//! Shared-memory index permutation: which rewrite a permutable shared binding
//! gets, and the PTX that applies it.
//!
//! Split out of `memory.rs`, which owns address computation and the load/store
//! forms. Bank-conflict avoidance is a separate question answered before any
//! shared declaration is written.

use std::fmt::Write as _;
use std::num::NonZeroU32;

use vyre_lower::analyses::{
    derive_shared_access_profiles, select_bank_conflict_strategy, BankConflictMitigation,
    SharedBindingAccessProfile, TargetBankGeometry,
};
use vyre_lower::KernelDescriptor;
use vyre_lower::MemoryClass;

use super::BodyCtx;
use crate::reg::{PtxType, Reg};

/// Shared-memory bank geometry, as every CUDA target since Kepler reports it:
/// thirty-two four-byte banks, thirty-two lanes to a warp, four-byte native
/// access width.
const fn bank_geometry() -> TargetBankGeometry {
    TargetBankGeometry {
        bank_count: 32,
        bank_width_bytes: 4,
        subgroup_lanes: 32,
        instruction_word_bytes: 4,
    }
}

/// A bijective rewrite of a shared binding's element index.
///
/// Bijective is the whole requirement: two lanes whose element indices differ
/// must still differ after the rewrite, or the kernel computes different values
/// than the unpermuted one. Both arms are proven one-to-one over the binding's
/// element range by [`shared_permutation_for`], which refuses a strategy whose
/// preconditions the binding does not meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SharedPermutation {
    /// `index + (index >> row_log2) * pad`, which spreads each row of a tile
    /// across one more bank than it occupied. That is the padded-row address
    /// `(index >> row_log2) * (row + pad) + (index & (row - 1))` in two
    /// instructions rather than four.
    PadRows {
        /// Log2 of the unpadded row length in elements.
        row_log2: u32,
        /// Elements added per row.
        pad_elements: u32,
    },
    /// `index ^ ((index >> stride_shift) & mask)`, which rewrites only bits
    /// below the swizzle width and so keeps every index inside its own aligned
    /// block.
    XorSwizzle {
        /// Shift selecting the bits mixed into the low bits.
        stride_shift: u32,
        /// Mask of the low bits the swizzle rewrites.
        mask: u32,
    },
}

impl SharedPermutation {
    /// The element index this permutation maps `index` to.
    pub(super) fn apply(self, index: u32) -> u32 {
        match self {
            Self::PadRows {
                row_log2,
                pad_elements,
            } => {
                // The same expression `emit_shared_permutation` writes, so the
                // host model and the emitted kernel cannot drift apart: adding
                // the row count times the pad to the index is the padded-row
                // address `q * (row + pad) + r`, and
                // `padded_address_matches_the_closed_form` pins the two equal.
                index.saturating_add((index >> row_log2).saturating_mul(pad_elements))
            }
            Self::XorSwizzle { stride_shift, mask } => index ^ ((index >> stride_shift) & mask),
        }
    }

    /// Elements the allocation needs so every permuted index stays in range.
    pub(super) fn extent(self, element_count: u32) -> u32 {
        match self {
            Self::PadRows {
                row_log2,
                pad_elements,
            } => {
                let row = 1_u32 << row_log2;
                (element_count >> row_log2).saturating_mul(row.saturating_add(pad_elements))
            }
            Self::XorSwizzle { .. } => element_count,
        }
    }
}

/// The permutation that applies `strategy` to `profile`, when it is one-to-one
/// over the binding's element range.
///
/// `None` keeps the unpermuted index. A strategy is refused rather than
/// approximated: an index rewrite that is not a bijection, or that can leave
/// the allocation, is a wrong kernel, not a slower one.
fn shared_permutation_for(
    profile: &SharedBindingAccessProfile,
    strategy: BankConflictMitigation,
) -> Option<SharedPermutation> {
    match strategy {
        BankConflictMitigation::NoRewrite => None,
        BankConflictMitigation::PadLines {
            pad_elements_per_row,
        } => {
            if pad_elements_per_row == 0 {
                return None;
            }
            // Padding is priced as stride plus pad, which is a row length only
            // when one row explains every strided phase. Two disagreeing
            // strides mean the price and the rewrite describe different tiles.
            let mut row: Option<u32> = None;
            for phase in &profile.phases {
                if phase.stride_elements <= 1 {
                    continue;
                }
                match row {
                    None => row = Some(phase.stride_elements),
                    Some(seen) if seen == phase.stride_elements => {}
                    Some(_) => return None,
                }
            }
            let row = row?;
            if !row.is_power_of_two()
                || row < 2
                || profile.element_count == 0
                || profile.element_count % row != 0
            {
                return None;
            }
            Some(SharedPermutation::PadRows {
                row_log2: row.trailing_zeros(),
                pad_elements: pad_elements_per_row,
            })
        }
        BankConflictMitigation::XorSwizzle {
            swizzle_bits,
            stride_shift,
        } => {
            // The rewritten bits have to sit strictly below the bits the shift
            // reads, or the extracted value depends on what the XOR changed and
            // the map stops being invertible.
            if swizzle_bits == 0 || swizzle_bits > 5 || stride_shift < swizzle_bits {
                return None;
            }
            let block = 1_u32 << swizzle_bits;
            if profile.element_count == 0 || profile.element_count % block != 0 {
                return None;
            }
            Some(SharedPermutation::XorSwizzle {
                stride_shift,
                mask: block - 1,
            })
        }
    }
}

impl BodyCtx<'_> {
    /// Choose one index permutation per permutable shared binding.
    ///
    /// Called before any shared declaration is written, because a padded
    /// binding is declared at its grown extent. A binding is permutable only
    /// when the neutral derivation proved every access to it is a scalar load
    /// or store with a known stride, and that no asynchronous transaction or
    /// fused bulk copy reaches it: both route around the single address site
    /// the rewrite happens at, and the derivation states that verdict so this
    /// crate keeps no second copy of the rule.
    pub(super) fn plan_shared_permutations(&mut self, desc: &KernelDescriptor) {
        let geometry = bank_geometry();
        let Some(banks) = NonZeroU32::new(geometry.bank_count) else {
            return;
        };
        for profile in derive_shared_access_profiles(desc, banks) {
            if profile.blocked_by.is_some() {
                continue;
            }
            let selection = select_bank_conflict_strategy(&profile.phases, &geometry);
            if !selection.accepted {
                continue;
            }
            if let Some(permutation) = shared_permutation_for(&profile, selection.strategy) {
                self.slot_to_shared_permutation
                    .insert(profile.binding_slot, permutation);
            }
        }
    }

    /// The permutation chosen for `binding_slot`, if it is a shared binding
    /// and one was chosen.
    pub(super) fn shared_permutation(
        &self,
        binding_slot: u32,
        memory_class: MemoryClass,
    ) -> Option<SharedPermutation> {
        if !matches!(memory_class, MemoryClass::Shared) {
            return None;
        }
        self.slot_to_shared_permutation.get(&binding_slot).copied()
    }

    /// Rewrite a shared element index through the binding's permutation.
    ///
    /// Returns `index_reg` unchanged when the binding has none, so a kernel
    /// with nothing to permute emits exactly the instructions it did before.
    pub(super) fn emit_shared_permutation(&mut self, binding_slot: u32, index_reg: Reg) -> Reg {
        let Some(permutation) = self.slot_to_shared_permutation.get(&binding_slot).copied() else {
            return index_reg;
        };
        match permutation {
            SharedPermutation::PadRows {
                row_log2,
                pad_elements,
            } => {
                // `i = q * row + r` with `r < row`, so the padded address
                // `q * (row + pad) + r` equals `i + q * pad`. Adding the row
                // count times the pad to the index takes a shift and a
                // multiply-add, where splitting the index and rebuilding it
                // takes a shift, a mask, a multiply and an add at every
                // shared access.
                let rows = self.alloc(PtxType::U32);
                let permuted = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    shr.u32    {rows}, {index_reg}, {row_log2};");
                let _ = writeln!(
                    self.text,
                    "    mad.lo.u32    {permuted}, {rows}, {pad_elements}, {index_reg};"
                );
                permuted
            }
            SharedPermutation::XorSwizzle { stride_shift, mask } => {
                let high = self.alloc(PtxType::U32);
                let bits = self.alloc(PtxType::U32);
                let permuted = self.alloc(PtxType::U32);
                let _ = writeln!(
                    self.text,
                    "    shr.u32    {high}, {index_reg}, {stride_shift};"
                );
                let _ = writeln!(self.text, "    and.b32    {bits}, {high}, {mask};");
                let _ = writeln!(self.text, "    xor.b32    {permuted}, {index_reg}, {bits};");
                permuted
            }
        }
    }
}

// Inline: `SharedPermutation` and `shared_permutation_for` are crate-private,
// and the property they carry is arithmetic. Whether the emitted kernel applies
// them is proven from the emitted text in
// `tests/regression_emit_fixes.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet;
    use vyre_lower::analyses::{AccessPhase, AccessPhaseProfile, CANDIDATE_MITIGATIONS};

    /// A profile stating one phase per entry of `strides`, all full width.
    fn profile(element_count: u32, strides: &[u32]) -> SharedBindingAccessProfile {
        SharedBindingAccessProfile {
            binding_slot: 1,
            element_count,
            phases: strides
                .iter()
                .map(|stride| AccessPhaseProfile {
                    phase: AccessPhase::ComputeRead,
                    stride_elements: *stride,
                    active_threads: 32,
                    access_weight: 1,
                })
                .collect(),
            blocked_by: None,
        }
    }

    /// Two lanes whose element indices differ must still differ after the
    /// rewrite, and no rewritten index may leave the allocation. Either failure
    /// is a kernel that computes different values than the unpermuted one, so
    /// every index in the declared range is checked rather than sampled.
    ///
    /// The strategies come from [`CANDIDATE_MITIGATIONS`], the same set
    /// `select_bank_conflict_strategy` ranks, so a candidate added there is
    /// proven here on the next run instead of reaching an emitted kernel
    /// unproven. The extents are the tile shapes a strategy's preconditions
    /// accept or refuse: powers of two of several row counts, and one extent
    /// that is not a whole number of rows.
    ///
    /// `PERMUTATION_FLOOR` is what makes a derivation that stops producing
    /// permutations fail instead of reporting a clean sweep of an empty set.
    #[test]
    fn every_selectable_strategy_is_one_to_one_inside_the_extent_it_declares() {
        /// Fewer accepted permutations than the two arms times the two padding
        /// widths and the two swizzle widths that the current extents admit.
        const PERMUTATION_FLOOR: usize = 8;

        let extents = [256_u32, 1000, 1024, 2048, 4096];
        let mut accepted = 0_usize;
        let mut arms = FxHashSet::default();

        for element_count in extents {
            for strategy in CANDIDATE_MITIGATIONS {
                let binding = profile(element_count, &[32]);
                let Some(permutation) = shared_permutation_for(&binding, strategy) else {
                    continue;
                };
                accepted += 1;
                arms.insert(std::mem::discriminant(&permutation));

                let extent = permutation.extent(element_count);
                let mut seen = FxHashSet::default();
                for index in 0..element_count {
                    let mapped = permutation.apply(index);
                    assert!(
                        mapped < extent,
                        "{strategy:?} over {element_count} elements became {permutation:?}, \
                         which maps {index} to {mapped}, outside an extent of {extent}"
                    );
                    assert!(
                        seen.insert(mapped),
                        "{strategy:?} over {element_count} elements became {permutation:?}, \
                         which maps two indices to {mapped}"
                    );
                }
            }
        }

        assert!(
            accepted >= PERMUTATION_FLOOR,
            "the derivation accepted {accepted} permutations over {} candidates and {} extents, \
             below the floor of {PERMUTATION_FLOOR}: a refusal that admits nothing proves nothing",
            CANDIDATE_MITIGATIONS.len(),
            extents.len()
        );
        assert_eq!(
            arms.len(),
            2,
            "both permutation arms must be exercised, and {} was",
            arms.len()
        );
    }

    /// The reduced address `index + (index >> row_log2) * pad` that both
    /// `apply` and the emitted kernel compute is the padded-row address
    /// `q * (row + pad) + r`. The reduction saves two instructions at every
    /// shared access, so it holds for every index inside the extent or the
    /// kernel addresses the wrong element.
    #[test]
    fn padded_address_matches_the_closed_form() {
        let mut cases = 0_usize;

        for element_count in [256_u32, 1024, 2048, 4096] {
            for strategy in CANDIDATE_MITIGATIONS {
                let binding = profile(element_count, &[32]);
                let Some(
                    permutation @ SharedPermutation::PadRows {
                        row_log2,
                        pad_elements,
                    },
                ) = shared_permutation_for(&binding, strategy)
                else {
                    continue;
                };
                cases += 1;

                let row = 1_u32 << row_log2;
                for index in 0..element_count {
                    let closed_form =
                        (index >> row_log2) * (row + pad_elements) + (index & (row - 1));
                    let reduced = permutation.apply(index);
                    assert_eq!(
                        reduced, closed_form,
                        "{permutation:?} over {element_count} elements maps {index} to \
                         {reduced} where the padded-row address is {closed_form}"
                    );
                }
            }
        }

        assert!(
            cases > 0,
            "no padding permutation was derived over {} candidates, so the equality \
             this test claims was never evaluated",
            CANDIDATE_MITIGATIONS.len()
        );
    }

    /// The chain from bank geometry to emitted rewrite. A column walk over 32
    /// four-byte banks is a 32-way conflict, one element of padding per row is
    /// the cheapest accepted candidate, and that candidate becomes a row
    /// permutation over 32-element rows.
    #[test]
    fn a_column_walk_selects_one_element_of_padding_per_row() {
        let profile = profile(1024, &[32, 32]);
        let selection = select_bank_conflict_strategy(&profile.phases, &bank_geometry());
        assert!(selection.accepted);
        assert_eq!(
            selection.strategy,
            BankConflictMitigation::PadLines {
                pad_elements_per_row: 1
            }
        );
        assert_eq!(
            shared_permutation_for(&profile, selection.strategy),
            Some(SharedPermutation::PadRows {
                row_log2: 5,
                pad_elements: 1
            })
        );
    }

    /// A strategy is refused rather than approximated. Every precondition is
    /// checked against a binding that misses exactly that one, because a
    /// rewrite that is not a bijection, or that can leave the allocation, is a
    /// wrong kernel rather than a slower one.
    #[test]
    fn a_strategy_the_binding_cannot_carry_is_refused() {
        let pad = |pad_elements_per_row| BankConflictMitigation::PadLines {
            pad_elements_per_row,
        };
        let swizzle = |swizzle_bits, stride_shift| BankConflictMitigation::XorSwizzle {
            swizzle_bits,
            stride_shift,
        };
        let refused = [
            // Nothing to pad by.
            (profile(1024, &[32]), pad(0)),
            // Two strides mean the price and the rewrite describe different
            // tiles, so neither row length is the one that was ranked.
            (profile(1024, &[32, 16]), pad(1)),
            // A row length that is not a power of two has no shift form.
            (profile(1024, &[24]), pad(1)),
            // No strided phase states a row length at all.
            (profile(1024, &[1]), pad(1)),
            // The extent is not a whole number of rows, so the last row would
            // be padded past the allocation.
            (profile(1000, &[32]), pad(1)),
            (profile(0, &[32]), pad(1)),
            // The rewritten bits are not strictly below the bits the shift
            // reads, so the extracted value depends on what the XOR changed.
            (profile(1024, &[32]), swizzle(3, 2)),
            (profile(1024, &[32]), swizzle(0, 4)),
            (profile(1024, &[32]), swizzle(6, 8)),
            // The extent is not a whole number of swizzle blocks.
            (profile(12, &[32]), swizzle(3, 4)),
        ];
        for (binding, strategy) in refused {
            assert_eq!(
                shared_permutation_for(&binding, strategy),
                None,
                "{strategy:?} must be refused for a binding of {} elements with \
                 strides {:?}",
                binding.element_count,
                binding
                    .phases
                    .iter()
                    .map(|phase| phase.stride_elements)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            shared_permutation_for(&profile(1024, &[32]), BankConflictMitigation::NoRewrite),
            None
        );
    }

    /// The permutation is value-preserving: storing values through the
    /// permuted address and reading them back recovers the exact original
    /// values for every element in the allocation.
    #[test]
    fn permutation_is_value_preserving_across_roundtrip() {
        let extents = [256_u32, 1024, 2048, 4096];
        for element_count in extents {
            for strategy in CANDIDATE_MITIGATIONS {
                let binding = profile(element_count, &[32]);
                let Some(permutation) = shared_permutation_for(&binding, strategy) else {
                    continue;
                };
                let extent = permutation.extent(element_count);
                let mut buffer = vec![0xDEAD_BEEF_u32; extent as usize];
                let original: Vec<u32> = (0..element_count)
                    .map(|i| i.wrapping_mul(17) ^ 0x5555_AAAA)
                    .collect();

                // Store through the permutation
                for i in 0..element_count {
                    let addr = permutation.apply(i);
                    assert!(addr < extent);
                    buffer[addr as usize] = original[i as usize];
                }

                // Read back through the permutation
                for i in 0..element_count {
                    let addr = permutation.apply(i);
                    assert_eq!(
                        buffer[addr as usize], original[i as usize],
                        "value mismatch after roundtrip through {strategy:?} for element {i}"
                    );
                }
            }
        }
    }

    /// The strategy-variant space is derived from source at run time so a new
    /// mitigation variant turns the suite red until the emitter records a
    /// decision for it.
    #[test]
    fn every_declared_mitigation_strategy_has_an_emitter_decision() {
        let path = vyre_test_support::monorepo::vyre_workspace_root()
            .join("vyre-lower/src/analyses/bank_conflict/strategy.rs");
        let source = vyre_test_support::read_source_file_bounded(&path).unwrap_or_else(|err| {
            panic!("Fix: cannot read the BankConflictMitigation declaration at {path:?}: {err}")
        });
        let body = vyre_test_support::braced_body(&source, "pub enum BankConflictMitigation {")
            .unwrap_or_else(|| {
                panic!("Fix: no `pub enum BankConflictMitigation` declaration in {path:?}; update this enumeration")
            });
        let declared = vyre_test_support::top_level_variant_names(body);
        assert!(
            declared.len() >= 3,
            "Fix: expected at least 3 declared variants, found {}",
            declared.len()
        );

        let samples = [
            BankConflictMitigation::NoRewrite,
            BankConflictMitigation::PadLines {
                pad_elements_per_row: 1,
            },
            BankConflictMitigation::XorSwizzle {
                swizzle_bits: 2,
                stride_shift: 3,
            },
        ];
        let covered: std::collections::BTreeSet<String> = samples
            .iter()
            .map(|s| match s {
                BankConflictMitigation::NoRewrite => "NoRewrite".to_string(),
                BankConflictMitigation::PadLines { .. } => "PadLines".to_string(),
                BankConflictMitigation::XorSwizzle { .. } => "XorSwizzle".to_string(),
            })
            .collect();
        let missing: Vec<&String> = declared.difference(&covered).collect();
        assert!(
            missing.is_empty(),
            "Fix: add emitter decision and coverage for newly declared BankConflictMitigation variant(s): {missing:?}"
        );
    }
}
