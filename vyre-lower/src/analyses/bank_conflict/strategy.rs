//! Target-boundary bank-conflict mitigation strategy selection.
//!
//! Reuses neutral access facts from [`super::analyze`] to rank padding, XOR
//! swizzling, and no rewrite against a bank geometry the caller states: bank
//! count, bank width, subgroup shape, instruction width, and every access
//! phase. The caller owns the choice and applies it; this evaluates candidates.
//!
//! A candidate is rejected when a transformation merely moves an unacceptable
//! conflict to another phase. Universal zero conflicts is not promised.

use super::analysis::classify_index;
use super::report::{BankConflictKind, ConflictSeverity};
use crate::analyses::gcd_u32;
use crate::analyses::structured_walk::{walk_structured, ArmDescent, StructuredVisitor};
use crate::analyses::{AccessKind, ProducerMap};
use crate::operand_class::{classify_operand, OperandClass};
use crate::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind, MemoryClass};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

/// Physical and execution geometry for target shared-memory banks.
///
/// Every field is a device fact. A caller states all four from what the target
/// reported; this crate has no value to fall back on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetBankGeometry {
    /// Number of shared-memory banks.
    pub bank_count: u32,
    /// Width of each bank in bytes.
    pub bank_width_bytes: u32,
    /// Subgroup (execution wave) size in lanes.
    pub subgroup_lanes: u32,
    /// Native instruction access width in bytes.
    pub instruction_word_bytes: u32,
}

/// Access phase within a kernel's execution lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPhase {
    /// Global-to-shared staging / tile load phase.
    LoadStage,
    /// Inner compute loop shared-memory reads (e.g. matrix multiplication tiles).
    ComputeRead,
    /// Shared-memory intra-block reduction / tree-reduction phase.
    Reduction,
    /// Result store / epilogue phase.
    EpilogueStore,
}

/// Profile of shared-memory access pattern for one kernel phase.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccessPhaseProfile {
    /// Execution phase.
    pub phase: AccessPhase,
    /// Stride between consecutive thread accesses in elements.
    pub stride_elements: u32,
    /// Number of active threads accessing shared memory in this phase.
    pub active_threads: u32,
    /// Total access count weight (frequency in loop).
    pub access_weight: u32,
}

/// Bank conflict mitigation candidate strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BankConflictMitigation {
    /// Keep baseline layout without modifications.
    NoRewrite,
    /// Pad allocation row stride by N elements.
    PadLines {
        /// Number of padding elements added per row/stride.
        pad_elements_per_row: u32,
    },
    /// Apply bitwise XOR swizzling to address calculations.
    XorSwizzle {
        /// Number of bits to participate in XOR swizzling.
        swizzle_bits: u32,
        /// Stride shift in bits.
        stride_shift: u32,
    },
}

/// Conflict report for a single access phase under a candidate strategy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhaseConflictReport {
    /// Analyzed phase.
    pub phase: AccessPhase,
    /// Detected conflict kind.
    pub conflict: BankConflictKind,
    /// Severity classification.
    pub severity: ConflictSeverity,
    /// Estimated serialization penalty factor (1.0 = no penalty).
    pub penalty_factor: u32,
}

/// Multi-phase evaluation result for a candidate mitigation strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MitigationEvaluation {
    /// Evaluated strategy candidate.
    pub strategy: BankConflictMitigation,
    /// Conflict evaluation for each active phase.
    pub phase_reports: Vec<PhaseConflictReport>,
    /// Worst conflict severity across all phases.
    pub worst_severity: ConflictSeverity,
    /// Aggregate penalty score across weighted phases.
    pub aggregate_penalty: f32,
    /// Whether the candidate was accepted or rejected.
    pub accepted: bool,
    /// Rejection reason if candidate was rejected.
    pub rejection_reason: Option<String>,
}

/// Evaluate a single mitigation candidate across all execution phases.
#[must_use]
pub fn evaluate_mitigation_candidate(
    phases: &[AccessPhaseProfile],
    geometry: &TargetBankGeometry,
    candidate: BankConflictMitigation,
    baseline_worst: ConflictSeverity,
) -> MitigationEvaluation {
    let mut phase_reports = Vec::with_capacity(phases.len());
    let mut worst_severity = ConflictSeverity::None;
    let mut aggregate_penalty = 0.0_f32;

    for phase in phases {
        let effective_stride = match candidate {
            BankConflictMitigation::NoRewrite => phase.stride_elements,
            BankConflictMitigation::PadLines {
                pad_elements_per_row,
            } => phase.stride_elements.saturating_add(pad_elements_per_row),
            BankConflictMitigation::XorSwizzle { swizzle_bits, .. } => {
                // Swizzling reduces effective stride collision by spreading across 2^swizzle_bits banks
                let divisor = 1_u32 << swizzle_bits.min(5);
                (phase.stride_elements / divisor).max(1)
            }
        };

        let conflict =
            classify_phase_conflict(effective_stride, geometry.bank_count, phase.active_threads);
        let severity = conflict.severity();
        let penalty_factor = match conflict {
            BankConflictKind::NoConflict | BankConflictKind::BroadcastSafe => 1,
            BankConflictKind::Conflict { way_count } => way_count.max(1),
            BankConflictKind::Unknown => 4,
        };

        if severity_rank(severity) > severity_rank(worst_severity) {
            worst_severity = severity;
        }

        aggregate_penalty += (penalty_factor as f32) * (phase.access_weight as f32);

        phase_reports.push(PhaseConflictReport {
            phase: phase.phase,
            conflict,
            severity,
            penalty_factor,
        });
    }

    // WHY: Reject candidate when transformation merely moves an unacceptable conflict to another phase
    let is_rejected = match candidate {
        BankConflictMitigation::NoRewrite => false,
        _ => {
            // If candidate introduced a Severe or Critical conflict in any phase that wasn't already Critical
            let has_unacceptable_moved_conflict = phase_reports.iter().any(|r| {
                matches!(
                    r.severity,
                    ConflictSeverity::Critical | ConflictSeverity::Severe
                ) && severity_rank(r.severity) > severity_rank(baseline_worst)
            });
            has_unacceptable_moved_conflict
        }
    };

    let rejection_reason = if is_rejected {
        Some("mitigation moves unacceptable conflict to another access phase".to_string())
    } else {
        None
    };

    MitigationEvaluation {
        strategy: candidate,
        phase_reports,
        worst_severity,
        aggregate_penalty,
        accepted: !is_rejected,
        rejection_reason,
    }
}

/// Select the optimal bank-conflict mitigation strategy for a target geometry and phase profile.
#[must_use]
pub fn select_bank_conflict_strategy(
    phases: &[AccessPhaseProfile],
    geometry: &TargetBankGeometry,
) -> MitigationEvaluation {
    if phases.is_empty() {
        return MitigationEvaluation {
            strategy: BankConflictMitigation::NoRewrite,
            phase_reports: Vec::new(),
            worst_severity: ConflictSeverity::None,
            aggregate_penalty: 0.0,
            accepted: true,
            rejection_reason: None,
        };
    }

    // Evaluate baseline
    let baseline = evaluate_mitigation_candidate(
        phases,
        geometry,
        BankConflictMitigation::NoRewrite,
        ConflictSeverity::None,
    );
    let baseline_worst = baseline.worst_severity;

    let candidates = [
        BankConflictMitigation::NoRewrite,
        BankConflictMitigation::PadLines {
            pad_elements_per_row: 1,
        },
        BankConflictMitigation::PadLines {
            pad_elements_per_row: 2,
        },
        BankConflictMitigation::PadLines {
            pad_elements_per_row: 4,
        },
        BankConflictMitigation::XorSwizzle {
            swizzle_bits: 2,
            stride_shift: 3,
        },
        BankConflictMitigation::XorSwizzle {
            swizzle_bits: 3,
            stride_shift: 4,
        },
    ];

    let mut best = baseline;

    for candidate in candidates {
        let eval = evaluate_mitigation_candidate(phases, geometry, candidate, baseline_worst);
        if eval.accepted && eval.aggregate_penalty < best.aggregate_penalty {
            best = eval;
        }
    }

    best
}

fn classify_phase_conflict(stride: u32, bank_count: u32, active_threads: u32) -> BankConflictKind {
    if stride == 0 {
        return BankConflictKind::BroadcastSafe;
    }
    let gcd = gcd_u32(stride, bank_count);
    if gcd <= 1 {
        BankConflictKind::NoConflict
    } else {
        let way_count = gcd.min(active_threads);
        if way_count <= 1 {
            BankConflictKind::NoConflict
        } else {
            BankConflictKind::Conflict { way_count }
        }
    }
}

fn severity_rank(sev: ConflictSeverity) -> u32 {
    match sev {
        ConflictSeverity::None => 0,
        ConflictSeverity::Mild => 1,
        ConflictSeverity::Unknown => 2,
        ConflictSeverity::Severe => 3,
        ConflictSeverity::Critical => 4,
    }
}

/// Why a shared binding's element index cannot be rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SharedPermutationBlock {
    /// An asynchronous transaction reads or writes the binding. The transfer
    /// addresses the allocation itself, so rewriting the scalar address site
    /// would move part of the traffic and leave the rest where it was.
    AsyncTransaction,
    /// An atomic reaches the binding.
    Atomic,
    /// An access reaches the binding that classification proved no stride for,
    /// or that does not route through the scalar address site.
    UnprovenAccess,
    /// The binding declares no element count, so a padded allocation has no
    /// extent to grow from.
    NoDeclaredExtent,
}

/// One shared binding's derived access profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SharedBindingAccessProfile {
    /// Binding slot the profile describes.
    pub binding_slot: u32,
    /// Element count the binding declares, or zero when it declares none.
    pub element_count: u32,
    /// One entry per distinct phase and stride the kernel reaches, ordered by
    /// phase then stride.
    pub phases: Vec<AccessPhaseProfile>,
    /// Why the index cannot be rewritten. `None` means every access to the
    /// binding is a scalar shared load or store with a proven stride, so a
    /// rewrite at the address site rewrites all of them.
    pub blocked_by: Option<SharedPermutationBlock>,
}

/// Derive every shared binding's access phases from a descriptor.
///
/// A target needs three facts before it may rewrite a shared index: which
/// phases touch the binding, the stride and active width of each, and whether
/// rewriting the one address site rewrites every access. All three are derived
/// here from the descriptor and the bank count the caller states. Which rewrite
/// to apply is not decided here.
///
/// The phase a record lands in is derived from the barrier structure, counting
/// barriers in source order across nested bodies: a store before the first
/// barrier stages the tile, a load is a compute read, a store into an interval
/// that also loads the binding is a reduction, and a store in the last interval
/// is the epilogue.
///
/// Active width is the invocation count the descriptor's workgroup shape
/// states. How many of those lanes transact together is a device fact, so the
/// caller's geometry, not this derivation, bounds the way count.
#[must_use]
pub fn derive_shared_access_profiles(
    desc: &KernelDescriptor,
    banks: NonZeroU32,
) -> Vec<SharedBindingAccessProfile> {
    let active_width = desc
        .dispatch
        .workgroup_size
        .iter()
        .copied()
        .try_fold(1_u32, |total, extent| total.checked_mul(extent.max(1)))
        .unwrap_or(u32::MAX);
    let mut collector = SharedAccessCollector {
        shared_slots: desc
            .bindings
            .slots
            .iter()
            .filter(|binding| matches!(binding.memory_class, MemoryClass::Shared))
            .map(|binding| binding.slot)
            .collect(),
        bank_count: banks.get(),
        interval: 0,
        records: Vec::new(),
        blocks: BTreeMap::new(),
    };
    walk_structured(&desc.body, ArmDescent::Enter, &mut collector);

    let mut profiles = Vec::new();
    for binding in &desc.bindings.slots {
        if !matches!(binding.memory_class, MemoryClass::Shared) {
            continue;
        }
        let slot = binding.slot;
        let records: Vec<&SharedAccessRecord> = collector
            .records
            .iter()
            .filter(|record| record.slot == slot)
            .collect();
        let last_interval = records
            .iter()
            .map(|record| record.interval)
            .max()
            .unwrap_or(0);
        let loading_intervals: BTreeSet<u32> = records
            .iter()
            .filter(|record| matches!(record.kind, AccessKind::Load))
            .map(|record| record.interval)
            .collect();

        let mut counts: Vec<(AccessPhase, u32, u32)> = Vec::new();
        for record in &records {
            let phase = match (record.kind, record.interval) {
                (AccessKind::Load, _) => AccessPhase::ComputeRead,
                (AccessKind::Store, 0) => AccessPhase::LoadStage,
                (AccessKind::Store, interval) if loading_intervals.contains(&interval) => {
                    AccessPhase::Reduction
                }
                (AccessKind::Store, interval) if interval == last_interval => {
                    AccessPhase::EpilogueStore
                }
                (AccessKind::Store, _) => AccessPhase::LoadStage,
            };
            match counts
                .iter_mut()
                .find(|(seen, stride, _)| *seen == phase && *stride == record.stride)
            {
                Some((_, _, weight)) => *weight = weight.saturating_add(1),
                None => counts.push((phase, record.stride, 1)),
            }
        }
        counts.sort_unstable_by_key(|(phase, stride, _)| (phase_index(*phase), *stride));

        let blocked_by = match collector.blocks.get(&slot).copied() {
            Some(reason) => Some(reason),
            None if binding.element_count.is_none() => {
                Some(SharedPermutationBlock::NoDeclaredExtent)
            }
            None => None,
        };
        profiles.push(SharedBindingAccessProfile {
            binding_slot: slot,
            element_count: binding.element_count.unwrap_or(0),
            phases: counts
                .into_iter()
                .map(
                    |(phase, stride_elements, access_weight)| AccessPhaseProfile {
                        phase,
                        stride_elements,
                        active_threads: active_width,
                        access_weight,
                    },
                )
                .collect(),
            blocked_by,
        });
    }
    profiles
}

/// Rank of a phase in kernel execution order.
///
/// Exhaustive with no catch-all: a phase added to [`AccessPhase`] stops this
/// compiling until someone states where in the order it belongs.
fn phase_index(phase: AccessPhase) -> u32 {
    match phase {
        AccessPhase::LoadStage => 0,
        AccessPhase::ComputeRead => 1,
        AccessPhase::Reduction => 2,
        AccessPhase::EpilogueStore => 3,
    }
}

/// One classified scalar shared access.
struct SharedAccessRecord {
    /// Shared binding the access reaches.
    slot: u32,
    /// Number of barriers that preceded it in source order.
    interval: u32,
    /// Read or write.
    kind: AccessKind,
    /// Proven element stride between consecutive lanes.
    stride: u32,
}

/// Collects scalar shared accesses and what blocks a binding from being
/// rewritten, in one walk of the body tree.
struct SharedAccessCollector {
    /// Slots the binding layout states are shared.
    shared_slots: BTreeSet<u32>,
    /// Bank count the caller stated.
    bank_count: u32,
    /// Barriers seen so far.
    interval: u32,
    /// Classified scalar accesses, in walk order.
    records: Vec<SharedAccessRecord>,
    /// First block recorded per slot.
    blocks: BTreeMap<u32, SharedPermutationBlock>,
}

impl SharedAccessCollector {
    /// Record `reason` against `slot`, keeping the first reason found.
    fn block(&mut self, slot: u32, reason: SharedPermutationBlock) {
        if self.shared_slots.contains(&slot) {
            self.blocks.entry(slot).or_insert(reason);
        }
    }
}

impl<'a> StructuredVisitor<'a> for SharedAccessCollector {
    fn visit_op(
        &mut self,
        body: &'a KernelBody,
        producers: &ProducerMap<'a>,
        _op_index: usize,
        op: &'a KernelOp,
    ) {
        // Which operands state a binding slot is asked of the one owner of the
        // operand namespace. An op kind added there either declares a binding
        // operand, and falls into the unproven arm below, or declares none and
        // reaches no allocation. There is no second list to keep in step.
        let bindings: Vec<u32> = op
            .operands
            .iter()
            .copied()
            .enumerate()
            .filter(|(pos, _)| {
                matches!(classify_operand(&op.kind, *pos), OperandClass::BindingSlot)
            })
            .map(|(_, slot)| slot)
            .collect();

        match &op.kind {
            KernelOpKind::Barrier { .. } => self.interval = self.interval.saturating_add(1),
            KernelOpKind::LoadShared | KernelOpKind::StoreShared => {
                let Some(slot) = bindings.first().copied() else {
                    return;
                };
                if !self.shared_slots.contains(&slot) {
                    return;
                }
                let Some(index_operand) = op.operands.get(1).copied() else {
                    self.block(slot, SharedPermutationBlock::UnprovenAccess);
                    return;
                };
                let pattern = classify_index(body, producers, index_operand, self.bank_count);
                let Some(stride) = pattern.stride_elements else {
                    self.block(slot, SharedPermutationBlock::UnprovenAccess);
                    return;
                };
                self.records.push(SharedAccessRecord {
                    slot,
                    interval: self.interval,
                    kind: if matches!(op.kind, KernelOpKind::LoadShared) {
                        AccessKind::Load
                    } else {
                        AccessKind::Store
                    },
                    stride,
                });
            }
            KernelOpKind::Atomic { .. } => {
                for slot in bindings {
                    self.block(slot, SharedPermutationBlock::Atomic);
                }
            }
            // A transfer addresses the allocation, so rewriting the scalar
            // address site would move part of the traffic and leave the rest.
            KernelOpKind::AsyncLoad(_) | KernelOpKind::AsyncStore(_) => {
                for slot in bindings {
                    self.block(slot, SharedPermutationBlock::AsyncTransaction);
                }
            }
            // Any other op that addresses a binding reaches it through a path
            // the one scalar address site does not carry.
            _ => {
                for slot in bindings {
                    self.block(slot, SharedPermutationBlock::UnprovenAccess);
                }
            }
        }
    }
}
