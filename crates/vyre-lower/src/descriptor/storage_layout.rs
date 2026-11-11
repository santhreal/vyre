//! Physical storage layout: what one workgroup allocates, how long each region
//! is live, and which regions share bytes.
//!
//! A descriptor stated its storage as a binding list and nothing else, so every
//! consumer that needed a byte total summed the declared element counts and
//! priced two regions with disjoint lifetimes as two allocations. The overlay is
//! stated here instead: a region carries the op span it is live across and the
//! offset a deterministic first-fit plan gave it, so a target reads one pool
//! size and the offsets inside it rather than deriving either. A region that is
//! declared and never accessed is live for the whole kernel, because nothing in
//! the descriptor proves its bytes are free.
//!
//! Two facts about registers ride here as well. `registers_per_invocation` is
//! the peak number of simultaneously live descriptor result ids, which is a
//! physical count over lowered SSA and is not the semantic-IR pressure estimate
//! the whole-program cost model reads. `fragment_words_per_invocation` is the
//! peak register-resident matrix fragment width one invocation contributes.
//!
//! Byte offsets are word-aligned at a minimum. The neutral descriptor addresses
//! storage in 32-bit words already, so a narrower alignment would state a region
//! start no operand encoding can name.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::{KernelBody, KernelDescriptor, KernelOpKind, MemoryClass};
use crate::analyses::child_body_operands;
use crate::operand_class::{classify_operand, OperandClass};

/// Current storage-layout schema version.
///
/// This advances when the layout gains or changes a field, which changes what a
/// target is promised about the pool it allocates.
pub const STORAGE_LAYOUT_VERSION: u16 = 1;

/// Smallest offset granularity a region start can be stated at, in bytes.
const WORD_BYTES: u32 = 4;

/// Op span one storage region is live across, in pre-order op indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageLifetime {
    /// First op index that accesses the region.
    pub first_op: u32,
    /// Last op index that accesses the region.
    pub last_op: u32,
}

/// One placed storage region: its size, its alignment, where it starts in its
/// pool, and the op span it is live across.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageRegion {
    /// Binding slot the region backs.
    pub slot: u32,
    /// Pool the region is placed in.
    pub class: MemoryClass,
    /// Region size in bytes.
    pub bytes: u64,
    /// Alignment the region start satisfies, in bytes.
    pub alignment: u32,
    /// Region start within its pool, in bytes.
    pub offset: u64,
    /// Op span the region is live across.
    pub lifetime: StorageLifetime,
}

/// What one workgroup allocates, with lifetime-overlaid regions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageLayout {
    /// Schema version the layout was written at.
    pub version: u16,
    /// Placed regions, ordered by pool then by placement order.
    pub regions: Vec<StorageRegion>,
    /// Bytes one workgroup allocates in workgroup-visible storage.
    pub shared_pool_bytes: u64,
    /// Bytes one workgroup allocates in invocation-private storage.
    pub private_pool_bytes: u64,
    /// Bytes the regions would need with no overlay at all.
    pub distinct_bytes: u64,
    /// Peak count of simultaneously live lowered result ids per invocation.
    pub registers_per_invocation: u32,
    /// Peak register-resident matrix fragment width per invocation, in words.
    pub fragment_words_per_invocation: u32,
}

/// Why a physical storage layout cannot be planned or cannot be relied on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageLayoutError {
    /// The layout was written by another version of this library.
    Version {
        /// Version the layout states.
        version: u16,
    },
    /// A workgroup-scoped binding states no element count, so its byte span is
    /// unknown and it cannot be placed in a pool.
    UnsizedRegion {
        /// Binding slot.
        slot: u32,
    },
    /// A region's byte span could not be computed from its element type.
    UnstatableRegion {
        /// Binding slot.
        slot: u32,
        /// Why the span is unstatable.
        reason: String,
    },
    /// A matrix fragment's register width could not be computed.
    FragmentUnstatable {
        /// Why the width is unstatable.
        reason: String,
    },
    /// A region spans zero bytes.
    ZeroBytes {
        /// Binding slot.
        slot: u32,
    },
    /// A region states no alignment.
    ZeroAlignment {
        /// Binding slot.
        slot: u32,
    },
    /// A region's offset is not a multiple of its own alignment.
    UnalignedOffset {
        /// Binding slot.
        slot: u32,
        /// Offset the layout states.
        offset: u64,
        /// Alignment the region requires.
        alignment: u32,
    },
    /// A region's lifetime ends before it begins.
    InvertedLifetime {
        /// Binding slot.
        slot: u32,
        /// First op the region is live at.
        first_op: u32,
        /// Last op the region is live at.
        last_op: u32,
    },
    /// Two regions claim the same binding slot.
    DuplicateSlot {
        /// Binding slot.
        slot: u32,
    },
    /// Two regions share bytes while both are live.
    OverlaidWhileLive {
        /// Binding slot of the earlier region.
        first: u32,
        /// Binding slot of the later region.
        second: u32,
    },
    /// A stated pool size is not the size the placed regions occupy.
    PoolBytesMismatch {
        /// Storage class of the pool.
        class: &'static str,
        /// Size the layout states.
        stated: u64,
        /// Size the regions occupy.
        planned: u64,
    },
    /// The stated unoverlaid total is not the sum of the region spans.
    DistinctBytesMismatch {
        /// Total the layout states.
        stated: u64,
        /// Total the regions sum to.
        summed: u64,
    },
    /// The workgroup pool is larger than the selected schedule allows.
    SharedCeilingExceeded {
        /// Size the pool occupies.
        pool_bytes: u64,
        /// Ceiling the schedule selected.
        ceiling: u64,
    },
    /// One invocation holds more live values than the selected schedule allows.
    RegisterCeilingExceeded {
        /// Live values one invocation holds.
        registers: u32,
        /// Ceiling the schedule selected.
        ceiling: u32,
    },
    /// Placing a region overflowed the byte address space.
    RegionOverflow {
        /// Binding slot.
        slot: u32,
    },
}

impl std::fmt::Display for StorageLayoutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Version { version } => write!(
                formatter,
                "storage layout version {version} is not {STORAGE_LAYOUT_VERSION}. Fix: re-lower the descriptor with this library."
            ),
            Self::UnsizedRegion { slot } => write!(
                formatter,
                "workgroup-scoped binding slot {slot} states no element count. Fix: declare the buffer with a fixed element count so its storage can be placed."
            ),
            Self::UnstatableRegion { slot, reason } => write!(
                formatter,
                "binding slot {slot} has no statable byte span: {reason}. Fix: declare the buffer with a fixed-width element type."
            ),
            Self::FragmentUnstatable { reason } => write!(
                formatter,
                "matrix fragment width is unstatable: {reason}. Fix: declare a fragment whose tile distributes across its lanes in whole words."
            ),
            Self::ZeroBytes { slot } => write!(
                formatter,
                "binding slot {slot} spans zero bytes. Fix: drop the binding or declare the elements it holds."
            ),
            Self::ZeroAlignment { slot } => write!(
                formatter,
                "binding slot {slot} states no alignment. Fix: place the region at word alignment or wider."
            ),
            Self::UnalignedOffset {
                slot,
                offset,
                alignment,
            } => write!(
                formatter,
                "binding slot {slot} is placed at offset {offset}, which is not a multiple of its {alignment}-byte alignment. Fix: keep the overlay planner the only writer of region offsets."
            ),
            Self::InvertedLifetime {
                slot,
                first_op,
                last_op,
            } => write!(
                formatter,
                "binding slot {slot} is live from op {first_op} to op {last_op}. Fix: derive the lifetime from the descriptor walk rather than stating it."
            ),
            Self::DuplicateSlot { slot } => write!(
                formatter,
                "binding slot {slot} has two storage regions. Fix: plan one region per declared binding."
            ),
            Self::OverlaidWhileLive { first, second } => write!(
                formatter,
                "binding slots {first} and {second} share bytes while both are live. Fix: overlay only regions with disjoint lifetimes."
            ),
            Self::PoolBytesMismatch {
                class,
                stated,
                planned,
            } => write!(
                formatter,
                "{class} pool states {stated} bytes and the placed regions occupy {planned}. Fix: publish the pool size the planner computed."
            ),
            Self::DistinctBytesMismatch { stated, summed } => write!(
                formatter,
                "layout states {stated} unoverlaid bytes and its regions sum to {summed}. Fix: publish the total the planner computed."
            ),
            Self::SharedCeilingExceeded {
                pool_bytes,
                ceiling,
            } => write!(
                formatter,
                "workgroup pool of {pool_bytes} bytes exceeds the {ceiling} bytes the selected schedule allows. Fix: select a schedule whose resource bound covers this storage, or tile the workload smaller."
            ),
            Self::RegisterCeilingExceeded {
                registers,
                ceiling,
            } => write!(
                formatter,
                "one invocation holds {registers} live values and the selected schedule allows {ceiling}. Fix: select a schedule with a wider register bound, or split the kernel."
            ),
            Self::RegionOverflow { slot } => write!(
                formatter,
                "placing binding slot {slot} overflowed the byte address space. Fix: declare a storage span that fits an unsigned 64-bit byte count."
            ),
        }
    }
}

impl std::error::Error for StorageLayoutError {}

impl StorageLifetime {
    /// A lifetime covering exactly one op.
    #[must_use]
    pub const fn at(op: u32) -> Self {
        Self {
            first_op: op,
            last_op: op,
        }
    }

    /// Whether both lifetimes are live at some common op.
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        self.first_op <= other.last_op && other.first_op <= self.last_op
    }
}

impl StorageRegion {
    /// First byte past this region.
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.offset + self.bytes
    }

    /// Whether both regions claim a common byte.
    #[must_use]
    pub const fn shares_bytes(&self, other: &Self) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }
}

impl StorageLayout {
    /// The region placed for `slot`.
    #[must_use]
    pub fn region(&self, slot: u32) -> Option<&StorageRegion> {
        self.regions.iter().find(|region| region.slot == slot)
    }

    /// Bytes the overlay saved against allocating every region separately.
    #[must_use]
    pub const fn overlaid_bytes(&self) -> u64 {
        self.distinct_bytes
            .saturating_sub(self.shared_pool_bytes)
            .saturating_sub(self.private_pool_bytes)
    }

    /// Check every fact a target is allowed to rely on.
    ///
    /// A zero ceiling is an unstated ceiling and is not enforced, which is how
    /// a schedule-free lowering and a backend that reports no limit are told
    /// apart from one that reports a limit of zero.
    ///
    /// # Errors
    ///
    /// Returns [`StorageLayoutError`] when the layout version is not this
    /// library's, a region is malformed, two live regions share bytes, a stated
    /// total is not the one the regions produce, or a stated ceiling is
    /// exceeded.
    pub fn validate(
        &self,
        shared_ceiling: u64,
        register_ceiling: u32,
    ) -> Result<(), StorageLayoutError> {
        if self.version != STORAGE_LAYOUT_VERSION {
            return Err(StorageLayoutError::Version {
                version: self.version,
            });
        }
        let mut seen: FxHashMap<u32, ()> = FxHashMap::default();
        let mut shared_planned = 0_u64;
        let mut private_planned = 0_u64;
        let mut summed = 0_u64;
        for region in &self.regions {
            if seen.insert(region.slot, ()).is_some() {
                return Err(StorageLayoutError::DuplicateSlot { slot: region.slot });
            }
            if region.bytes == 0 {
                return Err(StorageLayoutError::ZeroBytes { slot: region.slot });
            }
            if region.alignment == 0 {
                return Err(StorageLayoutError::ZeroAlignment { slot: region.slot });
            }
            if region.offset % u64::from(region.alignment) != 0 {
                return Err(StorageLayoutError::UnalignedOffset {
                    slot: region.slot,
                    offset: region.offset,
                    alignment: region.alignment,
                });
            }
            if region.lifetime.first_op > region.lifetime.last_op {
                return Err(StorageLayoutError::InvertedLifetime {
                    slot: region.slot,
                    first_op: region.lifetime.first_op,
                    last_op: region.lifetime.last_op,
                });
            }
            let end = region
                .offset
                .checked_add(region.bytes)
                .ok_or(StorageLayoutError::RegionOverflow { slot: region.slot })?;
            summed = summed
                .checked_add(region.bytes)
                .ok_or(StorageLayoutError::RegionOverflow { slot: region.slot })?;
            match region.class {
                MemoryClass::Shared => shared_planned = shared_planned.max(end),
                _ => private_planned = private_planned.max(end),
            }
        }
        for (position, region) in self.regions.iter().enumerate() {
            for other in self.regions.iter().skip(position + 1) {
                if region.class == other.class
                    && region.lifetime.overlaps(&other.lifetime)
                    && region.shares_bytes(other)
                {
                    return Err(StorageLayoutError::OverlaidWhileLive {
                        first: region.slot,
                        second: other.slot,
                    });
                }
            }
        }
        if self.shared_pool_bytes != shared_planned {
            return Err(StorageLayoutError::PoolBytesMismatch {
                class: "workgroup",
                stated: self.shared_pool_bytes,
                planned: shared_planned,
            });
        }
        if self.private_pool_bytes != private_planned {
            return Err(StorageLayoutError::PoolBytesMismatch {
                class: "invocation-private",
                stated: self.private_pool_bytes,
                planned: private_planned,
            });
        }
        if self.distinct_bytes != summed {
            return Err(StorageLayoutError::DistinctBytesMismatch {
                stated: self.distinct_bytes,
                summed,
            });
        }
        if shared_ceiling > 0 && self.shared_pool_bytes > shared_ceiling {
            return Err(StorageLayoutError::SharedCeilingExceeded {
                pool_bytes: self.shared_pool_bytes,
                ceiling: shared_ceiling,
            });
        }
        if register_ceiling > 0 && self.registers_per_invocation > register_ceiling {
            return Err(StorageLayoutError::RegisterCeilingExceeded {
                registers: self.registers_per_invocation,
                ceiling: register_ceiling,
            });
        }
        Ok(())
    }

    /// Plan the storage one workgroup of `descriptor` allocates.
    ///
    /// Every workgroup-scoped and invocation-private binding becomes one region,
    /// its lifetime is the op span across which the descriptor accesses it, and
    /// regions whose lifetimes are disjoint share bytes. Placement is deterministic:
    /// regions are placed in `(first live op, slot)` order at the lowest aligned
    /// offset no live neighbour occupies.
    ///
    /// # Errors
    ///
    /// Returns [`StorageLayoutError`] when a region has no statable byte span, a
    /// matrix fragment width is unstatable, or the planned layout does not check.
    pub fn plan(descriptor: &KernelDescriptor) -> Result<Self, StorageLayoutError> {
        let mut spans: FxHashMap<u32, StorageLifetime> = FxHashMap::default();
        let mut placed: Vec<u32> = Vec::new();
        for slot in &descriptor.bindings.slots {
            if matches!(
                slot.memory_class,
                MemoryClass::Shared | MemoryClass::Scratch
            ) {
                placed.push(slot.slot);
            }
        }
        let mut next_index = 0_u32;
        collect_spans(&descriptor.body, &placed, &mut next_index, &mut spans);
        let whole_kernel = StorageLifetime {
            first_op: 0,
            last_op: next_index.saturating_sub(1),
        };

        let mut regions = Vec::with_capacity(placed.len());
        for slot in &descriptor.bindings.slots {
            if !matches!(
                slot.memory_class,
                MemoryClass::Shared | MemoryClass::Scratch
            ) {
                continue;
            }
            let count = slot
                .element_count
                .ok_or(StorageLayoutError::UnsizedRegion { slot: slot.slot })?;
            let bytes = slot
                .element_type
                .packed_size_bytes(count as usize)
                .map_err(|reason| StorageLayoutError::UnstatableRegion {
                    slot: slot.slot,
                    reason,
                })?
                .ok_or_else(|| StorageLayoutError::UnstatableRegion {
                    slot: slot.slot,
                    reason: format!("element type {} has no fixed width", slot.element_type),
                })?;
            let element_bytes =
                u32::try_from(slot.element_type.size_bytes().unwrap_or(0)).unwrap_or(0);
            regions.push(StorageRegion {
                slot: slot.slot,
                class: slot.memory_class,
                bytes: bytes as u64,
                alignment: element_bytes.max(WORD_BYTES),
                offset: 0,
                lifetime: spans.get(&slot.slot).copied().unwrap_or(whole_kernel),
            });
        }

        let shared_pool_bytes = place_class(&mut regions, MemoryClass::Shared)?;
        let private_pool_bytes = place_class(&mut regions, MemoryClass::Scratch)?;
        let mut distinct_bytes = 0_u64;
        for region in &regions {
            distinct_bytes = distinct_bytes
                .checked_add(region.bytes)
                .ok_or(StorageLayoutError::RegionOverflow { slot: region.slot })?;
        }

        let layout = StorageLayout {
            version: STORAGE_LAYOUT_VERSION,
            regions,
            shared_pool_bytes,
            private_pool_bytes,
            distinct_bytes,
            registers_per_invocation: peak_live_results(&descriptor.body, 0),
            fragment_words_per_invocation: peak_fragment_words(&descriptor.body)?,
        };
        layout.validate(0, 0)?;
        Ok(layout)
    }
}

/// Place every region of one class at the lowest aligned offset no live
/// neighbour occupies, and return the bytes the class occupies.
fn place_class(
    regions: &mut [StorageRegion],
    class: MemoryClass,
) -> Result<u64, StorageLayoutError> {
    let mut order: Vec<usize> = (0..regions.len())
        .filter(|index| regions[*index].class == class)
        .collect();
    order.sort_by_key(|index| (regions[*index].lifetime.first_op, regions[*index].slot));
    let mut done: Vec<usize> = Vec::with_capacity(order.len());
    let mut pool = 0_u64;
    for index in order {
        let (slot, bytes, alignment, lifetime) = {
            let region = &regions[index];
            (
                region.slot,
                region.bytes,
                u64::from(region.alignment),
                region.lifetime,
            )
        };
        let mut blocked: Vec<(u64, u64)> = done
            .iter()
            .filter(|other| regions[**other].lifetime.overlaps(&lifetime))
            .map(|other| (regions[*other].offset, regions[*other].end()))
            .collect();
        blocked.sort_unstable();
        let mut offset = 0_u64;
        for (start, end) in blocked {
            if offset
                .checked_add(bytes)
                .ok_or(StorageLayoutError::RegionOverflow { slot })?
                <= start
            {
                break;
            }
            if end > offset {
                offset = align_up(end, alignment, slot)?;
            }
        }
        let end = offset
            .checked_add(bytes)
            .ok_or(StorageLayoutError::RegionOverflow { slot })?;
        regions[index].offset = offset;
        pool = pool.max(end);
        done.push(index);
    }
    Ok(pool)
}

fn align_up(value: u64, alignment: u64, slot: u32) -> Result<u64, StorageLayoutError> {
    value
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or(StorageLayoutError::RegionOverflow { slot })
}

/// Record the op span across which each placed slot is accessed.
///
/// Op indices run in descriptor pre-order, and a structured op's nested bodies
/// take the indices right after its own, so a slot touched inside a nested body
/// is live across the whole enclosing construct. A loop body executes more than
/// once, so a narrower span would let the planner overlay a region that is still
/// read on the next iteration.
fn collect_spans(
    body: &KernelBody,
    placed: &[u32],
    next_index: &mut u32,
    spans: &mut FxHashMap<u32, StorageLifetime>,
) {
    for op in &body.ops {
        let index = *next_index;
        *next_index = next_index.saturating_add(1);
        for (position, operand) in op.operands.iter().copied().enumerate() {
            if classify_operand(&op.kind, position) == OperandClass::BindingSlot
                && placed.contains(&operand)
            {
                touch(spans, operand, StorageLifetime::at(index));
            }
        }
        let mut nested: FxHashMap<u32, StorageLifetime> = FxHashMap::default();
        for child in child_body_operands(&op.kind, &op.operands) {
            if let Some(child_body) = body.child_bodies.get(child as usize) {
                collect_spans(child_body, placed, next_index, &mut nested);
            }
        }
        let construct_end = next_index.saturating_sub(1);
        for slot in nested.keys() {
            touch(
                spans,
                *slot,
                StorageLifetime {
                    first_op: index,
                    last_op: construct_end,
                },
            );
        }
    }
}

fn touch(spans: &mut FxHashMap<u32, StorageLifetime>, slot: u32, span: StorageLifetime) {
    spans
        .entry(slot)
        .and_modify(|known| {
            known.first_op = known.first_op.min(span.first_op);
            known.last_op = known.last_op.max(span.last_op);
        })
        .or_insert(span);
}

/// Peak simultaneously live descriptor result ids, counting the values an
/// enclosing scope holds while a nested body runs.
fn peak_live_results(body: &KernelBody, base: u32) -> u32 {
    let mut last_use: FxHashMap<u32, usize> = FxHashMap::default();
    for (index, op) in body.ops.iter().enumerate() {
        for (position, operand) in op.operands.iter().copied().enumerate() {
            if classify_operand(&op.kind, position) == OperandClass::ResultRef {
                last_use.insert(operand, index);
            }
        }
    }
    let mut live: Vec<u32> = Vec::new();
    let mut peak = base;
    for (index, op) in body.ops.iter().enumerate() {
        for result in op.result_ids() {
            if !live.contains(&result) {
                live.push(result);
            }
        }
        let held = base.saturating_add(u32::try_from(live.len()).unwrap_or(u32::MAX));
        peak = peak.max(held);
        for child in child_body_operands(&op.kind, &op.operands) {
            if let Some(child_body) = body.child_bodies.get(child as usize) {
                peak = peak.max(peak_live_results(child_body, held));
            }
        }
        live.retain(|value| last_use.get(value).is_some_and(|last| *last > index));
    }
    peak
}

/// Peak register-resident matrix fragment words one invocation contributes.
fn peak_fragment_words(body: &KernelBody) -> Result<u32, StorageLayoutError> {
    let mut peak = 0_u32;
    for op in &body.ops {
        if let KernelOpKind::MatrixMma(spec) = &op.kind {
            let words =
                spec.operand_words()
                    .map_err(|error| StorageLayoutError::FragmentUnstatable {
                        reason: error.to_string(),
                    })?;
            peak = peak.max(words[0].saturating_add(words[1]).saturating_add(words[2]));
        }
        for child in child_body_operands(&op.kind, &op.operands) {
            if let Some(child_body) = body.child_bodies.get(child as usize) {
                peak = peak.max(peak_fragment_words(child_body)?);
            }
        }
    }
    Ok(peak)
}
