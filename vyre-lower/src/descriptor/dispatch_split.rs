//! Cutting a descriptor into the dispatch segments a whole-grid fence separates.
//!
//! `KernelOpKind::Barrier { ordering: MemoryOrdering::GridSync }` synchronizes
//! every invocation of the dispatch. No shading language has an instruction for
//! it and only a cooperative launch satisfies it device-side, so on every other
//! route the fence is a launch boundary: the kernel becomes N kernels submitted
//! in order, and the boundary between them publishes every prior write.
//!
//! `vyre_foundation::transform::grid_sync_split` owns that cut at `Program`
//! level, for the planner and for dispatch-time splitting. This module owns it
//! at descriptor level, for an emitter that receives a fused descriptor and has
//! to lower the sequence the fences already describe. The two agree on which
//! placements can be cut: a fence promoted out of an unconditional wrapper is
//! cuttable, and a fence under a branch or a loop is not.
//!
//! The workgroup-scope orderings are not cut. They stay inside their segment and
//! lower to a barrier instruction.

use std::borrow::Cow;

use rustc_hash::{FxHashMap, FxHashSet};
use vyre_foundation::ir::MemoryOrdering;

use super::{DispatchSplitError, KernelBody, KernelDescriptor, KernelOp, KernelOpKind};
use crate::analyses::child_body_operands;
use crate::op_facts::{facts_for, NestedBodyControl};
use crate::operand_class::{classify_operand, OperandClass};

/// The dispatch segments `desc` is cut into at its whole-grid fences.
///
/// A descriptor with no fence yields exactly one borrowed segment equal to its
/// own body, so a caller may apply this unconditionally and a fence-free
/// descriptor emits the bytes it emitted before this cut existed.
///
/// Segments are returned in submission order. The fence ops themselves are
/// dropped: the boundary between two segments takes their place. Each segment is
/// self-contained, so a value an earlier segment computed and a later one reads
/// is recomputed in the later segment; a register does not survive a launch
/// boundary.
///
/// # Errors
///
/// Returns [`DispatchSplitError`] when a fence sits in a body whose execution
/// conditions differ from its container's, or when a value crossing a boundary
/// cannot be recomputed.
pub fn dispatch_segments(
    desc: &KernelDescriptor,
) -> Result<Vec<Cow<'_, KernelBody>>, DispatchSplitError> {
    if !body_contains_grid_fence(&desc.body) {
        return Ok(vec![Cow::Borrowed(&desc.body)]);
    }
    let items = linearize(&desc.body)?;
    let cuts = items.iter().filter(|item| item.is_none()).count();
    let mut segments = Vec::with_capacity(cuts + 1);
    let mut consumed = 0_usize;
    for group in items.split(|item| item.is_none()) {
        let prefix = carried_definitions(&items[..consumed], group)?;
        consumed += group.len() + 1;
        let mut builder = SegmentBuilder::default();
        for placed in prefix.into_iter().chain(group.iter().copied().flatten()) {
            builder.push(placed);
        }
        segments.push(Cow::Owned(builder.finish()));
    }
    Ok(segments)
}

/// Whether any body reachable from `body` carries a whole-grid fence.
///
/// The walk is deep on purpose. A fence under a branch or a loop is a legal
/// descriptor that reaches an emitter, and reporting it absent would send the
/// descriptor down the ordinary path where the fence lowers to a workgroup
/// barrier and the kernel runs with no cross-workgroup synchronization at all.
#[must_use]
pub fn body_contains_grid_fence(body: &KernelBody) -> bool {
    body.ops.iter().any(is_grid_fence)
        || body.child_bodies.iter().any(body_contains_grid_fence)
}

fn is_grid_fence(op: &KernelOp) -> bool {
    matches!(
        op.kind,
        KernelOpKind::Barrier {
            ordering: MemoryOrdering::GridSync
        }
    )
}

/// One op of the dispatch-level sequence, with the body its operand indices
/// resolve against.
#[derive(Clone, Copy)]
struct PlacedOp<'a> {
    op: &'a KernelOp,
    source: &'a KernelBody,
}

/// The dispatch-level op sequence of `body`, with `None` marking a fence.
///
/// Ops of an unconditional wrapper are spliced in where the wrapper stood, which
/// is what makes a fence inside the synthetic program-root region a
/// dispatch-level fence. Each spliced op keeps a reference to its own body, so a
/// literal-pool or child-body operand still resolves after the splice.
fn linearize(body: &KernelBody) -> Result<Vec<Option<PlacedOp<'_>>>, DispatchSplitError> {
    let mut items = Vec::with_capacity(body.ops.len());
    append_level(body, &mut items)?;
    Ok(items)
}

fn append_level<'a>(
    body: &'a KernelBody,
    items: &mut Vec<Option<PlacedOp<'a>>>,
) -> Result<(), DispatchSplitError> {
    for op in &body.ops {
        if is_grid_fence(op) {
            items.push(None);
            continue;
        }
        let children: Vec<&KernelBody> = child_body_operands(&op.kind, &op.operands)
            .filter_map(|index| body.child_bodies.get(index as usize))
            .collect();
        let Some(nested) = facts_for(&op.kind).nested_bodies else {
            items.push(Some(PlacedOp { op, source: body }));
            continue;
        };
        if !children.iter().any(|child| body_contains_grid_fence(child)) {
            items.push(Some(PlacedOp { op, source: body }));
            continue;
        }
        match nested.control {
            // Splicing every child, not only the fenced one, is what keeps the
            // sequence intact: an unconditional wrapper runs all of its bodies
            // in operand order, so dropping the unfenced ones would drop work.
            NestedBodyControl::Unconditional => {
                for child in children {
                    append_level(child, items)?;
                }
            }
            NestedBodyControl::Conditional => {
                return Err(DispatchSplitError::FenceUnderNestedControl {
                    construct: "structured branch arm",
                })
            }
            NestedBodyControl::Repeated => {
                return Err(DispatchSplitError::FenceUnderNestedControl {
                    construct: "structured loop body",
                })
            }
        }
    }
    Ok(())
}

/// The ops a segment reads and does not define, resolved out of the sequence
/// ahead of it and returned in that sequence's order.
///
/// A launch boundary ends every register, so a value the producing segment left
/// in one is recomputed here instead of forwarded. Only an op with no retained
/// effect is recomputed: a store, an atomic or a protocol step run twice is a
/// second effect, not a second copy of a value.
fn carried_definitions<'a>(
    earlier: &[Option<PlacedOp<'a>>],
    group: &[Option<PlacedOp<'a>>],
) -> Result<Vec<PlacedOp<'a>>, DispatchSplitError> {
    let defined: FxHashSet<u32> = group
        .iter()
        .flatten()
        .flat_map(|placed| placed.op.result_ids())
        .collect();
    let mut producers: FxHashMap<u32, (usize, PlacedOp<'a>)> = FxHashMap::default();
    for (position, placed) in earlier.iter().flatten().enumerate() {
        for result in placed.op.result_ids() {
            producers.insert(result, (position, *placed));
        }
    }

    let mut wanted: Vec<u32> = group
        .iter()
        .flatten()
        .flat_map(|placed| operand_result_refs(placed.op))
        .filter(|result| !defined.contains(result))
        .collect();
    let mut carried: FxHashMap<u32, (usize, PlacedOp<'a>)> = FxHashMap::default();
    while let Some(result) = wanted.pop() {
        if carried.contains_key(&result) {
            continue;
        }
        let Some((position, placed)) = producers.get(&result).copied() else {
            return Err(DispatchSplitError::UnresolvedCarrier { result });
        };
        if facts_for(&placed.op.kind).retained_effect {
            return Err(DispatchSplitError::EffectfulCarrier { result });
        }
        carried.insert(result, (position, placed));
        wanted.extend(operand_result_refs(placed.op));
    }

    let mut ordered: Vec<(usize, PlacedOp<'a>)> = carried.into_values().collect();
    ordered.sort_by_key(|(position, _)| *position);
    ordered.dedup_by_key(|(position, _)| *position);
    Ok(ordered.into_iter().map(|(_, placed)| placed).collect())
}

fn operand_result_refs(op: &KernelOp) -> impl Iterator<Item = u32> + '_ {
    op.operands
        .iter()
        .enumerate()
        .filter(|(position, _)| {
            classify_operand(&op.kind, *position) == OperandClass::ResultRef
        })
        .map(|(_, operand)| *operand)
}

/// One segment body under construction.
///
/// A segment holds only the child bodies and literals its own ops name, so every
/// child-body and literal-pool operand is renumbered as the op is copied in. The
/// alternative, cloning the whole parent's tables into every segment, would
/// leave each segment declaring workgroup scratch and literals it never reads.
#[derive(Default)]
struct SegmentBuilder {
    ops: Vec<KernelOp>,
    child_bodies: Vec<KernelBody>,
    literals: Vec<super::LiteralValue>,
}

impl SegmentBuilder {
    fn push(&mut self, placed: PlacedOp<'_>) {
        let mut op = placed.op.clone();
        for (position, operand) in op.operands.iter_mut().enumerate() {
            match classify_operand(&placed.op.kind, position) {
                OperandClass::ChildBodyIdx => {
                    let Some(child) = placed.source.child_bodies.get(*operand as usize) else {
                        continue;
                    };
                    *operand = u32::try_from(self.child_bodies.len()).unwrap_or(u32::MAX);
                    self.child_bodies.push(child.clone());
                }
                OperandClass::LiteralPoolIdx => {
                    let Some(literal) = placed.source.literals.get(*operand as usize) else {
                        continue;
                    };
                    *operand = u32::try_from(self.literals.len()).unwrap_or(u32::MAX);
                    self.literals.push(literal.clone());
                }
                OperandClass::ResultRef | OperandClass::BindingSlot | OperandClass::Other => {}
            }
        }
        self.ops.push(op);
    }

    fn finish(self) -> KernelBody {
        KernelBody {
            ops: self.ops,
            child_bodies: self.child_bodies,
            literals: self.literals,
        }
    }
}
