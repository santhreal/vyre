//! Cutting a grid-synchronizing `Program` into sequential dispatch segments.
//!
//! A `Node::Barrier { ordering: MemoryOrdering::GridSync }` is a whole-grid
//! fence. No shading language has an instruction for it, and only a cooperative
//! launch can satisfy it device-side, so on every other route the fence is a
//! CUT: the program becomes N programs dispatched in order, and the launch
//! boundary between them is the fence. Every prior write is globally visible
//! before the next launch reads.
//!
//! This module owns the IR half of that cut, which is the same on every
//! consumer: detection, hoisting a fence out of the wrappers that hide it,
//! segmenting the entry sequence, and making each segment self-contained.
//! `vyre_megakernel::grid_sync` applies it to a `ProgramGraph` before schedule
//! search, and `vyre_driver::grid_sync` applies it at dispatch time. Neither
//! owns the transform, because two copies of a fence walk that disagree by one
//! `Node` variant is a kernel that runs with no cross-block synchronization and
//! reports success.
//!
//! `MemoryOrdering::SeqCst` and the acquire/release orderings are workgroup-scope
//! and are not cut. They stay in the segment and lower to a barrier instruction.

mod let_propagation;

use std::collections::TryReserveError;
use std::sync::Arc;

use thiserror::Error;

use crate::allocation::try_reserve_vec_to_capacity;
use crate::ir::{Ident, MemoryOrdering, Node, Program};
use crate::visit::{any_descendant, child_bodies};

use let_propagation::propagate_let_bindings;

/// Failure to cut a program at its grid-sync fences.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GridSyncSplitError {
    /// Segment storage could not be reserved.
    #[error("grid-sync split could not reserve {capacity} slots for {field}. Fix: reduce the program's entry-sequence size or raise the process memory limit.")]
    Reservation {
        /// Storage being reserved.
        field: &'static str,
        /// Requested capacity in items.
        capacity: usize,
    },
    /// Barrier count and entry-node count disagree.
    #[error("grid-sync split counted {barriers} fences in an entry sequence of {nodes} nodes. Fix: count fences from the same hoisted sequence that is segmented.")]
    Accounting {
        /// Fences counted.
        barriers: usize,
        /// Nodes in the sequence being segmented.
        nodes: usize,
    },
}

fn reserve<T>(
    vec: &mut Vec<T>,
    capacity: usize,
    field: &'static str,
) -> Result<(), GridSyncSplitError> {
    try_reserve_vec_to_capacity(vec, capacity)
        .map_err(|_: TryReserveError| GridSyncSplitError::Reservation { field, capacity })
}

/// Walk past `Program::wrapped`'s synthetic outer Region. Real programs are
/// constructed via `wrapped`, which inserts a single outer Region around the
/// entry sequence; the split logic must operate on the inner sequence so a
/// `GridSync` fence inside the wrapper actually splits the program. Programs
/// constructed via `Program::new` use the entry sequence directly, in which case
/// it is returned unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
enum EntryWrapper {
    Region { generator: Ident },
    Block,
}

fn peel_entry_wrappers(program: &Program) -> (Vec<EntryWrapper>, &[Node]) {
    let mut wrappers = Vec::new();
    let mut entry = program.entry();
    loop {
        if entry.len() == 1 {
            match &entry[0] {
                Node::Region {
                    generator, body, ..
                } => {
                    wrappers.push(EntryWrapper::Region {
                        generator: generator.clone(),
                    });
                    entry = body.as_slice();
                    continue;
                }
                Node::Block(body) => {
                    wrappers.push(EntryWrapper::Block);
                    entry = body.as_slice();
                    continue;
                }
                // A genuine default over an open set: `Region` and `Block` are
                // the only wrappers `wrap_split_segment` can rebuild, so
                // peeling anything else would lose structure the segments must
                // be re-wrapped in. Stopping here leaves the node in the entry
                // sequence, where the split still sees it.
                _ => {}
            }
        }
        break;
    }
    (wrappers, entry)
}

/// The dispatch-level entry sequence, peeled past any synthetic outer wrapper.
#[must_use]
pub fn entry_sequence(program: &Program) -> &[Node] {
    peel_entry_wrappers(program).1
}

/// Whether `program` contains any `Node::Barrier { ordering: GridSync }`
/// anywhere under its dispatch-level entry sequence.
///
/// The walk is DEEP and must stay deep. `validate::barrier` rejects only a
/// barrier in divergent control flow, so a fence inside a uniform `If` or a
/// counted `Loop` is a legal program that reaches here. Detection and hoisting
/// have deliberately different depths: this reports the fence, while
/// [`split_on_grid_sync`] promotes only the ones it can promote legally, and the
/// consumer refuses the rest. Answering "no fence" for a nested one would route
/// the program down the ordinary dispatch path, where the fence lowers to a
/// workgroup barrier and the kernel silently runs unsynchronized.
#[must_use]
pub fn contains_grid_sync(program: &Program) -> bool {
    // O(1) negative gate: if the cached ProgramStats bitset records no Barrier
    // of any kind in the entire tree, there is no GridSync fence either. Skip
    // the entry-sequence walk, which pays a buffers/buffer_index dispatch on
    // every backend dispatch path.
    if !program.stats().has_node_barrier() {
        return false;
    }
    entry_sequence(program).iter().any(node_contains_grid_sync)
}

/// True when `node` or anything under it is a grid-sync fence.
///
/// Delegates to [`any_descendant`], which enumerates children through
/// [`crate::visit::child_bodies`], the one exhaustive owner of "which `Node`
/// variants contain other nodes".
///
/// A false negative here is the worst outcome in this file.
/// [`contains_grid_sync`] is the gate every route consults before committing to
/// a split, a cooperative launch, or a refusal. Reading a nested fence as absent
/// sends the program down the ORDINARY path, where the fence lowers to a
/// workgroup barrier and the kernel runs with no cross-block synchronization at
/// all: no error, no split, wrong answers. Compare the fence that IS detected
/// but cannot be hoisted, which reaches the emitter and is refused loudly.
fn node_contains_grid_sync(node: &Node) -> bool {
    any_descendant(node, &mut |candidate| {
        matches!(
            candidate,
            Node::Barrier {
                ordering: MemoryOrdering::GridSync,
                ..
            }
        )
    })
}

/// The induction variable of the innermost loop enclosing a grid-sync fence, if
/// any fence in `program` is loop-nested.
///
/// A loop body is emitted ONCE and branched back to, so one cut cannot express a
/// per-iteration whole-grid fence: the first iteration would be synchronized and
/// every later one would not. There is no correct segmentation, so the consumer
/// must refuse rather than cut, and the refusal names the loop returned here so
/// the fix is locatable in the source program.
#[must_use]
pub fn loop_nested_grid_sync(program: &Program) -> Option<Ident> {
    if !program.stats().has_node_barrier() {
        return None;
    }
    innermost_loop_over_grid_sync(entry_sequence(program), None)
}

fn innermost_loop_over_grid_sync(nodes: &[Node], enclosing: Option<&Ident>) -> Option<Ident> {
    for node in nodes {
        if matches!(
            node,
            Node::Barrier {
                ordering: MemoryOrdering::GridSync,
                ..
            }
        ) {
            if let Some(var) = enclosing {
                return Some(var.clone());
            }
        }
        let inner = match node {
            Node::Loop { var, .. } => Some(var),
            // Every other body-bearing variant leaves the enclosing loop
            // unchanged: a fence inside a `Block` inside a `Loop` is still
            // per-iteration. The bodies themselves come from `child_bodies`, so
            // a new nesting variant is descended into without an edit here.
            _ => enclosing,
        };
        for body in child_bodies(node) {
            if let Some(found) = innermost_loop_over_grid_sync(body, inner) {
                return Some(found);
            }
        }
    }
    None
}

/// Lift grid-sync fences out of unconditional `Block` and `Region` bodies so the
/// dispatch-level split can see them.
///
/// The returned sequence contains the same nodes in the same order, with each
/// hoistable fence promoted to a sibling of the container it came from and the
/// container split around it. A fence that cannot be hoisted is preserved
/// verbatim; see the catch-all arm below.
fn hoist_grid_sync_barriers(nodes: &[Node]) -> Vec<Node> {
    let mut new_nodes = Vec::new();
    for node in nodes {
        match node {
            Node::Block(body) => {
                let new_body = hoist_grid_sync_barriers(body);
                if new_body.iter().any(is_grid_sync_fence) {
                    let mut current = Vec::new();
                    for inner in new_body {
                        if is_grid_sync_fence(&inner) {
                            new_nodes.push(Node::Block(std::mem::take(&mut current)));
                            new_nodes.push(inner);
                        } else {
                            current.push(inner);
                        }
                    }
                    new_nodes.push(Node::Block(current));
                } else {
                    new_nodes.push(Node::Block(new_body));
                }
            }
            Node::Region {
                generator,
                source_region,
                body,
            } => {
                let new_body = hoist_grid_sync_barriers(body);
                if new_body.iter().any(is_grid_sync_fence) {
                    let mut current = Vec::new();
                    for inner in new_body {
                        if is_grid_sync_fence(&inner) {
                            new_nodes.push(Node::Region {
                                generator: generator.clone(),
                                source_region: source_region.clone(),
                                body: Arc::new(std::mem::take(&mut current)),
                            });
                            new_nodes.push(inner);
                        } else {
                            current.push(inner);
                        }
                    }
                    new_nodes.push(Node::Region {
                        generator: generator.clone(),
                        source_region: source_region.clone(),
                        body: Arc::new(current),
                    });
                } else {
                    new_nodes.push(Node::Region {
                        generator: generator.clone(),
                        source_region: source_region.clone(),
                        body: Arc::new(new_body),
                    });
                }
            }
            // KEPT, and load-bearing. Hoisting lifts a fence to the top of the
            // entry sequence, which is legal only where the enclosing body
            // executes unconditionally and in order, so only `Block` and
            // `Region` qualify. A fence under `If` or `Loop` executes
            // conditionally, and moving it out would change which invocations
            // reach it, so it is copied verbatim instead. That is not a silent
            // drop: `contains_grid_sync` still reports the program as
            // grid-syncing, [`loop_nested_grid_sync`] names the loop, and the
            // fence survives into a segment where the emitter refuses the shape.
            // A new body-carrying variant inherits the same conservative answer.
            other => {
                new_nodes.push(other.clone());
            }
        }
    }
    new_nodes
}

fn is_grid_sync_fence(node: &Node) -> bool {
    matches!(
        node,
        Node::Barrier {
            ordering: MemoryOrdering::GridSync,
            ..
        }
    )
}

/// Split `program` at every hoistable `Node::Barrier { GridSync }`.
///
/// Returns segments in execution order. The fence nodes themselves are dropped:
/// the launch boundary between segments takes their place.
///
/// Each segment is a complete `Program` sharing the original's buffer table,
/// workgroup size, and metadata; only the entry sequence changes, plus the `Let`
/// bindings a segment reads and did not define. Segments without executable
/// nodes are preserved, so an empty segment between two adjacent fences becomes
/// a no-op dispatch that completes with byte-identical inputs and outputs.
///
/// A program with no hoistable fence yields exactly one segment equal to the
/// input, so a caller may apply this unconditionally.
///
/// # Errors
///
/// Returns [`GridSyncSplitError`] when segment storage cannot be reserved or
/// when fence accounting overflows.
pub fn split_on_grid_sync(program: &Program) -> Result<Vec<Program>, GridSyncSplitError> {
    let (wrappers, inner) = peel_entry_wrappers(program);
    let hoisted = hoist_grid_sync_barriers(inner);
    let fences = hoisted
        .iter()
        .filter(|node| is_grid_sync_fence(node))
        .count();
    if fences == 0 {
        let mut segments = Vec::new();
        reserve(&mut segments, 1, "grid-sync no-op segment")?;
        segments.push(program.clone());
        return Ok(segments);
    }

    let segment_count = fences + 1;
    let executable = hoisted
        .len()
        .checked_sub(fences)
        .ok_or(GridSyncSplitError::Accounting {
            barriers: fences,
            nodes: hoisted.len(),
        })?;
    let segment_capacity = executable.div_ceil(segment_count);

    let mut raw_segments = Vec::new();
    reserve(&mut raw_segments, segment_count, "grid-sync segment list")?;
    let mut current = Vec::new();
    reserve(&mut current, segment_capacity, "grid-sync current segment")?;
    for node in &hoisted {
        if is_grid_sync_fence(node) {
            let mut next = Vec::new();
            reserve(&mut next, segment_capacity, "grid-sync next segment")?;
            raw_segments.push(std::mem::replace(&mut current, next));
        } else {
            current.push(node.clone());
        }
    }
    raw_segments.push(current);

    propagate_let_bindings(&mut raw_segments, &hoisted);

    let mut segments = Vec::new();
    reserve(
        &mut segments,
        raw_segments.len(),
        "grid-sync split segments",
    )?;
    for entry in raw_segments {
        segments.push(wrap_split_segment(program, &wrappers, entry));
    }
    Ok(segments)
}

fn wrap_split_segment(program: &Program, wrappers: &[EntryWrapper], entry: Vec<Node>) -> Program {
    // Re-wrap each segment in the same wrapper stack the source had, so tagged
    // and fused programs keep provenance and structure while the executable body
    // is split at launch boundaries.
    let mut wrapped = entry;
    for wrapper in wrappers.iter().rev() {
        match wrapper {
            EntryWrapper::Region { generator } => {
                wrapped = vec![Node::Region {
                    generator: generator.clone(),
                    source_region: None,
                    body: Arc::new(wrapped),
                }];
            }
            EntryWrapper::Block => {
                wrapped = vec![Node::Block(wrapped)];
            }
        }
    }
    program.with_rewritten_entry(wrapped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr};

    fn buffer() -> BufferDecl {
        BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
    }

    fn region(generator: &str, body: Vec<Node>) -> Node {
        Node::Region {
            generator: Ident::from(generator),
            source_region: None,
            body: Arc::new(body),
        }
    }

    fn store(index: u32, value: u32) -> Node {
        Node::store("buf", Expr::u32(index), Expr::u32(value))
    }

    fn segments(program: &Program) -> Vec<Program> {
        split_on_grid_sync(program).expect("split must not exhaust memory in a test program")
    }

    #[test]
    fn a_program_without_a_fence_yields_itself() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![region("a", vec![store(0, 1)])],
        );
        assert!(!contains_grid_sync(&program));
        let split = segments(&program);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].entry(), program.entry());
    }

    #[test]
    fn a_workgroup_barrier_is_not_a_cut() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![
                region("a", vec![store(0, 1)]),
                Node::barrier_with_ordering(MemoryOrdering::SeqCst),
                region("b", vec![store(1, 2)]),
            ],
        );
        assert!(!contains_grid_sync(&program));
        let split = segments(&program);
        assert_eq!(split.len(), 1);
        assert_eq!(
            entry_sequence(&split[0]).len(),
            3,
            "a workgroup barrier stays in the segment and lowers to an instruction"
        );
    }

    /// A fence buried in the synthetic wrapper `Program::wrapped` inserts is the
    /// shape that actually reaches an emitter: `fuse` writes the fence inside the
    /// region body it wraps, so a split that only looked at the outer entry
    /// sequence would see one node and refuse to cut.
    #[test]
    fn a_fence_inside_the_wrapper_still_cuts() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![Node::Block(vec![
                region("a", vec![store(0, 1)]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("b", vec![store(1, 2)]),
            ])],
        );
        assert!(contains_grid_sync(&program));
        let split = segments(&program);
        assert_eq!(split.len(), 2);
        for segment in &split {
            assert!(
                !contains_grid_sync(segment),
                "no segment may still carry the fence that produced it"
            );
        }
    }

    #[test]
    fn every_fence_adds_one_segment() {
        for fences in 0..5_u32 {
            let mut nodes = vec![region("head", vec![store(0, 1)])];
            for index in 0..fences {
                nodes.push(Node::barrier_with_ordering(MemoryOrdering::GridSync));
                nodes.push(region("tail", vec![store(1, index)]));
            }
            let program = Program::wrapped(vec![buffer()], [1, 1, 1], nodes);
            assert_eq!(
                segments(&program).len(),
                fences as usize + 1,
                "a cut at each of {fences} fences yields one more segment than fences"
            );
        }
    }

    #[test]
    fn each_segment_keeps_the_buffer_table_and_geometry() {
        let program = Program::wrapped(
            vec![buffer()],
            [256, 1, 1],
            vec![
                region("a", vec![store(0, 1)]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("b", vec![store(1, 2)]),
            ],
        );
        for segment in segments(&program) {
            assert_eq!(segment.workgroup_size(), [256, 1, 1]);
            assert_eq!(segment.buffers().len(), 1);
            assert_eq!(segment.buffers()[0].name(), "buf");
        }
    }

    /// A binding defined before the cut and read after it must travel with the
    /// segment that reads it, or that segment references a free variable.
    #[test]
    fn a_binding_read_after_the_cut_travels_with_the_segment() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![
                Node::Let {
                    name: Ident::from("base"),
                    value: Expr::u32(3),
                },
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                Node::store("buf", Expr::u32(0), Expr::Var(Ident::from("base"))),
            ],
        );
        let split = segments(&program);
        assert_eq!(split.len(), 2);
        assert!(
            entry_sequence(&split[1])
                .iter()
                .any(|node| matches!(node, Node::Let { name, .. } if name.as_str() == "base")),
            "the second segment reads `base`, so the binding must be hoisted into it"
        );
    }

    #[test]
    fn a_top_level_fence_is_not_loop_nested() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![
                region("a", vec![store(0, 1)]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("b", vec![store(1, 2)]),
            ],
        );
        assert_eq!(loop_nested_grid_sync(&program), None);
    }

    /// The refusal has to name a loop, so the query must return the INNERMOST
    /// one. Naming the outer loop of a nest points at code that is not where the
    /// fence is.
    #[test]
    fn a_nested_fence_reports_the_innermost_loop() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![Node::Loop {
                var: Ident::from("outer"),
                from: Expr::u32(0),
                to: Expr::u32(2),
                body: vec![Node::Loop {
                    var: Ident::from("inner"),
                    from: Expr::u32(0),
                    to: Expr::u32(2),
                    body: vec![Node::Block(vec![Node::barrier_with_ordering(
                        MemoryOrdering::GridSync,
                    )])],
                }],
            }],
        );
        assert_eq!(
            loop_nested_grid_sync(&program).as_ref().map(Ident::as_str),
            Some("inner"),
            "the refusal names the loop the fence is in, not an enclosing one"
        );
    }

    /// A loop-nested fence is detected and NOT hoisted. Both halves matter: a
    /// hoist would move the fence out of the iteration it belongs to, and losing
    /// the detection would route the program down the ordinary path where the
    /// fence lowers to a workgroup barrier.
    #[test]
    fn a_loop_nested_fence_survives_the_split_uncut() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![Node::Loop {
                var: Ident::from("round"),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: vec![
                    store(0, 1),
                    Node::barrier_with_ordering(MemoryOrdering::GridSync),
                    store(1, 2),
                ],
            }],
        );
        assert!(contains_grid_sync(&program));
        let split = segments(&program);
        assert_eq!(
            split.len(),
            1,
            "a fence inside a loop body is not a cut point"
        );
        assert!(
            contains_grid_sync(&split[0]),
            "the fence must remain visible so the consumer can refuse it"
        );
    }
}
