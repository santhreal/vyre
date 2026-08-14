//! Grid-sync barrier detection and the split of a program's entry sequence
//! into one segment per barrier.

use std::sync::Arc;

use vyre_foundation::ir::{Ident, Node, Program};
use vyre_foundation::memory_model::MemoryOrdering;

use super::let_propagation::propagate_let_bindings;
use super::reserve_grid_sync_vec;
use crate::backend::BackendError;

/// Walk past `Program::wrapped`'s synthetic outer Region. Real
/// programs are constructed via `wrapped`, which inserts a single
/// outer Region around the user's entry sequence; the split logic
/// must operate on the inner sequence so a `GridSync` barrier inside
/// the wrapper actually splits the program. Programs constructed
/// via `Program::new` use the entry sequence directly  -  in that
/// case we just return it unchanged.
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
                _ => {}
            }
        }
        break;
    }
    (wrappers, entry)
}

pub(super) fn entry_sequence(program: &Program) -> &[Node] {
    peel_entry_wrappers(program).1
}

/// Whether `program` contains any `Node::Barrier { ordering: GridSync }`
/// in its dispatch-level entry sequence (peeled past any synthetic
/// outer Region).
///
/// The check is intentionally shallow: nested grid-sync barriers
/// inside `Node::Loop` or inner `Node::Region` bodies are a contract
/// violation (`validate::barrier` rejects them) and never reach this
/// path. The split operates at the dispatch-level granularity.
#[must_use]
pub fn contains_grid_sync(program: &Program) -> bool {
    // O(1) negative gate: if the cached ProgramStats bitset records no
    // Barrier of any kind in the entire tree, there is definitely no
    // top-level GridSync barrier either. Skip the entry-sequence walk
    // (which itself is shallow but still pays a buffers/buffer_index
    // dispatch on every backend dispatch path).
    if !program.stats().has_node_barrier() {
        return false;
    }
    node_slice_contains_grid_sync(entry_sequence(program))
}

fn node_slice_contains_grid_sync(nodes: &[Node]) -> bool {
    nodes.iter().any(node_contains_grid_sync)
}

fn node_contains_grid_sync(node: &Node) -> bool {
    match node {
        Node::Barrier {
            ordering: MemoryOrdering::GridSync,
            ..
        } => true,
        Node::If {
            then, otherwise, ..
        } => node_slice_contains_grid_sync(then) || node_slice_contains_grid_sync(otherwise),
        Node::Loop { body, .. } | Node::Block(body) => node_slice_contains_grid_sync(body),
        Node::Region { body, .. } => node_slice_contains_grid_sync(body),
        _ => false,
    }
}

/// Split `program` at every top-level `Node::Barrier { GridSync }`.
///
/// Returns a vector of segments in execution order. The barrier nodes
/// themselves are dropped from the segments  -  the kernel-launch
/// boundary between segments takes their place.
///
/// Each returned segment is a complete `Program` that shares the
/// original's buffer table, workgroup size, and metadata; only the
/// entry sequence changes. Segments without any executable nodes are
/// preserved (an empty segment between two adjacent barriers becomes
/// a no-op kernel that completes with byte-identical inputs and
/// outputs).
#[must_use]
pub fn split_on_grid_sync(program: &Program) -> Vec<Program> {
    try_split_on_grid_sync(program).unwrap_or_default()
}

/// Fallible variant of [`split_on_grid_sync`] for production dispatch paths.
///
/// # Errors
/// Returns an actionable [`BackendError`] if segment storage cannot be
/// reserved or if split accounting overflows.
fn hoist_grid_sync_barriers(nodes: &[Node]) -> Vec<Node> {
    let mut new_nodes = Vec::new();
    for node in nodes {
        match node {
            Node::Block(body) => {
                let new_body = hoist_grid_sync_barriers(body);
                let has_barrier = new_body.iter().any(|n| {
                    matches!(
                        n,
                        Node::Barrier {
                            ordering: MemoryOrdering::GridSync,
                            ..
                        }
                    )
                });
                if has_barrier {
                    let mut current_segment = Vec::new();
                    for b_node in new_body {
                        if matches!(
                            b_node,
                            Node::Barrier {
                                ordering: MemoryOrdering::GridSync,
                                ..
                            }
                        ) {
                            new_nodes.push(Node::Block(std::mem::take(&mut current_segment)));
                            new_nodes.push(b_node);
                        } else {
                            current_segment.push(b_node);
                        }
                    }
                    new_nodes.push(Node::Block(current_segment));
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
                let has_barrier = new_body.iter().any(|n| {
                    matches!(
                        n,
                        Node::Barrier {
                            ordering: MemoryOrdering::GridSync,
                            ..
                        }
                    )
                });
                if has_barrier {
                    let mut current_segment = Vec::new();
                    for b_node in new_body {
                        if matches!(
                            b_node,
                            Node::Barrier {
                                ordering: MemoryOrdering::GridSync,
                                ..
                            }
                        ) {
                            new_nodes.push(Node::Region {
                                generator: generator.clone(),
                                source_region: source_region.clone(),
                                body: Arc::new(std::mem::take(&mut current_segment)),
                            });
                            new_nodes.push(b_node);
                        } else {
                            current_segment.push(b_node);
                        }
                    }
                    new_nodes.push(Node::Region {
                        generator: generator.clone(),
                        source_region: source_region.clone(),
                        body: Arc::new(current_segment),
                    });
                } else {
                    new_nodes.push(Node::Region {
                        generator: generator.clone(),
                        source_region: source_region.clone(),
                        body: Arc::new(new_body),
                    });
                }
            }
            other => {
                new_nodes.push(other.clone());
            }
        }
    }
    new_nodes
}

/// Fallible variant of [`split_on_grid_sync`] for production dispatch paths.
///
/// # Errors
/// Returns an actionable [`BackendError`] if segment storage cannot be
/// reserved or if split accounting overflows.
pub fn try_split_on_grid_sync(program: &Program) -> Result<Vec<Program>, BackendError> {
    let (wrappers, inner) = peel_entry_wrappers(program);
    let hoisted_inner = hoist_grid_sync_barriers(inner);
    let split_count = hoisted_inner
        .iter()
        .filter(|node| {
            matches!(
                node,
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                    ..
                }
            )
        })
        .count();
    if split_count == 0 {
        let mut segments = Vec::new();
        reserve_grid_sync_vec(&mut segments, 1, "grid-sync no-op segment")?;
        segments.push(program.clone());
        return Ok(segments);
    }

    let segment_count = split_count + 1;
    let executable_nodes = hoisted_inner.len().checked_sub(split_count).ok_or_else(|| {
        BackendError::InvalidProgram {
            fix: format!(
            "grid-sync split_count {split_count} exceeded entry node count {}. Fix: split_on_grid_sync must count barriers from the same entry sequence it segments.",
            hoisted_inner.len()
            ),
        }
    })?;
    let segment_capacity = executable_nodes.div_ceil(segment_count);

    let mut raw_segments = Vec::new();
    let mut current = Vec::new();
    reserve_grid_sync_vec(&mut current, segment_capacity, "grid-sync current segment")?;
    for node in &hoisted_inner {
        match node {
            Node::Barrier {
                ordering: MemoryOrdering::GridSync,
                ..
            } => {
                let mut next = Vec::new();
                reserve_grid_sync_vec(&mut next, segment_capacity, "grid-sync next segment")?;
                let entry = std::mem::replace(&mut current, next);
                raw_segments.push(entry);
            }
            other => {
                current.push(other.clone());
            }
        }
    }
    raw_segments.push(current);

    propagate_let_bindings(&mut raw_segments, &hoisted_inner);

    let mut segments = Vec::new();
    reserve_grid_sync_vec(
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
    // Re-wrap each segment in the same wrapper stack the source had,
    // so tagged/fused programs keep provenance and structure while the
    // executable body is split at launch boundaries.
    let mut wrapped_entry = entry;
    for wrapper in wrappers.iter().rev() {
        match wrapper {
            EntryWrapper::Region { generator } => {
                wrapped_entry = vec![Node::Region {
                    generator: generator.clone(),
                    source_region: None,
                    body: Arc::new(wrapped_entry),
                }];
            }
            EntryWrapper::Block => {
                wrapped_entry = vec![Node::Block(wrapped_entry)];
            }
        }
    }
    program.with_rewritten_entry(wrapped_entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_sync::test_programs::{buffer, region};
    use vyre_foundation::ir::Expr;

    /// Get the inner-segment node count for a wrapped or unwrapped Program.
    fn inner_len(program: &Program) -> usize {
        entry_sequence(program).len()
    }

    #[test]
    fn no_grid_sync_returns_single_segment() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![region(
                "a",
                vec![Node::store("buf", Expr::u32(0), Expr::u32(1))],
            )],
        );
        assert!(!contains_grid_sync(&program));
        let segments = split_on_grid_sync(&program);
        assert_eq!(segments.len(), 1);
        // Original entry was [Region("a", ...)] so the inner sequence is 1.
        assert_eq!(inner_len(&segments[0]), 1);
    }

    #[test]
    fn one_grid_sync_splits_into_two() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![
                region("a", vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("b", vec![Node::store("buf", Expr::u32(1), Expr::u32(2))]),
            ],
        );
        assert!(contains_grid_sync(&program));
        let segments = split_on_grid_sync(&program);
        assert_eq!(segments.len(), 2);
        assert_eq!(inner_len(&segments[0]), 1);
        assert_eq!(inner_len(&segments[1]), 1);
    }

    #[test]
    fn block_nested_grid_sync_splits_into_two() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![Node::Block(vec![
                region("a", vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("b", vec![Node::store("buf", Expr::u32(1), Expr::u32(2))]),
            ])],
        );
        assert!(contains_grid_sync(&program));
        let segments = split_on_grid_sync(&program);
        assert_eq!(segments.len(), 2);
        assert_eq!(inner_len(&segments[0]), 1);
        assert_eq!(inner_len(&segments[1]), 1);
    }

    #[test]
    fn three_grid_syncs_split_into_four() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![
                region("a", vec![Node::Return]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("b", vec![Node::Return]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("c", vec![Node::Return]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("d", vec![Node::Return]),
            ],
        );
        let segments = split_on_grid_sync(&program);
        assert_eq!(segments.len(), 4);
    }

    #[test]
    fn workgroup_barrier_does_not_split() {
        let program = Program::wrapped(
            vec![buffer()],
            [1, 1, 1],
            vec![
                region("a", vec![Node::Return]),
                Node::barrier_with_ordering(MemoryOrdering::SeqCst),
                region("b", vec![Node::Return]),
            ],
        );
        assert!(!contains_grid_sync(&program));
        let segments = split_on_grid_sync(&program);
        assert_eq!(segments.len(), 1);
        // Region("a"), Barrier(SeqCst), Region("b") = 3 inner nodes.
        assert_eq!(inner_len(&segments[0]), 3);
    }

    #[test]
    fn buffers_and_workgroup_size_propagate_to_each_segment() {
        let program = Program::wrapped(
            vec![buffer()],
            [256, 1, 1],
            vec![
                region("a", vec![Node::Return]),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                region("b", vec![Node::Return]),
            ],
        );
        let segments = split_on_grid_sync(&program);
        for seg in &segments {
            assert_eq!(seg.workgroup_size(), [256, 1, 1]);
            assert_eq!(seg.buffers().len(), 1);
            assert_eq!(seg.buffers()[0].name(), "buf");
        }
    }
}
