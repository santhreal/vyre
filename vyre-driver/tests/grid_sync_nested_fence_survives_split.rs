//! The grid-sync splitter must not silently swallow a fence it cannot honor.
//!
//! `contains_grid_sync` RECURSES into loop and branch bodies, while
//! `split_on_grid_sync` only splits at DISPATCH-LEVEL barriers. A fence nested
//! inside a `Node::Loop` therefore routes a program down the split path while
//! being no kind of split point, which raises the obvious worry: if the splitter
//! dropped that fence on the way through, the program would emit clean and run
//! unsynchronized, and the emitter's refusal of the shape would never fire.
//!
//! It does not drop it. `hoist_grid_sync_barriers` recurses into `Node::Block`
//! and `Node::Region` only; a `Node::Loop` falls to the catch-all and is copied
//! verbatim, fence and all. So the nested fence survives into a segment and
//! reaches the emitter, which refuses it. These tests pin that, because the
//! alternative is silent grid desynchronization.

use std::sync::Arc;

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::transform::visit;
use vyre_foundation::MemoryOrdering;

fn nested_fence_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(256)],
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from("grid-sync-nested-fence-probe"),
            source_region: None,
            body: Arc::new(vec![Node::loop_for(
                "iter",
                Expr::u32(0),
                Expr::u32(4),
                vec![
                    Node::store("state", Expr::gid_x(), Expr::u32(1)),
                    Node::barrier_with_ordering(MemoryOrdering::GridSync),
                ],
            )]),
        }],
    )
}

/// Grid-sync fences anywhere in `nodes`, at any depth.
///
/// Descent comes from `for_each_node` so this counter cannot go stale against a
/// new `Node` variant that carries a body: the hand-written version reached
/// four variants and returned zero for the rest, which for a test that proves a
/// fence SURVIVES is the failure mode that reads as a pass.
fn grid_sync_fences(nodes: &[Node]) -> usize {
    let mut fences = 0;
    visit::for_each_node(nodes, |node| {
        if matches!(
            node,
            Node::Barrier {
                ordering: MemoryOrdering::GridSync
            }
        ) {
            fences += 1;
        }
    });
    fences
}

/// A loop-nested fence is detected, is not a split point, and is still present
/// after splitting.
///
/// The defect locked out is the silent one: a splitter that removed the fence it
/// could not hoist would hand the backend a program that emits cleanly and runs
/// its second and later iterations with no cross-block synchronization at all.
/// The count must survive the round trip exactly, so dropping the fence fails
/// here rather than surfacing later as wandering wrong answers.
#[test]
fn a_loop_nested_grid_sync_fence_is_preserved_through_the_split() {
    let program = nested_fence_program();
    assert_eq!(
        grid_sync_fences(program.entry()),
        1,
        "the probe program must declare exactly one loop-nested grid fence"
    );
    assert!(
        vyre_driver::grid_sync::contains_grid_sync(&program),
        "contains_grid_sync must recurse into loop bodies; a nested fence that reads as absent \
         would route this program down the non-grid path with no fence and no error"
    );

    let segments = vyre_driver::grid_sync::split_on_grid_sync(&program);
    assert_eq!(
        segments.len(),
        1,
        "a fence nested in a loop is not a dispatch-level split point, so the program must stay \
         one segment"
    );
    assert_eq!(
        segments
            .iter()
            .map(|segment| grid_sync_fences(segment.entry()))
            .sum::<usize>(),
        1,
        "the nested fence must survive the split; a splitter that swallowed it would produce a \
         program that emits clean and runs unsynchronized"
    );
}

/// A dispatch-level fence IS a split point and is consumed by it.
///
/// This is the contrast that makes the test above meaningful: it shows the
/// splitter really does remove top-level fences (so "count preserved" above is a
/// statement about nesting, not about a splitter that never removes anything),
/// and that the launch boundary replaces the fence.
#[test]
fn a_dispatch_level_grid_sync_fence_becomes_a_launch_boundary() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(256)],
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from("grid-sync-nested-fence-probe"),
            source_region: None,
            body: Arc::new(vec![
                Node::store("state", Expr::gid_x(), Expr::u32(1)),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                Node::store("state", Expr::gid_x(), Expr::u32(2)),
            ]),
        }],
    );
    let segments = vyre_driver::grid_sync::split_on_grid_sync(&program);
    assert_eq!(
        segments.len(),
        2,
        "a dispatch-level fence must split the program into two launch segments"
    );
    assert_eq!(
        segments
            .iter()
            .map(|segment| grid_sync_fences(segment.entry()))
            .sum::<usize>(),
        0,
        "the launch boundary replaces the fence, so no segment may still carry it"
    );
}
