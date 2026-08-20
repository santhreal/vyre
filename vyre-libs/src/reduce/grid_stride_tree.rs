//! Multi-workgroup grid-stride tree reductions.
//!
//! Two-level reduction:
//! 1. Each workgroup loads a slice of input and reduces it locally across lanes
//!    using workgroup/subgroup tree reduction.
//! 2. Thread 0 of each workgroup atomically combines its workgroup partial into the
//!    scalar output.
//!
//! This distributes work across all SMs on the device and achieves near-peak
//! memory bandwidth on large inputs, while avoiding 1M-way atomic contention.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::workgroup_tree::{sum_u32_child, WorkgroupReductionScope};

/// Canonical op id for multi-workgroup grid-stride tree sum over u32 elements.
pub const SUM_U32_OP_ID: &str = "vyre-libs::reduce::grid_stride_tree_sum_u32";

/// Build a multi-workgroup tree reduction program for u32 sum.
#[must_use]
pub fn grid_stride_tree_sum_u32(
    values: &str,
    out: &str,
    count: u32,
    tile: u32,
) -> Program {
    let tile = tile.max(1);
    let items_per_thread = 16u32;
    let chunk_tile = tile.saturating_mul(items_per_thread);
    let scratch = "__grid_stride_tree_scratch";
    let local = Expr::LocalId { axis: 0 };
    let block = Expr::WorkgroupId { axis: 0 };

    let body = vec![
        Node::let_bind("local", local.clone()),
        Node::let_bind("block", block.clone()),
        Node::let_bind("acc", Expr::u32(0)),
        Node::if_then(
            Expr::lt(
                Expr::mul(block.clone(), Expr::u32(chunk_tile)),
                Expr::u32(count),
            ),
            vec![
                Node::loop_for(
                    "item",
                    Expr::u32(0),
                    Expr::u32(items_per_thread),
                    vec![
                        Node::let_bind(
                            "idx",
                            Expr::add(
                                Expr::add(
                                    Expr::mul(block.clone(), Expr::u32(chunk_tile)),
                                    Expr::mul(Expr::var("item"), Expr::u32(tile)),
                                ),
                                local.clone(),
                            ),
                        ),
                        Node::if_then(
                            Expr::lt(Expr::var("idx"), Expr::u32(count)),
                            vec![Node::assign(
                                "acc",
                                Expr::add(Expr::var("acc"), Expr::load(values, Expr::var("idx"))),
                            )],
                        ),
                    ],
                ),
            ],
        ),
        Node::store(scratch, local.clone(), Expr::var("acc")),
        Node::barrier(),
        sum_u32_child(
            SUM_U32_OP_ID,
            tile,
            scratch,
            WorkgroupReductionScope::EveryWorkgroup,
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(local, Expr::u32(0)),
                Expr::lt(
                    Expr::mul(block, Expr::u32(chunk_tile)),
                    Expr::u32(count),
                ),
            ),
            vec![Node::let_bind(
                "_prev",
                Expr::atomic_add(
                    out,
                    Expr::u32(0),
                    Expr::load(scratch, Expr::u32(0)),
                ),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::workgroup(scratch, tile, DataType::U32),
            BufferDecl::storage(out, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(SUM_U32_OP_ID, body)],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SUM_U32_OP_ID,
        || grid_stride_tree_sum_u32("values", "out", 4, 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[1, 2, 3, 4]), to_bytes(&[0])]]
        }),
        Some(|| vec![vec![vec![0x0a, 0x00, 0x00, 0x00]]]),
    )
    .with_laws(&["associative", "commutative", "identity"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_stride_tree_sum_u32_builds_valid_program() {
        let program = grid_stride_tree_sum_u32("values", "out", 1024, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 3);
        assert_eq!(program.buffers()[0].name.as_ref(), "values");
        assert_eq!(program.buffers()[1].name.as_ref(), "__grid_stride_tree_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "out");
    }
}
