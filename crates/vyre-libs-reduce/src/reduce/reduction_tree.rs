//! Reduction programs whose scratch fold is a `reduce` child.

use vyre_foundation::ir::Program;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};
use vyre_libs_builder::builder::reduction::{ReductionComposer, ReductionPhase};

use super::workgroup_tree::{self, WorkgroupReductionScope};

/// Build a tiled mean reduction program.
#[must_use]
pub fn tiled_mean(
    generator: &'static str,
    input: &str,
    output: &str,
    n: u32,
    tile: u32,
) -> Program {
    let tile = tile.max(1);
    let chunks = n.div_ceil(tile);
    let phase = ReductionPhase {
        accumulate: vyre_libs_builder::builder::strided_accumulate_child(
            generator,
            tile,
            chunks,
            n,
            "mean_acc",
            Expr::f32(0.0),
            "mean_scratch",
            |idx, acc| Expr::add(acc, Expr::load(input, idx)),
        ),
        reductions: vec![workgroup_tree::sum_f32_child(
            generator,
            tile,
            "mean_scratch",
            WorkgroupReductionScope::FirstWorkgroup,
        )],
        publish: vec![Node::Store {
            buffer: output.into(),
            index: Expr::u32(0),
            value: Expr::div(
                Expr::load("mean_scratch", Expr::u32(0)),
                Expr::f32(n as f32),
            ),
        }],
    };
    ReductionComposer::new(
        generator,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::workgroup("mean_scratch", tile, DataType::F32),
            BufferDecl::output(output, 1, DataType::F32).with_count(1),
        ],
        [tile, 1, 1],
    )
    .with_phase(phase)
    .build()
}

/// Build a 2-phase tiled Softmax program (Max -> SumExp -> Writeback).
#[must_use]
pub fn tiled_softmax(
    generator: &'static str,
    input: &str,
    output: &str,
    n: u32,
    workgroup_size: [u32; 3],
) -> Program {
    let tile = workgroup_size[0].max(1);
    let chunks = n.div_ceil(tile);
    let max_pass = ReductionPhase {
        accumulate: vyre_libs_builder::builder::strided_accumulate_child(
            generator,
            tile,
            chunks,
            n,
            "local_max",
            Expr::f32(f32::MIN),
            "softmax_scratch",
            |idx, acc| {
                let loaded = Expr::load(input, idx);
                Expr::select(
                    Expr::BinOp {
                        op: vyre_foundation::ir::BinOp::Gt,
                        left: Box::new(loaded.clone()),
                        right: Box::new(acc.clone()),
                    },
                    loaded,
                    acc,
                )
            },
        ),
        reductions: vec![workgroup_tree::max_f32_child(
            generator,
            tile,
            "softmax_scratch",
            WorkgroupReductionScope::FirstWorkgroup,
        )],
        publish: vec![Node::Store {
            buffer: "softmax_max".into(),
            index: Expr::u32(0),
            value: Expr::load("softmax_scratch", Expr::u32(0)),
        }],
    };

    let sum_pass = ReductionPhase {
        accumulate: vyre_libs_builder::builder::strided_accumulate_child(
            generator,
            tile,
            chunks,
            n,
            "local_sum",
            Expr::f32(0.0),
            "softmax_scratch",
            |idx, acc| {
                Expr::add(
                    acc,
                    Expr::UnOp {
                        op: vyre_foundation::ir::UnOp::Exp,
                        operand: Box::new(Expr::BinOp {
                            op: vyre_foundation::ir::BinOp::Sub,
                            left: Box::new(Expr::load(input, idx)),
                            right: Box::new(Expr::load("softmax_max", Expr::u32(0))),
                        }),
                    },
                )
            },
        ),
        reductions: vec![workgroup_tree::sum_f32_child(
            generator,
            tile,
            "softmax_scratch",
            WorkgroupReductionScope::FirstWorkgroup,
        )],
        publish: Vec::new(),
    };

    ReductionComposer::new(
        generator,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::workgroup("softmax_scratch", tile, DataType::F32),
            BufferDecl::workgroup("softmax_max", 1, DataType::F32),
            BufferDecl::output(output, 1, DataType::F32).with_count(n),
        ],
        workgroup_size,
    )
    .with_phases([max_pass, sum_pass])
    .with_writeback(vyre_libs_builder::builder::strided_writeback_child(
        generator,
        tile,
        chunks,
        n,
        output,
        vec![
            Node::let_bind("sum_val", Expr::load("softmax_scratch", Expr::u32(0))),
            Node::let_bind("max_val", Expr::load("softmax_max", Expr::u32(0))),
        ],
        |idx| Expr::BinOp {
            op: vyre_foundation::ir::BinOp::Div,
            left: Box::new(Expr::UnOp {
                op: vyre_foundation::ir::UnOp::Exp,
                operand: Box::new(Expr::BinOp {
                    op: vyre_foundation::ir::BinOp::Sub,
                    left: Box::new(Expr::load(input, idx)),
                    right: Box::new(Expr::var("max_val")),
                }),
            }),
            right: Box::new(Expr::var("sum_val")),
        },
    ))
    .build()
}

/// Build a tiled dot product reduction program.
#[must_use]
pub fn tiled_dot(
    generator: &'static str,
    lhs: &str,
    rhs: &str,
    output: &str,
    n: u32,
    tile: u32,
) -> Program {
    let tile = tile.max(1);
    let chunks = n.div_ceil(tile);
    let phase = ReductionPhase {
        accumulate: vyre_libs_builder::builder::strided_accumulate_child(
            generator,
            tile,
            chunks,
            n,
            "local_acc",
            Expr::u32(0),
            "dot_scratch",
            |idx, acc| {
                Expr::add(
                    acc,
                    Expr::mul(Expr::load(lhs, idx.clone()), Expr::load(rhs, idx)),
                )
            },
        ),
        reductions: vec![workgroup_tree::sum_u32_child(
            generator,
            tile,
            "dot_scratch",
            WorkgroupReductionScope::FirstWorkgroup,
        )],
        publish: vec![Node::Store {
            buffer: output.into(),
            index: Expr::u32(0),
            value: Expr::load("dot_scratch", Expr::u32(0)),
        }],
    };

    ReductionComposer::new(
        generator,
        vec![
            BufferDecl::storage(lhs, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(rhs, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::workgroup("dot_scratch", tile, DataType::U32),
            BufferDecl::output(output, 2, DataType::U32).with_count(1),
        ],
        [tile, 1, 1],
    )
    .with_phase(phase)
    .build()
}

/// Build an atomic scalar reduction program over u32 elements.
#[must_use]
pub(crate) fn atomic_scalar_reduction(
    op_id: &'static str,
    input: &str,
    output: &str,
    count: u32,
    kind: crate::reduce::atomic_scalar::AtomicReduceKind,
) -> Program {
    crate::reduce::atomic_scalar::atomic_reduce_u32(input, output, count, kind, op_id)
}
