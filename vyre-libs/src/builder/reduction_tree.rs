//! Reduction programs whose scratch fold is a `reduce` child.
//!
//! Behind `reduce` because every program here composes one: the workgroup-tree
//! folds the tiled mean, softmax and dot pipelines publish through, and the
//! atomic scalar reduction over u32 elements. `builder::reduction` holds the
//! composer and the phase record, which reach no dialect and compile wherever
//! `builder` does.

use vyre_foundation::ir::Program;
#[cfg(feature = "builder-ops")]
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

#[cfg(feature = "builder-ops")]
use crate::reduce::workgroup_tree::{self, WorkgroupReductionScope};

use super::reduction::ReductionComposer;
#[cfg(feature = "builder-ops")]
use super::reduction::ReductionPhase;

impl ReductionComposer {
    /// Build a tiled mean reduction program.
    #[cfg(feature = "builder-ops")]
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
            accumulate: crate::builder::strided_accumulate_child(
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
        Self::new(
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
    ///
    /// `nn::attention::softmax` is the only caller, and `nn-attention` already
    /// names `builder-ops` plus `reduce` through `nn-kernels`. The inline test
    /// reaches it on the bare kernel features.
    #[cfg(any(feature = "nn-attention", all(test, feature = "builder-ops")))]
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
            accumulate: crate::builder::strided_accumulate_child(
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
            accumulate: crate::builder::strided_accumulate_child(
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

        Self::new(
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
        .with_writeback(crate::builder::strided_writeback_child(
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
    #[cfg(feature = "builder-ops")]
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
            accumulate: crate::builder::strided_accumulate_child(
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

        Self::new(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "builder-ops")]
    fn tiled_mean_composition_structure() {
        let program = ReductionComposer::tiled_mean("test::mean", "in", "out", 1024, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 3);
        assert_eq!(program.buffers()[0].name.as_ref(), "in");
        assert_eq!(program.buffers()[1].name.as_ref(), "mean_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "out");
    }

    #[test]
    #[cfg(feature = "builder-ops")]
    fn tiled_softmax_composition_structure() {
        let program =
            ReductionComposer::tiled_softmax("test::softmax", "in", "out", 512, [256, 1, 1]);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 4);
        assert_eq!(program.buffers()[0].name.as_ref(), "in");
        assert_eq!(program.buffers()[1].name.as_ref(), "softmax_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "softmax_max");
        assert_eq!(program.buffers()[3].name.as_ref(), "out");
    }

    #[test]
    #[cfg(feature = "builder-ops")]
    fn tiled_dot_composition_structure() {
        let program = ReductionComposer::tiled_dot("test::dot", "lhs", "rhs", "out", 512, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 4);
        assert_eq!(program.buffers()[0].name.as_ref(), "lhs");
        assert_eq!(program.buffers()[1].name.as_ref(), "rhs");
        assert_eq!(program.buffers()[2].name.as_ref(), "dot_scratch");
        assert_eq!(program.buffers()[3].name.as_ref(), "out");
    }

    #[test]
    fn atomic_scalar_reductions_structure() {
        use crate::reduce::atomic_scalar::AtomicReduceKind;

        for kind in [
            AtomicReduceKind::Sum,
            AtomicReduceKind::Min,
            AtomicReduceKind::Max,
            AtomicReduceKind::PopcountSum,
            AtomicReduceKind::CountNonZero,
        ] {
            let p =
                ReductionComposer::atomic_scalar_reduction("test::atomic", "in", "out", 128, kind);
            assert_eq!(p.workgroup_size(), [256, 1, 1]);
        }
    }
}
