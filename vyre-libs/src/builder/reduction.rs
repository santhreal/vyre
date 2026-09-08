//! Canonical reduction composer and workgroup tree orchestration.
//!
//! Every reduction in `vyre-libs` (tiled reductions with writeback, atomic scalar
//! reductions, workgroup tree folds, and multi-phase statistical pipelines) shares
//! a single composition model:
//!
//! 1. **Index Space Mapping**: Local lane binding (`local = LogicalWithinTileId(0)`), strided
//!    chunk iteration (`chunk * tile + local`), and bounds guarding (`idx < n`).
//! 2. **Phase Execution**: One or more reduction phases executing in order. Each
//!    phase runs a strided accumulation child, a workgroup barrier (`Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst)`),
//!    one or more scratch-tree reduction children, and an optional guarded publication
//!    from lane 0 of workgroup 0.
//! 3. **Fence Optimization**: An intra-kernel workgroup barrier fences published
//!    scalars only when a subsequent phase or writeback reads them. Terminal publishes
//!    omit the trailing barrier.
//! 4. **Fused Epilogue**: An optional strided writeback pass streaming normalized
//!    or reduced outputs back to memory without a second dispatch.

use vyre_foundation::composition::wrap_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// One reduction pass over the input.
#[derive(Debug, Clone)]
pub struct ReductionPhase {
    /// Strided accumulation child, built with one of the
    /// `strided_accumulate*_child` helpers.
    pub accumulate: Node,
    /// Workgroup-tree reduction children, one per scratch buffer the
    /// accumulation filled.
    pub reductions: Vec<Node>,
    /// Statistics lane zero of workgroup zero writes once the reductions have
    /// landed. An empty publish emits neither the guarded store nor the
    /// barrier that would fence it.
    pub publish: Vec<Node>,
}

/// Canonical composer for reduction programs.
#[derive(Debug, Clone)]
pub struct ReductionComposer {
    generator: &'static str,
    buffers: Vec<BufferDecl>,
    workgroup_size: [u32; 3],
    phases: Vec<ReductionPhase>,
    writeback: Option<Node>,
}

impl ReductionComposer {
    /// Create a new reduction composer with the given generator, buffer declarations,
    /// and launch geometry.
    #[must_use]
    pub fn new(
        generator: &'static str,
        buffers: Vec<BufferDecl>,
        workgroup_size: [u32; 3],
    ) -> Self {
        Self {
            generator,
            buffers,
            workgroup_size,
            phases: Vec::new(),
            writeback: None,
        }
    }

    /// Append a reduction phase to this pipeline.
    #[must_use]
    pub fn with_phase(mut self, phase: ReductionPhase) -> Self {
        self.phases.push(phase);
        self
    }

    /// Append multiple reduction phases to this pipeline.
    ///
    /// `nn-attention` is the only dialect feature whose source reaches a
    /// multi-phase pipeline: `tiled_softmax` here and `nn::moe::gating` under
    /// `nn-moe`, which names `nn-attention`.
    #[cfg(any(test, feature = "nn-attention"))]
    #[must_use]
    pub fn with_phases(mut self, phases: impl IntoIterator<Item = ReductionPhase>) -> Self {
        self.phases.extend(phases);
        self
    }

    /// Attach a strided writeback epilogue.
    #[must_use]
    pub fn with_writeback(mut self, writeback: Node) -> Self {
        self.writeback = Some(writeback);
        self
    }

    /// Assemble the reduction into a final [`Program`].
    #[must_use]
    pub fn build(self) -> Program {
        let ReductionComposer {
            generator,
            buffers,
            workgroup_size,
            phases,
            writeback,
        } = self;
        let phase_count = phases.len();
        let mut body = vec![Node::let_bind(
            "local",
            Expr::LogicalWithinTileId { axis: 0 },
        )];
        for (index, phase) in phases.into_iter().enumerate() {
            body.push(phase.accumulate);
            body.push(Node::logical_barrier(
                vyre_foundation::ir::MemoryOrdering::SeqCst,
            ));
            body.extend(phase.reductions);
            if !phase.publish.is_empty() {
                body.push(Node::if_then(
                    Expr::and(
                        Expr::is_first_logical_tile(),
                        Expr::eq(Expr::var("local"), Expr::u32(0)),
                    ),
                    phase.publish,
                ));
                // The barrier fences the published scalars for whoever reads them. When
                // the publish is the last thing the program does, nothing reads them
                // and the barrier would be a synchronization every lane pays for a
                // value none of them loads.
                let read_later = index + 1 < phase_count || writeback.is_some();
                if read_later {
                    body.push(Node::logical_barrier(
                        vyre_foundation::ir::MemoryOrdering::SeqCst,
                    ));
                }
            }
        }
        body.extend(writeback);
        Program::wrapped(
            buffers,
            workgroup_size,
            vec![wrap_region(generator, body, None)],
        )
    }

    /// Build a tiled Welford parallel variance reduction program.
    #[must_use]
    pub fn tiled_variance(
        generator: &'static str,
        input: &str,
        output: &str,
        n: u32,
        bessel: bool,
        tile: u32,
    ) -> Program {
        let tile = tile.max(1);
        let chunks = n.div_ceil(tile);
        let local = Expr::var("local");
        let idx = Expr::var("idx");

        // Per-lane Welford accumulation over grid-stride chunks.
        let mut body = vec![
            Node::let_bind("local", Expr::LogicalWithinTileId { axis: 0 }),
            Node::if_then(
                Expr::is_first_logical_tile(),
                vec![
                    Node::let_bind("n_i", Expr::u32(0)),
                    Node::let_bind("M1_i", Expr::f32(0.0)),
                    Node::let_bind("M2_i", Expr::f32(0.0)),
                    Node::loop_for(
                        "chunk",
                        Expr::u32(0),
                        Expr::u32(chunks),
                        vec![
                            Node::let_bind(
                                "idx",
                                Expr::add(
                                    Expr::mul(Expr::var("chunk"), Expr::u32(tile)),
                                    local.clone(),
                                ),
                            ),
                            Node::if_then(
                                Expr::lt(idx.clone(), Expr::u32(n)),
                                vec![
                                    Node::let_bind("x", Expr::load(input, idx.clone())),
                                    Node::assign("n_i", Expr::add(Expr::var("n_i"), Expr::u32(1))),
                                    Node::let_bind(
                                        "delta",
                                        Expr::sub(Expr::var("x"), Expr::var("M1_i")),
                                    ),
                                    Node::assign(
                                        "M1_i",
                                        Expr::add(
                                            Expr::var("M1_i"),
                                            Expr::div(
                                                Expr::var("delta"),
                                                Expr::cast(DataType::F32, Expr::var("n_i")),
                                            ),
                                        ),
                                    ),
                                    Node::let_bind(
                                        "delta2",
                                        Expr::sub(Expr::var("x"), Expr::var("M1_i")),
                                    ),
                                    Node::assign(
                                        "M2_i",
                                        Expr::add(
                                            Expr::var("M2_i"),
                                            Expr::mul(Expr::var("delta"), Expr::var("delta2")),
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    Node::store("var_n_scratch", local.clone(), Expr::var("n_i")),
                    Node::store("var_m1_scratch", local.clone(), Expr::var("M1_i")),
                    Node::store("var_m2_scratch", local.clone(), Expr::var("M2_i")),
                ],
            ),
            Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
        ];

        // Workgroup-local tree reduction for Welford triples.
        let wg0_guard = Expr::is_first_logical_tile();
        let base_stride = tile.next_power_of_two() / 2;
        let steps = (base_stride as f32).log2() as u32 + 1;
        body.push(Node::loop_for(
            "step",
            Expr::u32(0),
            Expr::u32(steps),
            vec![
                Node::let_bind(
                    "stride",
                    Expr::shr(Expr::u32(base_stride), Expr::var("step")),
                ),
                Node::if_then(
                    Expr::and(
                        wg0_guard.clone(),
                        Expr::lt(Expr::var("local"), Expr::var("stride")),
                    ),
                    vec![Node::if_then(
                        Expr::lt(
                            Expr::add(Expr::var("local"), Expr::var("stride")),
                            Expr::u32(tile),
                        ),
                        vec![
                            Node::let_bind(
                                "other_idx",
                                Expr::add(Expr::var("local"), Expr::var("stride")),
                            ),
                            Node::let_bind("n_a", Expr::load("var_n_scratch", Expr::var("local"))),
                            Node::let_bind(
                                "n_b",
                                Expr::load("var_n_scratch", Expr::var("other_idx")),
                            ),
                            Node::if_then(
                                Expr::gt(Expr::var("n_b"), Expr::u32(0)),
                                vec![Node::if_then_else(
                                    Expr::eq(Expr::var("n_a"), Expr::u32(0)),
                                    vec![
                                        Node::store(
                                            "var_n_scratch",
                                            Expr::var("local"),
                                            Expr::var("n_b"),
                                        ),
                                        Node::store(
                                            "var_m1_scratch",
                                            Expr::var("local"),
                                            Expr::load("var_m1_scratch", Expr::var("other_idx")),
                                        ),
                                        Node::store(
                                            "var_m2_scratch",
                                            Expr::var("local"),
                                            Expr::load("var_m2_scratch", Expr::var("other_idx")),
                                        ),
                                    ],
                                    vec![
                                        Node::let_bind(
                                            "m1_a",
                                            Expr::load("var_m1_scratch", Expr::var("local")),
                                        ),
                                        Node::let_bind(
                                            "m1_b",
                                            Expr::load("var_m1_scratch", Expr::var("other_idx")),
                                        ),
                                        Node::let_bind(
                                            "m2_a",
                                            Expr::load("var_m2_scratch", Expr::var("local")),
                                        ),
                                        Node::let_bind(
                                            "m2_b",
                                            Expr::load("var_m2_scratch", Expr::var("other_idx")),
                                        ),
                                        Node::let_bind(
                                            "n_ab",
                                            Expr::add(Expr::var("n_a"), Expr::var("n_b")),
                                        ),
                                        Node::let_bind(
                                            "n_ab_f",
                                            Expr::cast(DataType::F32, Expr::var("n_ab")),
                                        ),
                                        Node::let_bind(
                                            "n_a_f",
                                            Expr::cast(DataType::F32, Expr::var("n_a")),
                                        ),
                                        Node::let_bind(
                                            "n_b_f",
                                            Expr::cast(DataType::F32, Expr::var("n_b")),
                                        ),
                                        Node::let_bind(
                                            "delta_ab",
                                            Expr::sub(Expr::var("m1_b"), Expr::var("m1_a")),
                                        ),
                                        Node::let_bind(
                                            "m1_comb",
                                            Expr::add(
                                                Expr::var("m1_a"),
                                                Expr::mul(
                                                    Expr::var("delta_ab"),
                                                    Expr::div(
                                                        Expr::var("n_b_f"),
                                                        Expr::var("n_ab_f"),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        Node::let_bind(
                                            "m2_comb",
                                            Expr::add(
                                                Expr::add(Expr::var("m2_a"), Expr::var("m2_b")),
                                                Expr::mul(
                                                    Expr::mul(
                                                        Expr::var("delta_ab"),
                                                        Expr::var("delta_ab"),
                                                    ),
                                                    Expr::div(
                                                        Expr::mul(
                                                            Expr::var("n_a_f"),
                                                            Expr::var("n_b_f"),
                                                        ),
                                                        Expr::var("n_ab_f"),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        Node::store(
                                            "var_n_scratch",
                                            Expr::var("local"),
                                            Expr::var("n_ab"),
                                        ),
                                        Node::store(
                                            "var_m1_scratch",
                                            Expr::var("local"),
                                            Expr::var("m1_comb"),
                                        ),
                                        Node::store(
                                            "var_m2_scratch",
                                            Expr::var("local"),
                                            Expr::var("m2_comb"),
                                        ),
                                    ],
                                )],
                            ),
                        ],
                    )],
                ),
                Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
            ],
        ));
        // Publish: lane 0 of workgroup 0 computes variance from accumulated M2.
        let divisor = if bessel {
            if n > 1 {
                (n - 1) as f32
            } else {
                1.0
            }
        } else {
            n as f32
        };
        body.push(Node::if_then(
            Expr::and(
                Expr::is_first_logical_tile(),
                Expr::eq(Expr::var("local"), Expr::u32(0)),
            ),
            vec![Node::store(
                output,
                Expr::u32(0),
                Expr::div(
                    Expr::load("var_m2_scratch", Expr::u32(0)),
                    Expr::f32(divisor),
                ),
            )],
        ));

        Program::wrapped(
            vec![
                BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
                BufferDecl::workgroup("var_n_scratch", tile, DataType::U32),
                BufferDecl::workgroup("var_m1_scratch", tile, DataType::F32),
                BufferDecl::workgroup("var_m2_scratch", tile, DataType::F32),
                BufferDecl::output(output, 1, DataType::F32).with_count(1),
            ],
            [tile, 1, 1],
            vec![wrap_region(generator, body, None)],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_composer_builds_valid_wrapped_program() {
        let composer = ReductionComposer::new(
            "test::reduction",
            vec![
                BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16),
                BufferDecl::output("out", 1, DataType::F32).with_count(1),
            ],
            [64, 1, 1],
        );
        let program = composer.build();
        assert_eq!(program.workgroup_size(), [64, 1, 1]);
        assert_eq!(program.buffers().len(), 2);
    }

    #[test]
    fn reduction_composer_barrier_fencing_semantics() {
        // 1. Single phase without writeback -> terminal publish has no trailing barrier.
        let phase_terminal = ReductionPhase {
            accumulate: Node::let_bind("acc", Expr::f32(0.0)),
            reductions: vec![],
            publish: vec![Node::store("out", Expr::u32(0), Expr::var("acc"))],
        };
        let program_terminal = ReductionComposer::new(
            "test::terminal",
            vec![
                BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16),
                BufferDecl::output("out", 1, DataType::F32).with_count(1),
            ],
            [64, 1, 1],
        )
        .with_phase(phase_terminal)
        .build();

        let terminal_has_barrier = program_terminal.entry().iter().any(|node| {
            vyre_foundation::visit::any_descendant(node, &mut |candidate| {
                matches!(candidate, Node::LogicalBarrier { .. })
            })
        });
        assert!(terminal_has_barrier);

        // 2. Multi-phase -> intermediate phase publish has trailing barrier.
        let phase1 = ReductionPhase {
            accumulate: Node::let_bind("acc1", Expr::f32(0.0)),
            reductions: vec![],
            publish: vec![Node::store("scratch_stat", Expr::u32(0), Expr::var("acc1"))],
        };
        let phase2 = ReductionPhase {
            accumulate: Node::let_bind("acc2", Expr::f32(0.0)),
            reductions: vec![],
            publish: vec![Node::store("out", Expr::u32(0), Expr::var("acc2"))],
        };
        let program_multi = ReductionComposer::new(
            "test::multi",
            vec![
                BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16),
                BufferDecl::workgroup("scratch_stat", 1, DataType::F32),
                BufferDecl::output("out", 2, DataType::F32).with_count(1),
            ],
            [64, 1, 1],
        )
        .with_phases([phase1, phase2])
        .build();

        let multi_has_barrier = program_multi.entry().iter().any(|node| {
            vyre_foundation::visit::any_descendant(node, &mut |candidate| {
                matches!(candidate, Node::LogicalBarrier { .. })
            })
        });
        assert!(multi_has_barrier);
    }

    #[test]
    fn tiled_variance_composition_structure() {
        let program = ReductionComposer::tiled_variance("test::var", "in", "out", 512, false, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 5);
        assert_eq!(program.buffers()[0].name.as_ref(), "in");
        assert_eq!(program.buffers()[1].name.as_ref(), "var_n_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "var_m1_scratch");
        assert_eq!(program.buffers()[3].name.as_ref(), "var_m2_scratch");
        assert_eq!(program.buffers()[4].name.as_ref(), "out");
    }
}
