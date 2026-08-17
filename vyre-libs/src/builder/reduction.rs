//! Canonical reduction composer and workgroup tree orchestration.
//!
//! Every reduction in `vyre-libs` (tiled reductions with writeback, atomic scalar
//! reductions, workgroup tree folds, multi-phase statistical pipelines, and prefix
//! scans) shares a single composition model:
//!
//! 1. **Index Space Mapping**: Local lane binding (`local = LocalId(0)`), strided
//!    chunk iteration (`chunk * tile + local`), and bounds guarding (`idx < n`).
//! 2. **Phase Execution**: One or more reduction phases executing in order. Each
//!    phase runs a strided accumulation child, a workgroup barrier (`Node::barrier()`),
//!    one or more scratch-tree reduction children, and an optional guarded publication
//!    from lane 0 of workgroup 0.
//! 3. **Fence Optimization**: An intra-kernel workgroup barrier fences published
//!    scalars only when a subsequent phase or writeback reads them. Terminal publishes
//!    omit the trailing barrier.
//! 4. **Fused Epilogue**: An optional strided writeback pass streaming normalized
//!    or reduced outputs back to memory without a second dispatch.

use vyre_foundation::composition::wrap_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[cfg(feature = "reduce")]
use crate::reduce::workgroup_tree::{self, WorkgroupReductionScope};

/// One reduction pass over the input.
#[derive(Debug, Clone)]
pub(crate) struct ReductionPhase {
    /// Strided accumulation child, built with one of the
    /// `strided_accumulate*_child` helpers.
    pub(crate) accumulate: Node,
    /// Workgroup-tree reduction children, one per scratch buffer the
    /// accumulation filled.
    pub(crate) reductions: Vec<Node>,
    /// Statistics lane zero of workgroup zero writes once the reductions have
    /// landed. An empty publish emits neither the guarded store nor the
    /// barrier that would fence it.
    pub(crate) publish: Vec<Node>,
}

impl ReductionPhase {
    /// Construct a new reduction phase.
    #[must_use]
    pub(crate) fn new(accumulate: Node, reductions: Vec<Node>, publish: Vec<Node>) -> Self {
        Self {
            accumulate,
            reductions,
            publish,
        }
    }
}

/// Specification for a tiled reduce-then-publish program.
#[derive(Debug, Clone)]
pub(crate) struct TiledReduceSpec {
    /// Region generator name recorded on the wrapping region.
    pub(crate) generator: &'static str,
    /// Buffer declarations in binding order.
    pub(crate) buffers: Vec<BufferDecl>,
    /// Workgroup size.
    pub(crate) workgroup: [u32; 3],
    /// Reduction passes, run in order.
    pub(crate) phases: Vec<ReductionPhase>,
    /// Final strided pass that writes the normalized output. A reduction whose
    /// result is the published scalar itself has no writeback.
    pub(crate) writeback: Option<Node>,
}

/// Canonical composer for reduction programs.
#[derive(Debug, Clone)]
pub(crate) struct ReductionComposer {
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
    pub(crate) fn new(
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

    /// Construct a composer from a [`TiledReduceSpec`].
    #[must_use]
    pub(crate) fn from_spec(spec: TiledReduceSpec) -> Self {
        Self {
            generator: spec.generator,
            buffers: spec.buffers,
            workgroup_size: spec.workgroup,
            phases: spec.phases,
            writeback: spec.writeback,
        }
    }

    /// Append a reduction phase to this pipeline.
    #[must_use]
    pub(crate) fn with_phase(mut self, phase: ReductionPhase) -> Self {
        self.phases.push(phase);
        self
    }

    /// Append multiple reduction phases to this pipeline.
    #[must_use]
    pub(crate) fn with_phases(mut self, phases: impl IntoIterator<Item = ReductionPhase>) -> Self {
        self.phases.extend(phases);
        self
    }

    /// Attach a strided writeback epilogue.
    #[must_use]
    pub(crate) fn with_writeback(mut self, writeback: Node) -> Self {
        self.writeback = Some(writeback);
        self
    }

    /// Assemble the reduction into a final [`Program`].
    #[must_use]
    pub(crate) fn build(self) -> Program {
        let ReductionComposer {
            generator,
            buffers,
            workgroup_size,
            phases,
            writeback,
        } = self;
        let phase_count = phases.len();
        let mut body = vec![Node::let_bind("local", Expr::LocalId { axis: 0 })];
        for (index, phase) in phases.into_iter().enumerate() {
            body.push(phase.accumulate);
            body.push(Node::barrier());
            body.extend(phase.reductions);
            if !phase.publish.is_empty() {
                body.push(Node::if_then(
                    Expr::and(
                        Expr::is_first_workgroup(),
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
                    body.push(Node::barrier());
                }
            }
        }
        body.extend(writeback);
        Program::wrapped(buffers, workgroup_size, vec![wrap_region(generator, body, None)])
    }

    /// Build a tiled mean reduction program.
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    #[must_use]
    pub(crate) fn tiled_mean(
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

    /// Build a tiled Welford parallel variance reduction program.
    #[must_use]
    pub(crate) fn tiled_variance(
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
            Node::let_bind("local", Expr::LocalId { axis: 0 }),
            Node::if_then(
                Expr::is_first_workgroup(),
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
            Node::barrier(),
        ];

        // Workgroup-local tree reduction for Welford triples.
        let wg0_guard = Expr::is_first_workgroup();
        let mut stride = tile.next_power_of_two() / 2;
        while stride > 0 {
            body.push(Node::if_then(
                Expr::and(
                    wg0_guard.clone(),
                    Expr::lt(Expr::var("local"), Expr::u32(stride)),
                ),
                vec![Node::if_then(
                    Expr::lt(
                        Expr::add(Expr::var("local"), Expr::u32(stride)),
                        Expr::u32(tile),
                    ),
                    vec![
                        Node::let_bind(
                            "other_idx",
                            Expr::add(Expr::var("local"), Expr::u32(stride)),
                        ),
                        Node::let_bind("n_a", Expr::load("var_n_scratch", Expr::var("local"))),
                        Node::let_bind("n_b", Expr::load("var_n_scratch", Expr::var("other_idx"))),
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
                                                Expr::div(Expr::var("n_b_f"), Expr::var("n_ab_f")),
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
            ));
            body.push(Node::barrier());
            stride /= 2;
        }

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
                Expr::is_first_workgroup(),
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

    /// Build a tiled RMS normalization program.
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    #[must_use]
    pub(crate) fn tiled_rms_norm(
        generator: &'static str,
        input: &str,
        output: &str,
        n: u32,
        eps: f32,
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
                "rms_acc",
                Expr::f32(0.0),
                "rms_scratch",
                |idx, acc| {
                    let val = Expr::load(input, idx);
                    Expr::add(acc, Expr::mul(val.clone(), val))
                },
            ),
            reductions: vec![workgroup_tree::sum_f32_child(
                generator,
                tile,
                "rms_scratch",
                WorkgroupReductionScope::FirstWorkgroup,
            )],
            publish: vec![Node::Store {
                buffer: "rms_mean_sq".into(),
                index: Expr::u32(0),
                value: Expr::div(
                    Expr::load("rms_scratch", Expr::u32(0)),
                    Expr::f32(n as f32),
                ),
            }],
        };

        Self::new(
            generator,
            vec![
                BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
                BufferDecl::workgroup("rms_scratch", tile, DataType::F32),
                BufferDecl::workgroup("rms_mean_sq", 1, DataType::F32),
                BufferDecl::output(output, 1, DataType::F32).with_count(n),
            ],
            [tile, 1, 1],
        )
        .with_phase(phase)
        .with_writeback(crate::builder::strided_writeback_child(
            generator,
            tile,
            chunks,
            n,
            output,
            vec![Node::let_bind(
                "inv_rms",
                Expr::div(
                    Expr::f32(1.0),
                    Expr::sqrt(Expr::add(
                        Expr::load("rms_mean_sq", Expr::u32(0)),
                        Expr::f32(eps),
                    )),
                ),
            )],
            |idx| Expr::mul(Expr::load(input, idx), Expr::var("inv_rms")),
        ))
        .build()
    }

    /// Build a tiled layer normalization program with fused dual accumulation.
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    #[must_use]
    pub(crate) fn tiled_layer_norm(
        generator: &'static str,
        input: &str,
        output: &str,
        n: u32,
        eps: f32,
        tile: u32,
    ) -> Program {
        let tile = tile.max(1);
        let chunks = n.div_ceil(tile);
        let phase = ReductionPhase {
            accumulate: crate::builder::strided_accumulate2_child(
                generator,
                tile,
                chunks,
                n,
                (
                    "sum_acc",
                    Expr::f32(0.0),
                    "ln_sum_scratch",
                    |idx, acc| Expr::add(acc, Expr::load(input, idx)),
                ),
                (
                    "sq_acc",
                    Expr::f32(0.0),
                    "ln_sq_scratch",
                    |idx, acc| {
                        let val = Expr::load(input, idx);
                        Expr::add(acc, Expr::mul(val.clone(), val))
                    },
                ),
            ),
            reductions: vec![
                workgroup_tree::sum_f32_child(
                    generator,
                    tile,
                    "ln_sum_scratch",
                    WorkgroupReductionScope::FirstWorkgroup,
                ),
                workgroup_tree::sum_f32_child(
                    generator,
                    tile,
                    "ln_sq_scratch",
                    WorkgroupReductionScope::FirstWorkgroup,
                ),
            ],
            publish: vec![
                Node::let_bind(
                    "mean",
                    Expr::div(
                        Expr::load("ln_sum_scratch", Expr::u32(0)),
                        Expr::f32(n as f32),
                    ),
                ),
                Node::Store {
                    buffer: "ln_mean".into(),
                    index: Expr::u32(0),
                    value: Expr::var("mean"),
                },
                Node::let_bind(
                    "mean_sq",
                    Expr::div(
                        Expr::load("ln_sq_scratch", Expr::u32(0)),
                        Expr::f32(n as f32),
                    ),
                ),
                Node::let_bind(
                    "variance",
                    Expr::sub(
                        Expr::var("mean_sq"),
                        Expr::mul(Expr::var("mean"), Expr::var("mean")),
                    ),
                ),
                Node::Store {
                    buffer: "ln_inv_std".into(),
                    index: Expr::u32(0),
                    value: Expr::div(
                        Expr::f32(1.0),
                        Expr::sqrt(Expr::add(
                            Expr::max(Expr::var("variance"), Expr::f32(0.0)),
                            Expr::f32(eps),
                        )),
                    ),
                },
            ],
        };

        Self::new(
            generator,
            vec![
                BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
                BufferDecl::workgroup("ln_sum_scratch", tile, DataType::F32),
                BufferDecl::workgroup("ln_sq_scratch", tile, DataType::F32),
                BufferDecl::workgroup("ln_mean", 1, DataType::F32),
                BufferDecl::workgroup("ln_inv_std", 1, DataType::F32),
                BufferDecl::output(output, 1, DataType::F32).with_count(n),
            ],
            [tile, 1, 1],
        )
        .with_phase(phase)
        .with_writeback(crate::builder::strided_writeback_child(
            generator,
            tile,
            chunks,
            n,
            output,
            vec![
                Node::let_bind("mean", Expr::load("ln_mean", Expr::u32(0))),
                Node::let_bind("inv_std", Expr::load("ln_inv_std", Expr::u32(0))),
            ],
            |idx| {
                Expr::mul(
                    Expr::sub(Expr::load(input, idx), Expr::var("mean")),
                    Expr::var("inv_std"),
                )
            },
        ))
        .build()
    }

    /// Build a 2-phase tiled Softmax program (Max -> SumExp -> Writeback).
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    #[must_use]
    pub(crate) fn tiled_softmax(
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
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    #[must_use]
    pub(crate) fn tiled_dot(
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
    #[cfg(feature = "reduce")]
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

    /// Build an atomic nonzero boolean reduction program over u32 elements.
    #[cfg(feature = "reduce")]
    #[must_use]
    pub(crate) fn atomic_nonzero_bool_reduction(
        op_id: &'static str,
        input: &str,
        output: &str,
        count: u32,
        kind: crate::reduce::atomic_scalar::AtomicBoolReduceKind,
    ) -> Program {
        crate::reduce::atomic_scalar::atomic_nonzero_bool_reduce_u32(input, output, count, kind, op_id)
    }

    /// Build a prefix scan program.
    #[cfg(feature = "reduce")]
    #[must_use]
    pub(crate) fn prefix_scan(
        op_id: &'static str,
        input: &str,
        output: &str,
        n: u32,
        kind: crate::math::prefix_scan::ScanKind,
    ) -> Program {
        crate::math::prefix_scan::prefix_scan_with_op_id(input, output, n, kind, op_id)
    }
}

/// Convenience function assembling a [`TiledReduceSpec`] into a [`Program`].
#[must_use]
pub(crate) fn tiled_reduce_program(spec: TiledReduceSpec) -> Program {
    ReductionComposer::from_spec(spec).build()
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
        let phase_terminal = ReductionPhase::new(
            Node::let_bind("acc", Expr::f32(0.0)),
            vec![],
            vec![Node::store("out", Expr::u32(0), Expr::var("acc"))],
        );
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

        let format_terminal = format!("{:?}", program_terminal.entry());
        // Barrier after accumulate exists
        assert!(format_terminal.contains("Barrier"));

        // 2. Multi-phase -> intermediate phase publish has trailing barrier.
        let phase1 = ReductionPhase::new(
            Node::let_bind("acc1", Expr::f32(0.0)),
            vec![],
            vec![Node::store("scratch_stat", Expr::u32(0), Expr::var("acc1"))],
        );
        let phase2 = ReductionPhase::new(
            Node::let_bind("acc2", Expr::f32(0.0)),
            vec![],
            vec![Node::store("out", Expr::u32(0), Expr::var("acc2"))],
        );
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

        let format_multi = format!("{:?}", program_multi.entry());
        assert!(format_multi.contains("Barrier"));
    }

    #[test]
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    fn tiled_mean_composition_structure() {
        let program = ReductionComposer::tiled_mean("test::mean", "in", "out", 1024, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 3);
        assert_eq!(program.buffers()[0].name.as_ref(), "in");
        assert_eq!(program.buffers()[1].name.as_ref(), "mean_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "out");
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

    #[test]
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    fn tiled_rms_norm_composition_structure() {
        let program = ReductionComposer::tiled_rms_norm("test::rms", "in", "out", 512, 1e-5, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 4);
        assert_eq!(program.buffers()[0].name.as_ref(), "in");
        assert_eq!(program.buffers()[1].name.as_ref(), "rms_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "rms_mean_sq");
        assert_eq!(program.buffers()[3].name.as_ref(), "out");
    }

    #[test]
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    fn tiled_layer_norm_composition_structure() {
        let program = ReductionComposer::tiled_layer_norm("test::ln", "in", "out", 512, 1e-5, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 6);
        assert_eq!(program.buffers()[0].name.as_ref(), "in");
        assert_eq!(program.buffers()[1].name.as_ref(), "ln_sum_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "ln_sq_scratch");
        assert_eq!(program.buffers()[3].name.as_ref(), "ln_mean");
        assert_eq!(program.buffers()[4].name.as_ref(), "ln_inv_std");
        assert_eq!(program.buffers()[5].name.as_ref(), "out");
    }

    #[test]
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
    fn tiled_softmax_composition_structure() {
        let program = ReductionComposer::tiled_softmax("test::softmax", "in", "out", 512, [256, 1, 1]);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 4);
        assert_eq!(program.buffers()[0].name.as_ref(), "in");
        assert_eq!(program.buffers()[1].name.as_ref(), "softmax_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "softmax_max");
        assert_eq!(program.buffers()[3].name.as_ref(), "out");
    }

    #[test]
    #[cfg(all(feature = "reduce", feature = "builder-ops"))]
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
    #[cfg(feature = "reduce")]
    fn atomic_scalar_reductions_structure() {
        use crate::reduce::atomic_scalar::{AtomicBoolReduceKind, AtomicReduceKind};

        for kind in [
            AtomicReduceKind::Sum,
            AtomicReduceKind::Min,
            AtomicReduceKind::Max,
            AtomicReduceKind::PopcountSum,
            AtomicReduceKind::CountNonZero,
        ] {
            let p = ReductionComposer::atomic_scalar_reduction("test::atomic", "in", "out", 128, kind);
            assert_eq!(p.workgroup_size(), [256, 1, 1]);
        }

        for kind in [AtomicBoolReduceKind::AnyNonZero, AtomicBoolReduceKind::AllNonZero] {
            let p = ReductionComposer::atomic_nonzero_bool_reduction("test::atomic_bool", "in", "out", 128, kind);
            assert_eq!(p.workgroup_size(), [256, 1, 1]);
        }
    }

    #[test]
    #[cfg(feature = "reduce")]
    fn prefix_scan_structure() {
        use crate::math::prefix_scan::ScanKind;
        let p_inc = ReductionComposer::prefix_scan("test::scan_inc", "in", "out", 64, ScanKind::InclusiveSum);
        assert_eq!(p_inc.workgroup_size(), [64, 1, 1]);
        let p_exc = ReductionComposer::prefix_scan("test::scan_exc", "in", "out", 64, ScanKind::ExclusiveSum);
        assert_eq!(p_exc.workgroup_size(), [64, 1, 1]);
    }
}
