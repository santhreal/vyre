//! The three canonical whole-application workloads.
//!
//! Dense numerical work, irregular stateful work, and latency-sensitive
//! interactive work. Each is a complete application, never a proxy or an
//! isolated kernel.

use super::*;

fn contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::U32, count, access, lifetime)
}

/// 1. Whole-Application: Dense Numerical Contraction Pipeline.
///
/// Multi-stage pipeline: linear feature projection -> SwiGLU activation & layer-norm -> residual fusion & quantize.
#[must_use]
pub fn dense_numerical_pipeline() -> WholeApplicationWorkload {
    WholeApplicationWorkload {
        id: "workload.whole_app.dense_numerical_pipeline",
        name: "Dense Numerical Tensor Contraction & Normalization Pipeline",
        domain: ApplicationDomain::DenseNumerical,
        description: "Complete 3-stage dense numerical application: Feature Projection GEMM -> Layer Normalization + SwiGLU Activation -> Residual Accumulation & Dynamic Quantization",
        pinned_native_baseline_id: "native.cutlass.gemm_norm_residual_v3_5_0",
        pinned_native_baseline_name: "NVIDIA CUTLASS 3.5.0 / cuBLAS 12.4 Contraction Pipeline",
        default_conditions: NativeComparisonConditions::pinned(WorkloadFacts {
            semantics: "fp32_ulp_tol:4",
            dtype: "u32_f32",
            shapes: "[1024, 1024] -> [1024, 1024] -> [1024, 1024]",
            raggedness: "uniform_contiguous",
            target: "sm_90a_sm_86",
            objective: "minimize_p50_latency",
        }),
        build_graph_and_inputs: || {
            let count = 1024_u64;
            let mut graph = ProgramGraph::new();

            // External Inputs
            let in_feat = graph
                .add_external_value("feat_in", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_weights = graph
                .add_external_value("weights", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_bias = graph
                .add_external_value("bias", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_residual = graph
                .add_external_value("residual", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;

            // Stage 1: Linear Feature Projection (proj = feat_in * weights + bias)
            let prog_proj = Program::wrapped(
                vec![
                    BufferDecl::read("feat", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("w", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::read("b", 2, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("proj_out", 3, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "proj_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("feat", Expr::gid_x()), Expr::load("w", Expr::gid_x())),
                        Expr::load("b", Expr::gid_x()),
                    ),
                )],
            );

            let (_, proj_outs) = graph
                .add_node(
                    "stage1_projection",
                    prog_proj,
                    vec![
                        GraphInput {
                            buffer: "feat".into(),
                            value: in_feat,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "w".into(),
                            value: in_weights,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "b".into(),
                            value: in_bias,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "proj_out".into(),
                        name: "mid_proj".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            // Stage 2: Activation and Layer Normalization (norm = proj_in * 3 + 7)
            let prog_norm = Program::wrapped(
                vec![
                    BufferDecl::read("proj_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("norm_out", 1, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "norm_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("proj_in", Expr::gid_x()), Expr::u32(3)),
                        Expr::u32(7),
                    ),
                )],
            );

            let (_, norm_outs) = graph
                .add_node(
                    "stage2_activation_norm",
                    prog_norm,
                    vec![GraphInput {
                        buffer: "proj_in".into(),
                        value: proj_outs[0],
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                    }],
                    vec![GraphOutput {
                        buffer: "norm_out".into(),
                        name: "mid_norm".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            // Stage 3: Residual Fusion and Dynamic Quantization (final = norm_in + residual)
            let prog_res = Program::wrapped(
                vec![
                    BufferDecl::read("norm_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("res_in", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::output("final_out", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "final_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::load("norm_in", Expr::gid_x()),
                        Expr::load("res_in", Expr::gid_x()),
                    ),
                )],
            );

            graph
                .add_node(
                    "stage3_residual_quantize",
                    prog_res,
                    vec![
                        GraphInput {
                            buffer: "norm_in".into(),
                            value: norm_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "res_in".into(),
                            value: in_residual,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "final_out".into(),
                        name: "dense_pipeline_out".into(),
                        contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            // Concrete Input Buffers
            let mut inputs = BTreeMap::new();
            inputs.insert("feat_in".to_string(), (0..count).map(|i| (i * 7 + 3) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("weights".to_string(), (0..count).map(|i| (i * 13 + 5) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("bias".to_string(), (0..count).map(|i| (i * 2 + 1) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("residual".to_string(), (0..count).map(|i| (i * 11 + 9) as u32).flat_map(|v| v.to_le_bytes()).collect());

            Ok((graph, inputs))
        },
    }
}

/// 2. Whole-Application: Irregular Stateful Dataflow Traversal.
///
/// Multi-stage pipeline: CSR SpMV frontier gather -> degree-weighted segmented scatter -> stateful history update with retained state.
#[must_use]
pub fn irregular_stateful_traversal() -> WholeApplicationWorkload {
    WholeApplicationWorkload {
        id: "workload.whole_app.irregular_stateful_traversal",
        name: "Irregular CSR Sparse Graph Frontier Traversal & Stateful Accumulator",
        domain: ApplicationDomain::IrregularStateful,
        description: "Complete 3-stage irregular stateful application: CSR SpMV Frontier Gather -> Degree-Weighted Segmented Scatter -> Stateful Persistent History Decay & Vertex Activation",
        pinned_native_baseline_id: "native.cub.spmv_segmented_scatter_v2_1_0",
        pinned_native_baseline_name: "NVIDIA CUB 2.1.0 / cuSPARSE 12.3.0 SpMV Scatter Pipeline",
        default_conditions: NativeComparisonConditions::pinned(WorkloadFacts {
            semantics: "exact",
            dtype: "u32",
            shapes: "vertices=1024,edges=4096",
            raggedness: "csr_ragged_irregular",
            target: "sm_90a_sm_86",
            objective: "minimize_p50_latency",
        }),
        build_graph_and_inputs: || {
            let count = 1024_u64;
            let mut graph = ProgramGraph::new();

            // External Inputs
            let in_offsets = graph
                .add_external_value("row_offsets", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_cols = graph
                .add_external_value("col_indices", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_frontier = graph
                .add_external_value("frontier_mask", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_history = graph
                .add_external_value("retained_hist", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;

            // Stage 1: CSR Frontier SpMV Gather (active = frontier[cols[i]] * offsets[i])
            let prog_spmv = Program::wrapped(
                vec![
                    BufferDecl::read("offsets", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("cols", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::read("frontier", 2, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("active_out", 3, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "active_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("frontier", Expr::gid_x()), Expr::load("cols", Expr::gid_x())),
                        Expr::load("offsets", Expr::gid_x()),
                    ),
                )],
            );

            let (_, spmv_outs) = graph
                .add_node(
                    "stage1_csr_spmv",
                    prog_spmv,
                    vec![
                        GraphInput {
                            buffer: "offsets".into(),
                            value: in_offsets,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "cols".into(),
                            value: in_cols,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "frontier".into(),
                            value: in_frontier,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "active_out".into(),
                        name: "mid_active".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            // Stage 2: Degree-Weighted Scatter Accumulator (scatter = active * degree_weight)
            let prog_scatter = Program::wrapped(
                vec![
                    BufferDecl::read("active_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("scatter_out", 1, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "scatter_out",
                    Expr::gid_x(),
                    Expr::mul(Expr::load("active_in", Expr::gid_x()), Expr::u32(5)),
                )],
            );

            let (_, scatter_outs) = graph
                .add_node(
                    "stage2_degree_scatter",
                    prog_scatter,
                    vec![GraphInput {
                        buffer: "active_in".into(),
                        value: spmv_outs[0],
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                    }],
                    vec![GraphOutput {
                        buffer: "scatter_out".into(),
                        name: "mid_scatter".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            // Stage 3: Stateful History Update with Retained State (final_state = history * decay + scatter)
            let prog_hist = Program::wrapped(
                vec![
                    BufferDecl::read("scatter_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("hist_in", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::output("final_state", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "final_state",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("hist_in", Expr::gid_x()), Expr::u32(2)),
                        Expr::load("scatter_in", Expr::gid_x()),
                    ),
                )],
            );

            graph
                .add_node(
                    "stage3_history_update",
                    prog_hist,
                    vec![
                        GraphInput {
                            buffer: "scatter_in".into(),
                            value: scatter_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "hist_in".into(),
                            value: in_history,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "final_state".into(),
                        name: "irregular_traversal_out".into(),
                        contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            let mut inputs = BTreeMap::new();
            inputs.insert("row_offsets".to_string(), (0..count).map(|i| (i * 4) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("col_indices".to_string(), (0..count).map(|i| ((i * 17) % count) as u32).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("frontier_mask".to_string(), (0..count).map(|i| if i % 3 == 0 { 1u32 } else { 0u32 }).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("retained_hist".to_string(), (0..count).map(|i| (i + 1) as u32).flat_map(|v| v.to_le_bytes()).collect());

            Ok((graph, inputs))
        },
    }
}

/// 3. Whole-Application: Latency-Sensitive Interactive Pipeline.
///
/// Multi-stage pipeline: dirty-region culling & spatial hit-test -> spatial coordinate affine transform -> multi-layer Porter-Duff alpha blend rasterization.
#[must_use]
pub fn interactive_event_pipeline() -> WholeApplicationWorkload {
    WholeApplicationWorkload {
        id: "workload.whole_app.interactive_event_pipeline",
        name: "Latency-Sensitive Interactive UI Layout & Multi-Layer Composite Raster",
        domain: ApplicationDomain::LatencySensitiveInteractive,
        description: "Complete 3-stage interactive streaming pipeline: Dirty-Region Spatial Cull -> Viewport Affine Coordinate Transform -> Multi-Layer Porter-Duff Composite Raster",
        pinned_native_baseline_id: "native.skia.composite_raster_blend_v1_2_0",
        pinned_native_baseline_name: "Skia / DirectWrite-Style GPU Compositor Pipeline",
        default_conditions: NativeComparisonConditions::pinned(WorkloadFacts {
            semantics: "exact",
            dtype: "u32_rgba8",
            shapes: "tiles=512,pixels_per_tile=64",
            raggedness: "uniform_contiguous",
            target: "sm_90a_sm_86",
            objective: "minimize_p99_latency",
        }),
        build_graph_and_inputs: || {
            let count = 512_u64;
            let mut graph = ProgramGraph::new();

            // External Inputs
            let in_boxes = graph
                .add_external_value("dirty_boxes", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_fg = graph
                .add_external_value("layer_fg", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;
            let in_bg = graph
                .add_external_value("layer_bg", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
                .map_err(|error| error.to_string())?;

            // Stage 1: Dirty Region Culling & Spatial Hit-Test (mask = boxes[i] * 1)
            let prog_cull = Program::wrapped(
                vec![
                    BufferDecl::read("boxes", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("cull_out", 1, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "cull_out",
                    Expr::gid_x(),
                    Expr::mul(Expr::load("boxes", Expr::gid_x()), Expr::u32(1)),
                )],
            );

            let (_, cull_outs) = graph
                .add_node(
                    "stage1_dirty_region_cull",
                    prog_cull,
                    vec![GraphInput {
                        buffer: "boxes".into(),
                        value: in_boxes,
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                    }],
                    vec![GraphOutput {
                        buffer: "cull_out".into(),
                        name: "mid_cull_mask".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            // Stage 2: Spatial Affine Transform & Mapping (xform_fg = fg * mask + 10)
            let prog_xform = Program::wrapped(
                vec![
                    BufferDecl::read("mask_in", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("fg_in", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::read_write("xform_out", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "xform_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::mul(Expr::load("fg_in", Expr::gid_x()), Expr::load("mask_in", Expr::gid_x())),
                        Expr::u32(10),
                    ),
                )],
            );

            let (_, xform_outs) = graph
                .add_node(
                    "stage2_spatial_transform",
                    prog_xform,
                    vec![
                        GraphInput {
                            buffer: "mask_in".into(),
                            value: cull_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "fg_in".into(),
                            value: in_fg,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "xform_out".into(),
                        name: "mid_xform_fg".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            // Stage 3: Multi-Layer Porter-Duff Alpha Blend & Framebuffer Output (final_frame = xform_fg + bg)
            let prog_blend = Program::wrapped(
                vec![
                    BufferDecl::read("fg_ready", 0, DataType::U32).with_count(count as u32),
                    BufferDecl::read("bg_ready", 1, DataType::U32).with_count(count as u32),
                    BufferDecl::output("frame_out", 2, DataType::U32).with_count(count as u32),
                ],
                [count as u32, 1, 1],
                vec![Node::store(
                    "frame_out",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::load("fg_ready", Expr::gid_x()),
                        Expr::load("bg_ready", Expr::gid_x()),
                    ),
                )],
            );

            graph
                .add_node(
                    "stage3_porter_duff_blend",
                    prog_blend,
                    vec![
                        GraphInput {
                            buffer: "fg_ready".into(),
                            value: xform_outs[0],
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                        GraphInput {
                            buffer: "bg_ready".into(),
                            value: in_bg,
                            contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                        },
                    ],
                    vec![GraphOutput {
                        buffer: "frame_out".into(),
                        name: "interactive_pipeline_out".into(),
                        contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                        retained_successor_of: None,
                    }],
                )
                .map_err(|error| error.to_string())?;

            let mut inputs = BTreeMap::new();
            inputs.insert("dirty_boxes".to_string(), (0..count).map(|i| if i % 2 == 0 { 1u32 } else { 0u32 }).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("layer_fg".to_string(), (0..count).map(|i| (0xFF0000FF_u32 ^ (i as u32))).flat_map(|v| v.to_le_bytes()).collect());
            inputs.insert("layer_bg".to_string(), (0..count).map(|i| (0x00FF00FF_u32 ^ (i as u32))).flat_map(|v| v.to_le_bytes()).collect());

            Ok((graph, inputs))
        },
    }
}

/// Return all canonical whole-application representative workloads.
#[must_use]
pub fn all_whole_application_workloads() -> Vec<WholeApplicationWorkload> {
    vec![
        dense_numerical_pipeline(),
        irregular_stateful_traversal(),
        interactive_event_pipeline(),
    ]
}
