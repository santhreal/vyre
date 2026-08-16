//! Semantic ownership, placement, and consolidation closure for registered operations.
//!
//! Composition is judged from the live program and registration metadata. Source
//! similarity remains useful for discovery, but it never establishes ownership.

use super::*;
use std::path::{Path, PathBuf};

/// Organization role of a production file under `vyre-libs/src/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum FileRole {
    /// Registered operation implementation.
    OperationImplementation,
    /// Shared semantic builder consumed by registered operations.
    SharedBuilder,
    /// Domain contract, type, or algorithm helper module.
    DomainContractOrType,
    /// Crate-level plumbing.
    CratePlumbing,
}

/// Classify all organization roles for one production file under `vyre-libs/src/`.
/// Returns all matching roles so overlapping/conflicting classifications are detected.
pub(super) fn classify_file_roles(
    path: &str,
    registered_sources: &BTreeSet<&str>,
) -> Vec<FileRole> {
    let mut roles = Vec::new();
    let normalized = path.replace('\\', "/");

    // 1. Operation Implementation: explicitly registered in operation registry
    if registered_sources.contains(normalized.as_str()) {
        roles.push(FileRole::OperationImplementation);
    }

    let Some(rel) = normalized.strip_prefix("vyre-libs/src/") else {
        return roles;
    };
    let parts: Vec<&str> = rel.split('/').collect();
    let filename = parts.last().copied().unwrap_or_default();

    // 2. Crate-level plumbing
    if matches!(
        rel,
        "lib.rs" | "prelude.rs" | "fixture_bytes.rs" | "test_parity_oracles.rs"
    ) || matches!(parts.first().copied(), Some("intern" | "plumbing"))
    {
        roles.push(FileRole::CratePlumbing);
    }

    // 3. Shared builder
    if parts.first().copied() == Some("builder")
        || matches!(
            filename,
            "builder.rs" | "builders.rs" | "build.rs" | "emit.rs"
        )
    {
        roles.push(FileRole::SharedBuilder);
    }

    // 4. Domain contract, type, or algorithm supporting module
    if is_authorized_domain_contract_or_type(rel) {
        roles.push(FileRole::DomainContractOrType);
    }

    roles
}

/// Whether a relative path under `vyre-libs/src/` is an authorized domain contract or type.
pub(super) fn is_authorized_domain_contract_or_type(rel_path: &str) -> bool {
    AUTHORIZED_DOMAIN_CONTRACTS.binary_search(&rel_path).is_ok()
}

/// Single-role classification helper for simple lookups.
pub(super) fn classify_file_role(
    path: &str,
    registered_sources: &BTreeSet<&str>,
) -> Option<FileRole> {
    let roles = classify_file_roles(path, registered_sources);
    if roles.len() == 1 {
        Some(roles[0])
    } else {
        None
    }
}

/// Judge semantic ownership in both directions: every attributed child must
/// exist, every operation with the same semantic body must have one owner, and
/// every file in `vyre-libs` must have exactly one mechanically checkable role.
pub(super) fn check_semantic_organization(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("[11/11] Semantic ownership, placement, and consolidation".to_string());
    let mut findings = Vec::new();
    let known = ops.iter().map(|op| op.id.as_str()).collect::<BTreeSet<_>>();
    let registered_sources = ops
        .iter()
        .map(|op| op.source_file.as_str())
        .collect::<BTreeSet<_>>();

    for op in ops {
        check_source_placement(op, &mut findings);
        for node in op.program.entry() {
            check_attribution(op, node, &known, &mut findings);
        }
    }

    for (index, left) in ops.iter().enumerate() {
        for right in ops.iter().skip(index + 1) {
            check_pair(left, right, &mut findings);
        }
    }

    check_vyre_libs_file_roles(&registered_sources, &mut findings);

    let count = findings.len();
    for finding in findings {
        report.find(finding);
    }
    if count == 0 {
        report
            .note("  semantic ownership and file roles are closed in both directions".to_string());
    }
    count
}

fn check_vyre_libs_file_roles(registered_sources: &BTreeSet<&str>, findings: &mut Vec<Finding>) {
    let Some(root) = workspace_root() else {
        findings.push(Finding::new(
            "workspace root is not reachable",
            "run from the vyre workspace checkout root",
        ));
        return;
    };
    let libs_src = root.join("vyre-libs/src");
    if !libs_src.is_dir() {
        findings.push(Finding::new(
            format!(
                "production source root `{}` is missing or is not a directory",
                libs_src.display()
            ),
            "restore a readable vyre-libs/src directory before judging file roles",
        ));
        return;
    }

    let files = match rust_files_under(&libs_src) {
        Ok(files) => files,
        Err(error) => {
            findings.push(Finding::new(
                format!(
                    "cannot walk production files under `{}`: {error}",
                    libs_src.display()
                ),
                "repair the unreadable path so every production file can be classified",
            ));
            return;
        }
    };
    for path in files {
        let rel_path = match path.strip_prefix(&root) {
            Ok(relative) => relative.to_string_lossy().replace('\\', "/"),
            Err(error) => {
                findings.push(Finding::new(
                    format!(
                        "production file `{}` is outside workspace root `{}`: {error}",
                        path.display(),
                        root.display()
                    ),
                    "keep the authoritative production walk inside the workspace root",
                ));
                continue;
            }
        };

        let roles = classify_file_roles(&rel_path, registered_sources);
        if roles.is_empty() {
            findings.push(Finding::in_file(
                &rel_path,
                format!("production file `{rel_path}` has no recognized organization role"),
                "assign it to an operation, a shared builder, a domain type/contract, or plumbing",
            ));
        } else if roles.len() > 1 {
            findings.push(Finding::in_file(
                &rel_path,
                format!(
                    "production file `{rel_path}` matches multiple conflicting organization roles: {roles:?}"
                ),
                "keep exactly one organization role per file (Section 182.12.3)",
            ));
        }
    }
}

fn rust_files_under(root: &Path) -> Result<Vec<PathBuf>, walkdir::Error> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|ext| ext.to_str()) == Some("rs")
        {
            files.push(entry.into_path());
        }
    }
    Ok(files)
}

pub(super) const AUTHORIZED_DOMAIN_CONTRACTS: &[&str] = &[
    "analysis/cost_model.rs",
    "analysis/dataflow_fixpoint/delta_maintenance.rs",
    "analysis/dataflow_fixpoint/dense_matrix.rs",
    "analysis/dataflow_fixpoint/fixpoint_comparison.rs",
    "analysis/dataflow_fixpoint/gpu_dispatch.rs",
    "analysis/dataflow_fixpoint/mod.rs",
    "analysis/dataflow_fixpoint/reference_gemm.rs",
    "analysis/dataflow_fixpoint/scc_decomposition.rs",
    "analysis/decision_telemetry.rs",
    "analysis/diagnostic_aggregation.rs",
    "analysis/effect_signature.rs",
    "analysis/incremental_invalidation.rs",
    "analysis/knowledge_compile_pass_precondition.rs",
    "analysis/mod.rs",
    "analysis/persistent_fixpoint_program.rs",
    "bitset/frontier.rs",
    "bitset/mod.rs",
    "bitset/relation.rs",
    "bitset/unary_word.rs",
    "decode/buffers.rs",
    "decode/inflate_tests.rs",
    "decode/mod.rs",
    "decode/scan.rs",
    "decode/streaming.rs",
    "device/device_resident_token_fact_graph.rs",
    "device/gpu_probe_contract.rs",
    "device/memory_ownership_contract.rs",
    "device/mod.rs",
    "encoding/bitset_compression.rs",
    "encoding/bitset_mask_algebra.rs",
    "encoding/bitset_summary.rs",
    "encoding/bitset_transform_pipeline.rs",
    "encoding/matching_diagnostic_compaction.rs",
    "encoding/matroid_exact_megakernel.rs",
    "encoding/matroid_megakernel_scheduler.rs",
    "encoding/mod.rs",
    "encoding/nn_attention_paging.rs",
    "encoding/parsing_dispatch_pipeline.rs",
    "encoding/reduce_dispatch_pipeline.rs",
    "encoding/reduction_metrics.rs",
    "encoding/scallop_provenance.rs",
    "encoding/scallop_provenance_wide.rs",
    "encoding/vsa_fingerprint.rs",
    "fixpoint/mod.rs",
    "fixpoint/routing_contract.rs",
    "geom/mod.rs",
    "graph/adaptive_traverse/cpu_reference.rs",
    "graph/adaptive_traverse/dense_step.rs",
    "graph/adaptive_traverse/four_russians.rs",
    "graph/adaptive_traverse/frontier_plan.rs",
    "graph/adaptive_traverse/mod.rs",
    "graph/adaptive_traverse/mode_selection.rs",
    "graph/adaptive_traverse/plan_cache_key.rs",
    "graph/adaptive_traverse/sparse_dense_step.rs",
    "graph/adaptive_traverse/test_graphs.rs",
    "graph/adjustment_set.rs",
    "graph/alias_registry.rs",
    "graph/chebyshev_filter.rs",
    "graph/csr_bidirectional.rs",
    "graph/csr_closure_entry_points.rs",
    "graph/csr_closure_inputs.rs",
    "graph/csr_forward_or_changed/batched_frontier_words.rs",
    "graph/csr_forward_or_changed/body.rs",
    "graph/csr_forward_or_changed/cpu_ref.rs",
    "graph/csr_forward_or_changed/dispatch_plan.rs",
    "graph/csr_forward_or_changed/launch_plan.rs",
    "graph/csr_forward_or_changed/layout.rs",
    "graph/csr_forward_or_changed/mod.rs",
    "graph/csr_forward_or_changed/plan.rs",
    "graph/csr_forward_or_changed/program_dispatch.rs",
    "graph/csr_forward_or_changed/program_parallel.rs",
    "graph/csr_forward_or_changed/program_parallel_batch.rs",
    "graph/csr_forward_or_changed/program_parallel_batch_global.rs",
    "graph/csr_forward_or_changed/program_serial.rs",
    "graph/csr_forward_or_changed/tests/batch_validation.rs",
    "graph/csr_forward_or_changed/tests/cpu_reference_and_dispatch.rs",
    "graph/csr_forward_or_changed/tests/dispatch_contract_tests.rs",
    "graph/csr_forward_or_changed/tests/dynamic_changed_slot_tests.rs",
    "graph/csr_forward_or_changed/tests/mod.rs",
    "graph/csr_forward_or_changed/validate.rs",
    "graph/csr_frontier_queue/cpu_reference.rs",
    "graph/csr_frontier_queue/emitted_program_shape.rs",
    "graph/csr_frontier_queue/graph_validation.rs",
    "graph/csr_frontier_queue/mod.rs",
    "graph/csr_frontier_queue/packed_word_compaction.rs",
    "graph/csr_frontier_queue/queue_compaction.rs",
    "graph/csr_frontier_queue/queue_traverse.rs",
    "graph/csr_frontier_queue/sizing_diagnostics.rs",
    "graph/csr_frontier_queue/word_block_scan.rs",
    "graph/csr_frontier_queue/word_block_scatter.rs",
    "graph/csr_frontier_shard.rs",
    "graph/csr_frontier_step.rs",
    "graph/csr_queue_delta/mod.rs",
    "graph/csr_queue_delta/strided.rs",
    "graph/csr_queue_split/mod.rs",
    "graph/csr_queue_split/tests/mod.rs",
    "graph/dispatch/adaptive_traverse/mod.rs",
    "graph/dispatch/adaptive_traverse/reference.rs",
    "graph/dispatch/adaptive_traverse/resident.rs",
    "graph/dispatch/adaptive_traverse/resident_scratch.rs",
    "graph/dispatch/adaptive_traverse/resident_steps.rs",
    "graph/dispatch/adaptive_traverse/tests/lifecycle_contracts.rs",
    "graph/dispatch/adaptive_traverse/tests/mod.rs",
    "graph/dispatch/adaptive_traverse/tests/recording_dispatcher.rs",
    "graph/dispatch/adaptive_traverse/tests/selector_layout_contracts.rs",
    "graph/dispatch/adaptive_traverse/tests/source_contracts.rs",
    "graph/dispatch/adaptive_traverse/tests/sparse_dense_contracts.rs",
    "graph/dispatch/adaptive_traverse/tests/sparse_queue_materializer_contracts.rs",
    "graph/dispatch/adaptive_traverse/tests/sparse_queue_shape_contracts.rs",
    "graph/dispatch/adaptive_traverse/tests/upload_contracts.rs",
    "graph/dispatch/adaptive_traverse/tests/zero_frontier_contracts.rs",
    "graph/dispatch/adaptive_traverse/upload.rs",
    "graph/dispatch/alias_registry/mod.rs",
    "graph/dispatch/alias_registry/tests/mod.rs",
    "graph/dispatch/csr_bidirectional/closure.rs",
    "graph/dispatch/csr_bidirectional/dispatch.rs",
    "graph/dispatch/csr_bidirectional/mod.rs",
    "graph/dispatch/csr_bidirectional/reference.rs",
    "graph/dispatch/csr_bidirectional/tests/mod.rs",
    "graph/dispatch/csr_bidirectional/tests/reference_closure_tests.rs",
    "graph/dispatch/csr_forward_or_changed/dispatch.rs",
    "graph/dispatch/csr_forward_or_changed/mod.rs",
    "graph/dispatch/csr_forward_or_changed/reference.rs",
    "graph/dispatch/csr_forward_or_changed/tests/mod.rs",
    "graph/dispatch/csr_forward_or_changed/tests/reference_contracts.rs",
    "graph/dispatch/csr_frontier_queue_batch_memory.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/dispatch.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/mod.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/budget_contracts.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/high_degree_contracts.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/lifecycle_contracts.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/materializer_contracts.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/mod.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/queue_capacity_contracts.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/recording_dispatcher.rs",
    "graph/dispatch/csr_frontier_queue_batch_resident/tests/sequence_contracts.rs",
    "graph/dispatch/csr_frontier_queue_programs/mod.rs",
    "graph/dispatch/csr_frontier_queue_programs/tests/mod.rs",
    "graph/dispatch/csr_frontier_queue_resident/mod.rs",
    "graph/dispatch/csr_frontier_queue_resident/query.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/high_degree_contracts.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/lifecycle_contracts.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/materializer_contracts.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/mod.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/queue_capacity_contracts.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/recording_dispatcher.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/sequence_contracts.rs",
    "graph/dispatch/csr_frontier_queue_resident/tests/upload_contracts.rs",
    "graph/dispatch/csr_frontier_queue_resident/upload.rs",
    "graph/dispatch/csr_frontier_queue_scratch.rs",
    "graph/dispatch/dispatch_bridge/inputs.rs",
    "graph/dispatch/dispatch_bridge/mod.rs",
    "graph/dispatch/dispatch_bridge/resident.rs",
    "graph/dispatch/dispatch_bridge/tests/mod.rs",
    "graph/dispatch/dispatch_bridge/u32_outputs.rs",
    "graph/dispatch/dominator_frontier/dispatch.rs",
    "graph/dispatch/dominator_frontier/mod.rs",
    "graph/dispatch/dominator_frontier/reference.rs",
    "graph/dispatch/dominator_frontier/tests/dispatcher_doubles.rs",
    "graph/dispatch/dominator_frontier/tests/mod.rs",
    "graph/dispatch/exploded/dispatch.rs",
    "graph/dispatch/exploded/mod.rs",
    "graph/dispatch/exploded/reference.rs",
    "graph/dispatch/exploded/tests/ifds_doubles.rs",
    "graph/dispatch/exploded/tests/mod.rs",
    "graph/dispatch/frontier.rs",
    "graph/dispatch/level_wave_pass.rs",
    "graph/dispatch/mod.rs",
    "graph/dispatch/motif/dispatch.rs",
    "graph/dispatch/motif/mod.rs",
    "graph/dispatch/motif/reference.rs",
    "graph/dispatch/motif/tests/mod.rs",
    "graph/dispatch/path_reconstruct/dispatch.rs",
    "graph/dispatch/path_reconstruct/mod.rs",
    "graph/dispatch/path_reconstruct/reference.rs",
    "graph/dispatch/path_reconstruct/tests/mod.rs",
    "graph/dispatch/persistent_bfs/dispatch.rs",
    "graph/dispatch/persistent_bfs/mod.rs",
    "graph/dispatch/persistent_bfs/reference.rs",
    "graph/dispatch/persistent_bfs/resident.rs",
    "graph/dispatch/persistent_bfs/resident_scratch.rs",
    "graph/dispatch/persistent_bfs/tests/cpu_reference_contracts.rs",
    "graph/dispatch/persistent_bfs/tests/mod.rs",
    "graph/dispatch/persistent_bfs/tests/resident_contracts.rs",
    "graph/dispatch/persistent_bfs/tests/via_dispatch_contracts.rs",
    "graph/dispatch/plan_cache.rs",
    "graph/dispatch/resident_handles.rs",
    "graph/dispatch/structural_kernel_pipeline/mod.rs",
    "graph/dispatch/structural_kernel_pipeline/tests/mod.rs",
    "graph/dispatch/toposort/dispatch.rs",
    "graph/dispatch/toposort/mod.rs",
    "graph/dispatch/toposort/reference.rs",
    "graph/dispatch/toposort/tests/mod.rs",
    "graph/dispatch/traversal_dispatch_pipeline/mod.rs",
    "graph/dispatch/traversal_dispatch_pipeline/tests/mod.rs",
    "graph/dispatch/union_find_emit/dispatch.rs",
    "graph/dispatch/union_find_emit/mod.rs",
    "graph/dispatch/union_find_emit/reference.rs",
    "graph/dispatch/union_find_emit/tests/mod.rs",
    "graph/dispatch/vast_tree_walk/mod.rs",
    "graph/dispatch/vast_tree_walk/tests/mod.rs",
    "graph/do_calculus.rs",
    "graph/dominator_tree/cooper_harvey_kennedy.rs",
    "graph/dominator_tree/cpu_ref.rs",
    "graph/dominator_tree/depth.rs",
    "graph/dominator_tree/dominator_vec_growth.rs",
    "graph/dominator_tree/intersect_step.rs",
    "graph/dominator_tree/lengauer_tarjan.rs",
    "graph/dominator_tree/mod.rs",
    "graph/dominator_tree/program.rs",
    "graph/dominator_tree/tests/mod.rs",
    "graph/edge_scan.rs",
    "graph/exploded/abi.rs",
    "graph/exploded/canonicalize.rs",
    "graph/exploded/cpu_ref.rs",
    "graph/exploded/dispatch_plan.rs",
    "graph/exploded/encoding.rs",
    "graph/exploded/layout.rs",
    "graph/exploded/mod.rs",
    "graph/exploded/program_ir.rs",
    "graph/exploded/program_key.rs",
    "graph/exploded/tests/cpu_reference_tests.rs",
    "graph/exploded/tests/dispatch_plan_tests.rs",
    "graph/exploded/tests/mod.rs",
    "graph/exploded/tests/rule_column_tests.rs",
    "graph/exploded/validation.rs",
    "graph/frontier_bits.rs",
    "graph/knowledge_compile.rs",
    "graph/level_wave.rs",
    "graph/matroid.rs",
    "graph/mod.rs",
    "graph/motif/cpu_ref.rs",
    "graph/motif/layout.rs",
    "graph/motif/mod.rs",
    "graph/motif/pattern.rs",
    "graph/motif/plan.rs",
    "graph/motif/program.rs",
    "graph/motif/tests/mod.rs",
    "graph/persistent_bfs/cpu_ref.rs",
    "graph/persistent_bfs/dispatch_plan.rs",
    "graph/persistent_bfs/hash.rs",
    "graph/persistent_bfs/layout.rs",
    "graph/persistent_bfs/mod.rs",
    "graph/persistent_bfs/plan.rs",
    "graph/persistent_bfs/program.rs",
    "graph/persistent_bfs/resident_plan.rs",
    "graph/persistent_bfs/tests/behavior_contracts/cpu_reference_contracts.rs",
    "graph/persistent_bfs/tests/behavior_contracts/device_parity.rs",
    "graph/persistent_bfs/tests/behavior_contracts/dispatch_layout_contracts.rs",
    "graph/persistent_bfs/tests/behavior_contracts/mod.rs",
    "graph/persistent_bfs/tests/behavior_contracts/program_sync_contracts.rs",
    "graph/persistent_bfs/tests/mod.rs",
    "graph/persistent_bfs/tests/validation_and_builders.rs",
    "graph/persistent_bfs/validate.rs",
    "graph/program_graph.rs",
    "graph/reachable.rs",
    "graph/state_index_frontier.rs",
    "graph/tensor_flow_forward_tests.rs",
    "graph/toposort/csr.rs",
    "graph/toposort/edge_list.rs",
    "graph/toposort/error.rs",
    "graph/toposort/mod.rs",
    "graph/toposort/plan.rs",
    "graph/toposort/program.rs",
    "graph/toposort/tests/mod.rs",
    "graph/vector_neighbor_graph.rs",
    "hash/hypervector.rs",
    "hash/mod.rs",
    "hash/sketch.rs",
    "hash/table.rs",
    "label/mod.rs",
    "label/nodeset_filter.rs",
    "llm/mod.rs",
    "logical/wrap.rs",
    "matching/anchor_dfa.rs",
    "matching/dfa_compile/compile.rs",
    "matching/dfa_compile/mod.rs",
    "matching/dfa_compile/tests/mod.rs",
    "matching/dfa_compile/wire.rs",
    "matching/mod.rs",
    "matching/nfa_to_dfa/dedup.rs",
    "matching/nfa_to_dfa/error.rs",
    "matching/nfa_to_dfa/mod.rs",
    "matching/nfa_to_dfa/state_set.rs",
    "matching/nfa_to_dfa/subset.rs",
    "matching/nfa_to_dfa/tests/mod.rs",
    "matching/region.rs",
    "matching/region_programs.rs",
    "matching/region_tests.rs",
    "math/bit_count_u32.rs",
    "math/broadcast/mod.rs",
    "math/chebyshev_recurrence.rs",
    "math/conv/mod.rs",
    "math/cpu_matrix.rs",
    "math/dp_clip.rs",
    "math/fft/complex_length.rs",
    "math/fft/mod.rs",
    "math/fixed.rs",
    "math/fixed_u32_matmul.rs",
    "math/fmm.rs",
    "math/fractional.rs",
    "math/kfac_block_inverse.rs",
    "math/linalg/matmul_tiled/body.rs",
    "math/linalg/matmul_tiled/mma_body.rs",
    "math/linalg/matmul_tiled/mma_fragment.rs",
    "math/linalg/matmul_tiled/mod.rs",
    "math/linalg/matmul_tiled/program.rs",
    "math/linalg/matmul_tiled/shape.rs",
    "math/linalg/matmul_tiled/tensor_core_policy.rs",
    "math/linalg/matmul_tiled/tile_coords.rs",
    "math/linalg/mod.rs",
    "math/matroid_intersection_full.rs",
    "math/mod.rs",
    "math/natural_gradient.rs",
    "math/ode_step.rs",
    "math/prefix_scan.rs",
    "math/quantized/cpu.rs",
    "math/quantized/i4_expressions.rs",
    "math/quantized/programs.rs",
    "math/quantized/tests/batched_matmul_contracts.rs",
    "math/quantized/tests/dot_contracts.rs",
    "math/quantized/tests/layout_contracts.rs",
    "math/quantized/tests/matvec_contracts.rs",
    "math/quantized/tests/mod.rs",
    "math/quantized/tests/pack_unpack_contracts.rs",
    "math/quantized/tests/zero_shape_contracts.rs",
    "math/scallop_persistent.rs",
    "math/scan/mod.rs",
    "math/score_denoise.rs",
    "math/semiring_gemm/tests/mod.rs",
    "math/semiring_gemm/wide.rs",
    "math/sinkhorn_iterate/f64_tests.rs",
    "math/sinkhorn_iterate/program.rs",
    "math/sinkhorn_iterate/reference.rs",
    "math/sinkhorn_iterate/reference_f64.rs",
    "math/sinkhorn_iterate/tests/mod.rs",
    "math/sos_certificate.rs",
    "math/sparse_selector.rs",
    "math/stream_compact.rs",
    "math/submodular_greedy.rs",
    "math/tensor_scc.rs",
    "math/tensor_train.rs",
    "math/u32_binary_map.rs",
    "math/welford.rs",
    "nfa/mod.rs",
    "nfa/subgroup_nfa.rs",
    "nn/activation/mod.rs",
    "nn/activation/unary.rs",
    "nn/attention/flash_attention_2.rs",
    "nn/attention/fused_tile_attention.rs",
    "nn/attention/gated_delta.rs",
    "nn/attention/gated_delta_chunked.rs",
    "nn/attention/gated_delta_spec.rs",
    "nn/attention/layout.rs",
    "nn/attention/mla.rs",
    "nn/attention/mod.rs",
    "nn/attention/planner.rs",
    "nn/attention_stability.rs",
    "nn/backward/mod.rs",
    "nn/backward/unary_f32.rs",
    "nn/conv/mod.rs",
    "nn/f32_stability.rs",
    "nn/inference_graph.rs",
    "nn/linear/layer/batch_matmul.rs",
    "nn/linear/layer/fused_activation.rs",
    "nn/linear/layer/linear_4bit/affine_grouped.rs",
    "nn/linear/layer/linear_4bit/affine_grouped_weight_reuse.rs",
    "nn/linear/layer/linear_4bit/grouped_layout.rs",
    "nn/linear/layer/linear_4bit/mod.rs",
    "nn/linear/layer/linear_4bit/planner_evidence.rs",
    "nn/linear/layer/linear_4bit/quantized_spec.rs",
    "nn/linear/layer/linear_4bit/unpack_on_demand.rs",
    "nn/linear/layer/mod.rs",
    "nn/linear/layer/tests/mod.rs",
    "nn/linear/layer/tests/relu_builder.rs",
    "nn/linear/layer/tests/rms_norm.rs",
    "nn/linear/layer/tests/tiled.rs",
    "nn/linear/mod.rs",
    "nn/mod.rs",
    "nn/model/composition.rs",
    "nn/model/dense_gated_mlp.rs",
    "nn/model/mod.rs",
    "nn/moe/expert_mlp.rs",
    "nn/moe/mod.rs",
    "nn/moe/moe_layer.rs",
    "nn/moe/topk_selection.rs",
    "nn/norm/gated_rms_norm.rs",
    "nn/norm/last_dim_l2_norm.rs",
    "nn/norm/mod.rs",
    "nn/optim/mod.rs",
    "nn/optim/muon_step.rs",
    "nn/quant/ggml.rs",
    "nn/quant/mod.rs",
    "nn/rms.rs",
    "opt/mod.rs",
    "parsing/ast_cse_constant_fold.rs",
    "parsing/ast_ops.rs",
    "parsing/bytecode_dispatch_table_pack.rs",
    "parsing/composition.rs",
    "parsing/core/ast/mod.rs",
    "parsing/core/ast/node.rs",
    "parsing/core/ast/shunting/mod.rs",
    "parsing/core/ast/shunting/operator.rs",
    "parsing/core/mod.rs",
    "parsing/go/lex.rs",
    "parsing/go/mod.rs",
    "parsing/go/parse/ast_ops.rs",
    "parsing/go/parse/mod.rs",
    "parsing/go/parse/structure.rs",
    "parsing/go/parse/token_predicates.rs",
    "parsing/lr_tables/action.rs",
    "parsing/lr_tables/c11_expr.rs",
    "parsing/lr_tables/mod.rs",
    "parsing/lr_tables/parser.rs",
    "parsing/lr_tables/table.rs",
    "parsing/mod.rs",
    "parsing/parallel_parse.rs",
    "parsing/python/mod.rs",
    "parsing/python/parse/mod.rs",
    "parsing/python/parse/walk.rs",
    "parsing/python/source_cache.rs",
    "parsing/python/tests/corpus.rs",
    "parsing/python/tests/mod.rs",
    "parsing/source_cache.rs",
    "parsing/vast.rs",
    "predicate/traversal.rs",
    "reasoning/adjustment_set_pass_dependency.rs",
    "reasoning/dnnf/compile.rs",
    "reasoning/dnnf/mod.rs",
    "reasoning/do_calculus_change_impact.rs",
    "reasoning/finite_category/adjoint.rs",
    "reasoning/finite_category/kan_extension.rs",
    "reasoning/finite_category/mod.rs",
    "reasoning/finite_category/yoneda.rs",
    "reasoning/functorial_pass_composition.rs",
    "reasoning/mod.rs",
    "reasoning/string_diagram_ir_rewrite.rs",
    "reasoning/zx_diagram/mod.rs",
    "reasoning/zx_diagram/rewrite.rs",
    "reduce/all.rs",
    "reduce/any.rs",
    "reduce/indexed_move.rs",
    "reduce/max.rs",
    "reduce/min.rs",
    "reduce/mod.rs",
    "representation/mod.rs",
    "rule/ast.rs",
    "rule/condition_op.rs",
    "rule/literal_false.rs",
    "rule/literal_true.rs",
    "rule/mod.rs",
    "rule/pattern_exists.rs",
    "rule/reference_eval.rs",
    "scan/classic_ac/bounded_ranges/mod.rs",
    "scan/classic_ac/bounded_ranges/prefilter/mod.rs",
    "scan/classic_ac/bounded_ranges/prefilter/suffix3.rs",
    "scan/classic_ac/bounded_ranges/regex_exact.rs",
    "scan/classic_ac/count_program/mod.rs",
    "scan/classic_ac/count_program/suffix2.rs",
    "scan/classic_ac/count_program/suffix3.rs",
    "scan/classic_ac/mod.rs",
    "scan/classic_ac/test_dispatch_and_decode.rs",
    "scan/dfa/mod.rs",
    "scan/fused_region_evidence.rs",
    "scan/haystack.rs",
    "scan/mod.rs",
    "scan/nfa/alloc.rs",
    "scan/nfa/mod.rs",
    "scan/nfa/plan.rs",
    "scan/nfa/shards.rs",
    "scan/nfa/tables.rs",
    "scan/post_process.rs",
    "scan/regex_anchored_window.rs",
    "scan/regex_compile/capture_mode.rs",
    "scan/regex_compile/char_class.rs",
    "scan/regex_compile/compile_error.rs",
    "scan/regex_compile/construct_budget.rs",
    "scan/regex_compile/hir_lowering.rs",
    "scan/regex_compile/match_extent.rs",
    "scan/regex_compile/mod.rs",
    "scan/regex_compile/nfa_builder.rs",
    "scan/regex_compile/set_compiler.rs",
    "scan/regex_dfa.rs",
    "scan/regex_region_admission.rs",
    "scan/scan_program.rs",
    "scan/substring/mod.rs",
    "scheduling/branch_compaction.rs",
    "scheduling/frontier_partitioning.rs",
    "scheduling/frontier_typed_ir.rs",
    "scheduling/megakernel_schedule.rs",
    "scheduling/mod.rs",
    "scheduling/multi_corpus_batching.rs",
    "scheduling/planar_rewrite_pass_scheduler.rs",
    "scheduling/polyhedral_fusion.rs",
    "scheduling/spectral_schedule.rs",
    "scheduling/submodular_cache_eviction.rs",
    "security/external_ifds.rs",
    "security/facts.rs",
    "security/family_mask.rs",
    "security/flow_composition.rs",
    "security/integer_overflow_arith.rs",
    "security/mod.rs",
    "security/predicate_catalog.rs",
    "security/relation_analyzer.rs",
    "security/reporter.rs",
    "security/sink_intersection.rs",
    "security/taint_kill.rs",
    "solvers/amg_pass_solver.rs",
    "solvers/bellman_tn_order.rs",
    "solvers/conv1d_latency_smoothing.rs",
    "solvers/dataflow_compaction_pipeline.rs",
    "solvers/differentiable_autotune.rs",
    "solvers/fmm_polyhedral_compress.rs",
    "solvers/kfac_autotune_step.rs",
    "solvers/mod.rs",
    "solvers/mori_zwanzig_region_coarsen.rs",
    "solvers/multigrid_matroid_solver.rs",
    "solvers/natural_gradient_autotuner.rs",
    "solvers/numerical_kernel_pipeline.rs",
    "solvers/persistent_homology_loop_signature.rs",
    "solvers/qsvt_matrix_function_fusion.rs",
    "solvers/quantized_dispatch/batched_matmul.rs",
    "solvers/quantized_dispatch/batched_matvec.rs",
    "solvers/quantized_dispatch/dot.rs",
    "solvers/quantized_dispatch/matvec.rs",
    "solvers/quantized_dispatch/mod.rs",
    "solvers/quantized_dispatch/shapes.rs",
    "solvers/quantized_dispatch/tests/batched_matmul_contracts.rs",
    "solvers/quantized_dispatch/tests/batched_matmul_top1_contracts.rs",
    "solvers/quantized_dispatch/tests/batched_matvec_contracts.rs",
    "solvers/quantized_dispatch/tests/dot_contracts.rs",
    "solvers/quantized_dispatch/tests/generated_contracts.rs",
    "solvers/quantized_dispatch/tests/matvec_contracts.rs",
    "solvers/quantized_dispatch/tests/mod.rs",
    "solvers/quantized_dispatch/tests/unpack_contracts.rs",
    "solvers/quantized_dispatch/top1.rs",
    "solvers/quantized_dispatch/unpack.rs",
    "solvers/scientific_kernel_pipeline.rs",
    "solvers/sheaf_heterophilic_dispatch.rs",
    "solvers/sheaf_spectral_clustering.rs",
    "solvers/sinkhorn_dispatch_clustering.rs",
    "solvers/sinkhorn_full_clustering.rs",
    "solvers/tensor_network_fusion_order.rs",
    "solvers/tensor_train_chain_fusion.rs",
    "solvers/tensor_train_compression.rs",
    "text/mod.rs",
    "text/utf8_validate/program.rs",
    "text/utf8_validate/reference.rs",
    "text/utf8_validate/sequence_rules.rs",
    "text/utf8_validate/tests/mod.rs",
    "topology/betti_persistence.rs",
    "topology/mod.rs",
    "topology/simplicial.rs",
    "topology/vietoris_rips.rs",
    "vfs/mod.rs",
    "visual/mod.rs",
    "visual/u32_word_bytes.rs",
];

fn check_source_placement(op: &OpInfo, findings: &mut Vec<Finding>) {
    let Some((_owner, declared_domain)) = operation_owner(&op.id) else {
        findings.push(Finding::new(
            format!("operation `{}` has no crate and domain namespace", op.id),
            "name it `<owner-crate>::<domain>::<operation>` so its canonical owner is mechanically decidable",
        ));
        return;
    };

    if op.source_file.is_empty() || op.source_file == "<unattributed>" {
        findings.push(Finding::new(
            format!("operation `{}` has no registration source attribution", op.id),
            "construct the registration through the track-caller constructor so the registry records its owning source file",
        ));
        return;
    }

    let normalized = op.source_file.replace('\\', "/");
    let Some((owning_crate, rest)) = normalized
        .split_once("vyre-libs/src/")
        .map(|(_, rest)| ("vyre-libs", rest))
        .or_else(|| {
            normalized
                .split_once("vyre-primitives/src/")
                .map(|(_, rest)| ("vyre-primitives", rest))
        })
    else {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` registration is outside `vyre-libs/src/` or `vyre-primitives/src/`",
                op.id
            ),
            "move the registration into the owning crate's source tree",
        ));
        return;
    };

    if op.tier == Tier::T2 && owning_crate != "vyre-primitives" {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "Category C hardware operation `{}` must live in `vyre-primitives/src/hardware/`",
                op.id
            ),
            "move the Category C hardware operation to `vyre-primitives/src/hardware/`",
        ));
        return;
    }

    let segments: Vec<&str> = rest
        .split('/')
        .map(|segment| segment.trim_end_matches(".rs"))
        .collect();
    let matches_domain = segments.contains(&declared_domain)
        || (declared_domain == "matching" && segments.contains(&"scan"))
        || (declared_domain == "quant" && segments.contains(&"nn"))
        || (declared_domain == "optim" && segments.contains(&"nn"));

    if !matches_domain {
        let source_domain = segments.first().copied().unwrap_or_default();
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` declares domain `{declared_domain}` but its registration lives in domain `{source_domain}`",
                op.id
            ),
            format!(
                "move the semantic owner to `{owning_crate}/src/{declared_domain}/`, or rename the operation into `{owning_crate}::{source_domain}::*` when its effects and contract are domain-specific"
            ),
        ));
    }
}

fn operation_owner(id: &str) -> Option<(&str, &str)> {
    let mut segments = id.split("::");
    let owner = segments.next()?;
    let domain = segments.next()?;
    (!owner.is_empty() && !domain.is_empty()).then_some((owner, domain))
}

fn check_attribution(
    op: &OpInfo,
    node: &Node,
    known: &BTreeSet<&str>,
    findings: &mut Vec<Finding>,
) {
    if let Node::Region {
        source_region: Some(parent),
        generator,
        ..
    } = node
    {
        let child = generator.as_str();
        if !known.contains(child) && !vyre_foundation::composition::is_anonymous_generator(child) {
            findings.push(Finding::in_file(
                &op.source_file,
                format!(
                    "operation `{}` attributes a composed region to unregistered child `{child}`",
                    op.id
                ),
                "register the child semantic owner and compose it by operation id, or mark the region anonymous when it owns no reusable operation",
            ));
        }
        if parent.as_str() != op.id && !known.contains(parent.as_str()) {
            findings.push(Finding::in_file(
                &op.source_file,
                format!(
                    "operation `{}` carries unknown composition parent `{}`",
                    op.id,
                    parent.as_str()
                ),
                "preserve the registered parent operation id when transplanting the child region",
            ));
        }
    }
    for body in vyre_foundation::visit::child_bodies(node) {
        for child in body {
            check_attribution(op, child, known, findings);
        }
    }
}

fn check_pair(left: &OpInfo, right: &OpInfo, findings: &mut Vec<Finding>) {
    if left.children.contains(&right.id) || right.children.contains(&left.id) {
        return;
    }

    if left.semantic_fingerprint == right.semantic_fingerprint {
        findings.push(consolidation_finding(
            left,
            right,
            "have byte-identical canonical programs after erasing only the owner id",
        ));
    }
}

fn consolidation_finding(left: &OpInfo, right: &OpInfo, evidence: &str) -> Finding {
    let left_domain = operation_owner(&left.id).map(|(_, domain)| domain);
    let right_domain = operation_owner(&right.id).map(|(_, domain)| domain);
    let direction = if left_domain == right_domain {
        "keep one parameterized semantic owner in that domain and make every caller compose it"
    } else if left.tier == Tier::T3 && right.tier == Tier::T3 {
        "promote the shared semantic body to the lowest common substrate domain and make both domains compose it"
    } else {
        "keep the lowest-level canonical owner and make the higher-level operation compose it"
    };
    Finding::new(
        format!(
            "operations `{}` and `{}` {evidence}, but neither composes the other",
            left.id, right.id
        ),
        direction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre::ir::{BufferDecl, DataType};

    fn fixture(id: &'static str, value: u32) -> OpInfo {
        build_info(
            id,
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(value))],
            ),
        )
    }
    fn fixture_distinct(id: &'static str) -> OpInfo {
        build_info(
            id,
            Program::wrapped(
                vec![
                    BufferDecl::read("in", 0, DataType::U32).with_count(1),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::load("in", Expr::u32(0)),
                )],
            ),
        )
    }

    /// WHY: exact semantic duplicates are the non-heuristic consolidation class.
    /// A same-domain copy and a cross-domain copy must both fail; differences in
    /// the literal body remain outside this exact-identity assertion.
    #[test]
    fn exact_semantic_duplicates_require_one_owner_in_every_domain_arrangement() {
        let same_domain_left = fixture("vyre-libs::math::left", 7);
        let same_domain_right = fixture("vyre-libs::math::right", 7);
        let cross_domain = fixture("vyre-libs::graph::right", 7);

        let mut findings = Vec::new();
        check_pair(&same_domain_left, &same_domain_right, &mut findings);
        check_pair(&same_domain_left, &cross_domain, &mut findings);
        assert_eq!(findings.len(), 2);
        assert!(findings[0].fix.contains("parameterized semantic owner"));
        assert!(findings[1].fix.contains("common substrate domain"));

        let distinct = fixture_distinct("vyre-libs::math::distinct");
        findings.clear();
        check_pair(&same_domain_left, &distinct, &mut findings);
        assert!(findings.is_empty());
    }

    #[test]
    fn source_domain_must_equal_registered_domain() {
        let mut op = fixture("vyre-libs::math::sum", 1);
        op.source_file = "vyre-libs/src/nn/sum.rs".to_string();
        let mut findings = Vec::new();
        check_source_placement(&op, &mut findings);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("declares domain `math`"));
    }

    /// WHY: an unreadable walk leaves production files unclassified. Dropping
    /// the walk error would let the semantic ownership gate report a clean tree
    /// for a subject universe it never observed.
    #[test]
    fn production_file_walk_fails_closed() {
        let root = PathBuf::from("/path/that/does/not/exist/vyre-libs/src");
        assert!(rust_files_under(&root).is_err());
    }

    #[test]
    fn file_roles_classify_every_vyre_libs_file_uniquely() {
        let registered = BTreeSet::from(["vyre-libs/src/math/sin.rs"]);
        assert_eq!(
            classify_file_roles("vyre-libs/src/math/sin.rs", &registered),
            vec![FileRole::OperationImplementation]
        );
        assert_eq!(
            classify_file_roles("vyre-libs/src/builder/elementwise.rs", &registered),
            vec![FileRole::SharedBuilder]
        );
        assert_eq!(
            classify_file_roles("vyre-libs/src/lib.rs", &registered),
            vec![FileRole::CratePlumbing]
        );
        assert_eq!(
            classify_file_roles(
                "vyre-libs/src/graph/csr_frontier_queue/graph_validation.rs",
                &registered
            ),
            vec![FileRole::DomainContractOrType]
        );
        assert_eq!(
            classify_file_roles("vyre-libs/src/dumping_ground.rs", &registered),
            vec![]
        );
    }

    /// WHY: Section 182.12.8 and fail-by-default require a newly added unclassified file to fail role closure.
    #[test]
    fn new_unregistered_nested_file_fails_role_closure() {
        let registered = BTreeSet::from(["vyre-libs/src/math/sin.rs"]);
        let new_file = "vyre-libs/src/foo/new_copy.rs";
        let roles = classify_file_roles(new_file, &registered);
        assert!(
            roles.is_empty(),
            "new unclassified file must have zero recognized roles"
        );
        assert_eq!(classify_file_role(new_file, &registered), None);

        let new_nested_math = "vyre-libs/src/math/unauthorized_new_copy.rs";
        let math_roles = classify_file_roles(new_nested_math, &registered);
        assert!(
            math_roles.is_empty(),
            "unauthorized nested file must fail role classification"
        );
    }

    /// WHY: Section 182.12.3 requires rejecting any file with more than one organization class.
    #[test]
    fn overlapping_file_roles_fails_role_closure() {
        let registered = BTreeSet::from(["vyre-libs/src/builder/elementwise.rs"]);
        let roles = classify_file_roles("vyre-libs/src/builder/elementwise.rs", &registered);
        assert_eq!(
            roles,
            vec![FileRole::OperationImplementation, FileRole::SharedBuilder],
            "file claiming both operation implementation and shared builder must report overlapping roles"
        );
        assert_eq!(
            classify_file_role("vyre-libs/src/builder/elementwise.rs", &registered),
            None
        );
    }
}
