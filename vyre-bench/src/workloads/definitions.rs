//! Representative complete graph and adversarial kernel-sized region workload definitions.
//!
//! BACKLOG row 47 requires:
//! "Cover representative complete graphs and adversarial kernel-sized regions."
//! "Version-pinned expert-written native kernels are compared under identical
//! semantics, dtype, shapes, raggedness, initial and final state, target, stream,
//! toolchain and flags, clock and power state, warmup, interleaving, repetitions,
//! cache state, and objective."

use serde::{Deserialize, Serialize};

use super::equality::NativeComparisonConditions;

/// Domain classification for representative workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WorkloadDomain {
    /// Dense numerical linear algebra and tensor contractions.
    DenseNumerical,
    /// Irregular graph, sparse matrix, and ragged sequence computation.
    IrregularGraph,
    /// Multi-head attention, sequence recurrence, and stateful scans.
    AttentionAndRecurrence,
    /// Whole-application connected dataflow graph pipeline.
    WholeApplicationDataflow,
}

/// Workload specification defining test configurations and conditions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadSpecification {
    /// Workload identifier.
    pub id: String,
    /// Human-readable title.
    pub name: String,
    /// Workload domain classification.
    pub domain: WorkloadDomain,
    /// Summary description.
    pub description: String,
    /// Whether this workload represents an adversarial stress region.
    pub is_adversarial: bool,
    /// Whether this workload represents a complete whole-application graph.
    pub is_complete_graph: bool,
    /// Pinned native baseline comparator identifier.
    pub pinned_native_baseline_id: String,
    /// Default baseline equality conditions.
    pub default_conditions: NativeComparisonConditions,
}

impl WorkloadSpecification {
    /// 1. Representative complete graph dataflow pipeline workload.
    #[must_use]
    pub fn complete_graph_pipeline() -> Self {
        Self {
            id: "workload.complete_graph.pipeline_5stage".to_string(),
            name: "Complete 5-Stage Graph Dataflow Pipeline".to_string(),
            domain: WorkloadDomain::WholeApplicationDataflow,
            description: "End-to-end multi-stage pipeline: embedding lookup -> tiled GEMM -> activation -> layer-norm -> scatter-reduce".to_string(),
            is_adversarial: false,
            is_complete_graph: true,
            pinned_native_baseline_id: "native.graph_pipeline.v1_0_0".to_string(),
            default_conditions: NativeComparisonConditions {
                semantics: Some("fp32_ulp_tol:4".to_string()),
                dtype: Some("f32".to_string()),
                shapes: Some("[1024, 4096] -> [4096, 4096] -> [1024, 4096]".to_string()),
                raggedness: Some("uniform_contiguous".to_string()),
                initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
                target: Some("sm_90a".to_string()),
                stream: Some("cuda_stream_non_blocking_0".to_string()),
                toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
                clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
                warmup: Some("300_warmup_iterations_discarded".to_string()),
                interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
                repetitions: Some("30_measured_samples_clt".to_string()),
                cache_state: Some("flushed_l2_between_iterations".to_string()),
                objective: Some("minimize_p50_latency".to_string()),
            },
        }
    }

    /// 2. Adversarial kernel-sized region: power-law ragged graph frontier traversal.
    #[must_use]
    pub fn adversarial_ragged_reduction() -> Self {
        Self {
            id: "workload.adversarial.ragged_power_law_csr".to_string(),
            name: "Adversarial Ragged Power-Law CSR Reduction".to_string(),
            domain: WorkloadDomain::IrregularGraph,
            description: "Adversarial skewed graph reduction with power-law degree distribution (alpha=1.5), inducing maximum warp divergence and bank conflicts".to_string(),
            is_adversarial: true,
            is_complete_graph: false,
            pinned_native_baseline_id: "native.cub.segmented_reduce_v2_1_0".to_string(),
            default_conditions: NativeComparisonConditions {
                semantics: Some("exact".to_string()),
                dtype: Some("u32".to_string()),
                shapes: Some("segments=65536,total_elements=16777216".to_string()),
                raggedness: Some("ragged_power_law:alpha=1.5".to_string()),
                initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
                target: Some("sm_90a".to_string()),
                stream: Some("cuda_stream_non_blocking_0".to_string()),
                toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
                clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
                warmup: Some("300_warmup_iterations_discarded".to_string()),
                interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
                repetitions: Some("30_measured_samples_clt".to_string()),
                cache_state: Some("flushed_l2_between_iterations".to_string()),
                objective: Some("minimize_p50_latency".to_string()),
            },
        }
    }

    /// 3. Dense matrix contraction workload compared against CUTLASS/cuBLAS.
    #[must_use]
    pub fn dense_contraction_gemm() -> Self {
        Self {
            id: "workload.dense.contraction_gemm_4096".to_string(),
            name: "Dense Contraction GEMM 4096x4096x4096".to_string(),
            domain: WorkloadDomain::DenseNumerical,
            description: "Square 4096 matrix multiplication comparing generic megakernel schedule search against CUTLASS 3.5.0 tensor core kernels".to_string(),
            is_adversarial: false,
            is_complete_graph: false,
            pinned_native_baseline_id: "native.cutlass.gemm_v3_5_0".to_string(),
            default_conditions: NativeComparisonConditions {
                semantics: Some("fp32_ulp_tol:4".to_string()),
                dtype: Some("f32".to_string()),
                shapes: Some("[4096, 4096]x[4096, 4096]".to_string()),
                raggedness: Some("uniform_contiguous".to_string()),
                initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
                target: Some("sm_90a".to_string()),
                stream: Some("cuda_stream_non_blocking_0".to_string()),
                toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
                clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
                warmup: Some("300_warmup_iterations_discarded".to_string()),
                interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
                repetitions: Some("30_measured_samples_clt".to_string()),
                cache_state: Some("flushed_l2_between_iterations".to_string()),
                objective: Some("maximize_throughput_gflops".to_string()),
            },
        }
    }

    /// 4. Multi-head self-attention recurrence workload compared against FlashAttention-2.
    #[must_use]
    pub fn attention_recurrence() -> Self {
        Self {
            id: "workload.attention.causal_mha_4096".to_string(),
            name: "Causal Multi-Head Attention Recurrence (Seq 4096)".to_string(),
            domain: WorkloadDomain::AttentionAndRecurrence,
            description: "Causal multi-head attention recurrence (batch=4, heads=32, seq=4096, dim=128) compared against FlashAttention-2 v2.5.8".to_string(),
            is_adversarial: false,
            is_complete_graph: false,
            pinned_native_baseline_id: "native.flash_attention.v2_5_8".to_string(),
            default_conditions: NativeComparisonConditions {
                semantics: Some("fp32_ulp_tol:8".to_string()),
                dtype: Some("f32".to_string()),
                shapes: Some("b=4,h=32,s=4096,d=128".to_string()),
                raggedness: Some("uniform_contiguous".to_string()),
                initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
                target: Some("sm_90a".to_string()),
                stream: Some("cuda_stream_non_blocking_0".to_string()),
                toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
                clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
                warmup: Some("300_warmup_iterations_discarded".to_string()),
                interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
                repetitions: Some("30_measured_samples_clt".to_string()),
                cache_state: Some("flushed_l2_between_iterations".to_string()),
                objective: Some("minimize_p50_latency".to_string()),
            },
        }
    }

    /// 5. Sparse SpMV matrix-vector product compared against cuSPARSE.
    #[must_use]
    pub fn sparse_spmv_csr() -> Self {
        Self {
            id: "workload.sparse.spmv_csr_10m".to_string(),
            name: "Sparse Matrix-Vector Multiplication CSR 10M Nonzeros".to_string(),
            domain: WorkloadDomain::IrregularGraph,
            description: "Compressed Sparse Row matrix-vector product over 10M nonzeros compared against cuSPARSE 12.3.0 SpMV".to_string(),
            is_adversarial: false,
            is_complete_graph: false,
            pinned_native_baseline_id: "native.cusparse.spmv_v12_3_0".to_string(),
            default_conditions: NativeComparisonConditions {
                semantics: Some("exact".to_string()),
                dtype: Some("f32".to_string()),
                shapes: Some("rows=1048576,cols=1048576,nnz=10000000".to_string()),
                raggedness: Some("csr_irregular_degree".to_string()),
                initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
                target: Some("sm_90a".to_string()),
                stream: Some("cuda_stream_non_blocking_0".to_string()),
                toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
                clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
                warmup: Some("300_warmup_iterations_discarded".to_string()),
                interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
                repetitions: Some("30_measured_samples_clt".to_string()),
                cache_state: Some("flushed_l2_between_iterations".to_string()),
                objective: Some("maximize_throughput_gb_s".to_string()),
            },
        }
    }

    /// 6. Inclusive scan workload compared against CUB.
    #[must_use]
    pub fn scan_inclusive() -> Self {
        Self {
            id: "workload.scan.inclusive_u32_1m".to_string(),
            name: "Device-Wide Inclusive Scan 1M u32".to_string(),
            domain: WorkloadDomain::DenseNumerical,
            description: "Device-wide inclusive prefix scan of 1,048,576 u32 elements compared against NVIDIA CUB 2.1.0 DeviceScan".to_string(),
            is_adversarial: false,
            is_complete_graph: false,
            pinned_native_baseline_id: "native.cub.inclusive_scan_v2_1_0".to_string(),
            default_conditions: NativeComparisonConditions {
                semantics: Some("exact".to_string()),
                dtype: Some("u32".to_string()),
                shapes: Some("[1048576]".to_string()),
                raggedness: Some("uniform_contiguous".to_string()),
                initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
                target: Some("sm_90a".to_string()),
                stream: Some("cuda_stream_non_blocking_0".to_string()),
                toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
                clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
                warmup: Some("300_warmup_iterations_discarded".to_string()),
                interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
                repetitions: Some("30_measured_samples_clt".to_string()),
                cache_state: Some("flushed_l2_between_iterations".to_string()),
                objective: Some("minimize_p50_latency".to_string()),
            },
        }
    }

    /// All canonical row 47 representative workloads.
    #[must_use]
    pub fn all_representative_workloads() -> Vec<Self> {
        vec![
            Self::complete_graph_pipeline(),
            Self::adversarial_ragged_reduction(),
            Self::dense_contraction_gemm(),
            Self::attention_recurrence(),
            Self::sparse_spmv_csr(),
            Self::scan_inclusive(),
        ]
    }
}
