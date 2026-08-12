#![allow(clippy::unnecessary_cast, clippy::needless_range_loop)]
fn conditional_metric_points(
    prefix: &str,
    resident_used: bool,
    device_reset_sequence: bool,
    resident_reset_bytes: u64,
) -> Vec<crate::api::metric::MetricPoint> {
    [
        ("resident_buffers", u64::from(resident_used)),
        ("device_reset_sequence", u64::from(device_reset_sequence)),
        ("resident_reset_bytes", resident_reset_bytes),
    ]
    .into_iter()
    .map(|(suffix, value)| crate::api::metric::MetricPoint {
        name: format!("{prefix}_{suffix}"),
        value,
    })
    .collect()
}

fn scaled_ratio_x1000(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    (u128::from(numerator) * 1000 / u128::from(denominator)).min(u128::from(u64::MAX)) as u64
}
fn generated_u32_triplet(
    count: u32,
    mut generate: impl FnMut(u32) -> (u32, u32, u32),
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut first = Vec::with_capacity(count as usize);
    let mut second = Vec::with_capacity(count as usize);
    let mut third = Vec::with_capacity(count as usize);
    for index in 0..count {
        let (a, b, c) = generate(index);
        first.push(a);
        second.push(b);
        third.push(c);
    }
    (first, second, third)
}
pub mod adaptive_routing;
pub mod adversarial;
pub mod alias_aware_optimizations;
pub mod attention;
pub mod bigint;
pub mod binary_search;
pub(crate) mod byte_pack;
pub mod c_parser;
pub mod compound_pipeline;
pub mod conditional_batch;
pub mod conditional_eval;
pub mod cpu_baselines;
pub mod crypto;
pub mod cuda_ptx_patterns;
pub mod dataflow_irregular;
pub mod dfa_match;
pub mod egraph_saturation;
pub mod elementwise;
pub mod gather;
pub(crate) mod gpu_case;
pub mod graph_frontier;
pub mod hashtable;
pub mod histogram;
pub mod interpreter;
pub mod lexer_transition;
pub mod literal_set;
pub mod literal_set_async_overlap;
pub mod literal_set_cold_start;
pub mod literal_set_decode_heavy;
pub mod literal_set_paged_corpus;
pub mod literal_set_vs_cpu;
pub mod lower_rewrite_impact;
pub mod matmul;
pub mod megakernel_condition;
pub mod megakernel_latency;
pub mod megakernel_truth;
#[cfg(target_os = "linux")]
pub mod nvme_gpu_ingest;
pub mod optimizer_impact;
pub mod quantized_linear;
pub(crate) mod queue_closure_profile;
pub(crate) mod queue_stage;
pub mod reduce_sum;
pub mod regex_bt;
pub mod release_workloads;
pub mod rust_frontend;
pub mod scan_ac_irregular;
pub(crate) mod skewed_graph;
pub mod stencil;
pub mod synthetic;
pub mod transpose;
