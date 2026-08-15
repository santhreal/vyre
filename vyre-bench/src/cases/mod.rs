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

/// The bench-wide 32-bit mixer.
///
/// Every case that needs a reproducible pseudo-random stream generates it from
/// this function, so a fixture built by one case and a fixture built by another
/// are comparable. It is `const` so fixtures can be evaluated at compile time.
pub(crate) const fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
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
pub(crate) mod attention;
pub mod bigint;
pub mod binary_search;
pub(crate) mod byte_pack;
#[cfg(test)]
mod clone_family_guard;
pub mod compound_pipeline;
pub(crate) mod conditional;
pub mod conditional_batch;
pub mod conditional_eval;
pub mod cpu_baselines;
pub mod crypto;
pub mod cuda_ptx_patterns;
pub mod dataflow_irregular;
pub(crate) mod dfa_match;
pub mod elementwise;
pub(crate) mod frontier_step;
pub(crate) mod gather;
pub mod graph_frontier;
pub(crate) mod harness;
pub mod hashtable;
pub(crate) mod histogram;
pub mod interpreter;
pub mod lexer_transition;
pub(crate) mod matmul;
pub mod megakernel_condition;
pub mod megakernel_latency;
pub mod megakernel_truth;
pub(crate) mod micro;
#[cfg(target_os = "linux")]
pub mod nvme_gpu_ingest;
pub mod optimizer_impact;
pub mod quantized_linear;
pub(crate) mod queue_closure;
pub(crate) mod queue_closure_oracle;
pub(crate) mod queue_closure_profile;
pub(crate) mod queue_materialize;
pub(crate) mod queue_stage;
pub(crate) mod queue_traverse_plan;
pub mod reduce_sum;
pub(crate) mod reference_sample;
pub mod regex_bt;
pub mod release_workloads;
pub(crate) mod resident_queue;
pub mod scan_ac_irregular;
pub(crate) mod skewed_graph;
pub(crate) mod stencil;
pub mod synthetic;
pub(crate) mod transpose;
pub(crate) mod triplet_pass;
