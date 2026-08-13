//! `runtime.adaptive_routing.gpu_resident.1m` - GPU-side workload routing.
//!
//! This benchmark models a scheduler that classifies one million independent
//! work items into skip/fast/deep/escalate routes using only resident GPU
//! state. It is deliberately not an arithmetic microbenchmark: the useful work
//! is per-item decisioning that would usually be orchestrated on the CPU
//! between kernels.
//!
//! The measured loop, the input length check and the baseline capture are
//! shared with `compound.pipeline.fused_filter.1m` and live in
//! [`super::triplet_pass`]; the routing program, the generated streams and the
//! per-item routing decision are here.

use super::mix32;
use super::harness::{CaseOps, ContractDescription, HarnessCase, WorkloadDescription};
use super::triplet_pass::{
    prepare_triplet, triplet_bytes_touched, triplet_measure, triplet_program, TripletPrepared,
    TripletSpec,
};
use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, Correctness, DeterminismClass,
    WorkloadClass,
};
use crate::api::metric::MetricPoint;
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const ITEM_COUNT: u32 = 1 << 20;
const ROUTE_SALT: u32 = 0x9E37_79B9;
const RISK_MASK: u32 = 0x3ff;

const SUITES: &[SuiteKind] = &[
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "runtime.adaptive_routing.gpu_resident.1m",
    name: "GPU Resident Adaptive Routing 1M",
    summary: "Classify one million resident work items into skip/fast/deep/escalate routes without CPU orchestration",
    tags: &[
        "runtime",
        "adaptive-routing",
        "resident",
        "scheduler",
        "release",
    ],
    layer: BenchLayer::Runtime,
    workload: WorkloadClass::Macro,
    determinism: DeterminismClass::Deterministic,
    owner_crate: "vyre-bench",
    suites: SUITES,
    needs_gpu: true,
    needs_network: false,
    min_vram_bytes: Some(ITEM_COUNT as u64 * 16),
    min_input_bytes: Some(ITEM_COUNT as u64 * 12),
    feature_set: &["runtime.adaptive-routing", "resident"],
    contract: Some(ContractDescription {
        primitive: "GPU-resident adaptive workload routing",
        baseline_crate: "rayon",
        baseline_name: "Rayon-parallel CPU scheduler over equivalent routing predicates",
        min_speedup_x: 10.0,
    }),
};

static OPS: CaseOps<TripletPrepared> = CaseOps {
    build: prepare_adaptive_routing,
    measure: triplet_measure,
    verify: verify_exact,
    program: triplet_program,
    fingerprint: None,
    bytes_touched: triplet_bytes_touched,
};

pub(crate) static ADAPTIVE_ROUTING: HarnessCase<TripletPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn verify_exact(run: &BenchRun) -> Result<Correctness, BenchError> {
    run.verify_exact_outputs()
}

fn prepare_adaptive_routing(ctx: &mut BenchContext) -> Result<TripletPrepared, BenchError> {
    prepare_triplet(
        ctx,
        TripletSpec {
            program: adaptive_routing_program(),
            streams: adaptive_routing_inputs(),
            lane: adaptive_route_word,
            stream_names: ["signals", "histories", "thresholds"],
            subject: "adaptive routing bench",
            metrics: adaptive_routing_metrics,
        },
    )
}

fn adaptive_routing_metrics(baseline_words: &[u32]) -> Vec<MetricPoint> {
    let escalated = baseline_words
        .iter()
        .filter(|&&route| (route >> 24) == 3)
        .count() as u64;
    vec![
        MetricPoint {
            name: "adaptive_routing_items".to_string(),
            value: u64::from(ITEM_COUNT),
        },
        MetricPoint {
            name: "adaptive_routing_predicates".to_string(),
            value: 3,
        },
        MetricPoint {
            name: "adaptive_routing_escalated".to_string(),
            value: escalated,
        },
    ]
}

fn adaptive_routing_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("signals", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(ITEM_COUNT),
            BufferDecl::storage("histories", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(ITEM_COUNT),
            BufferDecl::storage("thresholds", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(ITEM_COUNT),
            BufferDecl::output("routes", 3, DataType::U32).with_count(ITEM_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(ITEM_COUNT)),
                vec![
                    Node::let_bind("signal", Expr::load("signals", Expr::var("tid"))),
                    Node::let_bind("history", Expr::load("histories", Expr::var("tid"))),
                    Node::let_bind("threshold", Expr::load("thresholds", Expr::var("tid"))),
                    Node::let_bind(
                        "risk",
                        Expr::add(
                            Expr::bitand(
                                Expr::bitxor(
                                    Expr::var("signal"),
                                    Expr::mul(Expr::var("history"), Expr::u32(ROUTE_SALT)),
                                ),
                                Expr::u32(RISK_MASK),
                            ),
                            Expr::bitand(Expr::var("history"), Expr::u32(0xff)),
                        ),
                    ),
                    Node::let_bind("hot", Expr::ge(Expr::var("risk"), Expr::var("threshold"))),
                    Node::let_bind(
                        "unstable",
                        Expr::ge(
                            Expr::bitand(
                                Expr::shr(Expr::var("history"), Expr::u32(16)),
                                Expr::u32(7),
                            ),
                            Expr::u32(4),
                        ),
                    ),
                    Node::let_bind(
                        "escalate",
                        Expr::and(Expr::var("hot"), Expr::var("unstable")),
                    ),
                    Node::let_bind(
                        "route",
                        Expr::select(
                            Expr::var("escalate"),
                            Expr::u32(3),
                            Expr::select(
                                Expr::var("hot"),
                                Expr::u32(2),
                                Expr::select(Expr::var("unstable"), Expr::u32(1), Expr::u32(0)),
                            ),
                        ),
                    ),
                    Node::store(
                        "routes",
                        Expr::var("tid"),
                        Expr::bitor(
                            Expr::shl(Expr::var("route"), Expr::u32(24)),
                            Expr::bitand(Expr::var("risk"), Expr::u32(0x00ff_ffff)),
                        ),
                    ),
                ],
            ),
        ],
    )
}

fn adaptive_routing_inputs() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    super::generated_u32_triplet(ITEM_COUNT, |index| {
        (
            mix32(index ^ 0x4D59_5DF4),
            mix32(index.wrapping_mul(31).wrapping_add(0xA5A5_5A5A)),
            320 + (mix32(index ^ 0x517C_C1B7) & 511),
        )
    })
}

fn adaptive_route_word(signal: u32, history: u32, threshold: u32) -> u32 {
    let risk = ((signal ^ history.wrapping_mul(ROUTE_SALT)) & RISK_MASK) + (history & 0xff);
    let hot = risk >= threshold;
    let unstable = ((history >> 16) & 7) >= 4;
    let route = if hot && unstable {
        3
    } else if hot {
        2
    } else if unstable {
        1
    } else {
        0
    };
    (route << 24) | (risk & 0x00ff_ffff)
}


fn value_identity(value: &mut u32) -> u32 {
    *value
}

inventory::submit! {
    &ADAPTIVE_ROUTING as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_route_vectors_cover_every_decision_class() {
        let mut seen = [false; 4];
        for index in 0..12_288_u32 {
            let signal = mix32(index);
            let history = mix32(index ^ 0xCAFE_BABE);
            let threshold = 128 + (index & 767);
            let word = adaptive_route_word(signal, history, threshold);
            let route = (word >> 24) as usize;

            assert!(route < seen.len());
            assert!((word & 0x00ff_ffff) <= RISK_MASK + 0xff);
            seen[route] = true;
        }

        assert_eq!(seen, [true, true, true, true]);
    }

    #[test]
    fn hand_built_routes_pin_priority_encoding() {
        let history_stable = 0_u32;
        let history_unstable = 4 << 16;
        let signal = 0;
        let cold_threshold = 2048;
        let hot_threshold = 0;

        assert_eq!(
            adaptive_route_word(signal, history_stable, cold_threshold) >> 24,
            0
        );
        assert_eq!(
            adaptive_route_word(signal, history_unstable, cold_threshold) >> 24,
            1
        );
        assert_eq!(
            adaptive_route_word(signal, history_stable, hot_threshold) >> 24,
            2
        );
        assert_eq!(
            adaptive_route_word(signal, history_unstable, hot_threshold) >> 24,
            3
        );
    }

    /// The three generated streams reach the shared oracle at equal length; a
    /// mismatch would silently truncate the captured baseline.
    #[test]
    fn generated_streams_are_equal_length() {
        let (signals, histories, thresholds) = adaptive_routing_inputs();

        assert_eq!(signals.len(), ITEM_COUNT as usize);
        assert_eq!(histories.len(), ITEM_COUNT as usize);
        assert_eq!(thresholds.len(), ITEM_COUNT as usize);
    }

    /// The escalate metric counts exactly the route-3 words.
    #[test]
    fn escalated_metric_counts_route_three_words() {
        let words = [0 << 24, 3 << 24, 2 << 24, 3 << 24 | 5];

        let metrics = adaptive_routing_metrics(&words);

        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "adaptive_routing_escalated")
                .map(|metric| metric.value),
            Some(2)
        );
    }

    #[test]
    fn harness_case_keeps_its_registered_identity_and_contract() {
        assert_eq!(
            ADAPTIVE_ROUTING.id().0,
            "runtime.adaptive_routing.gpu_resident.1m"
        );
        assert_eq!(ADAPTIVE_ROUTING.suites(), SUITES);
        assert_eq!(
            ADAPTIVE_ROUTING
                .performance_contract()
                .expect("adaptive routing must keep its CPU-baseline contract")
                .baselines[0]
                .min_speedup_x,
            10.0
        );
    }
}
