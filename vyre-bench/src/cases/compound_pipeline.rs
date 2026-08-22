//! `compound.pipeline.fused_filter.1m` - fused literal/dataflow/score filtering.
//!
//! This workload is intentionally compound: each lane performs a literal-style
//! hash predicate, a dataflow liveness check, a score threshold, and a
//! taint-class compatibility check before writing one compact candidate score.
//! The point is to measure one resident GPU program that would otherwise be a
//! chain of CPU-side passes with intermediate materialization.
//!
//! The measured loop, the input length check and the baseline capture are
//! shared with `runtime.adaptive_routing.gpu_resident.1m` and live in
//! `super::triplet_pass`; the fused program, the generated streams and the
//! per-item acceptance decision are here.

use super::harness::{CaseOps, ContractDescription, HarnessCase, WorkloadDescription};
use super::mix32;
use super::triplet_pass::{
    prepare_triplet, triplet_bytes_touched, triplet_measure, triplet_program, TripletPrepared,
    TripletSpec,
};
use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, Correctness, WorkloadClass,
};
use crate::api::metric::MetricPoint;
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const ITEM_COUNT: u32 = 1 << 20;
const HASH_SALT: u32 = 2_654_435_761;
const SCORE_BASE: u32 = 500;
const SCORE_SPAN_MASK: u32 = 127;

const SUITES: &[SuiteKind] = &[
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "compound.pipeline.fused_filter.1m",
    name: "Compound Fused Filter 1M",
    summary: "One resident GPU pass fusing literal hash, dataflow liveness, score threshold, and taint-class filtering",
    tags: &[
        "compound",
        "resident",
        "dataflow",
        "matching",
        "release",
    ],
    layer: BenchLayer::Runtime,
    workload: WorkloadClass::Macro,
    suites: SUITES,
    min_vram_bytes: Some(ITEM_COUNT as u64 * 16),
    min_input_bytes: Some(ITEM_COUNT as u64 * 12),
    feature_set: &["compound.pipeline", "resident"],
    contract: Some(ContractDescription {
        primitive: "fused compound rule/dataflow filtering",
        baseline_crate: "rayon",
        baseline_name: "Rayon-parallel staged CPU filter with equivalent predicates",
        min_speedup_x: 10.0,
    }),
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<TripletPrepared> = CaseOps {
    build: prepare_compound_fused_filter,
    measure: triplet_measure,
    verify: verify_exact,
    program: triplet_program,
    fingerprint: None,
    bytes_touched: triplet_bytes_touched,
};

pub(crate) static COMPOUND_FUSED_FILTER: HarnessCase<TripletPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn verify_exact(run: &BenchRun) -> Result<Correctness, BenchError> {
    run.verify_exact_outputs()
}

fn prepare_compound_fused_filter(ctx: &mut BenchContext) -> Result<TripletPrepared, BenchError> {
    prepare_triplet(
        ctx,
        TripletSpec {
            program: compound_program(),
            streams: compound_inputs(),
            lane: compound_acceptance_value,
            stream_names: ["tokens", "scores", "states"],
            subject: "compound fused filter",
            metrics: compound_metrics,
        },
    )
}

fn compound_metrics(baseline_words: &[u32]) -> Vec<MetricPoint> {
    let accepted = baseline_words.iter().filter(|&&value| value != 0).count() as u64;
    vec![
        MetricPoint {
            name: "compound_items".to_string(),
            value: u64::from(ITEM_COUNT),
        },
        MetricPoint {
            name: "compound_fused_predicates".to_string(),
            value: 4,
        },
        MetricPoint {
            name: "compound_cpu_passes_elided".to_string(),
            value: 3,
        },
        MetricPoint {
            name: "compound_accepted_items".to_string(),
            value: accepted,
        },
    ]
}

fn compound_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("tokens", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(ITEM_COUNT),
            BufferDecl::storage("scores", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(ITEM_COUNT),
            BufferDecl::storage("states", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(ITEM_COUNT),
            BufferDecl::output("accepted", 3, DataType::U32).with_count(ITEM_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(ITEM_COUNT)),
                vec![
                    Node::let_bind("token", Expr::load("tokens", Expr::var("tid"))),
                    Node::let_bind("score", Expr::load("scores", Expr::var("tid"))),
                    Node::let_bind("state", Expr::load("states", Expr::var("tid"))),
                    Node::let_bind(
                        "mixed",
                        Expr::bitxor(
                            Expr::var("token"),
                            Expr::mul(Expr::var("state"), Expr::u32(HASH_SALT)),
                        ),
                    ),
                    Node::let_bind(
                        "score_floor",
                        Expr::add(
                            Expr::u32(SCORE_BASE),
                            Expr::bitand(
                                Expr::shr(Expr::var("state"), Expr::u32(8)),
                                Expr::u32(SCORE_SPAN_MASK),
                            ),
                        ),
                    ),
                    Node::let_bind(
                        "literal_hit",
                        Expr::eq(
                            Expr::bitand(Expr::var("mixed"), Expr::u32(0x1f)),
                            Expr::u32(0),
                        ),
                    ),
                    Node::let_bind(
                        "dataflow_live",
                        Expr::ne(Expr::bitand(Expr::var("state"), Expr::u32(1)), Expr::u32(0)),
                    ),
                    Node::let_bind(
                        "score_ok",
                        Expr::ge(Expr::var("score"), Expr::var("score_floor")),
                    ),
                    Node::let_bind(
                        "taint_class_ok",
                        Expr::eq(
                            Expr::bitand(Expr::shr(Expr::var("mixed"), Expr::u32(5)), Expr::u32(3)),
                            Expr::bitand(Expr::var("state"), Expr::u32(3)),
                        ),
                    ),
                    Node::let_bind(
                        "accepted_predicate",
                        Expr::and(
                            Expr::and(Expr::var("literal_hit"), Expr::var("dataflow_live")),
                            Expr::and(Expr::var("score_ok"), Expr::var("taint_class_ok")),
                        ),
                    ),
                    Node::store(
                        "accepted",
                        Expr::var("tid"),
                        Expr::select(
                            Expr::var("accepted_predicate"),
                            Expr::add(
                                Expr::bitand(Expr::var("mixed"), Expr::u32(0xffff)),
                                Expr::var("score"),
                            ),
                            Expr::u32(0),
                        ),
                    ),
                ],
            ),
        ],
    )
}

fn compound_inputs() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    super::generated_u32_triplet(ITEM_COUNT, |index| {
        (
            mix32(index ^ 0xA5A5_5A5A),
            440 + (mix32(index ^ 0x517C_C1B7) & 255),
            mix32(index.wrapping_mul(17).wrapping_add(0x9E37_79B9)) | 1,
        )
    })
}

fn compound_acceptance_value(token: u32, score: u32, state: u32) -> u32 {
    let mixed = token ^ state.wrapping_mul(HASH_SALT);
    let score_floor = SCORE_BASE + ((state >> 8) & SCORE_SPAN_MASK);
    let literal_hit = mixed & 0x1f == 0;
    let dataflow_live = state & 1 != 0;
    let score_ok = score >= score_floor;
    let taint_class_ok = ((mixed >> 5) & 3) == (state & 3);
    if literal_hit && dataflow_live && score_ok && taint_class_ok {
        (mixed & 0xffff).wrapping_add(score)
    } else {
        0
    }
}

inventory::submit! {
    &COMPOUND_FUSED_FILTER as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_acceptance_vectors_cover_thousands_of_state_classes() {
        for index in 0..8192_u32 {
            let state = mix32(index).wrapping_shl(1) | 1;
            let mixed = (index << 7) | ((state & 3) << 5);
            let token = mixed ^ state.wrapping_mul(HASH_SALT);
            let score = SCORE_BASE + ((state >> 8) & SCORE_SPAN_MASK) + (index & 7);
            let accepted = compound_acceptance_value(token, score, state);

            assert_eq!(accepted, (mixed & 0xffff).wrapping_add(score));
        }
    }

    #[test]
    fn generated_rejection_vectors_cover_each_fused_predicate() {
        for index in 0..4096_u32 {
            let live_state = mix32(index).wrapping_shl(1) | 1;
            let accepting_mixed = (index << 7) | ((live_state & 3) << 5);
            let accepting_token = accepting_mixed ^ live_state.wrapping_mul(HASH_SALT);
            let accepting_score = SCORE_BASE + ((live_state >> 8) & SCORE_SPAN_MASK);

            let literal_miss_token = (accepting_mixed | 1) ^ live_state.wrapping_mul(HASH_SALT);
            assert_eq!(
                compound_acceptance_value(literal_miss_token, accepting_score, live_state),
                0
            );

            let dead_state = live_state & !1;
            let dead_mixed = (index << 7) | ((dead_state & 3) << 5);
            let dead_token = dead_mixed ^ dead_state.wrapping_mul(HASH_SALT);
            let dead_score = SCORE_BASE + ((dead_state >> 8) & SCORE_SPAN_MASK);
            assert_eq!(
                compound_acceptance_value(dead_token, dead_score, dead_state),
                0
            );

            let low_score = accepting_score.saturating_sub(1);
            assert_eq!(
                compound_acceptance_value(accepting_token, low_score, live_state),
                0
            );

            let wrong_class = (((live_state & 3) + 1) & 3) << 5;
            let class_miss_mixed = (index << 7) | wrong_class;
            let class_miss_token = class_miss_mixed ^ live_state.wrapping_mul(HASH_SALT);
            assert_eq!(
                compound_acceptance_value(class_miss_token, accepting_score, live_state),
                0
            );
        }
    }

    /// The three generated streams reach the shared oracle at equal length; a
    /// mismatch would silently truncate the captured baseline.
    #[test]
    fn generated_streams_are_equal_length() {
        let (tokens, scores, states) = compound_inputs();

        assert_eq!(tokens.len(), ITEM_COUNT as usize);
        assert_eq!(scores.len(), ITEM_COUNT as usize);
        assert_eq!(states.len(), ITEM_COUNT as usize);
    }

    /// The accepted metric counts exactly the non-zero baseline words.
    #[test]
    fn accepted_metric_counts_non_zero_words() {
        let metrics = compound_metrics(&[0, 7, 0, 9]);

        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "compound_accepted_items")
                .map(|metric| metric.value),
            Some(2)
        );
    }

    #[test]
    fn harness_case_keeps_its_registered_identity_and_contract() {
        assert_eq!(
            COMPOUND_FUSED_FILTER.id().0,
            "compound.pipeline.fused_filter.1m"
        );
        assert_eq!(COMPOUND_FUSED_FILTER.suites(), SUITES);
        assert_eq!(
            COMPOUND_FUSED_FILTER
                .performance_contract()
                .expect("compound fused filter must keep its CPU-baseline contract")
                .baselines[0]
                .min_speedup_x,
            10.0
        );
    }
}
