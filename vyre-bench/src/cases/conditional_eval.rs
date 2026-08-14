//! `conditions.yara_like.eval.1m`  -  branchy rule-condition evaluation.
//!
//! This is the release proof workload for vyre's core claim: evaluate a large
//! set of conventional rule conditions faster than an optimized CPU path.
//! The CPU baseline is deliberately ordinary and strong: Rayon parallelism plus
//! scalar short-circuiting over pattern match flags, counts, offsets, filesize,
//! and entropy-style metadata. The GPU path executes the same condition graph as
//! vyre IR, one invocation per rule.
//!
//! The measured loop, the sparse-output verifier and the pattern-metadata
//! generator are shared with `conditions.yara_like.batch.16x64k` and live in
//! [`super::conditional`]; only the condition graph, the per-rule record layout
//! and the CPU oracle are here.

use super::byte_pack::u32_bytes;
use super::conditional::{
    conditional_measure, conditional_program, pattern_streams, verify_sparse_outputs,
    ConditionalLabels, ConditionalPrepared, HONEST_SUITES,
};
use super::harness::{CaseOps, ContractDescription, HarnessCase, WorkloadDescription};
use super::mix32;
use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, Correctness, DeterminismClass,
    WorkloadClass,
};
use crate::api::metric::elapsed_ns;
use crate::api::resident::{input_bytes_total, u32_counter_reset_program, ResidentInputSet};
use rayon::prelude::*;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const RULE_COUNT: u32 = 1 << 20;
const PATTERN_COUNT: u32 = 1 << 14;
const FILESIZE_BYTES: u32 = 10 * 1024 * 1024;
const ENTROPY_MILLIBITS: u32 = 712;
const FIRED_COUNT_RESOURCE_INDEX: usize = 12;
const FIRED_RULES_RESOURCE_INDEX: usize = 13;
const RESET_RESOURCE_INDICES: [usize; 1] = [FIRED_COUNT_RESOURCE_INDEX];
const CONDITIONAL_RESOURCE_INDICES: [usize; 14] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    FIRED_RULES_RESOURCE_INDEX,
];

pub(crate) const LABELS: ConditionalLabels = ConditionalLabels {
    metric_prefix: "conditional_eval",
    subject: "conditional eval resident sequence",
    fired_noun: "fired-rule",
    wire_context: "conditional-eval output",
};

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "conditions.yara_like.eval.1m",
    name: "YARA-like Conditional Eval 1M",
    summary: "Evaluate 1M branchy rule conditions over pattern flags, counts, offsets, filesize, and entropy metadata",
    tags: &[
        "honest",
        "conditions",
        "rule-engine",
        "cpu-favorable",
        "dataflow-adjacent",
    ],
    layer: BenchLayer::Honest,
    workload: WorkloadClass::Honest,
    determinism: DeterminismClass::Deterministic,
    owner_crate: "vyre-bench",
    suites: HONEST_SUITES,
    needs_gpu: true,
    needs_network: false,
    min_vram_bytes: Some(PATTERN_COUNT as u64 * 12 + RULE_COUNT as u64 * 40 + 4),
    min_input_bytes: None,
    feature_set: &[],
    contract: Some(ContractDescription {
        primitive: "YARA-like boolean rule-condition evaluation",
        baseline_crate: "rayon",
        baseline_name: "Rayon-parallel scalar short-circuit rule loop",
        min_speedup_x: 100.0,
    }),
};

static OPS: CaseOps<ConditionalPrepared> = CaseOps {
    build: prepare_conditional_eval,
    measure: conditional_measure,
    verify: verify_fired_rules,
    program: conditional_program,
    fingerprint: None,
    bytes_touched: bytes_touched,
};

pub(crate) static CONDITIONAL_EVAL: HarnessCase<ConditionalPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn verify_fired_rules(run: &BenchRun) -> Result<Correctness, BenchError> {
    verify_sparse_outputs(LABELS, &run.outputs, run.baseline_outputs.as_deref())
}

fn bytes_touched(prepared: &ConditionalPrepared) -> (u64, u64) {
    (prepared.input_bytes_total, RULE_COUNT as u64 * 4)
}

fn condition_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("matched", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(PATTERN_COUNT),
            BufferDecl::storage("counts", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(PATTERN_COUNT),
            BufferDecl::storage("offsets", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(PATTERN_COUNT),
            BufferDecl::storage("rule_a", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("rule_b", 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("rule_c", 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("rule_d", 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("min_count", 7, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("max_offset", 8, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("min_size", 9, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("max_size", 10, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::storage("entropy_limit", 11, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULE_COUNT),
            BufferDecl::read_write("fired_count", 12, DataType::U32).with_count(1),
            BufferDecl::output("fired_rules", 13, DataType::U32).with_count(RULE_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(RULE_COUNT)),
                vec![
                    Node::let_bind("pa", Expr::load("rule_a", Expr::var("tid"))),
                    Node::let_bind("pb", Expr::load("rule_b", Expr::var("tid"))),
                    Node::let_bind("pc", Expr::load("rule_c", Expr::var("tid"))),
                    Node::let_bind("pd", Expr::load("rule_d", Expr::var("tid"))),
                    Node::let_bind(
                        "both_literals",
                        Expr::and(
                            Expr::ne(Expr::load("matched", Expr::var("pa")), Expr::u32(0)),
                            Expr::ne(Expr::load("matched", Expr::var("pb")), Expr::u32(0)),
                        ),
                    ),
                    Node::let_bind(
                        "count_ok",
                        Expr::ge(
                            Expr::load("counts", Expr::var("pc")),
                            Expr::load("min_count", Expr::var("tid")),
                        ),
                    ),
                    Node::let_bind(
                        "offset_ok",
                        Expr::le(
                            Expr::load("offsets", Expr::var("pd")),
                            Expr::load("max_offset", Expr::var("tid")),
                        ),
                    ),
                    Node::let_bind(
                        "size_ok",
                        Expr::and(
                            Expr::ge(
                                Expr::u32(FILESIZE_BYTES),
                                Expr::load("min_size", Expr::var("tid")),
                            ),
                            Expr::le(
                                Expr::u32(FILESIZE_BYTES),
                                Expr::load("max_size", Expr::var("tid")),
                            ),
                        ),
                    ),
                    Node::let_bind(
                        "entropy_ok",
                        Expr::le(
                            Expr::u32(ENTROPY_MILLIBITS),
                            Expr::load("entropy_limit", Expr::var("tid")),
                        ),
                    ),
                    Node::let_bind(
                        "fired",
                        Expr::and(
                            Expr::and(Expr::var("both_literals"), Expr::var("count_ok")),
                            Expr::and(
                                Expr::var("offset_ok"),
                                Expr::and(Expr::var("size_ok"), Expr::var("entropy_ok")),
                            ),
                        ),
                    ),
                    Node::if_then(
                        Expr::var("fired"),
                        vec![
                            Node::let_bind(
                                "slot",
                                Expr::atomic_add("fired_count", Expr::u32(0), Expr::u32(1)),
                            ),
                            Node::store("fired_rules", Expr::var("slot"), Expr::var("tid")),
                        ],
                    ),
                ],
            ),
        ],
    )
}

fn prepare_conditional_eval(ctx: &mut BenchContext) -> Result<ConditionalPrepared, BenchError> {
    let program = condition_program();
    let reset_program = u32_counter_reset_program("fired_count");

    let (matched, counts, offsets) = pattern_streams(PATTERN_COUNT, FILESIZE_BYTES);

    let mut rule_a = Vec::with_capacity(RULE_COUNT as usize);
    let mut rule_b = Vec::with_capacity(RULE_COUNT as usize);
    let mut rule_c = Vec::with_capacity(RULE_COUNT as usize);
    let mut rule_d = Vec::with_capacity(RULE_COUNT as usize);
    let mut min_count = Vec::with_capacity(RULE_COUNT as usize);
    let mut max_offset = Vec::with_capacity(RULE_COUNT as usize);
    let mut min_size = Vec::with_capacity(RULE_COUNT as usize);
    let mut max_size = Vec::with_capacity(RULE_COUNT as usize);
    let mut entropy_limit = Vec::with_capacity(RULE_COUNT as usize);

    for rule in 0..RULE_COUNT {
        let seed = mix32(rule);
        rule_a.push(seed & (PATTERN_COUNT - 1));
        rule_b.push(mix32(seed ^ 0x9E37_79B9) & (PATTERN_COUNT - 1));
        rule_c.push(mix32(seed ^ 0x85EB_CA6B) & (PATTERN_COUNT - 1));
        rule_d.push(mix32(seed ^ 0xC2B2_AE35) & (PATTERN_COUNT - 1));
        min_count.push((seed >> 5) % 7 + 1);
        max_offset.push(FILESIZE_BYTES - ((seed >> 11) % (FILESIZE_BYTES / 2)));
        min_size.push(FILESIZE_BYTES - ((seed >> 17) & 4095));
        max_size.push(FILESIZE_BYTES + ((seed >> 3) & 8191));
        entropy_limit.push(600 + ((seed >> 9) % 320));
    }

    let inputs = vec![
        u32_bytes(&matched),
        u32_bytes(&counts),
        u32_bytes(&offsets),
        u32_bytes(&rule_a),
        u32_bytes(&rule_b),
        u32_bytes(&rule_c),
        u32_bytes(&rule_d),
        u32_bytes(&min_count),
        u32_bytes(&max_offset),
        u32_bytes(&min_size),
        u32_bytes(&max_size),
        u32_bytes(&entropy_limit),
    ];
    let input_bytes_total = input_bytes_total(&inputs);

    let resident = ResidentInputSet::upload_with_zeroed_outputs_optional(
        ctx,
        &inputs,
        &[4, RULE_COUNT as usize * 4],
        "conditional eval bench",
    )?;

    let baseline_start = std::time::Instant::now();
    let baseline_output = cpu_conditional_eval_raw(
        &matched,
        &counts,
        &offsets,
        &rule_a,
        &rule_b,
        &rule_c,
        &rule_d,
        &min_count,
        &max_offset,
        &min_size,
        &max_size,
        &entropy_limit,
    );
    let baseline_wall_ns = elapsed_ns(baseline_start);

    Ok(ConditionalPrepared {
        program,
        reset_program,
        inputs,
        input_bytes_total,
        baseline_output,
        baseline_wall_ns,
        resident,
        reset_indices: &RESET_RESOURCE_INDICES,
        condition_indices: &CONDITIONAL_RESOURCE_INDICES,
        fired_count_resource: FIRED_COUNT_RESOURCE_INDEX,
        fired_ids_resource: FIRED_RULES_RESOURCE_INDEX,
        eval_count: RULE_COUNT,
        labels: LABELS,
    })
}

#[allow(clippy::too_many_arguments)]
fn cpu_conditional_eval_raw(
    matched: &[u32],
    counts: &[u32],
    offsets: &[u32],
    rule_a: &[u32],
    rule_b: &[u32],
    rule_c: &[u32],
    rule_d: &[u32],
    min_count: &[u32],
    max_offset: &[u32],
    min_size: &[u32],
    max_size: &[u32],
    entropy_limit: &[u32],
) -> Vec<Vec<u8>> {
    let mut fired_rules: Vec<u32> = (0..RULE_COUNT as usize)
        .into_par_iter()
        .map(|rule| {
            if matched[rule_a[rule] as usize] == 0 {
                return None;
            }
            if matched[rule_b[rule] as usize] == 0 {
                return None;
            }
            if counts[rule_c[rule] as usize] < min_count[rule] {
                return None;
            }
            if offsets[rule_d[rule] as usize] > max_offset[rule] {
                return None;
            }
            if FILESIZE_BYTES < min_size[rule] || FILESIZE_BYTES > max_size[rule] {
                return None;
            }
            if ENTROPY_MILLIBITS > entropy_limit[rule] {
                return None;
            }
            Some(rule as u32)
        })
        .flatten()
        .collect();
    fired_rules.sort_unstable();
    let count = fired_rules.len() as u32;
    fired_rules.resize(RULE_COUNT as usize, 0);
    vec![u32_bytes(&[count]), u32_bytes(&fired_rules)]
}

inventory::submit! {
    &CONDITIONAL_EVAL as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This contract test keeps resident output resources aligned with their sparse binding indices.
    #[test]
    fn resident_sequence_indices_keep_sparse_outputs_in_binding_order() {
        assert_eq!(
            CONDITIONAL_RESOURCE_INDICES[FIRED_COUNT_RESOURCE_INDEX],
            FIRED_COUNT_RESOURCE_INDEX
        );
        assert_eq!(
            CONDITIONAL_RESOURCE_INDICES[FIRED_RULES_RESOURCE_INDEX],
            FIRED_RULES_RESOURCE_INDEX
        );
        assert_eq!(RESET_RESOURCE_INDICES, [FIRED_COUNT_RESOURCE_INDEX]);
    }

    /// This regression test proves resident conditional evaluation reports device resets without host reset traffic.
    #[test]
    fn metric_points_expose_device_reset_and_zero_host_reset_bytes() {
        let metrics = crate::cases::conditional_metric_points("conditional_eval", true, true, 0);

        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "conditional_eval_resident_buffers")
                .map(|metric| metric.value),
            Some(1)
        );
        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "conditional_eval_device_reset_sequence")
                .map(|metric| metric.value),
            Some(1)
        );
        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "conditional_eval_resident_reset_bytes")
                .map(|metric| metric.value),
            Some(0)
        );
    }

    /// The harness must publish this case under the id the release matrix pins.
    #[test]
    fn harness_case_keeps_its_registered_identity_and_contract() {
        assert_eq!(CONDITIONAL_EVAL.id().0, "conditions.yara_like.eval.1m");
        assert_eq!(CONDITIONAL_EVAL.suites(), HONEST_SUITES);

        let contract = CONDITIONAL_EVAL
            .performance_contract()
            .expect("the release proof workload must keep its CPU-baseline contract");

        assert_eq!(contract.baselines.len(), 1);
        assert_eq!(contract.baselines[0].min_speedup_x, 100.0);
        assert_eq!(contract.baselines[0].crate_name, "rayon");
    }
}
