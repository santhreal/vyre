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
    conditional_measure, conditional_program, file_metadata_predicates, fired_append,
    pattern_index_binds, pattern_streams, rule_conditions, rule_fires, stream_predicates,
    verify_sparse_outputs, ConditionalLabels, ConditionalPrepared, PatternStreams,
};
use super::harness::{CaseOps, ContractDescription, HarnessCase, WorkloadDescription};
use crate::api::case::{BenchCase, BenchContext, BenchError, BenchRun, Correctness};
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

static WORKLOAD: WorkloadDescription = WorkloadDescription::honest(
    "conditions.yara_like.eval.1m",
    "YARA-like Conditional Eval 1M",
    "Evaluate 1M branchy rule conditions over pattern flags, counts, offsets, filesize, and entropy metadata",
    &[
        "honest",
        "conditions",
        "rule-engine",
        "cpu-favorable",
        "dataflow-adjacent",
    ],
    PATTERN_COUNT as u64 * 12 + RULE_COUNT as u64 * 40 + 4,
    Some(ContractDescription {
        primitive: "YARA-like boolean rule-condition evaluation",
        baseline_crate: "rayon",
        baseline_name: "Rayon-parallel scalar short-circuit rule loop",
        min_speedup_x: 100.0,
    }),
);

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
                [
                    pattern_index_binds(
                        Expr::load("rule_a", Expr::var("tid")),
                        Expr::load("rule_b", Expr::var("tid")),
                        Expr::load("rule_c", Expr::var("tid")),
                        Expr::load("rule_d", Expr::var("tid")),
                    ),
                    stream_predicates(
                        Expr::load("min_count", Expr::var("tid")),
                        Expr::load("max_offset", Expr::var("tid")),
                    ),
                    // One file per run: its size and entropy are graph constants,
                    // not lane-indexed loads.
                    file_metadata_predicates(
                        Expr::u32(FILESIZE_BYTES),
                        Expr::load("min_size", Expr::var("tid")),
                        Expr::load("max_size", Expr::var("tid")),
                        Expr::u32(ENTROPY_MILLIBITS),
                        Expr::load("entropy_limit", Expr::var("tid")),
                    ),
                    fired_append("fired_rules"),
                ]
                .concat(),
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

    // One rule per lane, its nine parameters spread across nine buffers.
    for rule in 0..RULE_COUNT {
        let conditions = rule_conditions(rule, PATTERN_COUNT, FILESIZE_BYTES);
        rule_a.push(conditions.pattern_a);
        rule_b.push(conditions.pattern_b);
        rule_c.push(conditions.pattern_c);
        rule_d.push(conditions.pattern_d);
        min_count.push(conditions.min_count);
        max_offset.push(conditions.max_offset);
        min_size.push(conditions.min_size);
        max_size.push(conditions.max_size);
        entropy_limit.push(conditions.entropy_limit);
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
    let baseline_output = cpu_conditional_eval_raw(&PatternStreams {
        matched: &matched,
        counts: &counts,
        offsets: &offsets,
    });
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

/// The host oracle: the fired rule set, sorted, with its count in the first
/// buffer and the identifiers padded out to the sparse buffer's declared length.
fn cpu_conditional_eval_raw(streams: &PatternStreams<'_>) -> Vec<Vec<u8>> {
    let mut fired_rules: Vec<u32> = (0..RULE_COUNT)
        .into_par_iter()
        .filter(|rule| {
            rule_fires(
                streams,
                &rule_conditions(*rule, PATTERN_COUNT, FILESIZE_BYTES),
                FILESIZE_BYTES,
                ENTROPY_MILLIBITS,
            )
        })
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
        assert_eq!(
            CONDITIONAL_EVAL.suites(),
            crate::cases::harness::HONEST_SUITES
        );

        let contract = CONDITIONAL_EVAL
            .performance_contract()
            .expect("the release proof workload must keep its CPU-baseline contract");

        assert_eq!(contract.baselines.len(), 1);
        assert_eq!(contract.baselines[0].min_speedup_x, 100.0);
        assert_eq!(contract.baselines[0].crate_name, "rayon");
    }
}
