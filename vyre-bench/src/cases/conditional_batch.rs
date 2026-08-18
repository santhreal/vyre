//! `conditions.yara_like.batch.16x64k`  -  batched sparse rule-condition eval.
//!
//! Same measured loop as `conditions.yara_like.eval.1m`: the shared owner in
//! [`super::conditional`] runs the resident reset-plus-evaluate sequence, builds
//! the sample and verifies the sparse fired set. What is specific here is the
//! packed nine-word rule descriptor, the per-file size and entropy metadata,
//! and the fired identifier being a file-and-rule pair rather than a rule.

use super::byte_pack::u32_bytes;
use super::conditional::{
    conditional_measure, conditional_program, file_metadata_predicates, fired_append,
    pattern_index_binds, pattern_streams, rule_conditions, rule_fires, stream_predicates,
    verify_sparse_outputs, ConditionalLabels, ConditionalPrepared, PatternStreams,
};
use super::harness::{CaseOps, HarnessCase, WorkloadDescription};
use crate::api::case::{BenchCase, BenchContext, BenchError, BenchRun, Correctness};
use crate::api::metric::elapsed_ns;
use crate::api::resident::{input_bytes_total, u32_counter_reset_program, ResidentInputSet};
use rayon::prelude::*;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const RULES_PER_FILE: u32 = 1 << 16;
const FILE_COUNT: u32 = 16;
const EVAL_COUNT: u32 = RULES_PER_FILE * FILE_COUNT;
const PATTERN_COUNT: u32 = 1 << 14;
const BASE_FILESIZE_BYTES: u32 = 10 * 1024 * 1024;
const DESC_WORDS: u32 = 9;
const FIRED_COUNT_RESOURCE_INDEX: usize = 6;
const FIRED_PAIRS_RESOURCE_INDEX: usize = 7;
const RESET_RESOURCE_INDICES: [usize; 1] = [FIRED_COUNT_RESOURCE_INDEX];
const CONDITIONAL_BATCH_RESOURCE_INDICES: [usize; 8] = [
    0,
    1,
    2,
    3,
    4,
    5,
    FIRED_COUNT_RESOURCE_INDEX,
    FIRED_PAIRS_RESOURCE_INDEX,
];

pub(crate) const LABELS: ConditionalLabels = ConditionalLabels {
    metric_prefix: "conditional_batch",
    subject: "batched conditional resident sequence",
    fired_noun: "fired-pair",
    wire_context: "conditional-batch output",
};

static WORKLOAD: WorkloadDescription = WorkloadDescription::honest(
    "conditions.yara_like.batch.16x64k",
    "Batched YARA-like Conditional Eval 16x64K",
    "Evaluate 65,536 rule conditions across 16 files with sparse fired-pair output",
    &[
        "honest",
        "conditions",
        "rule-engine",
        "batched",
        "sparse-output",
    ],
    PATTERN_COUNT as u64 * 12 + RULES_PER_FILE as u64 * 36 + EVAL_COUNT as u64 * 4 + 128,
    None,
);

static OPS: CaseOps<ConditionalPrepared> = CaseOps {
    build: prepare_conditional_batch,
    measure: conditional_measure,
    verify: verify_fired_pairs,
    program: conditional_program,
    fingerprint: None,
    bytes_touched,
};

pub(crate) static CONDITIONAL_BATCH: HarnessCase<ConditionalPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn verify_fired_pairs(run: &BenchRun) -> Result<Correctness, BenchError> {
    verify_sparse_outputs(LABELS, &run.outputs, run.baseline_outputs.as_deref())
}

fn bytes_touched(prepared: &ConditionalPrepared) -> (u64, u64) {
    (prepared.input_bytes_total, EVAL_COUNT as u64 * 4 + 4)
}

/// Load word `offset` of this lane's packed rule descriptor.
fn descriptor_word(offset: u32) -> Expr {
    if offset == 0 {
        Expr::load("rule_desc", Expr::var("desc"))
    } else {
        Expr::load("rule_desc", Expr::add(Expr::var("desc"), Expr::u32(offset)))
    }
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
            BufferDecl::storage("rule_desc", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(RULES_PER_FILE * DESC_WORDS),
            BufferDecl::storage("file_sizes", 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(FILE_COUNT),
            BufferDecl::storage("file_entropy", 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(FILE_COUNT),
            BufferDecl::read_write("fired_count", 6, DataType::U32).with_count(1),
            BufferDecl::output("fired_pairs", 7, DataType::U32).with_count(EVAL_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(EVAL_COUNT)),
                [
                    vec![
                        Node::let_bind(
                            "file",
                            Expr::div(Expr::var("tid"), Expr::u32(RULES_PER_FILE)),
                        ),
                        Node::let_bind(
                            "rule",
                            Expr::rem(Expr::var("tid"), Expr::u32(RULES_PER_FILE)),
                        ),
                        Node::let_bind("desc", Expr::mul(Expr::var("rule"), Expr::u32(DESC_WORDS))),
                    ],
                    pattern_index_binds(
                        descriptor_word(0),
                        descriptor_word(1),
                        descriptor_word(2),
                        descriptor_word(3),
                    ),
                    stream_predicates(descriptor_word(4), descriptor_word(5)),
                    // Sixteen files per run, so this lane's file size is a load
                    // rather than a constant. Bound before the size predicate
                    // reads it twice.
                    vec![Node::let_bind(
                        "filesize",
                        Expr::load("file_sizes", Expr::var("file")),
                    )],
                    file_metadata_predicates(
                        Expr::var("filesize"),
                        descriptor_word(6),
                        descriptor_word(7),
                        Expr::load("file_entropy", Expr::var("file")),
                        descriptor_word(8),
                    ),
                    fired_append("fired_pairs"),
                ]
                .concat(),
            ),
        ],
    )
}

fn prepare_conditional_batch(ctx: &mut BenchContext) -> Result<ConditionalPrepared, BenchError> {
    let program = condition_program();
    let reset_program = u32_counter_reset_program("fired_count");

    let (matched, counts, offsets) = pattern_streams(PATTERN_COUNT, BASE_FILESIZE_BYTES);

    // One rule per lane within a file, its nine parameters packed side by side.
    let mut rule_desc = Vec::with_capacity((RULES_PER_FILE * DESC_WORDS) as usize);
    for rule in 0..RULES_PER_FILE {
        rule_desc.extend_from_slice(
            &rule_conditions(rule, PATTERN_COUNT, BASE_FILESIZE_BYTES).descriptor_words(),
        );
    }
    let file_sizes: Vec<u32> = (0..FILE_COUNT)
        .map(|file| BASE_FILESIZE_BYTES + file * 257)
        .collect();
    let file_entropy: Vec<u32> = (0..FILE_COUNT)
        .map(|file| 640 + ((file * 37) % 220))
        .collect();
    let inputs = vec![
        u32_bytes(&matched),
        u32_bytes(&counts),
        u32_bytes(&offsets),
        u32_bytes(&rule_desc),
        u32_bytes(&file_sizes),
        u32_bytes(&file_entropy),
    ];
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ResidentInputSet::upload_with_zeroed_outputs_optional(
        ctx,
        &inputs,
        &[4, EVAL_COUNT as usize * 4],
        "conditional batch bench",
    )?;
    let baseline_start = std::time::Instant::now();
    let baseline_output = cpu_batch(
        &PatternStreams {
            matched: &matched,
            counts: &counts,
            offsets: &offsets,
        },
        &file_sizes,
        &file_entropy,
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
        condition_indices: &CONDITIONAL_BATCH_RESOURCE_INDICES,
        fired_count_resource: FIRED_COUNT_RESOURCE_INDEX,
        fired_ids_resource: FIRED_PAIRS_RESOURCE_INDEX,
        eval_count: EVAL_COUNT,
        labels: LABELS,
    })
}

/// The host oracle: the fired file-and-rule pairs, sorted, with their count in
/// the first buffer and the identifiers padded out to the sparse buffer's
/// declared length.
fn cpu_batch(
    streams: &PatternStreams<'_>,
    file_sizes: &[u32],
    file_entropy: &[u32],
) -> Vec<Vec<u8>> {
    let mut fired: Vec<u32> = (0..EVAL_COUNT)
        .into_par_iter()
        .filter(|tid| {
            let file = (tid / RULES_PER_FILE) as usize;
            rule_fires(
                streams,
                &rule_conditions(tid % RULES_PER_FILE, PATTERN_COUNT, BASE_FILESIZE_BYTES),
                file_sizes[file],
                file_entropy[file],
            )
        })
        .collect();
    fired.sort_unstable();
    let count = fired.len() as u32;
    fired.resize(EVAL_COUNT as usize, 0);
    vec![u32_bytes(&[count]), u32_bytes(&fired)]
}

inventory::submit! {
    &CONDITIONAL_BATCH as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This contract test keeps resident output resources aligned with their sparse binding indices.
    #[test]
    fn resident_sequence_indices_keep_sparse_outputs_in_binding_order() {
        assert_eq!(
            CONDITIONAL_BATCH_RESOURCE_INDICES[FIRED_COUNT_RESOURCE_INDEX],
            FIRED_COUNT_RESOURCE_INDEX
        );
        assert_eq!(
            CONDITIONAL_BATCH_RESOURCE_INDICES[FIRED_PAIRS_RESOURCE_INDEX],
            FIRED_PAIRS_RESOURCE_INDEX
        );
        assert_eq!(RESET_RESOURCE_INDICES, [FIRED_COUNT_RESOURCE_INDEX]);
    }

    /// This regression test proves resident conditional batches report device resets without host reset traffic.
    #[test]
    fn metric_points_expose_device_reset_and_zero_host_reset_bytes() {
        let metrics = crate::cases::conditional_metric_points("conditional_batch", true, true, 0);

        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "conditional_batch_resident_buffers")
                .map(|metric| metric.value),
            Some(1)
        );
        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "conditional_batch_device_reset_sequence")
                .map(|metric| metric.value),
            Some(1)
        );
        assert_eq!(
            metrics
                .iter()
                .find(|metric| metric.name == "conditional_batch_resident_reset_bytes")
                .map(|metric| metric.value),
            Some(0)
        );
    }

    /// The batched case carries no CPU-baseline speedup contract; the 1M eval
    /// case is the release proof workload and owns that claim.
    #[test]
    fn harness_case_keeps_its_registered_identity_and_no_contract() {
        assert_eq!(
            CONDITIONAL_BATCH.id().0,
            "conditions.yara_like.batch.16x64k"
        );
        assert_eq!(
            CONDITIONAL_BATCH.suites(),
            crate::cases::harness::HONEST_SUITES
        );
        assert!(CONDITIONAL_BATCH.performance_contract().is_none());
    }
}
