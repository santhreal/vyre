//! Shared owner for the sparse rule-condition benchmark family.
//!
//! `conditions.yara_like.eval.1m` and `conditions.yara_like.batch.16x64k` both
//! evaluate a branchy rule-condition graph on GPU, append fired identifiers to
//! a device-resident sparse output through one atomic counter, and compare that
//! set against a Rayon CPU oracle. The two are not two rows of one table: the
//! 1M case keeps its rule parameters in twelve per-rule buffers and bakes one
//! file's size and entropy into the graph as constants, while the batched case
//! packs nine words per rule into one descriptor buffer and reads size and
//! entropy from per-file buffers indexed by lane. The layouts, the binding
//! counts and the appended identity all differ, and only the 1M case carries the
//! release speedup claim.
//!
//! What the two share is here: the rule parameters derived from a rule index,
//! the boolean predicate those parameters feed, the IR nodes that predicate
//! lowers to, the measured loop, the sample assembly and the sparse verifier.

use super::mix32;
use crate::api::case::{BenchContext, BenchError, BenchRun, Correctness};
use crate::api::metric::BenchMetrics;
use crate::api::resident::{
    dispatch_program_timed, stated_launch, transfer_accounting, ResidentInputSet,
};
use vyre_driver::{ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::{Expr, Node, Program};

/// Output slot of the atomic fired counter.
const FIRED_COUNT_OUTPUT: usize = 0;
/// Output slot of the sparse fired-identifier buffer.
const FIRED_IDS_OUTPUT: usize = 1;

/// The per-case wording the shared code needs to name what it is doing.
#[derive(Clone, Copy)]
pub(crate) struct ConditionalLabels {
    /// Prefix for the case's custom metric points.
    pub(crate) metric_prefix: &'static str,
    /// Subject used in resident-resource and workgroup error messages.
    pub(crate) subject: &'static str,
    /// Noun for the sparse identifiers, used in correctness messages.
    pub(crate) fired_noun: &'static str,
    /// Context passed to the little-endian word reader.
    pub(crate) wire_context: &'static str,
}

/// The device-resident state and host mirrors one conditional case needs.
pub(crate) struct ConditionalPrepared {
    pub(crate) program: Program,
    pub(crate) reset_program: Program,
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) input_bytes_total: u64,
    pub(crate) baseline_output: Vec<Vec<u8>>,
    pub(crate) baseline_wall_ns: u64,
    pub(crate) resident: Option<ResidentInputSet>,
    /// Resource indices bound by the counter-reset step.
    pub(crate) reset_indices: &'static [usize],
    /// Resource indices bound by the condition step.
    pub(crate) condition_indices: &'static [usize],
    /// Slot of the atomic counter within `condition_indices`.
    pub(crate) fired_count_resource: usize,
    /// Slot of the sparse output within `condition_indices`.
    pub(crate) fired_ids_resource: usize,
    /// Lanes the condition program launches.
    pub(crate) eval_count: u32,
    pub(crate) labels: ConditionalLabels,
}

/// One resident condition sequence: reset the counter, then evaluate.
struct ConditionalSequenceRun {
    outputs: Vec<Vec<u8>>,
    wall_ns: u64,
    dispatch_ns: Option<u64>,
}

/// Reset the atomic counter and evaluate the condition graph in one resident
/// sequence, reading back the counter and the populated prefix of the sparse
/// output.
fn dispatch_resident_conditional_sequence(
    ctx: &BenchContext,
    prepared: &ConditionalPrepared,
    resident: &ResidentInputSet,
) -> Result<ConditionalSequenceRun, BenchError> {
    let subject = prepared.labels.subject;
    let workgroup = prepared.program.workgroup_size();
    if let Some(override_workgroup) = ctx.dispatch_config.workgroup_override {
        if override_workgroup != workgroup {
            return Err(BenchError::ExecutionFailed(format!(
                "{subject} resident sequence uses program workgroup {workgroup:?}, but received override {override_workgroup:?}. Fix: run the resident condition sequence without a workgroup override or rebuild the resident sequence program."
            )));
        }
    }

    let reset_resources = resident.resources_for_indices(prepared.reset_indices, subject)?;
    let condition_resources =
        resident.resources_for_indices(prepared.condition_indices, subject)?;
    let steps = [
        ResidentDispatchStep {
            program: &prepared.reset_program,
            resources: &reset_resources,
            launch: Some(stated_launch(&prepared.reset_program, [1, 1, 1])?),
        },
        ResidentDispatchStep {
            program: &prepared.program,
            resources: &condition_resources,
            launch: Some(stated_launch(
                &prepared.program,
                [prepared.eval_count.div_ceil(workgroup[0]).max(1), 1, 1],
            )?),
        },
    ];
    let read_ranges = [
        ResidentReadRange {
            resource: &condition_resources[prepared.fired_count_resource],
            byte_offset: 0,
            byte_len: prepared.baseline_output[FIRED_COUNT_OUTPUT].len(),
        },
        ResidentReadRange {
            resource: &condition_resources[prepared.fired_ids_resource],
            byte_offset: 0,
            byte_len: prepared.baseline_output[FIRED_IDS_OUTPUT].len(),
        },
    ];

    let mut count_output = Vec::with_capacity(prepared.baseline_output[FIRED_COUNT_OUTPUT].len());
    let mut ids_output = Vec::with_capacity(prepared.baseline_output[FIRED_IDS_OUTPUT].len());
    let timing = ctx
        .dispatch_resident_sequence_read_ranges_timed_into(
            &steps,
            &read_ranges,
            &mut [&mut count_output, &mut ids_output],
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

    Ok(ConditionalSequenceRun {
        outputs: vec![count_output, ids_output],
        wall_ns: timing.wall_ns,
        dispatch_ns: timing.device_ns,
    })
}

/// One finished conditional measurement, however it was dispatched.
struct ConditionalSample {
    outputs: Vec<Vec<u8>>,
    wall_ns: u64,
    dispatch_ns: Option<u64>,
    resident_used: bool,
    device_reset_sequence: bool,
}

/// Run one measured sample: the resident reset-plus-evaluate sequence when the
/// backend keeps inputs on device, a single timed dispatch otherwise.
pub(crate) fn conditional_measure(
    ctx: &mut BenchContext,
    prepared: &mut ConditionalPrepared,
) -> Result<BenchRun, BenchError> {
    let sample = if let Some(resident) = &prepared.resident {
        let sequence = dispatch_resident_conditional_sequence(ctx, prepared, resident)?;
        ConditionalSample {
            outputs: sequence.outputs,
            wall_ns: sequence.wall_ns,
            dispatch_ns: sequence.dispatch_ns,
            resident_used: true,
            device_reset_sequence: true,
        }
    } else {
        let dispatch = dispatch_program_timed(
            ctx,
            &prepared.program,
            None,
            &prepared.inputs,
            &ctx.dispatch_config,
        )?;
        ConditionalSample {
            outputs: dispatch.timed.outputs,
            wall_ns: dispatch.timed.wall_ns,
            dispatch_ns: dispatch.timed.device_ns,
            resident_used: dispatch.resident_used,
            device_reset_sequence: false,
        }
    };

    Ok(conditional_bench_run(
        prepared,
        ctx.include_baseline_outputs,
        sample,
    ))
}

/// Assemble the measured sample from a finished sequence or single dispatch.
///
/// The resident path measures device time; it reaches the sample through
/// `dispatch_ns` here, so neither case can quietly report `None` again.
fn conditional_bench_run(
    prepared: &ConditionalPrepared,
    include_baseline_outputs: bool,
    sample: ConditionalSample,
) -> BenchRun {
    let ConditionalSample {
        outputs,
        wall_ns,
        dispatch_ns,
        resident_used,
        device_reset_sequence,
    } = sample;
    let input_bytes = prepared.input_bytes_total;
    let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;
    let accounting = transfer_accounting(input_bytes, output_bytes, resident_used);

    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(wall_ns),
            dispatch_ns,
            input_bytes: Some(input_bytes),
            output_bytes: Some(output_bytes),
            bytes_read: Some(accounting.bytes_read),
            bytes_written: Some(accounting.bytes_written),
            bytes_touched: Some(accounting.bytes_touched),
            custom: super::conditional_metric_points(
                prepared.labels.metric_prefix,
                resident_used,
                device_reset_sequence,
                0,
            ),
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(prepared.baseline_wall_ns),
            input_bytes: Some(input_bytes),
            output_bytes: Some(prepared.baseline_output.iter().map(Vec::len).sum::<usize>() as u64),
            ..Default::default()
        }),
        outputs,
        baseline_outputs: include_baseline_outputs.then(|| prepared.baseline_output.clone()),
    }
}

/// The single IR program the runner may recompile for either conditional case.
pub(crate) fn conditional_program(prepared: &ConditionalPrepared) -> Option<&Program> {
    Some(&prepared.program)
}

/// Compare the backend's sparse fired set against the baseline's.
///
/// Order is not part of the contract: the GPU appends through an atomic
/// counter, so both sides are sorted before comparison. Everything else is. A
/// differing count, a buffer shorter than its own reported count, and a
/// differing set are all correctness violations.
pub(crate) fn verify_sparse_outputs(
    labels: ConditionalLabels,
    outputs: &[Vec<u8>],
    baseline_outputs: Option<&[Vec<u8>]>,
) -> Result<Correctness, BenchError> {
    let noun = labels.fired_noun;
    let baseline = baseline_outputs.ok_or_else(|| {
        BenchError::CorrectnessViolation(format!(
            "{} did not capture baseline sparse {noun} output",
            labels.subject
        ))
    })?;
    if outputs.len() != 2 || baseline.len() != 2 {
        return Err(BenchError::CorrectnessViolation(format!(
            "sparse output count mismatch: backend returned {}, baseline returned {}",
            outputs.len(),
            baseline.len()
        )));
    }
    let backend_count = read_le_u32(labels, &outputs[FIRED_COUNT_OUTPUT], 0)? as usize;
    let baseline_count = read_le_u32(labels, &baseline[FIRED_COUNT_OUTPUT], 0)? as usize;
    if backend_count != baseline_count {
        return Err(BenchError::CorrectnessViolation(format!(
            "{noun} count mismatch: backend returned {backend_count}, baseline returned {baseline_count}"
        )));
    }
    if outputs[FIRED_IDS_OUTPUT].len() < backend_count.saturating_mul(4)
        || baseline[FIRED_IDS_OUTPUT].len() < baseline_count.saturating_mul(4)
    {
        return Err(BenchError::CorrectnessViolation(format!(
            "{noun} output buffer shorter than reported count"
        )));
    }
    let mut backend_ids = read_u32_prefix(labels, &outputs[FIRED_IDS_OUTPUT], backend_count)?;
    let mut baseline_ids = read_u32_prefix(labels, &baseline[FIRED_IDS_OUTPUT], baseline_count)?;
    backend_ids.sort_unstable();
    baseline_ids.sort_unstable();
    if backend_ids == baseline_ids {
        Ok(Correctness::Exact)
    } else {
        Err(BenchError::CorrectnessViolation(format!(
            "{noun} set differs between backend and baseline"
        )))
    }
}

fn read_le_u32(
    labels: ConditionalLabels,
    bytes: &[u8],
    word_index: usize,
) -> Result<u32, BenchError> {
    vyre_primitives::wire::read_u32_le_word(bytes, word_index, labels.wire_context)
        .map_err(BenchError::CorrectnessViolation)
}

fn read_u32_prefix(
    labels: ConditionalLabels,
    bytes: &[u8],
    count: usize,
) -> Result<Vec<u32>, BenchError> {
    (0..count)
        .map(|index| read_le_u32(labels, bytes, index))
        .collect()
}

/// The pattern-metadata streams both conditional workloads score rules against.
pub(crate) fn pattern_streams(
    pattern_count: u32,
    filesize_bytes: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    super::generated_u32_triplet(pattern_count, |index| {
        (
            u32::from((mix32(index) & 7) != 0),
            (mix32(index ^ 0xA5A5_5A5A) & 7) + 1,
            mix32(index ^ 0x517C_C1B7) % filesize_bytes,
        )
    })
}

/// One rule's nine condition parameters.
///
/// Both cases derive these from the rule index with the same mixer and the same
/// nine expressions; they differ only in where the words are stored. Two copies
/// of the derivation is two chances for one workload's rules to stop matching the
/// other's while both still look like the same benchmark.
pub(crate) struct RuleConditions {
    /// Pattern whose match flag must be set.
    pub(crate) pattern_a: u32,
    /// Second pattern whose match flag must be set.
    pub(crate) pattern_b: u32,
    /// Pattern whose match count is compared against `min_count`.
    pub(crate) pattern_c: u32,
    /// Pattern whose first offset is compared against `max_offset`.
    pub(crate) pattern_d: u32,
    pub(crate) min_count: u32,
    pub(crate) max_offset: u32,
    pub(crate) min_size: u32,
    pub(crate) max_size: u32,
    pub(crate) entropy_limit: u32,
}

impl RuleConditions {
    /// The parameters in the packed order the batched descriptor stores them.
    pub(crate) fn descriptor_words(&self) -> [u32; 9] {
        [
            self.pattern_a,
            self.pattern_b,
            self.pattern_c,
            self.pattern_d,
            self.min_count,
            self.max_offset,
            self.min_size,
            self.max_size,
            self.entropy_limit,
        ]
    }
}

/// Derive one rule's condition parameters from its index.
///
/// `pattern_count` must be a power of two: the pattern indices are masked, not
/// reduced, so a non-power-of-two count would skew the selection rather than
/// wrap it.
pub(crate) fn rule_conditions(
    rule: u32,
    pattern_count: u32,
    filesize_bytes: u32,
) -> RuleConditions {
    let seed = mix32(rule);
    RuleConditions {
        pattern_a: seed & (pattern_count - 1),
        pattern_b: mix32(seed ^ 0x9E37_79B9) & (pattern_count - 1),
        pattern_c: mix32(seed ^ 0x85EB_CA6B) & (pattern_count - 1),
        pattern_d: mix32(seed ^ 0xC2B2_AE35) & (pattern_count - 1),
        min_count: (seed >> 5) % 7 + 1,
        max_offset: filesize_bytes - ((seed >> 11) % (filesize_bytes / 2)),
        min_size: filesize_bytes - ((seed >> 17) & 4095),
        max_size: filesize_bytes + ((seed >> 3) & 8191),
        entropy_limit: 600 + ((seed >> 9) % 320),
    }
}

/// The pattern-metadata streams one rule is scored against.
pub(crate) struct PatternStreams<'a> {
    pub(crate) matched: &'a [u32],
    pub(crate) counts: &'a [u32],
    pub(crate) offsets: &'a [u32],
}

/// Whether one rule fires: both literals matched, enough matches of the third
/// pattern, the fourth pattern early enough, the file within the size window, and
/// its entropy at or below the limit.
///
/// This is the host oracle both cases are scored against, so a divergence between
/// two copies of it would read as a device correctness violation in whichever
/// case held the stale copy.
pub(crate) fn rule_fires(
    streams: &PatternStreams<'_>,
    rule: &RuleConditions,
    filesize_bytes: u32,
    entropy_millibits: u32,
) -> bool {
    streams.matched[rule.pattern_a as usize] != 0
        && streams.matched[rule.pattern_b as usize] != 0
        && streams.counts[rule.pattern_c as usize] >= rule.min_count
        && streams.offsets[rule.pattern_d as usize] <= rule.max_offset
        && filesize_bytes >= rule.min_size
        && filesize_bytes <= rule.max_size
        && entropy_millibits <= rule.entropy_limit
}

/// Bind the four pattern indices this lane's rule selects.
///
/// The initializers are the caller's: one case loads them from four per-rule
/// buffers, the other from four words of one packed descriptor. The bound names
/// are what the predicates below read.
pub(crate) fn pattern_index_binds(
    pattern_a: Expr,
    pattern_b: Expr,
    pattern_c: Expr,
    pattern_d: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("pa", pattern_a),
        Node::let_bind("pb", pattern_b),
        Node::let_bind("pc", pattern_c),
        Node::let_bind("pd", pattern_d),
    ]
}

/// The three predicates read from the pattern-metadata streams.
pub(crate) fn stream_predicates(min_count: Expr, max_offset: Expr) -> Vec<Node> {
    vec![
        Node::let_bind(
            "both_literals",
            Expr::and(
                Expr::ne(Expr::load("matched", Expr::var("pa")), Expr::u32(0)),
                Expr::ne(Expr::load("matched", Expr::var("pb")), Expr::u32(0)),
            ),
        ),
        Node::let_bind(
            "count_ok",
            Expr::ge(Expr::load("counts", Expr::var("pc")), min_count),
        ),
        Node::let_bind(
            "offset_ok",
            Expr::le(Expr::load("offsets", Expr::var("pd")), max_offset),
        ),
    ]
}

/// The two predicates read from the file's own metadata.
///
/// `filesize` and `entropy` are expressions rather than values because one case
/// evaluates a single file whose size and entropy are graph constants, and the
/// other reads them per lane from per-file buffers.
pub(crate) fn file_metadata_predicates(
    filesize: Expr,
    min_size: Expr,
    max_size: Expr,
    entropy: Expr,
    entropy_limit: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind(
            "size_ok",
            Expr::and(
                Expr::ge(filesize.clone(), min_size),
                Expr::le(filesize, max_size),
            ),
        ),
        Node::let_bind("entropy_ok", Expr::le(entropy, entropy_limit)),
    ]
}

/// Conjoin the five predicates and append this lane's identifier to the sparse
/// output through the atomic counter.
pub(crate) fn fired_append(fired_buffer: &str) -> Vec<Node> {
    vec![
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
                Node::store(fired_buffer, Expr::var("slot"), Expr::var("tid")),
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const LABELS: ConditionalLabels = ConditionalLabels {
        metric_prefix: "conditional_test",
        subject: "conditional test",
        fired_noun: "fired-rule",
        wire_context: "conditional-test output",
    };

    fn words(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn prepared() -> ConditionalPrepared {
        ConditionalPrepared {
            program: Program::wrapped(vec![], [256, 1, 1], vec![]),
            reset_program: Program::wrapped(vec![], [1, 1, 1], vec![]),
            inputs: vec![],
            input_bytes_total: 1_024,
            baseline_output: vec![words(&[1]), words(&[7])],
            baseline_wall_ns: 99,
            resident: None,
            reset_indices: &[0],
            condition_indices: &[0, 1],
            fired_count_resource: 0,
            fired_ids_resource: 1,
            eval_count: 16,
            labels: LABELS,
        }
    }

    /// A sparse buffer shorter than the count it reports is a correctness
    /// violation, not a silently truncated comparison.
    #[test]
    fn sparse_verifier_rejects_a_buffer_shorter_than_its_reported_count() {
        let error = verify_sparse_outputs(
            LABELS,
            &[words(&[3]), words(&[7])],
            Some(&[words(&[3]), words(&[7, 9, 11])]),
        )
        .expect_err("a truncated sparse buffer must never verify");

        assert!(
            error.to_string().contains("shorter than reported count"),
            "{error}"
        );
    }

    /// Atomic append order is not part of the contract; the set is.
    #[test]
    fn sparse_verifier_accepts_the_same_set_in_a_different_order() {
        let correctness = verify_sparse_outputs(
            LABELS,
            &[words(&[3]), words(&[11, 7, 9])],
            Some(&[words(&[3]), words(&[7, 9, 11])]),
        )
        .expect("atomic append order must not fail correctness");

        assert!(matches!(correctness, Correctness::Exact));
    }

    /// Equal counts with differing members must fail.
    #[test]
    fn sparse_verifier_rejects_equal_counts_with_different_members() {
        let error = verify_sparse_outputs(
            LABELS,
            &[words(&[2]), words(&[7, 9])],
            Some(&[words(&[2]), words(&[7, 10])]),
        )
        .expect_err("a differing fired set must never verify");

        assert!(
            error.to_string().contains("differs between backend"),
            "{error}"
        );
    }

    /// A run with no captured baseline cannot claim exactness.
    #[test]
    fn sparse_verifier_rejects_a_missing_baseline() {
        let error = verify_sparse_outputs(LABELS, &[words(&[1]), words(&[7])], None)
            .expect_err("a missing baseline must never verify");

        assert!(
            error.to_string().contains("did not capture baseline"),
            "{error}"
        );
    }

    /// Device timing measured by the resident sequence must reach the sample.
    /// Both conditional cases assemble their sample here, so neither can drop
    /// it the way the batched copy once did.
    #[test]
    fn measured_sample_propagates_device_dispatch_time() {
        let run = conditional_bench_run(
            &prepared(),
            false,
            ConditionalSample {
                outputs: vec![words(&[1]), words(&[7])],
                wall_ns: 4_242,
                dispatch_ns: Some(1_337),
                resident_used: true,
                device_reset_sequence: true,
            },
        );

        assert_eq!(run.metrics.dispatch_ns, Some(1_337));
        assert_eq!(run.metrics.wall_ns, Some(4_242));
        assert_eq!(run.metrics.input_bytes, Some(1_024));
    }

    /// The baseline copy is attached only when the runner asked for it.
    #[test]
    fn baseline_outputs_are_attached_only_on_request() {
        let prepared = prepared();
        let sample = || ConditionalSample {
            outputs: Vec::new(),
            wall_ns: 1,
            dispatch_ns: None,
            resident_used: false,
            device_reset_sequence: false,
        };
        let with = conditional_bench_run(&prepared, true, sample());
        let without = conditional_bench_run(&prepared, false, sample());

        assert_eq!(
            with.baseline_outputs.as_deref(),
            Some(&prepared.baseline_output[..])
        );
        assert!(without.baseline_outputs.is_none());
    }

    /// Both cases score rules against the same generated pattern metadata.
    #[test]
    fn pattern_streams_stay_within_their_declared_ranges() {
        let (matched, counts, offsets) = pattern_streams(1_024, 4_096);

        assert_eq!(matched.len(), 1_024);
        assert!(matched.iter().all(|value| *value <= 1));
        assert!(counts.iter().all(|value| (1..=8).contains(value)));
        assert!(offsets.iter().all(|value| *value < 4_096));
    }

    /// Golden pin on the rule derivation both cases now share.
    ///
    /// These are the values the two open-coded copies produced. A change to the
    /// mixer, the mask, or any of the nine expressions moves the recorded
    /// workload identity of both release cases, so it has to be a deliberate
    /// edit to this table rather than a side effect of touching one case.
    #[test]
    fn rule_parameters_are_pinned_to_the_recorded_workload() {
        let rule = rule_conditions(0, 1 << 14, 10 * 1024 * 1024);
        assert_eq!(
            rule.descriptor_words(),
            [0, 9_554, 10_354, 15_586, 1, 10_485_760, 10_485_760, 10_485_760, 600]
        );

        let rule = rule_conditions(7, 1 << 14, 10 * 1024 * 1024);
        assert_eq!(
            rule.descriptor_words(),
            [8_678, 1_983, 10_037, 3_661, 6, 9_268_876, 10_483_131, 10_490_940, 616]
        );
    }

    /// Every parameter has to be able to veto the rule on its own.
    ///
    /// A predicate dropped from the shared conjunction would still pass a test
    /// that only checks a firing rule, and would then disagree with the device
    /// program in exactly one direction: extra pairs reported as fired.
    #[test]
    fn each_condition_can_veto_the_rule_on_its_own() {
        let matched = [1_u32; 4];
        let counts = [8_u32; 4];
        let offsets = [16_u32; 4];
        let streams = PatternStreams {
            matched: &matched,
            counts: &counts,
            offsets: &offsets,
        };
        let firing = RuleConditions {
            pattern_a: 0,
            pattern_b: 1,
            pattern_c: 2,
            pattern_d: 3,
            min_count: 8,
            max_offset: 16,
            min_size: 1_000,
            max_size: 2_000,
            entropy_limit: 640,
        };
        assert!(rule_fires(&streams, &firing, 1_000, 640));
        assert!(rule_fires(&streams, &firing, 2_000, 0));

        let unmatched = [0_u32, 1, 1, 1];
        assert!(!rule_fires(
            &PatternStreams {
                matched: &unmatched,
                counts: &counts,
                offsets: &offsets,
            },
            &firing,
            1_000,
            640
        ));
        let second_unmatched = [1_u32, 0, 1, 1];
        assert!(!rule_fires(
            &PatternStreams {
                matched: &second_unmatched,
                counts: &counts,
                offsets: &offsets,
            },
            &firing,
            1_000,
            640
        ));

        for veto in [
            RuleConditions {
                min_count: 9,
                ..firing
            },
            RuleConditions {
                max_offset: 15,
                ..firing
            },
        ] {
            assert!(!rule_fires(&streams, &veto, 1_000, 640));
        }

        assert!(!rule_fires(&streams, &firing, 999, 640));
        assert!(!rule_fires(&streams, &firing, 2_001, 640));
        assert!(!rule_fires(&streams, &firing, 1_000, 641));
    }
}
