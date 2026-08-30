//! Observable contracts for conformance semantic requests and execution evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::{
    check_family_outputs, check_schedule_agreement, submit_under_every_schedule, ProductionSession,
    ScheduleAgreement, ScheduleDisagreement, ScheduleOutcome, CONFORMANCE_SCHEDULES,
};
use vyre_megakernel::{
    CompileObjective, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric, RequiredSchedule,
    ScheduleProduction, SearchBudget, SemanticExecutionError, SemanticExecutionOutput,
    SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};

#[derive(Debug, PartialEq, Eq)]
struct ObservedRequest {
    inputs: Vec<Vec<u8>>,
    objective: CompileObjective,
    budget: SearchBudget,
    target_facts: DeviceFacts,
}

struct RecordingExecutor {
    observed: Mutex<Option<ObservedRequest>>,
    artifact: Digest,
    payload: Digest,
    output: Vec<u8>,
}

impl SemanticExecutor for RecordingExecutor {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = request
            .inputs()
            .values()
            .map(|bytes| bytes.to_vec())
            .collect();
        *self.observed.lock().expect("recording executor lock") = Some(ObservedRequest {
            inputs,
            objective: *request.policy().objective(),
            budget: request.policy().budget(),
            target_facts: request.policy().target_facts(),
        });
        let terminal_values = request
            .logical()
            .graph()
            .values()
            .iter()
            .filter(|value| value.producer.is_some() && value.consumers.is_empty())
            .map(|value| value.id)
            .collect::<BTreeSet<_>>();
        let outputs = terminal_values
            .into_iter()
            .map(|value| (value, self.output.clone()))
            .collect::<BTreeMap<_, _>>();
        Ok(SemanticExecutionOutput {
            artifact: self.artifact,
            payload: self.payload,
            outputs,
        })
    }
}

/// WHY: conformance must pass schedule-free semantics and explicit target policy
/// to the compiler boundary, then retain the admitted identities it returns.
#[test]
fn semantic_request_and_admitted_output_cross_the_production_boundary() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1),
            BufferDecl::output("output", 1, DataType::U32).with_count(1),
        ],
        [8, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(0),
            Expr::load("input", Expr::u32(0)),
        )],
    );
    let artifact = Digest([17; 32]);
    let payload = Digest([29; 32]);
    let expected_output = 41_u32.to_le_bytes().to_vec();
    let executor = Arc::new(RecordingExecutor {
        observed: Mutex::new(None),
        artifact,
        payload,
        output: expected_output.clone(),
    });
    let target_facts = DeviceFacts::unknown();
    let budget = SearchBudget::new(19, 23, 1, 1, 31);
    let objective =
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 65_536);
    let policy = SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([7; 32]), BTreeMap::new()),
        target_facts,
        objective,
        budget,
    );
    let session = ProductionSession::with_executor(
        &program,
        executor.clone(),
        policy,
        "recording-semantic-backend",
    );
    let input = 37_u32.to_le_bytes();

    let execution = session
        .submit(&[&input])
        .expect("semantic executor must receive a valid canonical request");

    assert_eq!(execution.artifact, artifact);
    assert_eq!(execution.payload, payload);
    assert_eq!(execution.outputs, vec![expected_output]);
    assert_eq!(
        *executor.observed.lock().expect("recording executor lock"),
        Some(ObservedRequest {
            inputs: vec![input.to_vec()],
            objective,
            budget,
            target_facts,
        })
    );
}

/// Records the required schedule family of every request it is given.
struct ScheduleRecordingExecutor {
    seen: Mutex<Vec<Option<RequiredSchedule>>>,
    output: Vec<u8>,
}

impl SemanticExecutor for ScheduleRecordingExecutor {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        self.seen
            .lock()
            .expect("schedule recording executor lock")
            .push(request.policy().constraints().required_schedule());
        let outputs = request
            .logical()
            .graph()
            .values()
            .iter()
            .filter(|value| value.producer.is_some() && value.consumers.is_empty())
            .map(|value| (value.id, self.output.clone()))
            .collect::<BTreeMap<_, _>>();
        Ok(SemanticExecutionOutput {
            artifact: Digest([3; 32]),
            payload: Digest([5; 32]),
            outputs,
        })
    }
}

/// The one-node copy program every schedule-family case runs.
fn copy_one_word() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1),
            BufferDecl::output("output", 1, DataType::U32).with_count(1),
        ],
        [8, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(0),
            Expr::load("input", Expr::u32(0)),
        )],
    )
}

/// WHY: the requirement is stated on the policy and enforced inside the
/// compiler, so the two are joined only by the semantic request that carries it.
/// A request that dropped the family would run one schedule six times and every
/// case would pass, which is the shape of a conformance suite that proves
/// nothing. This asserts the family reaches the executor for every entry of
/// `CONFORMANCE_SCHEDULES`, in order.
#[test]
fn every_conformance_schedule_reaches_the_compiler_boundary() {
    let executor = Arc::new(ScheduleRecordingExecutor {
        seen: Mutex::new(Vec::new()),
        output: 11_u32.to_le_bytes().to_vec(),
    });
    let program = copy_one_word();
    let session = ProductionSession::with_executor(
        &program,
        executor.clone(),
        SemanticExecutionPolicy::new(
            ExternalFacts::new(Digest([7; 32]), BTreeMap::new()),
            DeviceFacts::unknown(),
            CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 65_536),
            SearchBudget::new(19, 23, 1, 1, 31),
        ),
        "recording-semantic-backend",
    );
    let input = 37_u32.to_le_bytes();

    let outcomes = submit_under_every_schedule(&session, &[&input])
        .expect("a recording executor refuses no family");

    assert_eq!(
        outcomes
            .iter()
            .map(|outcome| (outcome.schedule, outcome.required))
            .collect::<Vec<_>>(),
        CONFORMANCE_SCHEDULES.to_vec(),
        "every declared family must be run, under the name it is declared with"
    );
    assert_eq!(
        *executor
            .seen
            .lock()
            .expect("schedule recording executor lock"),
        CONFORMANCE_SCHEDULES
            .iter()
            .map(|(_, required)| Some(*required))
            .collect::<Vec<_>>(),
        "the family stated on the policy must reach the compiler boundary"
    );
}

/// WHY: a production added to the grammar is a schedule a conformance case can
/// run one semantic graph under, and one nobody has decided about is one nobody
/// checks. The roster is derived from `ScheduleProduction::ALL` at run time, so a
/// new production turns this red until it is either added to
/// `CONFORMANCE_SCHEDULES` or recorded here as a family conformance does not
/// range over, with the reason stated beside it.
#[test]
fn every_grammar_production_is_a_conformance_family_or_a_recorded_exclusion() {
    /// Productions no conformance family requires, and why.
    ///
    /// Each one is a refinement of a schedule another family already forces
    /// rather than a distinct way of executing the graph, so requiring it would
    /// run a variant of a case already run.
    const EXCLUDED: &[(ScheduleProduction, &str)] = &[
        (ScheduleProduction::Fission, "the inverse of the fused case"),
        (
            ScheduleProduction::Pipeline,
            "an overlap of the concurrent case",
        ),
        (
            ScheduleProduction::DispatchCut,
            "a submission boundary inside the multi-invocation case",
        ),
        (
            ScheduleProduction::AsymmetricJoin,
            "a fan-in shape of the fused case",
        ),
        (
            ScheduleProduction::Synchronization,
            "an ordering boundary inside every other family",
        ),
        (
            ScheduleProduction::MemoryPlacement,
            "a storage class, not an execution schedule",
        ),
        (
            ScheduleProduction::Prefetch,
            "a distance inside the concurrent case",
        ),
        (
            ScheduleProduction::Recomputation,
            "a materialization choice, not an execution schedule",
        ),
        (ScheduleProduction::AxisSplit, "a factor of the tiled case"),
        (
            ScheduleProduction::Vectorization,
            "a lane width inside the tiled case",
        ),
        (
            ScheduleProduction::AxisMapping,
            "a hierarchy level inside the tiled case",
        ),
        (
            ScheduleProduction::AxisReorder,
            "an axis order inside the tiled case",
        ),
    ];

    for production in ScheduleProduction::ALL {
        let required = CONFORMANCE_SCHEDULES
            .iter()
            .any(|(_, family)| *family == RequiredSchedule::Production(*production));
        let excluded = EXCLUDED.iter().any(|(family, _)| family == production);
        assert!(
            required != excluded,
            "production {} is {} of `CONFORMANCE_SCHEDULES` and {} of the \
             recorded exclusions; decide one",
            production.code(),
            if required { "a member" } else { "absent" },
            if excluded { "a member" } else { "absent" }
        );
    }
}

fn lanes(values: &[f32]) -> Vec<Vec<u8>> {
    vec![values.iter().flat_map(|v| v.to_le_bytes()).collect()]
}

/// WHY: an exact contract admits nothing but byte equality, so a family that
/// reorders a sum by one bit is a conformance failure rather than a tolerance.
#[test]
fn an_exact_contract_rejects_a_single_bit_of_reordering() {
    let baseline = lanes(&[1.0, 2.0]);
    let mut found = baseline.clone();
    found[0][0] ^= 1;
    let error = check_family_outputs("tiled", &baseline, &found, ScheduleAgreement::Exact)
        .expect_err("an exact contract must reject a changed byte");
    let ScheduleDisagreement::Lane {
        schedule,
        buffer,
        lane,
        ..
    } = error
    else {
        panic!("expected a lane disagreement, found {error:?}");
    };
    assert_eq!((schedule, buffer, lane), ("tiled", 0, 0));
    check_family_outputs("tiled", &baseline, &baseline, ScheduleAgreement::Exact)
        .expect("byte-identical outputs satisfy an exact contract");
}

/// WHY: a tolerance contract states a bound, so it has to admit a family inside
/// the bound and refuse one outside it; a bound that admits everything proves
/// nothing about a reassociated sum.
#[test]
fn a_tolerance_contract_admits_its_bound_and_refuses_beyond_it() {
    let baseline = lanes(&[1.0, 4.0]);
    let near = lanes(&[f32::from_bits(1.0f32.to_bits() + 2), 4.0]);
    let far = lanes(&[f32::from_bits(1.0f32.to_bits() + 5), 4.0]);
    let agreement = ScheduleAgreement::Float32Ulps { ulps: 2 };
    check_family_outputs("fused", &baseline, &near, agreement)
        .expect("a lane at the declared bound is admitted");
    let error = check_family_outputs("fused", &baseline, &far, agreement)
        .expect_err("a lane beyond the declared bound is refused");
    assert!(
        matches!(error, ScheduleDisagreement::Lane { distance: 5, .. }),
        "expected the measured distance, found {error:?}"
    );
}

/// WHY: a sign flip and a non-finite lane are class changes no unit-in-last-place
/// bound expresses, so only bit equality may admit them.
#[test]
fn a_tolerance_contract_never_admits_a_sign_or_class_change() {
    let agreement = ScheduleAgreement::Float32Ulps { ulps: u32::MAX - 1 };
    for (baseline, found) in [
        (lanes(&[0.0]), lanes(&[-0.0])),
        (lanes(&[1.0]), lanes(&[-1.0])),
        (lanes(&[1.0]), lanes(&[f32::NAN])),
        (lanes(&[1.0]), lanes(&[f32::INFINITY])),
    ] {
        let error = check_family_outputs("concurrent", &baseline, &found, agreement)
            .expect_err("a class change must be refused");
        assert!(
            matches!(
                error,
                ScheduleDisagreement::Lane {
                    distance: u32::MAX,
                    ..
                }
            ),
            "expected a class change, found {error:?}"
        );
    }
    let nan = lanes(&[f32::NAN]);
    check_family_outputs("concurrent", &nan, &nan, agreement)
        .expect("identical non-finite lanes are admitted");
}

/// WHY: a family that produces a different number or size of writable buffers
/// has changed the semantic contract, which no numeric tolerance covers.
#[test]
fn a_family_that_changes_the_output_shape_is_refused() {
    let baseline = lanes(&[1.0, 2.0]);
    let count = check_family_outputs("persistent", &baseline, &[], ScheduleAgreement::Exact)
        .expect_err("a missing buffer must be refused");
    assert_eq!(
        count,
        ScheduleDisagreement::BufferCount {
            schedule: "persistent",
            baseline: 1,
            found: 0,
        }
    );
    let length = check_family_outputs(
        "persistent",
        &baseline,
        &lanes(&[1.0]),
        ScheduleAgreement::Exact,
    )
    .expect_err("a shortened buffer must be refused");
    assert_eq!(
        length,
        ScheduleDisagreement::BufferLength {
            schedule: "persistent",
            buffer: 0,
            baseline: 8,
            found: 4,
        }
    );
    let unaligned = vec![vec![0u8; 6]];
    let alignment = check_family_outputs(
        "persistent",
        &unaligned,
        &unaligned,
        ScheduleAgreement::Float32Ulps { ulps: 0 },
    )
    .expect_err("a buffer that is not a whole number of lanes must be refused");
    assert_eq!(
        alignment,
        ScheduleDisagreement::LaneAlignment {
            schedule: "persistent",
            buffer: 0,
            bytes: 6,
        }
    );
}

/// WHY: comparison is against the unspecialized baseline, so a run whose
/// baseline produced nothing must fail rather than silently promote another
/// family to reference.
#[test]
fn agreement_without_a_baseline_execution_is_refused() {
    let outcomes = CONFORMANCE_SCHEDULES
        .iter()
        .map(|(schedule, required)| ScheduleOutcome {
            schedule,
            required: *required,
            execution: None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        check_schedule_agreement(&outcomes, ScheduleAgreement::Exact),
        Err(ScheduleDisagreement::NoBaseline)
    );
}
