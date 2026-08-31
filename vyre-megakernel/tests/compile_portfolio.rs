//! Contracts for the artifact set one objective retains.
//!
//! WHY: the objective states a coverage policy, an artifact ceiling, and an
//! aggregate byte ceiling, and nothing consumed them. A compile emitted one
//! artifact whatever the policy said, so an objective demanding one artifact per
//! workload class was accepted and then served by a single schedule selected for
//! the weighted average of arrangements that disagree. These tests defend the
//! joint decision that replaced it: a single-artifact path refuses a coverage
//! policy it cannot satisfy, portfolio selection enumerates the legal partitions
//! of the stated classes and retains the one the objective orders first, every
//! retained artifact records the arrangement it was selected for, and the
//! variant and aggregate bounds reject a set instead of pricing it.
//!
//! What these do not catch: nothing here proves the modeled figure of a
//! per-class artifact matches a device. That is the measurement protocol's
//! contract, not the portfolio's.

#![forbid(unsafe_code)]

use vyre_megakernel::{
    compile, compile_measured, compile_portfolio, compile_portfolio_measured, Artifact,
    ArtifactPortfolio, CompileObjective, CoveragePolicy, DeclaredConstraints, EmittedResources,
    FinalistEvaluator, ObjectiveMetric, PortfolioPolicy, PruneReason, RequiredSchedule,
    SearchBudget, TargetCompileError, TargetCompiler, TargetPayload, TargetPayloadFormat,
    TargetProfile, ValidatedCompileRequest, WorkloadAggregation, WorkloadClass, WorkloadProfile,
};

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{budget, fixture_request, launch_bound_device, validated, ARTIFACT_BYTES};

/// An interactive submission: one launch, one stream, half the workload.
fn interactive() -> WorkloadClass {
    WorkloadClass::new(1, 1, 500)
}

/// A batch submission: a thousand launches on four streams, half the workload.
fn batched() -> WorkloadClass {
    WorkloadClass::new(1_000, 4, 500)
}

/// A third arrangement, so a three-class profile can be partitioned.
fn concurrent() -> WorkloadClass {
    WorkloadClass::new(8, 8, 0)
}

/// The two-class profile the coverage cases share.
fn two_classes() -> WorkloadProfile {
    WorkloadProfile::of(interactive()).pushed(batched())
}

/// An objective over `profile` with `coverage` and an artifact ceiling.
fn objective(
    profile: WorkloadProfile,
    coverage: CoveragePolicy,
    variants: u32,
) -> CompileObjective {
    CompileObjective::minimize_latency()
        .with_workload(profile)
        .with_portfolio(PortfolioPolicy::new(coverage, variants))
        .with_bound(ObjectiveMetric::ArtifactBytes, ARTIFACT_BYTES)
}

fn request(objective: CompileObjective) -> ValidatedCompileRequest {
    validated(launch_bound_device(), budget(), objective)
}

/// Bytes of one artifact, which is what an aggregate byte bound counts.
fn artifact_bytes(artifact: &Artifact) -> u64 {
    u64::try_from(
        artifact
            .to_bytes()
            .expect("Fix: keep the artifact serializable")
            .len(),
    )
    .expect("Fix: keep the artifact length within u64")
}

/// Distinct artifact count the assignment reaches, derived from the assignment
/// rather than from the retained vector, so the two must agree.
fn assigned_variants(portfolio: &ArtifactPortfolio) -> usize {
    let mut seen = portfolio.assignment().to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

/// A target facet that refuses every artifact, so a guard that runs before
/// emission is proved to run before emission.
struct RefusingCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl RefusingCompiler {
    fn new() -> Self {
        Self {
            format: TargetPayloadFormat::new("test.refusing-target", 1)
                .expect("Fix: state a valid fixture payload format"),
            profile: TargetProfile::new("test.refusing-target", 1, [256, 1, 1], 256, 64 * 1024, 0)
                .expect("Fix: state a valid fixture target profile"),
        }
    }
}

impl TargetCompiler for RefusingCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(&self, _artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        Err(TargetCompileError::Unsupported(
            "the fixture target builds nothing".into(),
        ))
    }
}

struct RefusingEvaluator {
    compiler: RefusingCompiler,
}

impl FinalistEvaluator for RefusingEvaluator {
    fn target_compiler(&self) -> &dyn TargetCompiler {
        &self.compiler
    }

    fn resources(
        &self,
        _artifact: &Artifact,
        _payload: &TargetPayload,
    ) -> Result<Vec<EmittedResources>, TargetCompileError> {
        Err(TargetCompileError::Unsupported(
            "the fixture target reports nothing".into(),
        ))
    }

    fn measure(
        &self,
        _artifact: &Artifact,
        _payload: &TargetPayload,
    ) -> Result<u64, TargetCompileError> {
        Err(TargetCompileError::Unsupported(
            "the fixture target times nothing".into(),
        ))
    }

    fn device_state(&self) -> vyre_megakernel::measure::DeviceState {
        vyre_megakernel::measure::DeviceState::unreported()
    }
}

/// WHY: a compile that emits one artifact under a policy needing several used to
/// succeed and return the one, which is the defect: the caller asked for a set
/// and was handed a member of it with nothing saying so.
#[test]
fn a_single_artifact_compile_refuses_a_coverage_policy_it_cannot_satisfy() {
    let request = request(objective(
        two_classes(),
        CoveragePolicy::EveryWorkloadClass,
        2,
    ));
    let error =
        compile(&request).expect_err("one artifact cannot serve a per-class coverage policy");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC032_PORTFOLIO_COVERAGE_UNSATISFIED"
    );
    let fix = error
        .diagnostic
        .suggested_fix
        .as_deref()
        .expect("the diagnostic must state a corrective action");
    assert!(
        fix.contains("compile_portfolio"),
        "the fix must name the path that retains a set, got `{fix}`"
    );
}

/// WHY: both single-artifact entry points share one guard, and a guard wired
/// into one of two routes is how the second route keeps the old behavior.
#[test]
fn the_measured_single_artifact_compile_refuses_the_same_policy() {
    let request = request(objective(
        two_classes(),
        CoveragePolicy::EveryWorkloadClass,
        2,
    ));
    let evaluator = RefusingEvaluator {
        compiler: RefusingCompiler::new(),
    };
    let error = compile_measured(&request, &evaluator)
        .expect_err("one measured artifact cannot serve a per-class coverage policy");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC032_PORTFOLIO_COVERAGE_UNSATISFIED",
        "the coverage guard must refuse before the target is asked for anything"
    );
}

/// WHY: the common case must not change. An objective retaining one artifact has
/// to produce the artifact the single-artifact path produces, byte for byte, or
/// the portfolio path is a second compiler with its own answer.
#[test]
fn a_single_coverage_portfolio_retains_the_artifact_the_single_compile_selects() {
    let objective = CompileObjective::minimize_latency()
        .with_bound(ObjectiveMetric::ArtifactBytes, ARTIFACT_BYTES);
    let direct = compile(&request(objective)).expect("the single-artifact path must compile");
    let portfolio = compile_portfolio(&request(objective)).expect("the portfolio must compile");
    assert_eq!(portfolio.variants(), 1);
    assert_eq!(portfolio.assignment(), [0_u32].as_slice());
    assert_eq!(
        artifact_bytes(&direct),
        portfolio.aggregate_bytes(),
        "one retained artifact is the artifact the single path selects"
    );
    assert_eq!(
        direct.to_bytes().expect("bytes"),
        portfolio.artifacts()[0].to_bytes().expect("bytes"),
        "the two paths must not select different plans for the same objective"
    );
}

/// WHY: coverage is the whole point of a set. Every stated class must resolve to
/// a retained artifact, and the retained vector and the assignment must agree on
/// how many there are.
#[test]
fn every_workload_class_is_served_by_a_retained_artifact() {
    let profile = WorkloadProfile::of(interactive())
        .pushed(batched())
        .pushed(concurrent());
    let portfolio = compile_portfolio(&request(objective(
        profile,
        CoveragePolicy::EveryWorkloadClass,
        3,
    )))
    .expect("a three-class portfolio must compile");
    assert_eq!(portfolio.assignment().len(), 3, "every class is assigned");
    assert_eq!(assigned_variants(&portfolio), portfolio.artifacts().len());
    for class in 0..3 {
        assert!(
            portfolio.artifact_for_class(class).is_some(),
            "class {class} must be served by a retained artifact"
        );
    }
    assert!(portfolio.artifact_for_class(3).is_none());
    let expected = portfolio
        .artifacts()
        .iter()
        .map(artifact_bytes)
        .sum::<u64>();
    assert_eq!(
        portfolio.aggregate_bytes(),
        expected,
        "the aggregate is the bytes of the set, counted once per retained artifact"
    );
}

/// WHY: the assignment is a partition, and a partition enumerated per
/// relabelling of its parts costs an exponential factor of compiles and lets two
/// runs retain the same set under two labellings. Restricted growth is the
/// canonical form: part indices are dense and introduced in order.
#[test]
fn the_assignment_labels_parts_in_canonical_order() {
    let profile = WorkloadProfile::of(interactive())
        .pushed(batched())
        .pushed(concurrent());
    let portfolio = compile_portfolio(&request(objective(
        profile,
        CoveragePolicy::EveryWorkloadClass,
        3,
    )))
    .expect("a three-class portfolio must compile");
    let mut highest = 0_u32;
    for (position, part) in portfolio.assignment().iter().copied().enumerate() {
        if position == 0 {
            assert_eq!(part, 0, "the first class opens the first part");
        }
        assert!(
            part <= highest,
            "part {part} appears before part {highest} was opened"
        );
        if part == highest {
            highest += 1;
        }
    }
    assert_eq!(
        usize::try_from(highest).expect("part count fits"),
        portfolio.artifacts().len(),
        "every opened part retains exactly one artifact"
    );
}

/// WHY: a retained artifact that does not state which arrangement it was
/// selected for is indistinguishable from any other member of the set, and a
/// runtime holding several then picks by position.
#[test]
fn each_retained_artifact_records_the_arrangement_it_was_selected_for() {
    let portfolio = compile_portfolio(&request(objective(
        two_classes(),
        CoveragePolicy::EveryWorkloadClass,
        2,
    )))
    .expect("a two-class portfolio must compile");
    assert_eq!(portfolio.artifacts().len(), 2, "one artifact per class");
    for (index, class) in [interactive(), batched()].into_iter().enumerate() {
        let artifact = portfolio
            .artifact_for_class(index)
            .expect("every class is served");
        let recorded = artifact.provenance().objective;
        let stated = recorded.workload().as_slice();
        assert_eq!(stated.len(), 1, "a part optimizes for its own class alone");
        assert_eq!(stated[0].launch_batch, class.launch_batch);
        assert_eq!(stated[0].concurrent_streams, class.concurrent_streams);
        assert_eq!(
            stated[0].weight_permille, 1_000,
            "a part's own class carries the whole weight of that part"
        );
        assert_eq!(
            recorded.portfolio().coverage(),
            CoveragePolicy::Single,
            "a part is the set of classes one artifact serves"
        );
    }
}

/// WHY: the coverage policy is part of what a compile optimized. Two sets
/// selected under different policies for the same graph must not share artifact
/// identity, or a cache serves a per-class artifact to a caller that asked for
/// one covering every class.
#[test]
fn changing_the_coverage_policy_changes_the_retained_artifacts() {
    let merged = compile_portfolio(&request(objective(
        two_classes(),
        CoveragePolicy::Single,
        2,
    )))
    .expect("a merged portfolio must compile");
    let split = compile_portfolio(&request(objective(
        two_classes(),
        CoveragePolicy::EveryWorkloadClass,
        2,
    )))
    .expect("a split portfolio must compile");
    assert_eq!(merged.variants(), 1);
    assert_eq!(split.variants(), 2);
    let merged_bytes = merged.artifacts()[0].to_bytes().expect("bytes");
    for artifact in split.artifacts() {
        assert_ne!(
            artifact.to_bytes().expect("bytes"),
            merged_bytes,
            "a per-class artifact must not be identical to the covering one"
        );
    }
}

/// WHY: a bound is rejected, never priced. A variant bound below what the
/// coverage policy needs must fail naming the bound the caller wrote, not the
/// coverage policy it contradicts, and must never silently retain fewer
/// artifacts than the policy states.
#[test]
fn a_variant_bound_below_the_coverage_minimum_retains_nothing() {
    let objective = objective(two_classes(), CoveragePolicy::EveryWorkloadClass, 2)
        .with_bound(ObjectiveMetric::VariantCount, 1);
    let error = compile_portfolio(&request(objective))
        .expect_err("one artifact cannot cover two classes under a per-class policy");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC031_OBJECTIVE_BOUND_VIOLATED"
    );
    let path = error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
        .expect("the diagnostic must name the field it refused");
    assert_eq!(path, "request.objective.bounds.variant_count");
    assert!(
        error.diagnostic.message.contains("needs 2"),
        "the diagnostic must state what the coverage policy retains, got `{}`",
        error.diagnostic.message
    );
}

/// WHY: a variant bound of zero retains no artifact at all, and a search that
/// enumerated no partition used to report that as an empty candidate set, which
/// names the search instead of the bound the caller stated.
#[test]
fn a_variant_bound_of_zero_names_the_bound_it_refused_for() {
    let objective = objective(two_classes(), CoveragePolicy::EveryWorkloadClass, 2)
        .with_bound(ObjectiveMetric::VariantCount, 0);
    let error =
        compile_portfolio(&request(objective)).expect_err("a zero variant bound retains nothing");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC031_OBJECTIVE_BOUND_VIOLATED"
    );
    let path = error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
        .expect("the diagnostic must name the field it refused");
    assert_eq!(path, "request.objective.bounds.variant_count");
}

/// WHY: retaining a set costs bytes the caller has to keep. An aggregate ceiling
/// the set exceeds must refuse the set and report both figures, or a portfolio
/// grows past what the caller can store.
#[test]
fn an_aggregate_byte_bound_the_retained_set_exceeds_is_refused() {
    let objective = CompileObjective::minimize_latency()
        .with_workload(two_classes())
        .with_portfolio(
            PortfolioPolicy::new(CoveragePolicy::EveryWorkloadClass, 2)
                .with_max_aggregate_bytes(1_024),
        )
        .with_bound(ObjectiveMetric::ArtifactBytes, ARTIFACT_BYTES);
    let error =
        compile_portfolio(&request(objective)).expect_err("two artifacts do not fit in a kilobyte");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC031_OBJECTIVE_BOUND_VIOLATED"
    );
    let message = &error.diagnostic.message;
    assert!(
        message.contains("1024"),
        "the diagnostic must state the bound, got `{message}`"
    );
    let path = error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
        .expect("the diagnostic must name the field it refused");
    assert_eq!(path, "request.objective.portfolio.max_aggregate_bytes");
}

/// WHY: worst-case aggregation reads no weights, and a part restating weights to
/// sum to a thousand permille must stay legal under either aggregation. A part
/// objective the compiler itself refuses would make a legal request fail on a
/// derived record the caller never wrote.
#[test]
fn a_worst_case_profile_partitions_into_parts_the_compiler_accepts() {
    let profile = WorkloadProfile::of(WorkloadClass::new(1, 1, 0))
        .pushed(WorkloadClass::new(512, 2, 0))
        .with_aggregation(WorkloadAggregation::WorstCase);
    let portfolio = compile_portfolio(&request(objective(
        profile,
        CoveragePolicy::EveryWorkloadClass,
        2,
    )))
    .expect("a worst-case portfolio must compile");
    assert_eq!(portfolio.variants(), 2);
    for artifact in portfolio.artifacts() {
        assert_eq!(
            artifact.provenance().objective.workload().aggregation(),
            WorkloadAggregation::WorstCase,
            "a part keeps the aggregation the whole profile states"
        );
    }
}

/// WHY: the measured path must retain the same set the analytic path does when
/// no measurement is budgeted, and must refuse loudly rather than silently
/// falling back when the target builds nothing.
#[test]
fn a_measured_portfolio_reports_the_target_failure_it_hit() {
    let objective = objective(two_classes(), CoveragePolicy::EveryWorkloadClass, 2);
    let evaluator = RefusingEvaluator {
        compiler: RefusingCompiler::new(),
    };
    let measured_budget = SearchBudget::new(512, 200_000, 4, 8, 1_000_000_000);
    let request = validated(
        launch_bound_device().with_device_timestamps(true),
        measured_budget,
        objective,
    );
    let error = compile_portfolio_measured(&request, &evaluator)
        .expect_err("a target that builds nothing cannot produce a retained set");
    assert_eq!(error.diagnostic.code.as_str(), "MKC026_FINALIST_EVALUATION");
}

/// WHY: a partitioned compile restates the request once per part, and a part is
/// where a caller constraint is cheapest to lose: the whole request validated
/// with the requirement, so nothing downstream re-reads it. Dropping the
/// requirement from the restatement still compiles and still retains a set of
/// the ordered size, and the only visible difference is that a part selects a
/// family the caller forbade. Both assertions here are red against a
/// restatement that carries `None`: the selected plans stop being baseline
/// plans, and no part reports eliminating anything for the requirement.
#[test]
fn every_part_of_a_partitioned_compile_keeps_the_required_family() {
    let profile = WorkloadProfile::of(WorkloadClass::new(1, 1, 0))
        .pushed(WorkloadClass::new(512, 2, 0))
        .with_aggregation(WorkloadAggregation::WorstCase);
    let request = fixture_request(
        launch_bound_device(),
        budget(),
        objective(profile, CoveragePolicy::EveryWorkloadClass, 2),
    )
    .with_constraints(DeclaredConstraints::new().requiring_schedule(RequiredSchedule::Baseline))
    .validate()
    .expect("Fix: the fixture request must validate under a required family");

    let portfolio =
        compile_portfolio(&request).expect("the baseline is always in the candidate set");
    assert_eq!(portfolio.variants(), 2);

    let mut eliminated = 0_u32;
    for artifact in portfolio.artifacts() {
        let plan = artifact.selected_plan();
        assert!(
            plan.derivation.is_empty(),
            "a part selected under a required baseline applied {} production(s)",
            plan.derivation.len()
        );
        eliminated += plan
            .certificate
            .pruned_for(PruneReason::ScheduleRequirement);
    }
    assert!(
        eliminated > 0,
        "no part eliminated a candidate for the requirement, so the empty \
         derivations above prove nothing about the requirement reaching a part"
    );
}
