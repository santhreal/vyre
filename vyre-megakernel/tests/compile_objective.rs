//! What a compile optimizes, stated on the request and recorded in the artifact.
//!
//! Every contract here is about a decision the objective owns: which metric
//! orders candidates, which calibrated fact prices it, which hard bound rejects
//! one, how workload classes combine, and whether two objectives can share one
//! artifact identity. A compile that selects a plan without stating what it
//! optimized cannot be audited later, so the absence of a stated objective is a
//! validation failure rather than a default.

use std::collections::BTreeSet;

use vyre_megakernel::{
    compile, CompileObjective, CompileRequest, CoveragePolicy, MetricFigures, MetricSequence,
    ObjectiveBounds, ObjectiveMetric, PortfolioPolicy, RequiredFact, RiskStatistic,
    WorkloadAggregation, WorkloadClass, WorkloadProfile, OBJECTIVE_SCHEMA_VERSION,
};

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{
    bare_device, budget, facts, joined_graph, refused_field, rich_device, single_stage_graph,
};

/// Artifact byte ceiling every request here states, well above what the
/// fixtures emit, so an artifact-bytes bound is never what a test observes
/// unless it states a tighter one.
const CEILING: u64 = 4_000_000;

/// The latency objective the fixtures compile under.
fn latency() -> CompileObjective {
    CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, CEILING)
}

fn request(objective: CompileObjective) -> CompileRequest {
    CompileRequest::new(joined_graph(), facts(), rich_device(), budget(), objective)
}

fn error_code(objective: CompileObjective) -> String {
    let error = request(objective)
        .validate()
        .err()
        .expect("the objective must be refused");
    error.diagnostic.code.as_str().to_owned()
}

fn error_of(objective: CompileObjective) -> vyre_megakernel::CompileError {
    request(objective)
        .validate()
        .err()
        .expect("the objective must be refused")
}

fn source_variants(crate_relative: &str, declaration: &str) -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_crate_directory("vyre-megakernel")
        .join("src")
        .join(crate_relative);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {path:?} to derive the variant set: {err}"));
    let body = vyre_test_support::braced_body(&source, declaration)
        .unwrap_or_else(|| panic!("Fix: {path:?} no longer declares `{declaration}`"));
    vyre_test_support::top_level_variant_names(body)
}

fn listed<T: std::fmt::Debug>(all: &[T]) -> BTreeSet<String> {
    all.iter().map(|value| format!("{value:?}")).collect()
}

// ============================================================================
// Closure: every declared variant is reachable through the type's `ALL`
// ============================================================================

/// WHY: `ObjectiveBounds` and `MetricFigures` are positional arrays sized by
/// `ObjectiveMetric::ALL.len()` and indexed by `ObjectiveMetric::index`, which is
/// the variant's declaration position. A variant missing from `ALL` shrinks both
/// arrays by one slot while `index` keeps counting, so the last metric writes
/// out of bounds or two metrics share a slot. Rust has no reflection and the
/// enum is `#[non_exhaustive]`, so no exhaustive match in this crate can witness
/// the variant set; the declaration is the only witness.
#[test]
fn every_declared_metric_is_listed_once_at_its_own_index() {
    let declared = source_variants("objective/metric.rs", "pub enum ObjectiveMetric {");
    assert_eq!(
        declared,
        listed(ObjectiveMetric::ALL),
        "Fix: ObjectiveMetric::ALL and the declared variants disagree; add the new variant to ALL \
         in declaration order"
    );
    for (index, metric) in ObjectiveMetric::ALL.iter().enumerate() {
        assert_eq!(
            metric.index(),
            index,
            "Fix: ObjectiveMetric::{metric:?} is listed at position {index} but indexes slot {}; \
             list every metric in declaration order",
            metric.index()
        );
    }
}

/// WHY: the fact list is what a refusal names to a caller. A declared fact
/// missing from `ALL` is a fact no diagnostic can name, so a metric requiring it
/// fails with nothing actionable.
#[test]
fn every_declared_calibrated_fact_is_listed_and_names_its_builder() {
    let declared = source_variants("objective/metric.rs", "pub enum RequiredFact {");
    assert_eq!(declared, listed(RequiredFact::ALL));
    for fact in RequiredFact::ALL {
        assert!(
            !fact.name().is_empty() && !fact.supplied_by().is_empty(),
            "Fix: RequiredFact::{fact:?} states no name or no source; a refusal naming it would \
             tell a caller nothing"
        );
        assert!(
            ObjectiveMetric::ALL
                .iter()
                .any(|metric| metric.required_fact() == Some(*fact)),
            "Fix: RequiredFact::{fact:?} is required by no metric, so nothing ever checks it; \
             delete it or state the metric it prices"
        );
    }
}

/// WHY: the three policy enumerations decide an aggregate, a comparison, and a
/// retained artifact count. Each is read through its own `ALL` by validation and
/// by evidence projection, and a variant missing from one of those lists is a
/// policy no compile can be validated against.
#[test]
fn every_declared_policy_variant_is_listed() {
    assert_eq!(
        source_variants("objective/workload.rs", "pub enum WorkloadAggregation {"),
        listed(WorkloadAggregation::ALL)
    );
    assert_eq!(
        source_variants("objective/workload.rs", "pub enum RiskStatistic {"),
        listed(RiskStatistic::ALL)
    );
    assert_eq!(
        source_variants("objective/portfolio.rs", "pub enum CoveragePolicy {"),
        listed(CoveragePolicy::ALL)
    );
}

// ============================================================================
// Metric admissibility
// ============================================================================

/// WHY: ranking reads a figure per candidate. A metric whose figure only exists
/// after emission or after a whole portfolio is assembled would have to be
/// invented at ranking time, and an invented figure orders candidates by a
/// number no stage computed. Such a metric is admissible as a bound only, so
/// `is_orderable` must agree with what candidate scoring actually derives.
#[test]
fn a_metric_is_orderable_exactly_when_candidate_scoring_derives_it() {
    let selected =
        compile(&request(latency()).validate().expect("must validate")).expect("must compile");
    let cost = &selected.selected_plan().selection_cost;
    let figures = MetricFigures::derive(cost, rich_device(), WorkloadClass::single(), 1);
    for metric in ObjectiveMetric::ALL {
        let derived = figures.get(*metric).is_some();
        let needs_absent_fact = metric
            .required_fact()
            .is_some_and(|fact| fact == RequiredFact::EnergyRate);
        assert_eq!(
            derived,
            metric.is_orderable() && !needs_absent_fact,
            "Fix: ObjectiveMetric::{metric:?} claims orderable={} but candidate scoring derives \
             it={derived}; either derive the figure or state the metric as bound-only",
            metric.is_orderable()
        );
    }
}

/// WHY: a metric priced by a fact the device withheld cannot be ranked against
/// a guess. The refusal must name both the fact and the builder that supplies
/// it, because a caller who is told only "missing fact" has to read the compiler
/// to find out what to call.
#[test]
fn a_metric_whose_fact_the_device_withheld_is_refused_by_name() {
    let error = CompileRequest::new(
        joined_graph(),
        facts(),
        bare_device(),
        budget(),
        CompileObjective::maximize_throughput(64)
            .with_bound(ObjectiveMetric::ArtifactBytes, CEILING),
    )
    .validate()
    .err()
    .expect("a throughput objective needs a persistent-setup fact");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC030_MISSING_CALIBRATED_FACT"
    );
    assert!(
        error.diagnostic.message.contains("persistent_setup"),
        "the refusal must name the fact: {}",
        error.diagnostic.message
    );
    assert!(
        error
            .diagnostic
            .suggested_fix
            .as_deref()
            .is_some_and(|fix| fix.contains("DeviceFacts::with_launch_costs")),
        "the refusal must name the builder that supplies the fact: {:?}",
        error.diagnostic.suggested_fix
    );
}

/// WHY: the same objective on a device that reports the fact must compile.
/// A refusal that fires whatever the device said would make the whole metric
/// unusable rather than gated on calibration.
#[test]
fn the_same_metric_is_admitted_once_the_device_reports_its_fact() {
    let artifact = compile(
        &CompileRequest::new(
            joined_graph(),
            facts(),
            rich_device(),
            budget(),
            CompileObjective::maximize_throughput(64)
                .with_bound(ObjectiveMetric::ArtifactBytes, CEILING),
        )
        .validate()
        .expect("a throughput objective must validate against a calibrated device"),
    )
    .expect("compilation under a throughput objective must succeed");
    assert_eq!(
        artifact.provenance().objective.primary(),
        ObjectiveMetric::Throughput
    );
}

/// WHY: energy has no calibrated fact anywhere in this compiler. Ranking by it
/// would report an order no device earned, so it fails closed and names the
/// absence rather than silently pricing energy as zero.
#[test]
fn an_energy_objective_is_refused_on_every_device() {
    for device in [bare_device(), rich_device()] {
        let error = CompileRequest::new(
            joined_graph(),
            facts(),
            device,
            budget(),
            CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, CEILING)
                .with_primary(ObjectiveMetric::Energy),
        )
        .validate()
        .err()
        .expect("no device prices energy");
        assert_eq!(
            error.diagnostic.code.as_str(),
            "MKC030_MISSING_CALIBRATED_FACT"
        );
        assert!(error.diagnostic.message.contains("energy_rate"));
    }
}

/// WHY: a bound is checked where its figure is real, so bounding an unorderable
/// metric is legal. Ordering by one is not, and the two must not be conflated:
/// an artifact-bytes bound is the ceiling every production request states.
#[test]
fn an_unorderable_metric_bounds_but_never_orders() {
    assert!(request(latency()).validate().is_ok());
    let error = error_of(latency().with_primary(ObjectiveMetric::ArtifactBytes));
    assert_eq!(error.diagnostic.code.as_str(), "MKC029_INVALID_OBJECTIVE");
    assert!(
        error.diagnostic.message.contains("after emission"),
        "the refusal must say why the metric cannot order: {}",
        error.diagnostic.message
    );
    let tie_break = error_of(
        latency()
            .with_primary(ObjectiveMetric::Latency)
            .with_tie_breaker(ObjectiveMetric::VariantCount),
    );
    assert_eq!(
        tie_break.diagnostic.code.as_str(),
        "MKC029_INVALID_OBJECTIVE"
    );
}

// ============================================================================
// Validation: every inconsistent record is refused with its own path
// ============================================================================

/// WHY: each of these records states something no compile can act on. Ranking
/// under one would either divide by zero, weight a workload against a total it
/// does not sum to, or state a tie breaker that can never break a tie. Each is
/// refused as an invalid objective, and the diagnostic path names the field, so
/// a caller does not have to bisect its own request.
#[test]
fn every_inconsistent_objective_is_refused_at_its_own_field() {
    let cases: Vec<(&str, CompileObjective, &str)> = vec![
        (
            "zero amortization horizon",
            latency().with_amortization_launches(0),
            "request.objective.amortization_launches",
        ),
        (
            "a workload class that never runs",
            latency().with_workload(WorkloadProfile::of(WorkloadClass::new(0, 1, 1_000))),
            "request.objective.workload.classes[0]",
        ),
        (
            "a class with no stream",
            latency().with_workload(WorkloadProfile::of(WorkloadClass::new(1, 0, 1_000))),
            "request.objective.workload.classes[0]",
        ),
        (
            "weights that do not sum to one",
            latency().with_workload(
                WorkloadProfile::of(WorkloadClass::new(1, 1, 400))
                    .pushed(WorkloadClass::new(64, 1, 400)),
            ),
            "request.objective.workload.classes",
        ),
        (
            "a repeated tie breaker",
            latency()
                .with_tie_breaker(ObjectiveMetric::PeakMemory)
                .with_tie_breaker(ObjectiveMetric::PeakMemory),
            "request.objective.tie_breakers",
        ),
        (
            "the primary metric restated as a tie breaker",
            latency().with_tie_breaker(ObjectiveMetric::Latency),
            "request.objective.tie_breakers",
        ),
        (
            "a coverage requirement the artifact ceiling cannot reach",
            latency()
                .with_workload(
                    WorkloadProfile::of(WorkloadClass::new(1, 1, 500))
                        .pushed(WorkloadClass::new(64, 1, 500)),
                )
                .with_portfolio(PortfolioPolicy::new(CoveragePolicy::EveryWorkloadClass, 1)),
            "request.objective.portfolio.max_variants",
        ),
    ];
    for (name, objective, path) in cases {
        let error = error_of(objective);
        assert_eq!(
            error.diagnostic.code.as_str(),
            "MKC029_INVALID_OBJECTIVE",
            "{name} must be refused as an invalid objective"
        );
        assert_eq!(
            refused_field(&error),
            Some(path),
            "{name} must name its own field"
        );
    }
}

/// WHY: a worst-case profile ignores weights by construction, so weights that
/// do not sum to one thousand are not an inconsistency under it. Refusing them
/// anyway would force a caller to state weights it stated the compiler must not
/// read.
#[test]
fn worst_case_aggregation_does_not_require_normalized_weights() {
    let objective = latency().with_workload(
        WorkloadProfile::of(WorkloadClass::new(1, 1, 1))
            .pushed(WorkloadClass::new(64, 1, 1))
            .with_aggregation(WorkloadAggregation::WorstCase),
    );
    request(objective)
        .validate()
        .expect("worst-case aggregation reads no weight");
}

/// WHY: an objective built under a different schema states fields that mean
/// something else. Comparing against it would compare two different records, so
/// the skew is rejected rather than reinterpreted.
#[test]
fn an_objective_from_another_schema_is_rejected() {
    let current = latency();
    assert_eq!(current.version(), OBJECTIVE_SCHEMA_VERSION);
    let stale: CompileObjective = serde_json::from_str(
        &serde_json::to_string(&current)
            .expect("the objective must serialize")
            .replace(
                &format!("\"version\":{OBJECTIVE_SCHEMA_VERSION}"),
                "\"version\":0",
            ),
    )
    .expect("the projection must decode");
    assert_eq!(error_code(stale), "MKC029_INVALID_OBJECTIVE");
}

// ============================================================================
// Bounds
// ============================================================================

/// WHY: a bound is a refusal, not a preference. A compile whose whole legal
/// candidate set exceeds one must fail with the bound and the achieved figure
/// named, because "no plan" and "no plan under your limit" call for different
/// corrections.
#[test]
fn a_bound_the_whole_legal_set_exceeds_fails_with_the_figure_it_reached() {
    let error = compile(
        &request(latency().with_bound(ObjectiveMetric::Latency, 1))
            .validate()
            .expect("a tight bound is a legal objective"),
    )
    .expect_err("no plan can run in one nanosecond");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC031_OBJECTIVE_BOUND_VIOLATED"
    );
    assert!(
        error
            .diagnostic
            .message
            .contains("latency bound is 1 nanoseconds"),
        "the refusal must name the bound: {}",
        error.diagnostic.message
    );
    assert_eq!(refused_field(&error), Some("request.objective.bounds"));
}

/// WHY: the same graph under a bound its plans satisfy must compile. A bound
/// check that rejected every workload would be indistinguishable from one that
/// never ran.
#[test]
fn a_bound_the_selected_plan_satisfies_does_not_refuse_it() {
    let unbounded = compile(
        &request(latency())
            .validate()
            .expect("the plain objective must validate"),
    )
    .expect("compilation must succeed");
    let achieved = unbounded.selected_plan().selection_cost.total;
    let bounded = compile(
        &request(latency().with_bound(ObjectiveMetric::Latency, achieved))
            .validate()
            .expect("a satisfiable bound must validate"),
    )
    .expect("a bound the plan meets must not refuse it");
    assert_eq!(
        bounded.selected_plan().selection_cost.total,
        achieved,
        "a satisfied bound must not change which plan is selected"
    );
}

/// WHY: bounds are read positionally and reported in metric declaration order,
/// so a caller reading two violated bounds always hears about the same one. A
/// metric with no figure yet is not checked here, because the stage that owns
/// the figure checks it where the figure is real; checking it as zero would
/// admit an artifact nothing measured.
#[test]
fn bounds_report_the_first_violated_metric_and_skip_absent_figures() {
    let bounds = ObjectiveBounds::unbounded()
        .with_bound(ObjectiveMetric::Latency, 10)
        .with_bound(ObjectiveMetric::PeakMemory, 10)
        .with_bound(ObjectiveMetric::ArtifactBytes, 10);
    let figures = MetricFigures::empty()
        .with(ObjectiveMetric::Latency, 11)
        .with(ObjectiveMetric::PeakMemory, 99);
    let violation = bounds
        .first_violation(figures.as_array())
        .expect("a figure over its limit must be reported");
    assert_eq!(violation.metric, ObjectiveMetric::Latency);
    assert_eq!((violation.limit, violation.achieved), (10, 11));
    assert!(violation.statement().contains("11 nanoseconds"));

    let under = MetricFigures::empty().with(ObjectiveMetric::Latency, 10);
    assert!(
        bounds.first_violation(under.as_array()).is_none(),
        "a figure at its limit is inside it, and an absent artifact-bytes figure is not checked \
         at ranking time"
    );
    assert!(ObjectiveBounds::unbounded().is_unbounded());
    assert_eq!(bounds.stated().len(), 3);
}

// ============================================================================
// Workload aggregation and risk
// ============================================================================

/// WHY: a weighted aggregate is the figure the objective orders by, so it must
/// be reproducible on every host reading the same artifact and must never report
/// a figure below every class it combines. Integral permille weights and a
/// rounding-up division are what give both.
#[test]
fn a_weighted_aggregate_rounds_up_and_a_worst_case_aggregate_ignores_weights() {
    let profile =
        WorkloadProfile::of(WorkloadClass::new(1, 1, 750)).pushed(WorkloadClass::new(64, 1, 250));
    assert_eq!(profile.aggregate(&[100, 200]), 125);
    assert_eq!(profile.aggregate(&[1, 2]), 2, "a weighted mean rounds up");
    let worst = profile.with_aggregation(WorkloadAggregation::WorstCase);
    assert_eq!(worst.aggregate(&[100, 200]), 200);
    assert_eq!(
        worst.aggregate(&[200, 100]),
        200,
        "a worst case reads no weight"
    );
    assert!(WorkloadAggregation::Weighted.reads_weights());
    assert!(!WorkloadAggregation::WorstCase.reads_weights());
}

/// WHY: `CompileObjective::aggregate` is what ranking reads. A metric one class
/// reported and another did not has no aggregate, and reporting zero for it
/// would rank a candidate as free on a metric nothing priced.
#[test]
fn an_aggregate_metric_stays_absent_unless_every_class_reported_it() {
    let objective = latency().with_workload(
        WorkloadProfile::of(WorkloadClass::new(1, 1, 500)).pushed(WorkloadClass::new(1, 1, 500)),
    );
    let both = objective.aggregate(&[
        MetricFigures::empty().with(ObjectiveMetric::Latency, 100),
        MetricFigures::empty().with(ObjectiveMetric::Latency, 300),
    ]);
    assert_eq!(both.get(ObjectiveMetric::Latency), Some(200));
    let partial = objective.aggregate(&[
        MetricFigures::empty().with(ObjectiveMetric::Latency, 100),
        MetricFigures::empty().with(ObjectiveMetric::PeakMemory, 8),
    ]);
    assert_eq!(partial.get(ObjectiveMetric::Latency), None);
    assert_eq!(partial.get(ObjectiveMetric::PeakMemory), None);
}

/// WHY: two compiles that read different statistics of identical samples select
/// different winners, so the statistic is part of the objective and must read
/// the retained set by nearest rank. An interpolated percentile would report a
/// duration no launch took.
#[test]
fn every_risk_statistic_reads_a_sample_the_device_actually_produced() {
    let ordered = [10_u64, 20, 30, 40, 50];
    let trimmed = 27;
    for statistic in RiskStatistic::ALL {
        let read = statistic.read(&ordered, trimmed);
        match statistic.permille_rank() {
            None => assert_eq!(read, trimmed, "{}", statistic.name()),
            Some(_) => assert!(
                ordered.contains(&read),
                "RiskStatistic::{statistic:?} reported {read}, which no launch took"
            ),
        }
    }
    assert_eq!(RiskStatistic::Median.read(&ordered, trimmed), 30);
    assert_eq!(RiskStatistic::WorstCase.read(&ordered, trimmed), 50);
    assert_eq!(RiskStatistic::P95.read(&ordered, trimmed), 50);
    assert_eq!(
        RiskStatistic::Median.read(&[], trimmed),
        trimmed,
        "an empty retained set falls back to the estimate the protocol computed"
    );
}

/// WHY: search prunes candidates against the figure a plan cannot get below, so
/// that figure must never exceed the derived one. A lower bound above the real
/// figure prunes the winner.
#[test]
fn an_unavoidable_figure_never_exceeds_the_derived_figure() {
    let selected =
        compile(&request(latency()).validate().expect("must validate")).expect("must compile");
    let cost = &selected.selected_plan().selection_cost;
    for class in [
        WorkloadClass::single(),
        WorkloadClass::new(64, 1, 1_000),
        WorkloadClass::new(1, 8, 1_000),
        WorkloadClass::new(16, 4, 1_000),
    ] {
        let derived = MetricFigures::derive(cost, rich_device(), class, 8);
        let floor = MetricFigures::unavoidable(cost.launch_ns, class);
        for metric in [ObjectiveMetric::Latency, ObjectiveMetric::Throughput] {
            let (Some(floor), Some(derived)) = (floor.get(metric), derived.get(metric)) else {
                panic!("both figures must exist for {metric:?}");
            };
            assert!(
                floor <= derived,
                "the {metric:?} lower bound {floor} exceeds the derived figure {derived}, so \
                 search would prune the plan it selected"
            );
        }
    }
}

/// WHY: a fixed-capacity sequence keeps the objective `Copy`, which is what lets
/// request identity hash it by value. A full sequence must report itself instead
/// of dropping a key a caller stated.
#[test]
fn a_full_metric_sequence_reports_itself_rather_than_dropping_a_key() {
    let mut sequence = MetricSequence::empty();
    assert!(sequence.is_empty());
    for metric in ObjectiveMetric::ALL.iter().take(MetricSequence::CAPACITY) {
        sequence = sequence.pushed(*metric);
    }
    assert!(sequence.is_full());
    assert_eq!(sequence.len(), MetricSequence::CAPACITY);
    let overflowed = sequence.pushed(ObjectiveMetric::CompileWork);
    assert_eq!(overflowed, sequence);
    assert!(!overflowed.contains(ObjectiveMetric::CompileWork));
}

// ============================================================================
// Identity: no objective can reuse another's decision
// ============================================================================

/// Objectives that differ in exactly one field, each stated so that it is legal
/// on `rich_device()` and satisfiable by the fixture graph.
fn distinct_objectives() -> Vec<(&'static str, CompileObjective)> {
    vec![
        ("baseline", latency()),
        (
            "primary",
            latency().with_primary(ObjectiveMetric::PeakMemory),
        ),
        (
            "tie_breakers",
            latency().with_tie_breaker(ObjectiveMetric::PeakMemory),
        ),
        (
            "workload",
            latency().with_workload(WorkloadProfile::of(WorkloadClass::new(64, 2, 1_000))),
        ),
        ("risk", latency().with_risk(RiskStatistic::P99)),
        (
            "amortization_launches",
            latency().with_amortization_launches(4_096),
        ),
        (
            "bounds",
            latency().with_bound(ObjectiveMetric::PeakMemory, u64::MAX),
        ),
        (
            "artifact_bytes_bound",
            CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, CEILING - 1),
        ),
        (
            "portfolio",
            latency().with_portfolio(PortfolioPolicy::new(CoveragePolicy::Single, 8)),
        ),
        (
            "portfolio_aggregate_bytes",
            latency().with_portfolio(
                PortfolioPolicy::single().with_max_aggregate_bytes(16 * 1024 * 1024),
            ),
        ),
    ]
}

/// WHY: an artifact cached under one objective must never be served to a request
/// that stated another, or a latency artifact answers a throughput compile. The
/// request digest is what a cache keys on, so every objective field has to
/// change it. A field left out of identity is exactly the defect this proves
/// absent: mutate any one field, and the previous decision becomes unreachable.
#[test]
fn changing_any_objective_field_changes_the_request_identity() {
    let mut seen: Vec<(&str, vyre_megakernel::Digest)> = Vec::new();
    for (field, objective) in distinct_objectives() {
        let artifact = compile(
            &request(objective)
                .validate()
                .unwrap_or_else(|err| panic!("the {field} objective must validate: {err}")),
        )
        .unwrap_or_else(|err| panic!("the {field} objective must compile: {err}"));
        let digest = artifact.provenance().request;
        if let Some((other, _)) = seen.iter().find(|(_, other)| *other == digest) {
            panic!(
                "Fix: the {field} objective shares a request identity with {other}, so a cache \
                 would serve one compile's artifact to the other"
            );
        }
        seen.push((field, digest));
    }
    assert_eq!(seen.len(), distinct_objectives().len());
}

/// WHY: identity must also be stable. A digest that changed between two
/// identical requests would make every cache miss and every evidence comparison
/// meaningless, which is the failure mode opposite to the one above.
#[test]
fn the_same_objective_reproduces_one_request_identity() {
    let first = compile(&request(latency()).validate().expect("must validate"))
        .expect("must compile")
        .provenance()
        .request;
    let second = compile(&request(latency()).validate().expect("must validate"))
        .expect("must compile")
        .provenance()
        .request;
    assert_eq!(first, second);
}

/// WHY: the artifact states what its plan was selected to optimize, so a reader
/// holding only bytes can tell a latency artifact from a throughput one without
/// the request that produced it.
#[test]
fn the_artifact_records_the_objective_it_was_selected_under() {
    let objective = latency()
        .with_primary(ObjectiveMetric::PeakMemory)
        .with_tie_breaker(ObjectiveMetric::Latency)
        .with_risk(RiskStatistic::P95);
    let artifact = compile(&request(objective).validate().expect("must validate"))
        .expect("must compile")
        .to_bytes()
        .expect("must serialize");
    let decoded = vyre_megakernel::Artifact::from_bytes(&artifact).expect("must decode");
    assert_eq!(decoded.provenance().objective, objective);
    assert_eq!(
        decoded.provenance().objective.primary(),
        ObjectiveMetric::PeakMemory
    );
}

// ============================================================================
// The frontier the ranking preserved
// ============================================================================

/// WHY: an objective that orders by one metric hides how much choice the legal
/// set had. The recorded frontier is what tells a reader whether the tie
/// breakers decided the winner or whether one plan dominated every other, and it
/// always contains the selected plan, so it is never zero and never exceeds the
/// candidates explored.
#[test]
fn the_selected_plan_records_the_frontier_it_was_chosen_from() {
    for objective in [
        latency(),
        latency().with_primary(ObjectiveMetric::PeakMemory),
        latency()
            .with_primary(ObjectiveMetric::Latency)
            .with_tie_breaker(ObjectiveMetric::PeakMemory),
    ] {
        let artifact =
            compile(&request(objective).validate().expect("must validate")).expect("must compile");
        let plan = artifact.selected_plan();
        assert!(
            plan.pareto_frontier >= 1,
            "the selected plan is on its own frontier"
        );
        assert!(
            plan.pareto_frontier <= plan.candidates_explored,
            "the frontier cannot hold more plans than the search explored: {} of {}",
            plan.pareto_frontier,
            plan.candidates_explored
        );
    }
}

/// WHY: a single-stage graph has one legal plan, so its frontier is exactly one.
/// A frontier that counted every candidate, or that counted none, would pass the
/// range check above while reporting nothing.
#[test]
fn a_graph_with_one_legal_plan_records_a_frontier_of_one() {
    let artifact = compile(
        &CompileRequest::new(
            single_stage_graph(),
            facts(),
            rich_device(),
            budget(),
            latency(),
        )
        .validate()
        .expect("must validate"),
    )
    .expect("must compile");
    assert_eq!(artifact.selected_plan().pareto_frontier, 1);
}

/// WHY: the frontier is part of the persisted plan, so a reader holding only
/// bytes can tell how much choice the legal set had. A frontier the artifact
/// dropped on the way to disk reports nothing to that reader, and the decode
/// path's own mutation contracts prove a persisted frontier the search cannot
/// support is rejected.
#[test]
fn the_frontier_survives_a_round_trip_through_bytes() {
    let artifact =
        compile(&request(latency()).validate().expect("must validate")).expect("must compile");
    let bytes = artifact.to_bytes().expect("must serialize");
    let decoded = vyre_megakernel::Artifact::from_bytes(&bytes).expect("must decode");
    assert_eq!(
        decoded.selected_plan().pareto_frontier,
        artifact.selected_plan().pareto_frontier
    );
}

/// WHY: a device that reports fewer facts prices fewer plans apart, and the
/// objective must still select one. A selector that only worked on the rich
/// fixture device would fail on every real target that withheld a fact.
#[test]
fn a_bare_device_still_selects_under_a_stated_objective() {
    let artifact = compile(
        &CompileRequest::new(
            joined_graph(),
            facts(),
            bare_device(),
            budget(),
            latency().with_primary(ObjectiveMetric::PeakMemory),
        )
        .validate()
        .expect("peak memory needs no calibrated fact"),
    )
    .expect("must compile");
    assert_eq!(
        artifact.provenance().objective.primary(),
        ObjectiveMetric::PeakMemory
    );
    assert!(artifact.selected_plan().pareto_frontier >= 1);
}

/// WHY: a metric states the unit its bound is read in. A bound stated in bytes
/// against a nanosecond figure is a caller error the diagnostic must expose, so
/// every metric names a unit and no two ordering metrics share a slot.
#[test]
fn every_metric_states_a_name_and_a_unit() {
    let mut names = BTreeSet::new();
    for metric in ObjectiveMetric::ALL {
        assert!(!metric.unit().is_empty(), "{metric:?} states no unit");
        assert!(
            names.insert(metric.name()),
            "Fix: `{}` is the stable name of two metrics; a diagnostic naming it would be \
             ambiguous",
            metric.name()
        );
    }
    assert_eq!(names.len(), ObjectiveMetric::ALL.len());
}

/// WHY: the ordering metric list is what ranking reads, primary first. A
/// duplicate in it would compare one metric twice and never reach the next tie
/// breaker.
#[test]
fn the_ordering_metric_list_states_the_primary_first_and_each_metric_once() {
    let objective = latency()
        .with_primary(ObjectiveMetric::Latency)
        .with_tie_breaker(ObjectiveMetric::PeakMemory)
        .with_tie_breaker(ObjectiveMetric::ColdStart);
    assert_eq!(
        objective.ordering_metrics(),
        vec![
            ObjectiveMetric::Latency,
            ObjectiveMetric::PeakMemory,
            ObjectiveMetric::ColdStart
        ]
    );
    assert_eq!(objective.tie_breakers().len(), 2);
}

/// WHY: a portfolio ceiling that a coverage policy cannot reach is rejected at
/// validation, so `admits` and `minimum_variants` must agree with that check.
/// Two answers to "is this retained set legal" is how a compile retains a set
/// its own objective refuses.
#[test]
fn a_portfolio_admits_exactly_the_sets_its_coverage_and_ceiling_allow() {
    let single = PortfolioPolicy::single();
    assert!(single.admits(1, 1));
    assert!(!single.admits(2, 1), "the ceiling is one artifact");
    assert_eq!(CoveragePolicy::Single.minimum_variants(4), 1);
    assert_eq!(CoveragePolicy::EveryWorkloadClass.minimum_variants(4), 4);
    let every = PortfolioPolicy::new(CoveragePolicy::EveryWorkloadClass, 4);
    assert!(!every.admits(3, 4), "one class would be served by nothing");
    assert!(every.admits(4, 4));
    assert_eq!(single.max_aggregate_bytes(), None);
    assert_eq!(
        single.with_max_aggregate_bytes(1024).max_aggregate_bytes(),
        Some(1024)
    );
}

/// WHY: the artifact-bytes bound is the ceiling every production compile states,
/// and it is the only place that ceiling now lives. A request that states none
/// cannot be checked against anything, so it is refused rather than compiled
/// against an implicit limit.
#[test]
fn a_production_request_must_state_an_artifact_byte_ceiling() {
    for objective in [
        CompileObjective::minimize_latency(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 0),
    ] {
        let error = error_of(objective);
        assert_eq!(error.diagnostic.code.as_str(), "MKC013_ARTIFACT_LIMIT");
        assert_eq!(
            refused_field(&error),
            Some("request.objective.bounds.artifact_bytes"),
            "the refusal must name the bound that is missing"
        );
    }
}

/// WHY: the ceiling is enforced where the figure is real, on the serialized
/// artifact, and the failure keeps its own code so a caller can tell a
/// too-large artifact from a plan that violated a ranking bound.
///
/// The artifact states the bound it was compiled under, so the boundary is the
/// fixed point where the stated bound equals the resulting length.
#[test]
fn an_artifact_over_the_stated_byte_ceiling_fails_as_an_artifact_limit() {
    let bounded = |bytes: u64| {
        request(
            CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, bytes),
        )
        .validate()
        .expect("must validate")
    };
    let mut limit = 1_000_000;
    let mut exact = None;
    for _ in 0..8 {
        let len = compile(&bounded(limit))
            .expect("must compile")
            .to_bytes()
            .expect("must serialize")
            .len() as u64;
        if len == limit {
            exact = Some(limit);
            break;
        }
        limit = len;
    }
    let exact = exact.expect("the artifact length must settle on the bound it states");

    let error = compile(&bounded(exact - 1)).expect_err("one byte short must fail");
    assert_eq!(error.diagnostic.code.as_str(), "MKC013_ARTIFACT_LIMIT");
}
