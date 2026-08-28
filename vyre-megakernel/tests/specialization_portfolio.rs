//! A specialized set is compiled jointly, sealed whole, and selected from.
//!
//! WHY: three failures used to be reachable once a compile produced more than
//! one artifact for one graph. A caller could retain every proposed variant
//! whatever it cost, because nothing scored the set against the domain it
//! serves. A consumer could be handed some members and a guard table from
//! another compile, because the members were separate products. And a consumer
//! holding the right bytes for the wrong device could run them, because nothing
//! in the bytes stated which device they were compiled for.
//!
//! Each is proved here rather than documented: the variant ceiling is honoured
//! with the unspecialized baseline still in the candidate set, a member from
//! another compile is refused at seal, and an authenticated target that is not
//! the compiled-for target is refused before anything is selected.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::specialization::{
    compile_specialized_portfolio, AxisDomain, AxisValue, GuardTerm, PortfolioEnvelope,
    PortfolioVariant, RemainderKind, SpecializationAxis, SpecializationContract,
    SpecializedPortfolio, SpecializedRemainder, VariantGuard, MAX_PROPOSED_VARIANTS,
    PORTFOLIO_ENVELOPE_SCHEMA_VERSION, SPECIALIZATION_SCHEMA_VERSION,
};
use vyre_megakernel::{
    target_identity, Artifact, ArtifactEnvelope, CompileObjective, CompileRequest, CoveragePolicy,
    DeviceFacts, Digest, ExternalFacts, ObjectiveMetric, PortfolioPolicy, SearchBudget,
    ValidatedCompileRequest,
};

/// The axis every fixture guard reads.
fn tokens() -> SpecializationAxis {
    SpecializationAxis::SymbolicDimension {
        dimension: "tokens".to_string(),
    }
}

fn objective(max_variants: u32) -> CompileObjective {
    CompileObjective::minimize_latency()
        .with_bound(ObjectiveMetric::ArtifactBytes, 4_000_000)
        .with_portfolio(PortfolioPolicy::new(CoveragePolicy::Single, max_variants))
}

/// Live values one fixture node holds.
const LIVE_VALUES: u32 = 40;

/// Registers per invocation the fixture device grants at full occupancy.
const REGISTER_BUDGET: u32 = 32;

/// A device whose reported facts price the extent a variant was compiled for.
///
/// Every other term of the selection cost model is extent-independent over a
/// one-node graph: the launch count, the instruction count, and the rendezvous
/// count are the same whether the graph runs over 64 elements or 1024. What
/// moves with the extent is the traffic a group replays when it wants more
/// registers than the device grants, and that term is priced only against a
/// reported bandwidth. A device reporting neither ranks a 64-element variant
/// level with the 1024-element baseline, which is why it would be retained on a
/// tie-break rather than on its cost.
fn device() -> DeviceFacts {
    DeviceFacts::unknown()
        .with_occupancy(REGISTER_BUDGET, 65_536)
        .with_bandwidth_facts(1, 0)
}

/// A one-node graph over one symbolic dimension.
fn request(max_variants: u32, seed: u8) -> ValidatedCompileRequest {
    let value = |access, lifetime| ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("tokens".into())],
        access,
        lifetime,
    };
    let mut graph = ProgramGraph::new();
    let source = graph
        .add_external_value(
            "source",
            value(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: the fixture source value must be declarable.");
    let index = Expr::var("index");
    let mut statements = vec![Node::let_bind("index", Expr::LogicalIndex { axis: 0 })];
    let mut sum = Expr::load("source_in", index.clone());
    for step in 1..LIVE_VALUES {
        let name = format!("live{step}");
        statements.push(Node::let_bind(
            name.as_str(),
            Expr::add(Expr::load("source_in", index.clone()), Expr::u32(step)),
        ));
        sum = Expr::add(sum, Expr::var(name.as_str()));
    }
    statements.push(Node::store("sink_out", index, sum));
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("source_in", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("sink_out", 1, BufferAccess::WriteOnly, DataType::U32),
        ],
        [64, 1, 1],
        statements,
    );
    graph
        .add_node(
            "increment",
            program,
            vec![GraphInput {
                buffer: "source_in".into(),
                value: source,
                contract: value(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "sink_out".into(),
                name: "sink".into(),
                contract: value(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("Fix: the fixture node must be declarable.");
    let mut bindings = BTreeMap::new();
    bindings.insert("tokens".to_string(), 1024);
    CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([seed; 32]), bindings),
        device(),
        SearchBudget::new(32, 100_000, 4, 0, 1_000_000_000),
        objective(max_variants),
    )
    .validate()
    .expect("Fix: the fixture request must validate.")
}

fn contract() -> SpecializationContract {
    let mut axes = BTreeMap::new();
    axes.insert(tokens(), AxisDomain::Interval { low: 1, high: 1024 });
    SpecializationContract::new(axes).expect("Fix: the fixture contract must be declarable.")
}

fn in_range(low: u64, high: u64, precedence: u16) -> VariantGuard {
    VariantGuard::new(
        vec![GuardTerm::InRange {
            axis: tokens(),
            low,
            high,
        }],
        precedence,
    )
}

fn facts(tokens_extent: u64) -> BTreeMap<SpecializationAxis, AxisValue> {
    BTreeMap::from([(tokens(), AxisValue::Scalar(tokens_extent))])
}

/// The same graph and facts under another objective.
fn restated(
    stated: ValidatedCompileRequest,
    objective: CompileObjective,
) -> ValidatedCompileRequest {
    CompileRequest::new(
        stated.graph().clone(),
        stated.facts().clone(),
        stated.device(),
        SearchBudget::new(32, 100_000, 4, 0, 1_000_000_000),
        objective,
    )
    .validate()
    .expect("Fix: the restated request must validate.")
}

/// Seal a compiled set as one authenticated product for one target.
fn envelope(portfolio: &SpecializedPortfolio, identity: Digest) -> PortfolioEnvelope {
    let mut sealed = PortfolioEnvelope::new(
        portfolio.contract().clone(),
        portfolio.remainder().kind(),
        identity,
    );
    for variant in portfolio.variants() {
        sealed
            .attach_variant(
                variant.guard().clone(),
                ArtifactEnvelope::new(variant.artifact().clone()),
            )
            .expect("Fix: a retained variant must attach under its own guard.");
    }
    if let Some(artifact) = portfolio.remainder().artifact() {
        sealed
            .attach_remainder(ArtifactEnvelope::new(artifact.clone()))
            .expect("Fix: a generic remainder must attach.");
    }
    sealed
}

fn compiled_identity(portfolio: &SpecializedPortfolio) -> Digest {
    target_identity(device(), portfolio.objective())
}

#[test]
fn an_aggregate_ceiling_only_the_baseline_fits_retains_the_baseline() {
    let measured = vyre_megakernel::compile(&request(4, 0))
        .expect("Fix: the fixture request must compile.")
        .to_bytes()
        .expect("Fix: a compiled artifact must serialize.")
        .len() as u64;
    let tight = CompileObjective::minimize_latency()
        .with_bound(ObjectiveMetric::ArtifactBytes, 4_000_000)
        .with_portfolio(
            PortfolioPolicy::new(CoveragePolicy::Single, 4)
                .with_max_aggregate_bytes(measured + measured / 2),
        );
    let stated = restated(request(4, 0), tight);

    let portfolio = compile_specialized_portfolio(
        &stated,
        &contract(),
        &[in_range(1, 256, 0), in_range(257, 1024, 0)],
        RemainderKind::Generic,
    )
    .expect("Fix: a set that retains no variant is still a legal set.");

    assert!(
        portfolio.variants().is_empty(),
        "Fix: a ceiling only the unspecialized artifact fits under must retain no variant, so the \
         baseline is what the objective ordered rather than what was left over."
    );
    assert!(
        matches!(portfolio.remainder(), SpecializedRemainder::Generic { .. }),
        "Fix: the baseline must stay in the candidate set and be retained when no variant is."
    );
    assert_eq!(
        portfolio.proof().covered(),
        0,
        "Fix: with no retained guard the proof must charge every cell to the remainder."
    );
    assert!(
        portfolio.aggregate_bytes() <= measured + measured / 2,
        "Fix: the retained set must fit under the ceiling it was selected against."
    );
}

/// A variant is retained for the cells it makes cheaper, and only those.
///
/// Both proposals cover part of the declared domain, but only the narrow one
/// changes what the graph costs: it is compiled at a 64-element extent, so the
/// traffic its group replays is a sixteenth of the baseline's. The wide guard is
/// compiled at the same 1024-element extent as the generic remainder, so it
/// prices identically and retaining it would spend an artifact to serve cells
/// the remainder already serves at that price.
#[test]
fn a_guarded_extent_is_served_by_a_variant_compiled_for_it() {
    let stated = request(4, 1);
    let portfolio = compile_specialized_portfolio(
        &stated,
        &contract(),
        &[in_range(1, 64, 0), in_range(65, 1024, 0)],
        RemainderKind::Generic,
    )
    .expect("Fix: two disjoint guards over the declared domain must compile.");

    let retained: Vec<&VariantGuard> = portfolio
        .variants()
        .iter()
        .map(PortfolioVariant::guard)
        .collect();
    assert_eq!(
        retained,
        vec![&in_range(1, 64, 0)],
        "Fix: retain the guard whose variant is cheaper than the remainder over its own cells, \
         and no other: a guard compiled at the remainder's extent costs bytes and buys nothing."
    );
    assert!(
        portfolio.proof().gaps() > 0,
        "Fix: the cells no retained guard covers must be charged to the remainder, or the set \
         claims a coverage it did not prove."
    );

    let narrow = portfolio
        .select(&facts(8))
        .expect("Fix: a stated extent inside a guard must select that guard's variant.");
    let wide = portfolio
        .select(&facts(900))
        .expect("Fix: a stated extent no guard admits must fall to the generic remainder.");
    assert_eq!(
        narrow.digest(),
        portfolio.variants()[0].artifact().digest(),
        "Fix: selection must return the artifact the retained guard was compiled for."
    );
    assert_eq!(
        Some(wide.digest()),
        portfolio.remainder().artifact().map(Artifact::digest),
        "Fix: an extent outside every retained guard must be served by the remainder itself."
    );
    assert_ne!(
        narrow.digest(),
        wide.digest(),
        "Fix: a variant compiled at the largest extent its guard admits must differ from the \
         artifact compiled for the whole domain, or the guard bought nothing."
    );
    assert_eq!(
        portfolio.proof().served().iter().sum::<usize>(),
        portfolio.proof().covered(),
        "Fix: every covered cell must be charged to exactly one guard, so a guard that serves \
         nothing is visible."
    );
}

/// A workload outside the declared domain is refused whatever the remainder is.
///
/// The remainder is compiled over the declared domain, so answering a
/// 4096-element workload with it would run a 1024-element schedule against 4096
/// elements and drop the tail. The refusal names the axis, and it holds for
/// every declared remainder kind: carrying a generic member is not a licence to
/// serve a workload nothing proved it covers.
#[test]
fn a_workload_outside_the_declared_domain_is_refused_for_every_remainder_kind() {
    for kind in RemainderKind::ALL {
        let portfolio = compile_specialized_portfolio(
            &request(4, 2),
            &contract(),
            &[in_range(1, 1024, 0)],
            *kind,
        )
        .unwrap_or_else(|error| {
            panic!(
                "Fix: a guard covering the declared domain must compile under a {} remainder: \
                 {error}",
                kind.name()
            )
        });

        let Err(error) = portfolio.select(&facts(4096)) else {
            panic!(
                "Fix: an extent the declared domain does not hold must be refused under a {} \
                 remainder, not served by an artifact compiled for a smaller extent.",
                kind.name()
            )
        };
        assert_eq!(
            error.diagnostic.code.as_str(),
            "MKC039_UNSUPPORTED_WORKLOAD",
            "Fix: a workload no member covers must be refused as unsupported: {error}"
        );
        assert_eq!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|at| at.path.clone()),
            Some(tokens().field()),
            "Fix: the refusal must name the axis whose declared domain does not hold the value."
        );
    }
}

#[test]
fn more_proposals_than_the_search_bound_are_refused() {
    let stated = request(4, 3);
    let proposals: Vec<VariantGuard> = (0..=MAX_PROPOSED_VARIANTS)
        .map(|index| {
            let low = 1 + index as u64;
            in_range(low, low, 0)
        })
        .collect();
    let error =
        compile_specialized_portfolio(&stated, &contract(), &proposals, RemainderKind::Generic)
            .expect_err("Fix: an exponential subset search must be refused rather than attempted.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC034_INVALID_VARIANT_GUARD",
        "Fix: the proposal bound must be reported against the proposals: {error}"
    );
}

#[test]
fn an_aggregate_byte_ceiling_the_set_cannot_meet_is_refused() {
    let tight = CompileObjective::minimize_latency()
        .with_bound(ObjectiveMetric::ArtifactBytes, 4_000_000)
        .with_portfolio(
            PortfolioPolicy::new(CoveragePolicy::Single, 4).with_max_aggregate_bytes(8),
        );
    let stated = restated(request(4, 4), tight);

    let error = compile_specialized_portfolio(
        &stated,
        &contract(),
        &[in_range(1, 1024, 0)],
        RemainderKind::Generic,
    )
    .expect_err("Fix: a set no subset can fit inside must be refused, not truncated.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC031_OBJECTIVE_BOUND_VIOLATED",
        "Fix: an aggregate ceiling nothing meets must be reported as the bound it violated: \
         {error}"
    );
}

#[test]
fn a_sealed_set_round_trips_as_one_product() {
    let stated = request(4, 5);
    let portfolio = compile_specialized_portfolio(
        &stated,
        &contract(),
        &[in_range(1, 64, 0), in_range(65, 1024, 0)],
        RemainderKind::Generic,
    )
    .expect("Fix: the fixture set must compile.");
    let identity = compiled_identity(&portfolio);
    let sealed = envelope(&portfolio, identity);

    let bytes = sealed
        .to_bytes()
        .expect("Fix: a sealed set must encode as one product.");
    let decoded = PortfolioEnvelope::from_bytes(&bytes)
        .expect("Fix: canonical set bytes must decode and re-prove.");

    assert_eq!(
        decoded, sealed,
        "Fix: decoding must reconstruct the whole set, or a consumer reads a different set than \
         the compiler sealed."
    );
    assert_eq!(
        decoded.evaluation_order().len(),
        sealed.variants().len(),
        "Fix: every attached variant must appear exactly once in the evaluation order."
    );
}

#[test]
fn a_set_compiled_for_another_target_is_refused() {
    let stated = request(4, 6);
    let portfolio = compile_specialized_portfolio(
        &stated,
        &contract(),
        &[in_range(1, 1024, 0)],
        RemainderKind::Generic,
    )
    .expect("Fix: the fixture set must compile.");
    let sealed = envelope(&portfolio, compiled_identity(&portfolio));

    sealed
        .require_target_identity(compiled_identity(&portfolio))
        .expect("Fix: the target the set was compiled for must be accepted.");
    let error = sealed
        .require_target_identity(Digest([9; 32]))
        .expect_err("Fix: a target the set was not compiled for must be refused.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC038_TARGET_IDENTITY_MISMATCH",
        "Fix: a spoofed or stale target must be named as a target mismatch: {error}"
    );
}

#[test]
fn a_member_from_another_compile_does_not_seal() {
    let mut sealed = PortfolioEnvelope::new(
        contract(),
        RemainderKind::Generic,
        target_identity(DeviceFacts::unknown(), &objective(4)),
    );
    sealed
        .attach_variant(
            in_range(1, 1024, 0),
            ArtifactEnvelope::new(baseline_artifact(7)),
        )
        .expect("Fix: a variant valid under the contract must attach.");
    sealed
        .attach_remainder(ArtifactEnvelope::new(foreign_artifact()))
        .expect("Fix: attachment must accept the bytes and seal must be what rejects them.");

    let error = sealed
        .seal()
        .expect_err("Fix: a member produced by another compile must not seal into this set.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC037_PORTFOLIO_PROVENANCE_MISMATCH",
        "Fix: two members disagreeing on the graph they came from must be reported as a \
         provenance mismatch: {error}"
    );
}

#[test]
fn a_set_that_declares_a_remainder_and_carries_none_does_not_seal() {
    let mut sealed = PortfolioEnvelope::new(
        contract(),
        RemainderKind::Generic,
        target_identity(DeviceFacts::unknown(), &objective(4)),
    );
    sealed
        .attach_variant(
            in_range(1, 1024, 0),
            ArtifactEnvelope::new(baseline_artifact(9)),
        )
        .expect("Fix: a variant valid under the contract must attach.");

    let error = sealed.seal().expect_err(
        "Fix: a set declaring a generic remainder and carrying none must not seal, or a consumer \
         finds nothing to serve an uncovered request with.",
    );
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC014_MALFORMED_ARTIFACT",
        "Fix: a set missing its own declared remainder is malformed: {error}"
    );
}

#[test]
fn a_set_stating_another_specialization_schema_is_refused() {
    let mut sealed = PortfolioEnvelope::new(
        contract(),
        RemainderKind::Generic,
        target_identity(DeviceFacts::unknown(), &objective(4)),
    );
    sealed
        .attach_variant(
            in_range(1, 1024, 0),
            ArtifactEnvelope::new(baseline_artifact(11)),
        )
        .expect("Fix: a variant valid under the contract must attach.");
    sealed
        .attach_remainder(ArtifactEnvelope::new(baseline_artifact(12)))
        .expect("Fix: the generic remainder must attach.");
    let bytes = sealed
        .to_bytes()
        .expect("Fix: the fixture set must encode.");

    for (label, from, to, path) in [
        (
            "contract",
            format!("{{\"schema_version\":{SPECIALIZATION_SCHEMA_VERSION},\"axes\""),
            format!(
                "{{\"schema_version\":{},\"axes\"",
                SPECIALIZATION_SCHEMA_VERSION + 1
            ),
            "portfolio.contract.schema_version",
        ),
        (
            "body",
            format!("{{\"schema_version\":{PORTFOLIO_ENVELOPE_SCHEMA_VERSION},\"contract\""),
            format!(
                "{{\"schema_version\":{},\"contract\"",
                PORTFOLIO_ENVELOPE_SCHEMA_VERSION + 1
            ),
            "portfolio.body.schema_version",
        ),
    ] {
        let body = String::from_utf8(frame_body(&bytes).to_vec())
            .expect("Fix: a canonical set body must be UTF-8 JSON.");
        assert_eq!(
            body.matches(from.as_str()).count(),
            1,
            "Fix: the {label} schema must appear exactly once, or this case edits the wrong field."
        );
        let stale = reframe(body.replace(from.as_str(), to.as_str()).as_bytes());

        let error = PortfolioEnvelope::from_bytes(&stale)
            .expect_err("Fix: a set stating another schema must be refused, not read.");
        assert_eq!(
            error.diagnostic.code.as_str(),
            "MKC015_VERSION_SKEW",
            "Fix: a stale {label} schema must be reported as version skew: {error}"
        );
        assert_eq!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|location| location.path.as_deref()),
            Some(path),
            "Fix: the refusal must name the field that disagrees."
        );
    }
}

/// The canonical body of one framed product.
fn frame_body(frame: &[u8]) -> &[u8] {
    let body_len = u32::from_le_bytes(
        frame[6..10]
            .try_into()
            .expect("Fix: read the fixed header."),
    ) as usize;
    &frame[10..10 + body_len]
}

/// Re-frame an edited body so it authenticates and only its content differs.
fn reframe(body: &[u8]) -> Vec<u8> {
    let version = PORTFOLIO_ENVELOPE_SCHEMA_VERSION;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-megakernel-portfolio-v1\0");
    hasher.update(&version.to_le_bytes());
    hasher.update(&(body.len() as u64).to_le_bytes());
    hasher.update(body);
    let mut bytes = Vec::with_capacity(10 + body.len() + 32);
    bytes.extend_from_slice(b"VMP0");
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
    bytes.extend_from_slice(body);
    bytes.extend_from_slice(hasher.finalize().as_bytes());
    bytes
}

/// One artifact over the fixture graph, distinguished by `seed`.
fn baseline_artifact(seed: u8) -> Artifact {
    vyre_megakernel::compile(&request(4, seed)).expect("Fix: the fixture request must compile.")
}

/// One artifact over a different graph, so its source identity differs.
fn foreign_artifact() -> Artifact {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(4)],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let graph = ProgramGraph::from_program("foreign", program)
        .expect("Fix: the foreign fixture graph must be valid.");
    let stated = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([7; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(32, 100_000, 4, 0, 1_000_000_000),
        objective(4),
    )
    .validate()
    .expect("Fix: the foreign request must validate.");
    vyre_megakernel::compile(&stated).expect("Fix: the foreign request must compile.")
}

/// Facts that do not state exactly the declared axes are refused.
///
/// A guard reads the axes it names and nothing else, so an axis the facts leave
/// unstated falls through every guard to the remainder and is served by an
/// artifact nothing checked against it. An axis the contract never declared is
/// the same mistake stated out loud: the caller believes it selects on a fact
/// the set was never specialized over.
#[test]
fn facts_that_do_not_state_the_declared_axes_are_refused() {
    let portfolio = compile_specialized_portfolio(
        &request(4, 7),
        &contract(),
        &[in_range(1, 64, 0)],
        RemainderKind::Generic,
    )
    .expect("Fix: one guard with a generic remainder must compile.");

    let unstated = portfolio
        .select(&BTreeMap::new())
        .expect_err("Fix: facts that state no value for a declared axis must be refused.");
    assert_eq!(
        unstated.diagnostic.code.as_str(),
        "MKC039_UNSUPPORTED_WORKLOAD",
        "Fix: an axis nothing stated cannot be checked, so the workload is unsupported: \
         {unstated}"
    );

    let undeclared = portfolio
        .select(&BTreeMap::from([
            (tokens(), AxisValue::Scalar(8)),
            (SpecializationAxis::LaunchBatch, AxisValue::Scalar(4)),
        ]))
        .expect_err("Fix: facts naming an axis outside the contract must be refused.");
    assert_eq!(
        undeclared.diagnostic.code.as_str(),
        "MKC039_UNSUPPORTED_WORKLOAD",
        "Fix: selecting on an axis the set never specialized over must be refused: {undeclared}"
    );
    assert_eq!(
        undeclared
            .diagnostic
            .location
            .as_ref()
            .and_then(|at| at.path.clone()),
        Some(SpecializationAxis::LaunchBatch.field()),
        "Fix: the refusal must name the undeclared axis the facts stated."
    );
}
