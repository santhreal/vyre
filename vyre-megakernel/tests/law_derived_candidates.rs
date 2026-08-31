//! Candidate search reads the declared laws.
//!
//! WHY: two law-derivation mechanisms shipped with no in-tree consumer.
//! `derive_program_alternative` reads the laws a combine declares and rewrites
//! the expressions of a program; `derive_region_alternatives` composes the
//! region law families into equivalent programs. Candidate search built its set
//! from the schedule grammar alone, so a declared law authorized nothing any
//! selection ranked, and a law row could be added, removed, or broken without a
//! single test noticing.
//!
//! Every expected law set here is recomputed from the same derivations over the
//! same node programs at run time, never from a written-down list: a law row
//! added to either table, or a derivation candidate search stops calling, moves
//! the expected set and turns this red.
//!
//! What these do not catch: whether a law-derived candidate is ever *selected*.
//! Selection is a ranking decision the cost model owns, and a law that produces
//! a slower program is correctly passed over.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre_foundation::algebraic_reordering::every_declared_type_is_exact;
use vyre_foundation::ir::{
    BinOp, BufferAccess, DataType, Expr, Program, ProgramGraph, ValueLifetime,
};
use vyre_foundation::numeric::{ErrorMeasure, NumericContract, Reassociation};
use vyre_foundation::optimizer::law_saturation::{derive_program_alternative, LawSaturationBudget};
use vyre_foundation::optimizer::region_law::{
    derive_region_alternatives, law_numerical_contract, RegionDerivationBudget, REGION_LAWS,
};
use vyre_foundation::optimizer::rewrite_contract::NumericalContract;
use vyre_megakernel::{Artifact, SearchBudget, SearchCertificate};

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{
    artifact_of, artifact_of_within, contract, invocation, reduce_program, reduction_over, stage,
};

/// The element every graph here combines.
///
/// `U32` is exact, so every integer law is admitted without a grant and the
/// derivations are reachable before permission enters the picture.
fn element() -> DataType {
    DataType::U32
}

/// A reduction whose combine has a right identity written out.
///
/// `load + 0` is the shape the value-level derivation shrinks: the combine
/// declares an identity element, so the mirror unions the sum with the loaded
/// term and extraction returns the smaller one.
fn identity_program(input: &str, output: &str, element: &DataType) -> Program {
    reduction_over(
        input,
        output,
        element,
        Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::load(input, Expr::LocalId { axis: 0 })),
            right: Box::new(Expr::u32(0)),
        },
    )
}

/// A reduction whose combine states its literal on the left.
///
/// Non-canonical operand order is what the `canonical_operand_order` law's
/// rewrite changes, so this reaches the region derivation.
fn skewed_program(input: &str, output: &str, element: &DataType) -> Program {
    reduction_over(
        input,
        output,
        element,
        Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::u32(1)),
            right: Box::new(Expr::load(input, Expr::LocalId { axis: 0 })),
        },
    )
}

/// A reduction whose combine adds two literals together.
///
/// An operator application over literal operands is what the
/// `literal_evaluation` law's rewrite replaces.
fn foldable_program(input: &str, output: &str, element: &DataType) -> Program {
    reduction_over(
        input,
        output,
        element,
        Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::u32(2)),
                right: Box::new(Expr::u32(3)),
            }),
            right: Box::new(Expr::load(input, Expr::LocalId { axis: 0 })),
        },
    )
}

/// A three-stage chain whose stages the two law tables reach.
///
/// One stage per shape: a right identity for the value-level derivation, a
/// non-canonical operand order and a literal application for the region
/// derivation. A graph whose stages no law matches would let every contract
/// here pass while candidate construction called neither derivation.
fn law_reachable_graph(element: &DataType) -> ProgramGraph {
    chain_of(
        element,
        [identity_program, skewed_program, foldable_program],
    )
}

/// A chain of the same shape whose stages no declared law matches.
///
/// Every stage reduces a loaded element with nothing to canonicalize, fold, or
/// shrink. This is the control for the reachable chain: the two differ only in
/// what the law tables say about their node programs.
fn law_inert_graph(element: &DataType) -> ProgramGraph {
    chain_of(element, [reduce_program, reduce_program, reduce_program])
}

/// One graph node program built from an input buffer, an output buffer, and an
/// element type.
type Stage = fn(&str, &str, &DataType) -> Program;

/// A three-stage chain over `element`, one node per stage.
fn chain_of(element: &DataType, stages: [Stage; 3]) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let source = graph
        .add_external_value("in_a", invocation(element))
        .expect("Fix: the external input must be admitted");
    let first = stage(
        &mut graph,
        "n0",
        stages[0]("in_a", "mid_a", element),
        ("in_a", source),
        ("mid_a", invocation(element)),
    );
    let second = stage(
        &mut graph,
        "n1",
        stages[1]("mid_a", "mid_b", element),
        ("mid_a", first),
        ("mid_b", invocation(element)),
    );
    stage(
        &mut graph,
        "n2",
        stages[2]("mid_b", "out", element),
        ("mid_b", second),
        (
            "out",
            contract(element, BufferAccess::ReadWrite, ValueLifetime::Output),
        ),
    );
    graph
}

/// A contract that grants reassociation within a stated error budget.
fn permissive() -> NumericContract {
    NumericContract {
        measure: ErrorMeasure::Ulp { count: 4 },
        reassociation: Reassociation::WithinBudget,
        ..NumericContract::EXACT
    }
}

/// The contracts a request that asks for the exact result grants a law.
///
/// Stated here rather than recomputed from the request, so this side of the
/// comparison is an independent claim about what a bit-exact caller admits.
/// Integer results are identical wrapping included, which is why an exact
/// request still admits `IntegerWrapping`.
const EXACT_GRANTS: [NumericalContract; 2] = [
    NumericalContract::BitExact,
    NumericalContract::IntegerWrapping,
];

/// The contracts a request that grants everything a law may declare.
const EVERY_GRANT: [NumericalContract; 5] = [
    NumericalContract::BitExact,
    NumericalContract::IntegerWrapping,
    NumericalContract::FloatReassociation,
    NumericalContract::FloatContraction,
    NumericalContract::ReducedPrecision,
];

/// Region law names the derivation reports for every node program of `graph`.
fn region_names(graph: &ProgramGraph, grants: &[NumericalContract]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for node in graph.nodes() {
        let derived =
            derive_region_alternatives(&node.program, grants, RegionDerivationBudget::default())
                .expect("Fix: the registered pass set the region laws cite must be orderable");
        for alternative in &derived.alternatives {
            names.extend(alternative.chain.iter().map(|name| (*name).to_owned()));
        }
    }
    names
}

/// Value-level rewrite names the derivation reports for every node program.
///
/// `reassociates` states whether the request grants reordering a rounding
/// combine. An exact element type reads the exact laws either way.
fn value_names(graph: &ProgramGraph, reassociates: bool) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for node in graph.nodes() {
        let exact = reassociates || every_declared_type_is_exact(&node.program);
        let derived =
            derive_program_alternative(&node.program, exact, LawSaturationBudget::default())
                .expect("Fix: the expression mirror must build for a fixture program");
        if let Some(derived) = derived {
            names.extend(derived.chain.iter().map(|name| (*name).to_owned()));
        }
    }
    names
}

/// Every law name the certificate accounts for, cited or eliminated.
///
/// Admission may refuse a law-derived candidate on a device fact, so the set
/// candidate construction read is the union of the two records rather than the
/// cited one alone.
fn accounted_names(certificate: &SearchCertificate) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = certificate
        .cited_laws()
        .into_iter()
        .map(str::to_owned)
        .collect();
    for pruned in &certificate.law_pruned {
        names.extend(pruned.citation.laws.iter().cloned());
    }
    names
}

fn certificate_of(artifact: &Artifact) -> &SearchCertificate {
    &artifact.selected_plan().certificate
}

/// WHY: `derive_region_alternatives` had no consumer, so a region law family
/// could ship deriving nothing a selection ever saw. The expected set is
/// recomputed from the derivation, so a law row added to `REGION_LAWS` that
/// matches this graph moves both sides at once, and candidate construction
/// dropping the call moves only one.
#[test]
fn region_laws_reach_candidate_construction() {
    let element = element();
    let expected = region_names(&law_reachable_graph(&element), &EXACT_GRANTS);
    assert!(
        !expected.is_empty(),
        "no region law matches the fixture graph, so this contract proves nothing; give it a \
         program the law table reaches"
    );

    let artifact = artifact_of(law_reachable_graph(&element), None);
    let accounted = accounted_names(certificate_of(&artifact));
    let missing: Vec<&String> = expected.difference(&accounted).collect();
    assert!(
        missing.is_empty(),
        "candidate construction never accounted for region laws {missing:?}"
    );
}

/// WHY: `derive_alternatives` had no consumer either, and it is the other half
/// of the row. A value-level law that shrinks a term must reach the candidate
/// set through the same path.
#[test]
fn value_laws_reach_candidate_construction() {
    let element = element();
    let expected = value_names(&law_reachable_graph(&element), false);
    assert!(
        !expected.is_empty(),
        "no declared combine law rewrites the fixture graph, so this contract proves nothing"
    );

    let artifact = artifact_of(law_reachable_graph(&element), None);
    let accounted = accounted_names(certificate_of(&artifact));
    let missing: Vec<&String> = expected.difference(&accounted).collect();
    assert!(
        missing.is_empty(),
        "candidate construction never accounted for value-level rewrites {missing:?}"
    );
}

/// WHY: a ranked alternative that cannot name the laws it came from is
/// indistinguishable from a shape somebody wrote down, which is the property
/// the derivation exists to provide.
#[test]
fn every_law_derived_candidate_carries_a_chain() {
    let element = element();
    let artifact = artifact_of(law_reachable_graph(&element), None);
    let certificate = certificate_of(&artifact);
    let node_count = u32::try_from(law_reachable_graph(&element).nodes().len())
        .expect("Fix: the fixture graph must have a representable node count");

    assert!(
        !certificate.law_derived.is_empty() || !certificate.law_pruned.is_empty(),
        "no law-derived candidate was recorded for a graph the law tables reach"
    );
    for citation in &certificate.law_derived {
        assert!(
            !citation.laws.is_empty(),
            "a law-derived candidate cited no law: {citation:?}"
        );
        assert!(
            citation.node < node_count,
            "a citation names node {} of a {node_count}-node graph",
            citation.node
        );
    }
}

/// WHY: a law whose declared numerical contract the request did not grant must
/// authorize nothing. The non-bit-exact rows are read from `REGION_LAWS` at run
/// time, so a law added under a value-changing contract is covered the moment
/// it is declared.
#[test]
fn a_law_the_request_did_not_grant_is_absent() {
    let value_changing: Vec<&str> = REGION_LAWS
        .iter()
        .filter(|law| law_numerical_contract(law) != Some(NumericalContract::BitExact))
        .map(|law| law.name)
        .collect();
    assert!(
        !value_changing.is_empty(),
        "no declared law states a value-changing contract, so permission cannot be tested"
    );

    let element = element();
    let exact = artifact_of(law_reachable_graph(&element), Some(NumericContract::EXACT));
    let accounted = accounted_names(certificate_of(&exact));
    for name in &accounted {
        let Some(law) = REGION_LAWS.iter().find(|law| law.name == *name) else {
            continue;
        };
        let contract = law_numerical_contract(law);
        assert!(
            contract == Some(NumericalContract::BitExact)
                || contract == Some(NumericalContract::IntegerWrapping),
            "law {name} declares {contract:?}, a contract the exact request never granted"
        );
    }
}

/// WHY: the derivation must not replace the program as written. Every law the
/// selected plan names has to be one the search admitted, and a plan that names
/// none is the unfused baseline the grammar derived, which stays in the set. A
/// seeding step that overwrote the baseline instead of adding to it would leave
/// a plan citing a chain no certificate carries, which is what
/// `SelectedPlan::validate` rejects.
#[test]
fn the_selected_plan_names_only_admitted_laws() {
    let element = element();
    let artifact = artifact_of(law_reachable_graph(&element), None);
    let plan = artifact.selected_plan();

    plan.validate()
        .expect("Fix: the selected plan must name only law chains the search cited");
    for citation in &plan.law_derivation {
        assert!(
            plan.certificate.law_derived.contains(citation),
            "the plan names law chain {citation:?} the search never cited"
        );
    }

    let exact = artifact_of(law_reachable_graph(&element), Some(NumericContract::EXACT));
    let exact_names = accounted_names(certificate_of(&exact));
    let permissive_names = accounted_names(certificate_of(&artifact_of(
        law_reachable_graph(&element),
        Some(permissive()),
    )));
    let lost: Vec<&String> = exact_names.difference(&permissive_names).collect();
    assert!(
        lost.is_empty(),
        "granting a contract stopped accounting for laws {lost:?} the exact request reached"
    );
}

/// WHY: a certificate is the reproducible record of one search, so two searches
/// over the same request must record the same law set in the same order. A law
/// record whose order depends on a hash map iteration would make an artifact
/// irreproducible without changing any decision.
#[test]
fn the_law_record_is_canonical() {
    let element = element();
    let first = artifact_of(law_reachable_graph(&element), None);
    let second = artifact_of(law_reachable_graph(&element), None);

    let left = certificate_of(&first);
    let right = certificate_of(&second);
    assert_eq!(left.law_derived, right.law_derived);
    assert_eq!(left.law_pruned, right.law_pruned);
    assert_eq!(left.law_budget_reached, right.law_budget_reached);

    let mut sorted = left.law_derived.clone();
    sorted.sort_unstable();
    assert_eq!(
        left.law_derived, sorted,
        "the law record is not in canonical order"
    );
}

/// WHY: the row this suite closes asks for a proof that a law the fixture
/// matches changes the candidate set, not only that a derivation was called.
/// Two chains of identical shape differ in one respect: the law tables reach the
/// node programs of one and match nothing in the other. The inert chain's law
/// reachability is recomputed at run time, so a law row added later that does
/// match it turns this red instead of quietly making the control useless.
///
/// The inert compile is also the control for the baseline. A seeding step that
/// replaced the candidate set with law-derived plans would leave a graph no law
/// reaches with nothing to rank, and this compile would fail instead of
/// returning the program as written.
#[test]
fn a_matching_law_changes_the_candidate_set() {
    let element = element();
    let inert = law_inert_graph(&element);
    let mut inert_expected = region_names(&inert, &EVERY_GRANT);
    inert_expected.extend(value_names(&inert, true));
    assert!(
        inert_expected.is_empty(),
        "the control chain now matches laws {inert_expected:?}; state a control the law tables \
         do not reach, or this contract compares two reachable graphs"
    );

    let inert_artifact = artifact_of(law_inert_graph(&element), None);
    let inert_certificate = certificate_of(&inert_artifact);
    assert!(
        inert_certificate.law_derived.is_empty() && inert_certificate.law_pruned.is_empty(),
        "a graph no law reaches recorded law derivation: {:?} cited, {:?} eliminated",
        inert_certificate.law_derived,
        inert_certificate.law_pruned
    );
    assert!(
        inert_artifact.selected_plan().law_derivation.is_empty(),
        "the plan for a graph no law reaches names a law chain"
    );

    let reachable = accounted_names(certificate_of(&artifact_of(
        law_reachable_graph(&element),
        None,
    )));
    assert!(
        !reachable.is_empty(),
        "the reachable chain recorded no law, so the two chains produced the same candidate set"
    );
}

/// WHY: the candidate budget is the count an artifact authenticates, and law
/// derivation seeded the set outside it. Every law-derived alternative was
/// pushed before the first bound was consulted, so a graph the law tables reach
/// at more nodes than the budget grants candidates recorded more explored
/// candidates than the request authorized, and admission refused the artifact
/// with `MKC014_MALFORMED_ARTIFACT` at
/// `selected_plan.search_work.candidates_explored`. The count the fixture can
/// reach is recomputed from the certificate at run time, so a law row added to
/// either table widens the range this ranges over instead of leaving the
/// interesting caps untested.
#[test]
fn a_law_derived_candidate_stays_inside_the_authenticated_budget() {
    let element = element();
    let accounted = {
        let artifact = artifact_of(law_reachable_graph(&element), None);
        let certificate = certificate_of(&artifact);
        certificate.law_derived.len() + certificate.law_pruned.len()
    };
    assert!(
        accounted >= 2,
        "the fixture chain accounted for {accounted} law-derived candidates, so no cap below the \
         law-derived count is reachable and this contract proves nothing"
    );

    for max_candidates in 1..=u32::try_from(accounted).expect("law count must fit a budget") + 1 {
        let artifact = artifact_of_within(
            law_reachable_graph(&element),
            None,
            SearchBudget::new(max_candidates, 200_000, 4, 0, 1_000_000_000),
        );
        let plan = artifact.selected_plan();
        assert!(
            plan.candidates_explored <= max_candidates,
            "a budget of {max_candidates} candidates recorded {} explored",
            plan.candidates_explored
        );
        assert_eq!(
            plan.candidates_explored, plan.search_work.candidates_explored,
            "the plan and its search work disagree about the bounded set"
        );
        assert!(
            plan.candidates_explored >= 1,
            "a bounded search dropped the baseline the program was written as"
        );
    }
}
