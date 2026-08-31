//! A stated numeric budget decides which reorderings the search may keep.
//!
//! WHY: a rounding accumulation computes a different number in a different
//! order, so every schedule that changes combine order is eliminated when
//! nothing states how far the result may move. That is correct and it is also
//! why a floating reduction never reaches a tree order, a spatial partition or a
//! persistent queue: the compiler has no bound to check against. A caller that
//! states one gets the schedules whose error the bound covers, and a caller that
//! states a bound too tight for the reordering keeps the strict answer.
//!
//! The same graph is compiled three ways here, so the only variable is the
//! budget. What these cases do not prove: what the device measures. The bound is
//! proven on the IR; conformance measures the result.

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{numerically_pruned, reducing_artifact, REORDERING_PRODUCTIONS};

use vyre_foundation::ir::DataType;
use vyre_foundation::numeric::{
    ErrorMeasure, NumericContract, Reassociation, ScalarFormat, NUMERIC_CONTRACT_VERSION,
};
use vyre_megakernel::{Artifact, ExecutionMode, SearchCertificate};

fn artifact(element: DataType, budget: Option<NumericContract>) -> Artifact {
    reducing_artifact(&element, budget)
}

fn certificate(budget: Option<NumericContract>) -> SearchCertificate {
    artifact(DataType::F32, budget)
        .selected_plan()
        .certificate
        .clone()
}

/// A budget of one part in a thousand, which covers reordering a reduction of
/// the fixture's four thousand points many times over.
fn loose_budget() -> NumericContract {
    NumericContract::of(ScalarFormat::F32)
        .with_measure(ErrorMeasure::relative(1e-3))
        .reassociating(Reassociation::WithinBudget)
}

/// A budget of one part in a billion, which a reordered binary32 reduction of
/// four thousand points does not fit inside.
fn tight_budget() -> NumericContract {
    NumericContract::of(ScalarFormat::F32)
        .with_measure(ErrorMeasure::relative(1e-9))
        .reassociating(Reassociation::WithinBudget)
}

#[test]
fn a_rounding_reduction_loses_every_reordering_production_without_a_budget() {
    let stated = certificate(None);
    for production in REORDERING_PRODUCTIONS {
        assert_eq!(
            stated.admitted_by(production),
            0,
            "{production:?} reordered a rounding reduction with no bound to check against"
        );
        assert!(
            numerically_pruned(&stated, production) > 0,
            "{production:?} lost its candidates without stating Numerical: {stated:#?}"
        );
    }
}

#[test]
fn a_stated_budget_that_covers_the_reordering_admits_it() {
    let stated = certificate(Some(loose_budget()));
    for production in REORDERING_PRODUCTIONS {
        assert!(
            stated.admitted_by(production) > 0,
            "{production:?} stayed eliminated under a budget that covers it: {stated:#?}"
        );
        assert_eq!(
            numerically_pruned(&stated, production),
            0,
            "{production:?} was called numerical inside a budget that admits it"
        );
    }
}

/// WHY: a budget is a bound, not a switch. Admitting a reordering because a
/// caller stated any budget at all would turn the one number the caller controls
/// into permission to ignore it.
#[test]
fn a_budget_too_tight_for_the_reordering_keeps_the_strict_answer() {
    let stated = certificate(Some(tight_budget()));
    for production in REORDERING_PRODUCTIONS {
        assert_eq!(
            stated.admitted_by(production),
            0,
            "{production:?} reordered a reduction wider than the stated bound"
        );
        assert!(
            numerically_pruned(&stated, production) > 0,
            "{production:?} lost its candidates without stating Numerical: {stated:#?}"
        );
    }
}

/// WHY: the search must gain candidates from a budget, not merely relabel them.
/// A reordering that is admitted and then never explored is a bound that changed
/// a certificate and nothing else.
#[test]
fn a_stated_budget_widens_the_candidate_set() {
    let strict = certificate(None);
    let loose = certificate(Some(loose_budget()));
    let strict_total: u32 = REORDERING_PRODUCTIONS
        .iter()
        .map(|production| strict.admitted_by(*production))
        .sum();
    let loose_total: u32 = REORDERING_PRODUCTIONS
        .iter()
        .map(|production| loose.admitted_by(*production))
        .sum();
    assert_eq!(strict_total, 0);
    assert!(
        loose_total > 0,
        "a budget that admits reordering must reach schedules the strict answer refuses"
    );
}

/// WHY: a consumer reading the artifact must be able to state what the numbers
/// in it are worth without re-deriving the contracts from the programs, and a
/// conformance run needs the composed bound to compare a measurement against.
#[test]
fn the_artifact_records_the_contracts_the_plan_was_selected_under() {
    let recorded = artifact(DataType::F32, Some(loose_budget()));
    let record = &recorded.selected_plan().numeric_budget;
    assert_eq!(record.version, NUMERIC_CONTRACT_VERSION);
    assert_eq!(record.declared, Some(loose_budget()));
    assert_eq!(
        record.regions.len(),
        recorded.nodes().len(),
        "every region states a contract"
    );
    assert!(
        record
            .regions
            .iter()
            .all(|contract| contract.storage == ScalarFormat::F32),
        "the graph holds binary32, so every region does"
    );
    assert!(
        record.proven.measure != ErrorMeasure::Exact,
        "a rounding graph does not compose to an exact result"
    );
    assert_eq!(
        record.reordered,
        (0..u32::try_from(recorded.nodes().len()).expect("the fixture graph is small"))
            .collect::<Vec<_>>(),
        "the budget bought a resident route, so every rounding region is combined in an order \
         the program did not state, and the artifact says which"
    );
}

/// WHY: the record states what the plan did, not what it was allowed to do. A
/// plan that keeps every stated order and still lists reordered regions would
/// send a reader looking for an approximation that was never selected.
#[test]
fn a_plan_that_reorders_nothing_records_no_reordered_region() {
    let strict = artifact(DataType::F32, None);
    assert!(strict.selected_plan().numeric_budget.reordered.is_empty());
    assert_eq!(strict.selected_plan().numeric_budget.declared, None);
}

#[test]
fn an_exact_graph_records_exact_contracts() {
    let exact = artifact(DataType::U32, None);
    let record = &exact.selected_plan().numeric_budget;
    assert_eq!(record.proven.measure, ErrorMeasure::Exact);
    assert!(record
        .regions
        .iter()
        .all(|contract| contract.measure == ErrorMeasure::Exact));
    assert!(
        record.reordered.is_empty(),
        "an exact combine is the same value in every order, so no order is a departure"
    );
}

/// WHY: a resident kernel polling a work queue lets invocations reach a shared
/// accumulator in an order the program did not state, and the route is selected
/// after ranking, where the search is no longer looking. Selecting it over a
/// rounding accumulation with no stated budget computes a different number than
/// the program states and reports nothing, which is the defect this refuses.
/// The budget is what buys the route back, and an exact combine never needed
/// one.
#[test]
fn a_resident_route_over_ordered_combines_needs_a_stated_budget() {
    assert!(
        matches!(
            artifact(DataType::F32, None).selected_plan().execution,
            ExecutionMode::Static
        ),
        "an unbudgeted rounding accumulation does not get a route that reorders it"
    );
    assert!(
        matches!(
            artifact(DataType::F32, Some(loose_budget()))
                .selected_plan()
                .execution,
            ExecutionMode::Persistent { .. }
        ),
        "a stated budget wide enough for the new order buys the resident route"
    );
    assert!(
        matches!(
            artifact(DataType::F32, Some(NumericContract::of(ScalarFormat::F32)))
                .selected_plan()
                .execution,
            ExecutionMode::Static
        ),
        "a budget of zero units in the last place admits no reordering, so it buys nothing"
    );
    assert!(
        matches!(
            artifact(DataType::U32, None).selected_plan().execution,
            ExecutionMode::Persistent { .. }
        ),
        "an exact combine is the same value in every order, so the route needs no budget"
    );
}
