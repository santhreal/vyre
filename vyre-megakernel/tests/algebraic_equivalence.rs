//! A schedule may reorder a reduction only when the reduction reassociates.
//!
//! A spatial partition, a persistent queue, a pipeline, and an asymmetric join
//! all let independent workers reach a shared accumulator in an order the
//! schedule does not fix. Over integer addition that is the same number; over
//! floating-point addition it is a different one, and the difference is
//! data-dependent, so it reaches a caller as an accuracy report rather than as a
//! failure. Before this constraint the search admitted those candidates for a
//! rounding reduction and could select one.
//!
//! The two graphs here are the same kernel organization over two element types,
//! so the only variable is whether the reduction reassociates. The last case
//! derives the whole production vocabulary from `ScheduleProduction::ALL` and
//! requires every production the rounding graph loses to state `Numerical`,
//! which is what a production added later has to satisfy.

#[path = "support/search_fixtures.rs"]
mod search_fixtures;

use search_fixtures::{numerically_pruned, reducing_artifact, REORDERING_PRODUCTIONS};

use vyre_foundation::ir::DataType;
use vyre_megakernel::{ScheduleProduction, SearchCertificate};

/// Productions that move work without changing which invocations combine.
const ORDER_PRESERVING: [ScheduleProduction; 4] = [
    ScheduleProduction::DispatchCut,
    ScheduleProduction::Synchronization,
    ScheduleProduction::MemoryPlacement,
    ScheduleProduction::Prefetch,
];

fn certificate(element: &DataType) -> SearchCertificate {
    reducing_artifact(element, None)
        .selected_plan()
        .certificate
        .clone()
}

#[test]
fn an_exact_reduction_keeps_every_reordering_production() {
    let exact = certificate(&DataType::U32);

    for production in REORDERING_PRODUCTIONS {
        assert!(
            exact.admitted_by(production) > 0,
            "{production:?} admitted nothing over an exact reduction: {exact:#?}"
        );
        assert_eq!(
            numerically_pruned(&exact, production),
            0,
            "{production:?} was called numerical over an exact reduction"
        );
    }
}

#[test]
fn a_rounding_reduction_eliminates_every_reordering_production() {
    let rounding = certificate(&DataType::F32);

    for production in REORDERING_PRODUCTIONS {
        assert_eq!(
            rounding.admitted_by(production),
            0,
            "{production:?} admitted a candidate that reorders a rounding reduction"
        );
        assert!(
            numerically_pruned(&rounding, production) > 0,
            "{production:?} lost its candidates without stating Numerical: {rounding:#?}"
        );
    }
}

#[test]
fn a_rounding_reduction_still_compiles_through_the_order_preserving_productions() {
    let rounding = certificate(&DataType::F32);

    for production in ORDER_PRESERVING {
        assert!(
            rounding.admitted_by(production) > 0,
            "{production:?} admitted nothing, so the rounding graph lost more than reordering"
        );
    }
}

#[test]
fn every_production_an_exact_reduction_admits_is_kept_or_stated_numerical() {
    let exact = certificate(&DataType::U32);
    let rounding = certificate(&DataType::F32);

    for production in ScheduleProduction::ALL.iter().copied() {
        if exact.admitted_by(production) == 0 {
            continue;
        }
        if rounding.admitted_by(production) > 0 {
            continue;
        }
        assert!(
            numerically_pruned(&rounding, production) > 0,
            "{production:?} disappeared over a rounding reduction without stating Numerical"
        );
    }
}
