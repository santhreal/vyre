//! What a magnitude bound proves, and which schedule choice each proof enables.
//!
//! WHY: narrowing a format, narrowing an accumulator, selecting an approximate
//! instruction and reordering a sum are all legal over some values and wrong
//! over others. The deciding fact is the range: a value that fits binary16 may
//! be stored in it, partial sums that do not fit may not be accumulated in it,
//! an inverse is bounded only away from zero, and a sum of mixed signs cancels
//! so a reordered order is not within a per-step bound. Each case fixes one of
//! those, so a proof reused for a choice it was not carried out for fails.
//!
//! What these cases do not prove: where the range came from. Deriving one from
//! program values is the analysis that feeds this; here the bound is stated.

use vyre_foundation::numeric::{
    prove, Approximation, ContractRefusal, ErrorMeasure, MagnitudeRange, NumericChoice,
    NumericContract, Reassociation, ScalarFormat,
};

/// Every choice kind, derived from the enum rather than listed.
///
/// An exhaustive match with no catch-all arm is the closure: a choice added to
/// the enum stops this file compiling until someone records what proves it.
fn describe(choice: NumericChoice) -> &'static str {
    match choice {
        NumericChoice::StoreAs(_) => "store",
        NumericChoice::AccumulateIn { .. } => "accumulate",
        NumericChoice::Approximate { .. } => "approximate",
        NumericChoice::Reassociate { .. } => "reassociate",
        NumericChoice::ChunkReduction { .. } => "chunk",
    }
}

/// Every refusal kind, derived from the enum rather than listed.
fn refusal_kind(refusal: &ContractRefusal) -> &'static str {
    match refusal {
        ContractRefusal::FormatDisagrees { .. } => "format disagrees",
        ContractRefusal::UnboundedMagnitude { .. } => "unbounded magnitude",
        ContractRefusal::FormatMismatch { .. } => "format mismatch",
        ContractRefusal::BudgetExceeded { .. } => "budget exceeded",
        ContractRefusal::ReassociationRefused { .. } => "reassociation refused",
        ContractRefusal::ApproximationRefused => "approximation refused",
        ContractRefusal::RangeUnproven { .. } => "range unproven",
    }
}

fn unit_range() -> MagnitudeRange {
    MagnitudeRange::new(0.25, 1.0).expect("a bounded interval is a range")
}

/// A contract admitting one part in a hundred, which is wide enough that a
/// narrowing is a budget question rather than an arithmetic one.
fn loose_f32() -> NumericContract {
    NumericContract::of(ScalarFormat::F32)
        .with_measure(ErrorMeasure::relative(1e-2))
        .reassociating(Reassociation::WithinBudget)
}

#[test]
fn an_interval_is_a_range_only_when_it_bounds_something() {
    assert!(MagnitudeRange::new(f64::NAN, 1.0).is_none());
    assert!(MagnitudeRange::new(1.0, -1.0).is_none());
    let range = MagnitudeRange::new(-2.0, 0.5).expect("an ordered interval is a range");
    assert_eq!(range.peak(), 2.0);
    assert_eq!(range.floor(), 0.0, "a range across zero reaches zero");
    assert!(range.contains_zero());
    assert!(!range.single_signed());
}

#[test]
fn an_exponential_is_bounded_by_the_exponential_of_its_bound() {
    let bounded = MagnitudeRange::new(-3.0, 3.0)
        .expect("an interval")
        .exponential()
        .expect("a bounded exponent has a bounded exponential");
    assert!((bounded.high() - 3.0_f64.exp()).abs() < 1e-9);
    assert!(bounded.low() > 0.0, "an exponential is never zero");
    assert!(
        MagnitudeRange::new(0.0, 1000.0)
            .expect("an interval")
            .exponential()
            .is_none(),
        "an exponent nothing bounds has no representable exponential"
    );
}

#[test]
fn an_inverse_is_bounded_only_away_from_zero() {
    let inverted = MagnitudeRange::new(0.5, 2.0)
        .expect("an interval")
        .reciprocal()
        .expect("an interval away from zero inverts");
    assert!((inverted.low() - 0.5).abs() < 1e-12);
    assert!((inverted.high() - 2.0).abs() < 1e-12);
    assert!(
        MagnitudeRange::new(-1.0, 1.0)
            .expect("an interval")
            .reciprocal()
            .is_none(),
        "an inverse through zero is unbounded whatever the endpoints"
    );
}

/// WHY: a recurrent state is the shape a schedule most wants to run longer, and
/// the only thing that decides whether it stays finite is the gain. Treating a
/// growing recurrence as bounded is how a persistent schedule ships an overflow.
#[test]
fn a_recurrence_is_bounded_by_its_gain() {
    let start = MagnitudeRange::symmetric(1.0).expect("a symmetric interval");
    let contracting = start
        .affine_recurrence(0.5, 1.0, 1_000_000)
        .expect("a gain under one converges");
    assert!(contracting.peak() <= 3.0);
    assert!(
        start.affine_recurrence(1.5, 1.0, 4096).is_none(),
        "a gain over one grows past every representable bound"
    );
    let neutral = start
        .affine_recurrence(1.0, 1.0, 16)
        .expect("a unit gain accumulates its offset");
    assert!((neutral.peak() - 17.0).abs() < 1e-12);
}

#[test]
fn a_value_that_fits_may_be_stored_in_a_narrower_format() {
    let proof = prove(
        unit_range(),
        &loose_f32(),
        NumericChoice::StoreAs(ScalarFormat::F16),
    )
    .expect("a unit value fits binary16 inside a one-percent budget");
    assert_eq!(describe(proof.choice), "store");
    assert_eq!(proof.range, unit_range());
    assert!(proof.measure.magnitude() > 0.0, "narrowing costs something");
}

#[test]
fn a_value_that_does_not_fit_refuses_the_narrower_format() {
    let large = MagnitudeRange::new(0.0, 1.0e6).expect("an interval");
    let refusal = prove(
        large,
        &loose_f32(),
        NumericChoice::StoreAs(ScalarFormat::F16),
    )
    .expect_err("a million does not fit binary16");
    assert_eq!(refusal_kind(&refusal), "range unproven");
}

/// WHY: the values fit the accumulator and their sum does not. Checking the
/// value range alone is the defect this case exists to catch: it admits an
/// accumulator that overflows partway through a long reduction.
#[test]
fn partial_sums_decide_the_accumulator_not_the_values() {
    let values = MagnitudeRange::new(0.0, 1000.0).expect("an interval");
    assert!(
        values.fits(ScalarFormat::F16),
        "the individual values fit binary16"
    );
    let refusal = prove(
        values,
        &loose_f32(),
        NumericChoice::AccumulateIn {
            format: ScalarFormat::F16,
            terms: 4096,
        },
    )
    .expect_err("four thousand values of a thousand overflow binary16");
    assert_eq!(refusal_kind(&refusal), "range unproven");
}

#[test]
fn an_accumulator_that_holds_the_sums_is_admitted_when_the_budget_covers_it() {
    let values = MagnitudeRange::new(0.0, 1.0).expect("an interval");
    let proof = prove(
        values,
        &loose_f32(),
        NumericChoice::AccumulateIn {
            format: ScalarFormat::F32,
            terms: 1024,
        },
    )
    .expect("an f32 accumulator over a unit range stays inside one percent");
    assert_eq!(describe(proof.choice), "accumulate");
}

#[test]
fn a_contract_that_refuses_approximation_refuses_the_instruction() {
    let refusal = prove(
        unit_range(),
        &loose_f32(),
        NumericChoice::Approximate {
            measure: ErrorMeasure::relative(1e-6),
        },
    )
    .expect_err("the default contract admits no approximate instruction");
    assert_eq!(refusal_kind(&refusal), "approximation refused");

    let admitting = loose_f32().approximating(Approximation::Native {
        measure: ErrorMeasure::relative(1e-6),
    });
    let proof = prove(
        unit_range(),
        &admitting,
        NumericChoice::Approximate {
            measure: ErrorMeasure::relative(1e-6),
        },
    )
    .expect("a contract admitting one takes one inside its budget");
    assert_eq!(describe(proof.choice), "approximate");

    let too_coarse = prove(
        unit_range(),
        &admitting,
        NumericChoice::Approximate {
            measure: ErrorMeasure::relative(1e-1),
        },
    )
    .expect_err("a tenth is wider than the declared hundredth");
    assert_eq!(refusal_kind(&too_coarse), "budget exceeded");
}

/// WHY: reordering a sum is within a per-step bound only where the terms do not
/// cancel. Over a range spanning zero the exact sum can be arbitrarily small
/// beside its terms, so a relative bound proven per step says nothing about the
/// result, and a reordered schedule can return a different answer entirely.
#[test]
fn reassociation_needs_terms_that_do_not_cancel() {
    let signed = MagnitudeRange::new(-1.0, 1.0).expect("an interval");
    let refusal = prove(
        signed,
        &loose_f32(),
        NumericChoice::Reassociate { terms: 1024 },
    )
    .expect_err("a sum that cancels has no relative bound to reorder inside");
    assert_eq!(refusal_kind(&refusal), "range unproven");

    let proof = prove(
        unit_range(),
        &loose_f32(),
        NumericChoice::Reassociate { terms: 1024 },
    )
    .expect("same-signed terms reorder inside the budget");
    assert_eq!(describe(proof.choice), "reassociate");
}

#[test]
fn a_contract_that_states_the_order_refuses_reassociation() {
    let ordered = NumericContract::of(ScalarFormat::F32)
        .with_measure(ErrorMeasure::relative(1e-2))
        .reassociating(Reassociation::Forbidden);
    let refusal = prove(
        unit_range(),
        &ordered,
        NumericChoice::Reassociate { terms: 8 },
    )
    .expect_err("the stated order is the contract");
    assert_eq!(refusal_kind(&refusal), "reassociation refused");
}

#[test]
fn a_chunked_reduction_is_priced_between_the_two_orders() {
    let proof = prove(
        unit_range(),
        &loose_f32(),
        NumericChoice::ChunkReduction {
            terms: 1024,
            chunk: 32,
        },
    )
    .expect("thirty-two-value chunks stay inside one percent");
    assert_eq!(describe(proof.choice), "chunk");

    let sequential = prove(
        unit_range(),
        &loose_f32(),
        NumericChoice::ChunkReduction {
            terms: 1024,
            chunk: 1024,
        },
    )
    .expect("one chunk is the sequential order")
    .measure;
    assert!(
        sequential.magnitude() > proof.measure.magnitude(),
        "one long chunk rounds more often than thirty-two short ones"
    );

    let refusal = prove(
        unit_range(),
        &loose_f32(),
        NumericChoice::ChunkReduction {
            terms: 16,
            chunk: 64,
        },
    )
    .expect_err("a chunk longer than the reduction is not a chunking");
    assert_eq!(refusal_kind(&refusal), "range unproven");
}

/// WHY: an absolute bound becomes a relative one by dividing by the magnitude,
/// which is exactly the conversion a graph needs to compose one region's
/// absolute bound with the next region's relative one. Over a range that
/// reaches zero there is nothing to divide by, and a fraction produced anyway
/// would be a budget nothing measured.
#[test]
fn an_absolute_bound_reads_as_a_fraction_only_away_from_zero() {
    let away = MagnitudeRange::new(2.0, 4.0).expect("an interval");
    let fraction = away
        .relative_of(ErrorMeasure::absolute(0.5), ScalarFormat::F32)
        .expect("an absolute bound over a bounded magnitude is a fraction");
    assert!((fraction - 0.25).abs() < 1e-12);

    let across = MagnitudeRange::new(-1.0, 1.0).expect("an interval");
    let refusal = across
        .relative_of(ErrorMeasure::absolute(0.5), ScalarFormat::F32)
        .expect_err("a magnitude that reaches zero divides into nothing");
    assert_eq!(refusal_kind(&refusal), "range unproven");
}
