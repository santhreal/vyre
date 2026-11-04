//! What a numeric contract states, and what composing two of them proves.
//!
//! WHY: a schedule that reassociates a reduction, accumulates in a narrower
//! format or selects an approximate instruction changes the result. Before this
//! contract the only recorded numeric fact was a single f32 ULP count on an
//! operation registration, which cannot answer whether a whole graph stays
//! inside a caller's bound: a count has no composition rule, no reassociation
//! policy and no accumulator. Each case here fixes one composition rule, so a
//! rule that silently widens a budget fails rather than passing a wider result
//! off as the declared one.

use vyre_foundation::numeric::{
    Approximation, AtomicOrderSensitivity, ContractRefusal, Determinism, ErrorMeasure,
    NumericContract, Reassociation, ScalarFormat,
};
use vyre_spec::SubnormalBehavior;

/// Every measure kind, derived from the enum rather than listed.
///
/// An exhaustive match with no catch-all arm is the closure: a measure added to
/// the enum stops this file compiling until someone records what composing it
/// proves.
fn describe(measure: ErrorMeasure) -> &'static str {
    match measure {
        ErrorMeasure::Exact => "exact",
        ErrorMeasure::Ulp { .. } => "ulp",
        ErrorMeasure::Absolute { .. } => "absolute",
        ErrorMeasure::Relative { .. } => "relative",
    }
}

#[test]
fn an_exact_format_states_an_exact_contract() {
    let contract = NumericContract::of(ScalarFormat::U32);
    assert_eq!(contract.measure, ErrorMeasure::Exact);
    assert_eq!(contract.reassociation, Reassociation::Exact);
    assert_eq!(contract.determinism, Determinism::Deterministic);
    contract.check().expect("an integer contract is consistent");
}

#[test]
fn a_rounding_format_states_no_exact_result() {
    let contract = NumericContract::of(ScalarFormat::F32);
    assert_eq!(contract.measure, ErrorMeasure::Ulp { count: 0 });
    assert_eq!(contract.reassociation, Reassociation::Forbidden);
    contract.check().expect("an f32 contract is consistent");

    let claimed_exact = contract.clone().with_measure(ErrorMeasure::Exact);
    assert!(matches!(
        claimed_exact.check(),
        Err(ContractRefusal::FormatDisagrees {
            field: "measure",
            ..
        })
    ));

    let claimed_reassociable = contract.reassociating(Reassociation::Exact);
    assert!(matches!(
        claimed_reassociable.check(),
        Err(ContractRefusal::FormatDisagrees {
            field: "reassociation",
            ..
        })
    ));
}

#[test]
fn a_format_without_subnormals_refuses_to_preserve_them() {
    let mut contract = NumericContract::of(ScalarFormat::U32);
    contract.subnormal = SubnormalBehavior::PreservedIEEE;
    assert!(matches!(
        contract.check(),
        Err(ContractRefusal::FormatDisagrees {
            field: "subnormal",
            ..
        })
    ));
}

#[test]
fn an_approximation_that_contributes_nothing_is_refused() {
    let contract = NumericContract::of(ScalarFormat::F32).approximating(Approximation::Native {
        measure: ErrorMeasure::Exact,
    });
    assert!(matches!(
        contract.check(),
        Err(ContractRefusal::FormatDisagrees {
            field: "approximation",
            ..
        })
    ));
}

#[test]
fn two_ulp_bounds_over_one_format_add() {
    let first = NumericContract::of(ScalarFormat::F32).within_ulp(2);
    let second = NumericContract::of(ScalarFormat::F32).within_ulp(3);
    let composed = first.compose(&second).expect("one format composes");
    assert_eq!(composed.measure, ErrorMeasure::Ulp { count: 5 });
    assert_eq!(describe(composed.measure), "ulp");
}

#[test]
fn two_relative_bounds_compound() {
    let first = NumericContract::of(ScalarFormat::F32).with_measure(ErrorMeasure::relative(0.01));
    let second = NumericContract::of(ScalarFormat::F32).with_measure(ErrorMeasure::relative(0.02));
    let composed = first.compose(&second).expect("one format composes");
    let ErrorMeasure::Relative { bits } = composed.measure else {
        panic!("Fix: composing two relative bounds states a relative bound");
    };
    let expected = 0.01_f64.mul_add(0.02, 0.01 + 0.02);
    assert!((f64::from_bits(bits) - expected).abs() < 1e-12);
}

#[test]
fn an_absolute_bound_needs_a_magnitude_to_meet_a_relative_one() {
    let absolute = NumericContract::of(ScalarFormat::F32).with_measure(ErrorMeasure::absolute(0.5));
    let relative =
        NumericContract::of(ScalarFormat::F32).with_measure(ErrorMeasure::relative(0.01));
    assert!(matches!(
        absolute.compose(&relative),
        Err(ContractRefusal::UnboundedMagnitude { .. })
    ));
    let both = absolute
        .compose(&absolute)
        .expect("two absolute bounds add");
    assert_eq!(both.measure, ErrorMeasure::absolute(1.0));
    assert_eq!(describe(both.measure), "absolute");
}

#[test]
fn an_exact_stage_leaves_the_other_measure_alone() {
    let exact = NumericContract::of(ScalarFormat::F32);
    let bounded = NumericContract::of(ScalarFormat::F32).within_ulp(4);
    assert_eq!(
        exact
            .clone()
            .with_measure(ErrorMeasure::Exact)
            .compose(&bounded)
            .expect("composition holds")
            .measure,
        ErrorMeasure::Ulp { count: 4 }
    );
    assert_eq!(describe(ErrorMeasure::Exact), "exact");
}

#[test]
fn composition_keeps_the_weaker_promise() {
    let ordered = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(Reassociation::Forbidden)
        .under(Determinism::RunToRunVariable)
        .sensitive_to(AtomicOrderSensitivity::Sensitive);
    let free = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(Reassociation::WithinBudget);
    for (first, second, order) in [
        (&free, &ordered, "free then ordered"),
        (&ordered, &free, "ordered then free"),
    ] {
        let composed = first.compose(second).expect("one format composes");
        assert_eq!(
            composed.reassociation,
            Reassociation::Forbidden,
            "{order}: one ordered region orders the composition"
        );
        assert_eq!(
            composed.determinism,
            Determinism::RunToRunVariable,
            "{order}: one variable region makes the composition variable"
        );
        assert_eq!(
            composed.atomic_order,
            AtomicOrderSensitivity::Sensitive,
            "{order}: one order-sensitive region makes the composition sensitive"
        );
    }
}

#[test]
fn an_approximate_instruction_contributes_its_error() {
    let plain = NumericContract::of(ScalarFormat::F32).within_ulp(1);
    let approximate = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .approximating(Approximation::Native {
            measure: ErrorMeasure::Ulp { count: 8 },
        });
    let composed = plain.compose(&approximate).expect("one format composes");
    assert_eq!(composed.measure, ErrorMeasure::Ulp { count: 9 });
    assert_eq!(
        composed.approximation,
        Approximation::Native {
            measure: ErrorMeasure::Ulp { count: 8 }
        }
    );
}

#[test]
fn a_sequential_reduction_costs_a_step_per_term() {
    let sequential = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(Reassociation::Forbidden);
    let over_eight = sequential.over_reduction(8).expect("a bound scales");
    assert_eq!(over_eight.measure, ErrorMeasure::Ulp { count: 7 });
}

#[test]
fn a_tree_reduction_costs_a_step_per_level() {
    let tree = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(Reassociation::WithinBudget);
    assert_eq!(
        tree.over_reduction(8).expect("a bound scales").measure,
        ErrorMeasure::Ulp { count: 3 }
    );
    assert_eq!(
        tree.over_reduction(1024).expect("a bound scales").measure,
        ErrorMeasure::Ulp { count: 10 }
    );
}

#[test]
fn a_narrower_accumulator_widens_the_reduction() {
    let mixed = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(Reassociation::WithinBudget)
        .accumulating_in(ScalarFormat::F16);
    let plain = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(Reassociation::WithinBudget);
    let mixed_error = mixed
        .over_reduction(1024)
        .expect("a bound scales")
        .relative_error()
        .expect("a ulp bound reads as a fraction");
    let plain_error = plain
        .over_reduction(1024)
        .expect("a bound scales")
        .relative_error()
        .expect("a ulp bound reads as a fraction");
    assert!(
        mixed_error > plain_error * 1000.0,
        "an f16 accumulator rounds 2^13 times coarser than f32 storage: \
         mixed {mixed_error} against plain {plain_error}"
    );
}

#[test]
fn a_declared_budget_refuses_a_wider_proof() {
    let declared = NumericContract::of(ScalarFormat::F32).within_ulp(4);
    declared
        .admits(&ErrorMeasure::Ulp { count: 4 })
        .expect("the declared bound admits itself");
    assert!(matches!(
        declared.admits(&ErrorMeasure::Ulp { count: 5 }),
        Err(ContractRefusal::BudgetExceeded { .. })
    ));
}

#[test]
fn an_ordered_contract_refuses_a_reordering_schedule() {
    let ordered = NumericContract::of(ScalarFormat::F32).within_ulp(1);
    assert!(matches!(
        ordered.permits_reassociation(),
        Err(ContractRefusal::ReassociationRefused {
            stated: Reassociation::Forbidden
        })
    ));
    NumericContract::of(ScalarFormat::U32)
        .permits_reassociation()
        .expect("an exact combine reorders freely");
}

#[test]
fn a_contract_refuses_a_format_the_next_region_does_not_read() {
    let produced = NumericContract::of(ScalarFormat::F32).within_ulp(1);
    let consumed = NumericContract::of(ScalarFormat::U32);
    assert!(matches!(
        produced.compose(&consumed),
        Err(ContractRefusal::FormatMismatch { .. })
    ));
}

/// A codebook format has no uniform step, so a ULP bound over it has no
/// relative reading.
///
/// WHY: NF4 stores quantiles, not a mantissa. Reading a ULP count over it as a
/// fraction of the magnitude would invent a step the format does not have, and
/// every budget composed through that value would be a number nothing measured.
#[test]
fn a_ulp_bound_over_a_codebook_format_refuses_a_relative_reading() {
    let codebook = NumericContract::of(ScalarFormat::NF4).within_ulp(1);
    assert!(ScalarFormat::NF4.ulp_fraction().is_none());
    assert_eq!(
        codebook.relative_error(),
        Err(ContractRefusal::UnboundedMagnitude {
            measure: ErrorMeasure::Ulp { count: 1 },
        })
    );

    let stepped = NumericContract::of(ScalarFormat::F16).within_ulp(1);
    assert!(stepped.relative_error().is_ok_and(|error| error > 0.0));
}
