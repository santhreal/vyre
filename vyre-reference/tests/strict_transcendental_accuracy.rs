//! Does the strict f32 expansion compute the function it claims to?
//!
//! `vyre-foundation/tests/strict_transcendental_expansion.rs` judges the shape
//! of an expansion: only correctly-rounded operations, no decimal f32 literal,
//! no approximable operation left behind. Shape alone would be satisfied by an
//! expansion that returns its argument, so the numbers are judged here, on the
//! oracle, with no device involved.
//!
//! Two bounds are asserted for every operation, and they measure different
//! things:
//!
//! - Against the correctly-rounded f32 result, computed in f64 and rounded
//!   once. This is the accuracy of the expansion itself and it is the same
//!   number on every host, because the f64 result is nearer the true value than
//!   half an f32 ulp by six orders of magnitude.
//! - Against `ieee754::canonical_*`, which is `libm`. This is the shipped
//!   parity envelope, and it is looser because `libm`'s f32 transcendentals are
//!   not correctly rounded either: it lands one ulp from the correctly-rounded
//!   result on a small fraction of inputs, which is measured in
//!   `gap_transcendentals_parity.rs` in the wgpu driver's tests. Bit identity
//!   with `libm` is unreachable by any independent implementation, however
//!   accurate, which is why the contract is a bound and not an equality.
//!
//! What this does not catch: whether a device agrees with the oracle on the
//! same expanded program. That is the point of the whole exercise and it needs
//! a device, so `vyre-driver-wgpu/tests/gap_transcendentals_parity.rs` owns it.

use vyre_foundation::fp_expansion::{
    strict_expansion, NAN_RESULT_BITS, SIN_COS_DOMAIN_BITS, SIN_COS_SMALL_ARGUMENT_BITS,
};
use vyre_foundation::fp_parity::{canonical_f32, REFERENCE_TRANSCENDENTAL_ULP_BUDGET};
use vyre_foundation::ir::{Expr, UnOp};
use vyre_reference::{
    ieee754::{
        canonical_cos, canonical_exp, canonical_log, canonical_sin, canonical_sqrt,
        canonical_ulp_distance,
    },
    reference_eval_expr,
    value::Value,
    workgroup::InvocationIds,
    ReferenceMemory,
};

/// Accuracy of the expansion against the correctly-rounded f32 result.
///
/// One ulp for the three monotone functions and two for the circular pair,
/// which is what the minimax fits in `fp_expansion` measure over their whole
/// domains. Widening a row here is a regression in the polynomial, not a
/// tolerance to be relaxed.
fn correctly_rounded_budget(op: &UnOp) -> u32 {
    match op {
        UnOp::Sin | UnOp::Cos => 2,
        _ => 1,
    }
}

fn expansion(op: &UnOp, input: f32) -> Expr {
    strict_expansion(op, &Expr::f32(input))
        .unwrap_or_else(|| panic!("Fix: {op:?} must have a strict expansion"))
}

/// Evaluate one expansion on the reference interpreter.
///
/// The interpreter rounds and canonicalizes every f32 operation separately,
/// which is the arithmetic the strict mode requires a device to perform, so
/// this is the same evaluation the parity claim compares a device against.
fn evaluate(expr: &Expr) -> f32 {
    let program = vyre_foundation::ir::Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
    let value = reference_eval_expr(&program, &mut ReferenceMemory::empty(), InvocationIds::ZERO, expr)
    .expect("Fix: the reference interpreter must evaluate an expanded transcendental");
    match value {
        Value::Float(inner) => inner as f32,
        other => panic!("Fix: an f32 expansion must evaluate to a float, got {other:?}"),
    }
}

fn expanded(op: &UnOp, input: f32) -> f32 {
    evaluate(&expansion(op, input))
}

fn canonical(op: &UnOp, input: f32) -> f32 {
    match op {
        UnOp::Sin => canonical_sin(input),
        UnOp::Cos => canonical_cos(input),
        UnOp::Sqrt => canonical_sqrt(input),
        UnOp::Exp => canonical_exp(input),
        UnOp::Log => canonical_log(input),
        other => panic!("Fix: only sin/cos/sqrt/exp/log are covered, got {other:?}"),
    }
}

/// The correctly-rounded f32 result: evaluated in f64, rounded once, with the
/// expansion's stated subnormal flush applied to both ends.
///
/// `fp_expansion` states two departures from the exact real function, and the
/// flush is one of them: a subnormal argument is answered as the zero of its
/// own sign, and a subnormal result flushes the same way. `canonical_f32` is
/// the definition of that flush and every f32 the parity contract compares
/// passes through it, so an oracle that skipped it would measure the expansion
/// against a number no side of the contract produces. `sin(0x00000003)` is
/// where that showed: the true sine of a subnormal is that subnormal, three ulp
/// from the zero both the expansion and `libm` answer with.
///
/// The flush itself is a contract, not a tolerance, so it is asserted directly
/// in `a_subnormal_argument_is_answered_as_its_signed_zero` rather than left to
/// this bound.
fn correctly_rounded(op: &UnOp, input: f32) -> f32 {
    let wide = f64::from(canonical_f32(input));
    let exact = match op {
        UnOp::Sin => wide.sin(),
        UnOp::Cos => wide.cos(),
        UnOp::Sqrt => wide.sqrt(),
        UnOp::Exp => wide.exp(),
        UnOp::Log => wide.ln(),
        other => panic!("Fix: only sin/cos/sqrt/exp/log are covered, got {other:?}"),
    };
    canonical_f32(exact as f32)
}

/// A deterministic spread over `[low, high]`, dense enough to reach every
/// branch of a reduction and reproducible on every host.
fn sweep(low: f32, high: f32, count: u32) -> Vec<f32> {
    // A 64-bit LCG, so the sample set is fixed by the constants rather than by
    // a proptest seed a failure could not be replayed from.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut samples = Vec::with_capacity(count as usize + 2);
    samples.push(low);
    samples.push(high);
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        samples.push((f64::from(low) + unit * f64::from(high - low)) as f32);
    }
    samples
}

/// The f32 values adjacent to `center`, which is where a reduction's error is
/// largest and where a lost bit of the argument shows up first.
fn neighbours(center: f32, radius: i32) -> Vec<f32> {
    (-radius..=radius)
        .map(|offset| f32::from_bits(center.to_bits().wrapping_add(offset as u32)))
        .filter(|value| value.is_finite())
        .collect()
}

/// Judge one operation on every argument, both against the correctly-rounded
/// result and against the oracle.
///
/// Subnormal arguments are rejected rather than skipped. The flush is a stated
/// property of the contract and it is asserted directly by
/// `a_subnormal_argument_is_answered_as_its_signed_zero`; measuring an ulp
/// distance across it would measure the flush, not the polynomial, and a caller
/// that let one through would be reading the wrong number.
fn assert_within_budget(op: &UnOp, inputs: &[f32]) {
    let exact_budget = correctly_rounded_budget(op);
    let mut worst_exact = (0u32, 0.0f32);
    let mut worst_oracle = (0u32, 0.0f32);
    for &input in inputs {
        assert!(
            input == 0.0 || input.is_normal(),
            "{input:e} is neither zero nor a normal f32, so an ulp distance on it measures the \
             subnormal flush rather than the accuracy of {op:?}"
        );
        let produced = expanded(op, input);
        let exact = canonical_ulp_distance(produced, correctly_rounded(op, input));
        let oracle = canonical_ulp_distance(produced, canonical(op, input));
        assert!(
            exact <= exact_budget,
            "the strict expansion of {op:?}({input:e}) produced {:#010x}, {exact} ulp from the \
             correctly-rounded {:#010x}; the polynomial is budgeted for {exact_budget}",
            produced.to_bits(),
            correctly_rounded(op, input).to_bits()
        );
        assert!(
            oracle <= REFERENCE_TRANSCENDENTAL_ULP_BUDGET,
            "the strict expansion of {op:?}({input:e}) produced {:#010x}, {oracle} ulp from the \
             oracle {:#010x}; REFERENCE_TRANSCENDENTAL_ULP_BUDGET is {}",
            produced.to_bits(),
            canonical(op, input).to_bits(),
            REFERENCE_TRANSCENDENTAL_ULP_BUDGET
        );
        if exact > worst_exact.0 {
            worst_exact = (exact, input);
        }
        if oracle > worst_oracle.0 {
            worst_oracle = (oracle, input);
        }
    }
    eprintln!(
        "{op:?}: {} inputs, worst {} ulp from correctly rounded at {:e}, worst {} ulp from the oracle at {:e}",
        inputs.len(),
        worst_exact.0,
        worst_exact.1,
        worst_oracle.0,
        worst_oracle.1
    );
}

/// `sin` and `cos` over the domain the proptests draw from, plus every f32
/// adjacent to a multiple of `pi/2`.
///
/// The neighbours are the adversarial set: near a zero of `sin` the result is
/// tiny, so a reduction that loses one bit of the argument loses every
/// significant bit of the answer. A random sweep almost never lands there.
#[test]
fn circular_expansions_hold_their_budget_including_next_to_every_quadrant_boundary() {
    let mut inputs = sweep(-10.0, 10.0, 600);
    for quadrant in -6..=6 {
        let boundary = (f64::from(quadrant) * core::f64::consts::FRAC_PI_2) as f32;
        inputs.extend(neighbours(boundary, 24));
    }
    // The neighbours of zero are subnormal, which the flush owns rather than
    // the polynomial.
    inputs.retain(|value| value.abs() <= 10.0 && (*value == 0.0 || value.is_normal()));
    assert_within_budget(&UnOp::Sin, &inputs);
    assert_within_budget(&UnOp::Cos, &inputs);
}

/// The circular expansions hold the same budget out to the edge of their stated
/// domain, where the reduction quotient is at its largest and the three-limb
/// split of `pi/2` is doing all of its work.
#[test]
fn circular_expansions_hold_their_budget_out_to_the_domain_edge() {
    let domain = f32::from_bits(SIN_COS_DOMAIN_BITS);
    let inputs = sweep(-domain, domain, 800);
    assert_within_budget(&UnOp::Sin, &inputs);
    assert_within_budget(&UnOp::Cos, &inputs);
}

/// Outside the stated domain the circular expansions produce one quiet NaN.
///
/// The alternative is a saturated reduction, which answers a larger argument
/// with the value at the boundary: a wrong number rather than a refused one,
/// and indistinguishable from a right one.
#[test]
fn a_circular_argument_outside_the_domain_is_refused_with_one_quiet_nan() {
    let domain = f32::from_bits(SIN_COS_DOMAIN_BITS);
    let inside = [domain, -domain];
    let outside = [
        f32::from_bits(SIN_COS_DOMAIN_BITS + 1),
        f32::from_bits((SIN_COS_DOMAIN_BITS + 1) | 0x8000_0000),
        1.0e6,
        -1.0e6,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for op in [UnOp::Sin, UnOp::Cos] {
        for input in inside {
            let produced = expanded(&op, input);
            assert!(
                produced.abs() <= 1.0,
                "{op:?}({input}) is inside the domain and must produce a value in [-1, 1], got {produced}"
            );
        }
        for input in outside {
            assert_eq!(
                expanded(&op, input).to_bits(),
                NAN_RESULT_BITS,
                "{op:?}({input}) is outside the stated domain and must produce exactly the one \
                 quiet NaN, so a device and the reference agree on the payload"
            );
        }
    }
}

/// Below `2^-12` the circular expansions return the argument and one.
///
/// The reduction cannot improve on `sin(x) = x` there, and routing a small
/// magnitude through it would round twice for nothing. This also keeps the sign
/// of a zero, which `sin` must.
#[test]
fn a_small_circular_argument_is_returned_unchanged() {
    let threshold = f32::from_bits(SIN_COS_SMALL_ARGUMENT_BITS);
    for input in [
        0.0f32,
        -0.0f32,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(SIN_COS_SMALL_ARGUMENT_BITS - 1),
        threshold / 2.0,
    ] {
        let sine = expanded(&UnOp::Sin, input);
        let expected = if input.abs() < f32::MIN_POSITIVE {
            // A subnormal flushes to a zero of its own sign, which is what both
            // sides of the parity contract do to it.
            f32::from_bits(input.to_bits() & 0x8000_0000)
        } else {
            input
        };
        assert_eq!(
            sine.to_bits(),
            expected.to_bits(),
            "sin({input:e}) must return the argument itself below the small-argument threshold"
        );
        assert_eq!(
            expanded(&UnOp::Cos, input).to_bits(),
            1.0f32.to_bits(),
            "cos({input:e}) must be exactly one below the small-argument threshold"
        );
    }
    // The threshold itself goes through the reduction, and must still be right.
    assert_within_budget(&UnOp::Sin, &[threshold]);
    assert_within_budget(&UnOp::Cos, &[threshold]);
}

/// `exp` over the domain the proptests draw from, and over the full range where
/// the result is a normal f32.
#[test]
fn the_exponential_expansion_holds_its_budget() {
    let mut inputs = sweep(-10.0, 10.0, 600);
    inputs.extend(sweep(-87.0, 88.0, 600));
    assert_within_budget(&UnOp::Exp, &inputs);
}

/// `exp` saturates at both ends instead of producing an infinity or a
/// subnormal mid-expression.
#[test]
fn the_exponential_expansion_saturates_at_both_ends() {
    let cases: [(f32, u32); 7] = [
        (0.0, 1.0f32.to_bits()),
        (-0.0, 1.0f32.to_bits()),
        (f32::INFINITY, f32::INFINITY.to_bits()),
        (f32::NEG_INFINITY, 0),
        (f32::MAX, f32::INFINITY.to_bits()),
        (f32::MIN, 0),
        (f32::NAN, NAN_RESULT_BITS),
    ];
    for (input, expected) in cases {
        assert_eq!(
            expanded(&UnOp::Exp, input).to_bits(),
            expected,
            "exp({input}) must produce {expected:#010x}"
        );
    }
    // The largest argument whose result is finite, and the smallest whose
    // result is a normal f32: on the wrong side of either boundary the shipped
    // envelope would compare a zero against a subnormal.
    let last_finite = expanded(&UnOp::Exp, 88.72283);
    assert!(
        last_finite.is_finite() && last_finite > 3.0e38,
        "exp just below the overflow boundary must be finite and near f32::MAX, got {last_finite:e}"
    );
    let first_normal = expanded(&UnOp::Exp, -87.33);
    assert!(
        first_normal >= f32::MIN_POSITIVE,
        "exp just above the underflow boundary must be a normal f32, got {first_normal:e}"
    );
    assert_eq!(
        expanded(&UnOp::Exp, -88.0).to_bits(),
        0,
        "below the underflow boundary the result flushes to +0, because a subnormal cannot be \
         bit-identical across a device that flushes it and a reference that does not"
    );
}

/// `log` over the domain the proptests draw from, and across the whole
/// exponent range where the exponent contribution is exact.
#[test]
fn the_logarithm_expansion_holds_its_budget() {
    let mut inputs = sweep(0.000_001, 10.0, 600);
    // Log-uniform over the normal range, so every exponent field is reached.
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..600 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        inputs.push(
            (f64::from(f32::MIN_POSITIVE).ln() * (1.0 - unit) + f64::from(f32::MAX).ln() * unit)
                .exp() as f32,
        );
    }
    inputs.extend(neighbours(1.0, 32));
    inputs.extend(neighbours(2.0, 8));
    inputs.extend(neighbours(core::f32::consts::SQRT_2, 8));
    inputs.retain(|value| value.is_normal() && *value > 0.0);
    assert_within_budget(&UnOp::Log, &inputs);
}

/// `log` answers its domain edges the way IEEE-754 requires.
#[test]
fn the_logarithm_expansion_answers_its_domain_edges() {
    let cases: [(f32, u32); 9] = [
        (1.0, 0),
        (0.0, f32::NEG_INFINITY.to_bits()),
        (-0.0, f32::NEG_INFINITY.to_bits()),
        (f32::from_bits(1), f32::NEG_INFINITY.to_bits()),
        (-1.0, NAN_RESULT_BITS),
        (f32::MIN, NAN_RESULT_BITS),
        (f32::NEG_INFINITY, NAN_RESULT_BITS),
        (f32::NAN, NAN_RESULT_BITS),
        (f32::INFINITY, f32::INFINITY.to_bits()),
    ];
    for (input, expected) in cases {
        assert_eq!(
            expanded(&UnOp::Log, input).to_bits(),
            expected,
            "log({input}) must produce {expected:#010x}"
        );
    }
}

/// `sqrt` over the domain the proptests draw from, plus a sweep of the reduced
/// range the iteration actually sees and one of the extremes the exponent
/// halving has to get right.
#[test]
fn the_square_root_expansion_holds_its_budget() {
    let mut inputs = sweep(0.0, 10.0, 600);
    inputs.extend(sweep(1.0, 4.0, 600));
    let mut state = 0x0F1E_2D3C_4B5A_6978u64;
    for _ in 0..400 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Every exponent field, so both the odd and the even reduction path is
        // exercised across the whole range.
        inputs.push(f32::from_bits(
            ((state >> 32) as u32 & 0x7FFF_FFFF).clamp(0x0080_0000, 0x7F7F_FFFF),
        ));
    }
    inputs.extend([f32::MIN_POSITIVE, f32::MAX, 1.0, 4.0, 0.25]);
    inputs.retain(|value| value.is_normal() && *value > 0.0);
    assert_within_budget(&UnOp::Sqrt, &inputs);
}

/// `sqrt` answers its domain edges the way IEEE-754 requires, including the
/// sign of a zero.
#[test]
fn the_square_root_expansion_answers_its_domain_edges() {
    let cases: [(f32, u32); 8] = [
        (0.0, 0),
        (-0.0, 0x8000_0000),
        (1.0, 1.0f32.to_bits()),
        (4.0, 2.0f32.to_bits()),
        (-1.0, NAN_RESULT_BITS),
        (f32::NEG_INFINITY, NAN_RESULT_BITS),
        (f32::NAN, NAN_RESULT_BITS),
        (f32::INFINITY, f32::INFINITY.to_bits()),
    ];
    for (input, expected) in cases {
        assert_eq!(
            expanded(&UnOp::Sqrt, input).to_bits(),
            expected,
            "sqrt({input}) must produce {expected:#010x}"
        );
    }
}

/// A subnormal argument is answered as the zero it flushes to.
///
/// Not an accuracy claim: it is the parity claim. A device flushes a subnormal
/// f32 and the reference canonicalizes one to a zero of its own sign, so an
/// expansion that read the exponent field of a subnormal would compute from a
/// value the arithmetic around it cannot hold.
#[test]
fn a_subnormal_argument_is_answered_as_its_signed_zero() {
    for bits in [0x0000_0001u32, 0x0040_0000, 0x007F_FFFF] {
        for sign in [0u32, 0x8000_0000] {
            let subnormal = f32::from_bits(bits | sign);
            let zero = f32::from_bits(sign);
            for op in [UnOp::Sin, UnOp::Cos, UnOp::Sqrt, UnOp::Exp, UnOp::Log] {
                assert_eq!(
                    expanded(&op, subnormal).to_bits(),
                    expanded(&op, zero).to_bits(),
                    "{op:?} must answer the subnormal {subnormal:e} exactly as it answers the zero \
                     it flushes to"
                );
            }
        }
    }
}
