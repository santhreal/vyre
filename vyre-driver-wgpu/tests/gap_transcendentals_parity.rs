//! Bitwise CPU/GPU parity for the five f32 transcendentals.
//!
//! `BACKLOG.md` row 136 owns this contract. It was aspirational and every test
//! here was ignored, for three measured reasons:
//!
//! 1. WGSL hardware transcendentals are not correctly rounded. The spec defers
//!    to the hardware, which uses an approximation ROM good to a few ulps.
//! 2. Emitting a deterministic f32-only polynomial instead does not fix it on
//!    its own, because the WGSL backend CONTRACTS `a * b + c` into a fused
//!    multiply-add and a polynomial is nothing but a chain of multiply-adds.
//!    `f32_no_contraction_contract.rs` measures that directly: the device
//!    rounds once where the reference rounds twice.
//! 3. `vyre_reference::ieee754::canonical_*` is `libm`, and `libm` is not
//!    correctly rounded for f32 either. Measured over 600000 samples of
//!    `-10.0..10.0` by `the_reference_oracle_is_not_the_correctly_rounded_f32_sine`
//!    below, `libm::sinf` differs from the correctly-rounded f32 result by one
//!    ulp on a small fraction of them.
//!
//! The third reason is why the assertion had to change shape rather than merely
//! wait. Read as "the device must equal `libm` bit for bit" it is unreachable by
//! any independent implementation however accurate: bit identity with `libm`
//! means reproducing the f64 arithmetic it computes in, and WGSL has no f64. A
//! correctly-rounded expansion would disagree with the oracle on roughly two of
//! the 8000 samples one 1000-case run draws, so bit identity with the oracle and
//! numerical correctness are different targets and only the second is reachable.
//!
//! The contract enforced here is the reachable and stronger one. Under
//! `FloatLoweringMode::StrictIeee` the driver replaces every approximable f32
//! operation with the expansion in `vyre_foundation::fp_expansion`, which
//! contains only f32 add, subtract, multiply, minimum, maximum, comparison,
//! select, exact integer operations on exponent and mantissa fields, and
//! bit-preserving casts between f32 and u32. Every one of those is correctly
//! rounded, and the strict mode publishes each f32 product through an integer
//! reinterpretation so no adjacent add can absorb it. So:
//!
//! - The device and the reference interpreter, executing the same expanded
//!   program, must agree bit for bit. That is asserted below, and it is a
//!   stronger statement than the original one, because it holds for the whole
//!   program rather than for one operation whose oracle happens to agree.
//! - Accuracy stays tied to `libm` through
//!   `REFERENCE_TRANSCENDENTAL_ULP_BUDGET`, asserted here against the device
//!   result directly.
//!
//! Two departures are stated rather than hidden, and both are properties of the
//! expansion, documented on `vyre_foundation::fp_expansion`: a subnormal
//! argument flushes to a zero of its own sign, and `sin`/`cos` are defined for
//! `|x| <= 401` and produce one quiet NaN outside it. The domains drawn from
//! here are inside both.
//!
//! What this does not catch: whether the expansion is accurate anywhere the
//! proptests do not draw from. `vyre-reference/tests/strict_transcendental_accuracy.rs`
//! sweeps each full domain against the correctly-rounded result on the oracle,
//! with no device needed.

#![cfg(feature = "device-tests")]
#![cfg(feature = "parity-testing")]

use crate::harness::{bytes_f32, f32_bytes};

use proptest::prelude::*;
use std::sync::LazyLock;
use vyre::ir::{Expr, Program, UnOp};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::fp_expansion::expand_strict_transcendentals;
use vyre_foundation::fp_parity::{
    canonical_f32, FloatLoweringMode, BACKEND_TRANSCENDENTAL_ULP_BUDGET,
    REFERENCE_TRANSCENDENTAL_ULP_BUDGET,
};
use vyre_test_support::strict_float_programs::f32_lane_program;

static BACKEND: LazyLock<WgpuBackend> = LazyLock::new(|| {
    WgpuBackend::acquire()
        .expect("Fix: gap_transcendentals_parity requires a local GPU-backed wgpu backend")
});

fn backend() -> &'static WgpuBackend {
    &BACKEND
}

/// The dispatch that asks for one rounding per operation and no native
/// transcendental.
fn strict() -> DispatchConfig {
    // `DispatchConfig` is non-exhaustive, so a struct expression cannot name it
    // from outside `vyre-driver`. Mutating the default is the documented shape.
    let mut config = DispatchConfig::default();
    config.float_lowering = FloatLoweringMode::StrictIeee;
    config
}

/// `out[i] = op(in[i])`, one lane per element.
///
/// The same program is handed to the device and, after expansion, to the
/// reference. Nothing in it is specific to either side: that is what makes the
/// comparison a parity claim rather than two implementations of the same idea.
fn unary_program(op: UnOp, count: u32) -> Program {
    f32_lane_program(count, |index| Expr::UnOp {
        op,
        operand: Box::new(Expr::load("in", index.clone())),
    })
}

/// Whether this adapter's shader compiler preserves per-operation rounding.
///
/// The parity claim below is stated against a device that honors the strict
/// mode. Whether one does is not a property of the driver: the module leaves
/// the emitter contraction-free and the platform compiles it again, and a
/// platform running under relaxed floating-point rules folds the barrier,
/// substitutes approximate transcendentals, and reassociates the range
/// reduction. Measured on a Metal adapter: the multiply-add witness came back
/// fused and `exp(8.765743)` returned 4096.001 against a true 6410.825.
///
/// The backend answers from its own one-time measurement, so this reads the
/// shipped decision rather than taking a second one.
static STRICT_IS_HONORED: LazyLock<bool> =
    LazyLock::new(|| backend().honors_float_lowering(FloatLoweringMode::StrictIeee));

/// The contract an adapter that does not honor the strict mode must satisfy.
///
/// Two halves, because a bare refusal proves only that the driver declined. The
/// refusal itself is asserted, since a dispatch that answered a bit-identity
/// request with contracted arithmetic is exactly the defect this file exists to
/// catch and looks identical to a pass if the case simply returns. Then the
/// accuracy claim is made under the mode the refusal names: the default mode
/// reaches the device's native transcendental, whose contract is
/// [`BACKEND_TRANSCENDENTAL_ULP_BUDGET`] or [`absolute_allowance`] and not the
/// elementary budget, because a native `sin` is not a rounded multiply-add.
/// Measured on a Vulkan adapter, `sin(-6.0764728)` lands 14 ulp out and
/// `log(0.9228698)` 9, both inside that budget and both far outside the
/// elementary one. Asserting it here is what keeps this file a measurement on
/// every adapter rather than only on one that honors the strict mode.
fn assert_strict_is_refused_and_the_default_is_accurate(
    op: UnOp,
    inputs: &[f32],
) -> Result<(), TestCaseError> {
    let count = u32::try_from(inputs.len()).expect("Fix: lane count must fit in u32");
    let program = unary_program(op.clone(), count);
    let arguments = [f32_bytes(inputs), vec![0u8; inputs.len() * 4]];

    let Err(error) = backend().dispatch(&program, &arguments, &strict()) else {
        return Err(TestCaseError::fail(
            "the adapter reports that it does not honor the strict mode and then answered a \
             strict dispatch. One of the two is wrong, and a caller comparing those bits with \
             the oracle would be comparing contracted arithmetic",
        ));
    };
    let message = error.to_string();
    prop_assert!(
        message.contains(FloatLoweringMode::StrictIeee.cache_label())
            && message.contains(vyre_driver_wgpu::WGPU_BACKEND_ID),
        "a refusal must name the mode and the backend so a caller knows which to change: {message}"
    );

    let outputs = backend()
        .dispatch(&program, &arguments, &DispatchConfig::default())
        .map_err(|error| {
            TestCaseError::fail(format!(
                "the refusal directs a caller to the default mode, so that mode must answer: \
                 {error}"
            ))
        })?;
    let device = bytes_f32(&outputs[0]);
    prop_assert_eq!(
        device.len(),
        inputs.len(),
        "the device must return one result per lane"
    );
    let allowance = absolute_allowance(&op);
    for (x, gpu) in inputs.iter().zip(&device) {
        let oracle = canonical(&op, *x);
        let distance = ulp_distance(*gpu, oracle);
        let absolute = (gpu - oracle).abs();
        prop_assert!(
            distance <= BACKEND_TRANSCENDENTAL_ULP_BUDGET || absolute <= allowance,
            "{:?}({}) under the default mode produced {} ({:#010x}), {} ulp and {:e} absolute \
             from the oracle {} ({:#010x}); BACKEND_TRANSCENDENTAL_ULP_BUDGET is {} and the \
             absolute allowance for this operation is {:e}",
            op,
            x,
            gpu,
            gpu.to_bits(),
            distance,
            absolute,
            oracle,
            oracle.to_bits(),
            BACKEND_TRANSCENDENTAL_ULP_BUDGET,
            allowance
        );
    }
    Ok(())
}

/// The device result for the unexpanded program under the strict mode.
///
/// The driver expands it. Handing the device the original program is the point:
/// a caller asks for strict IEEE lowering, not for an expansion it wrote itself.
fn device_results(op: UnOp, inputs: &[f32]) -> Vec<f32> {
    let count = u32::try_from(inputs.len()).expect("Fix: lane count must fit in u32");
    let program = unary_program(op, count);
    let outputs = backend()
        .dispatch(
            &program,
            &[f32_bytes(inputs), vec![0u8; inputs.len() * 4]],
            &strict(),
        )
        .expect("Fix: a strict-mode transcendental program must dispatch");
    bytes_f32(&outputs[0])
}

/// The reference result for the expanded program.
fn reference_results(op: UnOp, inputs: &[f32]) -> Vec<f32> {
    let count = u32::try_from(inputs.len()).expect("Fix: lane count must fit in u32");
    let program = unary_program(op, count);
    let expanded = expand_strict_transcendentals(&program)
        .expect("Fix: the five row 136 operators must all have expansions")
        .expect("Fix: a program containing one of them must be rewritten");
    let values = vyre_reference::reference_inputs(
        &expanded,
        vec![f32_bytes(inputs), vec![0u8; inputs.len() * 4]],
    );
    let outputs = vyre_reference::reference_eval(&expanded, &values)
        .expect("Fix: the reference interpreter must evaluate an expanded program");
    bytes_f32(&outputs[0].to_bytes())
}

fn canonical(op: &UnOp, x: f32) -> f32 {
    use vyre_reference::ieee754::{
        canonical_cos, canonical_exp, canonical_log, canonical_sin, canonical_sqrt,
    };
    match op {
        UnOp::Sin => canonical_sin(x),
        UnOp::Cos => canonical_cos(x),
        UnOp::Sqrt => canonical_sqrt(x),
        UnOp::Exp => canonical_exp(x),
        UnOp::Log => canonical_log(x),
        other => panic!(
            "Fix: gap_transcendentals_parity only covers sin/cos/sqrt/exp/log, got {other:?}"
        ),
    }
}

/// Distance in units in the last place over the sign-magnitude ordering of f32.
///
/// A pair that is not comparable at all, a NaN against a number or two
/// infinities of different sign, returns `u32::MAX` so it can never read as a
/// near miss.
fn ulp_distance(left: f32, right: f32) -> u32 {
    if left.to_bits() == right.to_bits() {
        return 0;
    }
    if left.is_nan() || right.is_nan() || left.is_infinite() || right.is_infinite() {
        return u32::MAX;
    }
    let ordered = |value: f32| -> i64 {
        let bits = i64::from(value.to_bits());
        if value.is_sign_negative() {
            -(bits - i64::from(0x8000_0000u32))
        } else {
            bits
        }
    };
    u32::try_from((ordered(left) - ordered(right)).abs()).unwrap_or(u32::MAX)
}

/// Absolute error the device is permitted near a result of zero.
///
/// A ulp bound alone is not a contract any conforming device can meet. Ulp
/// width collapses toward a zero crossing, so the same absolute error reads as
/// a few ulp mid-range and hundreds next to a root: measured here,
/// `cos(7.7991967)` is 218 ulp out at an absolute error of 8.1e-7. The device
/// API states these two the way the hardware actually behaves, an absolute
/// bound near zero and a relative one away from it, so both halves are asserted
/// and a lane passes on whichever the argument falls under.
///
/// `sin` and `cos` carry 2^-11 over `[-pi, pi]` and no guarantee outside it;
/// the wider bound is applied everywhere rather than leaving the out-of-range
/// arguments the proptests draw unchecked. `log` carries 2^-21 near 1.0. `exp`
/// and `sqrt` have no zero crossing on the drawn domain and get no absolute
/// allowance, so they are held to the ulp budget alone.
fn absolute_allowance(op: &UnOp) -> f32 {
    match op {
        UnOp::Sin | UnOp::Cos => 4.882_812_5e-4,
        UnOp::Log => 4.768_371_5e-7,
        _ => 0.0,
    }
}

/// The whole contract for one operation and one batch of arguments.
///
/// Both halves are asserted on every lane. Bit identity is the strict-mode
/// claim; the ulp bound is the accuracy claim, and neither implies the other: an
/// expansion that returned its argument would satisfy the first and fail the
/// second, and a native transcendental would satisfy the second and fail the
/// first.
///
/// On an adapter that does not honor the strict mode there is no bit-identity
/// claim to make, so the contract becomes the refusal plus the accuracy of the
/// mode that refusal names. Neither half is skipped: the failure this file
/// exists to catch is a device answering a bit-identity request with contracted
/// arithmetic, and a bare return would let it through.
fn assert_strict_parity(op: UnOp, inputs: &[f32]) -> Result<(), TestCaseError> {
    // Both sides read the arguments the parity contract compares: a subnormal
    // flushes to a zero of its own sign before it reaches either.
    let inputs: Vec<f32> = inputs.iter().copied().map(canonical_f32).collect();
    if !*STRICT_IS_HONORED {
        return assert_strict_is_refused_and_the_default_is_accurate(op, &inputs);
    }
    let device = device_results(op.clone(), &inputs);
    let reference = reference_results(op.clone(), &inputs);
    prop_assert_eq!(
        device.len(),
        inputs.len(),
        "the device must return one result per lane"
    );
    prop_assert_eq!(
        reference.len(),
        inputs.len(),
        "the reference must return one result per lane"
    );
    for ((x, gpu), cpu) in inputs.iter().zip(&device).zip(&reference) {
        prop_assert_eq!(
            cpu.to_bits(),
            gpu.to_bits(),
            "{:?}({}) under FloatLoweringMode::StrictIeee: reference {} ({:#010x}) vs device {} \
             ({:#010x}) must be byte-identical. Both executed the same expanded program, built \
             only from correctly-rounded f32 operations and contraction-blocked, so a difference \
             is a real lowering fault. See this file's header.",
            op,
            x,
            cpu,
            cpu.to_bits(),
            gpu,
            gpu.to_bits()
        );
        let oracle = canonical(&op, *x);
        let distance = ulp_distance(*gpu, oracle);
        prop_assert!(
            distance <= REFERENCE_TRANSCENDENTAL_ULP_BUDGET,
            "{:?}({}) produced {} ({:#010x}), {} ulp from the oracle {} ({:#010x}); \
             REFERENCE_TRANSCENDENTAL_ULP_BUDGET is {}",
            op,
            x,
            gpu,
            gpu.to_bits(),
            distance,
            oracle,
            oracle.to_bits(),
            REFERENCE_TRANSCENDENTAL_ULP_BUDGET
        );
    }
    Ok(())
}

proptest! {
    // The sweep runs 1000 cases each. The count is declared here rather than
    // left to `PROPTEST_CASES` so a plain `cargo test` run is the proving run.
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    #[test]
    fn sin_bitwise_parity(xs in prop::collection::vec(-10.0f32..10.0f32, 1..=8)) {
        assert_strict_parity(UnOp::Sin, &xs)?;
    }

    #[test]
    fn cos_bitwise_parity(xs in prop::collection::vec(-10.0f32..10.0f32, 1..=8)) {
        assert_strict_parity(UnOp::Cos, &xs)?;
    }

    #[test]
    fn sqrt_bitwise_parity(xs in prop::collection::vec(0.0f32..10.0f32, 1..=8)) {
        assert_strict_parity(UnOp::Sqrt, &xs)?;
    }

    #[test]
    fn exp_bitwise_parity(xs in prop::collection::vec(-10.0f32..10.0f32, 1..=8)) {
        assert_strict_parity(UnOp::Exp, &xs)?;
    }

    #[test]
    fn log_bitwise_parity(xs in prop::collection::vec(0.000_001f32..10.0f32, 1..=8)) {
        assert_strict_parity(UnOp::Log, &xs)?;
    }
}

/// The default mode still reaches the device's native transcendental.
///
/// The strict mode is opt-in, and the native instruction carries its own
/// envelope: `BACKEND_ELEMENTARY_F32_ULP_BUDGET`, asserted against the
/// correctly-rounded oracle by the refusal path above. If the expansion ran
/// unconditionally that envelope would silently become a measurement of this
/// expansion instead. The witness is the argument domain: the expansion refuses
/// `|x| > 401` where a native `sin` answers it.
#[test]
fn the_default_mode_is_left_on_the_native_transcendental() {
    let outside = 1.0e6f32;
    let program = unary_program(UnOp::Sin, 1);
    let outputs = backend()
        .dispatch(
            &program,
            &[outside.to_bits().to_le_bytes().to_vec(), vec![0u8; 4]],
            &DispatchConfig::default(),
        )
        .expect("Fix: the default mode must dispatch a sin program");
    let native = bytes_f32(&outputs[0])[0];
    assert!(
        !native.is_nan(),
        "under the default mode sin({outside}) must reach the device's native instruction, which \
         answers an argument outside the strict expansion's stated domain. A NaN here means the \
         expansion ran without anyone asking for it."
    );
    if !*STRICT_IS_HONORED {
        // The default half above is the claim this test owns, and it held. The
        // strict half needs an adapter that accepts the mode, and this one
        // states it does not.
        return;
    }
    let strict_result = device_results(UnOp::Sin, &[outside]);
    assert!(
        strict_result[0].is_nan(),
        "under the strict mode sin({outside}) is outside the expansion's stated domain and must be \
         refused, not answered with the value at the domain boundary"
    );
}

/// The strict mode really is a different module, not the same one relabelled.
///
/// The mode participates in every cache key that admits an emitted artifact. If
/// it did not, a program compiled once under the default mode would be served to
/// a strict dispatch and this file would pass while measuring nothing. The
/// witness is the same one `f32_no_contraction_contract.rs` uses: an argument
/// where a fused multiply-add and two separate roundings differ.
#[test]
fn a_strict_dispatch_is_not_served_the_contracted_artifact() {
    const WITNESS: f32 = f32::from_bits(0x3F80_0800); // 1 + 2^-12, exact in f32.
    const SEPARATELY_ROUNDED: u32 = 0x3A00_0000; // 2^-11, two roundings.
    let program = f32_lane_program(1, |index| {
        Expr::add(
            Expr::mul(
                Expr::load("in", index.clone()),
                Expr::load("in", index.clone()),
            ),
            Expr::f32(-1.0),
        )
    });
    let run = |config: &DispatchConfig| {
        let outputs = backend()
            .dispatch(&program, &[f32_bytes(&[WITNESS]), vec![0u8; 4]], config)
            .expect("Fix: the multiply-add witness must dispatch");
        bytes_f32(&outputs[0])[0].to_bits()
    };
    // Order matters: the default mode runs first so a cache that ignored the
    // mode would answer the strict dispatch with the contracted artifact.
    let contracted = run(&DispatchConfig::default());
    if !*STRICT_IS_HONORED {
        // This adapter's platform compiler folds the barrier, which is the
        // measurement that made the mode refusable here. The cache-key claim is
        // then unobservable through a dispatch, and asserting the refusal is
        // what remains: a cache that ignored the mode would answer with the
        // contracted bits instead of refusing.
        let refused = backend()
            .dispatch(&program, &[f32_bytes(&[WITNESS]), vec![0u8; 4]], &strict())
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(
            refused.contains(FloatLoweringMode::StrictIeee.cache_label()),
            "an adapter that does not honor the strict mode must refuse it by name rather than \
             answer with the contracted artifact it produced {contracted:#010x} from; got \
             `{refused}`"
        );
        return;
    }
    let strict_bits = run(&strict());
    assert_eq!(
        strict_bits, SEPARATELY_ROUNDED,
        "under the strict mode the witness must round twice and produce {SEPARATELY_ROUNDED:#010x}, \
         got {strict_bits:#010x}. The default mode produced {contracted:#010x} on this device."
    );
}

/// Blocker 3, measured here instead of asserted in the header.
///
/// The assertions above no longer read "the device must equal `canonical_*` bit
/// for bit", and this is the measurement that decided that. `canonical_sin` is
/// `libm::sinf`, and a deterministic sweep of the same `-10.0..10.0` domain the
/// proptests draw from shows it is not the correctly-rounded f32 sine: it lands
/// one ulp away on a small fraction of inputs. The consequence is that an
/// expansion which IS correctly rounded would disagree with the oracle on those
/// inputs, so bit identity to the oracle and numerical correctness are different
/// targets and only one of them is reachable. Bitwise CPU/GPU parity is stated
/// against the reference executing the same expanded program.
///
/// Correct rounding is computed by evaluating in f64 and rounding once, which
/// is the correctly-rounded f32 result for every input where the f64 sine is
/// within half an f32 ulp of the true value, and that is every input here by
/// six orders of magnitude of margin.
///
/// What this does not catch: whether `libm`'s other four transcendentals are
/// correctly rounded. One counterexample class is enough to decide the shape of
/// the contract, and the strict expansion is held to a ULP envelope against all
/// five rather than to bit identity with any of them.
#[test]
fn the_reference_oracle_is_not_the_correctly_rounded_f32_sine() {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut mismatches = 0_u32;
    let mut worst_ulp = 0_u32;
    let samples = 600_000;
    for _ in 0..samples {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 11) as f64) / ((1_u64 << 53) as f64);
        let x = (unit * 20.0 - 10.0) as f32;
        let oracle = vyre_reference::ieee754::canonical_sin(x);
        let correctly_rounded = f64::from(x).sin() as f32;
        let distance = oracle.to_bits().abs_diff(correctly_rounded.to_bits());
        if distance != 0 {
            mismatches += 1;
            worst_ulp = worst_ulp.max(distance);
        }
    }
    assert!(
        mismatches > 0,
        "libm::sinf agreed with the correctly-rounded f32 sine on all {samples} \
         samples. If the oracle has become correctly rounded, the bitwise \
         contract in this file can be stated against it directly and row 136's \
         restatement of the CPU side needs revisiting. Record that decision \
         before relaxing this assertion."
    );
    assert_eq!(
        worst_ulp, 1,
        "the oracle deviates from correct rounding by {worst_ulp} ulp on \
         {mismatches} of {samples} samples. More than one ulp is an oracle \
         accuracy fault, not the rounding-tie effect this test records; \
         REFERENCE_TRANSCENDENTAL_ULP_BUDGET is the constant that governs it."
    );
}
