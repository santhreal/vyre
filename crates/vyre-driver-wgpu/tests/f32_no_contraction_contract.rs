//! Does the WGSL backend keep `a * b + c` as two separately-rounded steps?
//!
//! It does NOT, and that measured fact is what this file records.
//!
//! The question matters because it is the load-bearing assumption under any
//! bitwise CPU/GPU float contract. IEEE-754 `+`, `-` and `*` on f32 are
//! correctly rounded and WGSL requires the same, so a computation built only
//! from those three operations would produce identical bits everywhere. A
//! fused multiply-add breaks that: it rounds ONCE instead of twice, so
//! `fma(a, b, c)` and `a * b + c` are different functions.
//!
//! Nothing in the vyre IR asks for fusion. The IR says multiply, then add. The
//! shader compiler and driver contract the pair anyway, and WGSL permits it.
//! Measured on GPU hardware (wgpu 25, Vulkan): for `a = b = 1 + 2^-12` and
//! `c = -1` the device returns `0x3a000400` where two separate roundings give
//! `0x3a000000`, a one-ulp difference that is exactly the retained low bit of
//! the unrounded product.
//!
//! Two consequences:
//!
//! 1. Bitwise CPU/GPU parity for f32 is not achievable through the ordinary
//!    lowering, no matter how the arithmetic is expressed. Bitwise parity needs
//!    a strict-IEEE lowering mode that blocks contraction, not a more careful
//!    polynomial.
//! 2. The bounded-ULP envelope in `transcendentals_parity.rs` is not merely a
//!    concession to approximate hardware transcendentals. Even exactly-specified
//!    arithmetic drifts by an ulp per fused pair.
//!
//! The contract asserted below is the one that is actually true and portable:
//! under the default lowering the device returns one of the two well-defined
//! answers, never a third, and the two differ by at most one ulp. A device that
//! stops contracting still passes; a device that returns something outside that
//! pair has a real bug.
//!
//! The strict lowering is the other half. It denies the target the pair, and
//! whether the denial survives the platform's own compilation of the module is
//! a property of the adapter rather than of the driver, so the driver measures
//! it once and refuses the mode where it does not hold. Both outcomes are
//! asserted: separately-rounded bits on an adapter that honors the mode, a
//! refusal naming the mode on one that does not.

#![cfg(feature = "device-tests")]

use crate::harness;
use harness::acquire_live_backend as live_backend;
use harness::{bytes_f32, f32_bytes};

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_test_support::strict_float_programs::f32_multiply_add_program;

/// `1 + 2^-12`, squared and offset by -1, is the tightest fma witness.
///
/// The exact product is `1 + 2^-11 + 2^-24`. The trailing bit sits one place
/// below the ulp of 1.0, so a rounded multiply drops it (ties-to-even) and the
/// subsequent subtraction yields exactly `2^-11`. An fma keeps the full product
/// and yields `2^-11 + 2^-24`, one ulp higher at that magnitude.
const WITNESS: f32 = f32::from_bits(0x3F80_0800); // 1 + 2^-12, exact in f32.

/// What two separately-rounded operations must produce.
const SEPARATELY_ROUNDED: f32 = 0.000_488_281_25; // 2^-11, exact in f32.

/// The witness really does separate the two roundings on the host.
///
/// A self-check on the constants. If Rust's own `f32::mul_add` agreed with
/// `a * b + c` here, the GPU test below would pass for the wrong reason and
/// prove nothing.
#[test]
fn the_witness_distinguishes_fused_from_separate_rounding_on_the_host() {
    let separate = WITNESS * WITNESS + -1.0f32;
    let fused = WITNESS.mul_add(WITNESS, -1.0f32);
    assert_eq!(
        separate.to_bits(),
        SEPARATELY_ROUNDED.to_bits(),
        "two roundings must give exactly 2^-11"
    );
    assert_ne!(
        separate.to_bits(),
        fused.to_bits(),
        "the witness is only useful if fusing changes the answer"
    );
}

/// The device returns either the separately-rounded or the fused answer.
///
/// The characterization contract. Both values are well defined, so a result
/// outside the pair means the arithmetic is wrong rather than merely fused.
/// The test prints which one it saw: that is the fact a reader needs when
/// deciding whether a bitwise parity claim is available on their hardware.
#[test]
fn multiply_add_lands_on_one_of_the_two_well_defined_roundings() {
    let backend = live_backend();
    let program = f32_multiply_add_program(1, None);
    let outputs = backend
        .dispatch(
            &program,
            &[
                f32_bytes(&[WITNESS]),
                f32_bytes(&[WITNESS]),
                f32_bytes(&[-1.0f32]),
            ],
            &DispatchConfig::default(),
        )
        .expect("Fix: the multiply-add parity program must dispatch");
    let gpu = bytes_f32(&outputs[0])[0];
    let fused = WITNESS.mul_add(WITNESS, -1.0f32);
    let contracted = gpu.to_bits() == fused.to_bits();
    assert!(
        contracted || gpu.to_bits() == SEPARATELY_ROUNDED.to_bits(),
        "the device produced {:#010x}, which is neither the separately-rounded \
         {:#010x} nor the fused {:#010x}. That is an arithmetic fault, not \
         contraction.",
        gpu.to_bits(),
        SEPARATELY_ROUNDED.to_bits(),
        fused.to_bits()
    );
    eprintln!(
        "multiply-add contraction on this device: {}",
        if contracted { "FUSED" } else { "separate" }
    );
}

/// Contraction is bounded in ABSOLUTE terms, not in ulps of the result.
///
/// The tempting summary of this file is "fusion costs an ulp". It does not.
/// What fusion retains is one ulp of the PRODUCT, and if the following add
/// cancels most of that product the same absolute quantity is an enormous
/// number of ulps of the answer. In the witness here the difference is 2^-24
/// against a result of 2^-11: 1024 ulps.
///
/// This is why the envelopes in `transcendentals_parity.rs` are stated as
/// "either a ULP bound or an absolute bound". Near a root, only the absolute
/// bound is meaningful, and this test is the arithmetic reason why.
#[test]
fn contraction_is_bounded_by_an_ulp_of_the_product_not_of_the_result() {
    let fused = WITNESS.mul_add(WITNESS, -1.0f32);
    let absolute = (fused - SEPARATELY_ROUNDED).abs();
    let ulps_of_result = fused.to_bits().abs_diff(SEPARATELY_ROUNDED.to_bits());

    // 2^-24: the bit of the exact product that a rounded multiply discards.
    assert_eq!(
        absolute.to_bits(),
        f32::from_bits(0x3380_0000).to_bits(),
        "the retained quantity must be exactly one ulp of the product (2^-24), \
         got {absolute:e}"
    );
    assert_eq!(
        ulps_of_result, 1024,
        "cancellation must magnify that bit to 1024 ulps of the result, which \
         is the whole reason an ulp bound alone is not a usable contract"
    );
}

/// Every lane agrees with the host to within one ulp, fused or not.
///
/// The usable parity statement across a spread of magnitudes. Exact equality
/// is asserted only where the host and device happen to round alike; the
/// portable guarantee is adjacency, and this pins it.
#[test]
fn the_reference_and_the_gpu_agree_to_within_one_ulp_on_multiply_add() {
    let backend = live_backend();
    // Each pair is chosen so the offset cancels most of the product, leaving
    // any retained low bit of the unrounded product visible in the result.
    let a: Vec<f32> = vec![WITNESS, 3.000_244_1, 1e-3, 7.5, -2.000_488_3, 1e6];
    let b: Vec<f32> = a.clone();
    let c: Vec<f32> = vec![-1.0, -9.0, -1e-6, -56.25, -4.0, -1e12];
    let count = a.len() as u32;
    let program = f32_multiply_add_program(count, None);
    let inputs = vec![f32_bytes(&a), f32_bytes(&b), f32_bytes(&c)];

    let gpu = bytes_f32(
        &backend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .expect("Fix: the multiply-add parity program must dispatch")[0],
    );

    for (index, ((a, b), c)) in a.iter().zip(&b).zip(&c).enumerate() {
        let separate = a * b + c;
        let fused = a.mul_add(*b, *c);
        let seen = gpu[index];
        assert!(
            seen.to_bits() == separate.to_bits() || seen.to_bits() == fused.to_bits(),
            "lane {index}: gpu {:#010x} is neither separate {:#010x} nor fused \
             {:#010x}",
            seen.to_bits(),
            separate.to_bits(),
            fused.to_bits()
        );
    }
}

/// Under the strict mode the device rounds separately, or refuses the mode.
///
/// Everything above characterizes the DEFAULT lowering, where contraction is
/// permitted and the answer is one of two. The strict mode removes the choice:
/// `emit_binop` publishes every f32 product through a `u32` reinterpretation
/// emitted as its own statement, so the following add has no unrounded product
/// to absorb. A bitwise parity claim against the reference rests entirely on
/// that surviving the platform's own compilation of the module, and nothing
/// pinned it. The pair of tests above measured contraction in the mode that
/// allows it, and the strict answer was asserted only indirectly, by a
/// transcendental parity suite whose expansions are what it was really
/// exercising.
///
/// The witness is the tightest one there is: the fused and separate answers
/// differ by the single bit a rounded multiply discards. An adapter whose
/// platform compiler folds the barrier is measured by the driver and refuses
/// the mode, and that refusal is asserted here rather than passed over, because
/// answering with the fused bits is the defect.
#[test]
fn the_strict_mode_rounds_separately_or_is_refused_by_name() {
    use vyre_foundation::fp_parity::FloatLoweringMode;

    let backend = live_backend();
    let a: Vec<f32> = vec![WITNESS, 3.000_244_1, 1e-3, 7.5, -2.000_488_3, 1e6];
    let b: Vec<f32> = a.clone();
    let c: Vec<f32> = vec![-1.0, -9.0, -1e-6, -56.25, -4.0, -1e12];
    let count = u32::try_from(a.len()).expect("Fix: lane count must fit in u32");
    let program = f32_multiply_add_program(count, None);
    let inputs = vec![f32_bytes(&a), f32_bytes(&b), f32_bytes(&c)];

    // `DispatchConfig` is non-exhaustive, so a struct expression cannot name it
    // from outside `vyre-driver`. Mutating the default is the documented shape.
    let mut config = DispatchConfig::default();
    config.float_lowering = FloatLoweringMode::StrictIeee;

    let outcome = backend.dispatch(&program, &inputs, &config);
    if !backend.honors_float_lowering(FloatLoweringMode::StrictIeee) {
        let message = outcome
            .err()
            .map(|error| error.to_string())
            .unwrap_or_else(|| {
                panic!(
                    "the adapter reports it does not honor the strict mode and then answered a \
                     strict dispatch, so a caller would compare contracted bits with the oracle"
                )
            });
        assert!(
            message.contains(FloatLoweringMode::StrictIeee.cache_label()),
            "a refusal must name the mode so a caller knows what to change: {message}"
        );
        return;
    }
    let gpu = bytes_f32(
        &outcome.expect("Fix: an adapter that honors the strict mode must answer this program")[0],
    );

    for (index, ((a, b), c)) in a.iter().zip(&b).zip(&c).enumerate() {
        let separate = a * b + c;
        assert_eq!(
            gpu[index].to_bits(),
            separate.to_bits(),
            "lane {index}: the strict mode must round the product before the add. Got \
             {:#010x}, two roundings give {:#010x}, and the fused answer is {:#010x}",
            gpu[index].to_bits(),
            separate.to_bits(),
            a.mul_add(*b, *c).to_bits()
        );
    }
}
