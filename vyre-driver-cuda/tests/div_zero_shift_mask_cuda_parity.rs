//! Div/mod-by-zero and oversized-shift parity against the reference oracle on the
//! live CUDA device, the "undefined on hardware, TOTAL on the oracle" class
//! (Law 10), the PTX/CUDA twin of the wgpu `div_zero_shift_mask_parity` gate.
//!
//! The oracle defines three hardware-undefined ops with a single total contract:
//!   * `u32 x / 0`  -> `u32::MAX`   (oracle `div_u32`; PTX forces it with
//!                                   `emit_total_u32_div`: default 0xffffffff +
//!                                   `@pred bra` over the `div`)
//!   * `u32 x % 0`  -> `0`          (oracle `rem_u32`; PTX `emit_total_u32_mod`)
//!   * `u32 x << s` / `x >> s`, `s >= 32` -> amount taken `& 31` (oracle
//!                                   `shift_u32`; PTX masks with `and.b32 ...,31`)
//!
//! The generated scalar matrix already exercises the zero divisor (`lane % 13 ==
//! 0 => 0`), but it PRE-MASKS shift amounts (`RhsKind::Shift => value & 31`), so
//! it NEVER sends `s >= 32` and the PTX `and.b32 ...,31` mask is unverified on
//! hardware, exactly the source-read-vs-silicon gap the naga signed-`Modulo`
//! miscompile punished. These tests dispatch all three with oversized amounts and
//! literal-pinned zero-divisor sentinels, byte-for-byte against the oracle.
//!
//! (Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined, so
//! they are not emittable and not tested here; only the unsigned total cases.)

mod common;
use common::live_backend;

use vyre_driver::parity_harness::u32_binop_parity;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::Expr;

/// `out[i] = build(a[i], b[i])` over u32 buffers, dispatched on `ps`.
fn dispatch(backend: &CudaBackend, build: fn(Expr, Expr) -> Expr, ps: &[(u32, u32)]) -> Vec<u32> {
    u32_binop_parity(
        &|program, inputs| backend.dispatch_borrowed(program, inputs, &DispatchConfig::default()),
        build,
        ps,
        "div-zero / shift-mask parity",
    )
}

/// Divisor cases including the zero-divisor sentinels and normal control values.
fn div_cases() -> Vec<(u32, u32)> {
    vec![
        (10, 0),
        (0, 0),
        (u32::MAX, 0),
        (123, 0),
        (100, 7),
        (0, 5),
        (u32::MAX, 1),
        (4096, 4096),
    ]
}

#[test]
fn u32_div_by_zero_yields_max_on_cuda() {
    let backend = live_backend();
    let ps = div_cases();
    let gpu = dispatch(&backend, Expr::div, &ps);
    let expected: Vec<u32> = ps
        .iter()
        .map(|&(a, b)| if b == 0 { u32::MAX } else { a / b })
        .collect();
    // Pin the contract literally: every zero divisor -> u32::MAX, never `a`.
    assert_eq!(
        expected,
        vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX, 14, 0, u32::MAX, 1],
        "reference u32 div-by-zero contract drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32 `/ 0` diverged from the oracle (`x / 0 == u32::MAX`).\n  \
         cases: {ps:?}\n  expected: {expected:?}\n  gpu: {gpu:?}"
    );
}

#[test]
fn u32_mod_by_zero_yields_zero_on_cuda() {
    let backend = live_backend();
    let ps = div_cases();
    let gpu = dispatch(&backend, Expr::rem, &ps);
    let expected: Vec<u32> = ps
        .iter()
        .map(|&(a, b)| if b == 0 { 0 } else { a % b })
        .collect();
    assert_eq!(
        expected,
        vec![0, 0, 0, 0, 2, 0, 0, 0],
        "reference u32 mod-by-zero contract drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32 `% 0` diverged from the oracle (`x % 0 == 0`).\n  cases: {ps:?}\n  \
         expected: {expected:?}\n  gpu: {gpu:?}"
    );
}

/// (value, shift-amount), the amounts >= 32 exercise the `& 31` masking that a
/// non-masking lowering would get wrong (e.g. `1 << 32` would be 0, not 1).
fn shift_cases() -> Vec<(u32, u32)> {
    vec![
        (1, 0),
        (1, 31),
        (1, 32),
        (1, 33),
        (0xFF, 36),
        (0xFFFF_FFFF, 32),
        (0x1, 63),
        (0xDEAD_BEEF, 4),
    ]
}

#[test]
fn u32_oversized_shift_left_masks_amount_on_cuda() {
    let backend = live_backend();
    let ps = shift_cases();
    let gpu = dispatch(&backend, Expr::shl, &ps);
    // Oracle `shift_u32`: left << (right & 31). wrapping_shl masks identically.
    let expected: Vec<u32> = ps.iter().map(|&(v, s)| v.wrapping_shl(s)).collect();
    // Load-bearing oversized cases: 1<<32 -> 1 (NOT 0), 0xFFFFFFFF<<32 -> 0xFFFFFFFF,
    // 1<<33 -> 2, 1<<63 -> 0x80000000.
    assert_eq!(
        expected,
        vec![
            1,
            0x8000_0000,
            1,
            2,
            0xFF0,
            0xFFFF_FFFF,
            0x8000_0000,
            0xEADB_EEF0
        ],
        "reference u32 oversized shift-left contract drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32 `<<` with amount >= 32 diverged from the oracle (`<< (s & 31)`). A \
         non-masking shift would zero `1 << 32`.\n  cases: {ps:?}\n  expected: {expected:?}\n  \
         gpu: {gpu:?}"
    );
}

#[test]
fn u32_oversized_shift_right_masks_amount_on_cuda() {
    let backend = live_backend();
    let ps = shift_cases();
    let gpu = dispatch(&backend, Expr::shr, &ps);
    let expected: Vec<u32> = ps.iter().map(|&(v, s)| v.wrapping_shr(s)).collect();
    // 1>>32 -> 1 (32&31==0), 0xFFFFFFFF>>32 -> 0xFFFFFFFF, 0xFF>>36 -> 0xFF>>4 == 0xF.
    assert_eq!(
        expected,
        vec![1, 0, 1, 0, 0xF, 0xFFFF_FFFF, 0, 0x0DEA_DBEE],
        "reference u32 oversized shift-right contract drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32 `>>` with amount >= 32 diverged from the oracle (`>> (s & 31)`).\n  \
         cases: {ps:?}\n  expected: {expected:?}\n  gpu: {gpu:?}"
    );
}
