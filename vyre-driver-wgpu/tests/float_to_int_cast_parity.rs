//! Float→integer cast parity against the reference oracle (Rust saturating `as`).
//!
//! The reference oracle (`vyre-reference/src/execution/expr_cast.rs`) lowers
//! `Cast{F32→U32/I32}` with Rust's `as`, which is SATURATING since Rust 1.45:
//!   * in-range          → truncate toward zero
//!   * +∞ / overflow     → target MAX
//!   * −∞ / underflow     → target MIN (0 for unsigned)
//!   * **NaN             → 0**
//!
//! vyre-emit-naga emits a bare `As{Uint/Sint}`; naga's SPIR-V backend wraps it
//! in `FClamp(x, min, max)` then `ConvertFToU/S`, so ±∞ and out-of-range
//! saturate correctly. BUT `FClamp(NaN, …)` reduces to FMin/FMax of a NaN,
//! whose result is UNDEFINED per the SPIR-V GLSL.std.450 spec (it may return
//! min, max, or NaN by hardware). The output is an integer, so the differential
//! harness's ULP tolerance cannot mask a divergence. A legal kernel that casts a
//! possibly-NaN float (e.g. `(0.0/0.0) as u32`) can therefore silently disagree
//! with the oracle on the GPU — the div-by-zero / shift-mask silent-divergence
//! class (Law 10).
//!
//! This test dispatches the cast on the live GPU and asserts byte-for-byte
//! equality with the saturating reference, NaN included.

mod common;
use common::u32_bytes;

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_driver_wgpu::WgpuBackend;

/// The probe inputs: normal, out-of-range, and the IEEE specials.
fn cast_inputs() -> Vec<f32> {
    vec![
        42.9,
        -5.0,
        3.0e20,
        -3.0e20,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]
}

/// Build a kernel that loads each f32, casts it to `target`, and stores the
/// result word-for-word into `out`.
fn cast_program(target: DataType, n: u32) -> Program {
    let mut body = Vec::new();
    for i in 0..n {
        body.push(Node::store(
            "out",
            Expr::u32(i),
            Expr::cast(target.clone(), Expr::load("input", Expr::u32(i))),
        ));
    }
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::ReadWrite, target).with_count(n),
            BufferDecl::storage("input", 1, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        ],
        [1, 1, 1],
        body,
    )
}

fn run_cast(backend: &WgpuBackend, target: DataType) -> Vec<u32> {
    let inputs = cast_inputs();
    let n = inputs.len() as u32;
    let program = cast_program(target, n);
    // Pack the raw f32 bit patterns into the input buffer.
    let input_bytes = u32_bytes(&inputs.iter().map(|f| f.to_bits()).collect::<Vec<_>>());
    let out_init = u32_bytes(&vec![0u32; inputs.len()]);
    let outputs = backend
        .dispatch_borrowed(
            &program,
            &[out_init.as_slice(), input_bytes.as_slice()],
            &DispatchConfig::default(),
        )
        .expect("Fix: WGPU must dispatch the float→int cast parity contract.");
    // One output buffer of `n` u32 words.
    let bytes = &outputs[0];
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn f32_to_u32_cast_saturates_like_reference_including_nan() {
    let backend = WgpuBackend::acquire()
        .expect("Fix: float→int cast parity requires a live GPU backend.");
    let gpu = run_cast(&backend, DataType::U32);

    // Reference oracle = Rust saturating `as` (f32 widened to f64, then `as u32`).
    let expected: Vec<u32> = cast_inputs()
        .iter()
        .map(|&f| f64::from(f) as u32)
        .collect();

    // Pin the contract literally so a silent reference change can't hide a drift:
    // [42, 0(neg→0), MAX(overflow), 0(neg overflow→0), MAX(+∞), 0(−∞→0), 0(NaN)].
    assert_eq!(
        expected,
        vec![42, 0, u32::MAX, 0, u32::MAX, 0, 0],
        "reference saturating semantics drifted"
    );
    assert_eq!(
        gpu, expected,
        "GPU f32→u32 cast diverged from the saturating reference.\n  inputs:   {:?}\n  expected: {:?}\n  gpu:      {:?}\n\
         The NaN slot (last) is the likely diverger: FClamp(NaN) is SPIR-V-undefined.",
        cast_inputs(),
        expected,
        gpu
    );
}

#[test]
fn f32_to_i32_cast_saturates_like_reference_including_nan() {
    let backend = WgpuBackend::acquire()
        .expect("Fix: float→int cast parity requires a live GPU backend.");
    let gpu_bits = run_cast(&backend, DataType::I32);
    let gpu: Vec<i32> = gpu_bits.iter().map(|&w| w as i32).collect();

    let expected: Vec<i32> = cast_inputs()
        .iter()
        .map(|&f| f64::from(f) as i32)
        .collect();

    // [42, -5, MAX(overflow), MIN(neg overflow), MAX(+∞), MIN(−∞), 0(NaN)].
    assert_eq!(
        expected,
        vec![42, -5, i32::MAX, i32::MIN, i32::MAX, i32::MIN, 0],
        "reference saturating semantics drifted"
    );
    assert_eq!(
        gpu, expected,
        "GPU f32→i32 cast diverged from the saturating reference.\n  inputs:   {:?}\n  expected: {:?}\n  gpu:      {:?}",
        cast_inputs(),
        expected,
        gpu
    );
}
