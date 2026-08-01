//! Release gap #1: the aspirational bitwise transcendental contract.
//!
//! Every test in this file is `#[ignore]`d, deliberately and with a stated
//! reason. They are not dead: they are the exact assertions that must pass the
//! day gap #1 closes, kept compiling and runnable (`cargo test -- --ignored`)
//! so the contract cannot rot while it waits.
//!
//! Two independent blockers stand between here and a green run, both recorded
//! in BACKLOG.md under R65:
//!
//! 1. WGSL hardware transcendentals are not correctly rounded. The spec defers
//!    to the hardware, which uses an approximation ROM good to a few ulps.
//!    `vyre_reference::ieee754::canonical_*` is `libm`, which is correctly
//!    rounded, so the two cannot agree bit for bit.
//! 2. The obvious remedy, emitting a deterministic f32-only polynomial on both
//!    sides, does not work either. The WGSL backend CONTRACTS `a * b + c` into
//!    a fused multiply-add, and a polynomial is nothing but a chain of
//!    multiply-adds. See `f32_no_contraction_contract.rs`, which measures this
//!    directly: the device rounds once where the reference rounds twice.
//!
//! Closing the gap therefore needs a strict-IEEE lowering mode that blocks
//! contraction, plus f32/u32 bitcast ops in the IR so an expansion can touch
//! exponent fields at all. Until then the shipped contract is the bounded
//! envelope in `transcendentals_parity.rs`, which is enforced on every run.

#![cfg(feature = "parity-testing")]

use proptest::prelude::*;
use std::sync::OnceLock;
use vyre::ir::UnOp;
use vyre_driver_wgpu::WgpuBackend;

fn backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire()
            .expect("Fix: gap_transcendentals_parity requires a local GPU-backed wgpu backend")
    })
}

fn gpu_unary_many(backend: &WgpuBackend, op: UnOp, xs: &[f32]) -> Vec<f32> {
    backend
        .probe_op_many(op, xs)
        .expect("Fix: wgpu f32 unary batch probe must dispatch successfully")
}

fn cpu_canonical(op: &UnOp, x: f32) -> f32 {
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

fn assert_bitwise_parity(op: UnOp, x: f32, gpu: f32) {
    let cpu = cpu_canonical(&op, x);
    assert_eq!(
        cpu.to_bits(),
        gpu.to_bits(),
        "gap_transcendentals_parity: {op:?}({x}) cpu={cpu} ({:#010x}) vs gpu={gpu} ({:#010x}) \
         must be byte-identical per contracts/release.md gap #1",
        cpu.to_bits(),
        gpu.to_bits()
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    #[test]
    #[ignore = "release gap #1: WGSL transcendentals are not correctly rounded and the backend contracts multiply-add. See BACKLOG.md R65 and f32_no_contraction_contract.rs."]
    fn sin_bitwise_parity(xs in prop::collection::vec(-10.0f32..10.0f32, 1..=8)) {
        let gpu = gpu_unary_many(backend(), UnOp::Sin, &xs);
        for (x, gpu) in xs.into_iter().zip(gpu) {
            assert_bitwise_parity(UnOp::Sin, x, gpu);
        }
    }

    #[test]
    #[ignore = "release gap #1: WGSL transcendentals are not correctly rounded and the backend contracts multiply-add. See BACKLOG.md R65 and f32_no_contraction_contract.rs."]
    fn cos_bitwise_parity(xs in prop::collection::vec(-10.0f32..10.0f32, 1..=8)) {
        let gpu = gpu_unary_many(backend(), UnOp::Cos, &xs);
        for (x, gpu) in xs.into_iter().zip(gpu) {
            assert_bitwise_parity(UnOp::Cos, x, gpu);
        }
    }

    #[test]
    #[ignore = "release gap #1: WGSL transcendentals are not correctly rounded and the backend contracts multiply-add. See BACKLOG.md R65 and f32_no_contraction_contract.rs."]
    fn sqrt_bitwise_parity(xs in prop::collection::vec(0.0f32..10.0f32, 1..=8)) {
        let gpu = gpu_unary_many(backend(), UnOp::Sqrt, &xs);
        for (x, gpu) in xs.into_iter().zip(gpu) {
            assert_bitwise_parity(UnOp::Sqrt, x, gpu);
        }
    }

    #[test]
    #[ignore = "release gap #1: WGSL transcendentals are not correctly rounded and the backend contracts multiply-add. See BACKLOG.md R65 and f32_no_contraction_contract.rs."]
    fn exp_bitwise_parity(xs in prop::collection::vec(-10.0f32..10.0f32, 1..=8)) {
        let gpu = gpu_unary_many(backend(), UnOp::Exp, &xs);
        for (x, gpu) in xs.into_iter().zip(gpu) {
            assert_bitwise_parity(UnOp::Exp, x, gpu);
        }
    }

    #[test]
    #[ignore = "release gap #1: WGSL transcendentals are not correctly rounded and the backend contracts multiply-add. See BACKLOG.md R65 and f32_no_contraction_contract.rs."]
    fn log_bitwise_parity(xs in prop::collection::vec(0.000_001f32..10.0f32, 1..=8)) {
        let gpu = gpu_unary_many(backend(), UnOp::Log, &xs);
        for (x, gpu) in xs.into_iter().zip(gpu) {
            assert_bitwise_parity(UnOp::Log, x, gpu);
        }
    }
}
