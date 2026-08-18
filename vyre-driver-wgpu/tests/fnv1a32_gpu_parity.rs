//! FNV-1a 32-bit hash parity on the LIVE GPU, a real shipped vyre-primitives
//! workload whose GPU program is a LOOP with carried hash state.
//!
//! `vyre-primitives` tests `fnv1a32_program*` only through `reference_eval` (the
//! CPU oracle), exactly like BLAKE3 was before `blake3_compress_gpu_parity`. The
//! shape here is different and complementary: BLAKE3 is unrolled rounds, FNV is a
//! single `Node::Loop` that walks `input[0..n]` carrying the 32-bit hash across
//! iterations (`h = (h ^ (byte & 0xFF)) * FNV_PRIME`, a wrapping u32 multiply).
//! So this exercises loop-carried state + per-iteration `Mul`/`BitXor`/`BitAnd`
//! on real silicon end-to-end (none of which the unrolled BLAKE3 path covers).
//! If the loop carrier or the wrapping multiply miscompiled on the GPU, every
//! GPU FNV hash would be silently wrong and no test would catch it.
//!
//! Dispatched on the 5090 and asserted byte-for-byte against the `fnv1a32` Rust
//! reference (itself the proven contract, validated by
//! `vyre-primitives/tests/adversarial_hash.rs`).

mod harness;
use harness::{byte_stream_input_bytes, dispatch_single_u32_output, u32_bytes};

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::hash::fnv1a::fnv1a32_program;
use vyre_reference::composition_witness::multi_hash_witness;

fn reference_fnv1a32(bytes: &[u8]) -> u32 {
    multi_hash_witness(bytes).1
}
/// Dispatch the real `fnv1a32_program` on the GPU: one U32 word per source byte
/// (the builder masks each to its low 8 bits), single u32 hash out at `out[0]`.
fn gpu_fnv1a32(backend: &WgpuBackend, bytes: &[u8]) -> u32 {
    let n = bytes.len() as u32;
    let program = fnv1a32_program("input", "out", n);
    let input_b = byte_stream_input_bytes(bytes);
    let out_init = u32_bytes(&[0u32]);
    dispatch_single_u32_output(
        backend,
        &program,
        &[input_b.as_slice(), out_init.as_slice()],
        "Fix: WGPU must dispatch the FNV-1a32 loop program.",
    )
}

fn check(backend: &WgpuBackend, bytes: &[u8], label: &str) {
    let gpu = gpu_fnv1a32(backend, bytes);
    let expected = reference_fnv1a32(bytes);
    assert_eq!(
        gpu, expected,
        "GPU FNV-1a32 of {label} diverged from the Rust reference, the loop carrier \
         or wrapping multiply miscompiles on hardware.\n  gpu      = {gpu:#010x}\n  \
         expected = {expected:#010x}"
    );
}

#[test]
fn fnv1a32_abc_matches_reference_on_gpu() {
    let backend = WgpuBackend::acquire().expect("Fix: FNV GPU parity requires a live GPU.");
    // Drift-guard the reference against the published FNV-1a 32 vector for "abc".
    assert_eq!(
        reference_fnv1a32(b"abc"),
        0x1a47_e90b,
        "FNV-1a32 reference drifted for \"abc\""
    );
    check(&backend, b"abc", "\"abc\"");
}

#[test]
fn fnv1a32_varied_inputs_match_reference_on_gpu() {
    let backend = WgpuBackend::acquire().expect("Fix: FNV GPU parity requires a live GPU.");
    check(&backend, b"a", "\"a\"");
    check(&backend, b"hello, world", "\"hello, world\"");
    check(&backend, b"\x00\xff\x80\x7f\x01", "boundary bytes");
    // A 64-byte block exercises a longer loop with many carried iterations.
    let long: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(7));
    check(&backend, &long, "a 64-byte block");
}

#[test]
fn fnv1a32_distinguishes_inputs_on_gpu() {
    // Sanity that the GPU hash is actually data-dependent (not a constant): two
    // one-byte-different inputs must produce different GPU hashes, both matching
    // their references.
    let backend = WgpuBackend::acquire().expect("Fix: FNV GPU parity requires a live GPU.");
    let a = gpu_fnv1a32(&backend, b"hash-me-0");
    let b = gpu_fnv1a32(&backend, b"hash-me-1");
    assert_ne!(
        a, b,
        "FNV must distinguish one-byte-different inputs on the GPU"
    );
    assert_eq!(a, reference_fnv1a32(b"hash-me-0"));
    assert_eq!(b, reference_fnv1a32(b"hash-me-1"));
}
