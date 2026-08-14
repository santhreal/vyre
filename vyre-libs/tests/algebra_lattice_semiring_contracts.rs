//! Contract tests for lattice/semiring algebra and diversity sketches.
//!
//! Covers: lattice_join, lattice_meet, semiring_min_plus_mul, sketch_mix.
//! Properties tested: specific value correctness, algebraic laws,
//! boundary behaviour (size-0, size-1, all-ones, all-zeros), saturation,
//! and builder error paths (aliasing names).
//!
//! GPU acquisition: none  -  every test routes through the reference
//! interpreter or Reference oracle paths only.

#![cfg(feature = "math-algebra")]
#![allow(deprecated)]
mod common;
use common::{decode_u32_words, u32_bytes};
use vyre_foundation::ir::Program;
use vyre_reference::value::Value;

/// The two-input, one-output reference evaluation every case in this suite runs.
///
/// `out_bytes` is the declared width of the output buffer, which varies per
/// case; the input packing and the unwrap do not.
fn eval_pair(program: &Program, a: &[u32], b: &[u32], out_bytes: usize) -> Vec<u8> {
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(u32_bytes(a)),
            Value::from(u32_bytes(b)),
            Value::from(vec![0u8; out_bytes]),
        ],
    )
    .unwrap();
    outputs[0].to_bytes()
}

/// The sketch mix the diversity cases predict against.
fn mix(mut h: u32) -> u32 {
    h = h.wrapping_add(!(h << 15));
    h ^= h >> 12;
    h = h.wrapping_add(h << 2);
    h ^= h >> 4;
    h = h.wrapping_mul(2057);
    h ^= h >> 16;
    h
}

// ---------------------------------------------------------------------------
// Lattice Join (bitwise OR)
// ---------------------------------------------------------------------------

#[path = "contract_cases/algebra_lattice_semiring_contracts__lattice_join_specific_values.rs"]
mod algebra_lattice_semiring_contracts_lattice_join_specific_values;
#[path = "contract_cases/algebra_lattice_semiring_contracts__semiring_min_plus_mul_zero_is_identity.rs"]
mod algebra_lattice_semiring_contracts_semiring_min_plus_mul_zero_is_identity;
