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
use vyre_reference::value::Value;

// ---------------------------------------------------------------------------
// Lattice Join (bitwise OR)
// ---------------------------------------------------------------------------

#[path = "contract_cases/algebra_lattice_semiring_contracts__lattice_join_specific_values.rs"]
mod algebra_lattice_semiring_contracts_lattice_join_specific_values;
#[path = "contract_cases/algebra_lattice_semiring_contracts__semiring_min_plus_mul_zero_is_identity.rs"]
mod algebra_lattice_semiring_contracts_semiring_min_plus_mul_zero_is_identity;
