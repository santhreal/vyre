//! Shared f32 parity policy for reusable conform lenses.
//!
//! Integer and boolean buffers compare byte-identical. F32 buffers compare
//! under the program's ULP policy because GPU backends are allowed to use
//! native transcendental approximations while the CPU reference uses a
//! deterministic libm oracle.

pub use vyre_foundation::fp_parity::{
    compare_output_buffers, effective_tolerance, f32_buffer_matches, f32_ulp_tolerance,
    ulp_distance, BufferParity, BACKEND_ELEMENTARY_F32_ULP_BUDGET,
    BACKEND_TRANSCENDENTAL_ULP_BUDGET, REFERENCE_TRANSCENDENTAL_ULP_BUDGET,
};
