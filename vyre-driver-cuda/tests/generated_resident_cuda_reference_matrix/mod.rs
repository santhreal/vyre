//! Generated live CUDA-resident/reference differential matrix for release-path semantics.

#![cfg(feature = "device-tests")]

#[path = "../harness/mod.rs"]
mod harness;

mod case_defs;
mod case_tables;
mod generated_f32;
mod program_builders;
mod resident_reference;

mod atomic_cast_contracts;
mod bool_contracts;
mod f32_contracts;
mod integer_contracts;
mod memory_contracts;
use harness::{
    assert_f32_output_lanes, assert_u32_output_lanes, bool_bytes, bool_word, eq_word, f32_bytes,
    ge_word, generated_bool_cast_values, generated_f32_cast_values, generated_f32_fma_values,
    generated_i32_cast_values, generated_mixed_bool_values as generated_bool_values,
    generated_mixed_u32_values as generated_atomic_values, generated_u32_cast_values, gt_word,
    i32_bytes, le_word, live_backend, lt_word, ne_word, reference_outputs,
    resident_cuda_reference_outputs, u32_bytes, GENERATED_LANE_COUNT as LANE_COUNT,
    GENERATED_WORKGROUP_SIZE_X as WORKGROUP_SIZE_X,
};
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

const OUTPUT_BYTES: usize = LANE_COUNT * std::mem::size_of::<u32>();
const BUCKET_COUNT: usize = 8;
const BUCKET_MASK: u32 = BUCKET_COUNT as u32 - 1;
const MAX_F32_ULP: u32 = 1;

use case_defs::*;
use case_tables::*;
use generated_f32::*;
use program_builders::*;
use resident_reference::*;

/// One case in a resident matrix sweep.
struct ResidentMatrixCase {
    /// Case name, reported by every diagnostic the sweep emits.
    name: &'static str,
    /// Program under test.
    program: Program,
    /// Bindings in declaration order.
    inputs: Vec<Vec<u8>>,
}

/// Diff every case against the reference interpreter on the resident release
/// path, then prove the sweep compared every lane of every case.
///
/// The lane total is asserted rather than reported because
/// [`assert_u32_output_lanes`] returns the number of lanes it actually compared:
/// handed a truncated or empty output it compares zero and returns zero, which
/// would leave a sweep green while checking nothing.
fn assert_resident_u32_sweep(
    backend: &CudaBackend,
    fix: &str,
    cases: impl ExactSizeIterator<Item = ResidentMatrixCase>,
) {
    let expected_lanes = cases.len() * LANE_COUNT;
    let mut checked_lanes = 0usize;
    for case in cases {
        let outputs = resident_cuda_reference_outputs(
            backend,
            &case.program,
            &case.inputs,
            &[OUTPUT_BYTES],
            case.name,
        );
        checked_lanes += assert_u32_output_lanes(
            case.name,
            LANE_COUNT,
            &outputs.resident_cuda,
            &outputs.reference,
        );
    }
    assert_eq!(checked_lanes, expected_lanes, "{fix}");
}

/// [`assert_resident_u32_sweep`] for f32 outputs, compared with the strict edge
/// semantics [`assert_f32_output_lanes`] fixes at `max_ulp`.
fn assert_resident_f32_sweep(
    backend: &CudaBackend,
    fix: &str,
    max_ulp: u32,
    cases: impl ExactSizeIterator<Item = ResidentMatrixCase>,
) {
    let expected_lanes = cases.len() * LANE_COUNT;
    let mut checked_lanes = 0usize;
    for case in cases {
        let outputs = resident_cuda_reference_outputs(
            backend,
            &case.program,
            &case.inputs,
            &[OUTPUT_BYTES],
            case.name,
        );
        checked_lanes += assert_f32_output_lanes(
            case.name,
            LANE_COUNT,
            max_ulp,
            &outputs.resident_cuda,
            &outputs.reference,
        );
    }
    assert_eq!(checked_lanes, expected_lanes, "{fix}");
}
