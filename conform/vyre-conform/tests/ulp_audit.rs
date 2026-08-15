//! RELEASE TEST LANE 18  -  per-op F32 ULP audit.
//!
//! For every registered op whose fixtures contain F32 buffers:
//!   1. Dispatch the fixture through a linked dispatch-capable backend.
//!   2. Compute max ULP delta against CPU reference per output element.
//!   3. Assert delta ≤ the explicit F32 ULP budget for the program.
//!   4. Adversarial companion: feed finite normal values, signed zero, infinities,
//!      NaN, max finite, and denormals into every F32 input buffer. Finite normal
//!      companions assert the ULP bound. Architecture-edge companions assert
//!      successful dispatch and output shape while still recording observed ULP.
//!   5. Print a table of max-ULP-observed per op so regressions are visible.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre::ir::{DataType, Program};
use vyre_conform::production::ProductionSession;
use vyre_conform::witness_plan::{
    plan_witness_inputs_into, plan_witness_inputs_owned_into, WitnessInputPlan,
};
use vyre_foundation::fp_parity::{f32_ulp_tolerance, max_output_ulp};
use vyre_reference::value::Value;

type FixtureCases = Vec<Vec<Vec<u8>>>;
type FixtureFn = fn() -> FixtureCases;

struct UnifiedEntry {
    id: &'static str,
    build: Option<fn() -> Program>,
    test_inputs: Option<FixtureFn>,
    expected_output: Option<FixtureFn>,
}

impl UnifiedEntry {
    fn program(&self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }
}

fn all_entries() -> Vec<UnifiedEntry> {
    vyre_registry_link::operation::live_operation_registry()
        .iter()
        .map(|entry| UnifiedEntry {
            id: entry.id,
            build: entry.build,
            test_inputs: entry.test_inputs,
            expected_output: entry.expected_output,
        })
        .collect()
}

fn run_cpu_from_slices<'a>(
    program: &Program,
    inputs: &[&[u8]],
    values: &'a mut Vec<Value>,
    outputs: &'a mut Vec<Vec<u8>>,
) -> Result<&'a [Vec<u8>], String> {
    values.clear();
    for input in inputs {
        values.push(Value::from(*input));
    }
    let evaluated = vyre_reference::reference_eval(program, values).map_err(|e| e.to_string())?;
    outputs.clear();
    outputs.extend(evaluated.into_iter().map(|v| v.to_bytes()));
    Ok(outputs.as_slice())
}

fn backend_inputs_from_vectors<'a>(buffers: &'a [Vec<u8>], outputs: &mut Vec<&'a [u8]>) {
    outputs.clear();
    outputs.extend(buffers.iter().map(Vec::as_slice));
}

fn make_adversarial_inputs_into(
    base: &[Vec<u8>],
    program: &Program,
    input_indices: &[usize],
    value: f32,
    outputs: &mut Vec<Vec<u8>>,
) {
    if base.len() != input_indices.len() {
        panic!(
            "Fix: normalized adversarial input count {} does not match backend input count {}",
            base.len(),
            input_indices.len()
        );
    }
    outputs.clear();
    outputs.reserve(base.len());
    base.iter()
        .zip(input_indices.iter())
        .for_each(|(bytes, buffer_index)| {
            let decl = &program.buffers()[*buffer_index];
            let new = if decl.element() == DataType::F32 {
                let mut new = bytes.clone();
                assert_eq!(
                    new.len() % 4,
                    0,
                    "F32 buffer `{}` length {} not divisible by 4",
                    decl.name(),
                    new.len()
                );
                for chunk in new.chunks_exact_mut(4) {
                    chunk.copy_from_slice(&value.to_le_bytes());
                }
                new
            } else {
                bytes.clone()
            };
            outputs.push(new);
        });
}

const ADVERSARIAL_VALUES: &[f32] = &[
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::MIN_POSITIVE,
    f32::MAX,
    f32::from_bits(1),
    f32::from_bits(0x8000_0001),
    f32::from_bits(0x007f_ffff),
    f32::from_bits(0x807f_ffff),
];

fn adversarial_value_requires_ulp(value: f32) -> bool {
    value.is_finite() && value.abs() > f32::MIN_POSITIVE && value.abs() < f32::MAX
}

fn build_registered_backend() -> &'static vyre_driver::BackendRegistration {
    let selected = std::env::var("VYRE_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty());
    vyre_registry_link::backend::live_backend_registry()
        .expect("valid backend registry")
        .iter()
        .find(|registration| {
            // The reference oracle is what the audit compares against, so
            // auditing it against itself would measure zero ULP for every op.
            !registration.reference_oracle
                && vyre_driver::backend_dispatches(registration.id)
                    .expect("valid backend registry")
                && selected
                    .as_deref()
                    .is_none_or(|backend| registration.id == backend)
        })
        .expect(
            "Fix: a dispatch-capable backend must be registered for ULP audit. \
             Link a concrete driver crate into the test binary.",
        )
}

// ULP audit dispatches every registered op through a real dispatch-capable
// backend. Missing concrete GPU drivers must fail loudly instead of compiling
// this module out.
#[path = "contract_cases/ulp_audit__release_per_op_f32_ulp_audit.rs"]
mod ulp_audit_release_per_op_f32_ulp_audit;
