//! Witness fixture shapes, backend input planning, and synthesis of witness bytes for
//! operations that ship no explicit fixture.

use vyre_conform::witness_plan::{
    plan_witness_inputs_into, static_buffer_byte_len, WitnessInputPlan,
};

/// Per-case fixture bytes  -  one outer Vec per dispatch case, one
/// middle Vec per declared buffer, one inner Vec of raw byte content.
pub(crate) type FixtureCases = Vec<Vec<Vec<u8>>>;

/// Signature of the zero-argument closure a `SemanticOperation` ships as its
/// `test_inputs` / `expected_output` generator.
pub(crate) type FixtureFn = fn() -> FixtureCases;

pub(crate) type BackendDispatchPlan = WitnessInputPlan;

pub(crate) fn backend_dispatch_plan(
    program: &vyre::Program,
) -> Result<BackendDispatchPlan, String> {
    WitnessInputPlan::for_program(program)
}

pub(crate) fn backend_dispatch_inputs_with_plan_into<'a>(
    fixture_inputs: &'a [Vec<u8>],
    plan: &'a BackendDispatchPlan,
    backend_inputs: &mut Vec<&'a [u8]>,
) -> Result<(), String> {
    plan_witness_inputs_into(fixture_inputs, plan, backend_inputs)
}

pub(crate) fn synthesize_witness_cases(program: &vyre::Program) -> Result<FixtureCases, String> {
    let mut case = Vec::new();
    for buffer in program.buffers() {
        if buffer.kind() == vyre::ir::MemoryKind::Shared
            || buffer.is_output()
            || (buffer.is_pipeline_live_out()
                && matches!(buffer.access(), vyre::ir::BufferAccess::ReadWrite))
        {
            continue;
        }
        let byte_len = static_buffer_byte_len(buffer, "synthetic witness buffer")?;
        if byte_len == 0 {
            return Err(format!(
                "missing test_inputs for dynamically sized buffer `{}`. Fix: provide explicit witness bytes because synthetic conformance cannot infer runtime length.",
                buffer.name()
            ));
        }
        case.push(synthetic_buffer_bytes(&buffer.element(), byte_len));
    }
    if case.is_empty() {
        return Err(
            "missing test_inputs and Program has no synthesizable input buffers. Fix: provide explicit witness bytes for this op."
                .to_string(),
        );
    }
    Ok(vec![case])
}

fn synthetic_buffer_bytes(element: &vyre::ir::DataType, byte_len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; byte_len];
    match element {
        vyre::ir::DataType::F32 => {
            for chunk in bytes.chunks_exact_mut(4) {
                chunk.copy_from_slice(&1.0f32.to_le_bytes());
            }
        }
        vyre::ir::DataType::F64 => {
            for chunk in bytes.chunks_exact_mut(8) {
                chunk.copy_from_slice(&1.0f64.to_le_bytes());
            }
        }
        vyre::ir::DataType::F16 | vyre::ir::DataType::BF16 => {
            for chunk in bytes.chunks_exact_mut(2) {
                chunk.copy_from_slice(&0x3c00u16.to_le_bytes());
            }
        }
        _ => {}
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre::ir::{BufferAccess, BufferDecl, DataType, Node, Program};

    #[test]
    fn backend_dispatch_plan_accepts_fixture_backed_runtime_sized_read_input() {
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::output("out", 1, DataType::U32).with_count(1),
            ],
            [1, 1, 1],
            Vec::<Node>::new(),
        );
        let plan = backend_dispatch_plan(&program)
            .expect("Fix: runtime-sized read-only buffers must be fixture-backed, not rejected.");
        let case = vec![vec![0xA5; 12]];
        let mut backend_inputs = Vec::new();

        backend_dispatch_inputs_with_plan_into(&case, &plan, &mut backend_inputs)
            .expect("Fix: concrete fixture bytes must satisfy a runtime-sized input buffer.");

        assert_eq!(
            backend_inputs,
            vec![case[0].as_slice()],
            "Fix: dynamic fixture-backed inputs must be passed through byte-exactly."
        );
    }

    #[test]
    fn backend_dispatch_plan_rejects_omitted_runtime_sized_read_write_input() {
        let program = Program::wrapped(
            vec![BufferDecl::storage(
                "scratch",
                0,
                BufferAccess::ReadWrite,
                DataType::U32,
            )],
            [1, 1, 1],
            Vec::<Node>::new(),
        );
        let plan = backend_dispatch_plan(&program)
            .expect("Fix: dynamic read-write buffers may be fixture-backed per case.");
        let mut backend_inputs = Vec::new();

        let error = backend_dispatch_inputs_with_plan_into(&[], &plan, &mut backend_inputs)
            .expect_err("Fix: omitted dynamic read-write input must not be silently zeroed.");

        assert!(
            error.contains("runtime-sized read-write buffer"),
            "Fix: error must explain that dynamic read-write buffers need concrete fixture bytes, got: {error}"
        );
    }
}
