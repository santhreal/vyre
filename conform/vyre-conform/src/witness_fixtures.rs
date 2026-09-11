//! Witness fixture shapes, backend input planning, and synthesis of witness bytes for
//! operations that ship no explicit fixture.

use vyre_conform::witness_plan::static_buffer_byte_len;

/// Per-case fixture bytes  -  one outer Vec per dispatch case, one
/// middle Vec per declared buffer, one inner Vec of raw byte content.
pub(crate) type FixtureCases = Vec<Vec<Vec<u8>>>;

/// Signature of the zero-argument closure a `SemanticOperation` ships as its
/// `test_inputs` / `expected_output` generator.
pub(crate) type FixtureFn = fn() -> FixtureCases;

pub(crate) fn synthesize_witness_cases(program: &vyre::Program) -> Result<FixtureCases, String> {
    let mut case = Vec::new();
    for buffer in program.buffers() {
        if buffer.kind() == vyre::ir::MemoryKind::Shared
            || buffer.access() == vyre::ir::BufferAccess::Workgroup
            || buffer.is_backend_allocated_output()
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
