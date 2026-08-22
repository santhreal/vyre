//! Flat byte adapter that turns every CPU reference into a uniform byte-in,
//! byte-out contract.
//!
//! The parity engine compares raw bytes, not structured values. This module
//! exists so primitive ops can be tested with the same binary-diff harness
//! regardless of their internal Value representation.

use vyre_foundation::ir::{BufferAccess, DataType, Program};

use crate::reference_eval;
use crate::value::Value;
/// Execute a program from a concatenated single-case byte payload.
///
/// Every buffer accepted by [`crate::is_reference_input`] consumes exactly its
/// declared static element count from `input` (`count == 0` is treated as one
/// element for runtime-sized flat cases). A backend-allocated output takes no
/// payload bytes; the interpreter zero-fills it and its result is appended to
/// `output` after interpretation. Extra or truncated input bytes are rejected
/// so malformed conformance vectors cannot be hidden by padding or ignored
/// suffixes.
///
/// # Errors
///
/// Returns [`crate::ReferenceError`] if the program is invalid or execution fails.
///
/// # Examples
///
/// ```rust,ignore
/// let mut out = Vec::new();
/// vyre_reference::flat_cpu::run_flat(&program, &input_bytes, &mut out)?;
/// ```
pub fn run_flat(
    program: &Program,
    input: &[u8],
    output: &mut Vec<u8>,
) -> Result<(), crate::ReferenceError> {
    let mut offset = 0usize;
    let mut values = Vec::new();
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        // Every declared buffer must have a fixed width, whoever allocates it:
        // a variable-width output has no size for the readback either.
        let width = buffer_flat_width(buffer.name(), buffer.element(), buffer.count())?;
        // Only a buffer the artifact ABI reads consumes payload bytes. A
        // backend-allocated output is zero-filled by the interpreter.
        if !crate::is_reference_input(buffer) {
            continue;
        }
        let remaining = input.len().saturating_sub(offset);
        if remaining < width {
            return Err(crate::ReferenceError::new(format!(
                "flat CPU input for buffer `{}` is truncated: expected {width} byte(s), got {remaining}. Fix: provide the declared fixed-width element count for every buffer accepted by `vyre_reference::is_reference_input` before invoking the reference backend.",
                buffer.name()
            )));
        }
        let mut bytes = vec![0; width];
        bytes.copy_from_slice(&input[offset..offset + width]);
        offset += width;
        values.push(Value::from(bytes));
    }
    if offset != input.len() {
        let trailing = input.len() - offset;
        return Err(crate::ReferenceError::new(format!(
            "flat CPU input has {trailing} trailing byte(s) after consuming declared ReadOnly/Uniform buffers. Fix: provide exactly one fixed-width element per flat input buffer or split multi-case payloads before invoking the reference backend."
        )));
    }
    let values = reference_eval(program, &values)?;
    output.clear();
    for value in values {
        value.extend_bytes_width(0, output)?;
    }
    Ok(())
}

fn output_width(buffer_name: &str, data_type: DataType) -> Result<usize, crate::ReferenceError> {
    let min_bytes = data_type.min_bytes();
    if min_bytes == 0 {
        return Err(crate::ReferenceError::new(format!(
            "flat CPU buffer `{buffer_name}` uses variable-width element type {data_type:?}. Fix: use a fixed-width element type or route dynamic buffers through the structured reference evaluator."
        )));
    }
    Ok(min_bytes.max(4))
}

fn buffer_flat_width(
    buffer_name: &str,
    data_type: DataType,
    count: u32,
) -> Result<usize, crate::ReferenceError> {
    output_width(buffer_name, data_type)?
        .checked_mul(count.max(1) as usize)
        .ok_or_else(|| {
            crate::ReferenceError::new(
                "flat CPU buffer byte width overflows usize. Fix: split the flat conformance case or reduce the declared buffer count.",
            )
        })
}
