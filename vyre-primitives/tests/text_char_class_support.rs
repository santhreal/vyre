//! Shared executable helpers for packed character-class integration tests.

use vyre_primitives::text::char_class_u8;
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

/// Execute the packed-byte character classifier and decode exactly one class per input byte.
pub(crate) fn run_packed_u8_program(source: &[u8], table: &[u32; 256]) -> Vec<u32> {
    let program = char_class_u8("source", "classified", source.len() as u32);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(source.to_vec()),
            Value::from(pack_u32_slice(table)),
        ],
    )
    .expect("Fix: packed-u8 char_class reference evaluation must succeed");
    let mut classified = decode_u32_le_bytes_all(&outputs[0].to_bytes());
    classified.truncate(source.len());
    classified
}
