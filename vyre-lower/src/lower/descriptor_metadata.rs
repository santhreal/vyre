//! Buffer-to-binding metadata: memory class, binding visibility, and the
//! descriptor identity a lowered program carries.

use crate::descriptor::{BindingVisibility, MemoryClass};
use crate::error::LowerError;
use vyre_foundation::ir::{BufferAccess, BufferDecl, MemoryKind, Program};

pub(crate) fn memory_class(buffer: &BufferDecl) -> Result<MemoryClass, LowerError> {
    match (buffer.kind, &buffer.access) {
        (MemoryKind::Persistent, _) => Err(LowerError::UnsupportedConstruct(format!(
            "Persistent memory buffer `{}` cannot be lowered as a direct GPU binding. Fix: stage Persistent data through the host transfer path using AsyncLoad/AsyncStore into Global/Readonly memory before concrete GPU emission.",
            buffer.name()
        ))),
        (MemoryKind::Shared, _) | (_, BufferAccess::Workgroup) => Ok(MemoryClass::Shared),
        (MemoryKind::Local, _) => Ok(MemoryClass::Scratch),
        (MemoryKind::Uniform | MemoryKind::Push, _) | (_, BufferAccess::Uniform) => {
            Ok(MemoryClass::Uniform)
        }
        (MemoryKind::Readonly, _) | (_, BufferAccess::ReadOnly) => Ok(MemoryClass::Constant),
        (MemoryKind::Global, _) => Ok(MemoryClass::Global),
        (other, _) => Err(LowerError::UnsupportedConstruct(format!(
            "MemoryKind::{other:?} for buffer `{}` is not supported by neutral lowering. Fix: map the buffer to Global, Shared, Uniform, Readonly, Push, or Local before emission.",
            buffer.name()
        ))),
    }
}

pub(super) fn binding_visibility(access: &BufferAccess) -> BindingVisibility {
    match access {
        BufferAccess::ReadOnly | BufferAccess::Uniform => BindingVisibility::ReadOnly,
        BufferAccess::WriteOnly => BindingVisibility::WriteOnly,
        _ => BindingVisibility::ReadWrite,
    }
}

pub(super) fn fingerprint_id(program: &Program) -> String {
    // Direct hex table lookup is ~100x faster than per-byte write!() with
    // formatter dispatch. fingerprint is a fixed 32 bytes, so the output
    // is exactly 64 hex chars.
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let fingerprint = program.fingerprint();
    let mut out = String::with_capacity(fingerprint.len() * 2);
    for &byte in fingerprint.iter() {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
