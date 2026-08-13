#![allow(
    clippy::doc_lazy_continuation,
    clippy::double_must_use,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::match_like_matches_macro,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::nonminimal_bool,
    clippy::derivable_impls
)]
//! SPIR-V binary emitter for vyre `KernelDescriptor`.
//!
//! Substrate parity strategy: route the descriptor through
//! `vyre-emit-naga` to get a `naga::Module`, then use
//! `naga::back::spv::Writer` to produce a SPIR-V binary. This shares
//! the lossless lowering work with the wgpu/naga path  -  both backends
//! see the exact same naga::Module  -  and avoids forking a second
//! KernelOp → SPIR-V translation table.
//!
//! ## Op coverage
//!
//! Inherits from `vyre-emit-naga`. Anything that emit-naga refuses
//! also fails here. Anything emit-naga accepts gets converted to a
//! valid SPIR-V binary if naga's spv-out can lower it (essentially
//! everything except SPIR-V-specific extensions naga doesn't model).
//!
//! ## Validation gate
//!
//! `naga::valid::Validator` runs before `naga::back::spv::Writer`, so
//! any invalid module is rejected at the boundary. The emitted SPIR-V
//! binary is guaranteed to satisfy SPIR-V's structural requirements
//! per naga's spec compliance. Optional external `spirv-val` validation
//! sits in the integration-test surface (added when CI has spirv-tools).

use thiserror::Error;
use vyre_lower::KernelDescriptor;

pub mod patterns;

/// Errors produced while lowering and encoding a SPIR-V module.
#[derive(Debug, Error)]
pub enum EmitError {
    /// Shared descriptor-to-Naga lowering failed.
    #[error("naga emission failed: {0}")]
    NagaEmit(#[from] vyre_emit_naga::EmitError),

    /// Naga rejected the generated module during validation.
    #[error("naga validation failed: {0}")]
    NagaValidation(String),

    /// The SPIR-V writer could not be constructed for the module.
    #[error("SPIR-V writer construction failed: {0}")]
    WriterConstruction(String),

    /// The SPIR-V writer could not encode the module.
    #[error("SPIR-V writer.write failed: {0}")]
    WriterWrite(String),
}

/// Emit a SPIR-V binary from a `KernelDescriptor`.
///
/// Returns the raw SPIR-V words as a `Vec<u32>` (the canonical SPIR-V
/// representation per the spec). Callers that need bytes can convert
/// via `bytemuck::cast_slice` or by writing each word as little-endian
/// (SPIR-V is host-endian per spec but most consumers expect LE).
pub fn emit(desc: &KernelDescriptor) -> Result<Vec<u32>, EmitError> {
    let module = vyre_emit_naga::emit(desc).map_err(EmitError::NagaEmit)?;
    emit_from_naga_module(&module)
}

/// Emit SPIR-V only when `target` supports every descriptor requirement.
///
/// Capability admission is delegated to the Naga emitter, the single owner of
/// descriptor-to-Naga translation, before the shared module is encoded.
///
/// # Errors
///
/// Returns [`EmitError::NagaEmit`] with a stable unsupported-capability
/// diagnostic when the supplied target cannot admit the descriptor.
pub fn emit_with_capabilities(
    desc: &KernelDescriptor,
    target: &vyre_lower::EmissionTargetCapabilities,
) -> Result<Vec<u32>, EmitError> {
    let module =
        vyre_emit_naga::emit_with_capabilities(desc, target).map_err(EmitError::NagaEmit)?;
    emit_from_naga_module(&module)
}

/// Lower-level entry: emit SPIR-V from a pre-built naga::Module.
/// Useful when callers want to apply naga-level analyses or rewrites
/// between `vyre-emit-naga::emit` and SPIR-V conversion.
pub fn emit_from_naga_module(module: &naga::Module) -> Result<Vec<u32>, EmitError> {
    use naga::back::spv::{Options, PipelineOptions, Writer, WriterFlags};
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .map_err(|e| EmitError::NagaValidation(format!("{e:?}")))?;

    let options = Options {
        lang_version: (1, 3),
        capabilities: None,
        flags: WriterFlags::empty(),
        binding_map: Default::default(),
        zero_initialize_workgroup_memory:
            naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        force_loop_bounding: true,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        debug_info: None,
    };
    let pipeline = PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };

    let mut writer = Writer::new(&options).map_err(|e| {
        EmitError::WriterConstruction(format!(
            "{e:?}. Fix: upgrade naga or relax spv-out feature flags."
        ))
    })?;
    let mut out = Vec::new();
    writer
        .write(module, &info, Some(&pipeline), &None, &mut out)
        .map_err(|e| EmitError::WriterWrite(format!("{e:?}")))?;
    Ok(out)
}

/// Convenience: emit SPIR-V as raw little-endian bytes (the form most
/// runtime loaders accept directly).
pub fn emit_bytes(desc: &KernelDescriptor) -> Result<Vec<u8>, EmitError> {
    words_to_le_bytes(emit(desc)?)
}

fn words_to_le_bytes(words: Vec<u32>) -> Result<Vec<u8>, EmitError> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    Ok(bytes)
}

/// SPIR-V magic number  -  `0x07230203` per the spec. Useful for
/// integration tests and consumer-side sanity checks.
pub const SPIRV_MAGIC: u32 = 0x07230203;

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::DataType;
    use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
    use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

    fn one_store_kernel() -> KernelDescriptor {
        descriptor("store_one")
            .slot(global_rw(0, DataType::U32, "out"))
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops([lit(0, 0), lit(1, 1), effect(KernelOpKind::StoreGlobal, [0, 0, 1])])
                    .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
            )
            .build()
    }

    #[test]
    fn empty_kernel_emits_valid_spirv_with_magic_header() {
        let desc = descriptor("empty").dispatch(64, 1, 1).build();
        let words = emit(&desc).unwrap();
        assert!(!words.is_empty());
        assert_eq!(
            words[0], SPIRV_MAGIC,
            "first word must be the SPIR-V magic number"
        );
    }

    #[test]
    fn one_store_kernel_emits_non_trivial_spirv() {
        let words = emit(&one_store_kernel()).unwrap();
        assert!(
            words.len() > 16,
            "real kernel should produce more than the header"
        );
        assert_eq!(words[0], SPIRV_MAGIC);
    }

    #[test]
    fn emit_bytes_matches_words_in_le() {
        let desc = descriptor("empty").dispatch(64, 1, 1).build();
        let words = emit(&desc).unwrap();
        let bytes = emit_bytes(&desc).unwrap();
        assert_eq!(bytes.len(), words.len() * 4);
        let first_word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(first_word, SPIRV_MAGIC);
    }

    #[test]
    fn emit_with_unsupported_op_propagates_naga_error() {
        let desc = descriptor("bad")
            .body(
                body()
                    .ops([
                        op(KernelOpKind::SubgroupReduce {
                        op: vyre_lower::SubgroupReduceOp::Add,
                    }, [0], 0),
                    ]),
            )
            .build();
        let r = emit(&desc);
        assert!(matches!(r, Err(EmitError::NagaEmit(_))));
    }

    #[test]
    fn binop_add_emits_valid_spirv() {
        let kernel = descriptor("add")
            .body(
                body()
                    .ops([
                        lit(0, 0),
                        lit(1, 1),
                        op(KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add), [0, 1], 2),
                    ])
                    .literals([LiteralValue::U32(3), LiteralValue::U32(4)]),
            )
            .build();
        let words = emit(&kernel).unwrap();
        assert_eq!(words[0], SPIRV_MAGIC);
        assert!(words.len() > 16);
    }

    #[test]
    fn spirv_magic_constant_matches_spec() {
        assert_eq!(SPIRV_MAGIC, 0x0723_0203);
    }

    #[test]
    fn emit_from_naga_module_independently_consumable() {
        // Build a valid naga::Module via emit-naga, then convert.
        let module = vyre_emit_naga::emit(&descriptor("k").build())
        .unwrap();
        let words = emit_from_naga_module(&module).unwrap();
        assert_eq!(words[0], SPIRV_MAGIC);
    }
}
