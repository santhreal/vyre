//! SPIR-V backend module (C-B7).
//!
//! Emits the canonical `naga::Module` through `naga::back::spv::write_vec`
//! instead of `naga::back::wgsl::write_string`. Only the target writer changes.
//!
//! The backend runs through wgpu with the Vulkan adapter selected.
//! Testing byte-identity between the WGSL path (wgpu default) and
//! the SPIR-V path on the same Program is the conformance contract.
//!
//! This module ships:
//!
//! * `SpirvEmitter`  -  stateless helper that wraps
//!   `naga::back::spv::write_vec`.
//! * `SpirvBackend`  -  emission helper for Vulkan/SPIR-V validation.
//!   Live SPIR-V dispatch must acquire a concrete Vulkan backend or
//!   fail loudly with an actionable driver/probe error.
//!
//! The actual dispatch implementation shares machinery with the
//! WGPU dispatch via the driver-owned backend trait surface; this module
//! only needs to name itself and point wgpu at the Vulkan backend.

use naga::back::spv::{Options, PipelineOptions, WriterFlags};
use naga::valid::{Capabilities, ValidationFlags, Validator};

/// Stateless helper that emits SPIR-V bytecode from a validated
/// naga::Module.
pub struct SpirvEmitter;

impl SpirvEmitter {
    /// Validate + lower to SPIR-V. Returns the raw word stream
    /// ready for `wgpu::ShaderModule::create_shader_module` (with
    /// a SpirvRaw source) or equivalent.
    ///
    /// # Errors
    ///
    /// Returns a stringly-typed error containing the Fix: prose for
    /// any validation or emission failure. Callers wrap this in
    /// their local error type.
    pub fn emit(module: &naga::Module, entry: &str) -> Result<Vec<u32>, String> {
        // VYRE_NAGA_LOWER MEDIUM: the WGSL validator uses
        // `Capabilities::all()`. A Program that uses subgroup ops
        // passes WGSL validation but fails `Capabilities::empty()`
        // here, producing a split-brain "valid on WGSL, invalid on
        // SPIR-V" result even though the bytecode is from the same
        // naga::Module. Unify on `all()` so both back-ends accept
        // the same capability set. (Emission is still a best-effort
        // operation  -  naga rejects SPIR-V-incompatible constructs at
        // writer time with a specific error.)
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator.validate(module).map_err(|e| {
            format!(
                "SPIR-V emit: naga validation failed: {e}. Fix: reject the naga::Module and look for a malformed lowering."
            )
        })?;
        let options = Options::default();
        let pipeline = PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: entry.to_owned(),
        };
        let mut out = Vec::new();
        let mut writer = naga::back::spv::Writer::new(&options).map_err(|e| {
            format!(
                "SPIR-V emit: could not construct writer: {e}. Fix: upgrade naga or lower spv-out feature flags."
            )
        })?;
        writer
            .write(module, &info, Some(&pipeline), &None, &mut out)
            .map_err(|e| {
                format!(
                    "SPIR-V emit: writer.write failed: {e}. Fix: inspect the naga::Module for capabilities this adapter lacks."
                )
            })?;
        Ok(out)
    }

    /// Just the flags we expose; consumers inspect these at runtime
    /// to distinguish the emit variant.
    #[must_use]
    pub fn default_flags() -> WriterFlags {
        WriterFlags::empty()
    }
}

/// The SPIR-V backend identifier used by the dispatcher /
/// auto-picker.
pub const SPIRV_BACKEND_ID: &str = "spirv";
