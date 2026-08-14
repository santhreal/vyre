//! One owner for the target-compiler shell every backend needs.
//!
//! A target compiler is two things: a dialect that turns one selected lowering
//! into bytes, and a shell around it that holds the payload format, validates
//! the profile, walks the selected modules, infers the dispatch grid and copies
//! the canonical bindings through. Only the first is per-backend. The shell was
//! written once per backend and drifted only in the dialect name inside its
//! error strings, so `dup-scan` counted twenty-nine of spirv's seventy-six
//! target-compiler lines as duplicated against cuda, metal and wgpu.
//!
//! A backend now declares a [`TargetDialect`] and gets the shell. What it still
//! owns is the emit function and the numbers in the profile, which are facts
//! about the device, not plumbing.

use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, SelectedLowering, TargetCompileError,
    TargetCompiler, TargetPayload, TargetPayloadFormat, TargetProfile,
};

use crate::BackendError;

/// Target bytes for one selected lowering, plus the entry point that runs them.
///
/// The grid size, the workgroup size and the resource bindings are not here on
/// purpose: they follow from the selected lowering, so the shell derives them
/// and no dialect can disagree about them.
pub struct EmittedDialectModule {
    /// Entry point name inside the emitted module.
    pub entry_point: String,
    /// Emitted bytes in the dialect's own format.
    pub bytes: Vec<u8>,
    /// Dynamic shared memory the entry point requires, zero when it needs none.
    pub dynamic_shared_bytes: u32,
}

/// Emit one selected lowering as target bytes. The only per-backend half of a
/// target compiler.
pub type EmitSelected =
    fn(&SelectedLowering, &TargetProfile) -> Result<EmittedDialectModule, TargetCompileError>;

/// One backend's target dialect: its payload identity, its device limits and
/// its emitter.
pub struct TargetDialect {
    /// Backend identity reported in a compile failure.
    pub backend_id: &'static str,
    /// Dialect name as it appears in operator-facing messages, such as `PTX`.
    pub dialect: &'static str,
    /// Stable payload format identity, such as `ptx`.
    pub format: &'static str,
    /// Payload format version.
    pub format_version: u16,
    /// Compiler and materializer generation.
    pub generation: u64,
    /// Largest workgroup the target accepts, per dimension.
    pub max_workgroup_size: [u32; 3],
    /// Largest invocation count in one workgroup.
    pub max_invocations_per_workgroup: u32,
    /// Largest dynamic shared allocation the target accepts.
    pub max_dynamic_shared_bytes: u32,
    /// Subgroup width, zero when the target does not expose one.
    pub subgroup_size: u32,
    /// Dialect emitter.
    pub emit: EmitSelected,
}

impl TargetDialect {
    /// Build the validated compilation profile this dialect registers.
    pub fn profile(&self) -> Result<TargetProfile, BackendError> {
        TargetProfile::new(
            self.format,
            self.generation,
            self.max_workgroup_size,
            self.max_invocations_per_workgroup,
            self.max_dynamic_shared_bytes,
            self.subgroup_size,
        )
        .map_err(|error| self.invalid("profile", "repair the registered profile", &error))
    }

    /// Build the target compiler this dialect registers.
    pub fn compiler(&self) -> Result<Box<dyn TargetCompiler>, BackendError> {
        let format =
            TargetPayloadFormat::new(self.format, self.format_version).map_err(|error| {
                self.invalid("format", "repair the registered format identity", &error)
            })?;
        let profile = self.profile()?;
        Ok(Box::new(DialectCompiler {
            format,
            profile,
            emit: self.emit,
        }))
    }

    fn invalid(&self, part: &str, fix: &str, error: &impl std::fmt::Display) -> BackendError {
        BackendError::KernelCompileFailed {
            backend: self.backend_id.to_string(),
            compiler_message: format!(
                "{dialect} target {part} is invalid: {error}. Fix: {fix}.",
                dialect = self.dialect
            ),
        }
    }
}

struct DialectCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
    emit: EmitSelected,
}

impl TargetCompiler for DialectCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        let emit = self.emit;
        compile_selected_modules(
            artifact,
            self.format.clone(),
            self.profile.clone(),
            move |selected, profile| {
                let emitted = emit(selected, profile)?;
                let grid_size = crate::infer_dispatch_grid_for_count(
                    selected.logical_element_count,
                    selected.descriptor.dispatch.workgroup_size,
                )
                .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
                Ok(EmittedTargetModule {
                    entry_point: emitted.entry_point,
                    grid_size,
                    dynamic_shared_bytes: emitted.dynamic_shared_bytes,
                    workgroup_size: selected.descriptor.dispatch.workgroup_size,
                    resource_bindings: selected.canonical_bindings.clone(),
                    bytes: emitted.bytes,
                })
            },
        )
    }
}
