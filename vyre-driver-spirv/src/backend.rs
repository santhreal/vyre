//! SPIR-V target adapter over the canonical verified lower boundary and emitter.
use vyre_foundation::ir::Program;

/// Thin SPIR-V target adapter.
pub struct SpirvBackend;

impl SpirvBackend {
    /// Stable backend identifier.
    pub const BACKEND_ID: &'static str = super::SPIRV_BACKEND_ID;

    /// Construct a new backend instance. Always succeeds.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compile one Program through canonical verified lowering and emission.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when shared lowering or SPIR-V emission
    /// rejects the program.

    pub fn program_to_spv(program: &Program) -> Result<Vec<u32>, String> {
        let lowered = vyre_lower::lower_verified(program)
            .map_err(|error| format!("verified lowering failed before SPIR-V emission: {error}"))?;
        vyre_emit_spirv::emit(&lowered.descriptor)
            .map_err(|error| format!("SPIR-V emission failed: {error}"))
    }

    /// Compute the substrate VSA fingerprint of a vyre Program. Same
    /// fingerprint vyre-aot persists on `CompiledArtifact` and
    /// runtime validation caches use for their identity key; sharing the
    /// fingerprint across backends lets a single SPIR-V or PTX cache
    /// dedup against AOT artifacts.
    ///
    /// P-SPIRV-1: substrate consumption  -  vsa_fingerprint is the
    /// identity-by-meaning key that crosses backend boundaries.
    #[must_use]
    pub fn program_fingerprint(program: &Program) -> Vec<u32> {
        vyre_driver::program_vsa_fingerprint(program)
    }

    /// Snapshot the driver-tier observability surface
    /// ([`vyre_driver::observability::DriverObservability`]).
    #[must_use]
    pub fn observability_snapshot() -> vyre_driver::observability::DriverObservability {
        vyre_driver::observability::DriverObservability::snapshot()
    }

    /// SPIR-V module disk-cache directory. Same on-disk-key family as
    /// native-module and validation caches, keyed by VSA fingerprint
    /// via [`Self::program_fingerprint`].
    ///
    /// P-SPIRV-2: SPIR-V module blobs persist across runs.
    #[must_use]
    pub fn spv_disk_cache_dir() -> std::path::PathBuf {
        std::env::var_os("VYRE_SPV_CACHE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("vyre-spv-cache"))
    }
}

impl Default for SpirvBackend {
    fn default() -> Self {
        Self::new()
    }
}
