//! Domain-separated cache identity for PTX source and loaded CUDA modules.

use vyre_driver::input_identity::domain_separated_exact_input_key;
use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

use crate::backend::dispatch_phase_probe as probe;

pub(super) const PTX_LOWERING_CONTRACT: &[u8] =
    b"vyre-cuda-ptx-lowering-contract:v15:ssa-carrier-snapshots+f32-canonical+select-pred-normalization+bool-cast-boundary+f32-bool-nan-truthiness+bool-numeric-materialization+bool-memory-word-abi+f32-ne-unordered+masked-integer-shifts+no-mutable-loop-unroll+full-workgroup-entry+bounded-full-workgroup-stores+child-captured-producer-liveness+single-mad-dual-mul-liveness";
pub(super) const CUDA_PTX_SOURCE_FROM_PROGRAM_DOMAIN: &[u8] =
    b"vyre.cuda.ptx-source-cache.program.v1";
pub(super) const CUDA_MODULE_FROM_PTX_SOURCE_KEY_DOMAIN: &[u8] =
    b"vyre.cuda.module-cache.ptx-source-key.v1";
pub(super) const CUDA_MODULE_FROM_RAW_PTX_ARTIFACT_DOMAIN: &[u8] =
    b"vyre.cuda.module-cache.raw-ptx-artifact.v1";

/// Stable key for one PTX module on one CUDA architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ModuleCacheKey(pub(crate) [u8; 32]);

/// Stable key for cached PTX source before CUDA module loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PtxSourceCacheKey(pub(super) [u8; 32]);

impl PtxSourceCacheKey {
    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

fn vsa_fingerprint_cache_bytes(words: [u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (index, word) in words.iter().enumerate() {
        let offset = index * std::mem::size_of::<u32>();
        bytes[offset..offset + std::mem::size_of::<u32>()].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

pub(super) fn ptx_source_cache_key_from_program_identity(
    program: &Program,
    config: &DispatchConfig,
    ptx_target_sm: u32,
    subgroup_size: u32,
    feature_flags: vyre_driver::PipelineFeatureFlags,
) -> Result<PtxSourceCacheKey, BackendError> {
    let normalized_digest = probe::measure_nested(probe::Nested::PtxDigest, || {
        vyre_driver::try_normalized_program_cache_digest(program)
    })
    .map_err(|error| BackendError::new(format!("CUDA PTX source cache digest failed: {error}")))?;
    let vsa_bytes = probe::measure_nested(probe::Nested::PtxVsa, || {
        vsa_fingerprint_cache_bytes(vyre_driver::program_vsa_fingerprint_words(program))
    });
    let dispatch_policy_digest = vyre_driver::dispatch_policy_cache_digest(config);
    let feature_flag_bytes = feature_flags.bits().to_le_bytes();
    let key = domain_separated_exact_input_key(
        CUDA_PTX_SOURCE_FROM_PROGRAM_DOMAIN,
        u64::from(ptx_target_sm),
        u64::from(subgroup_size),
        &[
            PTX_LOWERING_CONTRACT,
            &normalized_digest,
            &vsa_bytes,
            &dispatch_policy_digest,
            &feature_flag_bytes,
        ],
    )?;
    Ok(PtxSourceCacheKey(key))
}

pub(super) fn module_cache_key_from_domain_digest(
    domain_tag: &[u8],
    compute_capability: (u32, u32),
    digest: &[u8; 32],
) -> Result<ModuleCacheKey, BackendError> {
    let key = domain_separated_exact_input_key(
        domain_tag,
        u64::from(compute_capability.0),
        u64::from(compute_capability.1),
        &[&digest[..]],
    )?;
    Ok(ModuleCacheKey(key))
}
