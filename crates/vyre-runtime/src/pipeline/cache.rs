//! Shared persistent cache for backend compiled-pipeline blobs.

use super::hashing::{
    dispatch_policy_cache_digest, try_normalized_program_cache_digest, PipelineDeviceFingerprint,
};
use crate::backend::DispatchConfig;
use vyre_foundation::ir::Program;

/// Capability bits that participate in pipeline-cache identity.
///
/// Two otherwise-identical pipelines compiled with different
/// `PipelineFeatureFlags` produce different cache keys  -  a pipeline
/// that assumed subgroup-op support cannot be reused on an adapter
/// that does not expose subgroup ops even if the shader bytes match.
///
/// Encoded as a bitfield so the wire form is compact and trivially
/// hashable. Bits `0x01..0x80` are allocated here; higher bits are
/// reserved for additive backend capability flags.
#[derive(
    Copy, Clone, Debug, Default, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct PipelineFeatureFlags(pub u32);

impl PipelineFeatureFlags {
    /// Pipeline was compiled against a lowering that emits subgroup /
    /// wave intrinsics.
    pub const SUBGROUP_OPS: Self = Self(1 << 0);
    /// Pipeline was compiled with native `f16` support.
    pub const F16: Self = Self(1 << 1);
    /// Pipeline was compiled with native `bf16` support.
    pub const BF16: Self = Self(1 << 2);
    /// Pipeline was compiled with tensor-core / matrix-engine
    /// intrinsics enabled.
    pub const TENSOR_CORES: Self = Self(1 << 3);
    /// Pipeline expects an async-compute queue at dispatch time.
    pub const ASYNC_COMPUTE: Self = Self(1 << 4);
    /// Pipeline expects push-constant support at dispatch time.
    pub const PUSH_CONSTANTS: Self = Self(1 << 5);
    /// Pipeline emits indirect-dispatch commands.
    pub const INDIRECT_DISPATCH: Self = Self(1 << 6);
    /// Pipeline was compiled for speculative (fused prefilter+confirmer)
    /// dispatch.
    pub const SPECULATIVE: Self = Self(1 << 7);
    /// Pipeline was compiled for persistent-thread (device-side work queue)
    /// dispatch.
    pub const PERSISTENT_THREAD: Self = Self(1 << 8);

    /// Empty flag set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Contains at least every bit of `other`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of two flag sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw bit representation.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Shared in-memory identity for a backend-compiled pipeline.
///
/// Backends may keep their own cache maps and compiled handles, but this object
/// keeps the identity facts single-sourced: normalized Program digest, dispatch
/// policy digest, device/runtime fingerprint, and the final digest used as the
/// lookup key.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PipelineCacheIdentity {
    /// Final tuple-boundary-preserving lookup digest.
    pub digest: [u8; 32],
    /// Normalized Program digest used by backend pipeline caches.
    pub program_digest: [u8; 32],
    /// Dispatch policy fields that alter generated backend code.
    pub policy_digest: [u8; 32],
    /// Backend/device/runtime identity participating in the final digest.
    pub device_fingerprint: PipelineDeviceFingerprint,
}

impl PipelineCacheIdentity {
    /// Build identity from already-computed shared components.
    #[must_use]
    pub fn from_parts(
        program_digest: [u8; 32],
        policy_digest: [u8; 32],
        device_fingerprint: PipelineDeviceFingerprint,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-pipeline-cache-identity-v1\0program\0");
        hasher.update(&program_digest);
        hasher.update(b"\0policy\0");
        hasher.update(&policy_digest);
        hasher.update(b"\0vendor\0");
        hasher.update(&device_fingerprint.vendor.to_le_bytes());
        hasher.update(b"\0device\0");
        hasher.update(&device_fingerprint.device.to_le_bytes());
        hasher.update(b"\0driver\0");
        hasher.update(&device_fingerprint.driver_digest);
        Self {
            digest: *hasher.finalize().as_bytes(),
            program_digest,
            policy_digest,
            device_fingerprint,
        }
    }

    /// Build identity from a public Program and dispatch config.
    ///
    /// # Errors
    ///
    /// Returns when the Program cannot be serialized into stable cache
    /// identity. Callers should surface the error through their backend error
    /// type instead of hashing invalid IR lossy.
    pub fn try_from_program(
        program: &Program,
        config: &DispatchConfig,
        device_fingerprint: PipelineDeviceFingerprint,
    ) -> Result<Self, String> {
        let program_digest = try_normalized_program_cache_digest(program)?;
        let policy_digest = dispatch_policy_cache_digest(config);
        Ok(Self::from_parts(
            program_digest,
            policy_digest,
            device_fingerprint,
        ))
    }
}

/// Evidence a backend can provide when a pipeline-cache lookup misses.
///
/// The classifier is intentionally backend-neutral: a concrete driver keeps its
/// own fast cache key, but records enough adjacent identity facts to explain why
/// a miss happened without duplicating per-backend reason logic.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PipelineCacheMissEvidence {
    /// Number of compiled pipeline entries present at lookup time.
    pub total_entries: usize,
    /// Entries whose normalized Program digest matched the requested program.
    pub same_program_entries: usize,
    /// Entries whose Program digest and dispatch policy digest both matched.
    pub same_program_and_policy_entries: usize,
    /// Entries whose Program digest, dispatch policy digest, and
    /// device/runtime fingerprint all matched.
    pub same_program_policy_and_device_entries: usize,
}

impl PipelineCacheMissEvidence {
    /// Build miss evidence from cached identities adjacent to the requested key.
    #[must_use]
    pub fn from_identities<'a>(
        cached: impl Iterator<Item = &'a PipelineCacheIdentity>,
        requested: &PipelineCacheIdentity,
    ) -> Self {
        let mut evidence = Self::default();
        for identity in cached {
            evidence.total_entries += 1;
            if identity.program_digest == requested.program_digest {
                evidence.same_program_entries += 1;
                if identity.policy_digest == requested.policy_digest {
                    evidence.same_program_and_policy_entries += 1;
                    if identity.device_fingerprint == requested.device_fingerprint {
                        evidence.same_program_policy_and_device_entries += 1;
                    }
                }
            }
        }
        evidence
    }
}

/// Backend-neutral cache-miss reason for operator telemetry.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PipelineCacheMissReason {
    /// The backend had no compiled entries at lookup time.
    EmptyCache,
    /// Existing entries were for different normalized programs.
    ProgramChanged,
    /// The same program existed, but dispatch policy changed the generated
    /// pipeline identity.
    DispatchPolicyChanged,
    /// Program and dispatch policy matched, but device/runtime identity changed.
    DeviceOrRuntimeChanged,
    /// The supplied evidence says adjacent identity matched but the final key
    /// still missed; this catches future key fields and malformed bookkeeping.
    KeyAbsent,
}

impl PipelineCacheMissReason {
    /// Classify a miss from backend-supplied adjacent identity evidence.
    #[must_use]
    pub const fn classify(evidence: PipelineCacheMissEvidence) -> Self {
        if evidence.total_entries == 0 {
            Self::EmptyCache
        } else if evidence.same_program_entries == 0 {
            Self::ProgramChanged
        } else if evidence.same_program_and_policy_entries == 0 {
            Self::DispatchPolicyChanged
        } else if evidence.same_program_policy_and_device_entries == 0 {
            Self::DeviceOrRuntimeChanged
        } else {
            Self::KeyAbsent
        }
    }

    /// Classify a miss directly from cached pipeline identities.
    #[must_use]
    pub fn classify_identities<'a>(
        cached: impl Iterator<Item = &'a PipelineCacheIdentity>,
        requested: &PipelineCacheIdentity,
    ) -> Self {
        Self::classify(PipelineCacheMissEvidence::from_identities(
            cached, requested,
        ))
    }

    /// Stable metric suffix for backend metric snapshots.
    #[must_use]
    pub const fn metric_suffix(self) -> &'static str {
        match self {
            Self::EmptyCache => "empty_cache",
            Self::ProgramChanged => "program_changed",
            Self::DispatchPolicyChanged => "dispatch_policy_changed",
            Self::DeviceOrRuntimeChanged => "device_or_runtime_changed",
            Self::KeyAbsent => "key_absent",
        }
    }
}

// Inline: `vyre_runtime::pipeline` is `pub(crate)`, so no integration test can reach what this suite
// exercises.
#[cfg(test)]
mod pipeline_cache_key_tests {
    use super::*;

    fn hash32(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn flag_containment_is_correct() {
        let a = PipelineFeatureFlags::SUBGROUP_OPS.union(PipelineFeatureFlags::F16);
        assert!(a.contains(PipelineFeatureFlags::SUBGROUP_OPS));
        assert!(a.contains(PipelineFeatureFlags::F16));
        assert!(!a.contains(PipelineFeatureFlags::TENSOR_CORES));
    }

    #[test]
    fn shared_cache_identity_separates_program_policy_and_device_facts() {
        let program_a = hash32(1);
        let program_b = hash32(2);
        let policy_a = hash32(3);
        let policy_b = hash32(4);
        let device_a = PipelineDeviceFingerprint::from_parts(1, 2, "driver-a", "runtime-a");
        let device_b = PipelineDeviceFingerprint::from_parts(1, 2, "driver-a", "runtime-b");

        let base = PipelineCacheIdentity::from_parts(program_a, policy_a, device_a);

        assert_eq!(base.program_digest, program_a);
        assert_eq!(base.policy_digest, policy_a);
        assert_eq!(base.device_fingerprint, device_a);
        assert_ne!(
            base.digest,
            PipelineCacheIdentity::from_parts(program_b, policy_a, device_a).digest,
            "Fix: shared pipeline cache identity must include the normalized Program digest."
        );
        assert_ne!(
            base.digest,
            PipelineCacheIdentity::from_parts(program_a, policy_b, device_a).digest,
            "Fix: shared pipeline cache identity must include dispatch policy as its own tuple field."
        );
        assert_ne!(
            base.digest,
            PipelineCacheIdentity::from_parts(program_a, policy_a, device_b).digest,
            "Fix: shared pipeline cache identity must include device/runtime fingerprint facts."
        );
    }

    #[test]
    fn miss_reason_classifies_adjacent_identity_evidence() {
        assert_eq!(
            PipelineCacheMissReason::classify(PipelineCacheMissEvidence {
                total_entries: 0,
                same_program_entries: 0,
                same_program_and_policy_entries: 0,
                same_program_policy_and_device_entries: 0,
            }),
            PipelineCacheMissReason::EmptyCache
        );
        assert_eq!(
            PipelineCacheMissReason::classify(PipelineCacheMissEvidence {
                total_entries: 3,
                same_program_entries: 0,
                same_program_and_policy_entries: 0,
                same_program_policy_and_device_entries: 0,
            }),
            PipelineCacheMissReason::ProgramChanged
        );
        assert_eq!(
            PipelineCacheMissReason::classify(PipelineCacheMissEvidence {
                total_entries: 3,
                same_program_entries: 2,
                same_program_and_policy_entries: 0,
                same_program_policy_and_device_entries: 0,
            }),
            PipelineCacheMissReason::DispatchPolicyChanged
        );
        assert_eq!(
            PipelineCacheMissReason::classify(PipelineCacheMissEvidence {
                total_entries: 3,
                same_program_entries: 2,
                same_program_and_policy_entries: 1,
                same_program_policy_and_device_entries: 0,
            }),
            PipelineCacheMissReason::DeviceOrRuntimeChanged
        );
        assert_eq!(
            PipelineCacheMissReason::classify(PipelineCacheMissEvidence {
                total_entries: 3,
                same_program_entries: 2,
                same_program_and_policy_entries: 1,
                same_program_policy_and_device_entries: 1,
            }),
            PipelineCacheMissReason::KeyAbsent
        );
    }

    #[test]
    fn miss_reason_metric_suffixes_are_stable_snake_case() {
        assert_eq!(
            PipelineCacheMissReason::EmptyCache.metric_suffix(),
            "empty_cache"
        );
        assert_eq!(
            PipelineCacheMissReason::ProgramChanged.metric_suffix(),
            "program_changed"
        );
        assert_eq!(
            PipelineCacheMissReason::DispatchPolicyChanged.metric_suffix(),
            "dispatch_policy_changed"
        );
        assert_eq!(
            PipelineCacheMissReason::DeviceOrRuntimeChanged.metric_suffix(),
            "device_or_runtime_changed"
        );
        assert_eq!(
            PipelineCacheMissReason::KeyAbsent.metric_suffix(),
            "key_absent"
        );
    }

    #[test]
    fn miss_reason_classifies_cached_shared_identities() {
        let program = hash32(1);
        let other_program = hash32(2);
        let policy = hash32(3);
        let other_policy = hash32(4);
        let device = PipelineDeviceFingerprint::from_parts(1, 2, "driver-a", "runtime-a");
        let other_device = PipelineDeviceFingerprint::from_parts(1, 2, "driver-a", "runtime-b");
        let requested = PipelineCacheIdentity::from_parts(program, policy, device);

        assert_eq!(
            PipelineCacheMissReason::classify_identities([].iter(), &requested),
            PipelineCacheMissReason::EmptyCache
        );
        assert_eq!(
            PipelineCacheMissReason::classify_identities(
                [PipelineCacheIdentity::from_parts(
                    other_program,
                    policy,
                    device
                )]
                .iter(),
                &requested,
            ),
            PipelineCacheMissReason::ProgramChanged
        );
        assert_eq!(
            PipelineCacheMissReason::classify_identities(
                [PipelineCacheIdentity::from_parts(
                    program,
                    other_policy,
                    device
                )]
                .iter(),
                &requested,
            ),
            PipelineCacheMissReason::DispatchPolicyChanged
        );
        assert_eq!(
            PipelineCacheMissReason::classify_identities(
                [PipelineCacheIdentity::from_parts(
                    program,
                    policy,
                    other_device
                )]
                .iter(),
                &requested,
            ),
            PipelineCacheMissReason::DeviceOrRuntimeChanged
        );
        assert_eq!(
            PipelineCacheMissReason::classify_identities(
                [PipelineCacheIdentity::from_parts(program, policy, device)].iter(),
                &requested,
            ),
            PipelineCacheMissReason::KeyAbsent
        );
    }
}
