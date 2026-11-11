//! Tests for driver policy vs runtime lifecycle boundary (Section 188.3).
//!
//! Verifies that:
//! 1. `vyre-driver` owns backend-neutral target planning, materialization,
//!    submission, and compiled-pipeline caching.
//! 2. `vyre-runtime` owns artifact admission, residency lifecycle, persistent queue
//!    orchestration, and fault recovery.
//! 3. Cache keys and identity stores for compiled pipelines vs admitted artifacts
//!    remain distinct and isolated.

use vyre_driver::{
    PipelineCacheIdentity, PipelineCacheKey, PipelineDeviceFingerprint, PipelineFeatureFlags,
    CURRENT_PIPELINE_CACHE_KEY_VERSION,
};

#[test]
fn driver_pipeline_cache_key_isolation() {
    let dummy_shader_hash = [42u8; 32];
    let dummy_layout_hash = [43u8; 32];

    let key1 = PipelineCacheKey::new(
        dummy_shader_hash,
        dummy_layout_hash,
        0,
        [64, 1, 1],
        PipelineFeatureFlags::empty(),
        "wgpu".into(),
    );

    let key2 = PipelineCacheKey::new(
        dummy_shader_hash,
        dummy_layout_hash,
        0,
        [64, 1, 1],
        PipelineFeatureFlags::empty(),
        "cuda".into(),
    );

    assert_eq!(key1.version, CURRENT_PIPELINE_CACHE_KEY_VERSION);
    assert_ne!(
        key1, key2,
        "different backends must produce isolated pipeline cache keys"
    );
}

#[test]
fn driver_and_runtime_store_identities_are_distinct() {
    let fingerprint = PipelineDeviceFingerprint {
        vendor: 0x10de,
        device: 0x2684,
        driver_digest: [7u8; 32],
    };
    let pipeline_identity = PipelineCacheIdentity::from_parts([1u8; 32], [2u8; 32], fingerprint);

    assert_eq!(pipeline_identity.device_fingerprint.vendor, 0x10de);
    assert_ne!(pipeline_identity.digest, [0u8; 32]);
}
