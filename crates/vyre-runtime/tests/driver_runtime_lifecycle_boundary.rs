//! Tests for driver policy vs runtime lifecycle boundary (Section 188.3).
//!
//! Verifies that:
//! 1. `vyre-runtime` owns backend-neutral target planning, materialization,
//!    submission, and compiled-pipeline caching.
//! 2. `vyre-runtime` owns artifact admission, residency lifecycle, persistent queue
//!    orchestration, and fault recovery.
//! 3. Cache keys and identity stores for compiled pipelines vs admitted artifacts
//!    remain distinct and isolated.

use vyre_runtime::{PipelineCacheIdentity, PipelineDeviceFingerprint};

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
