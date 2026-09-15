//! Public error-path contracts for multi-GPU stream sharding.

#![cfg(feature = "device-tests")]

use vyre_driver_wgpu::engine::multi_gpu::{self, shard_by_blake3};

/// Callers must be able to name and match the error returned by the public sharding API.
#[test]
fn zero_gpu_sharding_returns_publicly_nameable_error() {
    let error = shard_by_blake3(b"operator/input.bin", 0)
        .expect_err("zero visible GPUs must fail instead of inventing a device");

    assert_eq!(
        error,
        multi_gpu::StreamShardError::ZeroGpus,
        "the public function must return the error type exposed by the multi-GPU facade",
    );
}
