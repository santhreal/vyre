// Integration test module for the containing Vyre package.

#![allow(dead_code, unused_imports)]

#[allow(deprecated)]
pub(crate) mod c_fixture;

use vyre_driver_wgpu::WgpuBackend;

const LIVE_GPU_REQUIRED: &str =
    "WgpuBackend acquisition failed on a machine that must have a GPU. \
Fix: inspect WGPU adapter probing and driver visibility; live GPU tests must not silently skip.";

/// Acquire a fresh live WGPU backend for tests that need isolated backend state.
pub(crate) fn acquire_live_backend() -> WgpuBackend {
    WgpuBackend::acquire().expect(LIVE_GPU_REQUIRED)
}

/// Acquire the shared live WGPU backend for capability/adapter tests.
pub(crate) fn shared_live_backend() -> WgpuBackend {
    WgpuBackend::shared()
        .expect(LIVE_GPU_REQUIRED)
        .as_ref()
        .clone()
}

/// Pack little-endian `u32` lanes into backend dispatch bytes.
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

/// Alias used by C parser integration tests.
pub(crate) fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    u32_bytes(words)
}

/// Decode backend output bytes into little-endian `u32` lanes.
pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as bytes_u32;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;

/// Alias used by C parser integration tests.
pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as words_from_bytes;
