//! Contracts for `vyre_driver_wgpu::ext`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver_wgpu::WgpuBackend;

#[test]
fn dispatch_wgsl_reuses_backend_pipeline_cache() {
    let Ok(backend) = WgpuBackend::acquire() else {
        panic!("Fix: WGPU dispatch_wgsl cache test requires a live GPU adapter");
    };
    let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(1) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
if (gid.x >= params.y) {
    return;
}
output[gid.x] = input[gid.x] + 1u;
}
"#;
    let input: Vec<u8> = [41_u32, 99_u32]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();

    let first = backend
        .dispatch_wgsl(wgsl, &input, 8, 64)
        .expect("Fix: first raw WGSL dispatch must compile and run");
    let second = backend
        .dispatch_wgsl(wgsl, &input, 8, 64)
        .expect("Fix: second raw WGSL dispatch must reuse the cached pipeline and run");

    assert_eq!(first, second);
    assert_eq!(first, [42_u32, 100_u32].as_slice().as_bytes());
    assert_eq!(
        backend.wgsl_dispatch_pipeline_cache.len(),
        1,
        "Fix: identical dispatch_wgsl source must compile once per backend instance"
    );
}

trait U32SliceBytes {
    fn as_bytes(&self) -> &[u8];
}

impl U32SliceBytes for [u32] {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }
}
