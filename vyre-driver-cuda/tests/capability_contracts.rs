//! Live CUDA capability contracts for GPU-required Vyre hosts.

#![cfg(feature = "device-tests")]

use vyre_driver::DispatchConfig;
use vyre_driver::PipelineFeatureFlags;
use vyre_driver_cuda::{cuda_factory, CudaBackend, CudaDeviceCaps, CudaMegakernelDeviceKey};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

#[test]
fn cuda_device_probe_must_succeed_on_gpu_fleet() {
    let visible = CudaDeviceCaps::visible_device_count()
        .expect("Fix: CUDA driver/device probe failed on a GPU-required machine.");
    assert!(
        visible > 0,
        "Fix: CUDA reported zero visible devices on a GPU-required machine."
    );
    let all_devices = CudaDeviceCaps::probe_all()
        .expect("Fix: CUDA probe_all must enumerate every visible GPU without dropping errors.");
    assert_eq!(
        all_devices.len(),
        visible,
        "Fix: CUDA probe_all must return one capability record per visible device."
    );

    let backend = CudaBackend::acquire()
        .expect("Fix: CudaBackend::acquire must succeed on the GPU-required test host.");
    assert!(
        !backend.caps.name.trim().is_empty(),
        "Fix: CUDA device-name probe returned an empty adapter name."
    );
    assert!(
        backend.compute_capability() >= (8, 0),
        "Fix: expected CUDA device to report compute capability >= (8, 0), got {:?}.",
        backend.compute_capability()
    );
    assert!(
        backend.device_memory_bytes() >= 8 * 1024 * 1024 * 1024,
        "Fix: expected at least 8 GiB VRAM on the CUDA device, got {} bytes.",
        backend.device_memory_bytes()
    );
}

#[test]
fn cuda_backend_caps_match_driver_attributes() {
    let expected =
        CudaDeviceCaps::probe(0).expect("Fix: direct CUDA capability probe failed on device 0.");

    let backend = CudaBackend::acquire()
        .expect("Fix: CudaBackend::acquire failed after direct CUDA probe succeeded.");
    assert_eq!(
        backend.compute_capability(),
        expected.compute_capability,
        "Fix: CudaBackend compute capability must come from CUDA device attributes."
    );
    assert_eq!(
        backend.device_memory_bytes(),
        expected.total_memory,
        "Fix: CudaBackend memory limit must come from CUDA device attributes."
    );
    assert_eq!(
        backend.warp_size(),
        Some(expected.warp_size as u32),
        "Fix: CudaBackend subgroup width must come from CUDA warp-size attributes."
    );
    assert_eq!(
        backend.max_block_dim(),
        expected.max_block_dim.map(|axis| axis as u32),
        "Fix: CudaBackend per-axis block limits must come from CUDA max block dimension attributes."
    );
    assert_eq!(
        backend.caps.to_adapter_caps().max_workgroup_size,
        backend.max_block_dim(),
        "Fix: CUDA adapter caps and backend launch validation must share live per-axis block limits."
    );
    assert_eq!(
        backend.caps.to_adapter_caps().max_shared_memory_bytes,
        backend.max_shared_memory_per_block_bytes(),
        "Fix: CUDA shared-memory reporting must use the live CUDA device attribute."
    );
    assert!(
        backend.max_shared_memory_per_block_bytes() >= 16 * 1024,
        "Fix: CUDA shared-memory capability probe returned an impossible block budget: {} bytes.",
        backend.max_shared_memory_per_block_bytes()
    );
    assert_eq!(
        backend.target_sm(),
        expected.compute_capability.0 * 10 + expected.compute_capability.1,
        "Fix: CUDA physical target SM must be derived from the live device, not hardcoded."
    );
    assert!(
        backend.ptx_target_sm() <= backend.target_sm(),
        "Fix: CUDA PTX emitter target must not exceed the physical device target."
    );
    assert!(
        backend.ptx_target_sm() >= 70,
        "Fix: CUDA PTX emitter target must be selected by live driver probing and preserve modern NVIDIA instructions instead of falling back to an old baseline."
    );
    assert!(
        backend.hardware_supports_subgroup_ops(),
        "Fix: CUDA hardware probe must expose NVIDIA warp/subgroup capability."
    );
    assert!(
        backend.hardware_supports_f16(),
        "Fix: CUDA hardware probe must expose f16 arithmetic capability."
    );
    assert!(
        backend.hardware_supports_bf16(),
        "Fix: CUDA hardware probe must expose bf16 arithmetic capability."
    );
    assert!(
        backend.hardware_supports_tensor_cores(),
        "Fix: CUDA hardware probe must expose tensor-core capability."
    );
    assert!(
        backend.lowers_tensor_core_ops(),
        "Fix: CUDA must advertise tensor-core lowering now that vyre-lower promotes MMA descriptors and vyre-emit-ptx emits mma.sync."
    );
    let flags = backend.pipeline_feature_flags();
    assert!(
        flags.contains(PipelineFeatureFlags::SUBGROUP_OPS),
        "Fix: CUDA pipeline cache keys must include subgroup capability bits."
    );
    assert!(
        flags.contains(PipelineFeatureFlags::F16),
        "Fix: CUDA pipeline cache keys must include f16 capability bits."
    );
    assert!(
        flags.contains(PipelineFeatureFlags::BF16),
        "Fix: CUDA pipeline cache keys must include bf16 capability bits."
    );
    assert!(
        flags.contains(PipelineFeatureFlags::TENSOR_CORES),
        "Fix: CUDA pipeline cache keys must include tensor-core lowering bits now that PTX emits mma.sync."
    );
    let megakernel_device_key = CudaMegakernelDeviceKey::from(&backend.caps);
    assert_eq!(
        megakernel_device_key.sm_major as u32, expected.compute_capability.0,
        "Fix: CUDA megakernel plan cache key must include probed SM major version."
    );
    assert_eq!(
        megakernel_device_key.sm_minor as u32, expected.compute_capability.1,
        "Fix: CUDA megakernel plan cache key must include probed SM minor version."
    );
    assert_eq!(
        megakernel_device_key.warp_size as u32, expected.warp_size as u32,
        "Fix: CUDA megakernel plan cache key must include probed warp size."
    );
    assert_eq!(
        megakernel_device_key.supports_grid_sync,
        backend.hardware_supports_grid_sync(),
        "Fix: CUDA megakernel plan cache key must include live cooperative grid-sync capability."
    );
    assert_eq!(
        megakernel_device_key.supports_tensor_cores,
        backend.hardware_supports_tensor_cores(),
        "Fix: CUDA megakernel plan cache key must include live tensor-core capability."
    );
    assert_eq!(
        megakernel_device_key.max_workgroup_size,
        backend.max_threads_per_block(),
        "Fix: CUDA megakernel plan cache key must include live max workgroup size."
    );
    assert_eq!(
        backend.caps.compute_capability.0, expected.compute_capability.0,
        "Fix: CUDA device caps must include probed SM major version."
    );
    assert_eq!(
        backend.caps.compute_capability.1, expected.compute_capability.1,
        "Fix: CUDA device caps must include probed SM minor version."
    );
    assert_eq!(
        backend.caps.warp_size as u32, expected.warp_size as u32,
        "Fix: CUDA device caps must include probed warp size."
    );
    assert_eq!(
        backend.caps.cooperative_launch,
        backend.hardware_supports_grid_sync(),
        "Fix: CUDA device caps cooperative launch must match live cooperative grid-sync capability."
    );
    assert_eq!(
        backend.caps.hardware_supports_tensor_cores(),
        backend.hardware_supports_tensor_cores(),
        "Fix: CUDA device caps must match live tensor-core capability."
    );
    assert_eq!(
        backend.caps.max_threads_per_block as u32,
        backend.max_threads_per_block(),
        "Fix: CUDA device caps must include live max workgroup size."
    );
}

#[test]
fn cuda_is_canonical_dispatch_backend_when_linked() {
    assert!(
        vyre_driver::backend_precedence("cuda").expect("valid backend registry") < 10,
        "Fix: CUDA must outrank wgpu for this release when both live dispatch backends are linked."
    );
    assert!(
        vyre_driver::backend_dispatches("cuda").expect("valid backend registry"),
        "Fix: CUDA must advertise live dispatch capability for release routing."
    );
}

#[test]
fn vyre_backend_trait_reports_live_cuda_capabilities() {
    let backend = cuda_factory()
        .expect("Fix: CUDA backend factory must succeed on the GPU-required test host.");

    assert!(
        backend.supports_subgroup_ops(),
        "Fix: CUDA VyreBackend must advertise subgroup lowering now that PTX emits warp-sync subgroup operations."
    );
    assert_eq!(
        backend.subgroup_size(),
        Some(32),
        "Fix: CUDA subgroup_size must report NVIDIA warp width."
    );
    assert!(
        backend.supports_f16(),
        "Fix: CUDA VyreBackend must advertise f16 once PTX lowers f16 buffer load/store through f32 arithmetic."
    );
    assert!(
        backend.supports_bf16(),
        "Fix: CUDA VyreBackend must advertise bf16 once PTX lowers bf16 buffer load/store through deterministic f32 conversion."
    );
    assert!(
        backend.supports_tensor_cores(),
        "Fix: CUDA VyreBackend must advertise tensor-core execution now that lowering emits MatrixMma and PTX emits mma.sync."
    );
    assert!(
        backend.supports_async_compute(),
        "Fix: CUDA backend must report async CUDA hardware capability."
    );
    assert!(
        !backend.allows_host_grid_sync_split(),
        "Fix: CUDA must not permit hidden host-orchestrated GridSync splitting; missing native grid-sync lowering must be a loud release-path error."
    );
}

#[test]
fn preferred_dispatch_backend_is_cuda_not_cpu_or_wgpu_fallback() {
    let backend = vyre_driver::acquire_preferred_dispatch_backend()
        .expect("Fix: CUDA must be usable as the preferred dispatch backend on the GPU fleet.");
    assert_eq!(
        backend.id(),
        "cuda",
        "Fix: benchmark/smoke routing must select canonical CUDA when vyre-driver-cuda is linked; do not silently route GPU-required runs to another backend."
    );
    assert!(
        !backend.allows_host_grid_sync_split(),
        "Fix: registered CUDA acquisition must preserve the CUDA no-host-split policy through the shared registry wrapper."
    );
}

#[test]
fn cuda_grid_sync_capability_reflects_native_cooperative_lowering() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");

    // Native GridSync lowering now exists: the PTX emitter lowers
    // MemoryOrdering::GridSync to a monotonic-counter cooperative grid barrier
    // and the dispatch path launches it cooperatively, zeroing the module-scope
    // counter before each launch. This is proven byte-identical to the host
    // split by
    // cooperative_launch_contracts::native_cooperative_grid_sync_matches_host_split_cross_block.
    assert!(
        backend.lowers_grid_sync(),
        "Fix: CUDA lowers MemoryOrdering::GridSync to a native cooperative grid barrier; lowers_grid_sync() must report it now that the PTX emitter + cooperative launch path exist."
    );
    // supports_grid_sync() must describe *executable* native GridSync: native
    // lowering AND a device that can run a cooperative launch. With lowering
    // present that reduces to the hardware capability.
    assert_eq!(
        backend.supports_grid_sync(),
        backend.hardware_supports_grid_sync(),
        "Fix: with native cooperative lowering present, supports_grid_sync() must equal hardware_supports_grid_sync() (true exactly when the device can run a resident cooperative grid)."
    );
}

#[test]
fn cuda_registration_dispatch_borrowed_into_reuses_caller_output_slot() {
    let backend = cuda_factory()
        .expect("Fix: CUDA backend factory must succeed on the GPU-required test host.");
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(0xfeed_beef))],
    );
    let mut outputs = vec![Vec::with_capacity(64)];
    let outer_ptr = outputs.as_ptr();
    let slot_ptr = outputs[0].as_ptr();

    backend
        .dispatch_borrowed_into(&program, &[], &DispatchConfig::default(), &mut outputs)
        .expect(
            "Fix: CUDA dispatch_borrowed_into must execute through caller-owned output storage.",
        );

    assert_eq!(
        outputs.as_ptr(),
        outer_ptr,
        "Fix: CUDA registration dispatch_borrowed_into must preserve the caller-owned output slot vector."
    );
    assert_eq!(
        outputs[0].as_ptr(),
        slot_ptr,
        "Fix: CUDA registration dispatch_borrowed_into must collect readback bytes into the existing output allocation."
    );
    assert_eq!(
        outputs[0].as_slice(),
        &0xfeed_beef_u32.to_le_bytes(),
        "Fix: CUDA output-slot reuse must preserve byte-exact dispatch results."
    );
}
