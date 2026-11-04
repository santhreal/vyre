//! The feature set requested at device creation, and the probe that confirms
//! the request compiles.
//!
//! An adapter advertising a feature is not the same as a device that honors
//! it, so negotiation and verification are one concern and live together.

/// Snapshot of features that were actually enabled when the cached
/// device was created. Consumed by `WgpuBackend::supports_*` methods
/// so the VyreBackend capability reports are *honest*  -  a feature bit
/// is reported only if it was both advertised by the adapter AND
/// requested at device creation.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct EnabledFeatures {
    /// Wgpu timestamp queries feature.
    pub timestamp_query: bool,
    /// Wgpu timestamp writes directly on command encoders.
    pub timestamp_query_inside_encoders: bool,
    /// Wgpu subgroup feature.
    pub subgroup: bool,
    /// Wgpu subgroup barrier feature.
    pub subgroup_barrier: bool,
    /// Wgpu shader f16 feature.
    pub shader_f16: bool,
    /// Wgpu pipeline cache feature.
    pub pipeline_cache: bool,
    /// Wgpu push constants feature.
    pub push_constants: bool,
    /// Wgpu indirect first instance feature.
    pub indirect_first_instance: bool,
    /// Wgpu adapter max workgroup size limit.
    pub max_workgroup_size: [u32; 3],
    /// Wgpu adapter max storage buffer binding size limit.
    pub max_storage_buffer_binding_size: u64,
    /// Wgpu adapter max subgroup size.
    pub max_subgroup_size: u32,
    /// Wgpu adapter minimum subgroup size (I.6). `0` means the
    /// adapter did not report a subgroup size; consumers must treat
    /// subgroup-width-dependent planning as unavailable unless
    /// `crate::capabilities::supports_subgroup_ops` returns true.
    pub min_subgroup_size: u32,
}

/// wgpu only implements the persistent pipeline cache on the Vulkan and DX12
/// backends (`VK_EXT_pipeline_creation_cache_control` / `ID3D12PipelineLibrary`).
/// Apple's Metal backend (and GL) advertise the `PIPELINE_CACHE` adapter
/// feature under wgpu 25 but then fail `device_create_pipeline_cache_init`
/// with a fatal, un-catchable validation error in downstream GPU diagnostics.
/// Gate the request on a backend that actually honors it.
fn backend_implements_pipeline_cache(backend: wgpu::Backend) -> bool {
    matches!(backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12)
}

pub(super) fn enabled_features_for_adapter(
    adapter_features: wgpu::Features,
    adapter_limits: &wgpu::Limits,
    backend: wgpu::Backend,
) -> (wgpu::Features, EnabledFeatures) {
    let mut features = wgpu::Features::empty();
    let mut enabled = EnabledFeatures::default();
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        features |= wgpu::Features::TIMESTAMP_QUERY;
        enabled.timestamp_query = true;
    }
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
        features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        enabled.timestamp_query = true;
        enabled.timestamp_query_inside_encoders = true;
    }
    if crate::capabilities::supports_subgroup_for_adapter(adapter_features, adapter_limits) {
        features |= wgpu::Features::SUBGROUP;
        enabled.subgroup = true;
    }
    if adapter_features.contains(wgpu::Features::SUBGROUP_BARRIER) {
        features |= wgpu::Features::SUBGROUP_BARRIER;
        enabled.subgroup_barrier = true;
    }
    if adapter_features.contains(wgpu::Features::SHADER_F16) {
        features |= wgpu::Features::SHADER_F16;
        enabled.shader_f16 = true;
    }
    if adapter_features.contains(wgpu::Features::PIPELINE_CACHE)
        && backend_implements_pipeline_cache(backend)
    {
        features |= wgpu::Features::PIPELINE_CACHE;
        enabled.pipeline_cache = true;
    }
    if adapter_features.contains(wgpu::Features::PUSH_CONSTANTS) {
        features |= wgpu::Features::PUSH_CONSTANTS;
        enabled.push_constants = true;
    }
    if adapter_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        features |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
        enabled.indirect_first_instance = true;
    }

    enabled.max_workgroup_size = [
        adapter_limits.max_compute_workgroup_size_x,
        adapter_limits.max_compute_workgroup_size_y,
        adapter_limits.max_compute_workgroup_size_z,
    ];
    enabled.max_storage_buffer_binding_size =
        u64::from(adapter_limits.max_storage_buffer_binding_size);
    enabled.max_subgroup_size = adapter_limits.max_subgroup_size;
    enabled.min_subgroup_size = adapter_limits.min_subgroup_size;
    (features, enabled)
}

pub(super) fn subgroup_smoke_compiles(device: &wgpu::Device) -> std::result::Result<(), String> {
    const WGSL: &str = r#"
@compute @workgroup_size(32)
fn main(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32) {
    let total = subgroupAdd(lane + size);
    if (total == 0u) {
        return;
    }
}
"#;

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vyre subgroup capability probe"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vyre subgroup capability probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    match super::pop_error_scope_now(device) {
        Ok(None) => Ok(()),
        Ok(Some(error)) => Err(format!("validation error: {error}")),
        Err(error) => Err(error.to_string()),
    }
}

// Inline: the negotiation and its backend guard are crate-private, so an
// integration test cannot name either one.
#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the Apple-Silicon GPU crash: Metal (and GL / WebGPU)
    /// advertise the `PIPELINE_CACHE` adapter feature under wgpu 25 but then fail
    /// `device_create_pipeline_cache_init` with a fatal, un-catchable validation
    /// error. `enabled_features_for_adapter` MUST gate the feature off on those
    /// backends even when the adapter advertises it; dropping the
    /// `backend_implements_pipeline_cache` guard silently reintroduces a hard macOS
    /// crash that no Linux/Windows host would surface. These are pure functions of
    /// the backend enum, so this locks the cross-OS guard without a Mac or a GPU.
    #[test]
    fn pipeline_cache_enabled_only_on_backends_that_implement_it() {
        let limits = wgpu::Limits::default();
        let advertises = wgpu::Features::PIPELINE_CACHE;

        // Backends wgpu actually implements the persistent cache on -> enable it.
        for backend in [wgpu::Backend::Vulkan, wgpu::Backend::Dx12] {
            assert!(
                backend_implements_pipeline_cache(backend),
                "{backend:?} implements the persistent pipeline cache (Vulkan/DX12)"
            );
            let (features, enabled) = enabled_features_for_adapter(advertises, &limits, backend);
            assert!(
                features.contains(wgpu::Features::PIPELINE_CACHE) && enabled.pipeline_cache,
                "{backend:?} advertises AND implements PIPELINE_CACHE -> must be enabled"
            );
        }

        // Backends that advertise the feature but crash on init -> gate OFF.
        for backend in [
            wgpu::Backend::Metal,
            wgpu::Backend::Gl,
            wgpu::Backend::BrowserWebGpu,
            wgpu::Backend::Noop,
        ] {
            assert!(
                !backend_implements_pipeline_cache(backend),
                "{backend:?} does not implement the persistent pipeline cache"
            );
            let (features, enabled) = enabled_features_for_adapter(advertises, &limits, backend);
            assert!(
                !features.contains(wgpu::Features::PIPELINE_CACHE) && !enabled.pipeline_cache,
                "{backend:?} advertises PIPELINE_CACHE but crashes on init -> must be gated OFF (Apple-Silicon crash guard)"
            );
        }

        // An implementing backend that does NOT advertise the feature -> still off
        // (no phantom enable when the adapter never offered it).
        let (features, enabled) =
            enabled_features_for_adapter(wgpu::Features::empty(), &limits, wgpu::Backend::Vulkan);
        assert!(
            !features.contains(wgpu::Features::PIPELINE_CACHE) && !enabled.pipeline_cache,
            "Vulkan without the adapter feature must not phantom-enable PIPELINE_CACHE"
        );
    }
}
