use vyre_driver::DispatchConfig;
use vyre_foundation::ir::Program;
use vyre_foundation::serial::wire::framing::WIRE_FORMAT_VERSION;

pub(super) const NAGA_VERSION: &str = env!("VYRE_NAGA_VERSION");
pub(super) const WGSL_LOWERING_CONTRACT: &str =
    "vyre-wgpu-lowering-contract:v17:region-phi-named-carrier+ssa-carrier-snapshots+block-shadowed-carriers+carrier-rebind-invalidates-stale-blocks+restored-loop-and-block-carrier-scope+nonfinite-f32-bitcast+per-word-byte-compact+no-mutable-loop-unroll+licm-keeps-reassigned-loop-locals+runtime-storage-buffer-lengths+saturating-f32-to-int-cast";

pub(crate) struct CompiledPipelineCacheKey {
    pub(crate) hash: [u8; 32],
    pub(crate) adapter_fingerprint: String,
    pub(crate) cache_key: String,
    pub(crate) wgsl_blake3: String,
}

pub(crate) fn early_pipeline_cache_key(
    program: &Program,
    adapter_info: &wgpu::AdapterInfo,
    config: &DispatchConfig,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-early-pipeline-cache-v1\0program\0");
    hasher.update(&program.fingerprint());
    hasher.update(b"\0adapter\0");
    update_adapter_fingerprint(&mut hasher, adapter_info);
    hasher.update(b"\0abi\0");
    hasher.update(&WIRE_FORMAT_VERSION.to_le_bytes());
    hasher.update(b"\0naga\0");
    hasher.update(NAGA_VERSION.as_bytes());
    update_wgsl_lowering_contract(&mut hasher);
    hasher.update(b"\0policy\0");
    vyre_driver::update_dispatch_policy_cache_hash(&mut hasher, config);
    hasher.update(b"\0workgroup_override\0");
    if let Some(wg) = config.workgroup_override {
        for axis in wg {
            hasher.update(&axis.to_le_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

pub(crate) fn compiled_pipeline_cache_key(
    adapter_info: &wgpu::AdapterInfo,
    wgsl_source: &str,
) -> CompiledPipelineCacheKey {
    let adapter_fingerprint = adapter_fingerprint(adapter_info);
    let wgsl_blake3 = blake3_hex(wgsl_source.as_bytes());
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-compiled-pipeline-cache-v1\0");
    hasher.update(adapter_fingerprint.as_bytes());
    hasher.update(b"\0abi\0");
    hasher.update(&WIRE_FORMAT_VERSION.to_le_bytes());
    hasher.update(b"\0wgsl\0");
    hasher.update(wgsl_blake3.as_bytes());
    hasher.update(b"\0naga\0");
    hasher.update(NAGA_VERSION.as_bytes());
    let hash = *hasher.finalize().as_bytes();
    let cache_key = hex_hash(&hash);
    CompiledPipelineCacheKey {
        hash,
        adapter_fingerprint,
        cache_key,
        wgsl_blake3,
    }
}

pub(crate) fn wgsl_cache_key(
    norm_digest: &[u8],
    fingerprint: &str,
    config: &DispatchConfig,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-pipeline-cache-v7\0norm\0");
    hasher.update(norm_digest);
    hasher.update(b"\0adapter\0");
    hasher.update(fingerprint.as_bytes());
    hasher.update(b"\0abi\0");
    hasher.update(&WIRE_FORMAT_VERSION.to_le_bytes());
    hasher.update(b"\0naga\0");
    hasher.update(NAGA_VERSION.as_bytes());
    update_wgsl_lowering_contract(&mut hasher);
    hasher.update(b"\0policy\0");
    vyre_driver::update_dispatch_policy_cache_hash(&mut hasher, config);
    *hasher.finalize().as_bytes()
}

pub(super) fn update_wgsl_lowering_contract(hasher: &mut blake3::Hasher) {
    hasher.update(b"\0wgsl_lowering_contract\0");
    hasher.update(WGSL_LOWERING_CONTRACT.as_bytes());
}

pub(super) fn adapter_fingerprint(adapter_info: &wgpu::AdapterInfo) -> String {
    let mut fingerprint = String::new();
    fingerprint.push_str(adapter_backend_name(adapter_info.backend));
    fingerprint.push(':');
    push_hex_u32(&mut fingerprint, adapter_info.vendor);
    fingerprint.push(':');
    push_hex_u32(&mut fingerprint, adapter_info.device);
    fingerprint.push(':');
    fingerprint.push_str(&adapter_info.driver);
    fingerprint.push(':');
    fingerprint.push_str(&adapter_info.driver_info);
    fingerprint
}

fn adapter_backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Noop => "Noop",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "Dx12",
        wgpu::Backend::Gl => "Gl",
        wgpu::Backend::BrowserWebGpu => "BrowserWebGpu",
    }
}

fn update_adapter_fingerprint(hasher: &mut blake3::Hasher, adapter_info: &wgpu::AdapterInfo) {
    hasher.update(adapter_backend_name(adapter_info.backend).as_bytes());
    hasher.update(b"\0");
    hasher.update(&adapter_info.vendor.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(&adapter_info.device.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(adapter_info.driver.as_bytes());
    hasher.update(b"\0");
    hasher.update(adapter_info.driver_info.as_bytes());
}

pub(super) fn blake3_hex(bytes: &[u8]) -> String {
    hex_hash(blake3::hash(bytes).as_bytes())
}

pub(super) fn metadata_fingerprint(value: &str) -> [u8; 32] {
    *blake3::hash(value.as_bytes()).as_bytes()
}

pub(super) fn path_fingerprint(path: &std::path::Path) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-pipeline-cache-path-v1\0");
    hasher.update(path.as_os_str().as_encoded_bytes());
    let hex = hex_hash(hasher.finalize().as_bytes());
    format!("cache-path:{}", &hex[..16])
}

pub(super) fn hex_hash(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hex = [0_u8; 64];
    for (index, byte) in bytes.iter().enumerate() {
        let offset = index * 2;
        hex[offset] = HEX[hex_nibble_index(byte >> 4)];
        hex[offset + 1] = HEX[hex_nibble_index(byte & 0x0f)];
    }
    String::from_utf8_lossy(&hex).into_owned()
}

fn push_hex_u32(out: &mut String, value: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in value.to_be_bytes() {
        out.push(HEX[hex_nibble_index(byte >> 4)] as char);
        out.push(HEX[hex_nibble_index(byte & 0x0f)] as char);
    }
}

fn hex_nibble_index(nibble: u8) -> usize {
    debug_assert!(
        nibble < 16,
        "pipeline disk-cache hex encoding received a non-nibble byte"
    );
    usize::from(nibble)
}
