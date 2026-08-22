//! Hold the CUDA-first, WGPU-fallback backend policy to the tree and the probe.

use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

use serde::Serialize;
use vyre_driver::{
    acquire, acquire_preferred_dispatch_backend, backend_dispatches, backend_precedence,
};
use xtask::artifact_gate::Inspection;

const MAX_BACKEND_EVIDENCE_TEXT_BYTES: u64 = 4_194_304;

#[derive(Debug, Serialize)]
struct BackendMatrix {
    schema_version: u32,
    cuda_first: bool,
    wgpu_fallback_present: bool,
    preferred_backend_id: Option<String>,
    preferred_backend_gpu_only: bool,
    gpu_probe: GpuProbe,
    cuda_feature_markers: Vec<BackendFeatureMarker>,
    wgpu_feature_markers: Vec<BackendFeatureMarker>,
    capability_rows: Vec<BackendCapabilityRow>,
    hidden_fallback_findings: Vec<BackendSourceFinding>,
    hidden_fallback_scan_errors: Vec<String>,
    backends: Vec<BackendEntry>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GpuProbe {
    nvidia_smi_ok: bool,
    nvidia_smi_devices: Vec<String>,
    nvidia_smi_device_details: Vec<GpuProbeDevice>,
    nvidia_driver_version: Option<String>,
    nvidia_cuda_version: Option<String>,
    nvidia_smi_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct GpuProbeDevice {
    name: String,
    driver_version: String,
    memory_total_mib: Option<u64>,
    compute_capability_major: Option<u32>,
    compute_capability_minor: Option<u32>,
}

#[derive(Debug, Serialize)]
struct BackendEntry {
    id: String,
    precedence: u32,
    dispatches: bool,
    acquire_ok: bool,
    acquire_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BackendFeatureMarker {
    id: &'static str,
    path: String,
    exists: bool,
    read_error: Option<String>,
    source_bytes: usize,
    implementation_tokens: Vec<&'static str>,
    missing_tokens: Vec<&'static str>,
    unresolved_markers: Vec<&'static str>,
    role: &'static str,
}

#[derive(Debug, Serialize)]
struct BackendSourceFinding {
    path: String,
    line: usize,
    pattern: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct BackendCapabilityRow {
    backend_id: String,
    capability_id: String,
    probe_source: String,
    probed_value: Option<String>,
    supported: bool,
    unsupported_reason: Option<String>,
    fix: String,
}

struct BackendFeatureRequirement {
    id: &'static str,
    relative: &'static str,
    role: &'static str,
    tokens: &'static [&'static str],
}

const CUDA_FEATURE_MARKERS: &[BackendFeatureRequirement] = &[
    BackendFeatureRequirement {
        id: "tensor-core-fragment",
        relative: "vyre-emit-ptx/src/patterns/tensor_core_fragment/mod.rs",
        role: "Tensor-core/MMA lowering pattern",
        tokens: &["mma", "fragment"],
    },
    BackendFeatureRequirement {
        id: "ldmatrix-cp-async",
        relative: "vyre-emit-ptx/src/patterns/ldmatrix_cp_async/mod.rs",
        role: "Ampere+ async global-to-shared staging pattern",
        tokens: &[
            "supports_async_copy",
            "supports_ldmatrix",
            "KernelOpKind::StoreShared",
        ],
    },
    BackendFeatureRequirement {
        id: "predicated-execution",
        relative: "vyre-emit-ptx/src/patterns/predicated_execution/mod.rs",
        role: "Predicated execution pattern",
        tokens: &["predicate", "predicated"],
    },
    BackendFeatureRequirement {
        id: "instruction-scheduling",
        relative: "vyre-emit-ptx/src/patterns/instruction_scheduling/mod.rs",
        role: "PTX instruction scheduling pattern",
        tokens: &["schedule", "latency"],
    },
    BackendFeatureRequirement {
        id: "ptx-vector-load-gap-scheduling",
        relative: "vyre-emit-ptx/src/emitter/body.rs",
        role: "PTX fused vector-load latency-gap scheduling",
        tokens: &["vector-load gap", "find_latency_filler_avoiding_results"],
    },
    BackendFeatureRequirement {
        id: "ptx-compute-load-gap-scheduling",
        relative: "vyre-emit-ptx/src/emitter/schedule.rs",
        role: "PTX load-use latency-gap scheduling with independent compute fillers",
        tokens: &[
            "KernelOpKind::Fma",
            "KernelOpKind::MatrixMma",
            "KernelOpKind::SubgroupReduce",
        ],
    },
    BackendFeatureRequirement {
        id: "ptx-vector-load-fusion",
        relative: "vyre-emit-ptx/src/emitter/vector.rs",
        role: "PTX vector load fusion pattern",
        tokens: &["ld.global", "v4"],
    },
    BackendFeatureRequirement {
        id: "ptx-vector-store-fusion",
        relative: "vyre-emit-ptx/src/emitter/vector.rs",
        role: "PTX vector store fusion pattern",
        tokens: &["st.global", "v4"],
    },
    BackendFeatureRequirement {
        id: "async-copy-emitter",
        relative: "vyre-emit-ptx/src/emitter/async_copy.rs",
        role: "PTX async copy emitter",
        tokens: &["cp.async", "commit_group"],
    },
    BackendFeatureRequirement {
        id: "mma-emitter",
        relative: "vyre-emit-ptx/src/emitter/mma.rs",
        role: "PTX MMA emitter",
        tokens: &["mma", "sync"],
    },
    BackendFeatureRequirement {
        id: "cuda-resident-dispatch",
        relative: "vyre-driver-cuda/src/backend/resident_dispatch/mod.rs",
        role: "CUDA resident dispatch release path",
        tokens: &["dispatch_resident", "ptx"],
    },
    BackendFeatureRequirement {
        id: "cuda-resident-io",
        relative: "vyre-driver-cuda/src/backend/resident_io.rs",
        role: "CUDA resident input buffer uploads and device pointers",
        tokens: &["upload_resident_at_many", "resident_device_ptr"],
    },
    BackendFeatureRequirement {
        id: "cuda-resident-readback",
        relative: "vyre-driver-cuda/src/backend/resident_io_download.rs",
        role: "CUDA resident sparse readback batching",
        tokens: &[
            "download_resident_readbacks_many",
            "download_resident_readbacks_many_into",
        ],
    },
    BackendFeatureRequirement {
        id: "cuda-graph-launch",
        relative: "vyre-driver-cuda/src/backend/cuda_graph.rs",
        role: "CUDA graph launch path",
        tokens: &["record_cuda_graph", "cugraph"],
    },
    BackendFeatureRequirement {
        id: "cuda-module-cache",
        relative: "vyre-driver-cuda/src/backend/module_cache/module_registry.rs",
        role: "CUDA PTX module cache",
        tokens: &["function_for_ptx", "ptx_target_sm"],
    },
    BackendFeatureRequirement {
        id: "cuda-ptx-source-cache",
        relative: "vyre-driver-cuda/src/backend/module_cache/ptx_source_cache.rs",
        role: "CUDA PTX source cache before module load",
        tokens: &[
            "CudaPtxSourceCache",
            "CudaPtxSourceCacheSnapshot",
            "get_or_lower",
            "snapshot",
            "PTX_SOURCE_CACHE_SOFT_CAP",
            "evict_submodular",
        ],
    },
    BackendFeatureRequirement {
        id: "cuda-ptx-target-probe",
        relative: "vyre-driver-cuda/src/backend/ptx_target.rs",
        role: "CUDA loadable PTX target probing",
        tokens: &["select_loadable_ptx_target_sm", "cumoduleloaddata"],
    },
    BackendFeatureRequirement {
        id: "megakernel-paired-speculation",
        relative: "vyre-runtime/src/resident_work_queue/speculation.rs",
        role: "Megakernel paired speculative execution adoption policy",
        tokens: &[
            "PairedSpeculationWindow",
            "record_sample",
            "side_compile_cost_ns",
            "decide_speculation",
        ],
    },
];

const WGPU_FEATURE_MARKERS: &[BackendFeatureRequirement] = &[
    BackendFeatureRequirement {
        id: "wgpu-artifact-materializer",
        relative: "vyre-driver-wgpu/src/materializer.rs",
        role: "WGPU authenticated artifact materialization",
        tokens: &[
            "impl ArtifactMaterializer",
            "fn materialize",
            "TargetPayload",
        ],
    },
    BackendFeatureRequirement {
        id: "runtime-artifact-admission",
        relative: "vyre-runtime/src/artifact_admission/mod.rs",
        role: "Canonical runtime artifact admission and materialization",
        tokens: &["ArtifactSession", "materialize"],
    },
    BackendFeatureRequirement {
        id: "wgpu-readback-ring",
        relative: "vyre-driver-wgpu/src/runtime/readback_ring/ring.rs",
        role: "WGPU sparse/readback ring",
        tokens: &["ring", "readback"],
    },
    BackendFeatureRequirement {
        id: "wgpu-async-dispatch-prefetch",
        relative: "vyre-driver-wgpu/src/async_dispatch.rs",
        role: "WGPU non-blocking dispatch with predicted pipeline prefetch",
        tokens: &["dispatch_borrowed_async", "PipelinePrefetch"],
    },
    BackendFeatureRequirement {
        id: "wgpu-dispatch-scratch-reuse",
        relative: "vyre-driver-wgpu/src/engine/dispatch_scratch.rs",
        role: "WGPU dispatch hot-path scratch arena reuse",
        tokens: &["thread_local", "reset"],
    },
    BackendFeatureRequirement {
        id: "wgpu-disk-cache",
        relative: "vyre-driver-wgpu/src/pipeline/disk_cache/mod.rs",
        role: "WGPU pipeline disk cache",
        tokens: &["cache", "pipeline", "MAX_PENDING_DURABLE_CACHE_FILES"],
    },
    BackendFeatureRequirement {
        id: "wgpu-megakernel-dispatcher",
        relative: "vyre-driver-wgpu/src/pipeline/persistent.rs",
        role: "WGPU batched megakernel dispatch through persistent bindings",
        tokens: &[
            "dispatch_persistent_batched",
            "dispatch_borrowed_persistent_batched",
            "DispatchItem",
        ],
    },
    BackendFeatureRequirement {
        id: "wgpu-no-cpu-fallback",
        relative: "vyre-driver-wgpu/src/runtime/device/selector.rs",
        role: "WGPU adapter selection that refuses a CPU adapter",
        tokens: &["has_real_gpu_adapter", "DeviceType::Cpu"],
    },
    BackendFeatureRequirement {
        id: "megakernel-paired-speculation",
        relative: "vyre-runtime/src/resident_work_queue/speculation.rs",
        role: "Megakernel paired speculative execution adoption policy",
        tokens: &[
            "PairedSpeculationWindow",
            "record_sample",
            "side_compile_cost_ns",
            "decide_speculation",
        ],
    },
];

/// The feature marker ids the CUDA matrix emits, in declaration order.
///
/// The check that judges a recorded matrix reads this rather than a second
/// list: a marker added here has to appear in the artifact, and a marker
/// deleted here stops being required, with no third place to update.
#[must_use]
pub fn cuda_feature_marker_ids() -> Vec<&'static str> {
    CUDA_FEATURE_MARKERS
        .iter()
        .map(|requirement| requirement.id)
        .collect()
}

/// The feature marker ids the WGPU matrix emits, in declaration order.
#[must_use]
pub fn wgpu_feature_marker_ids() -> Vec<&'static str> {
    WGPU_FEATURE_MARKERS
        .iter()
        .map(|requirement| requirement.id)
        .collect()
}

const UNRESOLVED_MARKERS: &[&str] = &[
    "todo",
    "fixme",
    "placeholder",
    "stub",
    "todo!",
    "unimplemented!",
    "panic!(\"not implemented",
    "tbd",
];

const HIDDEN_FALLBACK_PATTERNS: &[&str] = &[
    "cpu fallback",
    "software fallback",
    "fallback dispatch",
    "falling back to cpu",
    "fallback to cpu",
];

const BACKEND_PRODUCTION_SCAN_ROOTS: &[&str] = &[
    "vyre-driver/src",
    "vyre-driver-cuda/src",
    "vyre-driver-wgpu/src",
    "vyre-runtime/src",
];

xtask::artifact_gate! {
    /// Holds the backend release policy evidence to the tree and to the recorded probe.
    BackendMatrixGate,
    name: "backend-matrix",
    help: "Judge the CUDA-first, WGPU-fallback backend policy. Proves, on any host, that every \
       backend implementation file the policy names exists and carries its implementation \
       tokens with no unresolved marker left in it, and that no backend production source \
       states a hidden fallback. Proves, from the recorded probe, that CUDA acquires first, \
       that the WGPU fallback acquires, that the preferred dispatch backend is never the \
       reference one, and that the host met the release GPU floor. The probe is only as \
       current as the run that recorded it; --write re-probes this host and rewrites the \
       artifact.",
    inspect: |ctx| inspect(&ctx.root),
}

/// The artifact this gate owns, relative to the workspace root.
const ARTIFACT: &str = "release/evidence/backends/backend-matrix.json";

/// Backend facts that come out of the source tree, with no device present.
///
/// These are checked on every run, including on a host with no GPU, because a
/// backend file that lost its implementation tokens is a defect of the tree and
/// not of the machine the evidence was recorded on.
struct SourceScan {
    cuda_feature_markers: Vec<BackendFeatureMarker>,
    wgpu_feature_markers: Vec<BackendFeatureMarker>,
    hidden_fallback_findings: Vec<BackendSourceFinding>,
    hidden_fallback_scan_errors: Vec<String>,
    blockers: Vec<String>,
}

fn scan_backend_sources(workspace_root: &Path) -> SourceScan {
    let mut blockers = Vec::new();
    let cuda_feature_markers = collect_cuda_feature_markers(workspace_root, &mut blockers);
    let wgpu_feature_markers =
        collect_feature_markers(workspace_root, WGPU_FEATURE_MARKERS, &mut blockers);
    let (hidden_fallback_findings, hidden_fallback_scan_errors) =
        scan_hidden_fallback_language(workspace_root, &mut blockers);
    for finding in &hidden_fallback_findings {
        blockers.push(format!(
            "backend production source `{}`:{} contains hidden fallback language `{}`",
            finding.path, finding.line, finding.pattern
        ));
    }
    SourceScan {
        cuda_feature_markers,
        wgpu_feature_markers,
        hidden_fallback_findings,
        hidden_fallback_scan_errors,
        blockers,
    }
}

fn inspect(workspace_root: &Path) -> Inspection {
    let mut inspection = Inspection::new();
    let scan = scan_backend_sources(workspace_root);
    for blocker in &scan.blockers {
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Repair the backend source the sentence names. This is read from the tree, so it is \
             true on every host and no re-probe clears it.",
        );
    }
    let (matrix, device_blockers) = probe_backend_matrix(scan);
    for blocker in &device_blockers {
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Repair the driver, the registry precedence or the device the sentence names, \
             then rerun with --write.",
        );
    }
    inspection.generates(ARTIFACT, &matrix);
    inspection
}
/// Probe this host's backend registry and devices, folding in the source scan.
///
/// The registry lookups used to panic. A gate that panics reports nothing at
/// all, so a registry that fails to start is a blocker like any other.
fn probe_backend_matrix(scan: SourceScan) -> (BackendMatrix, Vec<String>) {
    let mut blockers = Vec::new();
    let mut backends = Vec::new();
    match vyre_registry_link::backend::live_backend_registry_by_precedence() {
        Ok(registrations) => {
            for registration in registrations {
                let dispatches = match backend_dispatches(registration.id) {
                    Ok(dispatches) => dispatches,
                    Err(error) => {
                        blockers.push(format!(
                            "backend `{}` dispatch lookup failed: {error}",
                            registration.id
                        ));
                        false
                    }
                };
                let precedence = match backend_precedence(registration.id) {
                    Ok(precedence) => precedence,
                    Err(error) => {
                        blockers.push(format!(
                            "backend `{}` precedence lookup failed: {error}",
                            registration.id
                        ));
                        u32::MAX
                    }
                };
                let (acquire_ok, acquire_error) = match acquire(registration.id) {
                    Ok(_) => (true, None),
                    Err(error) => (false, Some(error.to_string())),
                };
                backends.push(BackendEntry {
                    id: registration.id.to_string(),
                    precedence,
                    dispatches,
                    acquire_ok,
                    acquire_error,
                });
            }
        }
        Err(error) => blockers.push(format!("backend registry startup failed: {error}")),
    }

    let cuda = backends.iter().find(|backend| backend.id == "cuda");
    let wgpu = backends.iter().find(|backend| backend.id == "wgpu");
    let preferred_backend = acquire_preferred_dispatch_backend();
    let preferred_backend_id = preferred_backend
        .as_ref()
        .ok()
        .map(|backend| backend.id().to_string());
    let preferred_backend_gpu_only = preferred_backend_id
        .as_deref()
        .is_some_and(|id| matches!(id, "cuda" | "wgpu"));
    let cuda_first = match (cuda, wgpu) {
        (Some(cuda), Some(wgpu)) => {
            cuda.dispatches && cuda.acquire_ok && cuda.precedence < wgpu.precedence
        }
        (Some(cuda), None) => cuda.dispatches && cuda.acquire_ok,
        _ => false,
    };
    let wgpu_fallback_present =
        wgpu.is_some_and(|backend| backend.dispatches && backend.acquire_ok);
    if !cuda_first {
        blockers.push(
            "CUDA is not the first acquired dispatch backend. Fix: link/configure CUDA and give it higher precedence than WGPU.".to_string(),
        );
    }
    if !wgpu_fallback_present {
        blockers.push(
            "WGPU fallback is not present and acquireable. Fix: link/configure vyre-driver-wgpu."
                .to_string(),
        );
    }
    if !preferred_backend_gpu_only {
        let detail = preferred_backend_id.as_deref().map_or_else(
            || {
                preferred_backend
                    .as_ref()
                    .err()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| {
                        "preferred backend acquisition returned no backend".to_string()
                    })
            },
            |id| format!("preferred backend was `{id}`"),
        );
        blockers.push(format!(
            "preferred runtime backend is not GPU-only ({detail}). Fix: acquire_preferred_dispatch_backend must never select cpu-ref/reference as an implicit fallback."
        ));
    }
    let gpu_probe = probe_nvidia_smi();
    if !gpu_probe.nvidia_smi_ok {
        blockers.push(
            "nvidia-smi -L did not report a GPU. Fix: repair CUDA/NVIDIA driver visibility before release benchmarking."
                .to_string(),
        );
    }
    if !gpu_probe
        .nvidia_smi_device_details
        .iter()
        .any(gpu_probe_device_meets_release_floor)
    {
        let (major, minor) = crate::gpu_release_floor::RELEASE_COMPUTE_CAPABILITY_FLOOR;
        blockers.push(format!(
            "nvidia-smi did not report a CUDA GPU meeting the release floor: >={} MiB VRAM and compute capability >={major}.{minor}",
            crate::gpu_release_floor::min_cuda_release_memory_mib()
        ));
    }
    let capability_rows = collect_backend_capability_rows(&backends, &gpu_probe);
    blockers.extend(capability_contract_blockers(&capability_rows));

    let device_blockers = blockers.clone();
    blockers.extend(scan.blockers);
    let matrix = BackendMatrix {
        schema_version: 3,
        cuda_first,
        wgpu_fallback_present,
        preferred_backend_id,
        preferred_backend_gpu_only,
        gpu_probe,
        cuda_feature_markers: scan.cuda_feature_markers,
        wgpu_feature_markers: scan.wgpu_feature_markers,
        capability_rows,
        hidden_fallback_findings: scan.hidden_fallback_findings,
        hidden_fallback_scan_errors: scan.hidden_fallback_scan_errors,
        backends,
        blockers,
    };
    (matrix, device_blockers)
}

fn collect_cuda_feature_markers(
    workspace_root: &Path,
    blockers: &mut Vec<String>,
) -> Vec<BackendFeatureMarker> {
    collect_feature_markers(workspace_root, CUDA_FEATURE_MARKERS, blockers)
}

fn collect_feature_markers(
    workspace_root: &Path,
    requirements: &'static [BackendFeatureRequirement],
    blockers: &mut Vec<String>,
) -> Vec<BackendFeatureMarker> {
    let mut markers = Vec::new();
    for requirement in requirements {
        let path = workspace_root.join(requirement.relative);
        let exists = path.is_file();
        let (text, read_error) = if exists {
            match read_marker_module(&path) {
                Ok(text) => (text, None),
                Err(error) => {
                    blockers.push(format!(
                        "backend feature marker `{}` could not be read at {}: {error}",
                        requirement.id,
                        path.display()
                    ));
                    (String::new(), Some(error.to_string()))
                }
            }
        } else {
            (String::new(), None)
        };
        let lowered = text.to_ascii_lowercase();
        let code_lowered = implementation_source(&text).to_ascii_lowercase();
        let missing_tokens = requirement
            .tokens
            .iter()
            .copied()
            .filter(|token| !code_lowered.contains(&token.to_ascii_lowercase()))
            .collect::<Vec<_>>();
        let unresolved_markers = UNRESOLVED_MARKERS
            .iter()
            .copied()
            .filter(|marker| lowered.contains(marker))
            .collect::<Vec<_>>();
        if !exists {
            blockers.push(format!(
                "backend feature marker `{}` is missing at {}",
                requirement.id,
                path.display()
            ));
        } else if text.trim().is_empty() {
            blockers.push(format!(
                "backend feature marker `{}` is empty",
                requirement.id
            ));
        }
        for token in &missing_tokens {
            blockers.push(format!(
                "backend feature marker `{}` does not contain implementation token `{token}`",
                requirement.id
            ));
        }
        for marker in &unresolved_markers {
            blockers.push(format!(
                "backend feature marker `{}` contains unresolved marker `{marker}`",
                requirement.id
            ));
        }
        markers.push(BackendFeatureMarker {
            id: requirement.id,
            path: requirement.relative.to_string(),
            exists,
            read_error,
            source_bytes: text.len(),
            implementation_tokens: requirement.tokens.to_vec(),
            missing_tokens,
            unresolved_markers,
            role: requirement.role,
        });
    }
    markers
}

fn collect_backend_capability_rows(
    backends: &[BackendEntry],
    gpu_probe: &GpuProbe,
) -> Vec<BackendCapabilityRow> {
    let cuda = backends.iter().find(|backend| backend.id == "cuda");
    let wgpu = backends.iter().find(|backend| backend.id == "wgpu");
    let mut rows = Vec::new();
    rows.push(registry_capability_row("cuda", cuda));
    rows.push(registry_capability_row("wgpu", wgpu));

    let cuda_sm = highest_cuda_compute_capability(gpu_probe);
    let cuda_sm_supported = cuda_sm.is_some_and(|(major, minor)| (major, minor) >= (8, 0));
    rows.push(BackendCapabilityRow {
        backend_id: "cuda".to_string(),
        capability_id: "live-sm-release-floor".to_string(),
        probe_source: "nvidia-smi --query-gpu=compute_cap".to_string(),
        probed_value: cuda_sm.map(|(major, minor)| format!("sm_{major}{minor}")),
        supported: cuda_sm_supported,
        unsupported_reason: (!cuda_sm_supported).then(|| {
            "no live NVIDIA device reported compute capability >= 8.0".to_string()
        }),
        fix: "Fix: repair CUDA driver/device visibility or route release benchmarks to a supported NVIDIA GPU.".to_string(),
    });

    let max_memory = max_cuda_memory_mib(gpu_probe);
    let memory_floor_mib = crate::gpu_release_floor::min_cuda_release_memory_mib();
    let memory_supported = max_memory.is_some_and(|mib| mib >= memory_floor_mib);
    rows.push(BackendCapabilityRow {
        backend_id: "cuda".to_string(),
        capability_id: "live-memory-release-floor".to_string(),
        probe_source: "nvidia-smi --query-gpu=memory.total".to_string(),
        probed_value: max_memory.map(|mib| format!("{mib} MiB")),
        supported: memory_supported,
        unsupported_reason: (!memory_supported)
            .then(|| format!("no live NVIDIA device reported >={memory_floor_mib} MiB memory")),
        fix: format!(
            "Fix: run release benchmark evidence on a CUDA GPU with at least {memory_floor_mib} MiB VRAM, which is the largest registered workload plus its CUDA context."
        ),
    });

    let warp_supported = gpu_probe.nvidia_smi_ok && cuda_sm.is_some();
    rows.push(BackendCapabilityRow {
        backend_id: "cuda".to_string(),
        capability_id: "warp-width-contract".to_string(),
        probe_source: "CUDA warp-size contract gated by live NVIDIA device probe".to_string(),
        probed_value: warp_supported.then(|| "32 lanes".to_string()),
        supported: warp_supported,
        unsupported_reason: (!warp_supported).then(|| {
            "CUDA warp-width contract is unavailable without a live NVIDIA GPU probe".to_string()
        }),
        fix: "Fix: expose a live CUDA device before using warp-width-sensitive benchmark claims."
            .to_string(),
    });

    rows.push(BackendCapabilityRow {
        backend_id: "cuda".to_string(),
        capability_id: "mlir-transform-support".to_string(),
        probe_source: "backend-matrix transform support registry".to_string(),
        probed_value: None,
        supported: false,
        unsupported_reason: Some(
            "CUDA backend does not expose a live MLIR transform-dialect capability probe"
                .to_string(),
        ),
        fix: "Fix: wire transform capability probing before claiming transform-scheduled CUDA lowering.".to_string(),
    });

    rows.push(BackendCapabilityRow {
        backend_id: "wgpu".to_string(),
        capability_id: "adapter-live-acquire".to_string(),
        probe_source: "vyre_driver::acquire(\"wgpu\")".to_string(),
        probed_value: wgpu.map(|backend| {
            format!(
                "dispatches={},acquire_ok={},precedence={}",
                backend.dispatches, backend.acquire_ok, backend.precedence
            )
        }),
        supported: wgpu.is_some_and(|backend| backend.dispatches && backend.acquire_ok),
        unsupported_reason: (!wgpu.is_some_and(|backend| backend.dispatches && backend.acquire_ok))
            .then(|| {
                wgpu.and_then(|backend| backend.acquire_error.clone())
                    .unwrap_or_else(|| "wgpu backend is not registered or acquireable".to_string())
            }),
        fix: "Fix: configure vyre-driver-wgpu so fallback evidence is backed by an acquireable adapter.".to_string(),
    });

    rows.push(BackendCapabilityRow {
        backend_id: "wgpu".to_string(),
        capability_id: "mlir-transform-support".to_string(),
        probe_source: "backend-matrix transform support registry".to_string(),
        probed_value: None,
        supported: false,
        unsupported_reason: Some(
            "WGPU backend does not expose a live MLIR transform-dialect capability probe"
                .to_string(),
        ),
        fix: "Fix: wire transform capability probing before claiming transform-scheduled WGPU lowering.".to_string(),
    });

    rows
}

fn registry_capability_row(
    backend_id: &str,
    backend: Option<&BackendEntry>,
) -> BackendCapabilityRow {
    let supported = backend.is_some_and(|backend| backend.dispatches && backend.acquire_ok);
    BackendCapabilityRow {
        backend_id: backend_id.to_string(),
        capability_id: "registered-dispatch-backend".to_string(),
        probe_source: "vyre_driver::backend registry plus acquire()".to_string(),
        probed_value: backend.map(|backend| {
            format!(
                "dispatches={},acquire_ok={},precedence={}",
                backend.dispatches, backend.acquire_ok, backend.precedence
            )
        }),
        supported,
        unsupported_reason: (!supported).then(|| {
            backend
                .and_then(|backend| backend.acquire_error.clone())
                .unwrap_or_else(|| format!("{backend_id} backend is not dispatchable/acquireable"))
        }),
        fix: format!(
            "Fix: register and configure `{backend_id}` before publishing backend support claims."
        ),
    }
}

fn highest_cuda_compute_capability(gpu_probe: &GpuProbe) -> Option<(u32, u32)> {
    gpu_probe
        .nvidia_smi_device_details
        .iter()
        .filter_map(|device| {
            Some((
                device.compute_capability_major?,
                device.compute_capability_minor?,
            ))
        })
        .max()
}

fn max_cuda_memory_mib(gpu_probe: &GpuProbe) -> Option<u64> {
    gpu_probe
        .nvidia_smi_device_details
        .iter()
        .filter_map(|device| device.memory_total_mib)
        .max()
}

/// Whether a live `nvidia-smi` device clears the release floors.
///
/// The comparison itself belongs to `crate::gpu_release_floor`; this only
/// widens the probe's `u32` capability pair to the shape that owner takes.
fn gpu_probe_device_meets_release_floor(device: &GpuProbeDevice) -> bool {
    let compute_capability = device
        .compute_capability_major
        .zip(device.compute_capability_minor)
        .map(|(major, minor)| (u64::from(major), u64::from(minor)));
    crate::gpu_release_floor::device_meets_release_floor(
        device.memory_total_mib,
        compute_capability,
    )
}

fn capability_contract_blockers(rows: &[BackendCapabilityRow]) -> Vec<String> {
    let mut blockers = Vec::new();
    if rows.is_empty() {
        blockers.push("backend capability matrix emitted zero capability rows".to_string());
        return blockers;
    }
    let mut seen = std::collections::BTreeSet::new();
    for row in rows {
        if row.backend_id.trim().is_empty() {
            blockers.push("backend capability row has blank backend_id".to_string());
        }
        if row.capability_id.trim().is_empty() {
            blockers.push(format!(
                "backend capability row for `{}` has blank capability_id",
                row.backend_id
            ));
        }
        if !seen.insert((row.backend_id.clone(), row.capability_id.clone())) {
            blockers.push(format!(
                "backend capability row duplicates `{}`/`{}`",
                row.backend_id, row.capability_id
            ));
        }
        if row.probe_source.trim().is_empty() {
            blockers.push(format!(
                "backend capability `{}`/`{}` has no probe_source",
                row.backend_id, row.capability_id
            ));
        }
        if row.supported && row.probed_value.as_deref().is_none_or(str::is_empty) {
            blockers.push(format!(
                "backend capability `{}`/`{}` is supported but has no probed_value",
                row.backend_id, row.capability_id
            ));
        }
        if !row.supported
            && row
                .unsupported_reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty())
        {
            blockers.push(format!(
                "backend capability `{}`/`{}` is unsupported but has no unsupported_reason",
                row.backend_id, row.capability_id
            ));
        }
        if row.fix.trim().is_empty() || !row.fix.starts_with("Fix:") {
            blockers.push(format!(
                "backend capability `{}`/`{}` must include actionable Fix text",
                row.backend_id, row.capability_id
            ));
        }
        let assumption_text = format!(
            "{} {} {}",
            row.probe_source,
            row.probed_value.as_deref().unwrap_or_default(),
            row.fix
        )
        .to_ascii_lowercase();
        if assumption_text.contains("hardcoded") || assumption_text.contains("assume gpu") {
            blockers.push(format!(
                "backend capability `{}`/`{}` contains hardcoded capability language",
                row.backend_id, row.capability_id
            ));
        }
    }
    blockers
}

fn scan_hidden_fallback_language(
    workspace_root: &Path,
    blockers: &mut Vec<String>,
) -> (Vec<BackendSourceFinding>, Vec<String>) {
    let mut findings = Vec::new();
    let mut scan_errors = Vec::new();
    for root in BACKEND_PRODUCTION_SCAN_ROOTS {
        scan_hidden_fallback_dir(
            &workspace_root.join(root),
            &mut findings,
            &mut scan_errors,
            blockers,
        );
    }
    (findings, scan_errors)
}

fn scan_hidden_fallback_dir(
    root: &Path,
    findings: &mut Vec<BackendSourceFinding>,
    scan_errors: &mut Vec<String>,
    blockers: &mut Vec<String>,
) {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            let message = format!(
                "hidden fallback scan could not read directory `{}`: {error}",
                root.display()
            );
            blockers.push(message.clone());
            scan_errors.push(message);
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                let message = format!(
                    "hidden fallback scan could not read entry in `{}`: {error}",
                    root.display()
                );
                blockers.push(message.clone());
                scan_errors.push(message);
                continue;
            }
        };
        let path = entry.path();
        if path.is_dir() {
            scan_hidden_fallback_dir(&path, findings, scan_errors, blockers);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        scan_hidden_fallback_file(&path, findings, scan_errors, blockers);
    }
}

fn scan_hidden_fallback_file(
    path: &Path,
    findings: &mut Vec<BackendSourceFinding>,
    scan_errors: &mut Vec<String>,
    blockers: &mut Vec<String>,
) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            let message = format!(
                "hidden fallback scan could not read source `{}`: {error}",
                path.display()
            );
            blockers.push(message.clone());
            scan_errors.push(message);
            return;
        }
    };
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        let lowered = line.to_ascii_lowercase();
        for &pattern in HIDDEN_FALLBACK_PATTERNS {
            if lowered.contains(pattern) {
                findings.push(BackendSourceFinding {
                    path: path.display().to_string(),
                    line: line_index + 1,
                    pattern,
                });
            }
        }
    }
}

/// The implementation text of a Rust source: comments and test items removed.
///
/// A feature marker names implementation text. A token that appears only in a
/// doc comment or inside a test outlives the feature it claims, so a marker
/// scored against the whole file passes over an empty implementation carrying
/// the right prose. String literals are kept, because emitted target text is
/// implementation.
fn implementation_source(text: &str) -> String {
    let without_comments = strip_comments(text);
    strip_test_items(&without_comments)
}

/// Copy `text` without line comments, block comments or their nesting.
///
/// Bytes are compared, not characters: every delimiter is ASCII, so a
/// multi-byte character is copied whole and never matches one.
fn strip_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut block_depth = 0usize;
    while index < bytes.len() {
        if block_depth > 0 {
            if bytes[index..].starts_with(b"/*") {
                block_depth += 1;
                index += 2;
            } else if bytes[index..].starts_with(b"*/") {
                block_depth -= 1;
                index += 2;
            } else {
                if bytes[index] == b'\n' {
                    out.push(b'\n');
                }
                index += 1;
            }
            continue;
        }
        if bytes[index..].starts_with(b"//") {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if bytes[index..].starts_with(b"/*") {
            block_depth = 1;
            index += 2;
            continue;
        }
        let end = string_literal_end(bytes, index);
        if end > index {
            out.extend_from_slice(&bytes[index..end]);
            index = end;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// One past the end of the string literal starting at `index`, or `index`
/// when no literal starts there. Raw literals close on their own hash count.
fn string_literal_end(bytes: &[u8], index: usize) -> usize {
    if bytes[index] == b'r' {
        let mut cursor = index + 1;
        let mut hashes = 0usize;
        while cursor < bytes.len() && bytes[cursor] == b'#' {
            hashes += 1;
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b'"' {
            cursor += 1;
            while cursor < bytes.len() {
                if bytes[cursor] == b'"' && closing_hashes(bytes, cursor + 1) >= hashes {
                    return cursor + 1 + hashes;
                }
                cursor += 1;
            }
            return bytes.len();
        }
        return index;
    }
    if bytes[index] != b'"' {
        return index;
    }
    let mut cursor = index + 1;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'\\' => cursor += 2,
            b'"' => return cursor + 1,
            _ => cursor += 1,
        }
    }
    bytes.len()
}

/// Hash characters run length at `index`.
fn closing_hashes(bytes: &[u8], index: usize) -> usize {
    let mut count = 0;
    while index + count < bytes.len() && bytes[index + count] == b'#' {
        count += 1;
    }
    count
}

/// Copy `code` without any item attributed `#[cfg(test)]` or `#[test]`.
///
/// `code` has already had its comments removed, so a string literal is the
/// only place an attribute or a brace can appear without meaning one.
fn strip_test_items(code: &str) -> String {
    const ATTRIBUTES: [&str; 2] = ["#[cfg(test)]", "#[test]"];
    let bytes = code.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut kept_from = 0;
    while index < bytes.len() {
        let end = string_literal_end(bytes, index);
        if end > index {
            index = end;
            continue;
        }
        if ATTRIBUTES
            .iter()
            .any(|attribute| bytes[index..].starts_with(attribute.as_bytes()))
        {
            out.extend_from_slice(&bytes[kept_from..index]);
            index = item_end(bytes, index);
            kept_from = index;
            continue;
        }
        index += 1;
    }
    out.extend_from_slice(&bytes[kept_from..]);
    String::from_utf8_lossy(&out).into_owned()
}

/// One past the end of the item starting at the attribute at `start`.
///
/// A braced item ends on its matching brace. An item that is a declaration
/// rather than a body, such as `#[cfg(test)] mod tests;`, ends on the
/// semicolon, so the rest of the file is not swallowed. A closing brace that
/// belongs to an enclosing item ends the scan where it stands.
fn item_end(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    let mut depth = 0usize;
    while index < bytes.len() {
        let end = string_literal_end(bytes, index);
        if end > index {
            index = end;
            continue;
        }
        match bytes[index] {
            b'{' => depth += 1,
            b'}' if depth <= 1 => return index + 1,
            b'}' => depth -= 1,
            b';' if depth == 0 => return index + 1,
            _ => {}
        }
        index += 1;
    }
    bytes.len()
}

fn probe_nvidia_smi() -> GpuProbe {
    match Command::new("nvidia-smi").arg("-L").output() {
        Ok(output) if output.status.success() => {
            let devices = String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let (driver_version, cuda_version) = probe_nvidia_smi_versions();
            let device_details = probe_nvidia_smi_device_details();
            GpuProbe {
                nvidia_smi_ok: !devices.is_empty(),
                nvidia_smi_devices: devices,
                nvidia_smi_device_details: device_details,
                nvidia_driver_version: driver_version,
                nvidia_cuda_version: cuda_version,
                nvidia_smi_error: None,
            }
        }
        Ok(output) => GpuProbe {
            nvidia_smi_ok: false,
            nvidia_smi_devices: Vec::new(),
            nvidia_smi_device_details: Vec::new(),
            nvidia_driver_version: None,
            nvidia_cuda_version: None,
            nvidia_smi_error: Some(String::from_utf8_lossy(&output.stderr).trim().to_string()),
        },
        Err(error) => GpuProbe {
            nvidia_smi_ok: false,
            nvidia_smi_devices: Vec::new(),
            nvidia_smi_device_details: Vec::new(),
            nvidia_driver_version: None,
            nvidia_cuda_version: None,
            nvidia_smi_error: Some(error.to_string()),
        },
    }
}

fn probe_nvidia_smi_versions() -> (Option<String>, Option<String>) {
    let Ok(output) = Command::new("nvidia-smi").output() else {
        return (None, None);
    };
    if !output.status.success() {
        return (None, None);
    }
    parse_nvidia_smi_versions(&String::from_utf8_lossy(&output.stdout))
}

fn probe_nvidia_smi_device_details() -> Vec<GpuProbeDevice> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(parse_nvidia_smi_device_detail)
        .collect()
}

fn parse_nvidia_smi_device_detail(line: &str) -> Option<GpuProbeDevice> {
    let (name, driver_version, memory_total_mib, compute_capability) =
        vyre_bench::probes::environment::parse_nvidia_smi_device_fields(line)?;
    Some(GpuProbeDevice {
        name,
        driver_version,
        memory_total_mib,
        compute_capability_major: compute_capability.map(|(major, _minor)| major),
        compute_capability_minor: compute_capability.map(|(_major, minor)| minor),
    })
}

fn parse_nvidia_smi_versions(text: &str) -> (Option<String>, Option<String>) {
    let mut driver_version = None;
    let mut cuda_version = None;
    for line in text.lines() {
        if let Some(value) = parse_header_value(line, "Driver Version:") {
            driver_version = Some(value);
        }
        if let Some(value) = parse_header_value(line, "CUDA Version:") {
            cuda_version = Some(value);
        }
    }
    (driver_version, cuda_version)
}

fn parse_header_value(line: &str, label: &str) -> Option<String> {
    let start = line.find(label)? + label.len();
    let rest = line.get(start..)?.trim_start();
    let end = [rest.find('|'), rest.find(' ')]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(rest.len());
    let value = rest.get(..end)?.trim();
    (!value.is_empty()).then(|| value.to_string())
}
fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_BACKEND_EVIDENCE_TEXT_BYTES, "backend evidence")
}

/// Read the source a feature marker names.
///
/// A citation ending in `mod.rs` or `lib.rs` names a module root, and every
/// production `.rs` file under its directory is part of that module. Splitting
/// an oversized module into submodules moves an implementation token out of the
/// root file without moving it out of the module, so reading the cited file
/// alone reports a token the module still defines as missing. `tests.rs` and
/// any `tests` directory are not read: their contents wear no `#[cfg(test)]`
/// attribute of their own, so `strip_test_items` leaves a bare helper there
/// standing, and a marker would pass on a token only a test defines. Each file
/// is read through the bounded read; the walk is bounded by the module
/// directory.
fn read_marker_module(path: &Path) -> io::Result<String> {
    let mut text = read_text_bounded(path)?;
    let root = path.file_name().and_then(|name| name.to_str());
    if !matches!(root, Some("mod.rs" | "lib.rs")) {
        return Ok(text);
    }
    let Some(directory) = path.parent() else {
        return Ok(text);
    };
    let mut pending = vec![directory.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let mut children = fs::read_dir(&directory)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<io::Result<Vec<_>>>()?;
        children.sort();
        for child in children {
            if child.is_dir() {
                if child.file_name().is_none_or(|name| name != "tests") {
                    pending.push(child);
                }
                continue;
            }
            if child == path
                || child.extension().is_none_or(|extension| extension != "rs")
                || child.file_name().is_some_and(|name| name == "tests.rs")
            {
                continue;
            }
            text.push('\n');
            text.push_str(&read_text_bounded(&child)?);
        }
    }
    Ok(text)
}

#[cfg(test)]
mod capability_contract_tests {
    use super::*;

    #[test]
    fn unsupported_capability_rows_require_reason_and_fix() {
        let rows = vec![BackendCapabilityRow {
            backend_id: "cuda".to_string(),
            capability_id: "mlir-transform-support".to_string(),
            probe_source: "backend-matrix transform support registry".to_string(),
            probed_value: None,
            supported: false,
            unsupported_reason: None,
            fix: "missing".to_string(),
        }];

        let blockers = capability_contract_blockers(&rows);

        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("unsupported_reason")));
        assert!(blockers.iter().any(|blocker| blocker.contains("Fix")));
    }

    #[test]
    fn cuda_capability_rows_include_live_sm_memory_and_warp_contracts() {
        let backends = vec![BackendEntry {
            id: "cuda".to_string(),
            precedence: 0,
            dispatches: true,
            acquire_ok: true,
            acquire_error: None,
        }];
        let probe = GpuProbe {
            nvidia_smi_ok: true,
            nvidia_smi_devices: vec!["GPU 0: NVIDIA RTX 5090".to_string()],
            nvidia_smi_device_details: vec![GpuProbeDevice {
                name: "NVIDIA RTX 5090".to_string(),
                driver_version: "580.0".to_string(),
                memory_total_mib: Some(32 * 1024),
                compute_capability_major: Some(12),
                compute_capability_minor: Some(0),
            }],
            nvidia_driver_version: Some("580.0".to_string()),
            nvidia_cuda_version: Some("13.0".to_string()),
            nvidia_smi_error: None,
        };

        let rows = collect_backend_capability_rows(&backends, &probe);

        assert!(rows.iter().any(|row| {
            row.backend_id == "cuda"
                && row.capability_id == "live-sm-release-floor"
                && row.supported
                && row.probed_value.as_deref() == Some("sm_120")
        }));
        assert!(rows.iter().any(|row| {
            row.backend_id == "cuda"
                && row.capability_id == "live-memory-release-floor"
                && row.supported
                && row.probed_value.as_deref() == Some("32768 MiB")
        }));
        assert!(rows.iter().any(|row| {
            row.backend_id == "cuda"
                && row.capability_id == "warp-width-contract"
                && row.supported
                && row.probed_value.as_deref() == Some("32 lanes")
        }));
        assert!(capability_contract_blockers(&rows).is_empty());
    }

    /// WHY: the memory row is the only thing that rejects a device too small
    /// for the workload it claims to have measured, and the floor it compares
    /// against is now derived from the catalog. A row that reports `supported`
    /// for a device one MiB short certifies a measurement nothing can hold.
    #[test]
    fn a_device_below_the_derived_memory_floor_is_unsupported() {
        let backends = vec![BackendEntry {
            id: "cuda".to_string(),
            precedence: 0,
            dispatches: true,
            acquire_ok: true,
            acquire_error: None,
        }];
        let short_by_one = crate::gpu_release_floor::min_cuda_release_memory_mib() - 1;
        let probe = GpuProbe {
            nvidia_smi_ok: true,
            nvidia_smi_devices: vec!["GPU 0: NVIDIA A2".to_string()],
            nvidia_smi_device_details: vec![GpuProbeDevice {
                name: "NVIDIA A2".to_string(),
                driver_version: "580.0".to_string(),
                memory_total_mib: Some(short_by_one),
                compute_capability_major: Some(8),
                compute_capability_minor: Some(6),
            }],
            nvidia_driver_version: Some("580.0".to_string()),
            nvidia_cuda_version: Some("13.0".to_string()),
            nvidia_smi_error: None,
        };

        let rows = collect_backend_capability_rows(&backends, &probe);

        let memory_row = rows
            .iter()
            .find(|row| {
                row.backend_id == "cuda" && row.capability_id == "live-memory-release-floor"
            })
            .expect("Fix: CUDA capability rows must always carry a live memory release floor row.");
        assert!(
            !memory_row.supported,
            "Fix: a device {short_by_one} MiB below the derived floor must not be a supported release device."
        );
        assert_eq!(
            memory_row.unsupported_reason.as_deref(),
            Some(
                format!(
                    "no live NVIDIA device reported >={} MiB memory",
                    crate::gpu_release_floor::min_cuda_release_memory_mib()
                )
                .as_str()
            ),
            "Fix: the unsupported reason must quote the derived floor, not a fixed number."
        );
        assert!(
            !gpu_probe_device_meets_release_floor(&probe.nvidia_smi_device_details[0]),
            "Fix: the release-floor predicate must reject a device short of the derived memory floor."
        );
    }
}

#[cfg(test)]
mod feature_marker_tests {
    use super::*;

    /// WHY: a marker scored against the whole file passes on prose alone. The
    /// feature can be deleted and the doc comment describing it keeps the
    /// marker green. What this does not catch is a token that names a real
    /// symbol which no longer does the work.
    #[test]
    fn a_token_that_lives_only_in_a_comment_or_a_test_is_not_implementation() {
        let source = r#"
//! cp.async staging, sm_80 and up.
/* ldmatrix is described here and nowhere else */
pub fn analyze() -> bool {
    let text = "    // cp.async.commit_group;";
    !text.is_empty() // trailing prose about ldmatrix
}
#[cfg(test)]
mod tests {
    #[test]
    fn ldmatrix_and_cp_async_are_named_in_the_test() {
        assert!(super::analyze());
    }
}
"#;

        let code = implementation_source(source);

        assert!(
            code.contains("cp.async.commit_group"),
            "Fix: an emitted target string is implementation text and must survive: {code}"
        );
        assert!(
            code.contains("pub fn analyze"),
            "Fix: the implementation must survive: {code}"
        );
        assert!(
            !code.contains("sm_80"),
            "Fix: a doc comment is not implementation text: {code}"
        );
        assert!(
            !code.contains("ldmatrix is described"),
            "Fix: a block comment is not implementation text: {code}"
        );
        assert!(
            !code.contains("trailing prose"),
            "Fix: a trailing comment is not implementation text: {code}"
        );
        assert!(
            !code.contains("ldmatrix_and_cp_async_are_named_in_the_test"),
            "Fix: a test item is not implementation text: {code}"
        );
    }

    /// WHY: the marker set is read from the two declarations at run time, so a
    /// marker added later that points at a test file fails here instead of
    /// passing on the test's prose.
    #[test]
    fn no_feature_marker_names_a_test_file() {
        let declared = CUDA_FEATURE_MARKERS.iter().chain(WGPU_FEATURE_MARKERS);
        for requirement in declared {
            let relative = requirement.relative;
            assert!(
                !relative.contains("/tests/")
                    && !relative.ends_with("/tests.rs")
                    && !relative.ends_with("_test.rs")
                    && !relative.ends_with("_tests.rs"),
                "Fix: feature marker `{}` names the test file `{relative}`. A marker names the \
                 implementation the feature lives in, because a test's prose outlives the feature.",
                requirement.id
            );
        }
    }

    /// WHY: the recorded matrix is judged against the ids the producer emits.
    /// A second hand-written list of ids goes stale the first time a marker is
    /// added or renamed, and the check then requires a marker nothing writes.
    #[test]
    fn the_required_marker_ids_are_the_ids_the_producer_emits() {
        assert_eq!(
            cuda_feature_marker_ids().len(),
            CUDA_FEATURE_MARKERS.len(),
            "Fix: every declared CUDA marker is required."
        );
        assert_eq!(
            wgpu_feature_marker_ids().len(),
            WGPU_FEATURE_MARKERS.len(),
            "Fix: every declared WGPU marker is required."
        );
        assert!(
            cuda_feature_marker_ids().contains(&"ldmatrix-cp-async"),
            "Fix: the async staging marker stays required."
        );
    }

    /// WHY: a marker citation names a module, and a module spans a directory.
    /// Splitting an oversized module into submodules moves an implementation
    /// token out of `mod.rs` while the module still defines it, and reading the
    /// cited file alone then reports the feature as deleted. This asserts both
    /// halves: the union finds the token, and the cited file alone does not, so
    /// it goes red against the single-file read. What it does not catch is a
    /// token that moved to a different module entirely, which is a real finding.
    #[test]
    fn a_module_root_citation_reads_the_submodule_a_token_moved_into() {
        let root = tempfile::tempdir().expect("Fix: the test needs a temporary directory.");
        let module = root.path().join("disk_cache");
        fs::create_dir_all(module.join("deep"))
            .expect("Fix: the test needs a nested module directory.");
        let cited = module.join("mod.rs");
        fs::write(&cited, "pub mod io;\npub mod deep;\n")
            .expect("Fix: the test needs a root file.");
        fs::write(
            module.join("io.rs"),
            "pub const MAX_PENDING_DURABLE_CACHE_FILES: usize = 64;\n",
        )
        .expect("Fix: the test needs a submodule file.");
        fs::write(module.join("deep").join("inner.rs"), "pub fn evict() {}\n")
            .expect("Fix: the test needs a nested submodule file.");

        let module_text = read_marker_module(&cited).expect("Fix: the module must be readable.");
        let cited_text = read_text_bounded(&cited).expect("Fix: the cited file must be readable.");

        assert!(
            module_text.contains("MAX_PENDING_DURABLE_CACHE_FILES"),
            "Fix: a token defined in a submodule is defined by the module: {module_text}"
        );
        assert!(
            module_text.contains("pub fn evict"),
            "Fix: the walk reaches a nested submodule: {module_text}"
        );
        assert!(
            !cited_text.contains("MAX_PENDING_DURABLE_CACHE_FILES"),
            "Fix: the single-file read is what this test must beat: {cited_text}"
        );
    }

    /// WHY: a citation that names one implementation file means that file. If
    /// the walk widened to every sibling, a marker could pass on a token its
    /// own file never defines, which is the vacuous case the gate exists to
    /// prevent.
    #[test]
    fn a_leaf_citation_reads_only_the_file_it_names() {
        let root = tempfile::tempdir().expect("Fix: the test needs a temporary directory.");
        let cited = root.path().join("resident_io.rs");
        fs::write(&cited, "pub fn upload_resident_inputs() {}\n")
            .expect("Fix: the test needs the cited file.");
        fs::write(
            root.path().join("resident_io_download.rs"),
            "pub fn download_resident_readbacks_many() {}\n",
        )
        .expect("Fix: the test needs a sibling file.");

        let text = read_marker_module(&cited).expect("Fix: the cited file must be readable.");

        assert!(
            text.contains("upload_resident_inputs"),
            "Fix: the cited file is read: {text}"
        );
        assert!(
            !text.contains("download_resident_readbacks_many"),
            "Fix: a leaf citation does not absorb its siblings: {text}"
        );
    }

    /// WHY: `no_feature_marker_names_a_test_file` guards the citation, and
    /// `strip_test_items` removes items carrying `#[cfg(test)]` or `#[test]`.
    /// Neither covers a whole test module file: `disk_cache/tests.rs` is
    /// declared `#[cfg(test)] mod tests;` in the root, so its own contents wear
    /// no attribute, and a bare helper const there survives into
    /// `implementation_source`. That would let a production marker pass on a
    /// token only a test defines, which is the vacuous case this gate exists to
    /// refuse. The module read excludes `tests.rs` and any `tests/` directory.
    /// What it does not catch: test material in a file named something else.
    #[test]
    fn a_module_root_citation_reads_no_test_material() {
        let root = tempfile::tempdir().expect("Fix: the test needs a temporary directory.");
        let module = root.path().join("disk_cache");
        fs::create_dir_all(module.join("tests"))
            .expect("Fix: the test needs a test directory inside the module.");
        let cited = module.join("mod.rs");
        fs::write(&cited, "pub mod io;\n#[cfg(test)]\nmod tests;\n")
            .expect("Fix: the test needs a root file.");
        fs::write(module.join("io.rs"), "pub fn evict() {}\n")
            .expect("Fix: the test needs a submodule file.");
        fs::write(
            module.join("tests.rs"),
            "pub(crate) const ONLY_A_TEST_DEFINES_THIS: usize = 1;\n",
        )
        .expect("Fix: the test needs a test module file.");
        fs::write(
            module.join("tests").join("helper.rs"),
            "pub(crate) const ONLY_A_TEST_HELPER_DEFINES_THIS: usize = 2;\n",
        )
        .expect("Fix: the test needs a test helper file.");

        let text = read_marker_module(&cited).expect("Fix: the module must be readable.");

        assert!(
            text.contains("pub fn evict"),
            "Fix: a production submodule is still read: {text}"
        );
        assert!(
            !text.contains("ONLY_A_TEST_DEFINES_THIS"),
            "Fix: a token only `tests.rs` defines must not satisfy a marker: {text}"
        );
        assert!(
            !text.contains("ONLY_A_TEST_HELPER_DEFINES_THIS"),
            "Fix: a token only a `tests/` file defines must not satisfy a marker: {text}"
        );
    }
}

#[cfg(test)]
mod artifact_ownership_tests {
    use super::*;

    /// WHY: The authoritative descriptor and backend producer must agree on
    /// the exact output path so comparison is immutable and write mutations
    /// are never undeclared.
    #[test]
    fn authoritative_descriptor_declares_exact_backend_matrix_artifact() {
        let descriptor = xtask::gate_metadata::descriptor_by_name("backend-matrix");
        let mut expected: Vec<&str> = vec![ARTIFACT];
        expected.sort_unstable();
        let mut actual: Vec<&str> = descriptor.artifacts.to_vec();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "Fix: backend-matrix gate descriptor must declare exactly the canonical backend evidence artifact (`{ARTIFACT}`)"
        );
    }
}
