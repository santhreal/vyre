//! Precompiled CUDA pipeline implementation.

use std::sync::{Arc, Mutex};

use smallvec::SmallVec;
use vyre_driver::accounting::checked_add_usize_lazy;
use vyre_driver::input_identity::{domain_separated_exact_input_key, ExactInputKey};
use vyre_driver::BindingRole;
use vyre_driver::{sealed, BackendError, DispatchConfig, LaunchPlan};
use vyre_foundation::ir::Program;

use crate::backend::allocations::DeviceAllocation;
use crate::backend::module_cache::PtxSourceCacheKey;
use crate::backend::{CachedCudaGraph, CudaBackend, CudaDispatchPlan, ModuleCacheKey};
use crate::device::CudaDeviceCaps;

mod compiled_dispatch;
mod materialized_cache;
mod static_params;

#[cfg(test)]
pub(crate) use materialized_cache::MAX_MATERIALIZED_OUTPUT_CACHE_BYTES_PER_PIPELINE;
pub(crate) use materialized_cache::{
    MaterializedPipelineOutputCache, MaterializedPipelineOutputCacheEntry,
};
use static_params::upload_static_launch_params;

/// Mutually exclusive GPU execution strategy selected for a compiled CUDA pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaPipelineExecutionStrategy {
    /// Cached CUDA graph record and replay (with materialized output cache).
    GraphReplay,
    /// Direct stream-ordered kernel dispatch with trap readback and barrier synchronization.
    DirectDispatch,
}

/// CUDA pipeline with PTX already lowered and loaded into the backend cache.
#[derive(Debug)]
pub(crate) struct CudaCompiledPipeline {
    backend: CudaBackend,
    program: Arc<Program>,
    ptx_src: Arc<str>,
    module_key: ModuleCacheKey,
    prepared: CudaDispatchPlan,
    compiled_config: DispatchConfig,
    graph_cache: Mutex<SmallVec<[CachedCudaGraph; MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE]>>,
    materialized_output_cache: Mutex<MaterializedPipelineOutputCache>,
    static_params: DeviceAllocation,
    id: String,
}

pub(crate) const MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE: usize = 32;
const CUDA_GRAPH_REPLAY_SMS_PER_LANE: usize = 8;
const CUDA_GRAPH_REPLAY_MIN_CONCURRENT_LANES: usize = 2;
const CUDA_GRAPH_REPLAY_VRAM_FRACTION_DENOMINATOR: u64 = 64;
const CUDA_COMPILED_PIPELINE_ID_DOMAIN: &[u8] = b"vyre.cuda.pipeline.compiled.v1";

fn cuda_compiled_pipeline_identity_key(
    ptx_source_key: &[u8; 32],
    module_key: &[u8; 32],
    launch: &LaunchPlan,
) -> Result<ExactInputKey, BackendError> {
    let element_count = launch.element_count.to_le_bytes();
    let workgroup_x = launch.workgroup[0].to_le_bytes();
    let workgroup_y = launch.workgroup[1].to_le_bytes();
    let workgroup_z = launch.workgroup[2].to_le_bytes();
    let grid_x = launch.grid[0].to_le_bytes();
    let grid_y = launch.grid[1].to_le_bytes();
    let grid_z = launch.grid[2].to_le_bytes();
    domain_separated_exact_input_key(
        CUDA_COMPILED_PIPELINE_ID_DOMAIN,
        0,
        0,
        &[
            ptx_source_key.as_slice(),
            module_key.as_slice(),
            element_count.as_slice(),
            workgroup_x.as_slice(),
            workgroup_y.as_slice(),
            workgroup_z.as_slice(),
            grid_x.as_slice(),
            grid_y.as_slice(),
            grid_z.as_slice(),
        ],
    )
}

impl CudaCompiledPipeline {
    /// Construct a pipeline from compiler-generated PTX.
    pub(crate) fn new(
        backend: CudaBackend,
        program: Arc<Program>,
        ptx_src: Arc<str>,
        ptx_source_key: PtxSourceCacheKey,
        module_key: ModuleCacheKey,
        config: &DispatchConfig,
        prepared: CudaDispatchPlan,
    ) -> Result<Self, BackendError> {
        Self::new_with_source_identity(
            backend,
            program,
            ptx_src,
            *ptx_source_key.as_bytes(),
            module_key,
            config,
            prepared,
        )
    }

    /// Construct a pipeline from authenticated immutable target PTX.
    pub(crate) fn new_from_target_payload(
        backend: CudaBackend,
        program: Arc<Program>,
        ptx_src: Arc<str>,
        module_key: ModuleCacheKey,
        config: &DispatchConfig,
        prepared: CudaDispatchPlan,
    ) -> Result<Self, BackendError> {
        let source_identity = *blake3::hash(ptx_src.as_bytes()).as_bytes();
        Self::new_with_source_identity(
            backend,
            program,
            ptx_src,
            source_identity,
            module_key,
            config,
            prepared,
        )
    }

    fn new_with_source_identity(
        backend: CudaBackend,
        program: Arc<Program>,
        ptx_src: Arc<str>,
        source_identity: [u8; 32],
        module_key: ModuleCacheKey,
        config: &DispatchConfig,
        prepared: CudaDispatchPlan,
    ) -> Result<Self, BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_PIPELINE_COMPILE_RANGE);
        let trace = crate::instrumentation::cuda_stage_trace_enabled();
        let started = std::time::Instant::now();
        if trace {
            tracing::debug!(
                "[cuda-pipeline] start entry={}",
                program.entry_op_id.as_deref().unwrap_or("<anonymous>")
            );
        }
        let digest =
            cuda_compiled_pipeline_identity_key(&source_identity, &module_key.0, &prepared.launch)?;
        if trace {
            tracing::debug!(
                "[cuda-pipeline] +{}ms digest ready",
                started.elapsed().as_millis()
            );
        }
        let static_params = upload_static_launch_params(&backend, &prepared.launch.param_words)?;
        if trace {
            tracing::debug!(
                "[cuda-pipeline] +{}ms static params ready bytes={}",
                started.elapsed().as_millis(),
                static_params.byte_len
            );
        }
        Ok(Self {
            backend,
            program,
            ptx_src,
            module_key,
            prepared,
            compiled_config: config.clone(),
            graph_cache: Mutex::new(SmallVec::new()),
            materialized_output_cache: Mutex::new(MaterializedPipelineOutputCache::default()),
            static_params,
            id: format!("cuda:{}", blake3::Hash::from(digest).to_hex()),
        })
    }

    /// Select the primary GPU execution strategy for this pipeline.
    ///
    /// Graph capture cannot read back device-side trap records because stream
    /// synchronization is forbidden during graph recording. Modules that declare
    /// traps, or cooperative kernels requiring multi-grid synchronization, are
    /// routed to direct stream-ordered dispatch so trap readback and grid barriers
    /// remain fail-closed.
    #[must_use]
    pub(crate) fn execution_strategy(&self) -> CudaPipelineExecutionStrategy {
        select_cuda_pipeline_execution_strategy(
            cuda_graph_replay_enabled(),
            self.prepared.cooperative,
            self.declares_trap(),
        )
    }

    /// Whether this pipeline's module declares a device-side trap record.
    ///
    /// Derived generically from module/program facts: the PTX text carrying
    /// `_vyre_trap_sidecar` or the source program declaring `CAP_TRAP`.
    #[must_use]
    pub(crate) fn declares_trap(&self) -> bool {
        crate::backend::module_cache::declares_trap_sidecar(&self.ptx_src)
            || self.program.stats().trap()
    }
}

impl Drop for CudaCompiledPipeline {
    fn drop(&mut self) {
        self.backend
            .transient_pool
            .release(std::mem::take(&mut self.static_params));
    }
}

impl sealed::Sealed for CudaCompiledPipeline {}

fn cuda_graph_replay_enabled() -> bool {
    crate::instrumentation::cuda_graph_replay_enabled()
}

fn select_cuda_pipeline_execution_strategy(
    graph_replay_enabled: bool,
    cooperative: bool,
    declares_trap: bool,
) -> CudaPipelineExecutionStrategy {
    if graph_replay_enabled && !cooperative && !declares_trap {
        CudaPipelineExecutionStrategy::GraphReplay
    } else {
        CudaPipelineExecutionStrategy::DirectDispatch
    }
}

pub(crate) fn cuda_graph_lane_count_for_batch(
    caps: &CudaDeviceCaps,
    prepared: &CudaDispatchPlan,
    batches: &[&[&[u8]]],
) -> Result<usize, BackendError> {
    if batches.is_empty() {
        return Ok(0);
    }
    let hardware_lanes = cuda_graph_hardware_lane_capacity(caps)?;
    let shape_bytes = cuda_graph_shape_cached_bytes(prepared, batches[0])?;
    let shape_bytes_u64 = u64::try_from(shape_bytes).map_err(|_| BackendError::InvalidProgram {
        fix: "Fix: CUDA graph replay shape byte count exceeds u64; split the replay batch before lane planning.".to_string(),
    })?;
    let host_memory_budget_cap = u64::try_from(usize::MAX).map_err(|source| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: host usize::MAX cannot fit u64 while planning CUDA graph lanes: {source}; use a supported host pointer width."
            ),
        }
    })?;
    let memory_budget_u64 = (caps.total_memory / CUDA_GRAPH_REPLAY_VRAM_FRACTION_DENOMINATOR)
        .max(shape_bytes_u64)
        .min(host_memory_budget_cap);
    let memory_budget = usize::try_from(memory_budget_u64).map_err(|source| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA graph replay memory budget {memory_budget_u64} cannot fit usize: {source}; split the replay batch before lane planning."
            ),
        }
    })?;
    let memory_lanes = if shape_bytes == 0 {
        MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE
    } else {
        (memory_budget / shape_bytes).clamp(1, MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE)
    };
    Ok(batches.len().min(hardware_lanes).min(memory_lanes).max(1))
}

fn cuda_graph_hardware_lane_capacity(caps: &CudaDeviceCaps) -> Result<usize, BackendError> {
    if !caps.concurrent_kernels {
        return Ok(1);
    }
    let sms = usize::try_from(caps.multi_processor_count_u32()).map_err(|source| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA multiprocessor count cannot fit usize during graph lane planning: {source}; reject corrupt device capabilities before compiling graph replay."
            ),
        }
    });
    let sms = sms?;
    let lanes = sms.div_ceil(CUDA_GRAPH_REPLAY_SMS_PER_LANE);
    Ok(lanes.clamp(
        CUDA_GRAPH_REPLAY_MIN_CONCURRENT_LANES,
        MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE,
    ))
}

fn cuda_graph_shape_cached_bytes(
    prepared: &CudaDispatchPlan,
    inputs: &[&[u8]],
) -> Result<usize, BackendError> {
    let mut bytes = bucketed_len(std::mem::size_of_val(
        prepared.launch.param_words.as_slice(),
    ))?;
    for binding in &prepared.bindings.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let byte_len = binding
            .input_index
            .and_then(|input_index| inputs.get(input_index).map(|input| input.len()))
            .or(binding.static_byte_len)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA graph replay shape cache found binding `{}` without a runtime input or static byte length. Preserve concrete binding byte lengths during dispatch planning instead of treating missing sizes as zero.",
                    binding.name
                ),
            })?;
        bytes = add_shape_bytes(bytes, bucketed_len(byte_len)?)?;
        if binding.input_index.is_some() {
            bytes = add_shape_bytes(bytes, bucketed_len(byte_len)?)?;
        }
        if binding.output_index.is_some() {
            bytes = add_shape_bytes(bytes, bucketed_len(byte_len)?)?;
        }
    }
    Ok(bytes)
}

fn add_shape_bytes(total: usize, component: usize) -> Result<usize, BackendError> {
    checked_add_usize_lazy(total, component, || {
        BackendError::InvalidProgram {
        fix: "Fix: CUDA graph replay cached shape byte count overflowed; split the replay batch before graph-cache lane planning.".to_string(),
    }
    })
}

fn bucketed_len(byte_len: usize) -> Result<usize, BackendError> {
    byte_len
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: CUDA graph replay bucketed shape byte count overflowed; split the oversized input or disable graph replay for this shape.".to_string(),
        })
}

// Inline: `pipeline` is a private module, so `cuda_compiled_pipeline_identity_key`,
// `cuda_graph_lane_count_for_batch` and the materialized output cache are
// unreachable from an integration test.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smallvec::smallvec;
    use vyre_driver::input_identity::exact_input_key;
    use vyre_driver::replace_output_buffers_preserving_slots;
    use vyre_driver::LaunchPlan;
    use vyre_driver::{Binding, BindingPlan, BindingRole};

    use crate::backend::CudaDispatchPlan;
    use crate::synthetic_device_caps::synthetic_sm120_envelope;

    use super::{
        cuda_compiled_pipeline_identity_key, cuda_graph_lane_count_for_batch,
        MaterializedPipelineOutputCache, MaterializedPipelineOutputCacheEntry,
        MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE, MAX_MATERIALIZED_OUTPUT_CACHE_BYTES_PER_PIPELINE,
    };

    fn single_input_output_plan(byte_len: usize) -> CudaDispatchPlan {
        CudaDispatchPlan {
            bindings: BindingPlan {
                bindings: vec![Binding {
                    name: Arc::from("state"),
                    binding: 0,
                    buffer_index: 0,
                    role: BindingRole::InputOutput,
                    element_size: 1,
                    preferred_alignment: 1,
                    element_count: byte_len as u32,
                    static_byte_len: Some(byte_len),
                    input_index: Some(0),
                    output_index: Some(0),
                }],
                input_indices: vec![0],
                output_indices: vec![0],
                shared_indices: vec![],
            },
            output_binding_indices: smallvec![0],
            launch: LaunchPlan {
                grid: [1, 1, 1],
                workgroup: [128, 1, 1],
                element_count: byte_len as u32,
                param_words: vec![1, 2, 3, 4],
                max_binding_alignment: 1,
            },
            cooperative: false,
            fixpoint_iterations: 1,
        }
    }

    fn generated_pipeline_identity_key(seed: u32, salt: u32) -> [u8; 32] {
        let mut out = [0_u8; 32];
        let mut state = seed ^ salt ^ 0xC0DA_CAFE;
        for (index, byte) in out.iter_mut().enumerate() {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .rotate_left((index as u32) & 15);
            *byte = (state >> ((index & 3) * 8)) as u8;
        }
        out
    }

    fn generated_pipeline_identity_launch(seed: u32) -> LaunchPlan {
        LaunchPlan {
            element_count: 1 + (seed % 4096),
            workgroup: [
                32 + (seed % 8) * 32,
                1 + (seed.rotate_left(3) % 4),
                1 + (seed.rotate_left(5) % 2),
            ],
            grid: [
                1 + (seed % 1024),
                1 + (seed.rotate_left(7) % 16),
                1 + (seed.rotate_left(11) % 8),
            ],
            param_words: Vec::new(),
            max_binding_alignment: std::mem::size_of::<u64>(),
        }
    }

    mod cache_core_contracts {
        use super::*;

        #[test]
        fn materialized_output_cache_hits_4096_generated_exact_inputs() {
            let mut cache = MaterializedPipelineOutputCache::default();
            for seed in 0_u32..4096 {
                let input_len = ((seed.wrapping_mul(19) ^ seed.rotate_left(3)) % 128 + 1) as usize;
                let output_len = ((seed.wrapping_mul(23) ^ seed.rotate_left(7)) % 128 + 1) as usize;
                let mut state = seed ^ 0xD15C_A11E;
                let mut input = Vec::with_capacity(input_len);
                for index in 0..input_len {
                    state = state
                        .wrapping_mul(1_664_525)
                        .wrapping_add(1_013_904_223)
                        .rotate_left((index as u32) & 15);
                    input.push((state >> ((index & 3) * 8)) as u8);
                }
                let mut output = Vec::with_capacity(output_len);
                for index in 0..output_len {
                    state = state
                        .wrapping_mul(22_695_477)
                        .wrapping_add(1)
                        .rotate_left((index as u32) & 7);
                    output.push((state ^ seed.rotate_left(index as u32 & 31)) as u8);
                }
                let outputs = vec![output];
                cache
                    .remember(&[input.as_slice()], &outputs)
                    .expect("Fix: generated materialized CUDA output cache insert must fit");

                let mut replayed = vec![Vec::with_capacity(output_len + 31)];
                assert!(
                    cache
                        .hit_into(&[input.as_slice()], &mut replayed)
                        .expect("Fix: generated materialized CUDA output cache hit must fit"),
                    "Fix: materialized CUDA output cache must hit immediately for generated exact input case {seed}."
                );
                assert_eq!(
                    replayed, outputs,
                    "Fix: materialized CUDA output cache must replay exact output bytes for generated case {seed}."
                );
                assert!(
                    cache.len() <= MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE,
                    "Fix: materialized CUDA output cache must enforce the bounded entry count."
                );
                assert!(
                    cache.byte_len() <= MAX_MATERIALIZED_OUTPUT_CACHE_BYTES_PER_PIPELINE,
                    "Fix: materialized CUDA output cache must enforce the bounded byte budget."
                );
            }
        }

        #[test]
        fn materialized_output_cache_replaces_same_key_without_double_counting_bytes() {
            let mut cache = MaterializedPipelineOutputCache::default();
            let input = b"same compiled CUDA graph replay input";
            let outputs_a = vec![b"old output".to_vec()];
            let outputs_b = vec![b"new output with a different byte length".to_vec()];

            cache
                .remember(&[input.as_slice()], &outputs_a)
                .expect("Fix: first materialized output cache insert must fit");
            assert_eq!(cache.len(), 1);
            let first_bytes = cache.byte_len();
            assert_eq!(first_bytes, input.len() + outputs_a[0].len());

            cache
                .remember(&[input.as_slice()], &outputs_b)
                .expect("Fix: same-key materialized output cache replacement must fit");
            assert_eq!(
                cache.len(),
                1,
                "Fix: same-key materialized output cache replacement must not create duplicate entries."
            );
            assert_eq!(
                cache.byte_len(),
                input.len() + outputs_b[0].len(),
                "Fix: same-key materialized output cache replacement must subtract the old entry before adding the new one."
            );

            let mut replayed = Vec::new();
            assert!(cache
                .hit_into(&[input.as_slice()], &mut replayed)
                .expect("Fix: same-key materialized output cache hit must fit"));
            assert_eq!(
                replayed, outputs_b,
                "Fix: same-key materialized output cache hit must return the newest output bytes."
            );
        }

        #[test]
        fn materialized_output_snapshot_survives_same_key_replacement() {
            let mut cache = MaterializedPipelineOutputCache::default();
            let input = b"snapshot input retained outside the CUDA graph cache lock";
            let outputs_a = vec![b"snapshot bytes copied after lock release".to_vec()];
            let outputs_b = vec![b"replacement bytes stored by another replay".to_vec()];

            cache
                .remember(&[input.as_slice()], &outputs_a)
                .expect("Fix: initial materialized output snapshot fixture insert must fit");
            let snapshot = cache
                .snapshot(&[input.as_slice()])
                .expect("Fix: materialized output snapshot lookup must fit")
                .expect("Fix: materialized output snapshot must exist for exact input");

            cache
                .remember(&[input.as_slice()], &outputs_b)
                .expect("Fix: same-key materialized output replacement must fit after snapshot");

            let mut replayed_from_snapshot = Vec::new();
            snapshot
                .copy_into(&mut replayed_from_snapshot)
                .expect("Fix: materialized output snapshot copy after replacement must fit");
            assert_eq!(
                replayed_from_snapshot, outputs_a,
                "Fix: CUDA materialized cache hit snapshots must keep immutable output ownership so dispatch can copy after releasing the cache lock."
            );

            let mut replayed_from_cache = Vec::new();
            assert!(cache
                .hit_into(&[input.as_slice()], &mut replayed_from_cache)
                .expect("Fix: post-replacement materialized cache hit must fit"));
            assert_eq!(
                replayed_from_cache, outputs_b,
                "Fix: same-key replacement must still expose the newest cached output after an older snapshot escapes the cache lock."
            );
        }
    }

    mod cache_pressure_contracts {
        use super::*;

        #[test]
        fn materialized_output_cache_prebuilt_entries_match_direct_remember_for_1024_cases() {
            for seed in 0_u32..1024 {
                let input_len = ((seed.wrapping_mul(11) ^ seed.rotate_left(13)) % 96 + 1) as usize;
                let output_len = ((seed.wrapping_mul(31) ^ seed.rotate_left(5)) % 96 + 1) as usize;
                let mut state = seed ^ 0xCACA_5000;
                let mut input = Vec::with_capacity(input_len);
                for index in 0..input_len {
                    state = state
                        .wrapping_mul(1_664_525)
                        .wrapping_add(1_013_904_223)
                        .rotate_left((index as u32) & 15);
                    input.push((state >> ((index & 3) * 8)) as u8);
                }
                let mut output = Vec::with_capacity(output_len);
                for index in 0..output_len {
                    state = state
                        .wrapping_mul(22_695_477)
                        .wrapping_add(1)
                        .rotate_left((index as u32) & 7);
                    output.push((state ^ seed.rotate_right(index as u32 & 31)) as u8);
                }
                let outputs = vec![output];
                let mut direct = MaterializedPipelineOutputCache::default();
                direct
                    .remember(&[input.as_slice()], &outputs)
                    .expect("Fix: direct materialized cache remember must fit");
                let mut prebuilt = MaterializedPipelineOutputCache::default();
                let entry =
                    MaterializedPipelineOutputCacheEntry::new(&[input.as_slice()], &outputs)
                        .expect("Fix: prebuilt materialized cache entry construction must fit");
                prebuilt
                    .remember_entry(entry)
                    .expect("Fix: prebuilt materialized cache entry insertion must fit");
                let input_key = exact_input_key(&[input.as_slice()])
                    .expect("Fix: generated materialized input key must fit");
                let mut keyed = MaterializedPipelineOutputCache::default();
                let keyed_entry = MaterializedPipelineOutputCacheEntry::new_with_key(
                    &[input.as_slice()],
                    &input_key,
                    &outputs,
                )
                .expect("Fix: keyed materialized cache entry construction must fit");
                keyed
                    .remember_entry(keyed_entry)
                    .expect("Fix: keyed materialized cache entry insertion must fit");

                let mut direct_replay = Vec::new();
                let mut prebuilt_replay = Vec::new();
                let mut keyed_replay = Vec::new();
                assert!(
                    direct
                        .hit_into(&[input.as_slice()], &mut direct_replay)
                        .expect("Fix: direct materialized cache hit must fit"),
                    "Fix: direct materialized cache must hit for generated case {seed}."
                );
                assert!(
                    prebuilt
                        .hit_into(&[input.as_slice()], &mut prebuilt_replay)
                        .expect("Fix: prebuilt materialized cache hit must fit"),
                    "Fix: prebuilt materialized cache must hit for generated case {seed}."
                );
                assert!(
                    keyed
                        .hit_into(&[input.as_slice()], &mut keyed_replay)
                        .expect("Fix: keyed materialized cache hit must fit"),
                    "Fix: keyed materialized cache must hit for generated case {seed}."
                );
                assert_eq!(
                    prebuilt_replay, direct_replay,
                    "Fix: prebuilt materialized cache insertion must preserve exact outputs for generated case {seed}."
                );
                assert_eq!(
                    keyed_replay, direct_replay,
                    "Fix: keyed materialized cache insertion must preserve exact outputs for generated case {seed}."
                );
                assert_eq!(
                    prebuilt.byte_len(),
                    direct.byte_len(),
                    "Fix: prebuilt materialized cache insertion must preserve byte accounting for generated case {seed}."
                );
                assert_eq!(
                    keyed.byte_len(),
                    direct.byte_len(),
                    "Fix: keyed materialized cache insertion must preserve byte accounting for generated case {seed}."
                );
            }
        }

        #[test]
        fn materialized_output_cache_evicts_oldest_entries_under_generated_pressure() {
            let mut cache = MaterializedPipelineOutputCache::default();
            let total_entries = MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE + 17;
            for seed in 0..total_entries {
                let input = (seed as u32).to_le_bytes().to_vec();
                let outputs = vec![vec![seed as u8; 8]];
                cache
                    .remember(&[input.as_slice()], &outputs)
                    .expect("Fix: generated materialized output cache pressure insert must fit");
            }

            assert_eq!(
                cache.len(),
                MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE,
                "Fix: materialized output cache must evict oldest entries instead of growing past its bounded lane-cache size."
            );
            assert_eq!(
                cache.byte_len(),
                MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE * (std::mem::size_of::<u32>() + 8),
                "Fix: materialized output cache byte accounting must track evicted entries exactly under generated pressure."
            );

            let evicted_input = 0_u32.to_le_bytes().to_vec();
            let mut evicted_replay = vec![b"sentinel".to_vec()];
            assert!(
                !cache
                    .hit_into(&[evicted_input.as_slice()], &mut evicted_replay)
                    .expect("Fix: evicted materialized output lookup must stay fallible"),
                "Fix: oldest generated materialized output entry must be evicted when capacity is exceeded."
            );
            assert_eq!(
                evicted_replay,
                vec![b"sentinel".to_vec()],
                "Fix: materialized output cache miss must not mutate caller-owned output buffers."
            );

            let retained_seed = (total_entries - 1) as u32;
            let retained_input = retained_seed.to_le_bytes().to_vec();
            let mut retained_replay = Vec::new();
            assert!(
                cache
                    .hit_into(&[retained_input.as_slice()], &mut retained_replay)
                    .expect("Fix: retained materialized output lookup must fit"),
                "Fix: newest generated materialized output entry must remain cached after pressure eviction."
            );
            assert_eq!(
                retained_replay,
                vec![vec![retained_seed as u8; 8]],
                "Fix: retained generated materialized output entry must replay exact bytes after evictions."
            );
        }

        #[test]

        fn materialized_output_cache_rejects_oversized_entries_without_polluting_cache() {
            let mut cache = MaterializedPipelineOutputCache::default();
            let input = b"oversized compiled CUDA graph replay input";
            let outputs = vec![vec![
                0xA5;
                MAX_MATERIALIZED_OUTPUT_CACHE_BYTES_PER_PIPELINE + 1
            ]];

            cache
                .remember(&[input.as_slice()], &outputs)
                .expect("Fix: oversized materialized output cache entry should be a typed no-admission path, not an allocation or dispatch failure.");

            assert_eq!(
                cache.len(),
                0,
                "Fix: oversized materialized output cache entries must not evict useful entries or consume cache slots."
            );
            assert_eq!(
                cache.byte_len(),
                0,
                "Fix: oversized materialized output cache entries must not perturb byte accounting."
            );
            let mut replay = Vec::new();
            assert!(
                !cache
                    .hit_into(&[input.as_slice()], &mut replay)
                    .expect("Fix: oversized no-admission lookup must remain fallible"),
                "Fix: oversized materialized output cache entries must not be observable as hits."
            );
        }
    }

    mod input_key_owner_contracts {
        use vyre_driver::input_identity::domain_separated_exact_input_key;

        use super::*;

        /// Input tuples that separately exercise a single slot, a tuple boundary, an
        /// empty slot between two non-empty ones, and a trailing empty slot.
        const INPUT_TUPLES: &[&[&[u8]]] = &[
            &[b"state"],
            &[b"ab", b"c"],
            &[b"ab", b"", b"c"],
            &[b"abc", b""],
            &[b"", b"abc"],
        ];

        fn outputs_for(inputs: &[&[u8]]) -> Vec<Vec<u8>> {
            vec![inputs
                .iter()
                .flat_map(|input| input.iter().rev().copied())
                .collect()]
        }

        #[test]
        fn materialized_cache_keys_inputs_with_the_shared_driver_envelope() {
            for inputs in INPUT_TUPLES {
                let outputs = outputs_for(inputs);
                let envelope_key = exact_input_key(inputs)
                    .expect("Fix: shared exact-input envelope must key the declared tuple");

                let entry = MaterializedPipelineOutputCacheEntry::new(inputs, &outputs).expect(
                    "Fix: materialized cache entry construction must fit the declared tuple",
                );
                assert_eq!(
                    entry.input_key(),
                    &envelope_key,
                    "Fix: the CUDA materialized output cache must key inputs with vyre_driver::input_identity::exact_input_key rather than a CUDA-private envelope, for tuple {inputs:?}."
                );

                let mut cache = MaterializedPipelineOutputCache::default();
                cache
                    .remember_entry(entry)
                    .expect("Fix: materialized cache insertion must fit the declared tuple");
                assert!(
                    cache.snapshot_with_key(inputs, &envelope_key).is_some(),
                    "Fix: an entry the cache stored must be reachable by the shared envelope key, for tuple {inputs:?}."
                );
            }
        }

        #[test]
        fn materialized_cache_rejects_a_resident_cache_domain_key_for_the_same_inputs() {
            for inputs in INPUT_TUPLES {
                let outputs = outputs_for(inputs);
                let mut cache = MaterializedPipelineOutputCache::default();
                cache
                    .remember(inputs, &outputs)
                    .expect("Fix: materialized cache remember must fit the declared tuple");

                let domain_key = domain_separated_exact_input_key(
                    b"vyre.cuda.optimizer.static-upload.v1",
                    0,
                    0,
                    inputs,
                )
                .expect("Fix: domain-separated key must fit the declared tuple");

                assert!(
                    cache.snapshot_with_key(inputs, &domain_key).is_none(),
                    "Fix: a resident-cache domain key must not reach materialized replay outputs, or the two caches alias for tuple {inputs:?}."
                );
                assert!(
                    cache
                        .snapshot(inputs)
                        .expect("Fix: materialized cache lookup must fit the declared tuple")
                        .is_some(),
                    "Fix: the same inputs must still hit under the plain replay envelope, for tuple {inputs:?}."
                );
            }
        }
    }

    mod pipeline_contracts {
        use super::*;

        #[test]
        fn cuda_compiled_pipeline_identity_uses_shared_domain_separated_contract() {
            for seed in 0_u32..2048 {
                let ptx_key = generated_pipeline_identity_key(seed, 0x5054_5820);
                let module_key = generated_pipeline_identity_key(seed, 0x4D4F_4420);
                let launch = generated_pipeline_identity_launch(seed);

                let key = cuda_compiled_pipeline_identity_key(&ptx_key, &module_key, &launch)
                    .expect("Fix: generated CUDA compiled pipeline key must fit");
                let changed_ptx = cuda_compiled_pipeline_identity_key(
                    &generated_pipeline_identity_key(seed ^ 1, 0x5054_5820),
                    &module_key,
                    &launch,
                )
                .expect("Fix: generated CUDA compiled pipeline PTX variant must fit");
                let changed_module = cuda_compiled_pipeline_identity_key(
                    &ptx_key,
                    &generated_pipeline_identity_key(seed ^ 1, 0x4D4F_4420),
                    &launch,
                )
                .expect("Fix: generated CUDA compiled pipeline module variant must fit");
                let mut changed_launch = launch.clone();
                changed_launch.grid[0] = changed_launch.grid[0].wrapping_add(1);
                let changed_launch_key =
                    cuda_compiled_pipeline_identity_key(&ptx_key, &module_key, &changed_launch)
                        .expect("Fix: generated CUDA compiled pipeline launch variant must fit");

                assert_ne!(key, changed_ptx);
                assert_ne!(key, changed_module);
                assert_ne!(key, changed_launch_key);
            }
        }

        #[test]
        fn cuda_pipeline_dynamic_dispatch_reuses_existing_output_slots() {
            let mut outputs = vec![Vec::with_capacity(8), Vec::with_capacity(4)];
            let outputs_addr = outputs.as_ptr() as usize;
            let first_slot_addr = outputs[0].as_ptr() as usize;
            let second_slot_addr = outputs[1].as_ptr() as usize;

            replace_output_buffers_preserving_slots(vec![vec![1, 2, 3], vec![4]], &mut outputs);

            assert_eq!(outputs, vec![vec![1, 2, 3], vec![4]]);
            assert_eq!(outputs.as_ptr() as usize, outputs_addr);
            assert_eq!(outputs[0].as_ptr() as usize, first_slot_addr);
            assert_eq!(outputs[1].as_ptr() as usize, second_slot_addr);
        }

        #[test]
        fn cuda_graph_lane_planner_scales_past_legacy_four_lane_cap() {
            let caps = synthetic_sm120_envelope(32 * 1024 * 1024 * 1024);
            let plan = single_input_output_plan(1024);
            let input = vec![7_u8; 1024];
            let row = [input.as_slice()];
            let batches: Vec<&[&[u8]]> = vec![row.as_slice(); 64];

            let lanes = cuda_graph_lane_count_for_batch(&caps, &plan, &batches)
                .expect("Fix: graph replay lane planning should fit");

            assert!(lanes > 4);
            assert_eq!(lanes, 22);
        }

        #[test]
        fn cuda_graph_lane_planner_caps_large_graphs_by_vram_budget() {
            let caps = synthetic_sm120_envelope(512 * 1024 * 1024);
            let plan = single_input_output_plan(64 * 1024 * 1024);
            let input = vec![1_u8; 64 * 1024 * 1024];
            let row = [input.as_slice()];
            let batches: Vec<&[&[u8]]> = vec![row.as_slice(); 64];

            let lanes = cuda_graph_lane_count_for_batch(&caps, &plan, &batches)
                .expect("Fix: graph replay lane planning should fit");

            assert_eq!(lanes, 1);
        }

        #[test]
        fn execution_strategy_covers_every_graph_cooperative_and_trap_combination() {
            use crate::pipeline::{
                select_cuda_pipeline_execution_strategy, CudaPipelineExecutionStrategy,
            };

            for graph_replay_enabled in [false, true] {
                for cooperative in [false, true] {
                    for declares_trap in [false, true] {
                        let expected =
                            if graph_replay_enabled && !cooperative && !declares_trap {
                                CudaPipelineExecutionStrategy::GraphReplay
                            } else {
                                CudaPipelineExecutionStrategy::DirectDispatch
                            };
                        assert_eq!(
                            select_cuda_pipeline_execution_strategy(
                                graph_replay_enabled,
                                cooperative,
                                declares_trap,
                            ),
                            expected,
                            "strategy mismatch for graph_replay_enabled={graph_replay_enabled}, cooperative={cooperative}, declares_trap={declares_trap}",
                        );
                    }
                }
            }
        }

        #[test]
        fn trap_declaration_detection_identifies_ir_stats_and_ptx_sidecar() {
            use vyre_foundation::ir::Program;

            let prog_plain = Arc::new(Program::empty());
            assert!(!prog_plain.stats().trap());

            let prog_trap = Arc::new(vyre_foundation::composition::trap_program(
                "test.trap.op",
                None,
                "domain violation",
            ));
            assert!(prog_trap.stats().trap());

            let ptx_plain = ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry main() { ret; }\n";
            assert!(!crate::backend::module_cache::declares_trap_sidecar(ptx_plain));

            let ptx_trap = format!(
                ".version 7.0\n.target sm_70\n.address_size 64\n.global .align 4 .u32 {}[4];\n.visible .entry main() {{ ret; }}\n",
                vyre_emit_ptx::TRAP_SIDECAR_SYMBOL
            );
            assert!(crate::backend::module_cache::declares_trap_sidecar(&ptx_trap));
        }
    }
}
