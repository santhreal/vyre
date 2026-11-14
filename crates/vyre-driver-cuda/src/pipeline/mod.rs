//! Precompiled CUDA pipeline implementation.

use std::sync::{Arc, Mutex};

use smallvec::SmallVec;
use vyre_runtime::input_identity::{domain_separated_exact_input_key, ExactInputKey};
use vyre_runtime::{
    dispatch_configs_share_launch_shape, sealed, BackendError, DispatchConfig, LaunchPlan,
};
use vyre_foundation::ir::Program;

use crate::backend::allocations::DeviceAllocation;
use crate::backend::{CachedCudaGraph, CudaBackend, CudaDispatchPlan, ModuleCacheKey};

mod compiled_dispatch;
mod compiled_graph_replay;
mod static_params;

use static_params::upload_static_launch_params;

/// Mutually exclusive GPU execution strategy selected for a compiled CUDA pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaPipelineExecutionStrategy {
    /// Cached CUDA graph record and replay.
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
    static_params: DeviceAllocation,
    id: String,
}

pub(crate) const MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE: usize = 32;
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

    /// What the driver assigned this pipeline's entry point after compiling its
    /// PTX for the acquired device.
    pub(crate) fn emitted_resources(
        &self,
    ) -> Result<vyre_compiler::EmittedResources, BackendError> {
        self.backend
            .module_resources_with_key(&self.ptx_src, self.module_key)
    }

    /// Device bytes the backend behind this pipeline holds resident.
    pub(crate) fn resident_device_bytes(&self) -> Result<u64, BackendError> {
        self.backend.resident_device_bytes()
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

/// Reject a dispatch whose config changes the launch shape the pipeline was
/// compiled for.
///
/// The dispatch plan, the uploaded static launch parameters and every recorded
/// CUDA graph are products of the compiled config, and an artifact module
/// dispatches with the config it was compiled under. A config that changes the
/// shape therefore has no compiled launch to run.
pub(crate) fn require_compiled_launch_shape(
    compiled: &DispatchConfig,
    runtime: &DispatchConfig,
) -> Result<(), BackendError> {
    if dispatch_configs_share_launch_shape(compiled, runtime) {
        return Ok(());
    }
    Err(BackendError::InvalidProgram {
        fix: "Fix: a compiled CUDA pipeline was dispatched with a config that changes its launch shape, as `vyre_runtime::dispatch_configs_share_launch_shape` defines it. Compile the artifact for that config and dispatch the instance it materializes.".to_string(),
    })
}

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

// Inline: `pipeline` is a private module, so `cuda_compiled_pipeline_identity_key`
// and the execution-strategy selector are unreachable from an integration test.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vyre_runtime::replace_output_buffers_preserving_slots;
    use vyre_runtime::LaunchPlan;

    use super::{cuda_compiled_pipeline_identity_key, require_compiled_launch_shape};

    /// A compiled pipeline runs only the launch shape it was compiled for: a
    /// config differing in a field that is not part of the shape dispatches, and
    /// one differing in a shape field is rejected with the recompile fix.
    #[test]
    fn dispatch_config_must_preserve_the_compiled_launch_shape() {
        let compiled = vyre_runtime::DispatchConfig::default();
        let mut timeout_only = compiled.clone();
        timeout_only.timeout = Some(std::time::Duration::from_millis(1));
        require_compiled_launch_shape(&compiled, &compiled)
            .expect("Fix: the compiled config must dispatch");
        require_compiled_launch_shape(&compiled, &timeout_only)
            .expect("Fix: a timeout is not part of the launch shape");

        let mut regridded = compiled.clone();
        regridded.grid_override = Some([2, 1, 1]);
        let mut refixed = compiled.clone();
        refixed.fixpoint_iterations = Some(2);
        for reshaped in [regridded, refixed] {
            let error = require_compiled_launch_shape(&compiled, &reshaped)
                .expect_err("Fix: a reshaped config must be rejected");
            let vyre_runtime::BackendError::InvalidProgram { fix } = error else {
                panic!("Fix: a reshaped config must report InvalidProgram, got {error:?}");
            };
            assert!(
                fix.starts_with("Fix: a compiled CUDA pipeline was dispatched with a config that changes its launch shape")
                    && fix.ends_with("Compile the artifact for that config and dispatch the instance it materializes."),
                "{fix}"
            );
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
        fn execution_strategy_covers_every_graph_cooperative_and_trap_combination() {
            use crate::pipeline::{
                select_cuda_pipeline_execution_strategy, CudaPipelineExecutionStrategy,
            };

            for graph_replay_enabled in [false, true] {
                for cooperative in [false, true] {
                    for declares_trap in [false, true] {
                        let expected = if graph_replay_enabled && !cooperative && !declares_trap {
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

            let ptx_plain =
                ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry main() { ret; }\n";
            assert!(!crate::backend::module_cache::declares_trap_sidecar(
                ptx_plain
            ));

            let ptx_trap = format!(
                ".version 7.0\n.target sm_70\n.address_size 64\n.global .align 4 .u32 {}[4];\n.visible .entry main() {{ ret; }}\n",
                vyre_emit::ptx::TRAP_SIDECAR_SYMBOL
            );
            assert!(crate::backend::module_cache::declares_trap_sidecar(
                &ptx_trap
            ));
        }
    }
}
