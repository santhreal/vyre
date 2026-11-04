//! Resident dispatch helpers for WGPU backend resources.

use crate::WgpuBackend;
use std::time::Instant;
use vyre_driver::CompiledPipeline;
use vyre_foundation::ir::Program;

/// Dispatch a program with backend-resident resources and return timing.
pub(crate) fn dispatch_resident_timed(
    backend: &WgpuBackend,
    program: &Program,
    resources: &[vyre_driver::Resource],
    config: &vyre_driver::DispatchConfig,
) -> Result<vyre_driver::TimedDispatchResult, vyre_driver::BackendError> {
    let started = Instant::now();
    if vyre_driver::grid_sync::contains_grid_sync(program) {
        return vyre_driver::grid_sync::dispatch_resident_with_grid_sync_split_timed(
            backend, program, resources, config,
        );
    }
    let pipeline = backend.compile_resident_pipeline_cached(program, config)?;
    let timed = pipeline.dispatch_persistent_handles_timed(resources, config)?;
    Ok(vyre_driver::TimedDispatchResult {
        outputs: timed.outputs,
        wall_ns: crate::numeric::WGPU_NUMERIC
            .elapsed_nanos_u64(started, "resident timed dispatch")?,
        device_ns: timed.device_ns,
        enqueue_ns: timed.enqueue_ns,
        wait_ns: timed.wait_ns,
    })
}
/// Submit resident work and return before compute or readback completes.
pub(crate) fn dispatch_resident_async(
    backend: &WgpuBackend,
    program: &Program,
    resources: &[vyre_driver::Resource],
    config: &vyre_driver::DispatchConfig,
) -> Result<Box<dyn vyre_driver::PendingDispatch>, vyre_driver::BackendError> {
    let started = Instant::now();
    if vyre_driver::grid_sync::contains_grid_sync(program) {
        let outputs = vyre_driver::grid_sync::dispatch_resident_with_grid_sync_split_timed(
            backend, program, resources, config,
        )?
        .outputs;
        return Ok(Box::new(crate::async_dispatch::WgpuPendingDispatch::ready(
            outputs,
            started,
            config.timeout,
        )));
    }
    let pipeline = backend.compile_resident_pipeline_cached(program, config)?;
    Ok(Box::new(pipeline.dispatch_persistent_handles_async(
        resources, config, started,
    )?))
}
