use vyre_driver::validation::LaunchGeometryLimits;
use vyre_driver::{resolve_launch_workgroup_for_geometry, LaunchGeometry};
use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

pub(super) fn wgpu_effective_dispatch_config(
    program: &Program,
    config: &DispatchConfig,
    device: &wgpu::Device,
    geometry: LaunchGeometry,
) -> Result<DispatchConfig, BackendError> {
    wgpu_effective_dispatch_config_for_limits(program, config, wgpu_launch_limits(device), geometry)
}

pub(crate) fn wgpu_effective_dispatch_config_for_limits(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    geometry: LaunchGeometry,
) -> Result<DispatchConfig, BackendError> {
    let mut effective = config.clone();
    if effective.launch.is_some()
        || (geometry == LaunchGeometry::Untracked && effective.workgroup_override.is_some())
    {
        return Ok(effective);
    }
    let element_count = wgpu_launch_element_count_for_tuning(program)?;
    let selected =
        resolve_launch_workgroup_for_geometry(program, &effective, limits, element_count, geometry);
    if selected != program.workgroup_size() {
        effective.workgroup_override = Some(selected);
    } else {
        effective.workgroup_override = None;
    }
    Ok(effective)
}

fn wgpu_launch_element_count_for_tuning(program: &Program) -> Result<u32, BackendError> {
    if program.output_buffer_indices().is_empty() {
        return Ok(0);
    }
    let layouts = vyre_driver::output_binding_layouts(program)?;
    let word_count = layouts
        .first()
        .map(|layout| layout.word_count)
        .unwrap_or_default();
    u32::try_from(word_count).map_err(|error| {
        BackendError::new(format!(
            "wgpu launch preparation cannot represent {word_count} output word(s) as u32: {error}. Fix: split the dispatch or provide an explicit workgroup/grid override."
        ))
    })
}

pub(crate) fn wgpu_launch_limits(device: &wgpu::Device) -> LaunchGeometryLimits {
    let limits = device.limits();
    LaunchGeometryLimits {
        backend: "wgpu",
        // The dialect compiles what it declares, not what the adapter allows,
        // so a geometry above its ceiling is rejected at payload construction
        // rather than launched.
        max_threads_per_block: crate::target_compiler::admissible_invocations_per_workgroup(
            limits.max_compute_invocations_per_workgroup,
        ),
        max_block_dim: crate::target_compiler::admissible_workgroup_size([
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        ]),
        max_grid_dim: [limits.max_compute_workgroups_per_dimension; 3],
        // WebGPU exposes no per-compute-unit thread budget, so wgpu reports
        // none. Zero keeps residency-aware launch decisions inert here rather
        // than deriving one from a number this API never supplies.
        max_threads_per_sm: 0,
    }
}
