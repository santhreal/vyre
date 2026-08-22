use crate::api::case::{
    prepared_as, BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata,
    BenchRequirements, BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase,
    WorkloadClass,
};
use crate::api::metric::{elapsed_ns, BenchMetrics, MetricPoint};
use vyre::ir::Program;
use vyre_driver::TimedDispatchResult;
use vyre_libs::reduce::{grid_stride_tree, sum};

pub struct ReduceSumBench;

const SMALL_COUNT: u32 = 32;
const LARGE_COUNT: u32 = 1 << 20;
const ROUTE_ATOMIC: u64 = 0;
const ROUTE_TREE: u64 = 1;

struct ReductionSizePrepared {
    count: u32,
    tree_tile: u32,
    tree_grid: Option<[u32; 3]>,
    values: Vec<u32>,
    atomic_program: Program,
    tree_program: Program,
    inputs: [Vec<u8>; 2],
    expected: Vec<u8>,
}

struct ReduceSumPrepared {
    small: ReductionSizePrepared,
    large: ReductionSizePrepared,
    baseline_wall_ns: u64,
}

impl BenchCase for ReduceSumBench {
    fn id(&self) -> BenchId {
        BenchId("foundation.reduce.sum.crossover".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Reduce Sum Atomic/Tree Crossover".to_string(),
            description:
                "Measure exact u32 atomic-scalar and workgroup-tree reductions at 32 and 1,048,576 elements, then route each size through its measured winner"
                    .to_string(),
            tags: vec![
                "compute".to_string(),
                "memory-bound".to_string(),
                "reduction".to_string(),
                "contention".to_string(),
                "adaptive-routing".to_string(),
            ],
            layer: BenchLayer::Foundation,
            workload: WorkloadClass::Micro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-bench".to_string(),
        }
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: Some(u64::from(LARGE_COUNT) * 4),
            min_input_bytes: Some(u64::from(LARGE_COUNT) * 4),
            feature_set: vec!["reduce.atomic-tree-crossover".to_string()],
        }
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_min_speedup(
            "measured-winner u32 reduction sum",
            "rayon",
            "rayon CPU reduction baseline",
            1.1,
        ))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        // The fused tree reduction carries a whole-grid fence, so it launches
        // cooperatively and every workgroup must be co-resident. One workgroup
        // per compute unit is the widest grid that holds at this tile, and it
        // is read from the probed device so the grid is a fact of the machine
        // being measured rather than a constant baked into the composition.
        let profile = ctx.preferred_backend.device_profile();
        let tree_blocks = profile.compute_units.max(1);
        let tile_ceiling = tree_tile_ceiling(&profile);
        let small = prepare_size(SMALL_COUNT, tree_blocks, tile_ceiling);
        let large = prepare_size(LARGE_COUNT, tree_blocks, tile_ceiling);

        let pool = crate::cases::cpu_baselines::baseline_pool();
        let mut durations = Vec::with_capacity(11);
        for _ in 0..11 {
            let start = std::time::Instant::now();
            let (_s, _l) = pool.install(|| {
                use rayon::prelude::*;
                let s: u32 = small.values.par_iter().copied().sum();
                let l: u32 = large.values.par_iter().copied().sum();
                (s, l)
            });
            durations.push(elapsed_ns(start));
        }
        durations.sort_unstable();
        let baseline_wall_ns = durations[durations.len() / 2];

        Ok(Box::new(ReduceSumPrepared {
            small,
            large,
            baseline_wall_ns,
        }))
    }

    fn program<'a>(&self, _prepared: &'a PreparedCase) -> Option<&'a Program> {
        None
    }

    fn workload_fingerprint_bytes(&self, prepared: &PreparedCase) -> Option<[u8; 32]> {
        let prepared = prepared.downcast_ref::<ReduceSumPrepared>()?;
        let mut hasher = blake3::Hasher::new();
        for size in [&prepared.small, &prepared.large] {
            hasher.update(&size.count.to_le_bytes());
            hasher.update(&size.tree_tile.to_le_bytes());
            hasher.update(&size.tree_grid.unwrap_or([0; 3])[0].to_le_bytes());
            hasher.update(&size.atomic_program.fingerprint());
            hasher.update(&size.tree_program.fingerprint());
        }
        Some(*hasher.finalize().as_bytes())
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared_as::<ReduceSumPrepared>(prepared, "reduce-sum crossover")?;

        let small = measure_size(ctx, &prepared.small, "small")?;
        let large = measure_size(ctx, &prepared.large, "large")?;
        let wall_ns = small
            .selected
            .wall_ns
            .saturating_add(large.selected.wall_ns);
        let dispatch_ns = match (small.selected.device_ns, large.selected.device_ns) {
            (Some(small_ns), Some(large_ns)) => Some(small_ns.saturating_add(large_ns)),
            _ => None,
        };
        let outputs = vec![
            small.selected.outputs.last().cloned().unwrap_or_default(),
            large.selected.outputs.last().cloned().unwrap_or_default(),
        ];
        let baseline_outputs = vec![
            prepared.small.expected.clone(),
            prepared.large.expected.clone(),
        ];

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(wall_ns),
                dispatch_ns,
                input_bytes: Some(u64::from(SMALL_COUNT.saturating_add(LARGE_COUNT)) * 4),
                output_bytes: Some(8),
                custom: vec![
                    metric("reduction_small_count", u64::from(SMALL_COUNT)),
                    metric("reduction_large_count", u64::from(LARGE_COUNT)),
                    metric(
                        "reduction_timing_source_device",
                        u64::from(small.device_timing),
                    ),
                    metric("reduction_small_atomic_ns", small.atomic_ns),
                    metric("reduction_small_tree_ns", small.tree_ns),
                    metric("reduction_small_selected_route", small.selected_route),
                    metric("reduction_large_atomic_ns", large.atomic_ns),
                    metric("reduction_large_tree_ns", large.tree_ns),
                    metric("reduction_large_selected_route", large.selected_route),
                    metric(
                        "reduction_small_atomic_contended_updates",
                        u64::from(SMALL_COUNT),
                    ),
                    metric(
                        "reduction_large_atomic_contended_updates",
                        u64::from(LARGE_COUNT),
                    ),
                    metric("reduction_small_tree_contended_updates", 0),
                    metric("reduction_large_tree_contended_updates", 0),
                    metric(
                        "reduction_small_tree_barrier_rounds",
                        u64::from(tree_barrier_rounds(prepared.small.tree_tile)),
                    ),
                    metric(
                        "reduction_large_tree_barrier_rounds",
                        u64::from(tree_barrier_rounds(prepared.large.tree_tile)),
                    ),
                ],
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(prepared.baseline_wall_ns),
                input_bytes: Some(u64::from(SMALL_COUNT.saturating_add(LARGE_COUNT)) * 4),
                output_bytes: Some(8),
                custom: vec![metric(
                    "flop_count",
                    u64::from(SMALL_COUNT.saturating_add(LARGE_COUNT)),
                )],
                ..Default::default()
            }),
            outputs,
            baseline_outputs: Some(baseline_outputs),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }

    fn bytes_touched(&self, _prepared: &PreparedCase) -> (u64, u64) {
        (u64::from(SMALL_COUNT.saturating_add(LARGE_COUNT)) * 4, 8)
    }
}

struct MeasuredSize {
    atomic_ns: u64,
    tree_ns: u64,
    selected_route: u64,
    device_timing: bool,
    selected: TimedDispatchResult,
}

/// Largest tile the tree reduction may launch on the measured device.
///
/// The tree halves its active lanes each round, so the tile is the admitted
/// workgroup extent floored to a power of two. Both limits come from the
/// probed device: a backend whose target dialect admits fewer invocations than
/// its adapter advertises rejects a payload sized for the adapter's number,
/// and WGSL admits the WebGPU spec baseline of 256 where CUDA admits 1024.
fn tree_tile_ceiling(profile: &vyre_driver::DeviceProfile) -> u32 {
    let admitted = profile.max_workgroup_size[0]
        .min(profile.max_invocations_per_workgroup)
        .max(1);
    1u32 << admitted.ilog2()
}

fn prepare_size(count: u32, tree_blocks: u32, tile_ceiling: u32) -> ReductionSizePrepared {
    let values: Vec<u32> = (0..count)
        .map(|index| index.wrapping_mul(17).wrapping_add(3) & 0xff)
        .collect();
    let expected = values.iter().copied().fold(0u32, u32::wrapping_add);
    let tree_tile = count.min(tile_ceiling).max(1).next_power_of_two();
    let tree_blocks =
        grid_stride_tree::grid_stride_tree_sum_u32_blocks(count, tree_tile, tree_blocks);
    ReductionSizePrepared {
        count,
        tree_tile,
        // The tree program's grid is a contract of the program at every block
        // count: pass 1 strides the input over exactly this many blocks and
        // sizes its partial buffer to them. Leaving the launch to inference
        // spans the widest declared buffer instead, which is the whole input.
        tree_grid: Some([tree_blocks, 1, 1]),
        values: values.clone(),
        atomic_program: sum::reduce_sum("values", "out", count),
        tree_program: grid_stride_tree::grid_stride_tree_sum_u32(
            "values",
            "out",
            count,
            tree_tile,
            tree_blocks,
        ),
        inputs: [
            crate::cases::byte_pack::u32_bytes(&values),
            crate::cases::byte_pack::u32_bytes(&[0]),
        ],
        expected: crate::cases::byte_pack::u32_bytes(&[expected]),
    }
}

fn measure_size(
    ctx: &BenchContext,
    prepared: &ReductionSizePrepared,
    size_name: &str,
) -> Result<MeasuredSize, BenchError> {
    let atomic = ctx
        .dispatch_timed(
            &prepared.atomic_program,
            &prepared.inputs,
            &ctx.dispatch_config,
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    verify_route_output(size_name, "atomic", &atomic.outputs, &prepared.expected)?;

    let mut tree_config = ctx.dispatch_config.clone();
    if let Some(grid) = prepared.tree_grid {
        tree_config.grid_override = Some(grid);
    }
    let tree = ctx
        .dispatch_timed(&prepared.tree_program, &prepared.inputs, &tree_config)
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    verify_route_output(size_name, "tree", &tree.outputs, &prepared.expected)?;

    let (atomic_ns, tree_ns) = match (atomic.device_ns, tree.device_ns) {
        (Some(a), Some(t)) if a > 0 && t > 0 => (a, t),
        (Some(a), Some(t)) => {
            return Err(BenchError::BackendFailed(format!(
                "{size_name} reduction routes reported zero device timing: atomic={a} ns, tree={t} ns"
            )));
        }
        (a, t) => {
            return Err(BenchError::BackendFailed(format!(
                "{size_name} reduction routes missing device timing: atomic={a:?}, tree={t:?}"
            )));
        }
    };
    if size_name == "large" && atomic_ns == tree_ns {
        return Err(BenchError::BackendFailed(format!(
            "large reduction routes reported identical device timings ({atomic_ns} ns); selector cannot determine measured winner"
        )));
    }
    let (selected_route, selected) = if atomic_ns <= tree_ns {
        (ROUTE_ATOMIC, atomic)
    } else {
        (ROUTE_TREE, tree)
    };

    Ok(MeasuredSize {
        atomic_ns,
        tree_ns,
        selected_route,
        device_timing: true,
        selected,
    })
}

fn verify_route_output(
    size_name: &str,
    route_name: &str,
    outputs: &[Vec<u8>],
    expected: &[u8],
) -> Result<(), BenchError> {
    if outputs.last().map(Vec::as_slice) == Some(expected) {
        return Ok(());
    }
    Err(BenchError::CorrectnessViolation(format!(
        "{size_name} {route_name} reduction output mismatch: expected {expected:02x?}, got {outputs:02x?}"
    )))
}

fn tree_barrier_rounds(tile: u32) -> u32 {
    tile.ilog2().saturating_add(1)
}

fn metric(name: &str, value: u64) -> MetricPoint {
    MetricPoint {
        name: name.to_string(),
        value,
    }
}
inventory::submit! {
    &ReduceSumBench as &'static dyn BenchCase
}
#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with(max_workgroup_x: u32, max_invocations: u32) -> vyre_driver::DeviceProfile {
        let mut profile = vyre_driver::DeviceProfile::conservative("test");
        profile.max_workgroup_size = [max_workgroup_x, 1, 1];
        profile.max_invocations_per_workgroup = max_invocations;
        profile
    }

    #[test]
    fn the_tile_ceiling_is_the_smaller_of_the_two_workgroup_facts() {
        assert_eq!(tree_tile_ceiling(&profile_with(1024, 1024)), 1024);
        assert_eq!(
            tree_tile_ceiling(&profile_with(1024, 256)),
            256,
            "Fix: a dialect that admits 256 invocations rejects a 1024-wide tile the adapter allows"
        );
        assert_eq!(tree_tile_ceiling(&profile_with(256, 1024)), 256);
        assert_eq!(
            tree_tile_ceiling(&profile_with(768, 768)),
            512,
            "Fix: the tree halves its lanes each round, so a tile must be a power of two"
        );
        assert_eq!(
            tree_tile_ceiling(&vyre_driver::DeviceProfile::conservative("unprobed")),
            1,
            "Fix: an unprobed backend admits one invocation, not a baked-in default"
        );
    }

    #[test]
    fn small_and_large_reduction_sizes_prepare_expected_values() {
        // The block count a 4090-class device reports; the clamp is what turns
        // it into a legal grid for each size. Both tile ceilings are real: CUDA
        // admits 1024 invocations per workgroup, WGSL admits 256.
        let compute_units = 170;

        for tile_ceiling in [1024, 256] {
            let small = prepare_size(SMALL_COUNT, compute_units, tile_ceiling);
            assert_eq!(small.count, 32);
            assert_eq!(small.tree_tile, 32);
            assert_eq!(small.inputs[0].len(), 32 * 4);
            assert_eq!(small.expected.len(), 4);
            assert_eq!(
                small.tree_grid, None,
                "Fix: 32 elements at tile 32 need one block, and a one-block tree infers its own grid"
            );

            let large = prepare_size(LARGE_COUNT, compute_units, tile_ceiling);
            assert_eq!(large.count, 1 << 20);
            assert_eq!(
                large.tree_tile, tile_ceiling,
                "Fix: a million elements fill whatever tile the device admits"
            );
            assert_eq!(large.inputs[0].len(), (1 << 20) * 4);
            assert_eq!(large.expected.len(), 4);
            assert_eq!(
                large.tree_grid,
                Some([compute_units, 1, 1]),
                "Fix: the launch must pin the block count the pass-one loop was built for"
            );

            // Pass two reduces the per-block partials inside one tile-wide
            // workgroup, so the block count is capped by the tile as well as by
            // the number of tiles the input fills.
            let saturated = prepare_size(LARGE_COUNT, 100_000, tile_ceiling);
            assert_eq!(
                saturated.tree_grid,
                Some([tile_ceiling.min(LARGE_COUNT / tile_ceiling), 1, 1]),
                "Fix: more blocks than tiles leaves blocks with nothing to reduce"
            );
        }
    }

    #[test]
    fn verify_route_output_rejects_mismatch_and_accepts_expected() {
        let expected = vec![1, 2, 3, 4];
        let matching = vec![vec![1, 2, 3, 4]];
        let mismatch = vec![vec![1, 2, 3, 5]];

        assert!(verify_route_output("test", "atomic", &matching, &expected).is_ok());
        assert!(verify_route_output("test", "atomic", &mismatch, &expected).is_err());
    }

    #[test]
    fn tree_barrier_rounds_computes_expected_log2_rounds() {
        assert_eq!(tree_barrier_rounds(32), 6);
        assert_eq!(tree_barrier_rounds(256), 9);
    }
}
