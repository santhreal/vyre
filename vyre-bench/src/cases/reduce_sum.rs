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
const MAX_TREE_TILE: u32 = 256;
const ROUTE_ATOMIC: u64 = 0;
const ROUTE_TREE: u64 = 1;

struct ReductionSizePrepared {
    count: u32,
    tree_tile: u32,
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

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let small = prepare_size(SMALL_COUNT);
        let large = prepare_size(LARGE_COUNT);

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
            small.selected.outputs[0].clone(),
            large.selected.outputs[0].clone(),
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

fn prepare_size(count: u32) -> ReductionSizePrepared {
    let values: Vec<u32> = (0..count)
        .map(|index| index.wrapping_mul(17).wrapping_add(3) & 0xff)
        .collect();
    let expected = values.iter().copied().fold(0u32, u32::wrapping_add);
    let tree_tile = count.min(MAX_TREE_TILE).max(1).next_power_of_two();
    ReductionSizePrepared {
        count,
        tree_tile,
        values: values.clone(),
        atomic_program: sum::reduce_sum("values", "out", count),
        tree_program: grid_stride_tree::grid_stride_tree_sum_u32("values", "out", count, tree_tile),
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

    let tree = ctx
        .dispatch_timed(
            &prepared.tree_program,
            &prepared.inputs,
            &ctx.dispatch_config,
        )
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
    if outputs == [expected] {
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

    #[test]
    fn small_and_large_reduction_sizes_prepare_expected_values() {
        let small = prepare_size(SMALL_COUNT);
        assert_eq!(small.count, 32);
        assert_eq!(small.tree_tile, 32);
        assert_eq!(small.inputs[0].len(), 32 * 4);
        assert_eq!(small.expected.len(), 4);

        let large = prepare_size(LARGE_COUNT);
        assert_eq!(large.count, 1 << 20);
        assert_eq!(large.tree_tile, MAX_TREE_TILE);
        assert_eq!(large.inputs[0].len(), (1 << 20) * 4);
        assert_eq!(large.expected.len(), 4);
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
