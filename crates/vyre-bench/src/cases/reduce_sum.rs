use crate::api::case::{
    host_input_bundle, prepared_as, BenchCase, BenchContext, BenchError, BenchId, BenchLayer,
    BenchMetadata, BenchRequirements, BenchRun, Correctness, DeterminismClass, PerformanceContract,
    PreparedCase, WorkloadClass,
};
use crate::api::metric::{elapsed_ns, BenchMetrics, MetricPoint};
use crate::api::resident::ResidentInputSet;
use vyre::ir::{BufferAccess, Program};
use vyre_driver::TimedDispatchResult;
use vyre_libs::reduce::{grid_stride_tree, sum};

pub struct ReduceSumBench;

const SMALL_COUNT: u32 = 32;
const LARGE_COUNT: u32 = 1 << 20;
const ROUTE_ATOMIC: u64 = 0;
const ROUTE_TREE: u64 = 1;

/// One reduction route the case measures at a given size.
///
/// The route owns the host input bundle it is dispatched with, derived from
/// its own program. A route added here is measured, is fingerprinted, and is
/// checked for ABI arity without anything else naming it.
///
/// The bundle reaches the device once. Every measured dispatch then binds the
/// resident resources uploaded from it, so reducing the same values a second
/// time moves no input bytes across the host boundary, which is the traffic an
/// application reducing one resident tensor repeatedly pays. The rayon
/// baseline sums a `Vec<u32>` that is already in host memory and copies
/// nothing per iteration, so the host side of the comparison has nothing to
/// skip and both sides read values they already hold.
pub struct ReductionRoute {
    /// Route name in diagnostics.
    pub name: &'static str,
    /// Route discriminant recorded in the selected-route metric.
    pub route_id: u64,
    /// The program dispatched for this route.
    pub program: Program,
    /// Host bytes bound in artifact ABI slot order.
    pub inputs: Vec<Vec<u8>>,
    /// Resident resources every dispatch binds, once the backend admits them.
    resident: Option<ResidentInputSet>,
    /// Resident resource index and seed bytes of every caller-seeded accumulator.
    reseed: Vec<(usize, Vec<u8>)>,
}

/// One measured input size and every route that reduces it.
pub struct ReductionSizePrepared {
    /// Element count reduced at this size.
    pub count: u32,
    /// Tile width the tree route reduces within.
    pub tree_tile: u32,
    /// The atomic route, then the fused tree route.
    pub routes: [ReductionRoute; 2],
    values: Vec<u32>,
    expected: Vec<u8>,
}

impl ReductionSizePrepared {
    /// The atomic route.
    fn atomic(&self) -> &ReductionRoute {
        &self.routes[0]
    }

    /// The fused tree route.
    fn tree(&self) -> &ReductionRoute {
        &self.routes[1]
    }
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
        // The tile is the widest workgroup extent the probed device admits, and
        // the tree builder derives its workgroup count from that tile and the
        // element count. A compiled artifact records the launch its own buffer
        // table implies, so neither this case nor the device profile can state
        // a grid the dispatch would honour.
        let profile = ctx.preferred_backend.device_profile();
        let tile_ceiling = tree_tile_ceiling(&profile);
        let mut small = prepare_size(SMALL_COUNT, tile_ceiling)?;
        let mut large = prepare_size(LARGE_COUNT, tile_ceiling)?;
        upload_resident_routes(ctx, &mut small)?;
        upload_resident_routes(ctx, &mut large)?;

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
            // Every route, not the two this case happens to declare: a route
            // added to the roster changes the workload and must change the
            // fingerprint that names it.
            for route in &size.routes {
                hasher.update(&route.route_id.to_le_bytes());
                hasher.update(&route.program.fingerprint());
            }
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

fn prepare_size(count: u32, tile_ceiling: u32) -> Result<ReductionSizePrepared, BenchError> {
    let values: Vec<u32> = (0..count)
        .map(|index| index.wrapping_mul(17).wrapping_add(3) & 0xff)
        .collect();
    let expected = values.iter().copied().fold(0u32, u32::wrapping_add);
    let tree_tile = count.min(tile_ceiling).max(1).next_power_of_two();

    let value_bytes = crate::cases::byte_pack::u32_bytes(&values);
    let out_seed = crate::cases::byte_pack::u32_bytes(&[0]);
    let named: [(&str, &[u8]); 2] = [("values", &value_bytes), ("out", &out_seed)];

    let atomic_program = sum::reduce_sum("values", "out", count);
    let tree_program =
        grid_stride_tree::grid_stride_tree_sum_u32("values", "out", count, tree_tile);

    Ok(ReductionSizePrepared {
        count,
        tree_tile,
        routes: [
            ReductionRoute {
                name: "atomic",
                route_id: ROUTE_ATOMIC,
                inputs: host_input_bundle(&atomic_program, &named)?,
                reseed: caller_seeded_accumulators(&atomic_program, &named)?,
                program: atomic_program,
                resident: None,
            },
            ReductionRoute {
                name: "tree",
                route_id: ROUTE_TREE,
                inputs: host_input_bundle(&tree_program, &named)?,
                reseed: caller_seeded_accumulators(&tree_program, &named)?,
                program: tree_program,
                resident: None,
            },
        ],
        values,
        expected: crate::cases::byte_pack::u32_bytes(&[expected]),
    })
}

/// Upload every route's input bundle once, before any measured dispatch.
///
/// A device backend without resident allocation is a failure rather than a
/// silent fall back to host bindings: the case would then measure a 4 MiB
/// upload per sample against a baseline that copies nothing, and the reported
/// speedup would describe the transfer instead of the reduction.
fn upload_resident_routes(
    ctx: &BenchContext,
    prepared: &mut ReductionSizePrepared,
) -> Result<(), BenchError> {
    for route in &mut prepared.routes {
        route.resident = ResidentInputSet::upload_program_ordered_with_zeroed_outputs_optional(
            ctx,
            &route.program,
            &route.inputs,
            "reduce-sum crossover",
        )?;
        if route.resident.is_none() && matches!(ctx.preferred_backend.id(), "cuda" | "wgpu") {
            return Err(BenchError::BackendFailed(format!(
                "{} lacks resident buffer allocation required to reduce a resident input without re-uploading it every dispatch",
                ctx.preferred_backend.id()
            )));
        }
    }
    Ok(())
}

/// Resident resource index and seed bytes of every accumulator the caller seeds.
///
/// A buffer that both consumes a host input slot and is read-write is an
/// accumulator: the dispatch adds into whatever bytes it finds there, so the
/// atomic route reads the total of every earlier dispatch unless the seed is
/// restored first. Resident inputs stay on the device between dispatches, and
/// this is the one binding that must not, so it is re-seeded from its own
/// declared bytes. The index is the position among non-shared bindings, which
/// is the order the resident resource handles are allocated in.
fn caller_seeded_accumulators(
    program: &Program,
    named: &[(&str, &[u8])],
) -> Result<Vec<(usize, Vec<u8>)>, BenchError> {
    program
        .buffers()
        .iter()
        .filter(|decl| decl.access != BufferAccess::Workgroup)
        .enumerate()
        .filter(|(_, decl)| {
            decl.consumes_host_input() && decl.access == BufferAccess::ReadWrite
        })
        .map(|(index, decl)| {
            named
                .iter()
                .find(|(name, _)| *name == decl.name())
                .map(|(_, bytes)| (index, bytes.to_vec()))
                .ok_or_else(|| {
                    BenchError::ExecutionFailed(format!(
                        "Fix: program buffer `{}` is a caller-seeded accumulator but the case supplied no seed bytes for it.",
                        decl.name()
                    ))
                })
        })
        .collect()
}

fn measure_size(
    ctx: &BenchContext,
    prepared: &ReductionSizePrepared,
    size_name: &str,
) -> Result<MeasuredSize, BenchError> {
    let atomic = dispatch_route(ctx, prepared.atomic(), size_name, &prepared.expected)?;
    let tree = dispatch_route(ctx, prepared.tree(), size_name, &prepared.expected)?;

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
        (prepared.atomic().route_id, atomic)
    } else {
        (prepared.tree().route_id, tree)
    };

    Ok(MeasuredSize {
        atomic_ns,
        tree_ns,
        selected_route,
        device_timing: true,
        selected,
    })
}

/// Dispatch one route against its resident resources and time it.
///
/// The accumulator seed is restored before every dispatch and is four bytes
/// wide; every other binding is already on the device and moves nothing. A
/// backend without residency dispatches the host bundle instead, which is the
/// only shape the comparison can take when the device cannot hold an input.
///
/// The route is dispatched twice and the second dispatch is the timed one. A
/// device timing is the elapsed time between two events recorded around the
/// launch, so whatever the previously dispatched route left the memory system
/// doing is inside that window. The 1,048,576-update atomic route is 1.37 ms
/// of contended accumulation and leaves a tail: the tree route timed directly
/// after it reads 35840 ns against 32864 ns timed before it, with the sample
/// spread three times wider, and at 32 elements whichever route ran first read
/// 3500 ns slower than the same route running second, which moved the reported
/// winner. The first of the two dispatches absorbs that tail, so a route is
/// timed from the device state its own previous launch left rather than from
/// whichever route the measurement loop happened to run before it.
fn dispatch_route(
    ctx: &BenchContext,
    route: &ReductionRoute,
    size_name: &str,
    expected: &[u8],
) -> Result<TimedDispatchResult, BenchError> {
    dispatch_route_once(ctx, route, size_name, expected)?;
    dispatch_route_once(ctx, route, size_name, expected)
}

fn dispatch_route_once(
    ctx: &BenchContext,
    route: &ReductionRoute,
    size_name: &str,
    expected: &[u8],
) -> Result<TimedDispatchResult, BenchError> {
    let config = &ctx.dispatch_config;
    let result = match &route.resident {
        Some(resident) => {
            for (index, seed) in &route.reseed {
                resident.upload_resource(*index, seed, route.name)?;
            }
            resident
                .dispatch_timed(ctx, &route.program, config)
                .map_err(|error| BenchError::BackendFailed(error.to_string()))?
        }
        None => ctx
            .dispatch_timed(&route.program, &route.inputs, config)
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?,
    };
    verify_route_output(size_name, route.name, &result.outputs, expected)?;
    Ok(result)
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

    /// Declared slot count of a route's partial buffer, when it has one.
    fn partial_slots(route: &ReductionRoute) -> Option<u32> {
        route
            .program
            .buffers()
            .iter()
            .find(|buffer| buffer.name().ends_with("_gst_partials"))
            .map(|buffer| buffer.count())
    }

    /// Workgroups a dispatch of `program` runs, read from the program alone.
    ///
    /// A launch spans the widest non-shared binding narrowed to the workgroups
    /// the program's tile guards admit, and a compiled artifact records that
    /// span, so this is the grid the tree route runs at whatever the case or
    /// the device profile would prefer.
    fn launched_workgroups(program: &Program) -> u32 {
        let span = program
            .buffers()
            .iter()
            .filter(|buffer| buffer.access != BufferAccess::Workgroup)
            .map(|buffer| buffer.count())
            .max()
            .unwrap_or(1);
        vyre_foundation::admitted_logical_span(program, span)
            .div_ceil(program.workgroup_size()[0].max(1))
    }

    /// The tree route sizes its partial buffer to the grid its launch runs.
    ///
    /// WHY: this case cannot state a grid, because a compiled artifact records
    /// the launch its own buffer table implies. A block count taken from the
    /// device profile therefore left the surplus workgroups reducing elements
    /// nothing read: one workgroup per compute unit built an 80-workgroup grid
    /// for a launch of 1024, and the reduction measured 0.53x of the rayon
    /// baseline against a release contract of 1.10x. Reading an unreported
    /// count as one workgroup measured 0.08x, which is the same defect at the
    /// other extreme.
    #[test]
    fn the_tree_route_sizes_its_partials_to_the_grid_its_launch_runs() {
        // Both tile ceilings are real: CUDA admits 1024 invocations per
        // workgroup, WGSL admits 256.
        for tile_ceiling in [1024u32, 256] {
            let small =
                prepare_size(SMALL_COUNT, tile_ceiling).expect("small reduction size prepares");
            assert_eq!(small.count, 32);
            assert_eq!(small.tree_tile, 32);
            assert_eq!(small.atomic().inputs[0].len(), 32 * 4);
            assert_eq!(small.expected.len(), 4);
            assert_eq!(
                partial_slots(small.tree()),
                None,
                "Fix: 32 elements fill one tile, so the tree route takes the single-block form and declares no partials"
            );

            let large =
                prepare_size(LARGE_COUNT, tile_ceiling).expect("large reduction size prepares");
            assert_eq!(large.count, 1 << 20);
            assert_eq!(
                large.tree_tile, tile_ceiling,
                "Fix: a million elements fill whatever tile the device admits"
            );
            assert_eq!(large.atomic().inputs[0].len(), (1 << 20) * 4);
            assert_eq!(large.expected.len(), 4);
            assert_eq!(
                partial_slots(large.tree()),
                Some(launched_workgroups(&large.tree().program)),
                "Fix: the launch runs a grid the partial buffer has no slot for, so those workgroups reduce elements nothing reads"
            );
            assert_eq!(
                partial_slots(large.tree()),
                Some(grid_stride_tree::grid_stride_tree_sum_u32_blocks(
                    LARGE_COUNT,
                    tile_ceiling
                )),
                "Fix: one workgroup per span of the input is the whole grid"
            );
        }
    }

    /// WHY: the combine pass used to read the partials from one tile-wide
    /// workgroup, which capped the block count at one tile. At a 256-lane tile
    /// a four-million-element input needs 512 blocks, and the cap handed the
    /// surplus 256 spans back to pass one as strided rereads under a launch
    /// that ran them anyway. The combine now strides the partials, so the
    /// block count follows the input at every tile.
    #[test]
    fn a_block_count_past_one_tile_still_gets_one_slot_each() {
        const PAST_ONE_TILE: u32 = 4 << 20;
        let prepared = prepare_size(PAST_ONE_TILE, 256).expect("reduction size prepares");
        assert_eq!(prepared.tree_tile, 256);
        let blocks = grid_stride_tree::grid_stride_tree_sum_u32_blocks(PAST_ONE_TILE, 256);
        assert!(
            blocks > prepared.tree_tile,
            "Fix: this case only means something while the block count exceeds one tile"
        );
        assert_eq!(partial_slots(prepared.tree()), Some(blocks));
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

    /// Every route stages exactly the host input buffers its own program
    /// declares.
    ///
    /// WHY: the two routes do not share an ABI. The atomic route reads `values`
    /// and accumulates into a host-staged `out`; the fused tree route's `out`
    /// is a pipeline-live output the backend allocates, so it declares one
    /// fewer host input. Binding one fixed pair of buffers to both routes made
    /// the artifact reject the tree launch for supplying two inputs where the
    /// ABI admits one, and the case never ran on any backend that took the
    /// fused route. The roster comes from `size.routes`, so a third route is
    /// held to this without this test being edited.
    #[test]
    fn every_reduction_route_stages_the_host_inputs_its_program_declares() {
        for count in [SMALL_COUNT, LARGE_COUNT] {
            let size = prepare_size(count, 1024).expect("prepared reduction size");
            for route in &size.routes {
                let declared = route
                    .program
                    .buffers()
                    .iter()
                    .filter(|buffer| buffer.consumes_host_input())
                    .count();
                assert_eq!(
                    route.inputs.len(),
                    declared,
                    "route `{}` at count {count} stages {} host input buffer(s) against an ABI declaring {declared}",
                    route.name,
                    route.inputs.len()
                );
            }
        }
    }

    /// The two routes differ in host input arity, which is why one bundle
    /// cannot serve both.
    ///
    /// Without this, a change that made both routes stage the host output would
    /// leave the test above green while erasing the distinction it exists for.
    #[test]
    fn the_fused_tree_route_declares_fewer_host_inputs_than_the_atomic_route() {
        let size = prepare_size(LARGE_COUNT, 1024).expect("prepared reduction size");
        assert!(
            size.tree().inputs.len() < size.atomic().inputs.len(),
            "the fused tree route's output is backend-allocated, so it stages fewer host inputs than the atomic route: tree={}, atomic={}",
            size.tree().inputs.len(),
            size.atomic().inputs.len()
        );
    }

    /// Only the accumulator is re-seeded between dispatches of resident inputs.
    ///
    /// WHY: resident inputs survive a dispatch, and the atomic route adds into
    /// `out` rather than storing to it, so a second dispatch over the same
    /// resident resources reports the running total of every earlier sample
    /// unless its seed is restored. The tree route stores its result, so
    /// re-seeding it would move bytes for nothing. Both facts come from the
    /// program, so a route added to the roster is planned without this test
    /// being edited. What this does not catch: whether the device honours the
    /// seed, which the per-sample exact-output check covers.
    #[test]
    fn only_a_caller_seeded_accumulator_is_reuploaded_between_dispatches() {
        for count in [SMALL_COUNT, LARGE_COUNT] {
            let size = prepare_size(count, 1024).expect("prepared reduction size");
            assert_eq!(
                size.atomic().reseed,
                vec![(1usize, vec![0u8, 0, 0, 0])],
                "the atomic route accumulates into its second binding, so that binding is re-seeded with four zero bytes at count {count}"
            );
            assert!(
                size.tree().reseed.is_empty(),
                "the fused tree route stores its result, so it re-seeds nothing at count {count}"
            );
        }
    }
}
