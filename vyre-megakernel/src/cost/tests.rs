use super::*;
use crate::facts::DataflowEdge;
use vyre_foundation::algebraic_reordering::ReorderingClass;
use vyre_foundation::validate::BackendCapabilities;

fn device(registers_per_invocation: u32) -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_occupancy(registers_per_invocation, 0)
        .with_bandwidth_facts(TRAFFIC_BYTES_PER_NS, TRAFFIC_BYTES_PER_NS)
}

/// A device with no register budget and the given workgroup scratch budget.
fn scratch_device(shared_scratch_bytes_per_workgroup: u32) -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_occupancy(0, shared_scratch_bytes_per_workgroup)
        .with_bandwidth_facts(TRAFFIC_BYTES_PER_NS, TRAFFIC_BYTES_PER_NS)
}

/// Two nodes, one value of `value_bytes` between them, each holding
/// `live_values` live values and touching that value.
fn two_node_facts(live_values: u64, value_bytes: u64) -> PlanningFacts {
    PlanningFacts {
        node_work: vec![1, 1],
        node_live_values: vec![live_values, live_values],
        node_workgroup_scratch: vec![Vec::new(), Vec::new()],
        node_declared_invocations: vec![256, 256],
        node_declared_workgroup: vec![[256, 1, 1], [256, 1, 1]],
        node_accepts_width: vec![true, true],
        node_reordering: vec![ReorderingClass::NoCombine; 2],
        node_instructions: vec![0, 0],
        node_barriers: vec![0, 0],
        node_grid_syncs: vec![0, 0],
        node_tensor_ops: vec![0, 0],
        node_divergent_regions: vec![0, 0],
        node_touched_bytes: vec![value_bytes, value_bytes],
        dataflow: vec![DataflowEdge {
            from: crate::ArtifactNodeId(0),
            to: crate::ArtifactNodeId(1),
            value: crate::ArtifactValueId(0),
        }],
        value_bytes: [(0_u32, value_bytes)].into_iter().collect(),
    }
}

fn dependencies() -> Vec<DependencyEdge> {
    vec![DependencyEdge {
        from: DependencyEndpoint::Node(crate::ArtifactNodeId(0)),
        to: DependencyEndpoint::Node(crate::ArtifactNodeId(1)),
        kind: DependencyKind::Data,
        value: Some(crate::ArtifactValueId(0)),
    }]
}

/// WHY: 150.13. Fusion is not free. When the fused group needs more registers
/// than one invocation holds, the device runs the group's traffic in more than
/// one resident pass, and past that cliff the unfused pair is cheaper even
/// though it launches twice and materializes the value between the launches.
/// A cost model with only launch and traffic terms ranks the fused candidate
/// first for every graph, which is what this test would have caught.
///
/// The extra pass has to move more than one launch buys. At 3788 bytes per
/// nanosecond a launch at the recorded floor pays for 16 MB, so the value is
/// 64 MiB: the fused group moves its 128 MiB of traffic a second time, and
/// the unfused pair materializes 64 MiB and launches once more.
#[test]
fn occupancy_cliff_ranks_the_unfused_candidate_first() {
    let facts = two_node_facts(96, 64 * 1024 * 1024);
    let dependencies = dependencies();
    let device = device(128);
    let fused = CandidatePlan::from_edges(2, &facts.dataflow);
    let unfused = CandidatePlan::baseline(2);
    assert_eq!(fused.group_count(), 1, "fixture must fuse both nodes");
    assert_eq!(unfused.group_count(), 2);

    let fused_cost = evaluate(&fused, &facts, &dependencies, device);
    let unfused_cost = evaluate(&unfused, &facts, &dependencies, device);
    assert_eq!(
        fused_cost.live_value_peak, 192,
        "the fused group holds both members' live values"
    );
    assert_eq!(
        fused_cost.occupancy_passes_peak, 2,
        "192 live values exceed a 128-register invocation"
    );
    assert_eq!(unfused_cost.occupancy_passes_peak, 1);
    assert!(
        unfused_cost.total < fused_cost.total,
        "past the cliff the unfused candidate must rank first: unfused {unfused_cost:?} fused {fused_cost:?}"
    );
}

/// WHY: 150.13 boundary. Exactly at the register budget there is no cliff, so
/// fusion keeps its launch and traffic saving. This is the adversarial side of
/// the test above: an off-by-one in `resident_passes` would charge a group that
/// fits, and would rank every fusion below its unfused pair.
#[test]
fn fusion_wins_when_the_group_fits_the_register_budget() {
    let facts = two_node_facts(64, 4 * 1024 * 1024);
    let dependencies = dependencies();
    let device = device(128);
    let fused = evaluate(
        &CandidatePlan::from_edges(2, &facts.dataflow),
        &facts,
        &dependencies,
        device,
    );
    let unfused = evaluate(&CandidatePlan::baseline(2), &facts, &dependencies, device);
    assert_eq!(
        fused.occupancy_passes_peak, 1,
        "128 live values fit a 128-register invocation"
    );
    assert_eq!(fused.occupancy_ns, 0);
    assert!(
        fused.total < unfused.total,
        "fused {fused:?} unfused {unfused:?}"
    );
}

/// WHY: the launch term decides how much occupancy a fusion is allowed to
/// cost, and what a launch costs is a property of the host. Pricing every
/// launch at the recorded floor made that trade the same everywhere, so a
/// host with cheap launches still fused past the point where the extra
/// resident pass costs more than the launch it saves.
///
/// One set of facts, two devices differing only in measured launch
/// overhead, opposite winners. `evaluate` and the floor are crate-private,
/// and the flip is a property of the arithmetic rather than of any one
/// graph, so it is pinned here and the published `launch_ns` field is
/// pinned from `tests/selection_cost_contract.rs`.
#[test]
fn a_measured_launch_overhead_reranks_a_fusion_the_floor_accepts() {
    let facts = two_node_facts(96, 4 * 1024 * 1024);
    let dependencies = dependencies();
    let fused_plan = CandidatePlan::from_edges(2, &facts.dataflow);
    let unfused_plan = CandidatePlan::baseline(2);

    let unmeasured = device(128);
    let fused = evaluate(&fused_plan, &facts, &dependencies, unmeasured);
    let unfused = evaluate(&unfused_plan, &facts, &dependencies, unmeasured);
    assert_eq!(fused.launch_ns, LAUNCH_COST_FLOOR_NS);
    assert_eq!(unfused.launch_ns, 2 * LAUNCH_COST_FLOOR_NS);
    assert!(
        fused.total < unfused.total,
        "at the floor the saved launch outweighs the extra resident pass: \
         fused {fused:?} unfused {unfused:?}"
    );

    let cheap_launches = device(128).with_launch_costs(512, 0);
    let fused = evaluate(&fused_plan, &facts, &dependencies, cheap_launches);
    let unfused = evaluate(&unfused_plan, &facts, &dependencies, cheap_launches);
    assert_eq!(
        fused.launch_ns, 512,
        "the measured figure prices the launch"
    );
    assert_eq!(unfused.launch_ns, 1024);
    assert!(
        unfused.total < fused.total,
        "a launch worth 512 ns does not pay for a second pass over 8 MiB: \
         fused {fused:?} unfused {unfused:?}"
    );
}

/// WHY: an unknown budget is not a budget of zero. A backend that reports no
/// register count must not make every group look infinitely over budget.
#[test]
fn unknown_occupancy_budget_charges_nothing() {
    let facts = two_node_facts(1_000_000, 4 * 1024 * 1024);
    let dependencies = dependencies();
    let cost = evaluate(
        &CandidatePlan::from_edges(2, &facts.dataflow),
        &facts,
        &dependencies,
        device(0),
    );
    assert_eq!(cost.occupancy_passes_peak, 1);
    assert_eq!(cost.occupancy_ns, 0);
}

/// Facts for two nodes whose scratch declarations are given, sharing one
/// value between them.
fn tiled_facts(first: &[(&str, u64)], second: &[(&str, u64)]) -> PlanningFacts {
    let mut facts = two_node_facts(1, 4 * 1024 * 1024);
    facts.node_workgroup_scratch = [first, second]
        .into_iter()
        .map(|declarations| {
            declarations
                .iter()
                .map(|(name, bytes)| (Ident::from(*name), *bytes))
                .collect()
        })
        .collect();
    facts
}

/// WHY: fusion unions buffers by name, so two ops fused over one tile hold
/// one tile. Charging the sum made a group that fits look like it needs two
/// resident passes, which ranked the tile-sharing fusion below the pair it
/// beats. That is the fused attention shape: the score tile is written by
/// one op and read by the next and never reaches memory.
#[test]
fn a_tile_two_members_share_is_charged_once() {
    let facts = tiled_facts(&[("tile", 32 * 1024)], &[("tile", 32 * 1024)]);
    let dependencies = dependencies();
    let device = scratch_device(48 * 1024);
    let fused = evaluate(
        &CandidatePlan::from_edges(2, &facts.dataflow),
        &facts,
        &dependencies,
        device,
    );
    assert_eq!(
        fused.shared_scratch_bytes,
        32 * 1024,
        "one name is one tile in the generated kernel"
    );
    assert_eq!(
        fused.occupancy_passes_peak, 1,
        "32 KiB fits a 48 KiB workgroup budget"
    );
    assert_eq!(fused.occupancy_ns, 0);
    let unfused = evaluate(&CandidatePlan::baseline(2), &facts, &dependencies, device);
    assert!(
        fused.total < unfused.total,
        "a fusion that shares its tile keeps its launch and traffic saving: \
         fused {fused:?} unfused {unfused:?}"
    );
}

/// WHY: the union is by name, not a blanket discount. Two members that
/// declare different tiles need both at once, and a group that stops
/// charging for them would rank a fusion the device cannot hold first.
#[test]
fn two_tiles_of_different_names_are_charged_together() {
    let facts = tiled_facts(&[("score", 32 * 1024)], &[("weights", 32 * 1024)]);
    let dependencies = dependencies();
    let device = scratch_device(48 * 1024);
    let fused = evaluate(
        &CandidatePlan::from_edges(2, &facts.dataflow),
        &facts,
        &dependencies,
        device,
    );
    assert_eq!(fused.shared_scratch_bytes, 64 * 1024);
    assert_eq!(
        fused.occupancy_passes_peak, 2,
        "64 KiB of distinct tiles exceeds a 48 KiB workgroup budget"
    );
    assert!(fused.occupancy_ns > 0);
}

/// WHY: fusion takes the larger count when two arms name one buffer, so the
/// group holds the larger tile and not the first one seen.
#[test]
fn a_shared_name_of_two_sizes_holds_the_larger() {
    let facts = tiled_facts(&[("tile", 16 * 1024)], &[("tile", 32 * 1024)]);
    let cost = evaluate(
        &CandidatePlan::from_edges(2, &facts.dataflow),
        &facts,
        &dependencies(),
        device(0),
    );
    assert_eq!(cost.shared_scratch_bytes, 32 * 1024);
}

/// Every `p50` of one metric across every recorded bench file in the tree,
/// paired with the file and the case id it came from.
///
/// The file is part of the answer because one case id appears in more than
/// one recording, with a different figure in each. Reading a metric by id
/// alone then answers from whichever file `read_dir` happened to yield
/// first, which is a filesystem fact: the same tree read the traffic rate
/// as 3623 on one host and 3788 on another, and the constant can only equal
/// one of them. The file list is sorted so a walk is a function of the
/// tree, and a metric is read out of the recording that named the case.
fn recorded(metric: &str) -> Vec<(std::path::PathBuf, String, u64)> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "json") {
                out.push(path);
            }
        }
    }

    let bench = vyre_test_support::monorepo::vyre_crate_directory("vyre-bench");
    let mut files = Vec::new();
    walk(&bench.join("snapshots"), &mut files);
    walk(&bench.join("baselines"), &mut files);
    assert!(!files.is_empty(), "no recorded bench file was found");
    files.sort();

    let mut found = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file).expect("a recorded bench file must read");
        let value: serde_json::Value =
            serde_json::from_str(&text).expect("a recorded bench file must parse");
        let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
            continue;
        };
        for case in cases {
            let Some(p50) = case
                .pointer(&format!("/metrics/{metric}/p50"))
                .and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            let id = case
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            found.push((file.clone(), id.to_string(), p50));
        }
    }
    found
}

/// WHY: both weights are durations read off a recording, and a constant that
/// drifts from the recording it cites prices every fusion decision against a
/// device nothing measured. This derives both from the files at run time, so a
/// cheaper recorded dispatch or a re-measured rate turns the suite red instead
/// of leaving a stale weight behind a doc comment. It proves nothing about
/// whether the model ranks a real plan correctly.
#[test]
fn the_cost_weights_are_the_figures_the_recordings_hold() {
    let dispatches = recorded("dispatch_ns");
    let (floor_file, floor_case, floor) = dispatches
        .iter()
        .min_by_key(|(file, id, p50)| (*p50, file.as_path(), id.as_str()))
        .expect("a recorded dispatch must exist");
    assert_eq!(
        LAUNCH_COST_FLOOR_NS, *floor,
        "the launch floor must be the cheapest recorded dispatch, now held by `{floor_case}`"
    );

    let rates = recorded("device_gb_s_x1000");
    let (_, _, rate) = rates
        .iter()
        .find(|(file, id, _)| file == floor_file && id == floor_case)
        .unwrap_or_else(|| {
            panic!(
                "the recording that fixes the floor must state its device rate: `{floor_case}` in {}",
                floor_file.display()
            )
        });
    assert_eq!(
        TRAFFIC_BYTES_PER_NS,
        rate.saturating_add(500).div_euclid(1_000),
        "the traffic rate must be the rate that case recorded, to the nearest byte"
    );
}

#[test]
fn unknown_device_evidence_omits_traffic_cost_terms() {
    let facts = two_node_facts(64, 4 * 1024 * 1024);
    let dependencies = dependencies();
    let unmeasured_device =
        DeviceFacts::new(BackendCapabilities::default(), 256).with_occupancy(128, 0);
    assert_eq!(
        unmeasured_device.calibrated_materialization_throughput_bytes_per_ns(),
        0
    );
    let unfused = evaluate(
        &CandidatePlan::baseline(2),
        &facts,
        &dependencies,
        unmeasured_device,
    );
    assert_eq!(
        unfused.materialization_ns, 0,
        "unmeasured bandwidth omits materialization term"
    );
    assert_eq!(unfused.occupancy_ns, 0);
}

#[test]
fn sub_rate_nonzero_bytes_round_up_to_one_nanosecond() {
    let facts = two_node_facts(64, 50);
    let dependencies = dependencies();
    let fast_bandwidth_device = DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_occupancy(128, 0)
        .with_bandwidth_facts(1000, 1000);
    let unfused = evaluate(
        &CandidatePlan::baseline(2),
        &facts,
        &dependencies,
        fast_bandwidth_device,
    );
    assert_eq!(
        unfused.materialization_ns, 1,
        "50 bytes with 1000 B/ns throughput must round up to 1 ns via ceiling division"
    );
}

#[test]
fn launch_and_bandwidth_ratios_favor_different_candidates_across_devices() {
    // Two nodes with 16MB dataflow.
    let facts = two_node_facts(64, 16 * 1024 * 1024);
    let dependencies = dependencies();
    let fused_plan = CandidatePlan::from_edges(2, &facts.dataflow);
    let unfused_plan = CandidatePlan::baseline(2);

    // Device A: High launch cost (50,000 ns), high bandwidth (10,000 B/ns).
    // Unfused pays 1 extra launch (50,000 ns) + 16MB traffic (1,678 ns) = total 101,678 ns.
    // Fused pays 1 launch (50,000 ns) + 0 traffic = total 50,000 ns. Fused wins!
    let device_a = DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_launch_costs(50_000, 0)
        .with_bandwidth_facts(10_000, 10_000)
        .with_occupancy(128, 0);
    let fused_a = evaluate(&fused_plan, &facts, &dependencies, device_a);
    let unfused_a = evaluate(&unfused_plan, &facts, &dependencies, device_a);
    assert!(
        fused_a.total < unfused_a.total,
        "fused must win on device with high launch cost"
    );

    // Device B: Low launch cost (100 ns), very low bandwidth (10 B/ns).
    // If fused group incurs extra occupancy pass on 128 live values:
    let occupancy_facts = two_node_facts(96, 16 * 1024 * 1024);
    let fused_occ = CandidatePlan::from_edges(2, &occupancy_facts.dataflow);
    let unfused_occ = CandidatePlan::baseline(2);
    let device_b = DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_launch_costs(100, 0)
        .with_bandwidth_facts(100, 100)
        .with_occupancy(128, 0);
    let fused_b = evaluate(&fused_occ, &occupancy_facts, &dependencies, device_b);
    let unfused_b = evaluate(&unfused_occ, &occupancy_facts, &dependencies, device_b);
    assert!(
        unfused_b.total < fused_b.total,
        "unfused must win when occupancy cliff on slow memory exceeds low launch cost"
    );
}

/// Facts for two nodes with the given per-node program counters.
fn counted_facts(
    instructions: u64,
    tensor_ops: u64,
    barriers: u64,
    grid_syncs: u64,
    divergent_regions: u64,
) -> PlanningFacts {
    let mut facts = two_node_facts(1, 4 * 1024 * 1024);
    facts.node_instructions = vec![instructions, instructions];
    facts.node_tensor_ops = vec![tensor_ops, tensor_ops];
    facts.node_barriers = vec![barriers, barriers];
    facts.node_grid_syncs = vec![grid_syncs, grid_syncs];
    facts.node_divergent_regions = vec![divergent_regions, divergent_regions];
    facts
}

/// WHY: a rate nothing measured is not a rate. Every term switched on by a
/// device rate must be omitted on a device that reports none, or the model
/// orders candidates on arithmetic no recording and no probe supports.
#[test]
fn a_device_that_reports_no_rate_is_charged_no_rate_term() {
    let facts = counted_facts(1_000, 40, 8, 2, 4);
    let cost = evaluate(
        &CandidatePlan::baseline(2),
        &facts,
        &dependencies(),
        device(0),
    );
    assert_eq!(cost.instructions, 2_000, "the count is recorded either way");
    assert_eq!(cost.tensor_ops, 80);
    assert_eq!(cost.barriers, 16);
    assert_eq!(cost.grid_syncs, 4);
    assert_eq!(cost.divergent_regions, 8);
    assert_eq!(cost.instruction_ns, 0);
    assert_eq!(cost.tensor_ns, 0);
    assert_eq!(cost.synchronization_ns, 0);
    assert_eq!(cost.divergence_ns, 0);
}

/// WHY: the instruction and matrix-engine terms are the device's own rates
/// applied to the counts the programs state, and each must be charged at its
/// own rate. Pricing tile statements at the scalar rate misprices every
/// candidate that fuses one.
#[test]
fn stated_work_is_charged_at_the_rate_the_device_reports() {
    let facts = counted_facts(1_000, 40, 0, 0, 0);
    let device = device(0).with_compute_throughput(100, 4);
    let cost = evaluate(&CandidatePlan::baseline(2), &facts, &dependencies(), device);
    assert_eq!(cost.instruction_ns, 20, "2000 instructions at 100 per ns");
    assert_eq!(cost.tensor_ns, 20, "80 tile statements at 4 per ns");
    assert!(cost.total >= 40);
}

/// WHY: a rendezvous costs device time no branch costs. A workgroup barrier the
/// program states is charged, and so is the grid rendezvous the program states.
#[test]
fn stated_rendezvous_is_charged_at_its_own_cost() {
    let facts = counted_facts(0, 0, 8, 2, 0);
    let device = device(0).with_synchronization_costs(50, 5_000);
    let cost = evaluate(&CandidatePlan::baseline(2), &facts, &dependencies(), device);
    assert_eq!(
        cost.synchronization_ns,
        16 * 50 + 4 * 5_000,
        "sixteen barriers and four grid rendezvous"
    );
}

/// WHY: a resident partition orders its stages inside one launch, so it pays a
/// whole-grid rendezvous where a sequential plan pays a dispatch boundary.
/// Pricing persistence as pure launch savings charged nothing for the
/// rendezvous it pays instead, so a resident plan outranked a sequential one on
/// a device where the rendezvous costs more than the launch it removes.
#[test]
fn a_resident_partition_pays_for_the_rendezvous_it_orders_stages_with() {
    let facts = counted_facts(0, 0, 0, 0, 0);
    let dependencies = dependencies();
    let device = device(0).with_synchronization_costs(0, 100_000);
    let sequential = CandidatePlan::baseline(2);
    let mut resident = CandidatePlan::baseline(2);
    resident.topology = ExecutionTopology::ResidentPartition {
        partitions: 2,
        mode: crate::ResidentPartitionMode::FixedSpatialMask,
    };
    let sequential_cost = evaluate(&sequential, &facts, &dependencies, device);
    let resident_cost = evaluate(&resident, &facts, &dependencies, device);
    assert_eq!(sequential_cost.grid_syncs, 0);
    assert_eq!(
        resident_cost.grid_syncs, 1,
        "two dependent stages inside one launch rendezvous once"
    );
    assert_eq!(resident_cost.synchronization_ns, 100_000);
    assert!(
        sequential_cost.total < resident_cost.total,
        "a rendezvous worth 100 us does not pay for one saved launch: \
         sequential {sequential_cost:?} resident {resident_cost:?}"
    );
}

/// WHY: a lane-gated region leaves the rest of the subgroup idle for its
/// duration. The count alone cannot say how long, so the term is a lower bound
/// of one idle instruction per lane, and it is charged only where the device
/// reports both a width and a rate.
#[test]
fn a_lane_gated_region_is_charged_for_the_lanes_it_idles() {
    let facts = counted_facts(0, 0, 0, 0, 16);
    let dependencies = dependencies();
    let priced = device(0)
        .with_compute_throughput(31, 0)
        .with_subgroup_size(32);
    let cost = evaluate(&CandidatePlan::baseline(2), &facts, &dependencies, priced);
    assert_eq!(
        cost.divergence_ns, 32,
        "32 gated regions idle 31 lanes each at 31 instructions per ns"
    );
    let no_width = device(0)
        .with_compute_throughput(31, 0)
        .with_subgroup_size(0);
    let cost = evaluate(&CandidatePlan::baseline(2), &facts, &dependencies, no_width);
    assert_eq!(
        cost.divergence_ns, 0,
        "a device that reports no subgroup width states no idle lanes"
    );
}

/// WHY: a repeated pass over a working set the device-wide cache holds does not
/// return to memory. Charging it as memory traffic priced the occupancy cliff at
/// full bandwidth cost on a device whose cache holds the whole group, which
/// ranked a fusion below a pair it beats.
#[test]
fn a_replayed_pass_the_cache_holds_is_not_charged_as_traffic() {
    let facts = two_node_facts(96, 4 * 1024 * 1024);
    let dependencies = dependencies();
    let uncached = device(128);
    let cached = device(128).with_cache_capacity(64 * 1024 * 1024);
    let fused = CandidatePlan::from_edges(2, &facts.dataflow);
    let uncached_cost = evaluate(&fused, &facts, &dependencies, uncached);
    let cached_cost = evaluate(&fused, &facts, &dependencies, cached);
    assert_eq!(uncached_cost.occupancy_passes_peak, 2);
    assert_eq!(
        cached_cost.occupancy_passes_peak, 2,
        "the cliff is unchanged"
    );
    assert_eq!(uncached_cost.cache_resident_bytes, 0);
    assert_eq!(
        cached_cost.cache_resident_bytes,
        8 * 1024 * 1024,
        "both members' touched bytes are replayed once and the cache holds them"
    );
    assert!(uncached_cost.occupancy_ns > 0);
    assert_eq!(
        cached_cost.occupancy_ns, 0,
        "a replay the cache serves costs no memory time"
    );
}

/// WHY: a group above the full-occupancy register budget spills, which is legal
/// and priced. Recording how far above it is separates that from the
/// architectural ceiling, which has no execution at all and is rejected rather
/// than priced.
#[test]
fn registers_above_the_occupancy_budget_are_recorded_as_spilled() {
    let facts = two_node_facts(96, 4 * 1024 * 1024);
    let dependencies = dependencies();
    let fused = CandidatePlan::from_edges(2, &facts.dataflow);
    let spilling = evaluate(&fused, &facts, &dependencies, device(128));
    assert_eq!(
        spilling.spill_registers_peak, 64,
        "192 live values against a 128-register budget spill 64"
    );
    let resident = evaluate(&fused, &facts, &dependencies, device(256));
    assert_eq!(resident.spill_registers_peak, 0);
    let unknown = evaluate(&fused, &facts, &dependencies, device(0));
    assert_eq!(
        unknown.spill_registers_peak, 0,
        "an unknown budget states nothing above it"
    );
}
