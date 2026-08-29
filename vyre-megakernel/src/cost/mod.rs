//! Open, reproducible whole-program candidate cost model.
//!
//! The unit is nanoseconds of expected device time, so every weight below is a
//! duration read off a recorded `vyre-bench` result rather than a preference.
//! The recorded files are `vyre-bench/snapshots/*.json` and
//! `vyre-bench/baselines/rtx_5090/*.json`; each constant names the case and the
//! metric that fixed it.
//!
//! Three things a fusion decision changes are priced: the launches it removes,
//! the bytes it stops writing between kernels, and the occupancy it costs when
//! the fused group needs more registers or more shared scratch than the device
//! keeps resident. The third is why fusion is not always cheaper, and it is a
//! cliff rather than a slope: a group that needs twice the resident budget runs
//! its traffic in two passes instead of one.
//!
//! What a launch costs is read off the device the request targets, not off the
//! recorded floor, because the trade the first two terms make is launches
//! against bytes and the answer differs by host. The floor prices a device that
//! measured nothing.
//!
//! `the_cost_weights_are_the_figures_the_recordings_hold` reads those files at
//! run time and derives both weights from them, so a weight that drifts from
//! its citation, or a cheaper dispatch landing in a new recording, is a red
//! test rather than a stale doc comment.
//!
//! A group's shared scratch is the union of its members' declarations, not the
//! sum of their totals. Fusion keeps one declaration per buffer name and takes
//! the larger count, so two ops fused over one tile hold one tile in the
//! generated kernel. Summing per-member totals charged that tile once per
//! member, which pushed the group over the device budget and ranked the
//! tile-sharing fusion below the pair it beats. That is the shape a fused
//! attention group has: the score is written to a tile and read from the same
//! tile by the value matmul, and it never reaches memory.
//!
//! The launch width the search proposes is not priced. No recorded case varies
//! width against a fixed program, so a width term would be a guess, and a guess
//! that orders the widths is worse than no term: the analytic ranking would claim
//! a result only a measurement has. Width candidates therefore tie on cost, are
//! ordered deterministically by `crate::select::rank`, and a measured
//! compilation decides between them on device time.
//!
//! ## Terms a device fact switches on
//!
//! Instruction throughput, matrix-engine throughput, barrier cost and grid
//! rendezvous cost are device facts, not recorded constants. No recording in
//! this tree fixes any of the four: the fastest recorded compute rate belongs to
//! one program's dynamic instruction stream, and this model counts the
//! instructions a program states, which is a different quantity. A rate read off
//! one and applied to the other would order candidates on arithmetic nothing
//! measured. So the count is recorded either way, the term is charged only when
//! the backend reports the matching rate, and a device that reports none is
//! ranked on launches and traffic exactly as before.
//!
//! The register budget is the registers an invocation holds while the device
//! stays fully occupied, never the architectural ceiling. A device that admits
//! 255 registers per invocation runs one group per compute unit at that
//! allocation, so ranking against the ceiling reported every candidate as
//! resident and the occupancy term never fired on the one backend that measures
//! a register count.
//!
//! A repeated pass over a working set the device-wide cache holds does not
//! return to memory, so the occupancy term charges only the bytes that exceed
//! the reported cache capacity. A device that reports no cache is charged for
//! all of them, which is the previous behaviour.
//!
//! ## What the target compiler reports outranks what this model estimated
//!
//! Every register and shared-byte figure above is an estimate read off the IR.
//! Once a candidate has been emitted and loaded, the target compiler and the
//! device state what the entry point actually allocates, which is the same
//! quantity measured instead of predicted. Reported-figure evaluation re-prices
//! a candidate with those figures in place of the estimate, so the ladder ranks
//! its finalists on the register allocation the device will run rather than the
//! one the IR suggested. A term the backend does not report stays zero and the
//! estimate stands for it.

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::Ident;

use crate::{
    candidate::{CandidatePlan, ExecutionTopology},
    dependency_order::group_stages,
    facts::PlanningFacts,
    DependencyEdge, DependencyEndpoint, DependencyKind, DeviceFacts, FusionGroupId,
};

/// Cost of one kernel launch on a device that reports no measured overhead.
///
/// `foundation.elementwise.add.1m` records `dispatch_ns` p50 4224
/// (`vyre-bench/snapshots/59a7d71f36292424c99b7530da59f7361bfab607.json`). It is
/// the cheapest whole dispatch in any recorded snapshot or baseline, so no
/// launch costs less than this and a device that measured nothing is priced at
/// that floor. A device that did measure is priced at its own figure by
/// [`launch_cost_ns`].
const LAUNCH_COST_FLOOR_NS: u64 = 4_224;

/// Bytes of traffic one nanosecond moves.
///
/// The same dispatch moves 12 MB in and 4 MB out (`bytes_read` 12000000,
/// `bytes_written` 4000000) inside those 4224 ns, and the snapshot records the
/// resulting rate directly as `device_gb_s_x1000` 3787878, which is 3788 bytes
/// per nanosecond. The same figure prices both the bytes a materialization
/// writes and the bytes an occupancy loss moves a second time.
///
/// The rate is the recorded one and carries no safety factor. A factor that
/// nothing measured would reprice traffic against launches, and the ratio
/// between those two terms is the whole question a fusion decision asks.
const TRAFFIC_BYTES_PER_NS: u64 = 3_788;

mod provenance;

pub use provenance::{CostTerm, CostTermRole, CostUnit, TERMS};

/// Reproducible components of the open compiler selection cost model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Sum of semantic IR nodes in the complete graph.
    ///
    /// Recorded as evidence and excluded from [`Self::total`]: it is the same for
    /// every candidate over one graph, and no recorded snapshot separates
    /// per-IR-node time from memory time, so pricing it would be a guess that
    /// changes no ranking.
    pub semantic_work: u64,
    /// Number of generated kernel launches.
    pub launches: u64,
    /// Number of values crossing generated-kernel boundaries.
    pub materializations: u64,
    /// Bytes those crossing values move.
    pub materialized_bytes: u64,
    /// Largest per-invocation live value count in any one fusion group.
    pub live_value_peak: u64,
    /// Largest shared scratch byte count any one fusion group declares, with
    /// buffers of one name counted once because fusion unions them.
    pub shared_scratch_bytes: u64,
    /// Largest number of resident passes any one group needs, one meaning the
    /// group fits the device budgets.
    pub occupancy_passes_peak: u64,
    /// Bytes the allocation plan holds at once under this grouping.
    ///
    /// The same liveness the placement plan is packed against, so the peak the
    /// objective ranks and the peak the artifact records are one number.
    pub planned_peak_bytes: u64,
    /// Instructions the selected programs state.
    pub instructions: u64,
    /// Matrix-engine statements the selected programs state.
    pub tensor_ops: u64,
    /// Workgroup-scoped rendezvous the selected programs state.
    pub barriers: u64,
    /// Whole-grid rendezvous the selected programs state.
    pub grid_syncs: u64,
    /// Lane-gated regions the selected programs state.
    pub divergent_regions: u64,
    /// Registers the worst group holds above the full-occupancy budget.
    ///
    /// Above this the target compiler spills, which is legal and priced through
    /// [`Self::occupancy_ns`]. Above the architectural ceiling there is no
    /// execution at all, and `crate::constraints` rejects the candidate rather
    /// than pricing it.
    pub spill_registers_peak: u64,
    /// Repeated-pass bytes the device-wide cache serves, excluded from
    /// [`Self::occupancy_ns`].
    pub cache_resident_bytes: u64,
    /// Local-memory spill bytes the target compiler reported, multiplied by the
    /// invocations the selected geometry launches.
    ///
    /// Zero until a candidate has been emitted and the backend reported a spill.
    /// Charged through [`Self::occupancy_ns`] at the same bandwidth fact as
    /// every other byte that reaches memory twice.
    pub reported_spill_bytes: u64,
    /// Launch term in nanoseconds.
    pub launch_ns: u64,
    /// Materialized-traffic term in nanoseconds.
    pub materialization_ns: u64,
    /// Occupancy term in nanoseconds.
    pub occupancy_ns: u64,
    /// Instruction term in nanoseconds, zero when the device reports no rate.
    pub instruction_ns: u64,
    /// Matrix-engine term in nanoseconds, zero when the device reports no rate.
    pub tensor_ns: u64,
    /// Rendezvous term in nanoseconds, zero when the device reports no cost.
    pub synchronization_ns: u64,
    /// Idle-lane term in nanoseconds, zero when the device reports no
    /// instruction rate or no subgroup width.
    pub divergence_ns: u64,
    /// Weighted total in nanoseconds, minimized by candidate selection.
    pub total: u64,
}

/// What one emitted entry point allocates, as the target compiler and the device
/// reported it, converted into the quantities this model prices.
///
/// One entry per fusion group, in group order, which is the order
/// `compile_selected_modules` emits them in. Zero means the backend reported
/// nothing for that term, and the analytic estimate stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReportedGroup {
    /// Registers the entry point allocates per invocation.
    pub(crate) registers_per_invocation: u32,
    /// Statically declared workgroup-scoped bytes.
    pub(crate) shared_memory_bytes: u32,
    /// Local-memory spill bytes across every invocation the entry launches.
    pub(crate) spill_traffic_bytes: u64,
}

pub(crate) fn evaluate(
    candidate: &CandidatePlan,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    device: DeviceFacts,
) -> CostBreakdown {
    evaluate_reported(candidate, facts, dependencies, device, &[])
}

/// Price one candidate with the figures its emitted entry points reported in
/// place of the estimates this model derived from the IR.
pub(crate) fn evaluate_reported(
    candidate: &CandidatePlan,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    device: DeviceFacts,
    reported: &[ReportedGroup],
) -> CostBreakdown {
    let semantic_work = facts
        .node_work
        .iter()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let group_count = candidate.group_count();
    let node_groups: Vec<FusionGroupId> = candidate
        .node_groups
        .iter()
        .copied()
        .map(FusionGroupId)
        .collect();
    let stages = group_stages(group_count, dependencies, &node_groups)
        .unwrap_or_else(|_| vec![0; group_count]);
    let stage_count = stages.iter().copied().max().map_or(0, |s| s as usize + 1);
    let mut stage_groups = vec![Vec::new(); stage_count];
    for (group, &stage) in stages.iter().enumerate() {
        stage_groups[stage as usize].push(group as u32);
    }

    let launches = if candidate.frontier_topology == crate::candidate::FrontierTopology::FusedWave {
        u64::try_from(stage_groups.len()).unwrap_or(1).max(1)
    } else {
        match candidate.topology {
            ExecutionTopology::Sequential => u64::try_from(group_count).unwrap_or(u64::MAX),
            ExecutionTopology::ConcurrentQueue { queues } => {
                let q = u64::from(queues.max(1));
                let mut total_launches = 0_u64;
                for groups in &stage_groups {
                    let count = u64::try_from(groups.len()).unwrap_or(1);
                    total_launches = total_launches.saturating_add(count.div_ceil(q));
                }
                total_launches.max(1)
            }
            ExecutionTopology::ResidentPartition { partitions, .. } => {
                let p = u64::from(partitions.max(1));
                let mut total_launches = 0_u64;
                for groups in &stage_groups {
                    let count = u64::try_from(groups.len()).unwrap_or(1);
                    total_launches = total_launches.saturating_add(count.div_ceil(p));
                }
                total_launches.max(1)
            }
        }
    };
    let crossing = dependencies.iter().filter(|edge| {
        if edge.kind != DependencyKind::Data {
            return false;
        }
        let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        else {
            return false;
        };
        candidate.node_groups.get(from.0 as usize) != candidate.node_groups.get(to.0 as usize)
    });
    let mut materializations = 0_u64;
    let mut materialized_bytes = 0_u64;
    for edge in crossing {
        materializations = materializations.saturating_add(1);
        let bytes = edge
            .value
            .and_then(|value| facts.value_bytes.get(&value.0).copied())
            .unwrap_or(0);
        materialized_bytes = materialized_bytes.saturating_add(bytes);
    }

    let mut live_value_peak = 0_u64;
    let mut shared_scratch_bytes = 0_u64;
    let mut occupancy_passes_peak = 1_u64;
    let mut occupancy_bytes = 0_u64;
    let mut spill_registers_peak = 0_u64;
    let mut cache_resident_bytes = 0_u64;
    let mut reported_spill_bytes = 0_u64;
    let register_budget = u64::from(device.registers_per_invocation());
    let scratch_budget = u64::from(device.shared_scratch_bytes_per_workgroup());
    let cache_capacity = device.cache_capacity_bytes();
    // A sequential or queued topology holds one group resident at a time; a
    // resident partition holds a whole stage. The peak is taken over whichever
    // of the two is co-resident, so both cases are one loop over launch units.
    let units: Vec<Vec<u32>> = match candidate.topology {
        ExecutionTopology::Sequential | ExecutionTopology::ConcurrentQueue { .. } => (0
            ..u32::try_from(group_count).unwrap_or(u32::MAX))
            .map(|group| vec![group])
            .collect(),
        ExecutionTopology::ResidentPartition { .. } => stage_groups.clone(),
    };
    for unit in &units {
        let mut estimated_live = 0_u64;
        let mut tiles: Vec<(&Ident, u64)> = Vec::new();
        let mut unit_bytes = 0_u64;
        let mut reported_registers = 0_u64;
        let mut reported_shared = 0_u64;
        for &group in unit {
            let reported = reported.get(group as usize).copied().unwrap_or_default();
            reported_registers =
                reported_registers.saturating_add(u64::from(reported.registers_per_invocation));
            reported_shared =
                reported_shared.saturating_add(u64::from(reported.shared_memory_bytes));
            reported_spill_bytes =
                reported_spill_bytes.saturating_add(reported.spill_traffic_bytes);
            for node in candidate.group_members(group) {
                estimated_live = estimated_live
                    .saturating_add(facts.node_live_values.get(node).copied().unwrap_or(0));
                for (name, bytes) in facts
                    .node_workgroup_scratch
                    .get(node)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    let held = tiles.iter().position(|(name_held, _)| *name_held == name);
                    match held {
                        Some(index) => tiles[index].1 = tiles[index].1.max(*bytes),
                        None => tiles.push((name, *bytes)),
                    }
                }
                unit_bytes = unit_bytes
                    .saturating_add(facts.node_touched_bytes.get(node).copied().unwrap_or(0));
            }
        }
        let estimated_scratch = tiles
            .iter()
            .fold(0_u64, |total, (_, bytes)| total.saturating_add(*bytes));
        // A reported figure is the allocation the device will run. It replaces
        // the estimate rather than joining it, because two answers to one
        // question are not evidence of a larger demand.
        let live = if reported_registers > 0 {
            reported_registers
        } else {
            estimated_live
        };
        let scratch = if reported_shared > 0 {
            reported_shared
        } else {
            estimated_scratch
        };
        live_value_peak = live_value_peak.max(live);
        shared_scratch_bytes = shared_scratch_bytes.max(scratch);
        let passes =
            resident_passes(live, register_budget).max(resident_passes(scratch, scratch_budget));
        occupancy_passes_peak = occupancy_passes_peak.max(passes);
        spill_registers_peak = spill_registers_peak.max(spilled(live, register_budget));
        let replays = passes.saturating_sub(1);
        let served = unit_bytes.min(cache_capacity).saturating_mul(replays);
        cache_resident_bytes = cache_resident_bytes.saturating_add(served);
        occupancy_bytes = occupancy_bytes
            .saturating_add(unit_bytes.saturating_mul(replays).saturating_sub(served));
    }
    occupancy_bytes = occupancy_bytes.saturating_add(reported_spill_bytes);

    let launch_ns = launches.saturating_mul(launch_cost_ns(device));
    let throughput = match device.calibrated_materialization_throughput_bytes_per_ns() {
        0 => device.peak_bandwidth_bytes_per_ns(),
        calibrated => calibrated,
    };
    let (materialization_ns, occupancy_ns) = if throughput == 0 {
        (0, 0)
    } else {
        (
            if materialized_bytes == 0 {
                0
            } else {
                materialized_bytes.div_ceil(throughput)
            },
            if occupancy_bytes == 0 {
                0
            } else {
                occupancy_bytes.div_ceil(throughput)
            },
        )
    };
    let instructions = sum(&facts.node_instructions);
    let tensor_ops = sum(&facts.node_tensor_ops);
    let barriers = sum(&facts.node_barriers);
    let divergent_regions = sum(&facts.node_divergent_regions);
    // A resident partition keeps its stages inside one launch, so ordering them
    // is a whole-grid rendezvous instead of a dispatch boundary. Pricing
    // persistence as pure launch savings charged nothing for the rendezvous it
    // pays instead.
    let rendezvous = match candidate.topology {
        ExecutionTopology::Sequential | ExecutionTopology::ConcurrentQueue { .. } => 0,
        ExecutionTopology::ResidentPartition { .. } => u64::try_from(stage_count)
            .unwrap_or(u64::MAX)
            .saturating_sub(1),
    };
    let grid_syncs = sum(&facts.node_grid_syncs).saturating_add(rendezvous);
    let planned_peak_bytes = crate::allocation::peak(&facts.value_liveness, &node_groups, &stages);
    let instruction_ns = rate_ns(instructions, device.compute_throughput_ops_per_ns());
    let tensor_ns = rate_ns(tensor_ops, device.tensor_throughput_ops_per_ns());
    let synchronization_ns = barriers
        .saturating_mul(device.barrier_ns())
        .saturating_add(grid_syncs.saturating_mul(device.grid_sync_ns()));
    let idle_lanes =
        divergent_regions.saturating_mul(u64::from(device.subgroup_size()).saturating_sub(1));
    let divergence_ns = rate_ns(idle_lanes, device.compute_throughput_ops_per_ns());
    let total = launch_ns
        .saturating_add(materialization_ns)
        .saturating_add(occupancy_ns)
        .saturating_add(instruction_ns)
        .saturating_add(tensor_ns)
        .saturating_add(synchronization_ns)
        .saturating_add(divergence_ns);
    CostBreakdown {
        semantic_work,
        launches,
        materializations,
        materialized_bytes,
        live_value_peak,
        shared_scratch_bytes,
        occupancy_passes_peak,
        planned_peak_bytes,
        instructions,
        tensor_ops,
        barriers,
        grid_syncs,
        divergent_regions,
        spill_registers_peak,
        cache_resident_bytes,
        reported_spill_bytes,
        launch_ns,
        materialization_ns,
        occupancy_ns,
        instruction_ns,
        tensor_ns,
        synchronization_ns,
        divergence_ns,
        total,
    }
}

/// Nanoseconds one kernel launch costs on `device`.
///
/// `DeviceFacts::per_launch_overhead_ns` is measured on the device the request
/// targets, and `crate::artifact` already trades it against persistent setup to
/// choose an execution mode. Pricing every launch at the recorded floor instead
/// made the launch term a constant multiple of the group count, so a host whose
/// launches cost five times the floor still had its fusions ranked as though
/// they saved the floor. The term now moves with the device, which is what lets
/// a persistent schedule outrank a per-op one on a host where launches are
/// expensive and lose on one where they are not.
///
/// A device that reports zero measured nothing, so the recorded floor stands in.
pub(crate) fn launch_cost_ns(device: DeviceFacts) -> u64 {
    match device.per_launch_overhead_ns() {
        0 => LAUNCH_COST_FLOOR_NS,
        measured => measured,
    }
}

/// Resident passes a group needs when it wants `demand` of a `budget`.
///
/// A workgroup holds a fixed number of registers and a fixed shared-scratch
/// allocation. A group that wants more than one budget's worth does not fail: the
/// target compiler spills, or the device schedules fewer workgroups, and the
/// group's traffic moves once per pass instead of once. The recorded shape of that
/// loss is `foundation.reduce.sum.1m`, which declares a 256-entry u32 workgroup
/// tile and records `dispatch_ns` p50 44448 against 9664 for
/// `foundation.elementwise.add.1m` over the same element count with no scratch
/// (`vyre-bench/baselines/rtx_5090/smoke_full_2026-04-30_11bccf28.json`), and
/// `adversarial.register_exhaustion.u32_1024`, which holds 100 live variables
/// across 1024 lanes for the register side of the same effect.
///
/// A zero budget means the backend reported none, so nothing is charged rather
/// than a guess.
fn resident_passes(demand: u64, budget: u64) -> u64 {
    if budget == 0 || demand == 0 {
        return 1;
    }
    demand.div_ceil(budget)
}

/// Sum of one per-node counter over the whole graph.
fn sum(counts: &[u64]) -> u64 {
    counts.iter().copied().fold(0_u64, u64::saturating_add)
}

/// Registers a group holds above the full-occupancy budget.
///
/// A group above the budget spills, which is legal: the target compiler moves
/// the excess to local memory and the device schedules fewer groups. A budget of
/// zero means the backend reported none, so nothing is above it.
fn spilled(demand: u64, budget: u64) -> u64 {
    if budget == 0 {
        return 0;
    }
    demand.saturating_sub(budget)
}

/// Nanoseconds `operations` take at `rate` operations per nanosecond.
///
/// A zero rate means the backend measured none, so the term is omitted rather
/// than priced at a rate nothing observed.
fn rate_ns(operations: u64, rate: u64) -> u64 {
    if rate == 0 || operations == 0 {
        return 0;
    }
    operations.div_ceil(rate)
}

#[cfg(test)]
mod tests;
