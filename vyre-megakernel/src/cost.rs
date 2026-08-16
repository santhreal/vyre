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
//! ordered deterministically by [`crate::select::rank`], and a measured
//! compilation decides between them on device time.

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::Ident;

use crate::{
    candidate::CandidatePlan, facts::PlanningFacts, DependencyEdge, DependencyEndpoint,
    DependencyKind, DeviceFacts,
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
    /// Launch term in nanoseconds.
    pub launch_ns: u64,
    /// Materialized-traffic term in nanoseconds.
    pub materialization_ns: u64,
    /// Occupancy term in nanoseconds.
    pub occupancy_ns: u64,
    /// Weighted total in nanoseconds, minimized by candidate selection.
    pub total: u64,
}

pub(crate) fn evaluate(
    candidate: &CandidatePlan,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    device: DeviceFacts,
) -> CostBreakdown {
    let semantic_work = facts
        .node_work
        .iter()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let launches = u64::try_from(candidate.group_count()).unwrap_or(u64::MAX);
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
    for group in 0..u32::try_from(candidate.group_count()).unwrap_or(u32::MAX) {
        let mut group_live = 0_u64;
        let mut group_tiles: Vec<(&Ident, u64)> = Vec::new();
        let mut group_bytes = 0_u64;
        for node in candidate.group_members(group) {
            group_live =
                group_live.saturating_add(facts.node_live_values.get(node).copied().unwrap_or(0));
            for (name, bytes) in facts
                .node_workgroup_scratch
                .get(node)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                let held = group_tiles
                    .iter()
                    .position(|(name_held, _)| *name_held == name);
                match held {
                    // Fusion keeps one declaration per name and takes the
                    // larger count, so the group holds the larger of the two.
                    Some(index) => group_tiles[index].1 = group_tiles[index].1.max(*bytes),
                    None => group_tiles.push((name, *bytes)),
                }
            }
            group_bytes = group_bytes
                .saturating_add(facts.node_touched_bytes.get(node).copied().unwrap_or(0));
        }
        let group_scratch = group_tiles
            .iter()
            .fold(0_u64, |total, (_, bytes)| total.saturating_add(*bytes));
        live_value_peak = live_value_peak.max(group_live);
        shared_scratch_bytes = shared_scratch_bytes.max(group_scratch);
        let passes = resident_passes(group_live, u64::from(device.registers_per_invocation())).max(
            resident_passes(
                group_scratch,
                u64::from(device.shared_scratch_bytes_per_workgroup()),
            ),
        );
        occupancy_passes_peak = occupancy_passes_peak.max(passes);
        occupancy_bytes =
            occupancy_bytes.saturating_add(group_bytes.saturating_mul(passes.saturating_sub(1)));
    }

    let launch_ns = launches.saturating_mul(launch_cost_ns(device));
    let materialization_ns = materialized_bytes / TRAFFIC_BYTES_PER_NS;
    let occupancy_ns = occupancy_bytes / TRAFFIC_BYTES_PER_NS;
    let total = launch_ns
        .saturating_add(materialization_ns)
        .saturating_add(occupancy_ns);
    CostBreakdown {
        semantic_work,
        launches,
        materializations,
        materialized_bytes,
        live_value_peak,
        shared_scratch_bytes,
        occupancy_passes_peak,
        launch_ns,
        materialization_ns,
        occupancy_ns,
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
fn launch_cost_ns(device: DeviceFacts) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::DataflowEdge;
    use vyre_foundation::validate::BackendCapabilities;

    fn device(registers_per_invocation: u32) -> DeviceFacts {
        DeviceFacts::new(BackendCapabilities::default(), 256)
            .with_occupancy(registers_per_invocation, 0)
    }

    /// A device with no register budget and the given workgroup scratch budget.
    fn scratch_device(shared_scratch_bytes_per_workgroup: u32) -> DeviceFacts {
        DeviceFacts::new(BackendCapabilities::default(), 256)
            .with_occupancy(0, shared_scratch_bytes_per_workgroup)
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
    /// paired with the case id it came from.
    fn recorded(metric: &str) -> Vec<(String, u64)> {
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
                found.push((id.to_string(), p50));
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
        let (floor_case, floor) = dispatches
            .iter()
            .min_by_key(|(id, p50)| (*p50, id.as_str()))
            .expect("a recorded dispatch must exist");
        assert_eq!(
            LAUNCH_COST_FLOOR_NS, *floor,
            "the launch floor must be the cheapest recorded dispatch, now held by `{floor_case}`"
        );

        let rates = recorded("device_gb_s_x1000");
        let (_, rate) = rates
            .iter()
            .find(|(id, _)| id == floor_case)
            .expect("the case that fixes the floor must record its device rate");
        assert_eq!(
            TRAFFIC_BYTES_PER_NS,
            rate.saturating_add(500).div_euclid(1_000),
            "the traffic rate must be the rate that case recorded, to the nearest byte"
        );
    }
}
