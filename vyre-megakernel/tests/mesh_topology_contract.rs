//! Mesh facts and the coordinated topology one selected schedule records.
//!
//! WHY: BACKLOG row 64 requires one owner for multi-device execution. These
//! cases defend the rules a decoded topology must satisfy without the graph:
//! every logical point is covered exactly once, every communicating device holds
//! work, stages are contiguous, and no stage can deadlock. They also defend the
//! authentication of the mesh facts a placement was selected against, because a
//! record whose identity does not cover its content lets a spoofed mesh be
//! selected against and submitted somewhere else.
//!
//! What these cases do not prove: that a placement is fast. Ranking is priced by
//! the objective and proven in the selection tests.

#![forbid(unsafe_code)]

use vyre_foundation::logical::LogicalExchangeKind;
use vyre_megakernel::allocation::DeviceSlot;
use vyre_megakernel::mesh::{
    CollectiveSupport, MeshFacts, MeshTopologyPlan, PartitionKind, RegionPartition,
    ShardAssignment, TransferAssignment, TransferOrigin, MESH_TOPOLOGY_VERSION,
};
use vyre_megakernel::{ArtifactNodeId, CompileError};

#[path = "../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::{
    atomic_output_graph, chained_graph, collective_output_graph, compile_graph,
    compile_graph_on_mesh, compile_graph_on_mesh_for_memory, in_place_input_graph, mesh_axis,
    mesh_device, mesh_link, mesh_request, two_device_mesh, wired_input_graph,
};

fn shard(index: u32, device: u16, points: u64) -> ShardAssignment {
    ShardAssignment {
        shard: index,
        device: DeviceSlot(device),
        coordinate: vec![u32::from(device)],
        points,
    }
}

fn split_partition(node: u32, points: u64) -> RegionPartition {
    RegionPartition {
        node: ArtifactNodeId(node),
        kind: PartitionKind::Data,
        axis: Some(0),
        region_points: points,
        shards: vec![shard(0, 0, points / 2), shard(1, 1, points - points / 2)],
    }
}

fn transfer(
    exchange: u32,
    kind: LogicalExchangeKind,
    from: u16,
    to: u16,
    stage: u32,
    overlaps: bool,
) -> TransferAssignment {
    TransferAssignment {
        origin: TransferOrigin::Exchange(exchange),
        kind,
        link: u32::from(from),
        from: DeviceSlot(from),
        to: DeviceSlot(to),
        bytes: 4_096,
        stage,
        overlaps,
    }
}

fn plan(partitions: Vec<RegionPartition>, transfers: Vec<TransferAssignment>) -> MeshTopologyPlan {
    let width = partitions
        .iter()
        .map(|partition| {
            if partition.kind.splits_axis() {
                u32::try_from(partition.shards.len()).unwrap_or(u32::MAX)
            } else {
                1
            }
        })
        .min()
        .unwrap_or(1);
    let anchor = partitions
        .first()
        .and_then(|partition| partition.shards.first())
        .map_or(DeviceSlot(0), |shard| shard.device);
    MeshTopologyPlan {
        version: MESH_TOPOLOGY_VERSION,
        mesh: two_device_mesh().authentication(),
        anchor,
        partitions,
        transfers,
        width,
        communication_ns: 0,
    }
}

fn diagnostic_path(error: &CompileError) -> String {
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
        .unwrap_or_default()
        .to_owned()
}

fn rejection_path(plan: &MeshTopologyPlan) -> String {
    diagnostic_path(
        &plan
            .validate()
            .expect_err("Fix: expect the topology rule under test to reject this plan"),
    )
}

#[test]
fn a_split_partition_and_its_ring_exchange_validate() {
    let plan = plan(
        vec![split_partition(0, 64)],
        vec![
            transfer(0, LogicalExchangeKind::AllReduce, 0, 1, 0, false),
            transfer(0, LogicalExchangeKind::AllReduce, 1, 0, 0, false),
        ],
    );
    plan.validate().expect("Fix: accept a legal ring exchange");
    assert_eq!(plan.devices(), vec![DeviceSlot(0), DeviceSlot(1)]);
    assert_eq!(plan.stage_count(), 1);
}

#[test]
fn shards_cover_every_logical_point_exactly_once() {
    let mut partition = split_partition(0, 64);
    partition.shards[1].points -= 1;
    assert_eq!(
        rejection_path(&plan(vec![partition], Vec::new())),
        "topology.partitions[0].shards"
    );
}

#[test]
fn a_shard_with_no_work_is_rejected() {
    let mut partition = split_partition(0, 64);
    partition.shards[0].points = 0;
    partition.shards[1].points = 64;
    assert_eq!(
        rejection_path(&plan(vec![partition], Vec::new())),
        "topology.partitions[0].shards[0].points"
    );
}

#[test]
fn a_replicated_shard_holds_the_whole_region() {
    let mut partition = split_partition(0, 64);
    partition.kind = PartitionKind::Replicated;
    partition.axis = None;
    assert_eq!(
        rejection_path(&plan(vec![partition], Vec::new())),
        "topology.partitions[0].shards[0].points"
    );
}

#[test]
fn an_axis_is_stated_exactly_when_the_partition_cuts_one() {
    let mut cutting = split_partition(0, 64);
    cutting.axis = None;
    assert_eq!(
        rejection_path(&plan(vec![cutting], Vec::new())),
        "topology.partitions[0].axis"
    );

    let mut replicated = RegionPartition {
        node: ArtifactNodeId(0),
        kind: PartitionKind::Replicated,
        axis: None,
        region_points: 64,
        shards: vec![shard(0, 0, 64), shard(1, 1, 64)],
    };
    replicated.axis = Some(0);
    assert_eq!(
        rejection_path(&plan(vec![replicated], Vec::new())),
        "topology.partitions[0].axis"
    );
}

#[test]
fn one_device_is_never_recorded_at_two_coordinates() {
    let mut partition = split_partition(0, 64);
    partition.shards[1].device = DeviceSlot(0);
    assert_eq!(
        rejection_path(&plan(vec![partition], Vec::new())),
        "topology.partitions[0].shards[1].coordinate"
    );
}

#[test]
fn two_partitions_never_place_one_node() {
    assert_eq!(
        rejection_path(&plan(
            vec![split_partition(0, 64), split_partition(0, 64)],
            Vec::new()
        )),
        "topology.partitions[1].node"
    );
}

#[test]
fn a_transfer_names_only_devices_that_hold_a_shard() {
    let partition = RegionPartition {
        node: ArtifactNodeId(0),
        kind: PartitionKind::Replicated,
        axis: None,
        region_points: 64,
        shards: vec![shard(0, 0, 64)],
    };
    assert_eq!(
        rejection_path(&plan(
            vec![partition],
            vec![transfer(
                0,
                LogicalExchangeKind::PointToPoint,
                0,
                1,
                0,
                false
            )]
        )),
        "topology.transfers[0].to"
    );
}

#[test]
fn a_transfer_that_moves_no_byte_is_rejected() {
    let mut moved = transfer(0, LogicalExchangeKind::PointToPoint, 0, 1, 0, false);
    moved.bytes = 0;
    assert_eq!(
        rejection_path(&plan(vec![split_partition(0, 64)], vec![moved])),
        "topology.transfers[0].bytes"
    );
}

#[test]
fn stages_are_contiguous_from_zero() {
    let mut late = transfer(0, LogicalExchangeKind::PointToPoint, 0, 1, 1, false);
    late.stage = 1;
    assert_eq!(
        rejection_path(&plan(vec![split_partition(0, 64)], vec![late])),
        "topology.transfers"
    );
}

#[test]
fn two_overlapping_transfers_never_share_a_link_in_one_stage() {
    let first = transfer(0, LogicalExchangeKind::AllGather, 0, 1, 0, true);
    let second = transfer(1, LogicalExchangeKind::AllGather, 0, 1, 0, true);
    assert_eq!(
        rejection_path(&plan(vec![split_partition(0, 64)], vec![first, second])),
        "topology.transfers[1].overlaps"
    );
}

#[test]
fn two_collectives_over_shared_devices_never_share_a_stage() {
    let first = transfer(0, LogicalExchangeKind::AllReduce, 0, 1, 0, false);
    let mut second = transfer(1, LogicalExchangeKind::AllReduce, 1, 0, 0, false);
    second.link = 1;
    assert_eq!(
        rejection_path(&plan(vec![split_partition(0, 64)], vec![first, second])),
        "topology.transfers[1].stage"
    );
}

#[test]
fn point_to_point_transfers_of_one_stage_never_wait_in_a_cycle() {
    let out = transfer(0, LogicalExchangeKind::PointToPoint, 0, 1, 0, false);
    let mut back = transfer(1, LogicalExchangeKind::PointToPoint, 1, 0, 0, false);
    back.link = 1;
    assert_eq!(
        rejection_path(&plan(vec![split_partition(0, 64)], vec![out, back])),
        "topology.transfers"
    );
}

#[test]
fn a_stored_topology_of_another_schema_is_refused() {
    let mut stale = plan(vec![split_partition(0, 64)], Vec::new());
    stale.version = MESH_TOPOLOGY_VERSION + 1;
    assert_eq!(rejection_path(&stale), "topology.version");
}

#[test]
fn a_mesh_states_one_device_per_coordinate() {
    let error = MeshFacts::new(
        vec![mesh_axis(2)],
        vec![mesh_device(0, 0, 1 << 30), mesh_device(1, 0, 1 << 30)],
        Vec::new(),
        CollectiveSupport::NONE,
    )
    .expect_err("Fix: reject two devices sharing one mesh coordinate");
    assert_eq!(diagnostic_path(&error), "mesh.devices");
}

#[test]
fn a_mesh_link_names_only_devices_the_mesh_states() {
    let error = MeshFacts::new(
        vec![mesh_axis(1)],
        vec![mesh_device(0, 0, 1 << 30)],
        vec![mesh_link(0, 7)],
        CollectiveSupport::ALL,
    )
    .expect_err("Fix: reject a link to a device the mesh omits");
    assert_eq!(diagnostic_path(&error), "mesh.links[0].to");
}

#[test]
fn mesh_identity_covers_every_mesh_fact() {
    let mesh = two_device_mesh();
    mesh.authenticate()
        .expect("Fix: accept facts whose identity covers them");
    let mut value =
        serde_json::to_value(&mesh).expect("Fix: serialize authenticated mesh facts to JSON");
    value["links"][0]["bandwidth_bytes_per_ns"] = serde_json::json!(1_024);
    let spoofed: MeshFacts =
        serde_json::from_value(value).expect("Fix: decode the tampered mesh record");
    let error = spoofed
        .authenticate()
        .expect_err("Fix: reject facts whose identity does not cover them");
    assert_eq!(diagnostic_path(&error), "mesh.authentication");
}

#[test]
fn a_mesh_axis_of_extent_zero_is_rejected() {
    let error = MeshFacts::new(
        vec![mesh_axis(0)],
        vec![mesh_device(0, 0, 0)],
        Vec::new(),
        CollectiveSupport::NONE,
    )
    .expect_err("Fix: reject a mesh axis with no coordinate");
    assert_eq!(diagnostic_path(&error), "mesh.axes[0].extent");
}

#[test]
fn every_exchange_kind_states_whether_the_mesh_carries_it() {
    let kinds = [
        LogicalExchangeKind::AllReduce,
        LogicalExchangeKind::AllGather,
        LogicalExchangeKind::ReduceScatter,
        LogicalExchangeKind::Broadcast,
        LogicalExchangeKind::PointToPoint,
    ];
    for kind in kinds {
        assert!(
            CollectiveSupport::ALL.carries(kind),
            "a mesh that carries every exchange must carry {kind:?}"
        );
        assert!(
            !CollectiveSupport::NONE.carries(kind),
            "a mesh that carries no exchange must reject {kind:?}"
        );
    }
}

/// WHY: a mesh placement must distribute the work, not duplicate it. A partition
/// that held the whole region on every device would report a mesh that runs
/// faster while holding more bytes than one device ever did, and the peak the
/// artifact records would be a figure no device holds.
#[test]
fn a_two_device_mesh_cuts_every_region_and_holds_the_bytes_one_device_holds() {
    let single = compile_graph(wired_input_graph(8), 0);
    let placed = compile_graph_on_mesh(wired_input_graph(8), 0, two_device_mesh());

    assert_eq!(single.topology().devices(), vec![DeviceSlot(0)]);
    assert_eq!(single.topology().width, 1);
    assert_eq!(
        placed.topology().devices(),
        vec![DeviceSlot(0), DeviceSlot(1)]
    );
    assert_eq!(placed.topology().width, 2);
    assert_eq!(placed.topology().anchor, DeviceSlot(0));
    assert_eq!(
        placed.allocation().aggregate_peak_bytes,
        single.allocation().aggregate_peak_bytes,
        "a partition distributes the bytes a single device would hold"
    );
    for partition in &placed.topology().partitions {
        assert_eq!(partition.shards.len(), 2, "every region is cut in two");
        let covered: u64 = partition.shards.iter().map(|shard| shard.points).sum();
        assert_eq!(covered, partition.region_points);
    }
    for peak in &placed.allocation().device_peaks {
        assert!(
            placed.topology().devices().contains(&peak.device),
            "device {} holds bytes the topology does not place",
            peak.device.0
        );
    }
}

/// WHY: the mesh a placement was selected against is part of what the artifact
/// is. Two compiles of one graph on different meshes that shared an identity
/// would let an artifact cut for two devices admit on one.
#[test]
fn the_artifact_identity_covers_the_mesh_it_was_placed_on() {
    let single = compile_graph(wired_input_graph(8), 0);
    let placed = compile_graph_on_mesh(wired_input_graph(8), 0, two_device_mesh());

    assert_ne!(single.digest(), placed.digest());
    assert_eq!(
        placed.topology().mesh,
        two_device_mesh().authentication(),
        "the topology names the mesh it was selected against"
    );
}

/// WHY: a region with no axis to cut has to keep a legal placement. Pricing the
/// illegal partition instead, or failing the compile, would make an
/// unpartitionable graph unbuildable on a machine that happens to hold two
/// devices.
#[test]
fn a_region_with_no_axis_to_cut_retains_the_single_device_placement() {
    let placed = compile_graph_on_mesh(wired_input_graph(1), 0, two_device_mesh());

    assert_eq!(placed.topology().devices(), vec![DeviceSlot(0)]);
    assert_eq!(placed.topology().width, 1);
    assert!(placed.topology().transfers.is_empty());
}

/// WHY: capacity is a legality question, not a price. A mesh whose devices
/// cannot hold their share has to say so and name both figures, because ranking
/// a placement no device can run is how an artifact that cannot launch gets
/// selected.
#[test]
fn a_mesh_that_cannot_hold_its_share_fails_with_both_figures() {
    let cramped = MeshFacts::new(
        vec![mesh_axis(2)],
        vec![mesh_device(0, 0, 1), mesh_device(1, 1, 1)],
        vec![mesh_link(0, 1), mesh_link(1, 0)],
        CollectiveSupport::ALL,
    )
    .expect("Fix: authenticate a mesh whose devices hold one byte");
    let request = mesh_request(wired_input_graph(8), cramped);

    let error = vyre_megakernel::compile(&request)
        .expect_err("Fix: refuse a placement no device of the mesh can hold");

    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC044_MESH_CAPACITY_EXCEEDED"
    );
    let message = error.diagnostic.message.as_ref();
    assert!(
        message.contains("holds 1 bytes") && message.contains("needs"),
        "the diagnostic must name the capacity and the need: {message}"
    );
}

/// WHY: an exchange the program states has to appear in the placement as routed
/// bytes over a link the mesh holds. A placement that cut every region and
/// recorded no transfer would run two devices that never combine their
/// contributions, and the plan would state a result neither device computed.
#[test]
fn a_stated_collective_is_routed_over_the_links_it_crosses() {
    let placed = compile_graph_on_mesh(collective_output_graph(8), 0, two_device_mesh());
    let topology = placed.topology();

    assert_eq!(topology.devices(), vec![DeviceSlot(0), DeviceSlot(1)]);
    assert_eq!(
        topology
            .transfers
            .iter()
            .map(|transfer| (
                transfer.from,
                transfer.to,
                transfer.stage,
                transfer.overlaps
            ))
            .collect::<Vec<_>>(),
        vec![
            (DeviceSlot(0), DeviceSlot(1), 0, false),
            (DeviceSlot(1), DeviceSlot(0), 0, false),
        ],
        "Fix: a combining exchange walks the participants as a ring in one stage \
         and never overlaps the stage that consumes it"
    );
    assert!(topology
        .transfers
        .iter()
        .all(|transfer| transfer.kind == LogicalExchangeKind::AllReduce && transfer.bytes > 0));
    assert!(
        topology.communication_ns > 0,
        "Fix: routed bytes cost time the ranking can read"
    );
}

/// WHY: a mesh that does not carry the exchange a program states cannot run the
/// partitioned placement. Recording it anyway would submit a collective the
/// devices have no path for, and pricing it would rank a plan that cannot
/// complete above the single-device plan that can.
#[test]
fn a_mesh_that_carries_no_collective_keeps_the_single_device_placement() {
    let silent = MeshFacts::new(
        vec![mesh_axis(2)],
        vec![mesh_device(0, 0, 1 << 30), mesh_device(1, 1, 1 << 30)],
        vec![mesh_link(0, 1), mesh_link(1, 0)],
        CollectiveSupport::NONE,
    )
    .expect("Fix: authenticate a mesh that carries no collective");

    let placed = vyre_megakernel::compile(&mesh_request(collective_output_graph(8), silent))
        .expect("Fix: keep compiling when the mesh carries no collective");

    assert_eq!(placed.topology().devices(), vec![DeviceSlot(0)]);
    assert!(placed.topology().transfers.is_empty());
}

/// WHY: a region that reads a value it also writes may read a point another
/// shard holds. Recording it as an independent element cut would let one shard
/// read bytes no device sent it, so the placement has to state that the domain
/// is spatial even though the cut looks the same.
#[test]
fn a_region_that_reads_what_it_writes_is_cut_as_a_spatial_domain() {
    let placed = compile_graph_on_mesh(in_place_input_graph(8), 0, two_device_mesh());
    let topology = placed.topology();

    assert_eq!(topology.devices(), vec![DeviceSlot(0), DeviceSlot(1)]);
    assert_eq!(
        topology
            .partitions
            .iter()
            .map(|partition| partition.kind)
            .collect::<Vec<_>>(),
        vec![PartitionKind::Spatial],
        "Fix: an in-place read makes the cut axis a spatial domain"
    );
}

/// WHY: an atomic update lands where the program computes, so a shard holds
/// contributions the other shards own. A placement that cut the region and
/// recorded no routing would drop every contribution that crossed a shard
/// boundary, and one that refused to cut it would leave a mesh unusable for the
/// scatter workloads that need it most.
#[test]
fn a_routed_region_routes_its_contributions_to_the_shard_that_owns_them() {
    let placed = compile_graph_on_mesh_for_memory(atomic_output_graph(8), 0, two_device_mesh());
    let topology = placed.topology();

    assert_eq!(topology.devices(), vec![DeviceSlot(0), DeviceSlot(1)]);
    assert_eq!(
        topology
            .partitions
            .iter()
            .map(|partition| partition.kind)
            .collect::<Vec<_>>(),
        vec![PartitionKind::Routed]
    );
    let node = topology.partitions[0].node;
    assert_eq!(
        topology
            .transfers
            .iter()
            .map(|transfer| (transfer.origin, transfer.from, transfer.to, transfer.stage))
            .collect::<Vec<_>>(),
        vec![
            (
                TransferOrigin::Routing { node },
                DeviceSlot(0),
                DeviceSlot(1),
                0
            ),
            (
                TransferOrigin::Routing { node },
                DeviceSlot(1),
                DeviceSlot(0),
                0
            ),
        ],
        "Fix: routing walks the shard devices and names the region it routes"
    );
    assert!(topology.transfers.iter().all(|transfer| {
        transfer.kind == LogicalExchangeKind::ReduceScatter
            && transfer.bytes > 0
            && !transfer.overlaps
    }));
}

/// WHY: a chain of one-point regions has no axis to cut, and one device holding
/// every region holds every value at once. Without a placement that spreads
/// whole regions, the three partition kinds that do not cut an axis would be
/// unreachable and a graph too large for one device would have no legal plan
/// besides the one that does not fit.
#[test]
fn a_chain_no_region_can_cut_runs_as_a_pipeline() {
    let placed = compile_graph_on_mesh_for_memory(chained_graph(), 0, two_device_mesh());
    let topology = placed.topology();

    assert_eq!(
        topology
            .partitions
            .iter()
            .map(|partition| (partition.kind, partition.shards.len()))
            .collect::<Vec<_>>(),
        vec![(PartitionKind::Pipeline, 1), (PartitionKind::Pipeline, 1)],
        "Fix: a pipeline runs each region whole on one device"
    );
    assert_eq!(topology.devices(), vec![DeviceSlot(0), DeviceSlot(1)]);
    assert_eq!(topology.width, 1, "no region is cut");
    assert_eq!(topology.pipeline_depth(), 2);
    assert_eq!(
        topology
            .transfers
            .iter()
            .map(|transfer| (
                transfer.origin,
                transfer.from,
                transfer.to,
                transfer.overlaps
            ))
            .collect::<Vec<_>>(),
        vec![(
            TransferOrigin::Handoff {
                producer: topology.partitions[0].node,
                consumer: topology.partitions[1].node,
            },
            DeviceSlot(0),
            DeviceSlot(1),
            true
        )],
        "Fix: the values one stage produces are handed to the device that consumes them"
    );
    let single = compile_graph(chained_graph(), 0);
    let mesh_peak = placed
        .allocation()
        .device_peaks
        .iter()
        .map(|peak| peak.peak_bytes)
        .max()
        .unwrap_or(0);
    assert!(
        mesh_peak < single.allocation().aggregate_peak_bytes,
        "Fix: a stage device holds fewer bytes than the one device that held every region"
    );
}

/// WHY: a pipeline stage is the whole region on one device. Two shards would
/// state that the region was cut while recording every point on both devices,
/// which is the duplicated-bytes defect the width and peak figures are derived
/// from.
#[test]
fn a_pipeline_partition_places_its_region_on_one_device() {
    let mut partition = split_partition(0, 64);
    partition.kind = PartitionKind::Pipeline;
    partition.axis = None;
    partition.shards[0].points = 64;
    partition.shards[1].points = 64;
    assert_eq!(
        rejection_path(&plan(vec![partition], Vec::new())),
        "topology.partitions[0].shards"
    );
}

/// WHY: a routed partition is legal only because the contributions a shard does
/// not own are sent to the shard that does. A topology that recorded the cut and
/// no routing transfer would run every shard against its own points alone.
#[test]
fn a_routed_partition_records_the_routing_it_depends_on() {
    let mut partition = split_partition(0, 64);
    partition.kind = PartitionKind::Routed;
    assert_eq!(
        rejection_path(&plan(vec![partition.clone()], Vec::new())),
        "topology.partitions[0].kind"
    );

    let routing = TransferAssignment {
        origin: TransferOrigin::Routing {
            node: ArtifactNodeId(0),
        },
        kind: LogicalExchangeKind::ReduceScatter,
        link: 0,
        from: DeviceSlot(0),
        to: DeviceSlot(1),
        bytes: 4_096,
        stage: 0,
        overlaps: false,
    };
    plan(vec![partition], vec![routing])
        .validate()
        .expect("Fix: accept a routed partition whose routing transfer is recorded");
}
