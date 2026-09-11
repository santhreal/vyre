//! Placement across a device mesh: the candidates, their communication cost, and
//! the one the stated objective orders first.
//!
//! Placement is a schedule decision, so it is made here and nowhere else. A
//! driver reports the mesh it authenticates and the links it measured; it does
//! not choose a partition. The runtime submits the topology this module selected;
//! it does not discover one.
//!
//! Every candidate is a generic transform over logical facts. The single-device
//! placement is always the first candidate and is never pruned, so a mesh that
//! cannot carry an exchange, lacks a link, or cannot hold a shard costs the
//! compile nothing: the placement that needs none of that is still there.

mod facts;
mod plan;

use vyre_foundation::logical::{
    LogicalExchange, LogicalExchangeKind, LogicalPartitionAxisKind, LogicalProgramGraph,
};

pub use facts::{CollectiveSupport, MeshAxis, MeshDevice, MeshFacts, MeshLink, MESH_FACTS_VERSION};
pub use plan::{
    implied_width, MeshTopologyPlan, PartitionKind, RegionPartition, ShardAssignment,
    TransferAssignment, TransferOrigin, MESH_TOPOLOGY_VERSION,
};

use crate::allocation::DeviceSlot;
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::identity::ArtifactNodeId;
use crate::objective::{CompileObjective, ObjectiveMetric};

/// Every placement of one logical program on one mesh, single device first.
///
/// # Errors
///
/// Returns when the mesh facts are not authentic, or when a logical region has
/// no point bound the placement could cut.
pub(crate) fn candidates(
    logical: &LogicalProgramGraph<'_>,
    mesh: &MeshFacts,
) -> Result<Vec<MeshTopologyPlan>, CompileError> {
    mesh.authenticate()?;
    let mut candidates = vec![single_device(logical, mesh)?];
    for (axis, extent) in mesh
        .axes()
        .iter()
        .enumerate()
        .map(|(axis, entry)| (axis, entry.extent))
        .filter(|(_, extent)| *extent > 1)
    {
        if let Some(candidate) = split_along(logical, mesh, axis, extent)? {
            candidates.push(candidate);
        }
        if let Some(candidate) = pipeline_along(logical, mesh, axis, extent)? {
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

/// The placement that puts every region on the first mesh device.
fn single_device(
    logical: &LogicalProgramGraph<'_>,
    mesh: &MeshFacts,
) -> Result<MeshTopologyPlan, CompileError> {
    let device = mesh.devices().first().ok_or_else(|| {
        failure(
            CompilerFailureKind::InvalidMeshFacts,
            "mesh.devices",
            "an authenticated mesh states no device",
            "report every device the target authenticates",
        )
    })?;
    let partitions = logical
        .regions()
        .iter()
        .map(|region| RegionPartition {
            node: ArtifactNodeId(region.node.0),
            kind: PartitionKind::Replicated,
            axis: None,
            region_points: region.max_points,
            shards: vec![ShardAssignment {
                shard: 0,
                device: device.slot,
                coordinate: device.coordinate.clone(),
                points: region.max_points,
            }],
        })
        .collect();
    let plan = MeshTopologyPlan::single_device(mesh.authentication(), device.slot, partitions);
    plan.validate()?;
    Ok(plan)
}

/// The placement that cuts every region along one mesh axis.
///
/// A multi-device placement is a pure partition: every region is cut, so the
/// bytes of one value are distributed rather than duplicated and the mesh holds
/// exactly the byte total one device would. A region that cannot be cut makes
/// this placement ineligible, and so does one that is not replicable unless its
/// cut axis is routed, because a routed shard sends the contributions it
/// computed to the shard that owns their destination instead of holding a second
/// copy of the region.
///
/// Returns `None` when the mesh cannot carry the placement: a coordinate with no
/// device, an uncuttable region, an exchange kind the mesh does not carry, or a
/// missing link. A rejected placement is not an error because the single-device
/// placement is always retained.
fn split_along(
    logical: &LogicalProgramGraph<'_>,
    mesh: &MeshFacts,
    axis: usize,
    extent: u32,
) -> Result<Option<MeshTopologyPlan>, CompileError> {
    let Some(row) = row_along(mesh, axis, extent) else {
        return Ok(None);
    };
    if logical.regions().is_empty() {
        return Ok(None);
    }
    let mut partitions = Vec::with_capacity(logical.regions().len());
    for region in logical.regions() {
        let Some((logical_axis, kind)) = region
            .partition
            .axes
            .iter()
            .find(|entry| entry.bound > 1)
            .map(|entry| (entry.axis, partition_kind(entry.kind)))
        else {
            return Ok(None);
        };
        // A routed region is not replicable and is still cuttable: its updates
        // land at data-dependent points, so a shard sends the contributions it
        // computed to the shard that owns their destination.
        if !region.partition.replicable && kind != PartitionKind::Routed {
            return Ok(None);
        }
        let shards = chunk(region.max_points, row.len());
        if shards.len() < 2 {
            return Ok(None);
        }
        partitions.push(RegionPartition {
            node: ArtifactNodeId(region.node.0),
            kind,
            axis: Some(logical_axis),
            region_points: region.max_points,
            shards: shards
                .into_iter()
                .enumerate()
                .map(|(index, points)| ShardAssignment {
                    shard: u32::try_from(index).unwrap_or(u32::MAX),
                    device: row[index].slot,
                    coordinate: row[index].coordinate.clone(),
                    points,
                })
                .collect(),
        });
    }
    let mut transfers = match route(logical.exchanges(), &partitions, mesh)? {
        Some(transfers) => transfers,
        None => return Ok(None),
    };
    match route_shards(logical, &partitions, mesh, next_stage(&transfers))? {
        Some(mut routing) => transfers.append(&mut routing),
        None => return Ok(None),
    }
    let mut plan = MeshTopologyPlan {
        version: MESH_TOPOLOGY_VERSION,
        mesh: mesh.authentication(),
        anchor: row[0].slot,
        width: implied_width(&partitions),
        partitions,
        transfers,
        communication_ns: 0,
    };
    plan.communication_ns = communication_ns(&plan, mesh);
    plan.validate()?;
    Ok(Some(plan))
}

/// The generic transform that cuts one logical axis kind.
const fn partition_kind(kind: LogicalPartitionAxisKind) -> PartitionKind {
    match kind {
        LogicalPartitionAxisKind::Elementwise => PartitionKind::Data,
        LogicalPartitionAxisKind::Spatial => PartitionKind::Spatial,
        LogicalPartitionAxisKind::Reduction => PartitionKind::Reduction,
        LogicalPartitionAxisKind::Sequence => PartitionKind::Sequence,
        LogicalPartitionAxisKind::Routed => PartitionKind::Routed,
    }
}

/// The devices one mesh axis addresses, anchored at the first device.
///
/// Returns `None` when the axis addresses a coordinate the mesh omits, which is
/// a mesh no placement along that axis can use.
fn row_along(mesh: &MeshFacts, axis: usize, extent: u32) -> Option<Vec<&MeshDevice>> {
    let anchor = mesh.devices().first()?;
    let mut row = Vec::with_capacity(extent as usize);
    for position in 0..extent {
        let mut coordinate = anchor.coordinate.clone();
        coordinate[axis] = position;
        row.push(
            mesh.devices()
                .iter()
                .find(|device| device.coordinate == coordinate)?,
        );
    }
    Some(row)
}

/// The placement that runs each region whole on one device of a mesh axis.
///
/// A pipeline places consecutive regions on consecutive devices and hands the
/// values one region produces to the device that consumes them, so a chain no
/// single device can hold in one submission runs as stages. Nothing is cut, so
/// this placement needs no partitionable region and holds for retained state and
/// ordered work that `split_along` rejects.
///
/// Returns `None` when the mesh omits a coordinate of the axis, the program has
/// fewer than two regions, the axis addresses fewer than two devices, or a
/// handoff has no link to travel over.
fn pipeline_along(
    logical: &LogicalProgramGraph<'_>,
    mesh: &MeshFacts,
    axis: usize,
    extent: u32,
) -> Result<Option<MeshTopologyPlan>, CompileError> {
    let regions = logical.regions();
    if regions.len() < 2 {
        return Ok(None);
    }
    let Some(row) = row_along(mesh, axis, extent) else {
        return Ok(None);
    };
    if row.len() < 2 {
        return Ok(None);
    }
    if !mesh
        .collectives()
        .carries(LogicalExchangeKind::PointToPoint)
    {
        return Ok(None);
    }
    let mut partitions = Vec::with_capacity(regions.len());
    let mut device_of = std::collections::BTreeMap::new();
    for (index, region) in regions.iter().enumerate() {
        if region.max_points == 0 {
            return Ok(None);
        }
        let stage = index * row.len() / regions.len();
        let device = row[stage.min(row.len() - 1)];
        device_of.insert(region.node.0, device.slot);
        partitions.push(RegionPartition {
            node: ArtifactNodeId(region.node.0),
            kind: PartitionKind::Pipeline,
            axis: None,
            region_points: region.max_points,
            shards: vec![ShardAssignment {
                shard: 0,
                device: device.slot,
                coordinate: device.coordinate.clone(),
                points: region.max_points,
            }],
        });
    }
    let mut stage = 0u32;
    let mut transfers = Vec::new();
    for region in regions {
        let Some(consumer) = device_of.get(&region.node.0).copied() else {
            return Ok(None);
        };
        for dependence in &region.dependencies {
            let Some(producer) = device_of.get(&dependence.predecessor.0).copied() else {
                continue;
            };
            if producer == consumer {
                continue;
            }
            let Some((link, _)) = mesh.link(producer, consumer) else {
                return Ok(None);
            };
            transfers.push(TransferAssignment {
                origin: TransferOrigin::Handoff {
                    producer: ArtifactNodeId(dependence.predecessor.0),
                    consumer: ArtifactNodeId(region.node.0),
                },
                kind: LogicalExchangeKind::PointToPoint,
                link: u32::try_from(link).unwrap_or(u32::MAX),
                from: producer,
                to: consumer,
                bytes: dependence.bytes.max(1),
                stage,
                // A handoff supplies the stage that consumes it, so the stage
                // issuing it waits for nothing of its own.
                overlaps: true,
            });
            stage += 1;
        }
    }
    if transfers.is_empty() {
        return Ok(None);
    }
    let mut plan = MeshTopologyPlan {
        version: MESH_TOPOLOGY_VERSION,
        mesh: mesh.authentication(),
        anchor: row[0].slot,
        partitions,
        transfers,
        width: 1,
        communication_ns: 0,
    };
    plan.communication_ns = communication_ns(&plan, mesh);
    plan.validate()?;
    Ok(Some(plan))
}

/// Cut `points` into at most `parts` positive chunks.
fn chunk(points: u64, parts: usize) -> Vec<u64> {
    let parts = u64::try_from(parts).unwrap_or(1).max(1);
    if points == 0 {
        return Vec::new();
    }
    let base = points / parts;
    let remainder = points % parts;
    (0..parts)
        .map(|index| base + u64::from(index < remainder))
        .filter(|chunk| *chunk > 0)
        .collect()
}

/// Assign every exchange whose participants span devices to mesh links.
///
/// Each exchange occupies its own stage: two collectives over intersecting
/// devices in one stage would need a global order the stage does not state, and
/// that is the ordering defect the plan refuses to record.
fn route(
    exchanges: &[LogicalExchange],
    partitions: &[RegionPartition],
    mesh: &MeshFacts,
) -> Result<Option<Vec<TransferAssignment>>, CompileError> {
    let mut transfers = Vec::new();
    let mut stage = 0u32;
    for (index, exchange) in exchanges.iter().enumerate() {
        let Some(partition) = partitions
            .iter()
            .find(|partition| partition.node.0 == exchange.node.0)
        else {
            continue;
        };
        let mut participants = partition
            .shards
            .iter()
            .map(|shard| shard.device)
            .collect::<Vec<_>>();
        participants.sort_unstable();
        participants.dedup();
        if participants.len() < 2 {
            continue;
        }
        if !mesh.collectives().carries(exchange.kind) {
            return Ok(None);
        }
        let exchange_index = u32::try_from(index).unwrap_or(u32::MAX);
        let hops = hops(exchange.kind, &participants);
        let mut stage_transfers = Vec::with_capacity(hops.len());
        for (from, to) in hops {
            let Some((link, _)) = mesh.link(from, to) else {
                return Ok(None);
            };
            stage_transfers.push(TransferAssignment {
                origin: TransferOrigin::Exchange(exchange_index),
                kind: exchange.kind,
                link: u32::try_from(link).unwrap_or(u32::MAX),
                from,
                to,
                bytes: exchange.bytes.max(1),
                stage,
                overlaps: overlaps(exchange.kind),
            });
        }
        if stage_transfers.is_empty() {
            continue;
        }
        transfers.append(&mut stage_transfers);
        stage += 1;
    }
    Ok(Some(transfers))
}

/// Stage a transfer appended after `transfers` belongs to.
fn next_stage(transfers: &[TransferAssignment]) -> u32 {
    transfers
        .last()
        .map_or(0, |transfer| transfer.stage.saturating_add(1))
}

/// Assign the shard-to-shard routing a data-dependent update needs.
///
/// A routed region computes contributions whose destination point is known only
/// at run time, so every shard holds contributions the other shards own. That is
/// a reduce-scatter over the shard devices: each participant contributes, and
/// each receives the combined contributions of the points it holds. The mesh
/// must carry that exchange for the placement to be legal.
///
/// Each routed region occupies its own stage, for the same reason a stated
/// collective does: two of them over intersecting devices in one stage would
/// need an order the stage does not state.
fn route_shards(
    logical: &LogicalProgramGraph<'_>,
    partitions: &[RegionPartition],
    mesh: &MeshFacts,
    start_stage: u32,
) -> Result<Option<Vec<TransferAssignment>>, CompileError> {
    let mut transfers = Vec::new();
    let mut stage = start_stage;
    for partition in partitions
        .iter()
        .filter(|partition| partition.kind == PartitionKind::Routed)
    {
        let mut participants = partition
            .shards
            .iter()
            .map(|shard| shard.device)
            .collect::<Vec<_>>();
        participants.sort_unstable();
        participants.dedup();
        if participants.len() < 2 {
            continue;
        }
        if !mesh
            .collectives()
            .carries(LogicalExchangeKind::ReduceScatter)
        {
            return Ok(None);
        }
        let Some(region) = logical
            .regions()
            .iter()
            .find(|region| region.node.0 == partition.node.0)
        else {
            continue;
        };
        // One shard routes the contributions it wrote, which is its share of the
        // bytes the region writes.
        let shards = u64::try_from(partition.shards.len()).unwrap_or(1).max(1);
        let bytes = (region.written_bytes / shards).max(1);
        let mut stage_transfers = Vec::with_capacity(participants.len());
        for (from, to) in hops(LogicalExchangeKind::ReduceScatter, &participants) {
            let Some((link, _)) = mesh.link(from, to) else {
                return Ok(None);
            };
            stage_transfers.push(TransferAssignment {
                origin: TransferOrigin::Routing {
                    node: partition.node,
                },
                kind: LogicalExchangeKind::ReduceScatter,
                link: u32::try_from(link).unwrap_or(u32::MAX),
                from,
                to,
                bytes,
                stage,
                overlaps: overlaps(LogicalExchangeKind::ReduceScatter),
            });
        }
        if stage_transfers.is_empty() {
            continue;
        }
        transfers.append(&mut stage_transfers);
        stage = stage.saturating_add(1);
    }
    Ok(Some(transfers))
}

/// Ordered device pairs one exchange kind moves bytes between.
///
/// A combining exchange walks the participants as a ring, so every link carries
/// one hop and no device is a hub. A broadcast is a star from the first
/// participant, which is what the semantics states. A point-to-point exchange is
/// one pair.
fn hops(kind: LogicalExchangeKind, participants: &[DeviceSlot]) -> Vec<(DeviceSlot, DeviceSlot)> {
    match kind {
        LogicalExchangeKind::AllReduce
        | LogicalExchangeKind::AllGather
        | LogicalExchangeKind::ReduceScatter => (0..participants.len())
            .map(|index| {
                (
                    participants[index],
                    participants[(index + 1) % participants.len()],
                )
            })
            .collect(),
        LogicalExchangeKind::Broadcast => participants[1..]
            .iter()
            .map(|target| (participants[0], *target))
            .collect(),
        LogicalExchangeKind::PointToPoint => vec![(participants[0], participants[1])],
    }
}

/// Whether one exchange kind may run concurrently with computation.
///
/// A gather or a broadcast supplies a later stage, so the stage that issues it
/// has no dependency on its result. A combining exchange produces the value its
/// own stage consumes, so overlapping it would consume bytes that have not
/// arrived.
const fn overlaps(kind: LogicalExchangeKind) -> bool {
    match kind {
        LogicalExchangeKind::AllGather
        | LogicalExchangeKind::Broadcast
        | LogicalExchangeKind::PointToPoint => true,
        LogicalExchangeKind::AllReduce | LogicalExchangeKind::ReduceScatter => false,
    }
}

/// Device nanoseconds one topology spends moving bytes.
///
/// Transfers that may overlap are charged the slowest of their stage rather than
/// their sum, because the stage waits for the last one to arrive. Transfers that
/// may not overlap are charged in full.
fn communication_ns(plan: &MeshTopologyPlan, mesh: &MeshFacts) -> u64 {
    let mut total = 0u64;
    for stage in 0..plan.stage_count() {
        let mut serial = 0u64;
        let mut overlapped = 0u64;
        for transfer in plan
            .transfers
            .iter()
            .filter(|transfer| transfer.stage == stage)
        {
            let cost = mesh
                .links()
                .get(transfer.link as usize)
                .map(|link| {
                    link.latency_ns
                        .saturating_add(transfer.bytes / link.bandwidth_bytes_per_ns.max(1))
                })
                .unwrap_or(0);
            if transfer.overlaps {
                overlapped = overlapped.max(cost);
            } else {
                serial = serial.saturating_add(cost);
            }
        }
        total = total.saturating_add(serial).saturating_add(overlapped);
    }
    total
}

/// The placement the objective orders first, given the winning schedule figures.
///
/// The ranking is joint over schedule and placement without a cross product: for
/// a fixed placement every candidate's adjusted figure is monotone in its own
/// figure, so the best pair is the best schedule under the best placement for
/// it. Ties keep the earlier candidate, and the single-device placement is first,
/// so a placement that does not pay for itself never displaces it.
///
/// Cutting a region shortens one submission, so latency divides by the width. A
/// pipeline leaves one submission as long as it was and runs consecutive
/// submissions on consecutive devices, so it divides throughput and the bytes one
/// device holds instead. Both are first-order models the objective ranks with;
/// measurement decides when the budget allows it.
pub(crate) fn choose(
    candidates: &[MeshTopologyPlan],
    objective: &CompileObjective,
    schedule_latency_ns: u64,
    schedule_peak_bytes: u64,
) -> usize {
    let mut best = 0usize;
    let mut best_figure = u64::MAX;
    for (index, candidate) in candidates.iter().enumerate() {
        let width = u64::from(candidate.width.max(1));
        let breadth = width.saturating_mul(u64::from(candidate.pipeline_depth()));
        let figure = match objective.primary() {
            ObjectiveMetric::Latency | ObjectiveMetric::ColdStart | ObjectiveMetric::Energy => {
                (schedule_latency_ns / width).saturating_add(candidate.communication_ns)
            }
            ObjectiveMetric::Throughput => {
                (schedule_latency_ns / breadth).saturating_add(candidate.communication_ns)
            }
            ObjectiveMetric::PeakMemory => schedule_peak_bytes / breadth,
            ObjectiveMetric::ArtifactBytes
            | ObjectiveMetric::VariantCount
            | ObjectiveMetric::CompileWork
            | ObjectiveMetric::MeasurementWork => {
                if index == 0 {
                    0
                } else {
                    u64::MAX
                }
            }
        };
        if figure < best_figure {
            best_figure = figure;
            best = index;
        }
    }
    best
}
