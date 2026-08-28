//! The one coordinated topology a selected schedule records for a whole mesh.
//!
//! A partition states how one logical region is cut and which device holds each
//! shard. A transfer states which link carries one logical exchange, in which
//! stage, and whether it overlaps computation. Nothing here names a vendor, an
//! interconnect, or a collective library: a partition is a generic transform and
//! a link is a bandwidth with a latency.
//!
//! Validation is self-contained on purpose. An artifact decoded from bytes has
//! this plan and the mesh identity it was selected against, not the graph, so
//! every rule below is checkable from the plan alone. The rules that need the
//! logical stage - which axes may be split, which exchanges exist, what the
//! participants are - are enforced where the plan is derived.

use serde::{Deserialize, Serialize};
use vyre_foundation::logical::LogicalExchangeKind;

use crate::allocation::{DevicePeak, DeviceSlot};
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::identity::{ArtifactNodeId, Digest};
use crate::mesh::MeshFacts;

/// Schema version of the mesh topology plan carried inside one artifact.
pub const MESH_TOPOLOGY_VERSION: u16 = 1;

/// How one logical region is cut across the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionKind {
    /// Every device holds the whole region.
    Replicated,
    /// Independent points are split across devices.
    Data,
    /// A spatial domain is split into subdomains, one per device.
    Spatial,
    /// A combined axis is split, and the partial results are combined by an exchange.
    Reduction,
    /// Successive stages of one dependency chain are placed on successive devices.
    Pipeline,
    /// A sequence axis is split into ordered segments.
    Sequence,
    /// Points are routed to devices by a data-dependent assignment.
    Routed,
}

impl PartitionKind {
    /// Whether this kind splits an axis rather than replicating the region.
    #[must_use]
    pub const fn splits_axis(self) -> bool {
        match self {
            Self::Replicated | Self::Pipeline => false,
            Self::Data | Self::Spatial | Self::Reduction | Self::Sequence | Self::Routed => true,
        }
    }

    /// Whether this kind places the whole region on one device.
    #[must_use]
    pub const fn one_device(self) -> bool {
        matches!(self, Self::Pipeline)
    }

    /// Stable neutral name of this kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Replicated => "replicated",
            Self::Data => "data",
            Self::Spatial => "spatial",
            Self::Reduction => "reduction",
            Self::Pipeline => "pipeline",
            Self::Sequence => "sequence",
            Self::Routed => "routed",
        }
    }
}

/// One shard of one region, on one device.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardAssignment {
    /// Shard index within its partition.
    pub shard: u32,
    /// Device holding this shard.
    pub device: DeviceSlot,
    /// Mesh coordinate of that device.
    pub coordinate: Vec<u32>,
    /// Logical points this shard computes.
    pub points: u64,
}

/// How one logical region is placed on the mesh.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionPartition {
    /// Canonical node whose region this partition places.
    pub node: ArtifactNodeId,
    /// Generic transform this partition applies.
    pub kind: PartitionKind,
    /// Logical axis the partition splits, absent when the region is replicated.
    pub axis: Option<u32>,
    /// Logical points of the whole region.
    pub region_points: u64,
    /// Shards in index order.
    pub shards: Vec<ShardAssignment>,
}

/// What one transfer carries.
///
/// A stated exchange is named by its index in the logical exchange list. A
/// handoff carries the values one region produces for another, which is a
/// dependence rather than an exchange the program states. A routing transfer
/// carries the contributions one shard of a region owes the shard that holds
/// their destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferOrigin {
    /// Index of the exchange the program states.
    Exchange(u32),
    /// Values one region hands to the region that consumes them.
    Handoff {
        /// Region that produces the values.
        producer: ArtifactNodeId,
        /// Region that consumes them.
        consumer: ArtifactNodeId,
    },
    /// Contributions routed between shards of one region.
    Routing {
        /// Region whose points are routed.
        node: ArtifactNodeId,
    },
}

impl TransferOrigin {
    /// Stable neutral description used in a diagnostic.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Exchange(index) => format!("exchange {index}"),
            Self::Handoff { producer, consumer } => {
                format!(
                    "the handoff from node {} to node {}",
                    producer.0, consumer.0
                )
            }
            Self::Routing { node } => format!("the routing of node {}", node.0),
        }
    }
}

/// One payload carried over one link, in one stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferAssignment {
    /// What this transfer carries.
    pub origin: TransferOrigin,
    /// Exchange semantics, recorded so a decoded plan needs no graph to read it.
    pub kind: LogicalExchangeKind,
    /// Mesh link index the bytes travel over.
    pub link: u32,
    /// Device the bytes leave.
    pub from: DeviceSlot,
    /// Device the bytes arrive at.
    pub to: DeviceSlot,
    /// Bytes this transfer moves.
    pub bytes: u64,
    /// Submission stage this transfer belongs to.
    pub stage: u32,
    /// Whether this transfer runs concurrently with computation of its stage.
    pub overlaps: bool,
}

/// Parallel width a placed region set implies.
///
/// A region cut along an axis contributes its shard count. A replicated or
/// pipelined region is computed whole on the device that holds it, so it
/// contributes one whatever its shard count. The width is the smallest of
/// those, because a submission runs no wider than its narrowest region.
#[must_use]
pub fn implied_width(partitions: &[RegionPartition]) -> u32 {
    partitions
        .iter()
        .map(|partition| {
            if partition.kind.splits_axis() {
                u32::try_from(partition.shards.len()).unwrap_or(u32::MAX)
            } else {
                1
            }
        })
        .min()
        .unwrap_or(1)
}

/// Every placement and communication decision one selected schedule made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshTopologyPlan {
    /// Plan schema version.
    pub version: u16,
    /// Identity of the mesh facts this plan was selected against.
    pub mesh: Digest,
    /// One partition per placed region, in node order.
    pub partitions: Vec<RegionPartition>,
    /// Device a single-device submission of this plan binds to.
    ///
    /// A plan that places no region still runs on one device, so the anchor is
    /// stated rather than derived from the placement.
    pub anchor: DeviceSlot,
    /// Transfers in stage order.
    pub transfers: Vec<TransferAssignment>,
    /// Smallest shard count any placed region is cut into.
    ///
    /// One means at least one region is computed whole on one device, which
    /// bounds what the placement can shorten.
    pub width: u32,
    /// Device nanoseconds this topology spends moving bytes.
    pub communication_ns: u64,
}

impl MeshTopologyPlan {
    /// The topology of a schedule that places everything on one device.
    #[must_use]
    pub fn single_device(
        mesh: Digest,
        anchor: DeviceSlot,
        partitions: Vec<RegionPartition>,
    ) -> Self {
        Self {
            version: MESH_TOPOLOGY_VERSION,
            mesh,
            anchor,
            partitions,
            transfers: Vec::new(),
            width: 1,
            communication_ns: 0,
        }
    }

    /// Devices a submission binds one payload to, in slot order.
    ///
    /// This is every device the topology places work on, and the anchor alone for
    /// a plan that allocates nothing.
    #[must_use]
    pub fn submission_devices(&self) -> Vec<DeviceSlot> {
        let devices = self.devices();
        if devices.is_empty() {
            return vec![self.anchor];
        }
        devices
    }

    /// Devices this topology places work on, in slot order.
    #[must_use]
    pub fn devices(&self) -> Vec<DeviceSlot> {
        let mut devices = self
            .partitions
            .iter()
            .flat_map(|partition| partition.shards.iter().map(|shard| shard.device))
            .collect::<Vec<_>>();
        devices.sort_unstable();
        devices.dedup();
        devices
    }

    /// Number of submission stages this topology records.
    #[must_use]
    pub fn stage_count(&self) -> u32 {
        self.transfers
            .iter()
            .map(|transfer| transfer.stage + 1)
            .max()
            .unwrap_or(0)
    }

    /// Devices holding consecutive stages of one pipelined submission.
    ///
    /// A cut region distributes its own points, which `width` states. A pipeline
    /// distributes whole regions instead: one device runs a later region of one
    /// submission while another runs an earlier region of the next, and each
    /// device holds only the values its own regions need. This is that count, and
    /// one when no region is pipelined.
    #[must_use]
    pub fn pipeline_depth(&self) -> u32 {
        let mut devices = self
            .partitions
            .iter()
            .filter(|partition| partition.kind.one_device())
            .flat_map(|partition| partition.shards.iter().map(|shard| shard.device))
            .collect::<Vec<_>>();
        devices.sort_unstable();
        devices.dedup();
        u32::try_from(devices.len()).unwrap_or(u32::MAX).max(1)
    }

    /// A device holding fewer bytes than its share of the plan is a capacity
    /// failure naming the device and both figures.
    ///
    /// # Errors
    ///
    /// Returns when a device of this topology reports a capacity below
    /// `peak_bytes` for that device, or holds a region while the mesh omits it.
    pub fn verify_capacity(
        &self,
        mesh: &MeshFacts,
        peaks: &[DevicePeak],
    ) -> Result<(), CompileError> {
        for peak in peaks {
            let Some(device) = mesh.device(peak.device) else {
                return Err(failure(
                    CompilerFailureKind::InvalidMeshTopology,
                    "artifact.allocation.device_peaks",
                    format!(
                        "the plan places {} bytes on device {} which the mesh omits",
                        peak.peak_bytes, peak.device.0
                    ),
                    "place work only on devices the authenticated mesh states",
                ));
            };
            if device.memory_capacity_bytes > 0 && peak.peak_bytes > device.memory_capacity_bytes {
                return Err(failure(
                    CompilerFailureKind::MeshCapacityExceeded,
                    format!("artifact.allocation.device_peaks[{}]", peak.device.0),
                    format!(
                        "device {} holds {} bytes and its share of the plan needs {}",
                        peak.device.0, device.memory_capacity_bytes, peak.peak_bytes
                    ),
                    "cut the work further, place it on a device with more memory, or reduce the graph",
                ));
            }
        }
        Ok(())
    }

    /// Reject a topology a consumer could not submit.
    ///
    /// # Errors
    ///
    /// Returns when the schema version is not current, a partition names no
    /// shard, a shard index repeats, a shard has no points, shard points do not
    /// account for the region, an axis is stated for a partition that cuts none
    /// or missing for one that cuts one, a partition that is not cut records a
    /// shard short of the region, a pipeline partition records more than one
    /// shard, a routed partition records no routing transfer, one device appears
    /// at two coordinates, a transfer moves no bytes, a transfer connects a
    /// device to itself, stages are not contiguous from zero, two overlapping
    /// transfers share one link in one stage, two collectives with intersecting
    /// participants share one stage, point-to-point transfers of one stage form
    /// a cycle, the recorded width is zero, or communication is priced where
    /// nothing is transferred.
    pub fn validate(&self) -> Result<(), CompileError> {
        if self.version != MESH_TOPOLOGY_VERSION {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "topology.version",
                format!(
                    "mesh topology schema {} is unsupported; expected {MESH_TOPOLOGY_VERSION}",
                    self.version
                ),
                "recompile the graph instead of reinterpreting a stored topology",
            ));
        }
        if self.width == 0 {
            return Err(invalid(
                "topology.width",
                "the topology records no parallel width",
                "record the smallest shard count any placed region is cut into, at least one",
            ));
        }
        if self.transfers.is_empty() && self.communication_ns != 0 {
            return Err(invalid(
                "topology.communication_ns",
                format!(
                    "the topology prices {} nanoseconds of communication and records no transfer",
                    self.communication_ns
                ),
                "price only the transfers the topology records",
            ));
        }
        let recorded = implied_width(&self.partitions);
        if self.width != recorded {
            return Err(invalid(
                "topology.width",
                format!(
                    "the topology records width {} and its partitions are cut {recorded} ways",
                    self.width
                ),
                "record the smallest shard count any placed region is cut into",
            ));
        }
        let placed = self.devices();
        if !placed.is_empty() && !placed.contains(&self.anchor) {
            return Err(invalid(
                "topology.anchor",
                format!(
                    "the anchor device {} holds no shard of this topology",
                    self.anchor.0
                ),
                "anchor the submission on a device the topology places work on",
            ));
        }
        let mut coordinate_of = std::collections::BTreeMap::<DeviceSlot, &[u32]>::new();
        let mut nodes = std::collections::BTreeSet::new();
        for (index, partition) in self.partitions.iter().enumerate() {
            let path = format!("topology.partitions[{index}]");
            if !nodes.insert(partition.node) {
                return Err(invalid(
                    format!("{path}.node"),
                    format!("two partitions place node {}", partition.node.0),
                    "record one partition per canonical node",
                ));
            }
            if partition.shards.is_empty() {
                return Err(invalid(
                    format!("{path}.shards"),
                    format!("the partition of node {} has no shard", partition.node.0),
                    "record one shard per device the region runs on",
                ));
            }
            if partition.kind.splits_axis() == partition.axis.is_none() {
                return Err(invalid(
                    format!("{path}.axis"),
                    format!(
                        "a {} partition {} a split axis",
                        partition.kind.as_str(),
                        if partition.kind.splits_axis() {
                            "states no"
                        } else {
                            "states"
                        }
                    ),
                    "state the split axis for every partition that cuts one, and none otherwise",
                ));
            }
            if partition.region_points == 0 {
                return Err(invalid(
                    format!("{path}.region_points"),
                    format!("the region of node {} has no points", partition.node.0),
                    "record the logical point bound of the region this partition places",
                ));
            }
            let mut total = 0u64;
            for (shard_index, shard) in partition.shards.iter().enumerate() {
                let shard_path = format!("{path}.shards[{shard_index}]");
                if shard.shard as usize != shard_index {
                    return Err(invalid(
                        format!("{shard_path}.shard"),
                        format!(
                            "shard index {} is recorded in position {shard_index}",
                            shard.shard
                        ),
                        "record shards in index order starting at zero",
                    ));
                }
                if shard.points == 0 {
                    return Err(invalid(
                        format!("{shard_path}.points"),
                        format!(
                            "shard {} of node {} computes no point",
                            shard.shard, partition.node.0
                        ),
                        "give every shard work, or record fewer shards",
                    ));
                }
                if let Some(recorded) = coordinate_of.get(&shard.device) {
                    if *recorded != shard.coordinate.as_slice() {
                        return Err(invalid(
                            format!("{shard_path}.coordinate"),
                            format!(
                                "device {} is recorded at two mesh coordinates",
                                shard.device.0
                            ),
                            "record one coordinate per device across the whole topology",
                        ));
                    }
                } else {
                    coordinate_of.insert(shard.device, shard.coordinate.as_slice());
                }
                if partition.kind.splits_axis() {
                    total = total.checked_add(shard.points).ok_or_else(|| {
                        invalid(
                            format!("{shard_path}.points"),
                            "shard points overflow the region bound",
                            "record shard point counts that sum to the region bound",
                        )
                    })?;
                } else if shard.points != partition.region_points {
                    return Err(invalid(
                        format!("{shard_path}.points"),
                        format!(
                            "a {} shard computes {} of {} region points",
                            partition.kind.as_str(),
                            shard.points,
                            partition.region_points
                        ),
                        "give every shard that is not cut the whole region",
                    ));
                }
            }
            if partition.kind.one_device() && partition.shards.len() != 1 {
                return Err(invalid(
                    format!("{path}.shards"),
                    format!(
                        "a {} partition of node {} records {} shards",
                        partition.kind.as_str(),
                        partition.node.0,
                        partition.shards.len()
                    ),
                    "place a pipeline stage on exactly one device",
                ));
            }
            if partition.kind == PartitionKind::Routed
                && !self.transfers.iter().any(|transfer| {
                    transfer.origin
                        == TransferOrigin::Routing {
                            node: partition.node,
                        }
                })
            {
                return Err(invalid(
                    format!("{path}.kind"),
                    format!(
                        "node {} is routed and the topology records no routing transfer for it",
                        partition.node.0
                    ),
                    "record the routing transfers a data-dependent assignment needs",
                ));
            }
            if partition.kind.splits_axis() && total != partition.region_points {
                return Err(invalid(
                    format!("{path}.shards"),
                    format!(
                        "shards of node {} compute {total} of {} region points",
                        partition.node.0, partition.region_points
                    ),
                    "cover every logical point exactly once, including the ragged tail",
                ));
            }
        }
        self.validate_transfers(&coordinate_of)
    }

    fn validate_transfers(
        &self,
        coordinate_of: &std::collections::BTreeMap<DeviceSlot, &[u32]>,
    ) -> Result<(), CompileError> {
        let stages = self.stage_count();
        let mut seen_stages = std::collections::BTreeSet::new();
        for (index, transfer) in self.transfers.iter().enumerate() {
            let path = format!("topology.transfers[{index}]");
            if transfer.bytes == 0 {
                return Err(invalid(
                    format!("{path}.bytes"),
                    format!("the transfer for {} moves no byte", transfer.origin.label()),
                    "record the exact payload bytes, or omit the transfer",
                ));
            }
            if transfer.from == transfer.to {
                return Err(invalid(
                    format!("{path}.to"),
                    format!("a transfer sends device {} to itself", transfer.from.0),
                    "record transfers between distinct devices",
                ));
            }
            for (field, slot) in [("from", transfer.from), ("to", transfer.to)] {
                if !coordinate_of.contains_key(&slot) {
                    return Err(invalid(
                        format!("{path}.{field}"),
                        format!("a transfer names device {} which holds no shard", slot.0),
                        "place work on every device the topology communicates with",
                    ));
                }
            }
            if transfer.stage >= stages {
                return Err(invalid(
                    format!("{path}.stage"),
                    format!(
                        "stage {} is outside the {stages} recorded stages",
                        transfer.stage
                    ),
                    "record stages contiguously from zero",
                ));
            }
            seen_stages.insert(transfer.stage);
            if index > 0 && self.transfers[index - 1].stage > transfer.stage {
                return Err(invalid(
                    format!("{path}.stage"),
                    "transfers are not recorded in stage order",
                    "record transfers in ascending stage order",
                ));
            }
        }
        for stage in 0..stages {
            if !seen_stages.contains(&stage) {
                return Err(invalid(
                    "topology.transfers",
                    format!("stage {stage} carries no transfer"),
                    "record stages contiguously from zero",
                ));
            }
            self.validate_stage(stage)?;
        }
        Ok(())
    }

    /// Reject one stage that cannot make progress.
    ///
    /// Two collectives whose participants intersect need one global order, and a
    /// stage states none, so both devices could wait in opposite orders. A cycle
    /// of point-to-point transfers is the same wait expressed directly.
    fn validate_stage(&self, stage: u32) -> Result<(), CompileError> {
        let transfers = self
            .transfers
            .iter()
            .enumerate()
            .filter(|(_, transfer)| transfer.stage == stage);
        let mut links = std::collections::BTreeSet::new();
        let mut collectives =
            Vec::<(TransferOrigin, std::collections::BTreeSet<DeviceSlot>)>::new();
        let mut edges = Vec::<(DeviceSlot, DeviceSlot)>::new();
        for (index, transfer) in transfers {
            if transfer.overlaps && !links.insert(transfer.link) {
                return Err(invalid(
                    format!("topology.transfers[{index}].overlaps"),
                    format!(
                        "two overlapping transfers of stage {stage} share link {}",
                        transfer.link
                    ),
                    "overlap a transfer only when its link carries nothing else in that stage",
                ));
            }
            if transfer.kind == LogicalExchangeKind::PointToPoint {
                edges.push((transfer.from, transfer.to));
                continue;
            }
            let participants = self
                .transfers
                .iter()
                .filter(|other| other.stage == stage && other.origin == transfer.origin)
                .flat_map(|other| [other.from, other.to])
                .collect::<std::collections::BTreeSet<_>>();
            for (other_origin, other) in &collectives {
                if *other_origin != transfer.origin && !other.is_disjoint(&participants) {
                    return Err(invalid(
                        format!("topology.transfers[{index}].stage"),
                        format!(
                            "collectives for {} and {} share a device in stage {stage}",
                            other_origin.label(),
                            transfer.origin.label()
                        ),
                        "order two collectives over the same devices in separate stages",
                    ));
                }
            }
            if !collectives
                .iter()
                .any(|(origin, _)| *origin == transfer.origin)
            {
                collectives.push((transfer.origin, participants));
            }
        }
        reject_cycle(&edges, stage)
    }
}

/// Reject a wait cycle among the point-to-point transfers of one stage.
fn reject_cycle(edges: &[(DeviceSlot, DeviceSlot)], stage: u32) -> Result<(), CompileError> {
    let mut nodes = edges
        .iter()
        .flat_map(|(from, to)| [*from, *to])
        .collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    let mut outstanding = nodes
        .iter()
        .map(|node| (*node, edges.iter().filter(|(_, to)| to == node).count()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut ready = outstanding
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(node, _)| *node)
        .collect::<Vec<_>>();
    let mut settled = 0usize;
    while let Some(node) = ready.pop() {
        settled += 1;
        for (from, to) in edges.iter().filter(|(from, _)| *from == node) {
            let _ = from;
            if let Some(count) = outstanding.get_mut(to) {
                *count -= 1;
                if *count == 0 {
                    ready.push(*to);
                }
            }
        }
    }
    if settled != nodes.len() {
        return Err(invalid(
            "topology.transfers",
            format!("point-to-point transfers of stage {stage} wait in a cycle"),
            "order the transfers of one stage so no device waits on its own sender",
        ));
    }
    Ok(())
}

fn invalid(
    path: impl Into<String>,
    message: impl Into<String>,
    fix: impl Into<String>,
) -> CompileError {
    failure(CompilerFailureKind::InvalidMeshTopology, path, message, fix)
}
