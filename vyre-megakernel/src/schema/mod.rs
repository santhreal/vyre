//! The canonical artifact schema: every versioned record, and the immutable
//! container that frames them.

mod geometry;
mod plan;
mod records;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::allocation::AllocationPlan;
use crate::error::{failure, serialization_failure, CompileError, CompilerFailureKind};
use crate::frame;
use crate::identity::{
    ArtifactNodeId, ArtifactValueId, DependencyEdge, DependencyEndpoint, Digest, FusionGroupId,
};
use crate::mesh::MeshTopologyPlan;

pub use geometry::{BarrierPhaseRecord, EntryPersistence, GeometryRecord, LaunchResourceIntent};
pub use plan::{ExecutionMode, FrontierTopology, NumericRecord, PlanMeasurement, SelectedPlan};
pub use records::{
    AbiAccess, ArtifactAbi, BarrierRecord, EntryAbiRecord, EntryResourceBinding, FusionRecord,
    FusionRejection, MaterializationReason, MaterializationRecord, NodeRecord, Provenance,
    ResourceAbiRecord, ResourceEnvelope, ResourceLifetime, ResourceNameCollision, ResourceRecord,
};

/// Current canonical artifact schema.
pub const ARTIFACT_SCHEMA_VERSION: u16 = 18;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactPayload {
    pub(crate) schema_version: u16,
    pub(crate) nodes: Vec<NodeRecord>,
    pub(crate) dependencies: Vec<DependencyEdge>,
    pub(crate) selected_plan: SelectedPlan,
    pub(crate) abi: ArtifactAbi,
    pub(crate) resources: Vec<ResourceRecord>,
    pub(crate) resource_envelope: ResourceEnvelope,
    pub(crate) geometry: Vec<GeometryRecord>,
    pub(crate) allocation: AllocationPlan,
    pub(crate) topology: MeshTopologyPlan,
    pub(crate) provenance: Provenance,
}

/// Versioned immutable canonical whole-program compiler result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifact {
    pub(crate) payload: ArtifactPayload,
    pub(crate) digest: Digest,
}

impl Artifact {
    /// Artifact schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.payload.schema_version
    }

    /// Canonical node records.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRecord] {
        &self.payload.nodes
    }

    /// Canonical typed dependency edges.
    #[must_use]
    pub fn dependencies(&self) -> &[DependencyEdge] {
        &self.payload.dependencies
    }

    /// Canonical fusion groups.
    #[must_use]
    pub fn fusion(&self) -> &[FusionRecord] {
        &self.payload.selected_plan.fusion
    }

    /// Canonical barrier boundaries.
    #[must_use]
    pub fn barriers(&self) -> &[BarrierRecord] {
        &self.payload.selected_plan.barriers
    }

    /// Canonical resource records.
    #[must_use]
    pub fn resources(&self) -> &[ResourceRecord] {
        &self.payload.resources
    }

    /// Canonical value identity for every resource name.
    ///
    /// A resource name carries descriptor binding identity, so an artifact that
    /// gave one name to two values would let a consumer bind the wrong buffer.
    /// Detection lives here because the resource set is the owner of value
    /// identity: a consumer that built its own lookup would have to repeat the
    /// check, and the two that did disagreed about whether it was an error.
    pub fn canonical_value_by_name(
        &self,
    ) -> Result<BTreeMap<&str, ArtifactValueId>, ResourceNameCollision> {
        let mut by_name = BTreeMap::<&str, ArtifactValueId>::new();
        for resource in self.resources() {
            if let Some(previous) = by_name.insert(resource.name.as_str(), resource.value) {
                if previous != resource.value {
                    return Err(ResourceNameCollision {
                        name: resource.name.clone(),
                        first: previous,
                        second: resource.value,
                    });
                }
            }
        }
        Ok(by_name)
    }

    /// Checked aggregate resource envelope.
    #[must_use]
    pub const fn resource_envelope(&self) -> ResourceEnvelope {
        self.payload.resource_envelope
    }

    /// Program-declared neutral geometry records.
    #[must_use]
    pub fn geometry(&self) -> &[GeometryRecord] {
        &self.payload.geometry
    }

    /// Every physical allocation and layout decision the selected schedule made.
    #[must_use]
    pub const fn allocation(&self) -> &AllocationPlan {
        &self.payload.allocation
    }

    /// The one coordinated topology the selected schedule placed on the mesh.
    #[must_use]
    pub const fn topology(&self) -> &MeshTopologyPlan {
        &self.payload.topology
    }

    /// Reject recorded geometry or physical storage a consumer could not submit.
    ///
    /// One record per node, every predecessor known, and every field a launch
    /// reads present. Decoded bytes reach this check before any consumer reads a
    /// launch out of them, because a partial geometry set is the one shape a
    /// consumer would fill in itself.
    ///
    /// # Errors
    ///
    /// Returns when a node has no geometry, two records name one node, a record
    /// is unlaunchable, the allocation plan is not bindable, or a placement and
    /// the resource row for one value disagree on lifetime, bytes or live range.
    pub fn validate_geometry(&self) -> Result<(), CompileError> {
        let mut recorded = BTreeMap::new();
        for record in self.geometry() {
            record.validate()?;
            if recorded.insert(record.node, record).is_some() {
                return Err(failure(
                    CompilerFailureKind::MalformedArtifact,
                    "artifact.body.geometry",
                    format!("node {} carries two geometry records", record.node.0),
                    "record one selected geometry per canonical node",
                ));
            }
        }
        for node in self.nodes() {
            if !recorded.contains_key(&node.id) {
                return Err(failure(
                    CompilerFailureKind::MalformedArtifact,
                    "artifact.body.geometry",
                    format!("node {} carries no selected geometry", node.id.0),
                    "compile the artifact with a compiler that records geometry for every node",
                ));
            }
        }
        for record in self.geometry() {
            for predecessor in &record.predecessors {
                if !recorded.contains_key(predecessor) {
                    return Err(failure(
                        CompilerFailureKind::MalformedArtifact,
                        "artifact.body.geometry",
                        format!(
                            "node {} depends on entry point {} the artifact does not carry",
                            record.node.0, predecessor.0
                        ),
                        "record the entry-point dependency order the selected plan implies",
                    ));
                }
            }
        }
        let mut position = BTreeMap::new();
        for (index, record) in self.geometry().iter().enumerate() {
            position.insert(record.node, index);
        }
        let mut group_of = BTreeMap::new();
        for group in self.fusion() {
            for member in &group.members {
                group_of.insert(*member, (group.id, group.stage));
            }
        }
        for (index, record) in self.geometry().iter().enumerate() {
            for predecessor in &record.predecessors {
                if position[predecessor] >= index {
                    return Err(failure(
                        CompilerFailureKind::MalformedArtifact,
                        "artifact.body.geometry",
                        format!(
                            "node {} is recorded before entry point {} it depends on",
                            record.node.0, predecessor.0
                        ),
                        "record the geometry set in the dependency order the selected plan implies",
                    ));
                }
                match (group_of.get(&record.node), group_of.get(predecessor)) {
                    (Some((consumer, _)), Some((producer, _))) if consumer == producer => {}
                    (Some((_, consumer_stage)), Some((_, producer_stage)))
                        if producer_stage < consumer_stage => {}
                    (Some(_), Some(_)) => {
                        return Err(failure(
                            CompilerFailureKind::MalformedArtifact,
                            "artifact.body.selected_plan",
                            format!(
                                "the group of node {} runs no later than the group producing entry point {}",
                                record.node.0, predecessor.0
                            ),
                            "record a later stage for every group that consumes another group's value",
                        ));
                    }
                    _ => {
                        return Err(failure(
                            CompilerFailureKind::MalformedArtifact,
                            "artifact.body.selected_plan",
                            format!(
                                "node {} or entry point {} belongs to no selected fusion group",
                                record.node.0, predecessor.0
                            ),
                            "assign every canonical node to one selected fusion group",
                        ));
                    }
                }
            }
        }
        self.allocation().validate()?;
        let rows: BTreeMap<ArtifactValueId, &ResourceRecord> = self
            .resources()
            .iter()
            .map(|resource| (resource.value, resource))
            .collect();
        // A partitioned value is placed once per device that holds a share, so
        // the row total is the sum of its placements, not any one of them.
        let mut held = BTreeMap::<ArtifactValueId, u64>::new();
        for region in &self.allocation().regions {
            for placement in &region.placements {
                let Some(row) = rows.get(&placement.value) else {
                    return Err(malformed_allocation(format!(
                        "allocation places value {} the artifact does not carry",
                        placement.value.0
                    )));
                };
                if row.lifetime != placement.lifetime {
                    return Err(malformed_allocation(format!(
                        "placement of value {} disagrees with its resource lifetime",
                        placement.value.0
                    )));
                }
                let placed = held.entry(placement.value).or_insert(0);
                *placed = placed.saturating_add(placement.bytes);
                if (row.first_stage, row.last_stage)
                    != (placement.first_stage, placement.last_stage)
                {
                    return Err(malformed_allocation(format!(
                        "placement of value {} states a live range the resource row does not",
                        placement.value.0
                    )));
                }
            }
        }
        for row in self.resources() {
            if row.byte_count > 0 && self.allocation().placement(row.value).is_none() {
                return Err(malformed_allocation(format!(
                    "value {} occupies bytes and the allocation plan places it nowhere",
                    row.value.0
                )));
            }
            let placed = held.get(&row.value).copied().unwrap_or(0);
            if row.byte_count > 0 && placed != row.byte_count {
                return Err(malformed_allocation(format!(
                    "the placements of value {} hold {placed} bytes and its resource row states {}",
                    row.value.0, row.byte_count
                )));
            }
        }
        self.topology().validate()?;
        // A device holding bytes the topology does not place is storage nothing
        // submits. The converse is legal: a device computing shards of a region
        // whose values are all caller-bound holds no artifact allocation.
        for peak in &self.allocation().device_peaks {
            if !self
                .topology()
                .devices()
                .iter()
                .any(|device| *device == peak.device)
            {
                return Err(malformed_allocation(format!(
                    "device {} holds bytes and the topology places no work on it",
                    peak.device.0
                )));
            }
        }
        let last_stage = self
            .fusion()
            .iter()
            .map(|group| group.stage)
            .max()
            .unwrap_or(0);
        validate_barrier_records(self.barriers(), self.dependencies(), &group_of, last_stage)?;
        Ok(())
    }

    /// Canonical materialization records.
    #[must_use]
    pub fn materializations(&self) -> &[MaterializationRecord] {
        &self.payload.selected_plan.materializations
    }

    /// Compiler-selected plan with recorded search bounds.
    #[must_use]
    pub const fn selected_plan(&self) -> &SelectedPlan {
        &self.payload.selected_plan
    }

    /// Canonical neutral resource and entry ABI.
    #[must_use]
    pub const fn abi(&self) -> &ArtifactAbi {
        &self.payload.abi
    }

    /// Deterministic artifact provenance.
    #[must_use]
    pub const fn provenance(&self) -> &Provenance {
        &self.payload.provenance
    }

    /// Canonical artifact content identity.
    #[must_use]
    pub const fn digest(&self) -> Digest {
        self.digest
    }

    /// Encode canonical versioned bytes with an authenticated body.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CompileError> {
        Ok(encode_payload(&self.payload)?.bytes)
    }

    /// Decode, authenticate, and reject non-canonical or incompatible bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let decoded = frame::ARTIFACT.decode(bytes)?;
        let payload: ArtifactPayload = serde_json::from_slice(decoded.body).map_err(|error| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                "artifact.body",
                error.to_string(),
                "supply a canonical body emitted by this crate",
            )
        })?;
        if payload.schema_version != decoded.version {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "artifact.body.schema_version",
                "body schema disagrees with framing schema",
                "recompile instead of rewriting artifact framing",
            ));
        }
        let canonical = serde_json::to_vec(&payload).map_err(serialization_failure)?;
        if canonical != decoded.body {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "artifact.body",
                "artifact body is valid JSON but not canonical JSON",
                "use the canonical bytes emitted by Artifact::to_bytes",
            ));
        }
        let artifact = Self {
            payload,
            digest: Digest(decoded.digest),
        };
        artifact.payload.selected_plan.validate()?;
        artifact.validate_geometry()?;
        // A compiled artifact cannot carry a duplicate resource name because graph
        // value names are unique, so this is a check on decoded bytes rather than on
        // this crate's own output. Refusing here keeps every consumer's name lookup
        // total: without it a tampered artifact resolves a descriptor binding to
        // whichever of the two values the consumer's map happened to keep.
        artifact.canonical_value_by_name().map_err(|collision| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                "artifact.body.resources",
                collision.to_string(),
                "emit one resource record per canonical value name",
            )
        })?;
        Ok(artifact)
    }
}

/// Refusal of a decoded plan that contradicts the resource rows beside it.
fn malformed_allocation(message: String) -> CompileError {
    failure(
        CompilerFailureKind::MalformedArtifact,
        "artifact.body.allocation",
        message,
        "record one placement per value the resource set carries, stating its lifetime, bytes and \
         live range",
    )
}

/// Admit the stage boundaries a decoded artifact carries.
///
/// A barrier record is what a backend submits between two stages, so a record
/// that states the wrong pair of stages, an edge that does not cross it, or that
/// omits an edge which does, is a schedule the device would run without the
/// ordering the plan was selected under. The boundaries are derived from one
/// topological stage assignment, so they cover stages 1 through the last one in
/// order, each after its immediate predecessor.
fn validate_barrier_records(
    barriers: &[BarrierRecord],
    dependencies: &[DependencyEdge],
    group_of: &BTreeMap<ArtifactNodeId, (FusionGroupId, u32)>,
    last_stage: u32,
) -> Result<(), CompileError> {
    let malformed = |message: String| {
        failure(
            CompilerFailureKind::MalformedArtifact,
            "artifact.body.selected_plan.barriers",
            message,
            "record one boundary per consecutive stage pair, stating every dependency edge that \
             crosses it and no other",
        )
    };
    let expected = usize::try_from(last_stage).unwrap_or(usize::MAX);
    if barriers.len() != expected {
        return Err(malformed(format!(
            "the plan runs {} stages and records {} boundaries",
            last_stage + 1,
            barriers.len()
        )));
    }
    let mut crossing = BTreeMap::<u32, Vec<u32>>::new();
    for (index, edge) in dependencies.iter().enumerate() {
        let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        else {
            continue;
        };
        let (Some((_, from_stage)), Some((_, to_stage))) = (group_of.get(&from), group_of.get(&to))
        else {
            return Err(malformed(format!(
                "dependency edge {index} connects a node that belongs to no selected fusion group"
            )));
        };
        if from_stage < to_stage {
            let id = u32::try_from(index)
                .map_err(|_| malformed("dependency edge identity exceeds u32".to_string()))?;
            crossing.entry(*to_stage).or_default().push(id);
        }
    }
    for (position, record) in barriers.iter().enumerate() {
        let after = u32::try_from(position + 1)
            .map_err(|_| malformed("stage identity exceeds u32".to_string()))?;
        if record.after_stage != after || record.before_stage + 1 != record.after_stage {
            return Err(malformed(format!(
                "boundary {position} admits stage {} after stage {}, and the derived order admits \
                 stage {after} after stage {}",
                record.after_stage,
                record.before_stage,
                after - 1
            )));
        }
        let derived = crossing.remove(&after).unwrap_or_default();
        if record.dependencies != derived {
            return Err(malformed(format!(
                "boundary into stage {after} states dependency edges {:?} and the edges crossing \
                 it are {derived:?}",
                record.dependencies
            )));
        }
    }
    if let Some((stage, edges)) = crossing.pop_first() {
        return Err(malformed(format!(
            "dependency edges {edges:?} cross into stage {stage} and no boundary admits them"
        )));
    }
    Ok(())
}

pub(crate) fn encode_payload(payload: &ArtifactPayload) -> Result<frame::Framed, CompileError> {
    let body = serde_json::to_vec(payload).map_err(serialization_failure)?;
    frame::ARTIFACT.encode(payload.schema_version, &body)
}
// Inline: the tamper this suite needs is a re-framed `ArtifactPayload`, and both
// `ArtifactPayload` and `frame::ARTIFACT` are crate-private. Recomputing the frame
// from its documented layout in an integration test would put a second copy of the
// digest domain outside the module that owns it.
#[cfg(test)]
mod tests;
