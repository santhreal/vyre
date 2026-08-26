//! The canonical artifact schema: every versioned record, and the immutable
//! container that frames them.

mod geometry;
mod plan;
mod records;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{failure, serialization_failure, CompileError, CompilerFailureKind};
use crate::frame;
use crate::identity::{ArtifactValueId, DependencyEdge, Digest};

pub use geometry::{
    BarrierPhaseRecord, EntryPersistence, GeometryRecord, LaunchResourceIntent, WorkspacePlan,
    WorkspaceRegion, WORKSPACE_REGION_ALIGNMENT,
};
pub use plan::{ExecutionMode, PlanMeasurement, SelectedPlan};
pub use records::{
    AbiAccess, ArtifactAbi, BarrierRecord, EntryAbiRecord, EntryResourceBinding, FusionRecord,
    FusionRejection, MaterializationReason, MaterializationRecord, NodeRecord, Provenance,
    ResourceAbiRecord, ResourceEnvelope, ResourceLifetime, ResourceNameCollision, ResourceRecord,
};

/// Current canonical artifact schema.
pub const ARTIFACT_SCHEMA_VERSION: u16 = 11;

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
    pub(crate) workspace: WorkspacePlan,
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

    /// Exact workspace allocation and cross-entry offsets.
    #[must_use]
    pub const fn workspace(&self) -> &WorkspacePlan {
        &self.payload.workspace
    }

    /// Reject recorded geometry or workspace a consumer could not submit.
    ///
    /// One record per node, every predecessor and workspace value known, and
    /// every field a launch reads present. Decoded bytes reach this check
    /// before any consumer reads a launch out of them, because a partial
    /// geometry set is the one shape a consumer would fill in itself.
    ///
    /// # Errors
    ///
    /// Returns when a node has no geometry, two records name one node, a
    /// record is unlaunchable, or the workspace plan names a value the
    /// artifact does not carry.
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
        for (index, group) in self.fusion().iter().enumerate() {
            for member in &group.members {
                group_of.insert(*member, index);
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
                    (Some(consumer), Some(producer)) if producer <= consumer => {}
                    (Some(_), Some(_)) => {
                        return Err(failure(
                            CompilerFailureKind::MalformedArtifact,
                            "artifact.body.selected_plan",
                            format!(
                                "the fusion group of node {} is planned before the group producing entry point {}",
                                record.node.0, predecessor.0
                            ),
                            "order the selected fusion groups in the dependency order they imply",
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
        self.workspace().validate()?;
        let lifetimes: BTreeMap<ArtifactValueId, ResourceLifetime> = self
            .resources()
            .iter()
            .map(|resource| (resource.value, resource.lifetime))
            .collect();
        for region in &self.workspace().regions {
            match lifetimes.get(&region.value) {
                Some(lifetime) if *lifetime == region.lifetime => {}
                Some(_) => {
                    return Err(failure(
                        CompilerFailureKind::MalformedArtifact,
                        "artifact.body.workspace",
                        format!(
                            "workspace region for value {} disagrees with its resource lifetime",
                            region.value.0
                        ),
                        "record the lifetime the resource set states",
                    ))
                }
                None => {
                    return Err(failure(
                        CompilerFailureKind::MalformedArtifact,
                        "artifact.body.workspace",
                        format!(
                            "workspace region names value {} the artifact does not carry",
                            region.value.0
                        ),
                        "reserve workspace only for values the resource set records",
                    ))
                }
            }
            if !recorded.contains_key(&region.first_entry)
                || !recorded.contains_key(&region.last_entry)
            {
                return Err(failure(
                    CompilerFailureKind::MalformedArtifact,
                    "artifact.body.workspace",
                    format!(
                        "workspace region for value {} names an entry point the artifact does not carry",
                        region.value.0
                    ),
                    "record the live range across entry points the selected plan implies",
                ));
            }
        }
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

pub(crate) fn encode_payload(payload: &ArtifactPayload) -> Result<frame::Framed, CompileError> {
    let body = serde_json::to_vec(payload).map_err(serialization_failure)?;
    frame::ARTIFACT.encode(payload.schema_version, &body)
}
// Inline: the tamper this suite needs is a re-framed `ArtifactPayload`, and both
// `ArtifactPayload` and `frame::ARTIFACT` are crate-private. Recomputing the frame
// from its documented layout in an integration test would put a second copy of the
// digest domain outside the module that owns it.
#[cfg(test)]
mod tests {
    //! WHY: a resource name carries descriptor binding identity, so an artifact that
    //! gives one name to two values lets a consumer bind the wrong buffer. Before
    //! this check the two consumers of the name lookup disagreed: the target
    //! compiler refused the collision while the runtime's `program_outputs` built a
    //! map that silently kept whichever value came last and projected it as the
    //! program output. Both now read `Artifact::canonical_value_by_name`, and decode
    //! refuses the bytes outright so neither has to be trusted to check.
    //!
    //! Does not catch: a collision reaching a consumer through some path that builds
    //! an `Artifact` without `from_bytes`. Nothing in the workspace does, because the
    //! compiler derives resources from graph value names and `ProgramGraph` already
    //! refuses a duplicate name, but a future constructor would need the same check.

    use super::*;
    use crate::cost::CostBreakdown;
    use crate::identity::{ArtifactNodeId, FusionGroupId};
    use crate::request::{SearchBudget, SearchWork};
    use vyre_foundation::schedule::{SchedulePhaseId, ScheduleTransform, SelectedSchedule};

    fn payload(resources: Vec<ResourceRecord>) -> ArtifactPayload {
        ArtifactPayload {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            nodes: Vec::new(),
            dependencies: Vec::new(),
            selected_plan: SelectedPlan {
                topology: crate::ExecutionTopology::Sequential,
                schedule: SelectedSchedule::synthetic(1),
                derivation: Vec::new(),
                certificate: crate::SearchCertificate::new(crate::SCHEDULE_GRAMMAR_VERSION),
                fusion: Vec::new(),
                barriers: Vec::new(),
                materializations: Vec::new(),
                candidates_explored: 1,
                search_budget: SearchBudget::new(1, 1, 0, 0, 1),
                search_work: SearchWork {
                    candidates_explored: 1,
                    ..SearchWork::default()
                },
                selection_cost: CostBreakdown::default(),
                pruned_fusions: Vec::new(),
                execution: ExecutionMode::Static,
                measurement: PlanMeasurement::Unbudgeted,
            },
            abi: ArtifactAbi {
                resources: Vec::new(),
                entries: Vec::new(),
            },
            resources,
            resource_envelope: ResourceEnvelope {
                total_bytes: 0,
                peak_live_bytes: 0,
            },
            geometry: Vec::new(),
            workspace: WorkspacePlan::default(),
            provenance: Provenance {
                source_graph: Digest([0; 32]),
                request: Digest([0; 32]),
                compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }

    fn resource(name: &str, value: u32) -> ResourceRecord {
        ResourceRecord {
            value: ArtifactValueId(value),
            name: name.to_string(),
            element_count: 1,
            byte_count: 4,
            lifetime: ResourceLifetime::Output,
            retained_predecessor: None,
            first_stage: 0,
            last_stage: 0,
        }
    }

    fn decode_payload(payload: ArtifactPayload) -> Result<Artifact, CompileError> {
        let framed = encode_payload(&payload).expect("fixture payload must frame");
        Artifact::from_bytes(&framed.bytes)
    }

    fn decode(resources: Vec<ResourceRecord>) -> Result<Artifact, CompileError> {
        decode_payload(payload(resources))
    }

    /// WHY: topology cardinality is part of the selected schedule. A zero
    /// cardinality used to deserialize successfully and reached target lowering
    /// as an execution topology that could not launch.
    #[test]
    fn selected_schedule_rejects_every_zero_topology_cardinality() {
        for topology in [
            crate::ExecutionTopology::ConcurrentQueue { queues: 0 },
            crate::ExecutionTopology::ResidentPartition {
                partitions: 0,
                mode: crate::ResidentPartitionMode::FixedSpatialMask,
            },
        ] {
            let mut invalid = payload(Vec::new());
            invalid.selected_plan.topology = topology;
            let error =
                decode_payload(invalid).expect_err("zero topology cardinality must not decode");
            assert!(
                error
                    .diagnostic
                    .location
                    .as_ref()
                    .and_then(|location| location.path.as_deref())
                    .is_some_and(|path| path.starts_with("artifact.body.selected_plan.topology")),
                "diagnostic must identify selected topology: {error}"
            );
        }
    }

    /// WHY: selected phase state and transform proof rows are authenticated
    /// together. A stale or edited phase must not reach physical lowering even
    /// when the surrounding artifact frame was recomputed.
    #[test]
    fn selected_schedule_rejects_mutated_phase_and_transform_provenance() {
        let mut valid = payload(Vec::new());
        valid
            .selected_plan
            .schedule
            .apply(ScheduleTransform::SetWorkgroup {
                phase: SchedulePhaseId(0),
                shape: [32, 1, 1],
            })
            .unwrap();

        let mut phase_mutation = valid.clone();
        phase_mutation.selected_plan.schedule.phases[0].workgroup[0] = 64;
        let phase_error =
            decode_payload(phase_mutation).expect_err("mutated phase geometry must fail replay");
        assert!(phase_error.diagnostic.message.contains("replay"));

        let mut provenance_mutation = valid;
        provenance_mutation.selected_plan.schedule.transforms[0]
            .provenance
            .inverse
            .previous_identity[0] ^= 1;
        let provenance_error = decode_payload(provenance_mutation)
            .expect_err("mutated transform provenance must fail replay");
        assert!(provenance_error.diagnostic.message.contains("evidence"));
    }

    /// WHY: duplicated candidate accounting can authenticate two incompatible
    /// answers for the same bounded search. Decode must reject the disagreement
    /// before any target consumes the schedule.
    #[test]
    fn selected_schedule_rejects_inconsistent_candidate_accounting() {
        let mut invalid = payload(Vec::new());
        invalid.selected_plan.search_work.candidates_explored = 2;
        let error =
            decode_payload(invalid).expect_err("candidate accounting disagreement must fail");
        assert!(
            error
                .diagnostic
                .message
                .contains("records 1 explored candidates"),
            "diagnostic must state both accounting owners: {error}"
        );
    }

    /// WHY: measurement samples are charged per target compilation actually
    /// performed. Unused compilation budget cannot authenticate samples for a
    /// finalist that was never compiled.
    #[test]
    fn selected_schedule_rejects_measurements_without_compiled_finalists() {
        let mut invalid = payload(Vec::new());
        invalid.selected_plan.search_budget = SearchBudget::new(1, 1, 2, 1, 1);
        invalid.selected_plan.search_work.target_compilations = 1;
        invalid.selected_plan.search_work.measurements = 2;
        let error =
            decode_payload(invalid).expect_err("samples without a compiled finalist must fail");
        assert!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|location| location.path.as_deref())
                .is_some_and(|path| path.ends_with("search_work.measurements")),
            "diagnostic must identify measurement accounting: {error}"
        );
    }

    /// WHY: a measured schedule with no launch or no elapsed device time is not
    /// measurement evidence and must not deserialize as one.
    #[test]
    fn selected_schedule_rejects_zero_measurement_evidence() {
        for measurement in [
            PlanMeasurement::Measured {
                launches: 0,
                median_ns: 1,
            },
            PlanMeasurement::Measured {
                launches: 1,
                median_ns: 0,
            },
        ] {
            let mut invalid = payload(Vec::new());
            invalid.selected_plan.measurement = measurement;
            let error =
                decode_payload(invalid).expect_err("zero measurement evidence must not decode");
            assert!(
                error
                    .diagnostic
                    .message
                    .contains("zero launch count or device time"),
                "diagnostic must identify invalid measurement evidence: {error}"
            );
        }
    }

    /// WHY: unbudgeted and untimed plans cannot authenticate device samples
    /// that their measurement state states were never used.
    #[test]
    fn selected_schedule_rejects_samples_on_every_unmeasured_state() {
        for measurement in [PlanMeasurement::Unbudgeted, PlanMeasurement::UntimedDevice] {
            let mut invalid = payload(Vec::new());
            invalid.selected_plan.search_budget = SearchBudget::new(1, 1, 1, 1, 1);
            invalid.selected_plan.search_work.target_compilations = 1;
            invalid.selected_plan.search_work.measurements = 1;
            invalid.selected_plan.measurement = measurement;
            let error = decode_payload(invalid)
                .expect_err("unmeasured schedules with device samples must not decode");
            assert!(
                error
                    .diagnostic
                    .message
                    .contains("unmeasured selection records 1 on-device measurements"),
                "diagnostic must identify contradictory measurement state: {error}"
            );
        }
    }

    #[test]
    fn one_name_for_two_values_is_refused_at_decode() {
        let error = decode(vec![resource("out", 1), resource("out", 2)])
            .expect_err("a name claimed by two values must not decode");

        assert_eq!(
            error.diagnostic.code.as_str(),
            CompilerFailureKind::MalformedArtifact.as_str()
        );
        let message = error.diagnostic.message.as_ref();
        assert!(
            message.contains("`out`") && message.contains("value 1") && message.contains("value 2"),
            "the diagnostic must name the reused name and both values it claims: {message}"
        );
    }

    #[test]
    fn one_name_repeated_for_one_value_is_not_a_collision() {
        let artifact = decode(vec![resource("out", 1), resource("out", 1)])
            .expect("a repeated record for one value names one binding");

        let by_name = artifact
            .canonical_value_by_name()
            .expect("no name claims two values");
        assert_eq!(by_name.get("out").copied(), Some(ArtifactValueId(1)));
    }

    #[test]
    fn every_resource_name_resolves_to_its_own_value() {
        let names = ["a", "b", "c"];
        let records = names
            .iter()
            .enumerate()
            .map(|(index, name)| resource(name, index as u32 + 7))
            .collect::<Vec<_>>();
        let artifact = decode(records.clone()).expect("distinct names must decode");

        let by_name = artifact
            .canonical_value_by_name()
            .expect("distinct names cannot collide");
        assert_eq!(by_name.len(), records.len());
        for record in &records {
            assert_eq!(
                by_name.get(record.name.as_str()).copied(),
                Some(record.value),
                "`{}` must resolve to its own value",
                record.name
            );
        }
    }

    fn entry_node(id: u32) -> NodeRecord {
        NodeRecord {
            id: ArtifactNodeId(id),
            name: format!("n{id}"),
            program: Vec::new(),
        }
    }

    fn launch(node: u32) -> GeometryRecord {
        crate::geometry_fixtures::geometry(node, 0, [32, 1, 1])
    }

    fn launchable() -> ArtifactPayload {
        let mut payload = payload(vec![ResourceRecord {
            value: ArtifactValueId(1),
            name: "scratch".to_string(),
            element_count: 64,
            byte_count: 256,
            lifetime: ResourceLifetime::Invocation,
            retained_predecessor: None,
            first_stage: 0,
            last_stage: 0,
        }]);
        payload.nodes = vec![entry_node(0)];
        payload.geometry = vec![launch(0)];
        payload.workspace = WorkspacePlan {
            total_bytes: 256,
            regions: vec![WorkspaceRegion {
                value: ArtifactValueId(1),
                offset: 0,
                bytes: 256,
                lifetime: ResourceLifetime::Invocation,
                first_entry: ArtifactNodeId(0),
                last_entry: ArtifactNodeId(0),
            }],
        };
        payload
    }

    fn rejection_path(payload: ArtifactPayload, case: &str) -> String {
        let error = decode_payload(payload).unwrap_err();
        error
            .diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.clone())
            .unwrap_or_else(|| panic!("{case}: diagnostic carries no path: {error}"))
    }

    /// WHY: recording geometry is what stops a consumer from computing one. A
    /// node with no record leaves exactly the hole a consumer fills in itself,
    /// and two records for one node let two consumers launch different shapes
    /// out of one authenticated artifact.
    #[test]
    fn decode_rejects_a_geometry_set_that_does_not_cover_every_node_exactly_once() {
        decode_payload(launchable()).expect("a covered node set decodes");

        let mut missing = launchable();
        missing.geometry.clear();
        let error = decode_payload(missing).expect_err("a node without geometry must not decode");
        assert!(
            error
                .diagnostic
                .message
                .contains("node 0 carries no selected geometry"),
            "diagnostic must name the uncovered node: {error}"
        );

        let mut doubled = launchable();
        doubled.geometry.push(launch(0));
        let error = decode_payload(doubled).expect_err("two records for one node must not decode");
        assert!(
            error
                .diagnostic
                .message
                .contains("node 0 carries two geometry records"),
            "diagnostic must name the doubly recorded node: {error}"
        );
    }

    /// WHY: field-level launchability is checked beside the records, and decode
    /// is the boundary that has to apply it. Bytes that reached a consumer with
    /// an unlaunchable record would be launched, because the consumer no longer
    /// has geometry of its own to fall back on.
    #[test]
    fn decode_refuses_an_unlaunchable_record_before_a_consumer_reads_it() {
        let mut invalid = launchable();
        invalid.geometry[0].grid = [1, 1, 1];
        assert_eq!(
            rejection_path(invalid, "grid that does not cover the recorded points"),
            "artifact.geometry[0].grid"
        );
    }

    /// WHY: a predecessor names the entry point a submission waits on. One the
    /// artifact does not carry is a dependency the runtime would drop, which
    /// reorders the submission instead of failing it.
    #[test]
    fn decode_rejects_a_predecessor_the_artifact_does_not_carry() {
        let mut invalid = launchable();
        invalid.geometry[0].predecessors = vec![ArtifactNodeId(9)];
        let error = decode_payload(invalid).expect_err("an unknown predecessor must not decode");
        assert!(
            error
                .diagnostic
                .message
                .contains("depends on entry point 9 the artifact does not carry"),
            "diagnostic must name the missing entry point: {error}"
        );
    }

    /// WHY: the recorded order is the submission order. A consumer submits the
    /// geometry set and the selected fusion groups in the order the artifact
    /// lists them, so a list that places an entry point before one it depends on
    /// runs a consumer of a value before its producer wrote it. Recording that
    /// order wrong is refused here rather than repaired by a topological sort in
    /// every consumer, because two consumers sorting independently is how the
    /// order stopped being the compiler's in the first place.
    #[test]
    fn decode_rejects_a_recorded_order_that_contradicts_the_dependency_dag() {
        // Node 1 produces for node 0, recorded in dependency order.
        let ordered = |dependent_first: bool, groups_swapped: bool| {
            let mut payload = launchable();
            payload.nodes = vec![entry_node(0), entry_node(1)];
            let mut dependent = launch(0);
            dependent.predecessors = vec![ArtifactNodeId(1)];
            payload.geometry = if dependent_first {
                vec![dependent, launch(1)]
            } else {
                vec![launch(1), dependent]
            };
            let group = |id: u32, node: u32| FusionRecord {
                id: FusionGroupId(id),
                members: vec![ArtifactNodeId(node)],
                stage: 0,
                legality: Vec::new(),
            };
            payload.selected_plan.fusion = if groups_swapped {
                vec![group(0, 0), group(1, 1)]
            } else {
                vec![group(1, 1), group(0, 0)]
            };
            payload
        };

        decode_payload(ordered(false, false)).expect("a dependency-ordered artifact decodes");

        let error = decode_payload(ordered(true, false))
            .expect_err("a dependent recorded first must not decode");
        assert!(
            error
                .diagnostic
                .message
                .contains("node 0 is recorded before entry point 1 it depends on"),
            "diagnostic must name the misordered pair: {error}"
        );

        assert_eq!(
            rejection_path(ordered(false, true), "consumer group planned first"),
            "artifact.body.selected_plan"
        );
    }

    /// WHY: the runtime allocates the recorded plan and binds its offsets
    /// verbatim, so a region the resource set does not back is a bind of a
    /// buffer that has no owner, and an unbindable plan must not survive decode.
    #[test]
    fn decode_rejects_a_workspace_the_resource_set_does_not_back() {
        let cases: Vec<(&str, fn(&mut ArtifactPayload), &str)> = vec![
            (
                "region for a value the artifact does not carry",
                |payload| payload.workspace.regions[0].value = ArtifactValueId(7),
                "artifact.body.workspace",
            ),
            (
                "region disagreeing with its resource lifetime",
                |payload| payload.workspace.regions[0].lifetime = ResourceLifetime::Retained,
                "artifact.body.workspace",
            ),
            (
                "live range naming an entry point the artifact does not carry",
                |payload| payload.workspace.regions[0].last_entry = ArtifactNodeId(4),
                "artifact.body.workspace",
            ),
            (
                "misaligned region",
                |payload| {
                    payload.workspace.regions[0].offset = 8;
                    payload.workspace.total_bytes = 264;
                },
                "artifact.workspace.regions[0].offset",
            ),
        ];
        for (case, mutate, expected) in cases {
            let mut invalid = launchable();
            mutate(&mut invalid);
            assert_eq!(rejection_path(invalid, case), expected, "{case}");
        }
    }

    /// WHY: a body written by the previous schema carries no geometry set and no
    /// workspace plan, so a reader that accepted one would submit an artifact
    /// with no recorded launch. The version is read before the body, so stale
    /// bytes are refused instead of partially interpreted.
    #[test]
    fn bytes_stamped_with_the_previous_schema_are_refused() {
        let framed = encode_payload(&launchable()).expect("fixture payload must frame");
        Artifact::from_bytes(&framed.bytes).expect("the current schema decodes");

        let mut stale = framed.bytes.clone();
        stale[4..6].copy_from_slice(&(ARTIFACT_SCHEMA_VERSION - 1).to_le_bytes());
        let error = Artifact::from_bytes(&stale).expect_err("a stale schema must not decode");
        assert_eq!(
            error.diagnostic.code.as_str(),
            CompilerFailureKind::VersionSkew.as_str()
        );
    }
}
