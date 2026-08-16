//! The canonical artifact schema: every versioned record, and the immutable
//! container that frames them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::DataType;

use crate::error::{failure, serialization_failure, CompileError, CompilerFailureKind};
use crate::frame;
use crate::identity::{ArtifactNodeId, ArtifactValueId, DependencyEdge, Digest, FusionGroupId};
use crate::request::{SearchBudget, SearchWork};
use crate::{cost, legality};

/// Current canonical artifact schema.
pub const ARTIFACT_SCHEMA_VERSION: u16 = 6;

/// Canonical executable-node payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeRecord {
    /// Graph node identity preserved from [`ProgramGraph`](vyre_foundation::ir::ProgramGraph).
    pub id: ArtifactNodeId,
    /// Stable diagnostic name; graph ID assignment never depends on lexical order.
    pub name: String,
    /// Canonical versioned program wire bytes.
    pub program: Vec<u8>,
}

/// Compiler-selected per-node launch geometry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryRecord {
    /// Node this geometry launches.
    pub node: ArtifactNodeId,
    /// Workgroup dimensions the search selected for this node.
    ///
    /// Every consumer launches this geometry. The workgroup the node program
    /// declares is an input to the search, not its result, so a consumer that
    /// reads the program instead of this record can launch a shape the emitted
    /// module does not have.
    pub workgroup_size: [u32; 3],
}

/// Graph value lifetime represented in the artifact schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLifetime {
    /// Immutable constant input.
    Constant,
    /// Temporary value for one submission.
    Invocation,
    /// Mutable value retained across submissions.
    Retained,
    /// Caller-visible graph output.
    Output,
}

/// Canonical resource and liveness fact for one typed graph value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRecord {
    /// Canonical value identity.
    pub value: ArtifactValueId,
    /// Stable graph value name.
    pub name: String,
    /// Resolved logical element count.
    pub element_count: u64,
    /// Canonical packed byte count.
    pub byte_count: u64,
    /// Semantic lifetime class.
    pub lifetime: ResourceLifetime,
    /// Prior retained value when this resource replaces retained state.
    pub retained_predecessor: Option<ArtifactValueId>,
    /// First barrier stage needing the value.
    pub first_stage: u32,
    /// Last barrier stage needing the value.
    pub last_stage: u32,
}

/// One resource name claimed by two canonical values.
///
/// Carried as its own type rather than a formatted error so each consumer can
/// report it in the error vocabulary of its own boundary while the detection
/// stays with the resource set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceNameCollision {
    /// The reused resource name.
    pub name: String,
    /// The value the name reached first.
    pub first: ArtifactValueId,
    /// The value that reused the name.
    pub second: ArtifactValueId,
}

impl std::fmt::Display for ResourceNameCollision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "artifact resource name `{}` names both value {} and value {}. Fix: resource names carry descriptor binding identity, so one artifact must not reuse a name for two values.",
            self.name, self.first.0, self.second.0
        )
    }
}

impl std::error::Error for ResourceNameCollision {}

/// Aggregate checked resource envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceEnvelope {
    /// Sum of all canonical value byte counts.
    pub total_bytes: u64,
    /// Maximum bytes simultaneously live in any artifact stage.
    pub peak_live_bytes: u64,
}

/// Canonical neutral resource access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbiAccess {
    /// Read-only resource.
    ReadOnly,
    /// Write-only resource.
    WriteOnly,
    /// Read-write resource.
    ReadWrite,
    /// Uniform read-only resource.
    Uniform,
}

/// One canonical resource slot in the whole-program ABI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAbiRecord {
    /// Dense canonical slot.
    pub slot: u32,
    /// Typed graph value occupying this slot.
    pub value: ArtifactValueId,
    /// Element representation.
    pub dtype: DataType,
    /// Required access.
    pub access: AbiAccess,
}

/// One canonical executable entry in the whole-program ABI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryAbiRecord {
    /// Typed graph node implemented by this entry.
    pub node: ArtifactNodeId,
    /// Input value identities in Program buffer order.
    pub inputs: Vec<ArtifactValueId>,
    /// Output value identities in Program buffer order.
    pub outputs: Vec<ArtifactValueId>,
}

/// Canonical resource and entry ABI projected to every target payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactAbi {
    /// Dense resource slots.
    pub resources: Vec<ResourceAbiRecord>,
    /// Executable entries.
    pub entries: Vec<EntryAbiRecord>,
}

/// Canonical compiler-selected fusion group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionRecord {
    /// Stable group identity.
    pub id: FusionGroupId,
    /// Typed group members.
    pub members: Vec<ArtifactNodeId>,
    /// Dependency stage selected for this group.
    pub stage: u32,
    /// Compiler-derived semantic-legality identities used to form the group.
    pub legality: Vec<Digest>,
}
/// Stable evidence that one proposed fusion was pruned before selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionRejection {
    /// Proposed producer.
    pub from: ArtifactNodeId,
    /// Proposed consumer.
    pub to: ArtifactNodeId,
    /// Connecting value.
    pub value: ArtifactValueId,
    /// Stable semantic rejection reason.
    pub reason: legality::FusionRejectionReason,
}

/// Dependency-completion boundary between canonical stages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarrierRecord {
    /// Stage that must complete.
    pub before_stage: u32,
    /// Stage admitted after completion.
    pub after_stage: u32,
    /// Sorted dependency-edge indices requiring the boundary.
    pub dependencies: Vec<u32>,
}

/// Reason a typed value crosses a fusion-group boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationReason {
    /// Value is consumed by a different fusion group.
    CrossGroupUse,
    /// Value is observable after artifact completion.
    Output,
    /// Value is retained for a later submission.
    Retained,
}

/// Canonical value materialization fact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationRecord {
    /// Materialized value.
    pub value: ArtifactValueId,
    /// Producing fusion group.
    pub producer: FusionGroupId,
    /// Earliest stage at which the value exists.
    pub stage: u32,
    /// Stable semantic reason.
    pub reason: MaterializationReason,
}

/// How the runtime executes one compiled artifact.
///
/// The compiler decides this, not the dispatcher: the decision needs the launch
/// count the caller declared and the device launch costs, both of which are
/// compile-time facts recorded in the request. A consumer executes the mode the
/// artifact names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// One kernel launch per stage per submission.
    Static,
    /// One resident kernel that polls a device-side work queue for the whole
    /// launch batch, paying one setup cost instead of one launch per item.
    Persistent {
        /// Launch overhead this mode removes, less the setup it pays, in
        /// nanoseconds. Computed from the device launch costs and the declared
        /// launch batch, and always positive: a non-positive figure is recorded
        /// as [`Self::Static`].
        saved_ns: u64,
    },
}

/// Whether a device measurement selected the plan, and what it measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanMeasurement {
    /// The search budget allowed no measurement, so the plan is the analytic
    /// winner and carries no measured device time.
    Unbudgeted,
    /// The device reports no launch timestamp, so nothing measured on it would
    /// be a device time.
    UntimedDevice,
    /// Selected by lowest median device time across measured finalists.
    Measured {
        /// Timestamped launches performed against the winning finalist.
        launches: u32,
        /// Median device time of those launches in nanoseconds.
        median_ns: u64,
    },
}

/// Immutable compiler-selected whole-program plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedPlan {
    /// Selected fusion groups.
    pub fusion: Vec<FusionRecord>,
    /// Required dependency-completion boundaries.
    pub barriers: Vec<BarrierRecord>,
    /// Required cross-stage value materializations.
    pub materializations: Vec<MaterializationRecord>,
    /// Number of legal candidates examined.
    pub candidates_explored: u32,
    /// Search bounds under which this plan was selected.
    pub search_budget: SearchBudget,
    /// Exact work charged against the bounded search.
    pub search_work: SearchWork,
    /// Open-model cost of the selected plan.
    pub selection_cost: cost::CostBreakdown,
    /// Illegal producer-consumer fusions pruned with stable reasons.
    pub pruned_fusions: Vec<FusionRejection>,
    /// How the runtime executes this artifact.
    pub execution: ExecutionMode,
    /// Whether a device measurement chose this plan over its finalists.
    pub measurement: PlanMeasurement,
}

/// Deterministic identities establishing how an artifact was produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Canonical source-graph content identity.
    pub source_graph: Digest,
    /// Canonical validated-request identity.
    pub request: Digest,
    /// Compiler crate version.
    pub compiler_version: String,
}

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

    fn payload(resources: Vec<ResourceRecord>) -> ArtifactPayload {
        ArtifactPayload {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            nodes: Vec::new(),
            dependencies: Vec::new(),
            selected_plan: SelectedPlan {
                fusion: Vec::new(),
                barriers: Vec::new(),
                materializations: Vec::new(),
                candidates_explored: 0,
                search_budget: SearchBudget::new(1, 1, 0, 0, 1),
                search_work: SearchWork::default(),
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

    fn decode(resources: Vec<ResourceRecord>) -> Result<Artifact, CompileError> {
        let payload = payload(resources);
        let framed = encode_payload(&payload).expect("fixture payload must frame");
        Artifact::from_bytes(&framed.bytes)
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
}
