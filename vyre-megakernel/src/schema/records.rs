//! Versioned artifact records: resources, the whole-program ABI, and the
//! fusion, barrier and materialization facts one compile selected.

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::DataType;

use crate::identity::{ArtifactNodeId, ArtifactValueId, Digest, FusionGroupId};
use crate::legality;
use crate::objective::CompileObjective;

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

/// One named Program buffer projected onto a canonical graph value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryResourceBinding {
    /// Program buffer name at the executable entry boundary.
    pub buffer: String,
    /// Canonical graph value bound to that buffer.
    pub value: ArtifactValueId,
}

/// One canonical executable entry in the whole-program ABI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryAbiRecord {
    /// Typed graph node implemented by this entry.
    pub node: ArtifactNodeId,
    /// Input value identities in Program buffer order.
    pub inputs: Vec<ArtifactValueId>,
    /// Input identities paired with their exact Program buffer names.
    pub input_bindings: Vec<EntryResourceBinding>,
    /// Output value identities in Program buffer order.
    pub outputs: Vec<ArtifactValueId>,
    /// Output identities paired with their exact Program buffer names.
    pub output_bindings: Vec<EntryResourceBinding>,
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

/// Deterministic identities establishing how an artifact was produced, and what
/// the plan inside it was selected to optimize.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Canonical source-graph content identity.
    pub source_graph: Digest,
    /// Canonical validated-request identity.
    pub request: Digest,
    /// Objective the recorded plan was selected under.
    ///
    /// A reader of the artifact can state what "best" meant for it: the request
    /// digest authenticates the whole request, and this states the part of it
    /// that decided the selection, so a latency artifact is never compared
    /// against a throughput one as though they answered the same question.
    pub objective: CompileObjective,
    /// Compiler crate version.
    pub compiler_version: String,
}
