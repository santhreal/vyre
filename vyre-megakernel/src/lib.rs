//! Backend-neutral compilation from validated typed graphs to immutable artifacts.
//!
//! # Ownership
//!
//! This crate owns the whole-program compile seam:
//! - input: a validated typed [`ProgramGraph`], immutable [`ExternalFacts`], and
//!   explicit [`SearchBudget`]
//! - output: one versioned immutable [`Artifact`] plus optional [`TargetPayload`]
//!   values in an [`ArtifactEnvelope`]
//!
//! Device admission, materialization, submission, queues, residency, and recovery
//! are consumers of this compiler product and do not alter artifact identity.

#![forbid(unsafe_code)]

mod artifact;
mod candidate;
/// Open, reproducible whole-program candidate cost model.
pub mod cost;
mod envelope;
mod facts;
/// Stable semantic legality decisions for whole-program fusion.
pub mod legality;
mod normalize;
mod search;
mod select;
/// Target compiler facets over compiler-selected modules and canonical ABI.
pub mod target;

pub use envelope::{
    ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetResourceAccess,
    TargetResourceBinding, TargetResourceMemory, ARTIFACT_ENVELOPE_SCHEMA_VERSION,
    TARGET_PAYLOAD_SCHEMA_VERSION,
};
pub use target::{
    artifact_abi, fuse_selected_module, selected_modules, SelectedModule, TargetCompileError,
    TargetCompiler,
};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vyre_foundation::ir::{
    BufferAccess, DataType, GraphValueId, ProgramGraph, ShapeDim, ValueLifetime,
};

/// Current canonical artifact schema.
pub const ARTIFACT_SCHEMA_VERSION: u16 = 4;
const ARTIFACT_MAGIC: &[u8; 4] = b"VMK0";
const ARTIFACT_HEADER_BYTES: usize = 10;
const ARTIFACT_DIGEST_BYTES: usize = 32;
const ARTIFACT_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-artifact-v4\0";
const SOURCE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-source-v2\0";
const REQUEST_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-request-v2\0";

/// Stable 256-bit content identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// Return the digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Canonical node identity inside an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactNodeId(pub u32);

/// Canonical value identity inside an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactValueId(pub u32);

/// Canonical fusion-group identity inside an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FusionGroupId(pub u32);

/// Dependency endpoint with an explicit identity domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyEndpoint {
    /// An executable graph node.
    Node(ArtifactNodeId),
    /// A typed graph value materialized at a boundary.
    Value(ArtifactValueId),
}

/// Semantic reason that one artifact record depends on another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyKind {
    /// A produced value is consumed by another node.
    Data,
    /// A retained value is replaced by a type-preserving successor.
    Retained,
    /// A value must exist beyond its producing fusion group.
    Materialization,
}

/// One canonical typed dependency edge.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Edge source.
    pub from: DependencyEndpoint,
    /// Edge destination.
    pub to: DependencyEndpoint,
    /// Stable semantic edge kind.
    pub kind: DependencyKind,
    /// Connected value for data, retained, and materialization edges.
    pub value: Option<ArtifactValueId>,
}

/// Explicit bounds for one whole-program schedule search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SearchBudget {
    /// Maximum legal candidates examined by the open cost model.
    pub max_candidates: u32,
    /// Maximum abstract CPU work units consumed by analysis and search.
    pub max_cpu_work: u64,
    /// Maximum target compilations used for finalist evaluation.
    pub max_target_compilations: u32,
    /// Maximum on-device measurements used for finalist evaluation.
    pub max_measurements: u32,
    /// Maximum elapsed search time in nanoseconds.
    pub max_elapsed_ns: u64,
}

impl SearchBudget {
    /// Construct an explicit bounded search budget.
    #[must_use]
    pub const fn new(
        max_candidates: u32,
        max_cpu_work: u64,
        max_target_compilations: u32,
        max_measurements: u32,
        max_elapsed_ns: u64,
    ) -> Self {
        Self {
            max_candidates,
            max_cpu_work,
            max_target_compilations,
            max_measurements,
            max_elapsed_ns,
        }
    }
}
/// Exact bounded work performed while selecting a whole-program plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SearchWork {
    /// Legal candidates scored by the open cost model.
    pub candidates_explored: u32,
    /// Abstract deterministic CPU work units consumed.
    pub cpu_work: u64,
    /// Target compilations performed for finalist evaluation.
    pub target_compilations: u32,
    /// On-device measurements performed for finalist evaluation.
    pub measurements: u32,
    /// Deterministic elapsed-budget units charged by the search.
    pub elapsed_ns: u64,
}

/// Stable external semantic facts not encoded by graph topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFacts {
    /// Digest of validated semantic configuration outside the graph.
    pub configuration_digest: Digest,
    /// Exact value for every symbolic graph dimension.
    pub symbolic_bindings: BTreeMap<String, u64>,
    /// Verified content identity for every constant graph value.
    pub constant_identities: BTreeMap<GraphValueId, Digest>,
}

impl ExternalFacts {
    /// Construct external facts with no constant identities.
    #[must_use]
    pub fn new(configuration_digest: Digest, symbolic_bindings: BTreeMap<String, u64>) -> Self {
        Self {
            configuration_digest,
            symbolic_bindings,
            constant_identities: BTreeMap::new(),
        }
    }
}

/// Unvalidated whole-program compilation request.
pub struct CompileRequest {
    graph: ProgramGraph,
    facts: ExternalFacts,
    search_budget: SearchBudget,
    max_artifact_bytes: u64,
}

impl CompileRequest {
    /// Construct a request. Call [`Self::validate`] before compilation.
    #[must_use]
    pub const fn new(
        graph: ProgramGraph,
        facts: ExternalFacts,
        search_budget: SearchBudget,
        max_artifact_bytes: u64,
    ) -> Self {
        Self {
            graph,
            facts,
            search_budget,
            max_artifact_bytes,
        }
    }

    /// Validate topology, programs, external facts, and resource bounds.
    pub fn validate(self) -> Result<ValidatedCompileRequest, CompileError> {
        if self.max_artifact_bytes == 0 {
            return Err(failure(
                DiagnosticCode::ArtifactLimit,
                "request.max_artifact_bytes",
                "artifact byte limit must be greater than zero",
                "supply a positive bounded artifact byte limit",
            ));
        }
        if self.search_budget.max_candidates == 0
            || self.search_budget.max_cpu_work == 0
            || self.search_budget.max_elapsed_ns == 0
        {
            return Err(failure(
                DiagnosticCode::InvalidSearchBudget,
                "request.search_budget",
                "candidate, CPU-work, and elapsed-work bounds must be positive",
                "supply explicit positive bounds for every mandatory search dimension",
            ));
        }
        self.graph.analyze().map_err(|error| {
            failure(
                DiagnosticCode::InvalidProgram,
                "request.graph",
                error.to_string(),
                "supply a structurally valid acyclic ProgramGraph",
            )
        })?;
        for node in self.graph.nodes() {
            node.program.validate().map_err(|error| {
                failure(
                    DiagnosticCode::InvalidProgram,
                    format!("request.graph.nodes[{}].program", node.id.0),
                    error.to_string(),
                    "supply a structurally valid typed program",
                )
            })?;
        }
        validate_bindings(&self.graph, &self.facts.symbolic_bindings)?;
        validate_constant_identities(&self.graph, &self.facts.constant_identities)?;
        Ok(ValidatedCompileRequest {
            graph: self.graph,
            facts: self.facts,
            search_budget: self.search_budget,
            max_artifact_bytes: self.max_artifact_bytes,
        })
    }
}

/// A graph and complete immutable facts that passed request validation.
pub struct ValidatedCompileRequest {
    graph: ProgramGraph,
    facts: ExternalFacts,
    search_budget: SearchBudget,
    max_artifact_bytes: u64,
}

impl ValidatedCompileRequest {
    /// Borrow the validated source graph.
    #[must_use]
    pub const fn graph(&self) -> &ProgramGraph {
        &self.graph
    }

    /// Borrow validated external semantic facts.
    #[must_use]
    pub const fn facts(&self) -> &ExternalFacts {
        &self.facts
    }

    /// Return the explicit bounded-search policy.
    #[must_use]
    pub const fn search_budget(&self) -> SearchBudget {
        self.search_budget
    }

    /// Return the maximum accepted artifact byte length.
    #[must_use]
    pub const fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }
}

/// Stable diagnostic reason code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DiagnosticCode {
    /// A program failed structural validation.
    InvalidProgram,
    /// A symbolic extent had no exact binding.
    MissingSymbol,
    /// A binding was supplied for no graph symbol.
    UnknownSymbol,
    /// An ordering constraint made the dependency graph cyclic.
    DependencyCycle,
    /// Checked size arithmetic overflowed.
    ResourceOverflow,
    /// A value representation has no static packed size.
    UnsizedResource,
    /// The canonical artifact exceeded the caller's bound.
    ArtifactLimit,
    /// Artifact framing or canonical body data was malformed.
    MalformedArtifact,
    /// Artifact schema is not supported by this compiler version.
    VersionSkew,
    /// Artifact content identity did not match its body.
    DigestMismatch,
    /// Target payload framing or metadata was malformed.
    MalformedTargetPayload,
    /// Target payload schema or format version is incompatible.
    TargetPayloadVersionSkew,
    /// Target payload content identity did not match its metadata and bytes.
    TargetPayloadDigestMismatch,
    /// Target payload metadata names a different neutral artifact record.
    TargetPayloadAssociationMismatch,
    /// No attached target payload satisfies the required format identity.
    IncompatibleTargetPayload,
    /// Mandatory schedule-search bounds are zero or otherwise invalid.
    InvalidSearchBudget,
    /// A constant graph value has no verified content identity.
    MissingConstantIdentity,
    /// A constant identity was supplied for a non-constant graph value.
    UnknownConstantIdentity,
}

impl DiagnosticCode {
    /// Stable ASCII code for logs and serialized evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidProgram => "MKC001_INVALID_PROGRAM",
            Self::MissingSymbol => "MKC002_MISSING_SYMBOL",
            Self::UnknownSymbol => "MKC003_UNKNOWN_SYMBOL",
            Self::DependencyCycle => "MKC010_DEPENDENCY_CYCLE",
            Self::ResourceOverflow => "MKC011_RESOURCE_OVERFLOW",
            Self::UnsizedResource => "MKC012_UNSIZED_RESOURCE",
            Self::ArtifactLimit => "MKC013_ARTIFACT_LIMIT",
            Self::MalformedArtifact => "MKC014_MALFORMED_ARTIFACT",
            Self::VersionSkew => "MKC015_VERSION_SKEW",
            Self::DigestMismatch => "MKC016_DIGEST_MISMATCH",
            Self::MalformedTargetPayload => "MKC017_MALFORMED_TARGET_PAYLOAD",
            Self::TargetPayloadVersionSkew => "MKC018_TARGET_PAYLOAD_VERSION_SKEW",
            Self::TargetPayloadDigestMismatch => "MKC019_TARGET_PAYLOAD_DIGEST_MISMATCH",
            Self::TargetPayloadAssociationMismatch => "MKC020_TARGET_PAYLOAD_ASSOCIATION_MISMATCH",
            Self::IncompatibleTargetPayload => "MKC021_INCOMPATIBLE_TARGET_PAYLOAD",
            Self::InvalidSearchBudget => "MKC022_INVALID_SEARCH_BUDGET",
            Self::MissingConstantIdentity => "MKC023_MISSING_CONSTANT_IDENTITY",
            Self::UnknownConstantIdentity => "MKC024_UNKNOWN_CONSTANT_IDENTITY",
        }
    }
}

/// Stable actionable compiler diagnostic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Machine-stable reason code.
    pub code: DiagnosticCode,
    /// Stable request or artifact path associated with the failure.
    pub path: String,
    /// Deterministic failure detail.
    pub message: String,
    /// Deterministic corrective action.
    pub fix: String,
}

/// Compilation or artifact-validation failure.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{diagnostic}")]
pub struct CompileError {
    /// Structured stable diagnostic.
    pub diagnostic: Diagnostic,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "{} at {}: {}. Fix: {}",
            self.code.as_str(),
            self.path,
            self.message,
            self.fix
        )
    }
}

/// Canonical executable-node payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeRecord {
    /// Graph node identity preserved from [`ProgramGraph`].
    pub id: ArtifactNodeId,
    /// Stable diagnostic name; graph ID assignment never depends on lexical order.
    pub name: String,
    /// Canonical versioned program wire bytes.
    pub program: Vec<u8>,
}

/// Neutral per-node logical geometry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryRecord {
    /// Node whose program declares the geometry.
    pub node: ArtifactNodeId,
    /// Program-declared workgroup dimensions.
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
    /// First barrier stage needing the value.
    pub first_stage: u32,
    /// Last barrier stage needing the value.
    pub last_stage: u32,
}

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
struct ArtifactPayload {
    schema_version: u16,
    nodes: Vec<NodeRecord>,
    dependencies: Vec<DependencyEdge>,
    selected_plan: SelectedPlan,
    abi: ArtifactAbi,
    resources: Vec<ResourceRecord>,
    resource_envelope: ResourceEnvelope,
    geometry: Vec<GeometryRecord>,
    provenance: Provenance,
}

/// Versioned immutable canonical whole-program compiler result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifact {
    payload: ArtifactPayload,
    digest: Digest,
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
        encode_payload(&self.payload)
    }

    /// Decode, authenticate, and reject non-canonical or incompatible bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        if bytes.len() < ARTIFACT_HEADER_BYTES + ARTIFACT_DIGEST_BYTES {
            return Err(failure(
                DiagnosticCode::MalformedArtifact,
                "artifact.header",
                "artifact is shorter than its fixed framing",
                "supply complete VMK0 bytes",
            ));
        }
        if &bytes[..4] != ARTIFACT_MAGIC {
            return Err(failure(
                DiagnosticCode::MalformedArtifact,
                "artifact.magic",
                "artifact magic is not VMK0",
                "supply canonical megakernel artifact bytes",
            ));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != ARTIFACT_SCHEMA_VERSION {
            return Err(failure(
                DiagnosticCode::VersionSkew,
                "artifact.schema_version",
                format!("schema {version} is unsupported; expected {ARTIFACT_SCHEMA_VERSION}"),
                "recompile the source graph with this compiler version",
            ));
        }
        let body_len = u32::from_le_bytes(bytes[6..10].try_into().expect("fixed slice")) as usize;
        let expected_len = ARTIFACT_HEADER_BYTES
            .checked_add(body_len)
            .and_then(|len| len.checked_add(ARTIFACT_DIGEST_BYTES))
            .ok_or_else(|| {
                failure(
                    DiagnosticCode::MalformedArtifact,
                    "artifact.body_length",
                    "framed body length overflowed addressable memory",
                    "supply bounded canonical artifact bytes",
                )
            })?;
        if bytes.len() != expected_len {
            return Err(failure(
                DiagnosticCode::MalformedArtifact,
                "artifact.body_length",
                format!(
                    "framing declares {expected_len} bytes but received {}",
                    bytes.len()
                ),
                "supply exactly one complete canonical artifact",
            ));
        }
        let body = &bytes[ARTIFACT_HEADER_BYTES..ARTIFACT_HEADER_BYTES + body_len];
        let expected_digest = artifact_digest(version, body);
        let encoded_digest: [u8; 32] = bytes[ARTIFACT_HEADER_BYTES + body_len..]
            .try_into()
            .expect("validated digest length");
        if expected_digest.0 != encoded_digest {
            return Err(failure(
                DiagnosticCode::DigestMismatch,
                "artifact.digest",
                "artifact body does not match its content identity",
                "discard the corrupted artifact and recompile",
            ));
        }
        let payload: ArtifactPayload = serde_json::from_slice(body).map_err(|error| {
            failure(
                DiagnosticCode::MalformedArtifact,
                "artifact.body",
                error.to_string(),
                "supply a canonical body emitted by this crate",
            )
        })?;
        if payload.schema_version != version {
            return Err(failure(
                DiagnosticCode::VersionSkew,
                "artifact.body.schema_version",
                "body schema disagrees with framing schema",
                "recompile instead of rewriting artifact framing",
            ));
        }
        let canonical = serde_json::to_vec(&payload).map_err(serialization_failure)?;
        if canonical != body {
            return Err(failure(
                DiagnosticCode::MalformedArtifact,
                "artifact.body",
                "artifact body is valid JSON but not canonical JSON",
                "use the canonical bytes emitted by Artifact::to_bytes",
            ));
        }
        Ok(Self {
            payload,
            digest: expected_digest,
        })
    }
}

/// Compile one validated typed graph into a canonical backend-neutral artifact.
pub fn compile(request: &ValidatedCompileRequest) -> Result<Artifact, CompileError> {
    let canonical_wire = request.graph.to_wire().map_err(|error| {
        failure(
            DiagnosticCode::InvalidProgram,
            "request.graph",
            error.to_string(),
            "supply a graph representable by the canonical foundation wire format",
        )
    })?;
    let source_graph = domain_digest(SOURCE_DIGEST_DOMAIN, &canonical_wire);

    let nodes = request
        .graph
        .nodes()
        .iter()
        .map(|node| {
            let program = node.program.canonical_wire_bytes().map_err(|error| {
                failure(
                    DiagnosticCode::InvalidProgram,
                    format!("request.graph.nodes[{}].program", node.id.0),
                    error.to_string(),
                    "supply canonical-wire-compatible typed IR",
                )
            })?;
            Ok(NodeRecord {
                id: ArtifactNodeId(node.id.0),
                name: node.name.clone(),
                program,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let geometry = request
        .graph
        .nodes()
        .iter()
        .map(|node| GeometryRecord {
            node: ArtifactNodeId(node.id.0),
            workgroup_size: node.program.workgroup_size,
        })
        .collect::<Vec<_>>();

    let normalized = normalize::normalize(&request.graph)?;
    let dependencies = normalized.dependencies;
    let artifact::ArtifactPlan {
        node_groups,
        stages,
        selected_plan,
    } = artifact::plan(&request.graph, &dependencies, request.search_budget)?;
    let (resources, resource_envelope) = build_resources(
        &request.graph,
        &request.facts.symbolic_bindings,
        &node_groups,
        &stages,
    )?;
    let abi = build_abi(&request.graph)?;
    let request_bytes =
        serde_json::to_vec(&RequestIdentity::from(request)).map_err(serialization_failure)?;
    let provenance = Provenance {
        source_graph,
        request: domain_digest(REQUEST_DIGEST_DOMAIN, &request_bytes),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let payload = ArtifactPayload {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        nodes,
        dependencies,
        selected_plan,
        abi,
        resources,
        resource_envelope,
        geometry,
        provenance,
    };
    let bytes = encode_payload(&payload)?;
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| overflow("artifact", "artifact length exceeds u64"))?;
    if byte_len > request.max_artifact_bytes {
        return Err(failure(
            DiagnosticCode::ArtifactLimit,
            "artifact",
            format!(
                "canonical artifact is {byte_len} bytes; limit is {}",
                request.max_artifact_bytes
            ),
            "raise the explicit artifact bound or reduce the source graph",
        ));
    }
    let digest: [u8; 32] = bytes[bytes.len() - ARTIFACT_DIGEST_BYTES..]
        .try_into()
        .expect("encoded digest length");
    Ok(Artifact {
        payload,
        digest: Digest(digest),
    })
}

#[derive(Serialize)]
struct RequestIdentity<'a> {
    configuration_digest: Digest,
    symbolic_bindings: &'a BTreeMap<String, u64>,
    constant_identities: Vec<(u32, Digest)>,
    search_budget: SearchBudget,
}

impl<'a> From<&'a ValidatedCompileRequest> for RequestIdentity<'a> {
    fn from(request: &'a ValidatedCompileRequest) -> Self {
        Self {
            configuration_digest: request.facts.configuration_digest,
            symbolic_bindings: &request.facts.symbolic_bindings,
            constant_identities: request
                .facts
                .constant_identities
                .iter()
                .map(|(id, digest)| (id.0, *digest))
                .collect(),
            search_budget: request.search_budget,
        }
    }
}

fn validate_bindings(
    graph: &ProgramGraph,
    bindings: &BTreeMap<String, u64>,
) -> Result<(), CompileError> {
    let symbols: BTreeSet<&str> = graph
        .values()
        .iter()
        .flat_map(|value| &value.contract.shape)
        .filter_map(|dim| match dim {
            ShapeDim::Known(_) => None,
            ShapeDim::Symbol(symbol) => Some(symbol.as_str()),
        })
        .collect();
    if let Some(symbol) = symbols
        .iter()
        .find(|symbol| !bindings.contains_key(**symbol))
    {
        return Err(failure(
            DiagnosticCode::MissingSymbol,
            format!("request.facts.symbolic_bindings.{symbol}"),
            "graph symbol has no exact extent",
            "bind every symbolic graph dimension before compilation",
        ));
    }
    if let Some(symbol) = bindings
        .keys()
        .find(|symbol| !symbols.contains(symbol.as_str()))
    {
        return Err(failure(
            DiagnosticCode::UnknownSymbol,
            format!("request.facts.symbolic_bindings.{symbol}"),
            "binding does not occur in the graph",
            "remove stale bindings or use the graph's exact symbol name",
        ));
    }
    Ok(())
}

fn validate_constant_identities(
    graph: &ProgramGraph,
    identities: &BTreeMap<GraphValueId, Digest>,
) -> Result<(), CompileError> {
    let constants = graph
        .values()
        .iter()
        .filter(|value| value.contract.lifetime == ValueLifetime::Constant)
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    if let Some(id) = constants.iter().find(|id| !identities.contains_key(*id)) {
        return Err(failure(
            DiagnosticCode::MissingConstantIdentity,
            format!("request.facts.constant_identities.{}", id.0),
            "constant graph value has no verified content identity",
            "supply one digest keyed by the constant GraphValueId",
        ));
    }
    if let Some(id) = identities.keys().find(|id| !constants.contains(id)) {
        return Err(failure(
            DiagnosticCode::UnknownConstantIdentity,
            format!("request.facts.constant_identities.{}", id.0),
            "constant identity names a non-constant or missing graph value",
            "remove stale identities and key constant content by GraphValueId",
        ));
    }
    Ok(())
}

fn build_abi(graph: &ProgramGraph) -> Result<ArtifactAbi, CompileError> {
    let resources = graph
        .values()
        .iter()
        .map(|value| {
            let access = match value.contract.access.clone() {
                BufferAccess::ReadOnly => AbiAccess::ReadOnly,
                BufferAccess::WriteOnly => AbiAccess::WriteOnly,
                BufferAccess::ReadWrite => AbiAccess::ReadWrite,
                BufferAccess::Uniform => AbiAccess::Uniform,
                unsupported => {
                    return Err(failure(
                        DiagnosticCode::InvalidProgram,
                        format!("request.graph.values[{}].contract.access", value.id.0),
                        format!("access {unsupported:?} has no artifact ABI representation"),
                        "lower workgroup/private resources inside the node Program",
                    ))
                }
            };
            Ok(ResourceAbiRecord {
                slot: value.id.0,
                value: ArtifactValueId(value.id.0),
                dtype: value.contract.dtype.clone(),
                access,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let entries = graph
        .nodes()
        .iter()
        .map(|node| EntryAbiRecord {
            node: ArtifactNodeId(node.id.0),
            inputs: node
                .inputs
                .iter()
                .map(|input| ArtifactValueId(input.value.0))
                .collect(),
            outputs: node
                .outputs
                .iter()
                .map(|output| ArtifactValueId(output.0))
                .collect(),
        })
        .collect();
    Ok(ArtifactAbi { resources, entries })
}

fn ensure_node_dag(
    count: usize,
    dependencies: &[DependencyEdge],
    code: DiagnosticCode,
) -> Result<(), CompileError> {
    let groups: Vec<_> = (0..count).map(|id| FusionGroupId(id as u32)).collect();
    ensure_group_dag(count, dependencies, &groups, code)
}

fn ensure_group_dag(
    count: usize,
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
    code: DiagnosticCode,
) -> Result<(), CompileError> {
    group_stages_inner(count, dependencies, node_groups)
        .map(|_| ())
        .map_err(|_| {
            failure(
                code,
                "artifact.dependencies",
                "dependency graph contains a cycle",
                "remove the cyclic semantic dependency",
            )
        })
}

fn group_stages(
    count: usize,
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
) -> Result<Vec<u32>, CompileError> {
    group_stages_inner(count, dependencies, node_groups).map_err(|_| {
        failure(
            DiagnosticCode::DependencyCycle,
            "artifact.dependencies",
            "selected-plan dependency graph contains a cycle",
            "fix compiler legality before plan selection",
        )
    })
}

fn group_stages_inner(
    count: usize,
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
) -> Result<Vec<u32>, ()> {
    let mut outgoing = vec![BTreeSet::<usize>::new(); count];
    let mut indegree = vec![0usize; count];
    for edge in dependencies {
        let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        else {
            continue;
        };
        let from = node_groups[from.0 as usize].0 as usize;
        let to = node_groups[to.0 as usize].0 as usize;
        if from != to && outgoing[from].insert(to) {
            indegree[to] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut stage = vec![0u32; count];
    let mut visited = 0usize;
    while let Some(next) = ready.pop_first() {
        visited += 1;
        for successor in outgoing[next].iter().copied() {
            stage[successor] = stage[successor].max(stage[next].checked_add(1).ok_or(())?);
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.insert(successor);
            }
        }
    }
    (visited == count).then_some(stage).ok_or(())
}

fn build_barriers(
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
    stages: &[u32],
) -> Result<Vec<BarrierRecord>, CompileError> {
    let max_stage = stages.iter().copied().max().unwrap_or(0);
    let mut barriers = Vec::new();
    for after_stage in 1..=max_stage {
        let mut edge_ids = Vec::new();
        for (index, edge) in dependencies.iter().enumerate() {
            let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) =
                (edge.from, edge.to)
            else {
                continue;
            };
            let from_stage = stages[node_groups[from.0 as usize].0 as usize];
            let to_stage = stages[node_groups[to.0 as usize].0 as usize];
            if from_stage < after_stage && to_stage == after_stage {
                edge_ids.push(
                    u32::try_from(index).map_err(|_| {
                        overflow("artifact.dependencies", "edge identity exceeds u32")
                    })?,
                );
            }
        }
        barriers.push(BarrierRecord {
            before_stage: after_stage - 1,
            after_stage,
            dependencies: edge_ids,
        });
    }
    Ok(barriers)
}

fn build_materializations(
    graph: &ProgramGraph,
    node_groups: &[FusionGroupId],
    stages: &[u32],
) -> Vec<MaterializationRecord> {
    let mut records = Vec::new();
    for value in graph.values() {
        let Some(producer) = value.producer else {
            continue;
        };
        let producer_node = ArtifactNodeId(producer.0);
        let producer_group = node_groups[producer_node.0 as usize];
        let producer_stage = stages[producer_group.0 as usize];
        let cross_group = value.consumers.iter().any(|consumer| {
            let consumer_node = ArtifactNodeId(consumer.0);
            node_groups[consumer_node.0 as usize] != producer_group
        });
        let reason = match value.contract.lifetime {
            ValueLifetime::Output => Some(MaterializationReason::Output),
            ValueLifetime::Retained => Some(MaterializationReason::Retained),
            _ if cross_group => Some(MaterializationReason::CrossGroupUse),
            _ => None,
        };
        if let Some(reason) = reason {
            records.push(MaterializationRecord {
                value: ArtifactValueId(value.id.0),
                producer: producer_group,
                stage: producer_stage,
                reason,
            });
        }
    }
    records.sort_by_key(|record| (record.value, record.reason as u8));
    records
}

fn build_resources(
    graph: &ProgramGraph,
    bindings: &BTreeMap<String, u64>,
    node_groups: &[FusionGroupId],
    stages: &[u32],
) -> Result<(Vec<ResourceRecord>, ResourceEnvelope), CompileError> {
    let final_stage = stages.iter().copied().max().unwrap_or(0);
    let mut resources = Vec::with_capacity(graph.values().len());
    for value in graph.values() {
        let mut element_count = 1u64;
        for dim in &value.contract.shape {
            let extent = match dim {
                ShapeDim::Known(extent) => *extent,
                ShapeDim::Symbol(symbol) => bindings[symbol],
            };
            element_count = element_count.checked_mul(extent).ok_or_else(|| {
                overflow(
                    format!("graph.values[{}].shape", value.name),
                    "shape element count exceeds u64",
                )
            })?;
        }
        let host_count = usize::try_from(element_count).map_err(|_| {
            overflow(
                format!("graph.values[{}].shape", value.name),
                "shape element count exceeds addressable packed-size input",
            )
        })?;
        let byte_count = value
            .contract
            .dtype
            .packed_size_bytes(host_count)
            .map_err(|message| overflow(format!("graph.values[{}].dtype", value.name), message))?
            .ok_or_else(|| {
                failure(
                    DiagnosticCode::UnsizedResource,
                    format!("graph.values[{}].dtype", value.name),
                    "value representation has no fixed packed byte size",
                    "resolve the representation to a fixed-width typed value before compilation",
                )
            })?;
        let byte_count = u64::try_from(byte_count).map_err(|_| {
            overflow(
                format!("graph.values[{}]", value.name),
                "packed byte count exceeds u64",
            )
        })?;
        let producer_stage = value.producer.map_or(0, |producer| {
            stages[node_groups[producer.0 as usize].0 as usize]
        });
        let mut last_stage = value
            .consumers
            .iter()
            .map(|consumer| stages[node_groups[consumer.0 as usize].0 as usize])
            .max()
            .unwrap_or(producer_stage);
        if matches!(
            value.contract.lifetime,
            ValueLifetime::Output | ValueLifetime::Retained
        ) {
            last_stage = last_stage.max(final_stage);
        }
        resources.push(ResourceRecord {
            value: ArtifactValueId(value.id.0),
            name: value.name.clone(),
            element_count,
            byte_count,
            lifetime: match value.contract.lifetime {
                ValueLifetime::Constant => ResourceLifetime::Constant,
                ValueLifetime::Invocation => ResourceLifetime::Invocation,
                ValueLifetime::Retained => ResourceLifetime::Retained,
                ValueLifetime::Output => ResourceLifetime::Output,
            },
            first_stage: producer_stage,
            last_stage,
        });
    }
    resources.sort_by_key(|resource| resource.value);
    let total_bytes = resources.iter().try_fold(0u64, |total, resource| {
        total.checked_add(resource.byte_count).ok_or_else(|| {
            overflow(
                "artifact.resource_envelope.total_bytes",
                "resource sum exceeds u64",
            )
        })
    })?;
    let mut peak_live_bytes = 0u64;
    for stage in 0..=final_stage {
        let live = resources
            .iter()
            .filter(|resource| resource.first_stage <= stage && stage <= resource.last_stage)
            .try_fold(0u64, |total, resource| {
                total.checked_add(resource.byte_count).ok_or_else(|| {
                    overflow(
                        "artifact.resource_envelope.peak_live_bytes",
                        "live resource sum exceeds u64",
                    )
                })
            })?;
        peak_live_bytes = peak_live_bytes.max(live);
    }
    Ok((
        resources,
        ResourceEnvelope {
            total_bytes,
            peak_live_bytes,
        },
    ))
}

fn encode_payload(payload: &ArtifactPayload) -> Result<Vec<u8>, CompileError> {
    let body = serde_json::to_vec(payload).map_err(serialization_failure)?;
    let body_len = u32::try_from(body.len()).map_err(|_| {
        overflow(
            "artifact.body",
            "canonical body exceeds the u32 framing limit",
        )
    })?;
    let digest = artifact_digest(payload.schema_version, &body);
    let capacity = ARTIFACT_HEADER_BYTES
        .checked_add(body.len())
        .and_then(|len| len.checked_add(ARTIFACT_DIGEST_BYTES))
        .ok_or_else(|| overflow("artifact", "encoded artifact length overflowed usize"))?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(ARTIFACT_MAGIC);
    bytes.extend_from_slice(&payload.schema_version.to_le_bytes());
    bytes.extend_from_slice(&body_len.to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes.extend_from_slice(&digest.0);
    Ok(bytes)
}

fn artifact_digest(version: u16, body: &[u8]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ARTIFACT_DIGEST_DOMAIN);
    hasher.update(&version.to_le_bytes());
    hasher.update(&(body.len() as u64).to_le_bytes());
    hasher.update(body);
    Digest(*hasher.finalize().as_bytes())
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Digest(*hasher.finalize().as_bytes())
}

fn serialization_failure(error: serde_json::Error) -> CompileError {
    failure(
        DiagnosticCode::MalformedArtifact,
        "artifact.body",
        error.to_string(),
        "use values representable by the canonical artifact schema",
    )
}

fn overflow(path: impl Into<String>, message: impl Into<String>) -> CompileError {
    failure(
        DiagnosticCode::ResourceOverflow,
        path,
        message,
        "reduce resolved extents or split the graph before compilation",
    )
}

fn failure(
    code: DiagnosticCode,
    path: impl Into<String>,
    message: impl Into<String>,
    fix: impl Into<String>,
) -> CompileError {
    CompileError {
        diagnostic: Diagnostic {
            code,
            path: path.into(),
            message: message.into(),
            fix: fix.into(),
        },
    }
}
