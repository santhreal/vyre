//! Backend-neutral compilation from validated typed graphs to immutable artifacts.
//!
//! This crate owns canonical artifact construction only. Admission, execution,
//! and lifecycle policy belong to consumers above this boundary.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vyre_foundation::ir::{
    GraphNodeId, GraphValueId, ProgramGraph, ShapeDim, ValueLifetime,
};

/// Current canonical artifact schema.
pub const ARTIFACT_SCHEMA_VERSION: u16 = 1;
const ARTIFACT_MAGIC: &[u8; 4] = b"VMK0";
const ARTIFACT_HEADER_BYTES: usize = 10;
const ARTIFACT_DIGEST_BYTES: usize = 32;
const ARTIFACT_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-artifact-v1\0";
const SOURCE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-source-v1\0";
const REQUEST_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-request-v1\0";

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

/// Intended execution lifetime encoded without selecting an execution substrate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactRoute {
    /// Compile an artifact whose resources may be released after one completion.
    Static,
    /// Compile an artifact whose retained resources may span multiple submissions.
    Persistent,
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
    State,
    /// Caller-supplied semantic ordering not represented by a value flow.
    Order,
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
    /// Connected value for data, state, and materialization edges.
    pub value: Option<ArtifactValueId>,
}

/// Caller-proven semantic order between two stable graph node names.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OrderConstraint {
    /// Stable predecessor node name.
    pub before: String,
    /// Stable successor node name.
    pub after: String,
}

/// Proof input permitting two directly dependent nodes to share a fusion group.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FusionPermission {
    /// Stable predecessor node name.
    pub before: String,
    /// Stable successor node name.
    pub after: String,
    /// Identity of the semantic-legality evidence supplied below this boundary.
    pub legality_digest: Digest,
}

/// Immutable compilation inputs plus an admission-only artifact byte bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    /// Intended artifact lifetime.
    pub route: ArtifactRoute,
    /// Exact value for every symbolic graph dimension.
    pub symbolic_bindings: BTreeMap<String, u64>,
    /// Additional semantic ordering constraints.
    pub order_constraints: Vec<OrderConstraint>,
    /// Explicit semantic fusion permissions.
    pub fusion_permissions: Vec<FusionPermission>,
    /// Maximum accepted canonical artifact byte length.
    pub max_artifact_bytes: u64,
}

impl CompileOptions {
    /// Create options with no additional order or fusion facts.
    #[must_use]
    pub fn new(
        route: ArtifactRoute,
        symbolic_bindings: BTreeMap<String, u64>,
        max_artifact_bytes: u64,
    ) -> Self {
        Self {
            route,
            symbolic_bindings,
            order_constraints: Vec::new(),
            fusion_permissions: Vec::new(),
            max_artifact_bytes,
        }
    }
}

/// A graph and complete immutable inputs that passed request validation.
pub struct ValidatedCompileRequest {
    graph: ProgramGraph,
    options: CompileOptions,
}

impl ValidatedCompileRequest {
    /// Validate graph programs, symbolic bindings, and named constraints atomically.
    pub fn new(graph: ProgramGraph, mut options: CompileOptions) -> Result<Self, CompileError> {
        if options.max_artifact_bytes == 0 {
            return Err(failure(
                DiagnosticCode::ArtifactLimit,
                "options.max_artifact_bytes",
                "artifact byte limit must be greater than zero",
                "supply a positive bounded artifact byte limit",
            ));
        }
        for node in graph.nodes() {
            node.program.validate().map_err(|error| {
                failure(
                    DiagnosticCode::InvalidProgram,
                    format!("graph.nodes[{}].program", node.name),
                    error.to_string(),
                    "supply a structurally valid typed program",
                )
            })?;
        }
        validate_bindings(&graph, &options.symbolic_bindings)?;
        options.order_constraints.sort();
        options.fusion_permissions.sort();
        reject_duplicates(&options.order_constraints, "options.order_constraints")?;
        reject_duplicates(&options.fusion_permissions, "options.fusion_permissions")?;
        let names: BTreeSet<&str> = graph.nodes().iter().map(|node| node.name.as_str()).collect();
        for (index, edge) in options.order_constraints.iter().enumerate() {
            validate_named_edge(&names, &edge.before, &edge.after, format!("options.order_constraints[{index}]"))?;
        }
        for (index, permission) in options.fusion_permissions.iter().enumerate() {
            validate_named_edge(
                &names,
                &permission.before,
                &permission.after,
                format!("options.fusion_permissions[{index}]"),
            )?;
            if permission.legality_digest.0 == [0; 32] {
                return Err(failure(
                    DiagnosticCode::MissingFusionEvidence,
                    format!("options.fusion_permissions[{index}].legality_digest"),
                    "fusion legality identity must not be the all-zero sentinel",
                    "supply the digest of validated semantic-legality evidence",
                ));
            }
        }
        Ok(Self { graph, options })
    }

    /// Borrow the validated source graph.
    #[must_use]
    pub const fn graph(&self) -> &ProgramGraph {
        &self.graph
    }

    /// Borrow the validated immutable options.
    #[must_use]
    pub const fn options(&self) -> &CompileOptions {
        &self.options
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
    /// A named edge endpoint did not exist.
    UnknownNode,
    /// An edge pointed from a node to itself.
    SelfEdge,
    /// The same request fact appeared more than once.
    DuplicateFact,
    /// Fusion evidence used the reserved empty identity.
    MissingFusionEvidence,
    /// Fusion was requested without a direct dependency.
    UnconnectedFusion,
    /// Fused groups made the dependency quotient cyclic.
    FusionCycle,
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
}

impl DiagnosticCode {
    /// Stable ASCII code for logs and serialized evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidProgram => "MKC001_INVALID_PROGRAM",
            Self::MissingSymbol => "MKC002_MISSING_SYMBOL",
            Self::UnknownSymbol => "MKC003_UNKNOWN_SYMBOL",
            Self::UnknownNode => "MKC004_UNKNOWN_NODE",
            Self::SelfEdge => "MKC005_SELF_EDGE",
            Self::DuplicateFact => "MKC006_DUPLICATE_FACT",
            Self::MissingFusionEvidence => "MKC007_MISSING_FUSION_EVIDENCE",
            Self::UnconnectedFusion => "MKC008_UNCONNECTED_FUSION",
            Self::FusionCycle => "MKC009_FUSION_CYCLE",
            Self::DependencyCycle => "MKC010_DEPENDENCY_CYCLE",
            Self::ResourceOverflow => "MKC011_RESOURCE_OVERFLOW",
            Self::UnsizedResource => "MKC012_UNSIZED_RESOURCE",
            Self::ArtifactLimit => "MKC013_ARTIFACT_LIMIT",
            Self::MalformedArtifact => "MKC014_MALFORMED_ARTIFACT",
            Self::VersionSkew => "MKC015_VERSION_SKEW",
            Self::DigestMismatch => "MKC016_DIGEST_MISMATCH",
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
    /// Canonical name-sorted node identity.
    pub id: ArtifactNodeId,
    /// Stable graph node name.
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
    /// Immutable input retained by consumers.
    Immutable,
    /// Temporary value for one submission.
    Invocation,
    /// Mutable retained state.
    Retained,
    /// Observable graph output.
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

/// Canonical group formed only from caller-proven fusion permissions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionRecord {
    /// Stable group identity.
    pub id: FusionGroupId,
    /// Name-sorted group members.
    pub members: Vec<ArtifactNodeId>,
    /// Sorted semantic-legality evidence identities used to form the group.
    pub legality: Vec<Digest>,
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
    RetainedState,
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
    route: ArtifactRoute,
    nodes: Vec<NodeRecord>,
    dependencies: Vec<DependencyEdge>,
    fusion: Vec<FusionRecord>,
    barriers: Vec<BarrierRecord>,
    resources: Vec<ResourceRecord>,
    resource_envelope: ResourceEnvelope,
    geometry: Vec<GeometryRecord>,
    materializations: Vec<MaterializationRecord>,
    provenance: Provenance,
}

/// Versioned immutable canonical artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MegakernelArtifact {
    payload: ArtifactPayload,
    digest: Digest,
}

impl MegakernelArtifact {
    /// Artifact schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.payload.schema_version
    }

    /// Intended artifact route.
    #[must_use]
    pub const fn route(&self) -> ArtifactRoute {
        self.payload.route
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
        &self.payload.fusion
    }

    /// Canonical barrier boundaries.
    #[must_use]
    pub fn barriers(&self) -> &[BarrierRecord] {
        &self.payload.barriers
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
        &self.payload.materializations
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
                format!("framing declares {expected_len} bytes but received {}", bytes.len()),
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
                "use the canonical bytes emitted by MegakernelArtifact::to_bytes",
            ));
        }
        Ok(Self {
            payload,
            digest: expected_digest,
        })
    }
}

/// Compile one validated typed graph into a canonical backend-neutral artifact.
pub fn compile(request: &ValidatedCompileRequest) -> Result<MegakernelArtifact, CompileError> {
    let canonical_graph = canonical_graph(&request.graph)?;
    let canonical_wire = canonical_graph.to_wire().map_err(|error| {
        failure(
            DiagnosticCode::InvalidProgram,
            "graph",
            error.to_string(),
            "supply a graph representable by the canonical foundation wire format",
        )
    })?;
    let source_graph = domain_digest(SOURCE_DIGEST_DOMAIN, &canonical_wire);

    let mut nodes_by_name: Vec<_> = request.graph.nodes().iter().collect();
    nodes_by_name.sort_by(|left, right| left.name.cmp(&right.name));
    let node_ids: BTreeMap<&str, ArtifactNodeId> = nodes_by_name
        .iter()
        .enumerate()
        .map(|(index, node)| {
            u32::try_from(index)
                .map(|id| (node.name.as_str(), ArtifactNodeId(id)))
                .map_err(|_| overflow("graph.nodes", "node identity exceeds u32"))
        })
        .collect::<Result<_, _>>()?;

    let mut values_by_name: Vec<_> = request.graph.values().iter().collect();
    values_by_name.sort_by(|left, right| left.name.cmp(&right.name));
    let value_ids: BTreeMap<&str, ArtifactValueId> = values_by_name
        .iter()
        .enumerate()
        .map(|(index, value)| {
            u32::try_from(index)
                .map(|id| (value.name.as_str(), ArtifactValueId(id)))
                .map_err(|_| overflow("graph.values", "value identity exceeds u32"))
        })
        .collect::<Result<_, _>>()?;
    let value_name_by_old: BTreeMap<GraphValueId, &str> = request
        .graph
        .values()
        .iter()
        .map(|value| (value.id, value.name.as_str()))
        .collect();
    let old_node_name: BTreeMap<GraphNodeId, &str> = request
        .graph
        .nodes()
        .iter()
        .map(|node| (node.id, node.name.as_str()))
        .collect();

    let nodes = nodes_by_name
        .iter()
        .map(|node| {
            let program = node.program.canonical_wire_bytes().map_err(|error| {
                failure(
                    DiagnosticCode::InvalidProgram,
                    format!("graph.nodes[{}].program", node.name),
                    error.to_string(),
                    "supply canonical-wire-compatible typed IR",
                )
            })?;
            Ok(NodeRecord {
                id: node_ids[node.name.as_str()],
                name: node.name.clone(),
                program,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let geometry = nodes_by_name
        .iter()
        .map(|node| GeometryRecord {
            node: node_ids[node.name.as_str()],
            workgroup_size: node.program.workgroup_size,
        })
        .collect::<Vec<_>>();

    let mut dependencies = Vec::new();
    for value in request.graph.values() {
        let value_id = value_ids[value.name.as_str()];
        if let Some(producer) = value.producer {
            let from = ArtifactNodeId(node_ids[old_node_name[&producer]].0);
            for consumer in &value.consumers {
                dependencies.push(DependencyEdge {
                    from: DependencyEndpoint::Node(from),
                    to: DependencyEndpoint::Node(node_ids[old_node_name[consumer]]),
                    kind: DependencyKind::Data,
                    value: Some(value_id),
                });
            }
            if matches!(value.contract.lifetime, ValueLifetime::Output | ValueLifetime::SequenceState) {
                dependencies.push(DependencyEdge {
                    from: DependencyEndpoint::Node(from),
                    to: DependencyEndpoint::Value(value_id),
                    kind: DependencyKind::Materialization,
                    value: Some(value_id),
                });
            }
        }
        if let (Some(prior), Some(successor_node)) = (value.state_successor_of, value.producer) {
            let prior = &request.graph.values()[prior.0 as usize];
            if let Some(prior_node) = prior.producer {
                if prior_node != successor_node {
                    dependencies.push(DependencyEdge {
                        from: DependencyEndpoint::Node(node_ids[old_node_name[&prior_node]]),
                        to: DependencyEndpoint::Node(node_ids[old_node_name[&successor_node]]),
                        kind: DependencyKind::State,
                        value: Some(value_id),
                    });
                }
            }
        }
    }
    for edge in &request.options.order_constraints {
        dependencies.push(DependencyEdge {
            from: DependencyEndpoint::Node(node_ids[edge.before.as_str()]),
            to: DependencyEndpoint::Node(node_ids[edge.after.as_str()]),
            kind: DependencyKind::Order,
            value: None,
        });
    }
    dependencies.sort();
    dependencies.dedup();
    ensure_node_dag(nodes.len(), &dependencies, DiagnosticCode::DependencyCycle)?;

    let (fusion, node_groups) = build_fusion(
        &nodes,
        &geometry,
        &dependencies,
        &request.options.fusion_permissions,
        &node_ids,
    )?;
    let group_stages = group_stages(fusion.len(), &dependencies, &node_groups)?;
    let barriers = build_barriers(&dependencies, &node_groups, &group_stages)?;
    let materializations = build_materializations(
        &request.graph,
        &value_ids,
        &old_node_name,
        &node_ids,
        &node_groups,
        &group_stages,
    );
    let (resources, resource_envelope) = build_resources(
        &request.graph,
        &request.options.symbolic_bindings,
        &value_ids,
        &old_node_name,
        &node_ids,
        &node_groups,
        &group_stages,
    )?;
    let request_bytes = serde_json::to_vec(&RequestIdentity::from(&request.options))
        .map_err(serialization_failure)?;
    let provenance = Provenance {
        source_graph,
        request: domain_digest(REQUEST_DIGEST_DOMAIN, &request_bytes),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let payload = ArtifactPayload {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        route: request.options.route,
        nodes,
        dependencies,
        fusion,
        barriers,
        resources,
        resource_envelope,
        geometry,
        materializations,
        provenance,
    };
    let bytes = encode_payload(&payload)?;
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| overflow("artifact", "artifact length exceeds u64"))?;
    if byte_len > request.options.max_artifact_bytes {
        return Err(failure(
            DiagnosticCode::ArtifactLimit,
            "artifact",
            format!("canonical artifact is {byte_len} bytes; limit is {}", request.options.max_artifact_bytes),
            "raise the explicit artifact bound or reduce the source graph",
        ));
    }
    let digest: [u8; 32] = bytes[bytes.len() - ARTIFACT_DIGEST_BYTES..]
        .try_into()
        .expect("encoded digest length");
    Ok(MegakernelArtifact {
        payload,
        digest: Digest(digest),
    })
}

#[derive(Serialize)]
struct RequestIdentity<'a> {
    route: ArtifactRoute,
    symbolic_bindings: &'a BTreeMap<String, u64>,
    order_constraints: &'a [OrderConstraint],
    fusion_permissions: &'a [FusionPermission],
}

impl<'a> From<&'a CompileOptions> for RequestIdentity<'a> {
    fn from(options: &'a CompileOptions) -> Self {
        Self {
            route: options.route,
            symbolic_bindings: &options.symbolic_bindings,
            order_constraints: &options.order_constraints,
            fusion_permissions: &options.fusion_permissions,
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
    if let Some(symbol) = symbols.iter().find(|symbol| !bindings.contains_key(**symbol)) {
        return Err(failure(
            DiagnosticCode::MissingSymbol,
            format!("options.symbolic_bindings.{symbol}"),
            "graph symbol has no exact extent",
            "bind every symbolic graph dimension before compilation",
        ));
    }
    if let Some(symbol) = bindings.keys().find(|symbol| !symbols.contains(symbol.as_str())) {
        return Err(failure(
            DiagnosticCode::UnknownSymbol,
            format!("options.symbolic_bindings.{symbol}"),
            "binding does not occur in the graph",
            "remove stale bindings or use the graph's exact symbol name",
        ));
    }
    Ok(())
}

fn reject_duplicates<T: Ord>(values: &[T], path: &str) -> Result<(), CompileError> {
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(failure(
            DiagnosticCode::DuplicateFact,
            path,
            "request contains a duplicate canonical fact",
            "supply each order constraint or fusion permission exactly once",
        ));
    }
    Ok(())
}

fn validate_named_edge(
    names: &BTreeSet<&str>,
    before: &str,
    after: &str,
    path: String,
) -> Result<(), CompileError> {
    if !names.contains(before) {
        return Err(failure(
            DiagnosticCode::UnknownNode,
            format!("{path}.before"),
            format!("node `{before}` does not exist"),
            "use a stable node name from the validated graph",
        ));
    }
    if !names.contains(after) {
        return Err(failure(
            DiagnosticCode::UnknownNode,
            format!("{path}.after"),
            format!("node `{after}` does not exist"),
            "use a stable node name from the validated graph",
        ));
    }
    if before == after {
        return Err(failure(
            DiagnosticCode::SelfEdge,
            path,
            format!("node `{before}` cannot depend on itself"),
            "remove the self edge or name two distinct nodes",
        ));
    }
    Ok(())
}

fn canonical_graph(graph: &ProgramGraph) -> Result<ProgramGraph, CompileError> {
    let mut result = ProgramGraph::new();
    let mut value_map = BTreeMap::<GraphValueId, GraphValueId>::new();
    let mut external: Vec<_> = graph.values().iter().filter(|value| value.producer.is_none()).collect();
    external.sort_by(|left, right| left.name.cmp(&right.name));
    for value in external {
        let id = result
            .add_external_value(value.name.clone(), value.contract.clone())
            .map_err(graph_failure)?;
        value_map.insert(value.id, id);
    }
    let mut pending: BTreeMap<&str, _> = graph.nodes().iter().map(|node| (node.name.as_str(), node)).collect();
    while !pending.is_empty() {
        let ready_name = pending
            .iter()
            .find(|(_, node)| node.inputs.iter().all(|input| value_map.contains_key(&input.value)))
            .map(|(name, _)| *name)
            .ok_or_else(|| failure(
                DiagnosticCode::DependencyCycle,
                "graph",
                "no canonical topological node is available",
                "remove cyclic semantic ordering",
            ))?;
        let node = pending.remove(ready_name).expect("selected pending node");
        let mut inputs = node.inputs.clone();
        inputs.sort_by(|left, right| left.buffer.cmp(&right.buffer).then_with(|| left.value.cmp(&right.value)));
        for input in &mut inputs {
            input.value = value_map[&input.value];
        }
        let old_by_name: BTreeMap<&str, GraphValueId> = node
            .outputs
            .iter()
            .zip(&node.output_ports)
            .map(|(id, port)| (port.name.as_str(), *id))
            .collect();
        let mut outputs = node.output_ports.clone();
        outputs.sort_by(|left, right| left.name.cmp(&right.name));
        for output in &mut outputs {
            output.state_successor_of = output.state_successor_of.map(|prior| value_map[&prior]);
        }
        let (_, new_ids) = result
            .add_node(
                node.name.clone(),
                node.program.canonicalized(),
                inputs,
                outputs.clone(),
            )
            .map_err(graph_failure)?;
        for (output, new_id) in outputs.iter().zip(new_ids) {
            value_map.insert(old_by_name[output.name.as_str()], new_id);
        }
    }
    Ok(result)
}

fn graph_failure(error: vyre_foundation::ir::ProgramGraphError) -> CompileError {
    failure(
        DiagnosticCode::InvalidProgram,
        "graph",
        error.to_string(),
        "supply a canonical-wire-compatible validated graph",
    )
}

fn build_fusion(
    nodes: &[NodeRecord],
    geometry: &[GeometryRecord],
    dependencies: &[DependencyEdge],
    permissions: &[FusionPermission],
    node_ids: &BTreeMap<&str, ArtifactNodeId>,
) -> Result<(Vec<FusionRecord>, Vec<FusionGroupId>), CompileError> {
    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    let direct: BTreeSet<(ArtifactNodeId, ArtifactNodeId)> = dependencies
        .iter()
        .filter_map(|edge| match (edge.from, edge.to) {
            (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) => Some((from, to)),
            _ => None,
        })
        .collect();
    for (index, permission) in permissions.iter().enumerate() {
        let before = node_ids[permission.before.as_str()];
        let after = node_ids[permission.after.as_str()];
        if !direct.contains(&(before, after)) {
            return Err(failure(
                DiagnosticCode::UnconnectedFusion,
                format!("options.fusion_permissions[{index}]"),
                "fusion permission does not follow a direct dependency",
                "supply evidence only for directly dependent nodes",
            ));
        }
        if geometry[before.0 as usize].workgroup_size != geometry[after.0 as usize].workgroup_size {
            return Err(failure(
                DiagnosticCode::UnconnectedFusion,
                format!("options.fusion_permissions[{index}]"),
                "fusion permission spans unequal program-declared geometry",
                "retain separate groups or align semantic geometry before compilation",
            ));
        }
        union(&mut parent, before.0 as usize, after.0 as usize);
    }
    for index in 0..parent.len() {
        parent[index] = find(&mut parent, index);
    }
    let mut members = BTreeMap::<usize, Vec<ArtifactNodeId>>::new();
    for (index, root) in parent.iter().copied().enumerate() {
        members.entry(root).or_default().push(ArtifactNodeId(index as u32));
    }
    let mut groups: Vec<_> = members.into_values().collect();
    groups.sort_by_key(|group| group[0]);
    let mut node_groups = vec![FusionGroupId(0); nodes.len()];
    let fusion = groups
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            let id = FusionGroupId(index as u32);
            for node in &group {
                node_groups[node.0 as usize] = id;
            }
            let node_set: BTreeSet<_> = group.iter().copied().collect();
            let mut legality: Vec<_> = permissions
                .iter()
                .filter(|permission| {
                    node_set.contains(&node_ids[permission.before.as_str()])
                        && node_set.contains(&node_ids[permission.after.as_str()])
                })
                .map(|permission| permission.legality_digest)
                .collect();
            legality.sort();
            legality.dedup();
            FusionRecord { id, members: group, legality }
        })
        .collect::<Vec<_>>();
    ensure_group_dag(fusion.len(), dependencies, &node_groups, DiagnosticCode::FusionCycle)?;
    Ok((fusion, node_groups))
}

fn find(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left = find(parent, left);
    let right = find(parent, right);
    if left != right {
        let (root, child) = if left < right { (left, right) } else { (right, left) };
        parent[child] = root;
    }
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
    group_stages_inner(count, dependencies, node_groups).map(|_| ()).map_err(|_| {
        failure(
            code,
            "artifact.dependencies",
            "dependency quotient contains a cycle",
            "remove the cyclic order constraint or fusion permission",
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
            DiagnosticCode::FusionCycle,
            "artifact.fusion",
            "fused dependency quotient contains a cycle",
            "remove a fusion permission that spans an intervening dependency",
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
        let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to) else {
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
            let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to) else {
                continue;
            };
            let from_stage = stages[node_groups[from.0 as usize].0 as usize];
            let to_stage = stages[node_groups[to.0 as usize].0 as usize];
            if from_stage < after_stage && to_stage == after_stage {
                edge_ids.push(u32::try_from(index).map_err(|_| overflow("artifact.dependencies", "edge identity exceeds u32"))?);
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
    value_ids: &BTreeMap<&str, ArtifactValueId>,
    old_node_name: &BTreeMap<GraphNodeId, &str>,
    node_ids: &BTreeMap<&str, ArtifactNodeId>,
    node_groups: &[FusionGroupId],
    stages: &[u32],
) -> Vec<MaterializationRecord> {
    let mut records = Vec::new();
    for value in graph.values() {
        let Some(producer) = value.producer else { continue };
        let producer_node = node_ids[old_node_name[&producer]];
        let producer_group = node_groups[producer_node.0 as usize];
        let producer_stage = stages[producer_group.0 as usize];
        let cross_group = value.consumers.iter().any(|consumer| {
            let consumer_node = node_ids[old_node_name[consumer]];
            node_groups[consumer_node.0 as usize] != producer_group
        });
        let reason = match value.contract.lifetime {
            ValueLifetime::Output => Some(MaterializationReason::Output),
            ValueLifetime::SequenceState => Some(MaterializationReason::RetainedState),
            _ if cross_group => Some(MaterializationReason::CrossGroupUse),
            _ => None,
        };
        if let Some(reason) = reason {
            records.push(MaterializationRecord {
                value: value_ids[value.name.as_str()],
                producer: producer_group,
                stage: producer_stage,
                reason,
            });
        }
    }
    records.sort_by_key(|record| (record.value, record.reason as u8));
    records
}

#[allow(clippy::too_many_arguments)]
fn build_resources(
    graph: &ProgramGraph,
    bindings: &BTreeMap<String, u64>,
    value_ids: &BTreeMap<&str, ArtifactValueId>,
    old_node_name: &BTreeMap<GraphNodeId, &str>,
    node_ids: &BTreeMap<&str, ArtifactNodeId>,
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
            let node = node_ids[old_node_name[&producer]];
            stages[node_groups[node.0 as usize].0 as usize]
        });
        let mut last_stage = value
            .consumers
            .iter()
            .map(|consumer| {
                let node = node_ids[old_node_name[consumer]];
                stages[node_groups[node.0 as usize].0 as usize]
            })
            .max()
            .unwrap_or(producer_stage);
        if matches!(value.contract.lifetime, ValueLifetime::Output | ValueLifetime::SequenceState) {
            last_stage = last_stage.max(final_stage);
        }
        resources.push(ResourceRecord {
            value: value_ids[value.name.as_str()],
            name: value.name.clone(),
            element_count,
            byte_count,
            lifetime: match value.contract.lifetime {
                ValueLifetime::ImmutableWeight => ResourceLifetime::Immutable,
                ValueLifetime::Invocation => ResourceLifetime::Invocation,
                ValueLifetime::SequenceState => ResourceLifetime::Retained,
                ValueLifetime::Output => ResourceLifetime::Output,
            },
            first_stage: producer_stage,
            last_stage,
        });
    }
    resources.sort_by_key(|resource| resource.value);
    let total_bytes = resources.iter().try_fold(0u64, |total, resource| {
        total.checked_add(resource.byte_count).ok_or_else(|| {
            overflow("artifact.resource_envelope.total_bytes", "resource sum exceeds u64")
        })
    })?;
    let mut peak_live_bytes = 0u64;
    for stage in 0..=final_stage {
        let live = resources
            .iter()
            .filter(|resource| resource.first_stage <= stage && stage <= resource.last_stage)
            .try_fold(0u64, |total, resource| {
                total.checked_add(resource.byte_count).ok_or_else(|| {
                    overflow("artifact.resource_envelope.peak_live_bytes", "live resource sum exceeds u64")
                })
            })?;
        peak_live_bytes = peak_live_bytes.max(live);
    }
    Ok((resources, ResourceEnvelope { total_bytes, peak_live_bytes }))
}

fn encode_payload(payload: &ArtifactPayload) -> Result<Vec<u8>, CompileError> {
    let body = serde_json::to_vec(payload).map_err(serialization_failure)?;
    let body_len = u32::try_from(body.len()).map_err(|_| {
        overflow("artifact.body", "canonical body exceeds the u32 framing limit")
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
