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
mod frame;
/// Whole-grid fence detection shared by the compiler and the driver.
pub mod grid_sync;
/// Stable semantic legality decisions for whole-program fusion.
pub mod legality;
mod normalize;
mod search;
mod select;
/// Target compiler facets over compiler-selected modules and canonical ABI.
pub(crate) mod target;

pub use envelope::{
    ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetProfile,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
    ARTIFACT_ENVELOPE_SCHEMA_VERSION, TARGET_PAYLOAD_SCHEMA_VERSION,
};
pub use target::SelectedModule;
pub use target::{
    attach_target, compile_selected_modules, EmittedTargetModule, SelectedLowering,
    TargetCompileError, TargetCompiler, TargetModuleBundle, TargetModuleImage,
    TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
};

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;
pub use vyre_foundation::diagnostics::Diagnostic;
use vyre_foundation::diagnostics::{DiagnosticStage, OpLocation, RetryClass};
use vyre_foundation::ir::{
    BufferAccess, DataType, GraphValueId, Program, ProgramGraph, ProgramGraphValue, ShapeDim,
    ValueLifetime,
};
use vyre_foundation::program_caps;
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};

/// Current canonical artifact schema.
pub const ARTIFACT_SCHEMA_VERSION: u16 = 5;
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

/// Live device facts the whole-program compiler selects against.
///
/// Every field is a fact about the device that will run the artifact. A zero
/// occupancy budget or launch cost means the backend reported no number for
/// it, and the cost term that field feeds is then omitted rather than guessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceFacts {
    capabilities: BackendCapabilities,
    supports_cooperative_launch: bool,
    supports_device_timestamps: bool,
    max_invocations_per_workgroup: u32,
    registers_per_invocation: u32,
    shared_scratch_bytes_per_workgroup: u32,
    per_launch_overhead_ns: u64,
    persistent_setup_overhead_ns: u64,
}

impl DeviceFacts {
    /// Facts for a caller that has no device.
    ///
    /// Every capability is absent and every budget is zero, so validation grants
    /// nothing: a program that needs a gated capability is rejected instead of
    /// being compiled against an assumed device. A zero budget is unknown rather
    /// than a limit of zero, so no size gate fires and no cost term is charged.
    /// Use this only where no backend is reachable; a caller holding a backend
    /// passes its live facts.
    #[must_use]
    pub const fn unknown() -> Self {
        Self::new(
            BackendCapabilities {
                supports_subgroup_ops: false,
                supports_indirect_dispatch: false,
                supports_specialization_constants: false,
                supports_distributed_collectives: false,
                has_mul_high: false,
                has_dual_issue_fp32_int32: false,
                has_tensor_core_int: false,
                has_native_f16: false,
                has_warp_shuffle: false,
                has_shared_memory: false,
                has_transcendental_polynomial_emit: false,
                max_native_int_width: 0,
            },
            0,
        )
    }

    /// Construct facts from the live capability snapshot and invocation limit.
    ///
    /// Cooperative launch, launch timestamps, occupancy budgets, and launch
    /// costs start absent. A backend that measures one supplies it through the
    /// matching `with_` method.
    #[must_use]
    pub const fn new(capabilities: BackendCapabilities, max_invocations_per_workgroup: u32) -> Self {
        Self {
            capabilities,
            supports_cooperative_launch: false,
            supports_device_timestamps: false,
            max_invocations_per_workgroup,
            registers_per_invocation: 0,
            shared_scratch_bytes_per_workgroup: 0,
            per_launch_overhead_ns: 0,
            persistent_setup_overhead_ns: 0,
        }
    }

    /// Record whether the device can launch a cooperative grid.
    #[must_use]
    pub const fn with_cooperative_launch(mut self, supported: bool) -> Self {
        self.supports_cooperative_launch = supported;
        self
    }

    /// Record whether the device timestamps a launch on the device itself.
    #[must_use]
    pub const fn with_device_timestamps(mut self, supported: bool) -> Self {
        self.supports_device_timestamps = supported;
        self
    }

    /// Record the per-invocation register budget and the per-workgroup
    /// shared-scratch budget.
    #[must_use]
    pub const fn with_occupancy(
        mut self,
        registers_per_invocation: u32,
        shared_scratch_bytes_per_workgroup: u32,
    ) -> Self {
        self.registers_per_invocation = registers_per_invocation;
        self.shared_scratch_bytes_per_workgroup = shared_scratch_bytes_per_workgroup;
        self
    }

    /// Record measured host launch cost and persistent-mode setup cost.
    #[must_use]
    pub const fn with_launch_costs(
        mut self,
        per_launch_overhead_ns: u64,
        persistent_setup_overhead_ns: u64,
    ) -> Self {
        self.per_launch_overhead_ns = per_launch_overhead_ns;
        self.persistent_setup_overhead_ns = persistent_setup_overhead_ns;
        self
    }

    /// Live IR capability snapshot advertised by the device.
    #[must_use]
    pub const fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }

    /// Whether a whole-grid fence can run inside one kernel on this device.
    #[must_use]
    pub const fn supports_cooperative_launch(&self) -> bool {
        self.supports_cooperative_launch
    }

    /// Whether a search measurement can carry a device timestamp.
    #[must_use]
    pub const fn supports_device_timestamps(&self) -> bool {
        self.supports_device_timestamps
    }

    /// Largest legal invocation count in one workgroup.
    #[must_use]
    pub const fn max_invocations_per_workgroup(&self) -> u32 {
        self.max_invocations_per_workgroup
    }

    /// Registers one invocation holds before the target compiler spills, or zero
    /// when the backend reports no budget.
    #[must_use]
    pub const fn registers_per_invocation(&self) -> u32 {
        self.registers_per_invocation
    }

    /// Shared scratch bytes one workgroup holds, or zero when the backend
    /// reports no budget.
    #[must_use]
    pub const fn shared_scratch_bytes_per_workgroup(&self) -> u32 {
        self.shared_scratch_bytes_per_workgroup
    }

    /// Host cost of one kernel launch in nanoseconds, or zero when unmeasured.
    #[must_use]
    pub const fn per_launch_overhead_ns(&self) -> u64 {
        self.per_launch_overhead_ns
    }

    /// One-time cost of bringing up persistent execution in nanoseconds, or
    /// zero when unmeasured.
    #[must_use]
    pub const fn persistent_setup_overhead_ns(&self) -> u64 {
        self.persistent_setup_overhead_ns
    }
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
    /// Launches the caller will submit against this artifact.
    ///
    /// Persistent execution pays a one-time setup cost and saves one launch
    /// overhead per submission, so the count the caller expects decides whether
    /// that trade is profitable. One submission never amortizes it.
    pub expected_launch_batch: u32,
}

impl ExternalFacts {
    /// Construct external facts with no constant identities and one launch.
    #[must_use]
    pub fn new(configuration_digest: Digest, symbolic_bindings: BTreeMap<String, u64>) -> Self {
        Self {
            configuration_digest,
            symbolic_bindings,
            constant_identities: BTreeMap::new(),
            expected_launch_batch: 1,
        }
    }

    /// Record how many launches the caller will submit against the artifact.
    #[must_use]
    pub fn with_expected_launch_batch(mut self, expected_launch_batch: u32) -> Self {
        self.expected_launch_batch = expected_launch_batch;
        self
    }
}

/// Unvalidated whole-program compilation request.
pub struct CompileRequest {
    graph: ProgramGraph,
    facts: ExternalFacts,
    device: DeviceFacts,
    search_budget: SearchBudget,
    max_artifact_bytes: u64,
}

impl CompileRequest {
    /// Construct a request. Call [`Self::validate`] before compilation.
    #[must_use]
    pub const fn new(
        graph: ProgramGraph,
        facts: ExternalFacts,
        device: DeviceFacts,
        search_budget: SearchBudget,
        max_artifact_bytes: u64,
    ) -> Self {
        Self {
            graph,
            facts,
            device,
            search_budget,
            max_artifact_bytes,
        }
    }

    /// Validate topology, programs, device facts, external facts, and bounds.
    pub fn validate(self) -> Result<ValidatedCompileRequest, CompileError> {
        if self.max_artifact_bytes == 0 {
            return Err(failure(
                CompilerFailureKind::ArtifactLimit,
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
                CompilerFailureKind::InvalidSearchBudget,
                "request.search_budget",
                "candidate, CPU-work, and elapsed-work bounds must be positive",
                "supply explicit positive bounds for every mandatory search dimension",
            ));
        }
        if self.facts.expected_launch_batch == 0 {
            return Err(failure(
                CompilerFailureKind::InvalidDeviceFacts,
                "request.facts.expected_launch_batch",
                "expected launch batch is zero, so the artifact would never run",
                "supply the number of launches the caller will submit, at least one",
            ));
        }
        self.graph.analyze().map_err(|error| {
            failure(
                CompilerFailureKind::InvalidProgram,
                "request.graph",
                error.to_string(),
                "supply a structurally valid acyclic ProgramGraph",
            )
        })?;
        validate_node_programs(&self.graph, self.device.capabilities)?;
        validate_device_support(&self.graph, self.device)?;
        validate_bindings(&self.graph, &self.facts.symbolic_bindings)?;
        validate_constant_identities(&self.graph, &self.facts.constant_identities)?;
        Ok(ValidatedCompileRequest {
            graph: self.graph,
            facts: self.facts,
            device: self.device,
            search_budget: self.search_budget,
            max_artifact_bytes: self.max_artifact_bytes,
        })
    }
}

/// Validate every node program against the live capability snapshot.
fn validate_node_programs(
    graph: &ProgramGraph,
    capabilities: BackendCapabilities,
) -> Result<(), CompileError> {
    for node in graph.nodes() {
        let report = validate_with_options(
            &node.program,
            ValidationOptions::universal().with_backend_capabilities(capabilities),
        );
        if let Some(issue) = report.errors.into_iter().next() {
            let path = format!("request.graph.nodes[{}].program", node.id.0);
            let mut diagnostic = issue.diagnostic();
            if let Some(location) = diagnostic.location.as_mut() {
                location.path = Some(path);
                location.graph_node = Some(node.id.0);
            }
            return Err(CompileError { diagnostic });
        }
    }
    Ok(())
}

/// Reject a graph the live device cannot execute.
///
/// Foundation node validation covers the capability bits it knows about:
/// subgroup expressions and distributed collectives. This gate covers the rest
/// of the live snapshot, plus the two device facts no instruction expresses.
///
/// A whole-grid fence is a launch property, not an instruction property, so a
/// program that fences the grid on a device that cannot launch a cooperative
/// grid has no correct execution and is refused here instead of deadlocking at
/// dispatch. The declared workgroup is checked against the live invocation and
/// shared-scratch limits for the same reason: a group the device will not accept
/// is a compile-time fact, not a dispatch failure.
fn validate_device_support(graph: &ProgramGraph, device: DeviceFacts) -> Result<(), CompileError> {
    let capabilities = device.capabilities;
    for node in graph.nodes() {
        let path = format!("request.graph.nodes[{}].program", node.id.0);
        if grid_sync::requires_grid_sync(&node.program) && !device.supports_cooperative_launch {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                "program fences the whole grid but the device cannot launch a cooperative grid",
                "split the program at the grid fence into one node per segment, or compile for a device that reports cooperative launch",
            ));
        }
        let required = program_caps::scan(&node.program);
        let shared_scratch_bytes = workgroup_scratch_bytes(&node.program);
        let unmet = [
            (
                required.tensor_ops && !capabilities.has_tensor_core_int,
                "program uses tensor-core operands but the device reports no tensor-core integer support",
                "lower the tensor operation to scalar arithmetic, or compile for a device with tensor cores",
            ),
            (
                required.f16 && !capabilities.has_native_f16,
                "program uses binary16 operands but the device reports no native f16 arithmetic",
                "widen the f16 operands to f32, or compile for a device with native f16",
            ),
            (
                required.subgroup_ops && !capabilities.has_warp_shuffle,
                "program uses subgroup operations but the device reports no warp shuffle",
                "remove the subgroup operation, or compile for a device with warp-level shuffle",
            ),
            (
                required.indirect_dispatch && !capabilities.supports_indirect_dispatch,
                "program dispatches indirectly but the device reports no indirect dispatch",
                "resolve the dispatch extent on the host, or compile for a device with indirect dispatch",
            ),
            (
                shared_scratch_bytes > 0 && !capabilities.has_shared_memory,
                "program declares workgroup-scoped scratch but the device reports no shared memory",
                "move the scratch buffer to global memory, or compile for a device with shared memory",
            ),
        ];
        if let Some((_, message, fix)) = unmet.into_iter().find(|(unmet, _, _)| *unmet) {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                message,
                fix,
            ));
        }
        let declared = node.program.workgroup_size;
        let invocations = u64::from(declared[0])
            .saturating_mul(u64::from(declared[1]))
            .saturating_mul(u64::from(declared[2]));
        if device.max_invocations_per_workgroup > 0
            && invocations > u64::from(device.max_invocations_per_workgroup)
        {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                format!(
                    "program declares {invocations} invocations per workgroup; the device accepts {}",
                    device.max_invocations_per_workgroup
                ),
                "declare a workgroup within the live device invocation limit",
            ));
        }
        if device.shared_scratch_bytes_per_workgroup > 0
            && shared_scratch_bytes > u64::from(device.shared_scratch_bytes_per_workgroup)
        {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                format!(
                    "program declares {shared_scratch_bytes} workgroup scratch bytes; the device accepts {}",
                    device.shared_scratch_bytes_per_workgroup
                ),
                "reduce the workgroup-scoped scratch to the live device budget",
            ));
        }
    }
    Ok(())
}

/// Workgroup-scoped scratch bytes one program declares.
fn workgroup_scratch_bytes(program: &Program) -> u64 {
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access == BufferAccess::Workgroup)
        .fold(0_u64, |total, buffer| {
            let count = usize::try_from(buffer.count).unwrap_or(usize::MAX);
            let bytes = buffer
                .element
                .packed_size_bytes(count)
                .ok()
                .flatten()
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(0);
            total.saturating_add(bytes)
        })
}

/// A graph and complete immutable facts that passed request validation.
pub struct ValidatedCompileRequest {
    graph: ProgramGraph,
    facts: ExternalFacts,
    device: DeviceFacts,
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

    /// Return the live device facts the plan was selected against.
    #[must_use]
    pub const fn device(&self) -> DeviceFacts {
        self.device
    }
}

/// Compiler-internal failure classification projected into the shared diagnostic protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CompilerFailureKind {
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
    /// Device facts contradict what any device can report.
    InvalidDeviceFacts,
    /// A finalist could not be compiled for the target or timed on the device.
    FinalistEvaluation,
}

impl CompilerFailureKind {
    /// Stable ASCII code for logs and serialized evidence.
    #[must_use]
    const fn as_str(self) -> &'static str {
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
            Self::InvalidDeviceFacts => "MKC025_INVALID_DEVICE_FACTS",
            Self::FinalistEvaluation => "MKC026_FINALIST_EVALUATION",
        }
    }
}

const fn diagnostic_stage(code: CompilerFailureKind) -> DiagnosticStage {
    match code {
        CompilerFailureKind::InvalidProgram
        | CompilerFailureKind::MissingSymbol
        | CompilerFailureKind::UnknownSymbol
        | CompilerFailureKind::InvalidSearchBudget
        | CompilerFailureKind::MissingConstantIdentity
        | CompilerFailureKind::UnknownConstantIdentity
        | CompilerFailureKind::InvalidDeviceFacts => DiagnosticStage::Validate,
        CompilerFailureKind::DependencyCycle | CompilerFailureKind::FinalistEvaluation => {
            DiagnosticStage::Plan
        }
        CompilerFailureKind::ResourceOverflow | CompilerFailureKind::UnsizedResource => {
            DiagnosticStage::Lower
        }
        CompilerFailureKind::ArtifactLimit => DiagnosticStage::Emit,
        CompilerFailureKind::MalformedArtifact
        | CompilerFailureKind::VersionSkew
        | CompilerFailureKind::DigestMismatch
        | CompilerFailureKind::MalformedTargetPayload
        | CompilerFailureKind::TargetPayloadVersionSkew
        | CompilerFailureKind::TargetPayloadDigestMismatch
        | CompilerFailureKind::TargetPayloadAssociationMismatch
        | CompilerFailureKind::IncompatibleTargetPayload => DiagnosticStage::Admit,
    }
}

const fn diagnostic_retry(code: CompilerFailureKind) -> RetryClass {
    match code {
        CompilerFailureKind::VersionSkew
        | CompilerFailureKind::TargetPayloadVersionSkew
        | CompilerFailureKind::IncompatibleTargetPayload => RetryClass::RecompileSource,
        _ => RetryClass::Never,
    }
}

/// Compilation or artifact-validation failure.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{diagnostic}")]
pub struct CompileError {
    /// Structured stable diagnostic.
    pub diagnostic: Diagnostic,
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
        Ok(Self {
            payload,
            digest: Digest(decoded.digest),
        })
    }
}

/// Everything one compilation derives once and every finalist reuses.
struct CompileContext {
    source_graph: Digest,
    nodes: Vec<NodeRecord>,
    dependencies: Vec<DependencyEdge>,
    facts: facts::PlanningFacts,
    ranked: Vec<select::Selection>,
    pruned_fusions: Vec<FusionRejection>,
    work: SearchWork,
}

/// Rank every legal candidate for one validated request.
fn prepare(request: &ValidatedCompileRequest) -> Result<CompileContext, CompileError> {
    let canonical_wire = request.graph.to_wire().map_err(|error| {
        failure(
            CompilerFailureKind::InvalidProgram,
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
                    CompilerFailureKind::InvalidProgram,
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
    let normalized = normalize::normalize(&request.graph)?;
    let dependencies = normalized.dependencies;
    let planning_facts = facts::derive(
        &request.graph,
        &dependencies,
        &request.facts.symbolic_bindings,
    )?;
    let search = search::explore(
        &request.graph,
        &planning_facts,
        &dependencies,
        request.search_budget,
        request.device,
    );
    let ranked = select::rank(
        search.candidates,
        &planning_facts,
        &dependencies,
        request.device,
    );
    if ranked.is_empty() {
        return Err(failure(
            CompilerFailureKind::InvalidSearchBudget,
            "search.candidates",
            "schedule search scored no candidate plan",
            "raise the candidate bound so the unfused baseline plan is explored",
        ));
    }
    let pruned_fusions = search
        .rejected
        .into_iter()
        .map(|rejection| FusionRejection {
            from: rejection.edge.from,
            to: rejection.edge.to,
            value: rejection.edge.value,
            reason: rejection.reason,
        })
        .collect();
    Ok(CompileContext {
        source_graph,
        nodes,
        dependencies,
        facts: planning_facts,
        ranked,
        pruned_fusions,
        work: search.work,
    })
}

/// Turn one ranked candidate into a complete canonical artifact.
fn assemble(
    request: &ValidatedCompileRequest,
    context: &CompileContext,
    selection: &select::Selection,
    work: SearchWork,
    measurement: PlanMeasurement,
) -> Result<Artifact, CompileError> {
    let artifact::ArtifactPlan {
        node_groups,
        stages,
        geometry,
        selected_plan,
    } = artifact::plan(artifact::PlanInputs {
        graph: &request.graph,
        dependencies: &context.dependencies,
        facts: &context.facts,
        selection,
        pruned_fusions: &context.pruned_fusions,
        external: &request.facts,
        device: request.device,
        budget: request.search_budget,
        work,
        measurement,
    })?;
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
        source_graph: context.source_graph,
        request: domain_digest(REQUEST_DIGEST_DOMAIN, &request_bytes),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let payload = ArtifactPayload {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        nodes: context.nodes.clone(),
        dependencies: context.dependencies.clone(),
        selected_plan,
        abi,
        resources,
        resource_envelope,
        geometry,
        provenance,
    };
    let framed = encode_payload(&payload)?;
    let byte_len = u64::try_from(framed.bytes.len())
        .map_err(|_| overflow("artifact", "artifact length exceeds u64"))?;
    if byte_len > request.max_artifact_bytes {
        return Err(failure(
            CompilerFailureKind::ArtifactLimit,
            "artifact",
            format!(
                "canonical artifact is {byte_len} bytes; limit is {}",
                request.max_artifact_bytes
            ),
            "raise the explicit artifact bound or reduce the source graph",
        ));
    }
    Ok(Artifact {
        payload,
        digest: Digest(framed.digest),
    })
}

/// Compile one validated typed graph into a canonical backend-neutral artifact.
///
/// This path ranks candidates with the open cost model alone and records the
/// winner as [`PlanMeasurement::Unbudgeted`]. A request that budgets on-device
/// measurements is rejected here rather than compiled without spending them:
/// [`compile_measured`] is the only path that can honour that budget.
pub fn compile(request: &ValidatedCompileRequest) -> Result<Artifact, CompileError> {
    if request.search_budget.max_measurements > 0 {
        return Err(failure(
            CompilerFailureKind::InvalidSearchBudget,
            "request.search_budget.max_measurements",
            "analytic compilation cannot spend an on-device measurement budget",
            "compile through compile_measured with a finalist evaluator, or set max_measurements to zero",
        ));
    }
    let context = prepare(request)?;
    let selection = first_ranked(&context)?;
    assemble(
        request,
        &context,
        selection,
        context.work,
        PlanMeasurement::Unbudgeted,
    )
}

/// Device access the compiler borrows to time its finalists.
///
/// The compiler owns which plans are finalists and how their times are compared.
/// The caller owns the device: it supplies the target compiler that turns one
/// artifact into loadable bytes, and a launch that returns the device time of
/// one execution. Nothing here acquires a device, so a caller without one calls
/// [`compile`] instead.
pub trait FinalistEvaluator {
    /// Target compiler that turns one candidate artifact into target bytes.
    fn target_compiler(&self) -> &dyn TargetCompiler;

    /// Launch `payload` once and return the device time of that launch in
    /// nanoseconds. The time must come from the device, not the host clock.
    fn measure(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<u64, TargetCompileError>;
}

/// Compile with the ranked finalists compiled for the target and timed on the
/// device, selecting the plan with the lowest median device time.
///
/// The analytic ranking chooses which plans are worth a target compilation. The
/// top `max_target_compilations` of them are compiled and each launched
/// `max_measurements` times; the winner is the finalist with the lowest median.
/// [`SearchWork::target_compilations`] and [`SearchWork::measurements`] carry the
/// counts actually spent, and the recorded [`PlanMeasurement`] states whether a
/// measurement decided the plan at all: a zero measurement budget records
/// [`PlanMeasurement::Unbudgeted`] and a device with no launch timestamps records
/// [`PlanMeasurement::UntimedDevice`], neither of which is reported as a measured
/// selection.
pub fn compile_measured(
    request: &ValidatedCompileRequest,
    evaluator: &dyn FinalistEvaluator,
) -> Result<Artifact, CompileError> {
    let context = prepare(request)?;
    let budget = request.search_budget;
    if budget.max_measurements == 0 || budget.max_target_compilations == 0 {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
            context.work,
            PlanMeasurement::Unbudgeted,
        );
    }
    if !request.device.supports_device_timestamps() {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
            context.work,
            PlanMeasurement::UntimedDevice,
        );
    }

    let finalists = context
        .ranked
        .len()
        .min(budget.max_target_compilations as usize);
    let started = Instant::now();
    let mut work = context.work;
    let mut winner: Option<(usize, u64, u32)> = None;
    for index in 0..finalists {
        let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if elapsed >= budget.max_elapsed_ns {
            break;
        }
        let provisional = assemble(
            request,
            &context,
            &context.ranked[index],
            context.work,
            PlanMeasurement::Unbudgeted,
        )?;
        let payload = evaluator
            .target_compiler()
            .compile(&provisional)
            .map_err(|error| finalist_failure(index, &error))?;
        work.target_compilations = work.target_compilations.saturating_add(1);
        let mut samples = Vec::with_capacity(budget.max_measurements as usize);
        for _ in 0..budget.max_measurements {
            let sample = evaluator
                .measure(&provisional, &payload)
                .map_err(|error| finalist_failure(index, &error))?;
            samples.push(sample);
            work.measurements = work.measurements.saturating_add(1);
        }
        samples.sort_unstable();
        let launches = u32::try_from(samples.len()).unwrap_or(u32::MAX);
        let Some(median) = samples.get(samples.len() / 2).copied() else {
            continue;
        };
        if winner.is_none_or(|(_, best, _)| median < best) {
            winner = Some((index, median, launches));
        }
    }
    work.elapsed_ns = work
        .elapsed_ns
        .saturating_add(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    match winner {
        Some((index, median_ns, launches)) => assemble(
            request,
            &context,
            &context.ranked[index],
            work,
            PlanMeasurement::Measured {
                launches,
                median_ns,
            },
        ),
        None => assemble(
            request,
            &context,
            first_ranked(&context)?,
            work,
            PlanMeasurement::Unbudgeted,
        ),
    }
}

fn first_ranked(context: &CompileContext) -> Result<&select::Selection, CompileError> {
    context.ranked.first().ok_or_else(|| {
        failure(
            CompilerFailureKind::InvalidSearchBudget,
            "search.candidates",
            "schedule search scored no candidate plan",
            "raise the candidate bound so the unfused baseline plan is explored",
        )
    })
}

fn finalist_failure(index: usize, error: &TargetCompileError) -> CompileError {
    failure(
        CompilerFailureKind::FinalistEvaluation,
        format!("search.finalists[{index}]"),
        error.to_string(),
        "supply a finalist evaluator whose target compiler and device accept every ranked plan",
    )
}

/// Every fact that makes one compilation of one graph produce one artifact.
///
/// Device facts belong here because the plan is selected against them: the same
/// graph compiled for a device with a different capability snapshot, invocation
/// limit, occupancy budget, or launch cost is a different compilation and must
/// not reuse a cached artifact.
#[derive(Serialize)]
struct RequestIdentity<'a> {
    configuration_digest: Digest,
    symbolic_bindings: &'a BTreeMap<String, u64>,
    constant_identities: Vec<(u32, Digest)>,
    expected_launch_batch: u32,
    search_budget: SearchBudget,
    device_capabilities: DeviceCapabilityIdentity,
    device_cooperative_launch: bool,
    device_timestamps: bool,
    device_max_invocations_per_workgroup: u32,
    device_registers_per_invocation: u32,
    device_shared_scratch_bytes_per_workgroup: u32,
    device_per_launch_overhead_ns: u64,
    device_persistent_setup_overhead_ns: u64,
}

/// Serializable projection of the live capability snapshot.
///
/// [`BackendCapabilities`] is owned by the foundation validator and carries no
/// serialization, so artifact identity projects every field it exposes. A field
/// added there and not added here would silently share one artifact identity
/// between two devices that disagree.
#[derive(Serialize)]
struct DeviceCapabilityIdentity {
    supports_subgroup_ops: bool,
    supports_indirect_dispatch: bool,
    supports_specialization_constants: bool,
    supports_distributed_collectives: bool,
    has_mul_high: bool,
    has_dual_issue_fp32_int32: bool,
    has_tensor_core_int: bool,
    has_native_f16: bool,
    has_warp_shuffle: bool,
    has_shared_memory: bool,
    has_transcendental_polynomial_emit: bool,
    max_native_int_width: u32,
}

impl From<BackendCapabilities> for DeviceCapabilityIdentity {
    fn from(capabilities: BackendCapabilities) -> Self {
        let BackendCapabilities {
            supports_subgroup_ops,
            supports_indirect_dispatch,
            supports_specialization_constants,
            supports_distributed_collectives,
            has_mul_high,
            has_dual_issue_fp32_int32,
            has_tensor_core_int,
            has_native_f16,
            has_warp_shuffle,
            has_shared_memory,
            has_transcendental_polynomial_emit,
            max_native_int_width,
        } = capabilities;
        Self {
            supports_subgroup_ops,
            supports_indirect_dispatch,
            supports_specialization_constants,
            supports_distributed_collectives,
            has_mul_high,
            has_dual_issue_fp32_int32,
            has_tensor_core_int,
            has_native_f16,
            has_warp_shuffle,
            has_shared_memory,
            has_transcendental_polynomial_emit,
            max_native_int_width,
        }
    }
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
            expected_launch_batch: request.facts.expected_launch_batch,
            search_budget: request.search_budget,
            device_capabilities: request.device.capabilities().into(),
            device_cooperative_launch: request.device.supports_cooperative_launch(),
            device_timestamps: request.device.supports_device_timestamps(),
            device_max_invocations_per_workgroup: request.device.max_invocations_per_workgroup(),
            device_registers_per_invocation: request.device.registers_per_invocation(),
            device_shared_scratch_bytes_per_workgroup: request
                .device
                .shared_scratch_bytes_per_workgroup(),
            device_per_launch_overhead_ns: request.device.per_launch_overhead_ns(),
            device_persistent_setup_overhead_ns: request.device.persistent_setup_overhead_ns(),
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
            CompilerFailureKind::MissingSymbol,
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
            CompilerFailureKind::UnknownSymbol,
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
            CompilerFailureKind::MissingConstantIdentity,
            format!("request.facts.constant_identities.{}", id.0),
            "constant graph value has no verified content identity",
            "supply one digest keyed by the constant GraphValueId",
        ));
    }
    if let Some(id) = identities.keys().find(|id| !constants.contains(id)) {
        return Err(failure(
            CompilerFailureKind::UnknownConstantIdentity,
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
                        CompilerFailureKind::InvalidProgram,
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
    code: CompilerFailureKind,
) -> Result<(), CompileError> {
    let groups: Vec<_> = (0..count).map(|id| FusionGroupId(id as u32)).collect();
    ensure_group_dag(count, dependencies, &groups, code)
}

fn ensure_group_dag(
    count: usize,
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
    code: CompilerFailureKind,
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
            CompilerFailureKind::DependencyCycle,
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

/// Exact element count of one graph value under validated bindings.
fn value_element_count(
    value: &ProgramGraphValue,
    bindings: &BTreeMap<String, u64>,
) -> Result<u64, CompileError> {
    let mut element_count = 1u64;
    for dim in &value.contract.shape {
        let extent = match dim {
            ShapeDim::Known(extent) => *extent,
            ShapeDim::Symbol(symbol) => *bindings.get(symbol).ok_or_else(|| {
                failure(
                    CompilerFailureKind::MissingSymbol,
                    format!("graph.values[{}].shape", value.name),
                    format!("symbolic extent `{symbol}` has no exact binding"),
                    "bind every symbolic graph dimension before compilation",
                )
            })?,
        };
        element_count = element_count.checked_mul(extent).ok_or_else(|| {
            overflow(
                format!("graph.values[{}].shape", value.name),
                "shape element count exceeds u64",
            )
        })?;
    }
    Ok(element_count)
}

/// Exact packed byte length of one graph value under validated bindings.
///
/// The cost model prices materialized traffic in bytes and the resource records
/// report the same number, so both read it here.
fn value_byte_count(
    value: &ProgramGraphValue,
    bindings: &BTreeMap<String, u64>,
) -> Result<u64, CompileError> {
    let element_count = value_element_count(value, bindings)?;
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
                CompilerFailureKind::UnsizedResource,
                format!("graph.values[{}].dtype", value.name),
                "value representation has no fixed packed byte size",
                "resolve the representation to a fixed-width typed value before compilation",
            )
        })?;
    u64::try_from(byte_count).map_err(|_| {
        overflow(
            format!("graph.values[{}]", value.name),
            "packed byte count exceeds u64",
        )
    })
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
        let element_count = value_element_count(value, bindings)?;
        let byte_count = value_byte_count(value, bindings)?;
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

fn encode_payload(payload: &ArtifactPayload) -> Result<frame::Framed, CompileError> {
    let body = serde_json::to_vec(payload).map_err(serialization_failure)?;
    frame::ARTIFACT.encode(payload.schema_version, &body)
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
        CompilerFailureKind::MalformedArtifact,
        "artifact.body",
        error.to_string(),
        "use values representable by the canonical artifact schema",
    )
}

fn overflow(path: impl Into<String>, message: impl Into<String>) -> CompileError {
    failure(
        CompilerFailureKind::ResourceOverflow,
        path,
        message,
        "reduce resolved extents or split the graph before compilation",
    )
}

fn failure(
    code: CompilerFailureKind,
    path: impl Into<String>,
    message: impl Into<String>,
    fix: impl Into<String>,
) -> CompileError {
    let stage = diagnostic_stage(code);
    let retry = diagnostic_retry(code);
    CompileError {
        diagnostic: Diagnostic::error(code.as_str(), message.into())
            .with_stage(stage)
            .with_location(OpLocation::op("vyre-megakernel").with_path(path))
            .with_fix(fix.into())
            .with_retry(retry),
    }
}
