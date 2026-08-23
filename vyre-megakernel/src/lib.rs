//! Backend-neutral compilation from validated typed graphs to immutable artifacts.
//!
//! # Ownership
//!
//! This crate owns the selected-schedule and target-payload stages:
//! - input: a validated schedule-free
//!   [`LogicalProgramGraph`](vyre_foundation::logical::LogicalProgramGraph),
//!   immutable [`ExternalFacts`], and explicit [`SearchBudget`]
//! - output: one validated [`SelectedPlan`] in a versioned immutable [`Artifact`]
//!   plus optional authenticated [`TargetPayload`] values in an
//!   [`ArtifactEnvelope`]
//!
//! Device admission, materialization, submission, queues, residency, and recovery
//! are consumers of this compiler product and do not alter artifact identity.
//!
//! [`ProgramGraph`]: vyre_foundation::ir::ProgramGraph

mod artifact;
mod candidate;
mod compile;
/// Open, reproducible whole-program candidate cost model.
pub mod cost;
mod dependency_order;
mod device_facts;
mod envelope;
mod error;
mod facts;
mod frame;
#[cfg(test)]
#[path = "../tests/graph_fixtures/mod.rs"]
mod graph_fixtures;
/// Whole-grid fence detection, and the planner cut that removes it.
pub mod grid_sync;
mod identity;
/// Stable semantic legality decisions for whole-program fusion.
pub mod legality;
mod normalize;
mod request;
mod request_identity;
mod resource_records;
mod schema;
mod search;
mod select;
/// Target compiler facets over compiler-selected modules and canonical ABI.
pub(crate) mod target;

pub use candidate::{ExecutionTopology, ResidentPartitionMode};
pub use compile::{compile, compile_measured, FinalistEvaluator};
pub use device_facts::DeviceFacts;
pub use envelope::{
    ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetProfile,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
    ARTIFACT_ENVELOPE_SCHEMA_VERSION, TARGET_PAYLOAD_SCHEMA_VERSION,
};
pub use error::CompileError;
pub use identity::{
    ArtifactNodeId, ArtifactValueId, DependencyEdge, DependencyEndpoint, DependencyKind, Digest,
    FusionGroupId,
};
pub use request::{
    CompileRequest, ExternalFacts, SearchBudget, SearchWork, ValidatedCompileRequest,
};
pub use schema::{
    AbiAccess, Artifact, ArtifactAbi, BarrierRecord, EntryAbiRecord, EntryResourceBinding,
    ExecutionMode, FusionRecord, FusionRejection, GeometryRecord, MaterializationReason,
    MaterializationRecord, NodeRecord, PlanMeasurement, Provenance, ResourceAbiRecord,
    ResourceEnvelope, ResourceLifetime, ResourceNameCollision, ResourceRecord, SelectedPlan,
    ARTIFACT_SCHEMA_VERSION,
};
pub use target::SelectedModule;
pub use target::{
    attach_target, compile_selected_modules, EmittedTargetModule, SelectedLowering,
    TargetCompileError, TargetCompiler, TargetModuleBundle, TargetModuleImage,
    TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
};
pub use vyre_foundation::diagnostics::Diagnostic;

pub(crate) use dependency_order::{build_barriers, ensure_node_dag, group_stages};
pub(crate) use device_facts::workgroup_scratch_declarations;
pub(crate) use error::{failure, CompilerFailureKind};
pub(crate) use identity::domain_digest;
pub(crate) use resource_records::{build_materializations, value_byte_count};
