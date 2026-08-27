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
mod certificate;
mod compile;
mod constraints;
/// Open, reproducible whole-program candidate cost model.
pub mod cost;
mod dependency_order;
mod derive;
mod device_facts;
mod envelope;
mod error;
mod execution;
mod facts;
mod frame;
#[cfg(test)]
#[path = "../tests/geometry_fixtures/mod.rs"]
mod geometry_fixtures;
/// Versioned production grammar candidate search derives plans from, re-exported
/// as `ScheduleProduction`, `DerivationStep` and `SCHEDULE_GRAMMAR_VERSION`.
mod grammar;
#[cfg(test)]
#[path = "../tests/graph_fixtures/mod.rs"]
mod graph_fixtures;
/// Whole-grid fence detection, and the planner cut that removes it.
pub mod grid_sync;
mod identity;
/// Stable semantic legality decisions for whole-program fusion.
pub mod legality;
mod level_stage;
/// Versioned protocol budgeted device measurement runs under, and the evidence
/// one measured selection retains.
pub mod measure;
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
pub use certificate::{DerivedFamily, PruneReason, PrunedFamily, SearchCertificate};
pub use compile::{compile, compile_measured, EmittedResources, FinalistEvaluator};
pub use device_facts::DeviceFacts;
pub use envelope::{
    ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetProfile,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
    ARTIFACT_ENVELOPE_SCHEMA_VERSION, TARGET_PAYLOAD_SCHEMA_VERSION,
};
pub use error::CompileError;
pub use execution::{
    execute_single_program, writable_graph_value_buffers, writable_graph_values, CompileObjective,
    SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionPolicy,
    SemanticExecutionRequest, SemanticExecutor, SingleProgramExecutionOutput,
};
pub use grammar::{DerivationStep, ScheduleProduction, SCHEDULE_GRAMMAR_VERSION};
pub use identity::{
    ArtifactNodeId, ArtifactValueId, DependencyEdge, DependencyEndpoint, DependencyKind, Digest,
    FusionGroupId,
};
pub use level_stage::{registered_level_stage, PayloadAttachment};
pub use request::{
    CompileRequest, ExternalFacts, SearchBudget, SearchWork, ValidatedCompileRequest,
};
pub use schema::{
    AbiAccess, Artifact, ArtifactAbi, BarrierPhaseRecord, BarrierRecord, EntryAbiRecord,
    EntryPersistence, EntryResourceBinding, ExecutionMode, FusionRecord, FusionRejection,
    GeometryRecord, LaunchResourceIntent, MaterializationReason, MaterializationRecord, NodeRecord,
    PlanMeasurement, Provenance, ResourceAbiRecord, ResourceEnvelope, ResourceLifetime,
    ResourceNameCollision, ResourceRecord, SelectedPlan, WorkspacePlan, WorkspaceRegion,
    ARTIFACT_SCHEMA_VERSION, WORKSPACE_REGION_ALIGNMENT,
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
