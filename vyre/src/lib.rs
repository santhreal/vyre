#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Every lint below is allowed for a documented reason. New lints from
// nursery/pedantic/restriction are NOT auto-allowed  -  broad blanket allows
// were removed deliberately so that future clippy findings surface as CI
// warnings instead of being silently swallowed.
#![allow(
    // Auto-generated op wrappers replay derive attributes by design.
    clippy::duplicated_attributes,
    // GPU buffer layout types (bind-group slot tuples) are inherently complex.
    clippy::type_complexity,
    // Shader-side math and wire-format POD structs do intentional integer
    // casts; the conform gate verifies byte-identity with the CPU reference.
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    // Explicit clones on Copy improve readability in serial layers where
    // semantic ownership matters more than cycle count.
    clippy::clone_on_copy,
    // Three-branch comparisons are natural in range-check oracles.
    clippy::comparison_chain,
    // Vyre uses explicit invariant violations (expect/unwrap) with `Fix:`
    // prose  -  not graceful degradation  -  per the engineering standard.
    clippy::expect_used,
    // Generic collections take external hashers by design.
    clippy::implicit_hasher,
    // SHA/hash compressors use the canonical single-letter state vars
    // (a,b,c,d,e,f,g,h per FIPS 180-4).
    clippy::many_single_char_names,
    // Error prose is centralized in the `Error` enum; per-fn `# Errors`
    // sections duplicate that contract.
    clippy::missing_errors_doc,
    // Panics document invariant violations with `Fix:` prose inline.
    clippy::missing_panics_doc,
    // Template-generated ops don't always merit `#[must_use]`.
    clippy::must_use_candidate,
    // Builder APIs take owned values by design.
    clippy::needless_pass_by_value,
    // Indexed arithmetic is clearer than iterator chains for GPU-shape loops.
    clippy::needless_range_loop,
    // Generated target-text strings use `r##` for quote safety.
    clippy::needless_raw_string_hashes,
    // Type names repeat module names for cross-crate discoverability.
    clippy::module_name_repetitions,
    // `mod X` in `X.rs` is the canonical vyre module layout.
    clippy::module_inception,
    // Math code uses short similar names (a/A, x/X) by convention.
    clippy::similar_names,
    // Internal helpers with stdlib-adjacent names are intentional for clarity.
    clippy::should_implement_trait,
    // Enforcer dispatch arms can share a body but represent distinct cases.
    clippy::match_same_arms,
    // Hot paths in the pipeline assemble strings incrementally.
    clippy::format_push_string,
    // GPU kernel dispatchers take many parameters by design (buffer slots).
    clippy::too_many_arguments,
    // Hash compressors and regex compilers have long inlined bodies.
    clippy::too_many_lines,
    // Trait signatures force `&T` for small Copy types.
    clippy::trivially_copy_pass_by_ref,
    // `Result<T, E>` with a single error variant keeps the API
    // forward-compatible as new error variants land.
    clippy::unnecessary_wraps,
    // Or-patterns are expanded for readability in large match tables.
    clippy::unnested_or_patterns,
    // GPU buffer sizes like `0x12345678` are more readable without `_`
    // separators in shader contexts.
    clippy::unreadable_literal,
    // Prose doc comments use type names that clippy wants backticked; our
    // doc style sentences already read naturally.
    clippy::doc_markdown
)]
#![cfg_attr(not(test), deny(clippy::todo, clippy::unimplemented))]
//! # vyre
//!
//! Stable facade for frontend IR, whole-program compilation, authenticated
//! artifact materialization, typed submission, and reference semantics.
//!
//! Production execution starts from a validated `ProgramGraph`, compiles one
//! immutable artifact through [`compiler`], and materializes its authenticated
//! target payload through [`ArtifactSession`]. Raw `Program` dispatch remains
//! available only in explicit reference and conformance adapters.

/// The vyre Program model.
///
/// This module defines `Program`, the frozen, serializable model that every
/// frontend emits and every backend consumes. It has zero external
/// dependencies so that spec tools can parse it without pulling in GPU
/// libraries.
/// Public API re-export.
pub use vyre_foundation::ir;

/// Soundness lattice for dataflow primitives. Canonical home is
/// `vyre-foundation`; re-exported here so vyre-libs (and any downstream
/// consumer) reaches it via `vyre::soundness`. Per the LEGO discipline,
/// vyre never imports from domain dataflow crates  -  this is the originating definition.
pub use vyre_foundation::soundness;

// Layer 1 and Layer 2 operation specifications live in vyre-libs.
// The crate root remains the single stable import surface for consumers.

/// Wire-format CPU-reference byte ABI contract.
/// Public API re-export.
#[cfg(any(test, feature = "cpu-parity"))]
pub use vyre_foundation::cpu_op;
/// CPU reference implementations shared across backends.
/// Public API re-export.
#[cfg(any(test, feature = "cpu-parity"))]
pub use vyre_foundation::cpu_references;
/// Substrate-neutral memory ordering model.
/// Public API re-export.
pub use vyre_foundation::memory_model;
/// Substrate-neutral memory ordering type.
/// Public API re-export.
pub use vyre_foundation::MemoryOrdering;

/// Whole-program compiler request, artifact, payload, and target-facet APIs.
pub use vyre_megakernel as compiler;

/// Canonical compiler artifact and request types.
pub use vyre_megakernel::{
    Artifact, ArtifactEnvelope, CompileRequest, ExternalFacts, SearchBudget, TargetPayload,
    TargetPayloadFormat, ValidatedCompileRequest,
};

/// Authenticated artifact admission, materialization, submission, and recovery.
pub use vyre_runtime::{
    admit_artifact, admit_envelope, ArtifactAdmissionError, ArtifactSession, ArtifactSessionError,
    PersistentExecutor, RetainedArtifactSession, RetryClass,
};

/// Canonical scan compilation, session, paging, residency, and readback owner.
pub use vyre_scan as scan;

/// Distribution-aware runtime algorithm selection.
/// Public API re-export.
pub use vyre_driver::routing;

/// Substrate-neutral execution planning for performance and accuracy tracks.
/// Public API re-export.
pub use vyre_foundation::execution_plan;

/// Unified error types for the entire crate.
/// Public API re-export.
pub use vyre_driver::error;

/// Structured, machine-readable diagnostics.
/// Public API re-export.
pub use vyre_driver::diagnostics;

/// Backend trait surface  -  `VyreBackend`, `Executable`,
/// `Streamable`, `DispatchConfig`, `BackendError`,
/// `ErrorCode`. The whole backend contract every driver crate
/// implements against.
/// Public API re-export.
/// Public API re-export.
pub use vyre_driver::backend;
/// Re-export of the native scan match result type from the foundation crate.
/// Public API re-export.
/// Public API re-export.
pub use vyre_foundation::match_result;

/// Pipeline-mode dispatch: compile a Program once, dispatch repeatedly.
/// Public API re-export.
/// Public API re-export.
pub use vyre_driver::pipeline;

// Previously: pub mod bytecode  -  a 637-LOC stack-machine VM publicly
// re-exported from core. Deleted 2026-04-17. The NFA scan micro-interpreter
// that carried the remaining bytecode was deleted 2026-04-19. Rule evaluators
// compose ops in vyre IR directly. No interpreter surface remains in core.

pub use vyre_driver::{
    ArtifactInstance, BackendError, BackendRegistration, BindingSet, CompiledPipeline, Completion,
    Device, DeviceIdentity, DispatchConfig, Error, Executable, Memory, MemoryRef, OutputBuffers,
    ResidentGraphReuseTelemetry, ResidentGraphReuseTelemetryError, Submission, TypedDispatchExt,
    VyreBackend,
};

/// Persistent-thread dispatch policy for dispatch paths.
pub use vyre_driver::persistent::PersistentThreadMode;
/// Speculation policy for dispatch paths.
pub use vyre_driver::speculate::SpeculationMode;

/// Re-export of the core IR program type and validation entry point.
///
/// `Program` is the frozen IR container. `validate` is the function that
/// checks a program for structural and semantic correctness before it is
/// handed to a backend.
pub use ir::{validate, InterpCtx, NodeId, NodeStorage, OpId, Program, Value};

/// Re-export of the native scan match result type.
///
/// `Match` represents a byte-range hit produced by pattern-scanning engines.
pub use vyre_foundation::match_result::Match;

/// Domain-neutral byte-range type.
pub use vyre_foundation::ByteRange;
