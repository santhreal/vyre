//! Causal Introspection & Unified Causal-Span Schema (Backlog Row 117).
//!
//! Provides stable typed identities connecting source regions to schedule operators,
//! physical instructions, payload entries, resources, measurements, and diagnostics.
//! Supports compile-time-off, sampled, and full modes with zero allocation when disabled.

pub mod event;
pub mod id;
pub mod mode;
pub mod phase;
pub mod receipt;
pub mod tracer;

pub use event::{AlternativeSchedule, CacheHitRecord, CausalEvent, CounterfactualDecision, PruneReason};
pub use id::{CausalSpanId, SourceSpanRef, TraceId};
pub use mode::CausalTraceMode;
pub use phase::{exhaustiveness_check_causal_phase, CausalPhase};
pub use receipt::{CausalError, CausalReceipt, CAUSAL_RECEIPT_SCHEMA_VERSION};
pub use tracer::{next_span_id, CausalTracer};
