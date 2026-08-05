//! Backend-neutral C source ingestion and lowering to Vyre typed IR.
//!
//! This crate owns C parsing and semantic lowering only. It does not select a
//! backend, dispatch programs, emit executable objects, or manage runtime
//! services. Execution belongs to driver and harness crates that consume the
//! returned [`vyre_foundation::ir::Program`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// C source ingestion and typed-IR construction.
pub mod pipeline;

pub use pipeline::{
    lower_source, lower_translation_unit, parse_source, parse_source_bytes, CFrontendError,
    ParsedTranslationUnit, MAX_SOURCE_BYTES,
};
