//! The generated documentation views built from the live operation registry.
//!
//! The operation schema is read out of the registry rather than parsed from
//! source, and the catalog, the op matrix, the op list and the optimizer pass
//! reference are all rendered from it. Every view emits TOML or JSON, so the
//! markdown cell renderers the catalog and the op inventory once shared are
//! gone with the tables they filled. The documentation lifecycle gate itself
//! reads only files and stays in `xtask::docs`.

pub mod catalog;
pub mod error_codes;
pub mod list_ops;
pub mod op_matrix;
pub mod operation_schema;
pub mod optimization_docs;
