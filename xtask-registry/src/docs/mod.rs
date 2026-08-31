//! The generated documentation views built from the live operation registry.
//!
//! The operation schema is read out of the registry rather than parsed from
//! source, and the catalog, the op matrix, the op list and the optimizer pass
//! reference are all rendered from it. The documentation lifecycle gate itself
//! reads only files and stays in `xtask::docs`.

pub mod catalog;
pub mod error_codes;
pub mod list_ops;
pub mod op_matrix;
pub mod operation_schema;
pub mod optimization_docs;
pub mod schema_cells;
