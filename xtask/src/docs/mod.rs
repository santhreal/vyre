//! The subcommands that own the generated documentation surface.
//!
//! Each of these writes or checks a file under `docs/`: the canonical
//! operation schema and the browsing views built from it, the optimizer pass
//! reference, the op matrix, the research source ledger, and the
//! documentation lifecycle gate.

pub(crate) mod catalog;
pub(crate) mod docs_check;
pub(crate) mod list_ops;
pub(crate) mod op_matrix;
pub(crate) mod operation_schema;
pub(crate) mod optimization_docs;
pub(crate) mod research_key;
pub(crate) mod research_source_ledger;
