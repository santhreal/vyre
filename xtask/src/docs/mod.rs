//! The subcommands that own the generated documentation surface.
//!
//! Each of these writes or checks a file under `docs/`: the canonical
//! operation schema and the browsing views built from it, the optimizer pass
//! reference, the op matrix, the research source ledger, and the
//! documentation lifecycle gate.

pub mod docs_check;
pub mod research_key;
pub mod research_source_ledger;
