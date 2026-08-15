//! The subcommands that own the generated documentation surface.
//!
//! Each of these writes or checks a file under `docs/` or a generated section of
//! a crate README: the canonical operation schema and the browsing views built
//! from it, the optimizer pass reference, the op matrix, the research source
//! ledger, the documentation lifecycle gate, and the command-line contract every
//! shipped binary answers for.

pub mod cli_docs;
pub mod docs_check;
pub mod research_key;
pub mod research_source_ledger;

use crate::gate::Gate;

/// Every gate this module owns.
pub static GATES: &[&dyn Gate] = &[&cli_docs::CliDocs, &docs_check::DocsCheck];
