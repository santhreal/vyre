//! The subcommands that own the generated documentation surface.
//!
//! Each of these writes or checks a file under `docs/` or a generated section of
//! a crate README: the canonical operation schema and the browsing views built
//! from it, the optimizer pass reference, the op matrix, the research source
//! ledger, the documentation lifecycle gate, and the command-line contract every
//! shipped binary answers for.

pub mod cli_docs;
pub mod docs_check;
pub mod docs_coupling;
pub mod docs_register;
pub mod research_key;
pub mod research_source_ledger;

use crate::gate::GateBehavior;

/// Every documentation gate behavior implemented in this module.
pub static GATES: &[(&str, &dyn GateBehavior)] = &[
    ("cli-docs", &cli_docs::CliDocs),
    ("docs-check", &docs_check::DocsCheck),
    ("docs-coupling", &docs_coupling::DocsCoupling),
    ("docs-register", &docs_register::DocsRegister),
];
