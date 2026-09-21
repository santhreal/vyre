//! In-process gate execution over a fixture checkout, for the tree contracts.
//!
//! Only the generator contracts in this harness run a gate in process, so the
//! runner is declared here rather than in `tests/workspace_sources`. The
//! crate's other test binary has a subprocess runner of its own under the same
//! name, which is why one shared item cannot serve both.

use std::path::Path;

use xtask::gate::{GateBehavior, GateCtx, Report};

/// Run one gate over a fixture checkout, in check or write mode.
///
/// The fixture has to be a real checkout: every gate here reads the tree
/// through `git ls-files`, so a directory of untracked files reads as an empty
/// workspace and the gate would report nothing at all.
pub(crate) fn run_gate(
    name: &str,
    gate: &'static dyn GateBehavior,
    root: &Path,
    write: bool,
) -> Report {
    let args = if write {
        vec!["--write".to_string()]
    } else {
        Vec::new()
    };
    let desc = xtask::gate_metadata::descriptor_by_name(name);
    let registered = xtask::gate::RegisteredGate::new(desc, gate);
    registered
        .run(&GateCtx::new(root.to_path_buf(), args))
        .unwrap_or_else(|error| panic!("Fix: {name} must run: {error:?}"))
}
