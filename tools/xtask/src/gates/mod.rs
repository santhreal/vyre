//! Canonical structural and product-contract gates for the vyre workspace.

pub mod ci_steps;
pub mod crate_ownership;
pub mod docs_register;
pub mod dup_scan;
pub mod layering;
pub mod lint_one_policy;
pub mod lockfile_clean;
pub mod repo_hygiene;
pub mod workspace_membership;

use std::path::Path;

/// One registered structural verification check.
#[derive(Clone, Copy)]
pub struct Gate {
    /// Canonical CLI subcommand name.
    pub name: &'static str,
    /// Single-sentence summary of the verified invariant.
    pub summary: &'static str,
    /// Verification function returning findings on failure.
    pub check: fn(&Path) -> Result<(), Vec<String>>,
}

/// Canonical roster of active workspace gates, sorted by name.
pub const GATES: &[Gate] = &[
    Gate {
        name: "ci-steps",
        summary: "Verify CI workflows omit CARGO_TARGET_DIR, guard self-hosted GPU jobs, and keep the 85% coverage floor.",
        check: ci_steps::check,
    },
    Gate {
        name: "crate-ownership",
        summary: "Verify CRATE_OWNERSHIP.toml entries, publication classes, and canonical package authors.",
        check: crate_ownership::check,
    },
    Gate {
        name: "docs-register",
        summary: "Verify tracked prose count <= 5, competitor-name absence, target-dir absence, and man-page hygiene.",
        check: docs_register::check,
    },
    Gate {
        name: "dup-scan",
        summary: "Scan normalized Rust source windows for multi-file code duplication.",
        check: dup_scan::check,
    },
    Gate {
        name: "layering",
        summary: "Verify acyclic downward-only dependency edges across workspace layers.",
        check: layering::check,
    },
    Gate {
        name: "lint-one-policy",
        summary: "Verify workspace lint inheritance and absence of todo!/unimplemented! in production sources.",
        check: lint_one_policy::check,
    },
    Gate {
        name: "lockfile-clean",
        summary: "Verify Cargo.lock parses cleanly with a single version per workspace member.",
        check: lockfile_clean::check,
    },
    Gate {
        name: "repo-hygiene",
        summary: "Verify absence of scratch paths, merge artifacts, and missing CHANGELOG header.",
        check: repo_hygiene::check,
    },
    Gate {
        name: "workspace-membership",
        summary: "Verify root workspace.members matches member manifests on disk and path dependencies resolve.",
        check: workspace_membership::check,
    },
];

/// Run one gate by canonical name.
pub fn run_named(root: &Path, name: &str) -> Result<(), Vec<String>> {
    let Some(gate) = GATES.iter().find(|g| g.name == name) else {
        return Err(vec![format!(
            "unknown gate `{name}`. Run `cargo run -p xtask -- list-gates` for registered names."
        )]);
    };
    (gate.check)(root)
}

/// Run every registered gate concurrently across scoped threads.
pub fn run_all(root: &Path) -> Result<(), Vec<(String, Vec<String>)>> {
    let failures: Vec<(String, Vec<String>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = GATES
            .iter()
            .map(|gate| scope.spawn(move || (gate.name, (gate.check)(root))))
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| {
                let (name, res) = handle.join().expect("gate thread panicked");
                res.err().map(|errs| (name.to_owned(), errs))
            })
            .collect()
    });
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}
