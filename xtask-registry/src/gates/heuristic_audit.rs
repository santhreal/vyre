//! `heuristic-audit`  -  surface hand-rolled heuristics that should
//! be replaced by recursion-thesis self-consumers.
//!
//! The recursion thesis says every ad-hoc heuristic in vyre's
//! optimizer / scheduler / cache layer is technical debt  -  a place
//! where vyre is using less than the math it ships. This subcommand
//! greps for the canonical "I should be a self-consumer call" markers
//! so they show up on every CI run instead of getting forgotten.
//!
//! Default mode: warning. `--strict` exits non-zero  -  the gate.

use std::io;
use std::path::{Path, PathBuf};

use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

const VYRE_ROOT: &str = "libs/performance/matching/vyre";
const MAX_HEURISTIC_AUDIT_SOURCE_BYTES: u64 = 2_097_152;

/// Crates whose source we audit. Excludes test fixtures, examples,
/// benchmarks, and documentation.
const CRATES: &[&str] = &[
    "vyre-foundation",
    "vyre-driver",
    "vyre-driver-wgpu",
    "vyre-driver-cuda",
    "vyre-driver-spirv",
    "vyre-runtime",
    "vyre-libs",
    "vyre-aot",
    "vyre-spec",
];

/// Markers that flag a hand-rolled heuristic. Each pattern points at
/// a known class of "use math here" debt. Adding a new pattern
/// requires a one-line note explaining what self-consumer should
/// replace it.
const MARKERS: &[(&str, &str)] = &[
    // Fusion / cost heuristics → tensor_network_fusion_order (#35).
    (
        "Heuristic fusion pressure",
        "use tensor_network_fusion_order::optimal_fusion_order",
    ),
    (
        "// HEURISTIC",
        "audit + replace with the appropriate self-consumer",
    ),
    // Per-pass match-on-Node validators → knowledge_compile_pass_precondition (#38).
    (
        "// hand-rolled validator",
        "use knowledge_compile_pass_precondition::pass_applies",
    ),
    // Pass-dependency hand-curation → adjustment_set_pass_dependency (#37).
    (
        "// pass dependency table",
        "derive via adjustment_set_pass_dependency::ordering_is_safe",
    ),
    // Sequential host-driven fixpoint loops → persistent_fixpoint.
    (
        "// host-side fixpoint",
        "use vyre_primitives::fixpoint::persistent_fixpoint",
    ),
    // LRU eviction / hit-rate heuristics → submodular_cache_eviction (#45).
    (
        "// LRU eviction",
        "use submodular_cache_eviction::select_retention_set",
    ),
    // Plain-gradient autotuner → natural_gradient_autotuner (#56).
    (
        "// plain gradient autotune",
        "use natural_gradient_autotuner::autotune_step",
    ),
    // Hand-coded cache invalidation → do_calculus_change_impact (#36).
    (
        "// hand-coded invalidation",
        "use do_calculus_change_impact",
    ),
];

/// Reports hand-rolled heuristics that should be self-consumer calls.
pub struct HeuristicAudit;

impl Gate for HeuristicAudit {
    fn name(&self) -> &'static str {
        "heuristic-audit"
    }

    fn help(&self) -> &'static str {
        "Report hand-rolled heuristics that should be self-consumer calls"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let vyre_dir = resolve_vyre_dir(&ctx.root);

        let mut hits: Vec<(PathBuf, usize, &str, &str)> = Vec::new();
        let mut scan_errors = Vec::new();
        for crate_name in CRATES {
            let src = vyre_dir.join(crate_name).join("src");
            if !src.exists() {
                scan_errors.push(format!(
                    "heuristic audit crate source root `{}` does not exist",
                    src.display()
                ));
                continue;
            }
            scan_dir(&src, &mut hits, &mut scan_errors);
        }

        let mut report = Report::clean();
        report.note(format!("{} crate(s) audited", CRATES.len()));
        for error in &scan_errors {
            report.find(Finding::new(
                format!("{error}, so the heuristic audit is incomplete"),
                "make every audited production source root and file readable, then run the gate again",
            ));
        }
        for (path, line, marker, fix) in &hits {
            report.find(Finding::at(
                path.clone(),
                *line as u32,
                format!("hand-rolled heuristic `{marker}`"),
                format!("replace it with {fix}"),
            ));
        }
        Ok(report)
    }
}

fn scan_dir(
    dir: &Path,
    findings: &mut Vec<(PathBuf, usize, &'static str, &'static str)>,
    scan_errors: &mut Vec<String>,
) {
    // Heuristic markers in tests are intentional fixtures, not production debt.
    let sources = xtask::tree_walk::pruned_by(dir, |name| {
        !matches!(name, "tests" | "fuzz" | "benches" | "examples")
            && !xtask::tree_walk::BUILD_OUTPUT_AND_VCS.contains(&name)
    });
    for entry in sources {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                scan_errors.push(format!(
                    "could not read heuristic audit entry under `{}`: {error}",
                    dir.display()
                ));
                continue;
            }
        };
        let path = entry.into_path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let body = match read_text_bounded(&path) {
            Ok(body) => body,
            Err(error) => {
                scan_errors.push(format!(
                    "could not read heuristic audit source `{}`: {error}",
                    path.display()
                ));
                continue;
            }
        };
        for (lineno, line) in body.lines().enumerate() {
            for &(marker, fix) in MARKERS {
                if line.contains(marker) {
                    findings.push((path.clone(), lineno + 1, marker, fix));
                }
            }
        }
    }
}

fn resolve_vyre_dir(workspace_root: &Path) -> PathBuf {
    if workspace_root.join("vyre-foundation").join("src").is_dir() {
        workspace_root.to_path_buf()
    } else {
        workspace_root.join(VYRE_ROOT)
    }
}

fn is_workspace_root(path: &Path) -> io::Result<bool> {
    let manifest = path.join("Cargo.toml");
    let text = read_text_bounded(&manifest)?;
    Ok(text.contains("[workspace]") && text.contains("members"))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_HEURISTIC_AUDIT_SOURCE_BYTES, "heuristic audit")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// The audit must scan a standalone Vyre workspace instead of appending its monorepo path twice.
    #[test]
    fn resolves_standalone_vyre_workspace_root() {
        let root = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir_all(root.path().join("vyre-foundation/src"))
            .expect("standalone Vyre source root");

        assert_eq!(resolve_vyre_dir(root.path()), root.path());
    }

    /// The audit must retain support for running from the enclosing Santh workspace.
    #[test]
    fn resolves_vyre_root_inside_monorepo() {
        let root = tempfile::tempdir().expect("temporary workspace");
        let expected = root.path().join(VYRE_ROOT);
        fs::create_dir_all(expected.join("vyre-foundation/src")).expect("nested Vyre source root");

        assert_eq!(resolve_vyre_dir(root.path()), expected);
    }

    /// A standalone workspace must win when an unrelated nested path also resembles Vyre.
    #[test]
    fn standalone_workspace_takes_precedence_over_nested_candidate() {
        let root = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir_all(root.path().join("vyre-foundation/src"))
            .expect("standalone Vyre source root");
        fs::create_dir_all(root.path().join(VYRE_ROOT).join("vyre-foundation/src"))
            .expect("nested Vyre source root");

        assert_eq!(resolve_vyre_dir(root.path()), root.path());
    }

    /// Duplicate marker patterns would hide audit categories behind repeated findings.
    #[test]
    fn markers_have_unique_patterns() {
        let mut patterns: Vec<&str> = MARKERS.iter().map(|(p, _)| *p).collect();
        patterns.sort();
        let original_len = patterns.len();
        patterns.dedup();
        assert_eq!(
            patterns.len(),
            original_len,
            "duplicate marker pattern in MARKERS"
        );
    }
}
