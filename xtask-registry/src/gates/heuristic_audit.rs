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
use xtask::gates::scan;

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
///
/// A marker is the note an author leaves in a plain comment, so it is spelled
/// without the `//` that opens one: see [`declared_marker`] for why the
/// distinction between a note and a doc comment is the whole rule.
const MARKERS: &[(&str, &str)] = &[
    // Fusion / cost heuristics → tensor_network_fusion_order (#35).
    (
        "Heuristic fusion pressure",
        "use tensor_network_fusion_order::optimal_fusion_order",
    ),
    (
        "HEURISTIC",
        "audit + replace with the appropriate self-consumer",
    ),
    // Per-pass match-on-Node validators → knowledge_compile_pass_precondition (#38).
    (
        "hand-rolled validator",
        "use knowledge_compile_pass_precondition::pass_applies",
    ),
    // Pass-dependency hand-curation → adjustment_set_pass_dependency (#37).
    (
        "pass dependency table",
        "derive via adjustment_set_pass_dependency::ordering_is_safe",
    ),
    // Sequential host-driven fixpoint loops → persistent_fixpoint.
    (
        "host-side fixpoint",
        "use vyre_primitives::fixpoint::persistent_fixpoint",
    ),
    // LRU eviction / hit-rate heuristics → submodular_cache_eviction (#45).
    (
        "LRU eviction",
        "use submodular_cache_eviction::select_retention_set",
    ),
    // Plain-gradient autotuner → natural_gradient_autotuner (#56).
    (
        "plain gradient autotune",
        "use natural_gradient_autotuner::autotune_step",
    ),
    // Hand-coded cache invalidation → do_calculus_change_impact (#36).
    ("hand-coded invalidation", "use do_calculus_change_impact"),
];

/// The debt marker a line declares, with the self-consumer that replaces it.
///
/// A marker is an author's note that the code beside it is doing by hand what
/// vyre already ships as math, so it opens the comment that carries it. Three
/// shapes are not that note, and the scan read all three as one:
///
/// A doc comment describes an item to whoever calls it, so `/// LRU eviction
/// kicks the oldest entry` states a cache's policy to its reader. Reading it as
/// a declaration of debt reported three descriptions, one of them on a test,
/// and two in crates that cannot reach the composition the fix names.
///
/// A sentence that mentions a policy while explaining something else is prose:
/// `// ... which is the scan an LRU eviction is already paying for once` argues
/// why one key is cloned per eviction. Requiring the marker to open the comment
/// is what separates a tag an author placed from a term a paragraph used.
///
/// A string literal holds the marker table itself and every fixture built from
/// it, which is why the line arrives already masked.
fn declared_marker(line: &str) -> Option<(&'static str, &'static str)> {
    let comment = plain_comment_body(line)?.trim_start();
    MARKERS
        .iter()
        .copied()
        .find(|(marker, _)| comment.starts_with(marker))
}

/// The text after `//` when the line carries a plain comment, not a doc one.
///
/// Exactly three slashes open a doc comment and `//!` opens an inner one; four
/// or more are an ordinary comment an author may have drawn a rule with.
fn plain_comment_body(line: &str) -> Option<&str> {
    let at = line.find("//")?;
    let slashes = line[at..].bytes().take_while(|byte| *byte == b'/').count();
    let body = &line[at + slashes..];
    if slashes == 3 || body.starts_with('!') {
        return None;
    }
    Some(body)
}

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
    // A marker in a test tree is an intentional fixture, not production debt.
    // An inline `#[cfg(test)]` module is the same fixture in another place, and
    // `production_markers` is what excludes it.
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
        for (lineno, marker, fix) in production_markers(&body) {
            findings.push((path.clone(), lineno, marker, fix));
        }
    }
}

/// Every declared marker in `body` outside a `#[cfg(test)]` item, one-based.
fn production_markers(body: &str) -> Vec<(usize, &'static str, &'static str)> {
    let masked = scan::mask_literals(body);
    let lines: Vec<&str> = masked.lines().collect();
    let test_only = scan::cfg_test_lines(&lines);
    let mut found = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if test_only[index] {
            continue;
        }
        if let Some((marker, fix)) = declared_marker(line) {
            found.push((index + 1, marker, fix));
        }
    }
    found
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

    /// WHY: a marker is the note inside a comment, so a table spelling the `//`
    /// as well would match `///` too and read every doc comment as debt.
    #[test]
    fn no_marker_spells_the_comment_that_carries_it() {
        for (marker, _) in MARKERS {
            assert!(
                !marker.contains("//"),
                "Fix: spell `{marker}` as the note alone; the scan finds the comment."
            );
        }
    }

    /// WHY: the rule that fired on three doc comments. A cache whose doc states
    /// its eviction policy is describing itself to a caller, and two of the
    /// three crates cannot even reach the composition the fix names.
    #[test]
    fn a_doc_comment_naming_a_policy_is_not_a_declared_marker() {
        assert_eq!(
            production_markers("/// LRU eviction kicks the oldest entry.\n"),
            Vec::new()
        );
        assert_eq!(
            production_markers("//! LRU eviction policy with promotion.\n"),
            Vec::new()
        );
    }

    /// WHY: the shape of the last false finding. `HotPathHints::record` explains
    /// why one key is cloned per eviction and names the policy mid-sentence; the
    /// note is an argument about an allocation, not a request for a composition
    /// the crate cannot depend on.
    #[test]
    fn a_policy_named_mid_sentence_is_prose_not_a_marker() {
        let body = concat!(
            "// Pairing every key with its timestamp before taking the minimum\n",
            "// cloned the whole map on each eviction, which is the scan an LRU\n",
            "// eviction is already paying for once.\n",
        );
        assert_eq!(production_markers(body), Vec::new());
    }

    /// WHY: the gate exists for the note an author leaves beside the code, and
    /// the note is reported at its own line.
    #[test]
    fn a_plain_comment_marker_is_a_finding_at_its_line() {
        let found = production_markers("fn evict() {\n    // LRU eviction by hand\n}\n");
        assert_eq!(
            found,
            vec![(
                2,
                "LRU eviction",
                "use submodular_cache_eviction::select_retention_set"
            )]
        );
    }

    /// WHY: a fixture is a fixture wherever it lives. The walk prunes a `tests`
    /// directory, and an inline module is the same code one directory up.
    #[test]
    fn a_marker_inside_an_inline_test_module_is_not_production_debt() {
        let body = concat!(
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    // LRU eviction fixture\n",
            "    fn f() {}\n",
            "}\n",
            "// LRU eviction by hand\n",
        );
        assert_eq!(
            production_markers(body),
            vec![(
                6,
                "LRU eviction",
                "use submodular_cache_eviction::select_retention_set"
            )]
        );
    }

    /// WHY: this table is source text holding every marker it looks for, so an
    /// unmasked scan reports the gate itself.
    #[test]
    fn a_marker_inside_a_string_literal_is_not_a_declared_marker() {
        assert_eq!(
            production_markers("const M: &str = \"// LRU eviction\";\n"),
            Vec::new()
        );
    }
}
