//! Platform crate documentation names capabilities, not consumers.
//!
//! Dependency rules catch a `use` edge. Semantic coupling arrives through prose:
//! a comment that names the downstream product tells a reader the platform knows
//! who is calling it. The scanned set is enumerated from the tree rather than
//! listed, because the listed version named seventeen documents, sixteen of which
//! the documentation deletion removed, and skipped each missing one silently. A
//! guard that scans one file while naming seventeen reports a boundary it never
//! measured. The comment match is anchored after leading whitespace, which the
//! shell version required to be present, so a crate-level `//!` block at column
//! zero was outside its scope entirely.

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Crates that must stay consumer-neutral.
const PLATFORM_CRATES: &[&str] = &[
    "vyre",
    "vyre-spec",
    "vyre-macros",
    "vyre-foundation",
    "vyre-primitives",
    "vyre-libs",
    "vyre-reference",
    "vyre-driver",
    "vyre-driver-cuda",
    "vyre-driver-wgpu",
    "vyre-driver-spirv",
    "vyre-runtime",
    "vyre-pass-engine",
];

/// Documents the earlier guard named, kept so the coverage it lost is visible.
///
/// A row naming a document that no longer exists is a finding rather than a
/// deletion: it records what the guard stopped covering, and retiring a row is
/// the operator's call.
const DECLARED_DOCUMENTS: &[&str] = &[
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/HOT_PATH_PROOFS.md",
    "docs/MATH_PRIMITIVES_PLACEMENT.md",
    "docs/PREDICATE_EXPR_DUALITY.md",
    "docs/ERROR_SURFACE.md",
    "docs/RELEASE.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/TESTING_PROGRAM.md",
    "docs/RECURSION_THESIS.md",
    "docs/RUNTIME_PIPELINE.md",
    "docs/library-tiers.md",
    "docs/megakernel-wiring.md",
    "docs/ops-catalog.md",
    "docs/parsing-and-frontends.md",
    "docs/region-chain.md",
    "docs/consumer-integration.md",
];

/// Data files that carry prose and are scanned like a document.
const PLATFORM_TEXT_FILES: &[&str] = &["docs/optimization/OP_MATRIX.toml"];

/// Names that must not appear in platform documentation.
const CONSUMER_NAMES: &[&str] = &["weir", "surgec", "gossan", "keyhog"];

/// Documents that may name the partner product, because the release train ships
/// both under joint tags and a runbook nobody can follow is not a runbook. The
/// list is a data file two guards read, after they disagreed about which
/// documents were exempt.
const RELEASE_COORDINATION: &str = "vyre-lints/rules/release_coordination_docs.txt";

/// Platform documentation stays consumer-neutral.
pub struct PlatformConsumerDocs;

impl Gate for PlatformConsumerDocs {
    fn name(&self) -> &'static str {
        "platform-consumer-docs"
    }

    fn help(&self) -> &'static str {
        "platform documentation that names a downstream consumer"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let exempt = release_coordination_entries(&tree)?;

        for declared in DECLARED_DOCUMENTS {
            if !tree.exists(declared) {
                report.find(Finding::in_file(
                    *declared,
                    "the consumer-neutrality guard names a document that does not exist",
                    "delete the row; the scanned set is enumerated from the tree, so a row \
                     naming a missing document only records coverage that was lost",
                ));
            }
        }

        for crate_name in PLATFORM_CRATES {
            let source_root = format!("{crate_name}/src");
            if !tree.exists(&source_root) {
                report.find(Finding::in_file(
                    source_root,
                    "the guard names a platform crate that does not exist",
                    "delete the row, or restore the crate; a crate that is not scanned is \
                     not held to the boundary",
                ));
                continue;
            }
            for hit in tree.hits(&tree.rust(&[&source_root])?, |line| {
                is_comment(line) && names_consumer(line)
            })? {
                report.find(Finding::at(
                    hit.file,
                    hit.line,
                    format!("a platform comment names a downstream consumer: {}", hit.text),
                    "describe the capability generically; a consumer name belongs in the \
                     consumer crate or in release integration evidence",
                ));
            }
            for document in ["README.md", "ARCHITECTURE.md", "CONFIG.md"] {
                let path = format!("{crate_name}/{document}");
                if tree.exists(&path) {
                    scan_document(&tree, &path, &exempt, &mut report)?;
                }
            }
        }

        // Enumerated, not listed: a document added or deleted changes the scanned
        // set without anyone editing a table.
        let mut documents: Vec<String> = tree
            .paths()
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .filter(|path| {
                path.ends_with(".md") && (!path.contains('/') || path.starts_with("docs/"))
            })
            .collect();
        documents.extend(
            PLATFORM_TEXT_FILES
                .iter()
                .filter(|path| tree.exists(path))
                .map(|path| (*path).to_string()),
        );
        report.note(format!("{} document(s) scanned", documents.len()));
        for document in &documents {
            scan_document(&tree, document, &exempt, &mut report)?;
        }

        Ok(report)
    }
}

/// Scan one document, unless it is exempt.
fn scan_document(
    tree: &Tree,
    path: &str,
    exempt: &[String],
    report: &mut Report,
) -> Result<(), GateError> {
    if is_release_coordination(path, exempt) {
        return Ok(());
    }
    for (number, line) in crate::gates::scan::numbered(&tree.read(path)?) {
        if names_consumer(line) {
            report.find(Finding::at(
                path,
                number,
                format!("platform documentation names a downstream consumer: {}", line.trim()),
                "describe the capability generically, or move the passage into the consumer \
                 documentation that owns the integration",
            ));
        }
    }
    Ok(())
}

/// Whether a line names a consumer as a whole word.
fn names_consumer(line: &str) -> bool {
    let lowered = line.to_ascii_lowercase();
    CONSUMER_NAMES.iter().any(|name| {
        lowered.match_indices(name).any(|(at, _)| {
            let before = lowered[..at].chars().next_back();
            let after = lowered[at + name.len()..].chars().next();
            !boundary_is_word(before) && !boundary_is_word(after)
        })
    })
}

fn boundary_is_word(character: Option<char>) -> bool {
    character.is_some_and(|value| value.is_ascii_alphanumeric() || value == '_')
}

/// Whether a Rust line is a comment. Prose is what this gate reads; a string
/// literal naming a consumer is a code question, not a documentation one.
fn is_comment(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//!")
        || trimmed.starts_with("///")
        || trimmed.starts_with("// ")
        || trimmed.starts_with('*')
        || line.contains("/*")
}

/// The exemption entries, comments and whitespace stripped.
fn release_coordination_entries(tree: &Tree) -> Result<Vec<String>, GateError> {
    Ok(tree
        .read(RELEASE_COORDINATION)?
        .lines()
        .map(|line| {
            line.split('#')
                .next()
                .unwrap_or_default()
                .split_whitespace()
                .collect::<String>()
        })
        .filter(|entry| !entry.is_empty())
        .collect())
}

/// Whether a path is exempt. An entry ending in a slash exempts the documents
/// under it; anything else matches the path or its final component.
fn is_release_coordination(path: &str, exempt: &[String]) -> bool {
    exempt.iter().any(|entry| {
        if entry.ends_with('/') {
            path.contains(entry.as_str()) && path.ends_with(".md")
        } else {
            path == entry || path.ends_with(&format!("/{entry}"))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the name is matched as a word. A gate that fires on a substring
    /// reports every unrelated identifier that happens to contain one of these
    /// four names, and a gate with false positives gets muted.
    #[test]
    fn a_consumer_name_matches_as_a_word() {
        assert!(names_consumer("// the weir consumer drives this"));
        assert!(names_consumer("//! Weir integration notes"));
        assert!(!names_consumer("// weirdly shaped input"));
        assert!(!names_consumer("// see also gossanite_mineral"));
    }

    /// WHY: the exemption is a data file two guards read, and the directory form
    /// is what covers the release notes tree. Getting the directory form wrong
    /// silently re-gates every runbook, which is how the two guards disagreed.
    #[test]
    fn the_exemption_covers_a_directory_and_a_bare_name() {
        let exempt = vec!["docs/release/".to_string(), "CHANGELOG.md".to_string()];
        assert!(is_release_coordination("docs/release/0.5.0.md", &exempt));
        assert!(is_release_coordination("CHANGELOG.md", &exempt));
        assert!(is_release_coordination("vyre-libs/CHANGELOG.md", &exempt));
        assert!(!is_release_coordination("docs/release/notes.toml", &exempt));
        assert!(!is_release_coordination("README.md", &exempt));
    }

    /// WHY: the rule reads prose, so a comment is what counts. Widening it to
    /// every line would make the gate a code rule it was never scoped as, and
    /// narrowing it below these four forms is how a block comment escapes.
    #[test]
    fn only_comment_lines_are_read() {
        assert!(is_comment("    /// doc comment"));
        assert!(is_comment("//! crate doc"));
        assert!(is_comment("    // ordinary comment"));
        assert!(is_comment("    * continuation of a block comment"));
        assert!(is_comment("let x = 1; /* trailing block */"));
        assert!(!is_comment("    let name = \"weir\";"));
    }
}
