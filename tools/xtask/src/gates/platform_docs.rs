//! Published crates and the workspace documentation name capabilities, not
//! consumers.
//!
//! Dependency rules catch a `use` edge. Semantic coupling arrives through prose:
//! a comment or a document that names the downstream product states that the
//! platform depends on who calls it. The subject is derived, not listed: every
//! Rust comment and Markdown file under a crate whose layer ships
//! ([`crate_registry::shipped_crates`]), every Markdown file at the root or under
//! `docs/`, and the data files in `PLATFORM_TEXT_FILES`. A crate given a
//! shipped layer in `docs/CRATE_OWNERSHIP.toml` is scanned on the next run.

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::crate_registry;
use crate::gates::scan::{under, Tree};

/// Data files that hold prose and are scanned like a document.
const PLATFORM_TEXT_FILES: &[&str] = &["docs/optimization/OP_MATRIX.toml"];

/// Names that must not appear in platform documentation.
const CONSUMER_NAMES: &[&str] = &["weir", "surgec", "gossan", "keyhog"];

/// Documents that may name the partner product, because the release train ships
/// both under joint tags and a runbook must name the other repository.
const RELEASE_COORDINATION: &str = "tools/xtask/release-coordination-docs.txt";

/// Platform documentation stays consumer-neutral.
pub struct PlatformConsumerDocs;

impl crate::gate::GateBehavior for PlatformConsumerDocs {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let exempt = release_coordination_entries(&tree)?;
        let shipped: Vec<String> = crate_registry::shipped_crates(&tree)?
            .into_iter()
            .map(|row| row.path)
            .collect();
        if shipped.is_empty() {
            return Err(GateError::new(
                "docs/CRATE_OWNERSHIP.toml places no crate in a shipped layer, so no platform \
                 crate is scanned",
                "restore the [[crate]] rows; an empty subject passes every rule",
            ));
        }
        report.cover_complete("shipped crates", shipped.len());
        let roots: Vec<&str> = shipped.iter().map(String::as_str).collect();

        for hit in tree.hits(&tree.rust(&roots)?, |line| {
            is_comment(line) && names_consumer(line)
        })? {
            report.find(Finding::at(
                hit.file,
                hit.line,
                format!(
                    "a platform comment names a downstream consumer: {}",
                    hit.text
                ),
                "describe the capability generically; a consumer name belongs in the \
                 consumer crate or in release integration evidence",
            ));
        }

        let mut documents: Vec<String> = tree
            .paths()
            .iter()
            .filter(|path| {
                path.extension().and_then(|extension| extension.to_str()) == Some("md")
                    && (path.parent() == Some(std::path::Path::new(""))
                        || under(path, "docs")
                        || roots.iter().any(|root| under(path, root)))
            })
            .map(|path| path.to_string_lossy().into_owned())
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
                format!(
                    "platform documentation names a downstream consumer: {}",
                    line.trim()
                ),
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
    trimmed.starts_with("//") || trimmed.starts_with('*') || line.contains("/*")
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
        assert!(is_release_coordination(
            "crates/vyre-libs/CHANGELOG.md",
            &exempt
        ));
        assert!(!is_release_coordination("docs/release/notes.toml", &exempt));
        assert!(!is_release_coordination("README.md", &exempt));
    }

    /// WHY: the rule reads prose, so a comment is what counts. Widening it to
    /// every line would make the gate a code rule it was never scoped as, and
    /// narrowing it below these forms is how a block comment escapes.
    #[test]
    fn only_comment_lines_are_read() {
        assert!(is_comment("    /// doc comment"));
        assert!(is_comment("//! crate doc"));
        assert!(is_comment("    // ordinary comment"));
        assert!(is_comment("    //unspaced comment"));
        assert!(is_comment("    * continuation of a block comment"));
        assert!(is_comment("let x = 1; /* trailing block */"));
        assert!(!is_comment("    let name = \"weir\";"));
    }

    /// WHY: the subject is every crate in a shipped layer, not a list. A list
    /// held thirteen crates while forty-three shipped, and a second gate covered
    /// six of them through `tests/` and nested documents only. A shipped crate's
    /// test comment, its nested document and a crate added under a nested path
    /// are scanned whether or not Cargo publishes it; a tooling crate is not,
    /// and a source string is code.
    #[test]
    fn every_shipped_crate_is_scanned_through_its_whole_directory() {
        use crate::gate::GateBehavior;
        use crate::gates::fixture_checkout;

        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"shipped\", \"nest/added\", \"internal\"]\n",
            ),
            ("tools/xtask/release-coordination-docs.txt", "CHANGELOG.md\n"),
            (
                "docs/CRATE_OWNERSHIP.toml",
                "[[layer]]\nname = \"runtime\"\nrank = 0\n\n[[layer]]\nname = \"tooling\"\nrank = 1\n\n\
                 [[crate]]\npackage = \"shipped\"\npath = \"shipped\"\nlayer = \"runtime\"\n\n\
                 [[crate]]\npackage = \"added\"\npath = \"nest/added\"\nlayer = \"runtime\"\n\n\
                 [[crate]]\npackage = \"internal\"\npath = \"internal\"\nlayer = \"tooling\"\n",
            ),
            (
                "shipped/Cargo.toml",
                "[package]\nname = \"shipped\"\nversion = \"0.0.0\"\n",
            ),
            ("shipped/src/lib.rs", "pub fn f() -> &'static str { \"weir\" }\n"),
            ("shipped/tests/case.rs", "// drives the gossan path\n"),
            ("shipped/docs/guide.md", "Built for Keyhog.\n"),
            (
                "nest/added/Cargo.toml",
                "[package]\nname = \"added\"\nversion = \"0.0.0\"\npublish = false\n",
            ),
            ("nest/added/src/lib.rs", "//! The surgec bridge.\n"),
            (
                "internal/Cargo.toml",
                "[package]\nname = \"internal\"\nversion = \"0.0.0\"\n",
            ),
            ("internal/src/lib.rs", "// weir fixture\n"),
            ("internal/NOTES.md", "weir\n"),
            ("CHANGELOG.md", "Renamed the weir bridge.\n"),
        ]);
        let report = PlatformConsumerDocs
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the fixture checkout is readable");
        let mut files: Vec<String> = report
            .findings
            .iter()
            .filter_map(|finding| finding.file.as_ref())
            .map(|file| file.display().to_string())
            .collect();
        files.sort();
        assert_eq!(
            files,
            [
                "nest/added/src/lib.rs",
                "shipped/docs/guide.md",
                "shipped/tests/case.rs"
            ]
        );
    }
}
