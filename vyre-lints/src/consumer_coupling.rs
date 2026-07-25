//! Consumer-name coupling guard.
//!
//! Platform crates and current public docs must describe capabilities, not
//! downstream products. The guard scans current Markdown, Rust comments,
//! Rust string literals, and path names. Historical archives, tests, examples,
//! and fixtures are intentionally exempt because they may preserve migration
//! context or consumer integration examples.

use crate::{paths::workspace_relative, Violation, ViolationKind};
use anyhow::Result;
use std::path::Path;

const CONSUMER_NAMES: &[&str] = &["weir", "surgec", "gossan", "keyhog", "flare-native"];

const EXEMPT_PATH_FRAGMENTS: &[&str] = &[
    "/docs/archive/",
    "/docs/legacy/",
    "/.internals/",
    "/tests/",
    "/benches/",
    "/examples/",
    "/fixtures/",
    "/target/",
];

/// Release-coordination runbooks name the products in the combined release train on
/// purpose. The tags, publish order, and per-product versions an operator types are
/// literal facts (`vyre-0.4.1-weir-0.0.1` is a git tag, not a description), so a
/// capability paraphrase makes the instructions unfollowable. A blanket text
/// substitution over these files previously produced invalid identifiers with spaces
/// in them, including `git tag vyre-0.4.1-dataflow consumer-0.0.1` and the xtask
/// subcommand `vyre-dataflow consumer-release-gate`.
///
/// The exemption is deliberately narrow: release runbooks only. Architecture docs,
/// API docs, guides, and every Rust source file stay under the guard, because those
/// describe the platform's own surface, where naming a consumer is real coupling.
/// `CHANGELOG.md` is on this list for the same reason: a rename entry has to print the
/// identifier that was removed. A migration table reading `weir_alias` to
/// `alias_import` is the whole value of the entry, and paraphrasing the left column
/// leaves a consumer with no way to find the symbol it must edit. The changelog
/// records what the API used to be called, which is history rather than coupling.
/// The exemption list, as shipped.
///
/// It lives in a data file rather than in this array so that
/// `scripts/check_platform_consumer_docs.sh` can read the same list. The two
/// guards previously carried separate copies and disagreed: `docs/RELEASE.md`
/// was exempt here and scanned there, so the same line passed one gate and
/// failed the other.
const RELEASE_COORDINATION_DOCS_FILE: &str = include_str!("../rules/release_coordination_docs.txt");

/// Parses [`RELEASE_COORDINATION_DOCS_FILE`] into its path entries.
///
/// One path per line; `#` comments and blank lines are ignored.
fn release_coordination_docs() -> impl Iterator<Item = &'static str> {
    RELEASE_COORDINATION_DOCS_FILE
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

pub fn scan_tree(root: &Path) -> Result<Vec<Violation>> {
    let mut all = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();
        if !path.is_file() || !is_scanned_extension(path) {
            continue;
        }
        let workspace_rel = workspace_relative(path);
        if is_exempt_path(&workspace_rel) {
            continue;
        }
        if let Some((column, name)) = find_consumer_name_in_path(&workspace_rel) {
            all.push(Violation {
                file: workspace_rel.clone(),
                line: 1,
                column: column as u32,
                kind: ViolationKind::ConsumerCoupling,
                message: consumer_coupling_message(name, "path"),
            });
        }
        all.extend(scan_file(path, &workspace_rel)?);
    }
    Ok(all)
}

fn is_scanned_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "md")
    )
}

fn is_exempt_path(workspace_rel: &str) -> bool {
    let wrapped = format!("/{workspace_rel}");
    if EXEMPT_PATH_FRAGMENTS
        .iter()
        .any(|fragment| wrapped.contains(fragment))
    {
        return true;
    }
    is_release_coordination_doc(workspace_rel)
}

/// Whether this path is a release runbook, matched on the workspace-relative suffix so
/// the check works the same for an absolute scan root and a temp-dir fixture.
///
/// A directory entry only exempts Markdown inside it. Without that, a `.rs` file placed
/// under `docs/release/` would inherit the runbook exemption, and no exemption is ever
/// meant to cover Rust source: a runbook naming a downstream product is a literal
/// operator instruction, while source naming one is the coupling this lint exists for.
fn is_release_coordination_doc(workspace_rel: &str) -> bool {
    let normalized = workspace_rel.replace('\\', "/");
    release_coordination_docs().any(|entry| {
        if let Some(dir) = entry.strip_suffix('/') {
            normalized.contains(&format!("{dir}/")) && normalized.ends_with(".md")
        } else {
            normalized == entry || normalized.ends_with(&format!("/{entry}"))
        }
    })
}

fn scan_file(path: &Path, workspace_rel: &str) -> Result<Vec<Violation>> {
    let source = crate::read_source_bounded(path)?;
    let is_markdown = path.extension().and_then(|ext| ext.to_str()) == Some("md");
    let mut violations = Vec::new();

    for (line_idx, line) in source.lines().enumerate() {
        for (column_offset, segment, context) in scanned_segments(line, is_markdown) {
            if let Some((column, name)) = find_consumer_name(segment) {
                violations.push(Violation {
                    file: workspace_rel.to_string(),
                    line: (line_idx + 1) as u32,
                    column: (column_offset + column) as u32,
                    kind: ViolationKind::ConsumerCoupling,
                    message: consumer_coupling_message(name, context),
                });
            }
        }
    }
    Ok(violations)
}

fn consumer_coupling_message(name: &str, context: &str) -> String {
    format!(
        "platform {context} mentions downstream consumer `{name}`. Fix: use a capability name such as dataflow, static analysis, scan, or consumer integration."
    )
}

fn scanned_segments(line: &str, is_markdown: bool) -> Vec<(usize, &str, &'static str)> {
    if is_markdown {
        return vec![(0, line, "markdown")];
    }
    if is_comment_line(line) {
        return vec![(0, line, "comment")];
    }
    rust_string_literal_segments(line)
        .into_iter()
        .map(|(offset, segment)| (offset, segment, "string literal"))
        .collect()
}

fn is_comment_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed.starts_with('*')
        || trimmed.starts_with("*/")
}

fn rust_string_literal_segments(line: &str) -> Vec<(usize, &str)> {
    let bytes = line.as_bytes();
    let mut segments = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let Some(start_rel) = line[cursor..].find('"') else {
            break;
        };
        let start_quote = cursor + start_rel;
        if start_quote > 0 && bytes[start_quote - 1] == b'\'' {
            cursor = start_quote + 1;
            continue;
        }
        let content_start = start_quote + 1;
        let mut idx = content_start;
        let mut escaped = false;
        while idx < bytes.len() {
            let byte = bytes[idx];
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                segments.push((content_start, &line[content_start..idx]));
                cursor = idx + 1;
                break;
            }
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }
    }
    segments
}

fn find_consumer_name(line: &str) -> Option<(usize, &'static str)> {
    let lower = line.to_ascii_lowercase();
    for name in CONSUMER_NAMES {
        let mut search_from = 0usize;
        while let Some(rel_idx) = lower[search_from..].find(name) {
            let idx = search_from + rel_idx;
            let end = idx + name.len();
            if is_start_boundary(lower.as_bytes(), idx) && is_end_boundary(lower.as_bytes(), end) {
                return Some((idx, *name));
            }
            search_from = end;
        }
    }
    None
}

fn find_consumer_name_in_path(path: &str) -> Option<(usize, &'static str)> {
    let lower = path.to_ascii_lowercase();
    for name in CONSUMER_NAMES {
        let mut search_from = 0usize;
        while let Some(rel_idx) = lower[search_from..].find(name) {
            let idx = search_from + rel_idx;
            let end = idx + name.len();
            if is_path_start_boundary(lower.as_bytes(), idx)
                && is_path_end_boundary(lower.as_bytes(), end)
            {
                return Some((idx, *name));
            }
            search_from = end;
        }
    }
    None
}

fn is_start_boundary(bytes: &[u8], idx: usize) -> bool {
    if idx == 0 {
        return true;
    }
    !bytes[idx - 1].is_ascii_alphanumeric() && bytes[idx - 1] != b'_'
}

fn is_end_boundary(bytes: &[u8], idx: usize) -> bool {
    if idx >= bytes.len() {
        return true;
    }
    !bytes[idx].is_ascii_alphanumeric() && bytes[idx] != b'_'
}

fn is_path_start_boundary(bytes: &[u8], idx: usize) -> bool {
    if idx == 0 {
        return true;
    }
    !bytes[idx - 1].is_ascii_alphanumeric()
}

fn is_path_end_boundary(bytes: &[u8], idx: usize) -> bool {
    if idx >= bytes.len() {
        return true;
    }
    !bytes[idx].is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_boundary_rejects_substrings() {
        assert!(find_consumer_name("// weir dataflow").is_some());
        assert!(find_consumer_name("// weird control flow").is_none());
        assert!(find_consumer_name("// keyhog-style coupling").is_some());
    }

    #[test]
    fn rust_string_literal_segments_ignore_identifiers_and_chars() {
        let segments = rust_string_literal_segments(
            "let keyhog_counter = 'k'; let label = \"surgec adapter\"; let raw = r#\"weir phase\"#;",
        );
        assert_eq!(segments.len(), 2);
        assert!(segments
            .iter()
            .any(|(_, segment)| *segment == "surgec adapter"));
        assert!(segments.iter().any(|(_, segment)| *segment == "weir phase"));
    }

    #[test]
    fn path_boundary_treats_underscore_as_separator() {
        assert_eq!(
            find_consumer_name_in_path("vyre-libs/src/security/surgec_bridge/mod.rs"),
            Some((23, "surgec"))
        );
        assert_eq!(find_consumer_name_in_path("docs/weird.md"), None);
    }

    /// The changelog must be able to print a removed identifier in a rename migration
    /// table, because paraphrasing the old name leaves a consumer unable to find the
    /// symbol it has to edit.
    #[test]
    fn changelog_is_a_release_coordination_doc() {
        assert!(is_release_coordination_doc("CHANGELOG.md"));
        assert!(is_release_coordination_doc(
            "libs/performance/matching/vyre/CHANGELOG.md"
        ));
    }

    /// The exemption is a whole-filename match, not a prefix or a substring. A doc that
    /// merely mentions the changelog in its name stays under the guard, so the
    /// exemption cannot be widened by naming a file after an exempt one.
    #[test]
    fn changelog_exemption_does_not_leak_to_neighbouring_documents() {
        assert!(!is_release_coordination_doc("docs/CHANGELOG_NOTES.md"));
        assert!(!is_release_coordination_doc("docs/OLD-CHANGELOG.md"));
        assert!(!is_release_coordination_doc("changelog.md"));
    }

    /// No exemption ever covers Rust source. A release runbook naming a downstream
    /// product is a literal operator instruction; a `.rs` file naming one is coupling.
    #[test]
    fn release_coordination_exemption_never_covers_rust_source() {
        for path in [
            "docs/release/v0.7.0.rs",
            "xtask/src/release.rs",
            "vyre-lints/src/consumer_coupling.rs",
        ] {
            assert!(
                !is_release_coordination_doc(path),
                "{path} must stay under the consumer-coupling guard"
            );
        }
    }
}
