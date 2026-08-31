//! The `platform-boundary` gate: platform crate docs stay consumer-neutral.
//!
//! The tier system is meaningless if platform crate docs name downstream
//! consumers. This gate scans Rust comments and Markdown in platform crates for
//! known consumer names and reports each one with its file and line.

use std::fs;
use std::path::{Path, PathBuf};

use crate::gate::{Finding, GateCtx, GateError, Report};

const PLATFORM_ROOTS: &[&str] = &[
    "vyre-foundation",
    "vyre-primitives",
    "vyre-libs",
    "vyre-driver",
    "vyre-runtime",
    "vyre-pass-engine",
];

const FORBIDDEN_CONSUMERS: &[&str] = &["surgec", "weir", "gossan", "keyhog"];
const MAX_PLATFORM_BOUNDARY_FILE_BYTES: u64 = 16_777_216;

/// One consumer name found in one platform line.
#[derive(Debug, Clone, Eq, PartialEq)]
struct Hit {
    path: PathBuf,
    line: usize,
    term: &'static str,
    text: String,
}

/// Reports a downstream consumer name written into a platform crate's prose.
pub struct PlatformBoundary;

impl crate::gate::GateBehavior for PlatformBoundary {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut hits = Vec::new();
        let mut errors = Vec::new();
        for relative in PLATFORM_ROOTS {
            scan_tree(&ctx.root.join(relative), &ctx.root, &mut hits, &mut errors);
        }

        let mut report = Report::from_messages(
            errors,
            "make every platform source and doc file readable; a file this gate cannot read is a file it cannot judge",
        );
        report.cover_complete("platform source roots", PLATFORM_ROOTS.len());
        report.note(format!(
            "scanned {} platform crates for {} consumer names",
            PLATFORM_ROOTS.len(),
            FORBIDDEN_CONSUMERS.len()
        ));
        for hit in hits {
            report.find(Finding::at(
                hit.path,
                u32::try_from(hit.line).unwrap_or(u32::MAX),
                format!("names the downstream consumer `{}`: {}", hit.term, hit.text.trim()),
                "replace the downstream name with neutral platform, dataflow or frontend wording, or move the doc to the consumer-owned crate",
            ));
        }
        Ok(report)
    }
}

fn scan_tree(root: &Path, workspace: &Path, hits: &mut Vec<Hit>, errors: &mut Vec<String>) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => {
                errors.push(format!("{}: {error}", path.display()));
                continue;
            }
        };
        if metadata.is_dir() {
            let entries = match fs::read_dir(&path) {
                Ok(entries) => entries,
                Err(error) => {
                    errors.push(format!("{}: {error}", path.display()));
                    continue;
                }
            };
            for entry in entries {
                match entry {
                    Ok(entry) => stack.push(entry.path()),
                    Err(error) => errors.push(format!("{}: {error}", path.display())),
                }
            }
            continue;
        }
        if !is_scanned_file(&path) {
            continue;
        }
        if metadata.len() > MAX_PLATFORM_BOUNDARY_FILE_BYTES {
            errors.push(format!(
                "{} exceeds {MAX_PLATFORM_BOUNDARY_FILE_BYTES} byte platform-boundary read cap",
                path.display()
            ));
            continue;
        }
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) => {
                errors.push(format!("{}: {error}", path.display()));
                continue;
            }
        };
        collect_hits(&path, workspace, &text, hits);
    }
}

fn is_scanned_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "md")
    )
}

fn collect_hits(path: &Path, workspace: &Path, text: &str, hits: &mut Vec<Hit>) {
    let markdown = path.extension().and_then(|ext| ext.to_str()) == Some("md");
    for (line_index, line) in text.lines().enumerate() {
        if !markdown && !is_rust_comment_line(line) {
            continue;
        }
        for term in FORBIDDEN_CONSUMERS {
            if contains_word_case_insensitive(line, term) {
                hits.push(Hit {
                    path: path.strip_prefix(workspace).unwrap_or(path).to_path_buf(),
                    line: line_index + 1,
                    term,
                    text: line.to_string(),
                });
            }
        }
    }
}

fn is_rust_comment_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed.starts_with('*')
        || trimmed.starts_with("*/")
}

fn contains_word_case_insensitive(line: &str, needle: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let mut search_from = 0;
    while let Some(offset) = lower[search_from..].find(needle) {
        let start = search_from + offset;
        let end = start + needle.len();
        if is_left_word_boundary(&lower, start) && is_right_word_boundary(&lower, end) {
            return true;
        }
        search_from = end;
    }
    false
}

fn is_left_word_boundary(text: &str, byte_index: usize) -> bool {
    if byte_index == 0 {
        return true;
    }
    is_non_word_byte(text.as_bytes()[byte_index - 1])
}

fn is_right_word_boundary(text: &str, byte_index: usize) -> bool {
    match text.as_bytes().get(byte_index) {
        None => true,
        Some(byte) => is_non_word_byte(*byte),
    }
}

fn is_non_word_byte(byte: u8) -> bool {
    !byte.is_ascii_alphanumeric() && byte != b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_consumer_names_in_comments_but_not_identifiers() {
        let mut findings = Vec::new();
        collect_hits(
            Path::new("vyre-libs/src/example.rs"),
            Path::new(""),
            "let weir_internal = 1;\n//! Weir owns this downstream wording\n// keyhog should not appear here",
            &mut findings,
        );
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].term, "weir");
        assert_eq!(findings[1].term, "keyhog");
    }

    #[test]
    fn scans_markdown_docs_for_consumer_names() {
        let mut findings = Vec::new();
        collect_hits(
            Path::new("vyre-primitives/README.md"),
            Path::new(""),
            "# Graph primitives\n\nThis platform doc mentions SurgeC and Gossan.",
            &mut findings,
        );

        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].term, "surgec");
        assert_eq!(findings[1].term, "gossan");
    }

    #[test]
    fn honors_word_boundaries() {
        assert!(!contains_word_case_insensitive("wearing a wire", "weir"));
        assert!(contains_word_case_insensitive("consumer: WEIR", "weir"));
    }
}
