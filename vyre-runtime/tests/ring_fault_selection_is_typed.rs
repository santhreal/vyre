//! WHY: a rejection test is only as strong as the fault it selects. Two weak
//! assertion shapes recurred across this crate and neither could fail on a call
//! that started failing for a different reason.
//! `assert!(matches!(err, PipelineError::Backend(_)))` is satisfied by every
//! backend string the runtime can produce, and
//! `assert!(err.to_string().contains("Fix:"))` is satisfied by every variant of
//! `PipelineError`, because `pipeline_error_closure.rs` already proves each one
//! carries a `Fix:` clause. Both passed while asserting nothing about the fault
//! under test, so a truncated metrics window, a misaligned control buffer and a
//! double publish into an in-flight slot were interchangeable.
//!
//! Closes: the assertion shape, not the incidents. The deny list below is
//! derived from a run-time scan of every `.rs` file under `src/` and `tests/`,
//! so it is not a pinned count and cannot drift behind the tree. A new weak
//! assertion anywhere in the crate turns this test red and names its file and
//! line, whether it lands in an existing file or a new one.
//!
//! Does not catch: a message bound to a local before it is asserted, such as
//! `let msg = err.to_string(); assert!(msg.contains("Fix:"))`. Resolving that
//! needs the binding's type, which a text scan does not have. It also does not
//! prove that a typed assertion pins the right payload values; the variant a
//! given call reports is asserted at that call's own test.

use std::fs;
use std::path::{Path, PathBuf};

/// One rejected assertion, ready to print as `path:line`.
struct Finding {
    path: String,
    line: usize,
    shape: &'static str,
}

fn crate_root() -> PathBuf {
    vyre_test_support::monorepo::vyre_crate_directory("vyre-runtime")
}

/// Every `.rs` file under `src/` and `tests/`, minus the two files whose own
/// subject is the weak shape this test bans.
fn scanned_sources() -> Vec<(String, String)> {
    let root = crate_root();
    let mut sources = Vec::new();
    for dir in ["src", "tests"] {
        collect_rust_files(&root.join(dir), &root, &mut sources);
    }
    assert!(
        !sources.is_empty(),
        "the scan found no Rust sources under {}; a path change would make this test vacuous",
        root.display()
    );
    sources
}

fn collect_rust_files(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("the scan must be able to read {}: {error}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|error| panic!("directory entry in {}: {error}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, root, out);
            continue;
        }
        if path.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        // This file and `pipeline_error_closure.rs` both quote the banned shapes
        // as their subject, so scanning them would report their own prose.
        let name = path.file_name().unwrap_or_default();
        if name == "ring_fault_selection_is_typed.rs" || name == "pipeline_error_closure.rs" {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("the scan must be able to read {relative}: {error}"));
        out.push((relative, text));
    }
}

/// Source with every comment byte replaced by a space, so a doc comment that
/// quotes a banned shape is not mistaken for one.
fn without_comments(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    let mut idx = 0;
    while idx < chars.len() {
        let c = chars[idx];
        match c {
            '"' => {
                out.push(c);
                idx += 1;
                while idx < chars.len() {
                    let inner = chars[idx];
                    out.push(inner);
                    idx += 1;
                    if inner == '\\' && idx < chars.len() {
                        out.push(chars[idx]);
                        idx += 1;
                    } else if inner == '"' {
                        break;
                    }
                }
            }
            '/' if chars.get(idx + 1) == Some(&'/') => {
                while idx < chars.len() && chars[idx] != '\n' {
                    out.push(' ');
                    idx += 1;
                }
            }
            '/' if chars.get(idx + 1) == Some(&'*') => {
                let mut depth = 1usize;
                out.push(' ');
                out.push(' ');
                idx += 2;
                while idx < chars.len() && depth > 0 {
                    if chars[idx] == '/' && chars.get(idx + 1) == Some(&'*') {
                        depth += 1;
                        out.push(' ');
                        out.push(' ');
                        idx += 2;
                        continue;
                    }
                    if chars[idx] == '*' && chars.get(idx + 1) == Some(&'/') {
                        depth -= 1;
                        out.push(' ');
                        out.push(' ');
                        idx += 2;
                        continue;
                    }
                    out.push(if chars[idx] == '\n' { '\n' } else { ' ' });
                    idx += 1;
                }
            }
            _ => {
                out.push(c);
                idx += 1;
            }
        }
    }
    out.into_iter().collect()
}

/// The source with all whitespace removed, plus the source line each retained
/// character came from, so a match reports a real line number.
fn flatten(text: &str) -> (Vec<char>, Vec<usize>) {
    let mut flat = Vec::new();
    let mut lines = Vec::new();
    let mut line = 1usize;
    for c in text.chars() {
        if c == '\n' {
            line += 1;
        }
        if c.is_whitespace() {
            continue;
        }
        flat.push(c);
        lines.push(line);
    }
    (flat, lines)
}

fn find_from(haystack: &[char], needle: &[char], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (from..=haystack.len() - needle.len())
        .find(|&start| &haystack[start..start + needle.len()] == needle)
}

/// The characters between the parenthesis at `open` and its match, skipping
/// string literals so a `(` inside a message cannot unbalance the walk.
fn balanced_args(flat: &[char], open: usize) -> Option<&[char]> {
    let mut depth = 0usize;
    let mut idx = open;
    while idx < flat.len() {
        match flat[idx] {
            '"' => {
                idx += 1;
                while idx < flat.len() {
                    if flat[idx] == '\\' {
                        idx += 2;
                        continue;
                    }
                    if flat[idx] == '"' {
                        break;
                    }
                    idx += 1;
                }
            }
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&flat[open + 1..idx]);
                }
            }
            _ => {}
        }
        idx += 1;
    }
    None
}

/// A `matches!` whose `PipelineError` pattern binds nothing: the body is
/// exactly `{ .. }` or `(_)`, so every payload the variant carries is discarded
/// and any other value of that variant satisfies it.
fn has_shape_only_pipeline_error_pattern(args: &[char]) -> bool {
    let matches_token: Vec<char> = "matches!(".chars().collect();
    let variant_token: Vec<char> = "PipelineError::".chars().collect();
    if find_from(args, &matches_token, 0).is_none() {
        return false;
    }
    let mut cursor = 0usize;
    while let Some(hit) = find_from(args, &variant_token, cursor) {
        let mut after = hit + variant_token.len();
        while after < args.len() && (args[after].is_alphanumeric() || args[after] == '_') {
            after += 1;
        }
        let rest = &args[after..];
        if rest.starts_with(&['{', '.', '.', '}']) || rest.starts_with(&['(', '_', ')']) {
            return true;
        }
        cursor = hit + variant_token.len();
    }
    false
}

fn scan(path: &str, text: &str, findings: &mut Vec<Finding>) {
    let (flat, lines) = flatten(&without_comments(text));

    let fix_substring: Vec<char> = ".to_string().contains(\"Fix:\")".chars().collect();
    let mut cursor = 0usize;
    while let Some(hit) = find_from(&flat, &fix_substring, cursor) {
        findings.push(Finding {
            path: path.to_string(),
            line: lines[hit],
            shape: "to_string().contains(\"Fix:\") on a runtime error: every PipelineError \
                    variant carries a Fix: clause, so this cannot fail on the wrong fault. \
                    Destructure the variant and assert its payload instead",
        });
        cursor = hit + fix_substring.len();
    }

    for token in ["assert!(", "debug_assert!("] {
        let token: Vec<char> = token.chars().collect();
        let mut cursor = 0usize;
        while let Some(hit) = find_from(&flat, &token, cursor) {
            let open = hit + token.len() - 1;
            if let Some(args) = balanced_args(&flat, open) {
                if has_shape_only_pipeline_error_pattern(args) {
                    findings.push(Finding {
                        path: path.to_string(),
                        line: lines[hit],
                        shape: "shape-only matches! on a PipelineError variant: a `{ .. }` or \
                                `(_)` body discards the payload, so a different value of the \
                                same variant satisfies it. Bind the payload with a \
                                `let ... else { panic!(..) }` and assert its fields",
                    });
                }
            }
            cursor = hit + token.len();
        }
    }
}

/// Every weak assertion shape is denied crate-wide, and the deny list is the
/// scan result rather than a recorded number, so a new one turns this red.
#[test]
fn no_assertion_selects_a_pipeline_error_by_shape_or_by_fix_clause() {
    let mut findings = Vec::new();
    for (path, text) in scanned_sources() {
        scan(&path, &text, &mut findings);
    }

    assert!(
        findings.is_empty(),
        "these assertions cannot fail on the fault they name:\n{}",
        findings
            .iter()
            .map(|finding| format!("  {}:{}: {}", finding.path, finding.line, finding.shape))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The scan itself has to be able to see both shapes, otherwise a clean run
/// proves only that the scanner is broken.
#[test]
fn the_scan_reports_both_banned_shapes_and_ignores_a_typed_assertion() {
    let weak = r#"
fn a() {
    assert!(matches!(err, PipelineError::Backend(_)));
    assert!(matches!(
        err,
        PipelineError::QueueFull { .. }
    ));
    assert!(err.to_string().contains("Fix:"));
}
"#;
    let mut findings = Vec::new();
    scan("probe.rs", weak, &mut findings);
    let mut lines: Vec<usize> = findings.iter().map(|finding| finding.line).collect();
    lines.sort_unstable();
    assert_eq!(
        lines,
        vec![3, 4, 8],
        "the scan must report each weak assertion once, at its own line: {:?}",
        findings
            .iter()
            .map(|finding| (finding.line, finding.shape))
            .collect::<Vec<_>>()
    );

    let typed = r#"
//! A doc comment quoting matches!(err, PipelineError::QueueFull { .. }) is prose.
fn b() {
    assert!(
        matches!(
            err,
            PipelineError::RingEncoding {
                fault: RingEncodingFault::Capacity,
                ..
            }
        ),
        "a capacity fault, got {err:?}"
    );
    let PipelineError::Protocol(ProtocolError::MissingWord { word_idx, .. }) = err else {
        panic!("got {err:?}")
    };
    assert_eq!(word_idx, control::METRICS_BASE as usize);
}
"#;
    let mut clean = Vec::new();
    scan("probe.rs", typed, &mut clean);
    assert!(
        clean.is_empty(),
        "a bound fault discriminant, a let-else destructure, and a doc comment are not weak \
         assertions: {:?}",
        clean
            .iter()
            .map(|finding| (finding.line, finding.shape))
            .collect::<Vec<_>>()
    );
}
