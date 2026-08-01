//! Clean-checkout build governance.
//!
//! A clone of the public repository contains exactly the files git tracks. Anything
//! a crate embeds at compile time with `include_str!` / `include_bytes!` must
//! therefore be tracked, or the workspace does not build for anyone but the operator
//! who happens to have the untracked file on disk.
//!
//! This suite exists because that is precisely what happened: 31 governance tests
//! embedded `docs/optimization/ALL_AXES_ACCELERATION_PLAN.md`, which `.gitignore`
//! marks as private operator state and never publishes. `cargo test` failed to
//! compile on every fresh clone with `couldn't read ...: No such file or directory`,
//! and nothing caught it because the file was always present locally.
//!
//! The failure mode is silent and one-directional: adding a private file to
//! `.gitignore` (or embedding a path that was never added) breaks consumers without
//! breaking the author's own build. These tests are the trip-wire.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("conform crate must live under the vyre repository")
}

/// Every path git tracks, repo-relative and slash-separated.
///
/// Fails loudly rather than degrading to an empty set: an empty set would make every
/// assertion below vacuously pass, turning this suite into decoration.
fn tracked_paths(root: &Path) -> BTreeSet<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to run `git ls-files` in {}: {error}\n\
                 Fix: run this suite inside the vyre git work tree; it verifies what a \
                 clean clone contains and cannot do that without git.",
                root.display()
            )
        });
    assert!(
        output.status.success(),
        "`git ls-files` failed in {}: {}\nFix: run this suite inside the vyre git work tree.",
        root.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let tracked: BTreeSet<String> = String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|entry| !entry.is_empty())
        .map(str::to_owned)
        .collect();
    assert!(
        tracked.len() > 1000,
        "`git ls-files` reported only {} tracked paths, which cannot be the vyre workspace; \
         refusing to pass on an empty or truncated file list.",
        tracked.len()
    );
    tracked
}

/// Every present `*.rs` file tracked by git, skipping build output, vendored
/// trees, and worktree deletions that a clean checkout no longer scans.
fn rust_sources(root: &Path, tracked: &BTreeSet<String>) -> Vec<PathBuf> {
    tracked
        .iter()
        .filter(|path| path.ends_with(".rs"))
        .filter(|path| !path.starts_with("vendor/"))
        .map(|path| root.join(path))
        .filter(|path| path.is_file())
        .collect()
}

/// Extract the literal path argument of each `include_str!` / `include_bytes!`.
///
/// Only occurrences in *code* count. The macro name also appears inside ordinary
/// string literals (a lint in `xtask/src/research_audit/collectors.rs` greps for
/// `"include_str!("`) and inside raw-string test fixtures that hold sample source, and
/// a scanner that ignores literal context reports those as real embeds. Non-literal
/// path forms (`concat!`, `env!("OUT_DIR")`) are skipped deliberately: they resolve at
/// build time from generated state, not from the checkout.
fn embedded_paths(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    let mut i = 0usize;

    // Reads a `"..."` or `r#".."#` literal starting at `at`, returning its content and
    // the index just past its close.
    fn read_string(bytes: &[u8], at: usize) -> Option<(String, usize)> {
        let mut i = at;
        let mut hashes = 0usize;
        let raw = if bytes[i] == b'r' {
            i += 1;
            while i < bytes.len() && bytes[i] == b'#' {
                hashes += 1;
                i += 1;
            }
            true
        } else {
            false
        };
        if i >= bytes.len() || bytes[i] != b'"' {
            return None;
        }
        i += 1;
        let start = i;
        while i < bytes.len() {
            if !raw && bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                let closing = bytes[i + 1..].iter().take_while(|b| **b == b'#').count();
                if !raw || closing >= hashes {
                    let text = String::from_utf8_lossy(&bytes[start..i]).into_owned();
                    return Some((text, i + 1 + if raw { hashes } else { 0 }));
                }
            }
            i += 1;
        }
        None
    }

    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

    while i < bytes.len() {
        // Skip comments.
        if bytes[i] == b'/' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'/' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if bytes[i + 1] == b'*' {
                let mut depth = 1usize;
                i += 2;
                while i < bytes.len() && depth > 0 {
                    if bytes[i..].starts_with(b"/*") {
                        depth += 1;
                        i += 2;
                    } else if bytes[i..].starts_with(b"*/") {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                continue;
            }
        }
        // Skip character literals, so `'"'` does not open a string.
        if bytes[i] == b'\'' && i + 2 < bytes.len() {
            let escaped = bytes[i + 1] == b'\\';
            let close = if escaped { i + 3 } else { i + 2 };
            if close < bytes.len() && bytes[close] == b'\'' {
                i = close + 1;
                continue;
            }
        }
        // Skip whole string literals, including raw strings.
        if bytes[i] == b'"'
            || (bytes[i] == b'r' && i + 1 < bytes.len() && matches!(bytes[i + 1], b'"' | b'#'))
        {
            let starts_ident = i > 0 && is_ident(bytes[i - 1]);
            if !starts_ident {
                if let Some((_, next)) = read_string(bytes, i) {
                    i = next;
                    continue;
                }
            }
        }

        let matched = ["include_str!", "include_bytes!"].into_iter().find(|name| {
            bytes[i..].starts_with(name.as_bytes()) && !(i > 0 && is_ident(bytes[i - 1]))
        });
        let Some(name) = matched else {
            i += 1;
            continue;
        };

        let mut j = i + name.len();
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || !matches!(bytes[j], b'(' | b'[' | b'{') {
            i += name.len();
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if let Some((path, next)) = read_string(bytes, j) {
            found.push(path);
            i = next;
        } else {
            // concat!/env!/macro-generated path: not a checkout dependency.
            i += name.len();
        }
    }
    found
}

/// Resolve `../` and `./` segments without touching the filesystem, so a missing
/// file is reported by the assertion rather than by `canonicalize` failing.
fn normalize(path: &Path) -> String {
    let text = path.to_string_lossy();
    let mut parts: Vec<&str> = Vec::new();
    for component in text.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    // Keep the root marker: dropping it turned `/home/x/vyre/...` into `home/x/vyre/...`
    // and made every repo-relative prefix strip fail.
    let joined = parts.join("/");
    if text.starts_with('/') {
        format!("/{joined}")
    } else {
        joined
    }
}

#[test]
fn every_compile_time_embedded_file_is_tracked_by_git() {
    let root = repo_root();
    let root_prefix = format!("{}/", normalize(&root));
    let tracked = tracked_paths(&root);

    let mut violations: Vec<String> = Vec::new();
    for source in rust_sources(&root, &tracked) {
        let text = std::fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", source.display()));
        let source_dir = source
            .parent()
            .expect("a source file has a parent directory");
        let source_rel = normalize(&source)
            .strip_prefix(&root_prefix)
            .expect("source lives under the repo root")
            .to_owned();

        for embedded in embedded_paths(&text) {
            let absolute = normalize(&source_dir.join(&embedded));
            let Some(relative) = absolute.strip_prefix(&root_prefix) else {
                violations.push(format!(
                    "{source_rel} embeds {embedded}, which resolves outside the repository"
                ));
                continue;
            };
            if !tracked.contains(relative) {
                violations.push(format!("{source_rel} embeds untracked {relative}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "{} compile-time embed(s) reference files a clean clone does not contain, so \
         `cargo build` fails for every consumer:\n  {}\n\
         Fix: commit the referenced file, or assert against a committed artifact instead. \
         Never embed a path matched by .gitignore.",
        violations.len(),
        violations.join("\n  ")
    );
}

/// The specific regression: the private acceleration plan must never be embedded again.
///
/// Scoped to compile-time embedding, not to any mention of the path. `xtask` legitimately
/// names the plan as the default argument of `acceleration-plan-gate --plan PATH`, an
/// operator command that reads it at run time and errors loudly when it is absent. That
/// is a private-input tool, not a build dependency. Embedding the same file with
/// `include_str!` is what broke every clean clone.
#[test]
fn the_private_acceleration_plan_is_never_embedded_at_compile_time() {
    let root = repo_root();
    let root_prefix = format!("{}/", normalize(&root));
    let tracked = tracked_paths(&root);

    let mut offenders: Vec<String> = Vec::new();
    for source in rust_sources(&root, &tracked) {
        let text = std::fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", source.display()));
        let source_dir = source
            .parent()
            .expect("a source file has a parent directory");
        let source_rel = normalize(&source)
            .strip_prefix(&root_prefix)
            .expect("source lives under the repo root")
            .to_owned();

        for embedded in embedded_paths(&text) {
            if normalize(&source_dir.join(&embedded)).ends_with("ALL_AXES_ACCELERATION_PLAN.md") {
                offenders.push(format!("{source_rel} embeds {embedded}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "docs/optimization/ALL_AXES_ACCELERATION_PLAN.md is private operator state that \
         .gitignore excludes from the public repository, so embedding it breaks `cargo build` \
         on every clean clone. {} offender(s):\n  {}\n\
         Fix: assert the governance contract against the committed \
         docs/optimization/*.toml artifacts, which carry the same VX row ranges.",
        offenders.len(),
        offenders.join("\n  ")
    );
}

#[test]
fn gitignored_paths_are_absent_from_the_tracked_file_list() {
    let root = repo_root();
    let tracked = tracked_paths(&root);

    let listed: Vec<&String> = tracked.iter().collect();
    let mut stdin_payload = String::new();
    for path in &listed {
        stdin_payload.push_str(path);
        stdin_payload.push('\n');
    }

    let output = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["check-ignore", "--no-index", "--stdin"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .expect("stdin was piped")
                .write_all(stdin_payload.as_bytes())?;
            child.wait_with_output()
        })
        .expect("git check-ignore must run inside the vyre git work tree");

    // `check-ignore` exits 1 when nothing matched, which is the state we want.
    let matched: Vec<&str> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        // Historical, deliberately-tracked operator docs that shipped before the
        // .gitignore policy widened, and that shipped rustdoc still links to.
        .filter(|line| !line.starts_with("audits/"))
        .filter(|line| !line.starts_with("docs/archive/"))
        .filter(|line| !line.starts_with("docs/legacy/"))
        .map(|line| Box::leak(line.to_owned().into_boxed_str()) as &str)
        .collect();

    assert!(
        matched.is_empty(),
        "{} tracked file(s) match a .gitignore rule, so the repository both publishes and \
         claims to exclude them:\n  {}\n\
         Fix: either untrack the file (private operator state) or narrow the .gitignore rule \
         (genuinely public content).",
        matched.len(),
        matched.join("\n  ")
    );
}
