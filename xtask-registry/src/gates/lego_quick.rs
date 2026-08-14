//! `cargo xtask lego-quick`  -  fast pre-commit gate.
//!
//! Runs the file-only subset of `lego-audit` against the staged diff
//! only. Target wall-clock ≤ 2s on a 10-file diff so it can sit in
//! `.git/hooks/pre-commit` without writers reaching for `--no-verify`.
//!
//! Three default checks, no inventory walk, no fingerprinting:
//!
//! 1. **Raw IR construction** (delegates to `vyre_lints`): no
//!    `Node::*` / `Expr::*` constructors in `vyre-libs/src/**`.
//! 2. **Cross-dialect reach-through**: a Tier-3 dialect under
//!    `vyre-libs/src/<dialect>/` cannot import from
//!    `crate::<other_dialect>` or `vyre_libs::<other_dialect>`.
//! 3. **Large-file advisory**: a staged `*.rs` file over the
//!    `LARGE_FILE_ADVISORY_LINES` line guideline is *flagged for review*,
//!    not failed. Crossing 500 lines is a prompt to ask whether the file
//!    has grown a second responsibility worth splitting, it is a
//!    guideline, not a law. The hard size cap (a genuine god-file ceiling
//!    with a ratcheted per-file exception list) lives in
//!    `scripts/check_max_file_size.sh`, not here.
//!
//! The full op-fingerprint reinvention check (`lego-audit` check 1)
//! requires loading every registered operation and remains a separate CI gate.
//!
//! Exit code 0 on clean and on advisory-only runs; 1 only when a hard
//! raw-IR or cross-dialect boundary is violated.

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use xtask::gates::use_paths::{collect_use_paths, is_test_source_path};

/// Line count at which a source file is *flagged for a split-by-responsibility
/// review*. This is a guideline, not a law: crossing it does not fail the gate.
/// The hard god-file ceiling is enforced separately by
/// `scripts/check_max_file_size.sh` with its ratcheted per-file exception list.
const LARGE_FILE_ADVISORY_LINES: usize = 500;
const MAX_LEGO_QUICK_SOURCE_BYTES: u64 = 2_097_152;

pub(crate) fn run(args: &[String]) {
    let staged_only = !args.iter().any(|a| a == "--all");
    let root = match workspace_root() {
        Some(r) => r,
        None => {
            eprintln!(
                "Fix: cargo_full run --bin xtask -- lego-quick must run from a git checkout of the vyre workspace."
            );
            process::exit(1);
        }
    };

    let files = if staged_only {
        match staged_rust_files(&root) {
            Ok(files) => files,
            Err(err) => {
                eprintln!(
                    "Fix: failed to list staged files via `git diff --cached --name-only`: {err}"
                );
                process::exit(1);
            }
        }
    } else {
        all_rust_files(&root)
    };

    if files.is_empty() {
        println!("lego-quick: no staged Rust files; nothing to check.");
        return;
    }

    let mut findings: Vec<Finding> = Vec::new();
    findings.extend(check_raw_ir(&root, &files));
    findings.extend(check_cross_dialect(&root, &files));
    findings.extend(check_god_files(&root, &files));

    let check_count = 3;

    // `large-file` findings are advisory: they flag a split-by-responsibility
    // review, they do not fail the gate. Everything else is a hard law.
    let sort_key = |f: &Finding| (f.file.clone(), f.line, f.category.clone());
    let (mut advisories, mut blockers): (Vec<Finding>, Vec<Finding>) = findings
        .into_iter()
        .partition(|f| f.category == "large-file");
    advisories.sort_by_key(|finding| sort_key(finding));
    blockers.sort_by_key(|finding| sort_key(finding));

    if blockers.is_empty() && advisories.is_empty() {
        println!(
            "lego-quick: ✓ {} staged Rust file(s) clean ({} checks).",
            files.len(),
            check_count
        );
        return;
    }

    for f in &blockers {
        println!(
            "  ✗ {}:{} | {} | {} | fix: {}",
            f.file, f.line, f.category, f.message, f.fix
        );
    }
    for f in &advisories {
        println!(
            "  • {}:{} | {} | {} | {}",
            f.file, f.line, f.category, f.message, f.fix
        );
    }
    println!();

    if blockers.is_empty() {
        println!(
            "lego-quick: ✓ {} staged Rust file(s) pass ({} checks); \
             {} large-file advisory note(s) for review (non-blocking).",
            files.len(),
            check_count,
            advisories.len()
        );
        return;
    }

    println!(
        "lego-quick: FAILED  -  {} blocking finding(s) ({} advisory) across {} staged file(s). \
         Resolve the blocking findings before commit, or run \
         `cargo_full run --bin xtask -- lego-quick --all` to scan the whole tree.",
        blockers.len(),
        advisories.len(),
        files.len()
    );
    process::exit(1);
}

#[derive(Debug)]
struct Finding {
    file: String,
    line: u32,
    category: String,
    message: String,
    fix: String,
}

fn workspace_root() -> Option<PathBuf> {
    Some(xtask::checkout::checkout_root())
}

fn staged_rust_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let output = Command::new("git")
        .args(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.ends_with(".rs") {
            continue;
        }
        let path = root.join(trimmed);
        if path.is_file() {
            out.push(path);
        }
    }
    Ok(out)
}

fn all_rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".git" | "target" | "target-codex" | "target-fusion-fix"
            )
        })
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path.to_path_buf());
        }
    }
    out
}

fn check_raw_ir(root: &Path, files: &[PathBuf]) -> Vec<Finding> {
    let allowlist_path = root.join("vyre-lints").join("allowlist.toml");
    let allow = match vyre_lints::allowlist::load(&allowlist_path) {
        Ok(a) => a,
        Err(_) => vyre_lints::allowlist::Allowlist::empty(),
    };

    // The files this run is responsible for, spelled the way the lint
    // reports them. In staged mode that is the staged diff; under `--all`
    // it is every source file under a measured root.
    let considered = considered_sources(files, &allow);
    if considered.is_empty() {
        return Vec::new();
    }

    // One walk per measured root. `scan_tree` recurses, so calling it per
    // candidate file reparsed the whole tree once per file, and for a file
    // directly in a root that meant reparsing every descendant.
    let mut violations = Vec::new();
    for measured_root in allow.measured_roots() {
        let Ok(found) = vyre_lints::raw_ir_in_libs::scan_tree(&root.join(measured_root), &allow)
        else {
            continue;
        };
        violations.extend(found);
    }

    violations
        .into_iter()
        .filter(|violation| considered.contains(&violation.file))
        .map(|violation| Finding {
            file: violation.file,
            line: violation.line,
            category: "raw-ir".to_string(),
            message: violation.message,
            fix: "compose registered operations instead of constructing Node/Expr directly"
                .to_string(),
        })
        .collect()
}

/// Source files under a measured root that this run is responsible for.
///
/// The roots come from `vyre-lints/allowlist.toml` rather than a hardcoded
/// crate name, so relocating a composition domain between crates does not
/// silently move thousands of sites into or out of the pinned count.
fn considered_sources(
    files: &[PathBuf],
    allow: &vyre_lints::allowlist::Allowlist,
) -> BTreeSet<String> {
    files
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .filter_map(|path| {
            allow
                .measured_roots()
                .iter()
                .find_map(|measured_root| workspace_relative_under(&path, measured_root))
        })
        .filter(|workspace_rel| !allow.contains(workspace_rel))
        .collect()
}

/// The workspace-relative form of `path` when it lies under `measured_root`.
fn workspace_relative_under(path: &str, measured_root: &str) -> Option<String> {
    let marker = format!("{measured_root}/");
    path.find(&marker).map(|idx| path[idx..].to_string())
}

fn workspace_relative(path: &str, marker: &str) -> String {
    match path.find(marker) {
        Some(idx) => path[idx..].to_string(),
        None => path.to_string(),
    }
}

/// Check 4 (file-only): a Tier-3 dialect under `vyre-libs/src/<X>/`
/// must not import `crate::<Y>::...` or `vyre_libs::<Y>::...` for
/// `Y != X`. The cross-dialect coupling belongs in `vyre-primitives`.
fn check_cross_dialect(root: &Path, files: &[PathBuf]) -> Vec<Finding> {
    let libs_root = root.join("vyre-libs").join("src");
    let dialects: Vec<String> = match std::fs::read_dir(&libs_root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| {
                !matches!(
                    n.as_str(),
                    "region" | "tensor_ref" | "builder" | "buffer_names" | "descriptor"
                )
            })
            .collect(),
        Err(_) => return Vec::new(),
    };

    let mut out = Vec::new();
    for path in files {
        if is_test_source_path(path) {
            continue;
        }
        let path_str = path.to_string_lossy();
        let Some(idx) = path_str.find("vyre-libs/src/") else {
            continue;
        };
        let after = &path_str[idx + "vyre-libs/src/".len()..];
        let Some(this_dialect) = after.split('/').next() else {
            continue;
        };
        if !dialects.iter().any(|d| d == this_dialect) {
            continue;
        }
        let Ok(text) = read_text_bounded(path) else {
            continue;
        };
        let Ok(file) = syn::parse_file(&text) else {
            out.push(Finding {
                file: workspace_relative(&path_str, "vyre-libs/"),
                line: 0,
                category: "parse".to_string(),
                message: "failed to parse Rust source".to_string(),
                fix: "make the file syntactically valid before committing".to_string(),
            });
            continue;
        };
        for use_path in collect_use_paths(&file) {
            if use_path.is_public {
                continue;
            }
            for other in &dialects {
                if other == this_dialect {
                    continue;
                }
                if use_path.imports_dialect(other) {
                    out.push(Finding {
                        file: workspace_relative(&path_str, "vyre-libs/"),
                        line: use_path.line as u32,
                        category: "cross-dialect".to_string(),
                        message: format!(
                            "imports `{}` from sibling dialect `{}`",
                            use_path.segments.join("::"),
                            other
                        ),
                        fix: "hoist the shared piece into vyre-primitives, or route via a public re-export at crate root".to_string(),
                    });
                }
            }
        }
    }
    out
}

fn check_god_files(root: &Path, files: &[PathBuf]) -> Vec<Finding> {
    let mut out = Vec::new();
    for path in files {
        let Ok(text) = read_text_bounded(path) else {
            continue;
        };
        let line_count = text.lines().count();
        if line_count <= LARGE_FILE_ADVISORY_LINES {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());
        out.push(Finding {
            file: rel,
            line: line_count as u32,
            category: "large-file".to_string(),
            message: format!(
                "{line_count} lines is over the {LARGE_FILE_ADVISORY_LINES}-line review guideline"
            ),
            fix: "review whether this file has grown a second responsibility; \
                  split by responsibility if so (this is advisory, not a build failure)"
                .to_string(),
        });
    }
    out
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_LEGO_QUICK_SOURCE_BYTES, "lego quick")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write(dir: &Path, rel: &str, body: &str) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn large_file_check_flags_oversize_as_advisory() {
        let dir = TempDir::new().unwrap();
        let body = "fn _f() {}\n".repeat(LARGE_FILE_ADVISORY_LINES + 5);
        let p = write(dir.path(), "vyre-libs/src/math/big.rs", &body);
        let findings = check_god_files(dir.path(), &[p]);
        assert_eq!(findings.len(), 1);
        // Advisory category: partitioned out of the blocking set at exit, so a
        // large file is flagged for review but never fails the gate.
        assert_eq!(findings[0].category, "large-file");
    }

    /// WHY: `check_raw_ir` was rewritten from one recursive `scan_tree` per
    /// candidate file to a single walk plus a filter. The old shape reparsed
    /// the crate once per source, and a file sitting directly in
    /// `vyre-libs/src` dragged every descendant through syn on each pass.
    ///
    /// This test pins the reporting contract the rewrite has to preserve: one
    /// finding per offending file, none for a clean one. It does NOT catch the
    /// performance defect itself, which was proven by measurement, and it does
    /// not check the line numbers the lint assigns, which are the lint's own
    /// contract. It does fail on a duplicate, which is the way a filter-based
    /// rewrite breaks.
    #[test]
    fn each_raw_ir_violation_is_reported_once_across_sibling_files() {
        let dir = TempDir::new().unwrap();
        let offender = "use vyre_foundation::ir::Expr;\npub fn f() -> Expr { Expr::u32(0) }\n";
        let first = write(dir.path(), "vyre-libs/src/math/first.rs", offender);
        let second = write(dir.path(), "vyre-libs/src/math/second.rs", offender);
        let clean = write(dir.path(), "vyre-libs/src/math/clean.rs", "pub fn g() {}\n");

        let findings = check_raw_ir(dir.path(), &[first, second, clean]);

        // Walk order is not a contract; the gate sorts before printing. A
        // duplicate still fails this, which is the point.
        let mut reported: Vec<&str> = findings.iter().map(|f| f.file.as_str()).collect();
        reported.sort_unstable();
        assert_eq!(
            reported,
            vec![
                "vyre-libs/src/math/first.rs",
                "vyre-libs/src/math/second.rs"
            ],
            "each offending file is named exactly once and the clean file not at all"
        );
        assert!(findings.iter().all(|f| f.category == "raw-ir"));
    }

    /// WHY: the one-walk rewrite scans the whole crate and then filters, so a
    /// violation in a file the run is not responsible for must not leak into
    /// the report. Staged mode depends on that.
    #[test]
    fn a_violation_outside_the_considered_set_is_not_reported() {
        let dir = TempDir::new().unwrap();
        let offender = "use vyre_foundation::ir::Expr;\npub fn f() -> Expr { Expr::u32(0) }\n";
        let staged = write(dir.path(), "vyre-libs/src/math/staged.rs", offender);
        write(dir.path(), "vyre-libs/src/math/unstaged.rs", offender);

        let findings = check_raw_ir(dir.path(), &[staged]);

        let reported: Vec<&str> = findings.iter().map(|f| f.file.as_str()).collect();
        assert_eq!(reported, vec!["vyre-libs/src/math/staged.rs"]);
    }

    #[test]
    fn large_file_check_quiet_within_guideline() {
        let dir = TempDir::new().unwrap();
        let body = "fn _f() {}\n".repeat(50);
        let p = write(dir.path(), "vyre-libs/src/math/small.rs", &body);
        let findings = check_god_files(dir.path(), &[p]);
        assert!(findings.is_empty());
    }

    #[test]
    fn cross_dialect_check_flags_sibling_import() {
        let dir = TempDir::new().unwrap();
        // Set up two dialects so the dialect-name discovery succeeds.
        write(dir.path(), "vyre-libs/src/math/mod.rs", "");
        write(dir.path(), "vyre-libs/src/parsing/mod.rs", "");
        let p = write(
            dir.path(),
            "vyre-libs/src/math/uses_parsing.rs",
            "use crate::parsing::lexer;\nfn _f() {}\n",
        );
        let findings = check_cross_dialect(dir.path(), &[p]);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].category, "cross-dialect");
        assert!(findings[0].message.contains("parsing"));
    }

    #[test]
    fn cross_dialect_check_allows_same_dialect_import() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "vyre-libs/src/math/mod.rs", "");
        write(dir.path(), "vyre-libs/src/parsing/mod.rs", "");
        let p = write(
            dir.path(),
            "vyre-libs/src/math/uses_self.rs",
            "use crate::math::reduce;\nfn _f() {}\n",
        );
        let findings = check_cross_dialect(dir.path(), &[p]);
        assert!(findings.is_empty());
    }

    #[test]
    fn cross_dialect_check_allows_vyre_primitives_import() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "vyre-libs/src/math/mod.rs", "");
        write(dir.path(), "vyre-libs/src/parsing/mod.rs", "");
        let p = write(
            dir.path(),
            "vyre-libs/src/math/uses_primitives.rs",
            "use vyre_primitives::reduce_sum;\nfn _f() {}\n",
        );
        let findings = check_cross_dialect(dir.path(), &[p]);
        assert!(findings.is_empty());
    }
}
