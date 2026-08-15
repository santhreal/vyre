//! `cargo xtask cross-target` - every platform the source claims compiles.
//!
//! # The defect this exists for
//!
//! A dependency whose build script picks a C or assembly path off the target
//! triple turns a cross-compile into a hunt for that target's C toolchain, and
//! the host cannot see the failure: the host has a C compiler, the host build is
//! green, and no other target is ever attempted. blake3 sat in this tree that
//! way. Its build script compiles `c/blake3_neon.c` on aarch64 and
//! `c/blake3_*_x86-64_unix.S` on x86_64, and disabling only the aarch64 arm made
//! both Apple aarch64 targets compile while `x86_64-apple-darwin` failed with the
//! byte-identical error. Seven real compile errors in the Apple backend had been
//! sitting behind that for as long as the flag was wrong.
//!
//! # Why the criterion is a build
//!
//! The cheap proxy does not work. Judging a manifest - "this package has a build
//! script and a `cc` build-dependency" - is red on a tree that is right, because
//! a crate can declare `cc` unconditionally and have a feature that stops the
//! invocation. The only thing that answers the question is compiling for the
//! target, so that is what this does.
//!
//! # Where the platform list comes from
//!
//! The `target_os` literals in workspace source. A `cfg` arm for a platform is
//! the claim that the tree supports it; a comment naming one is not. A
//! `target_os` with no triple in [`TRIPLE_FOR_OS`] is a finding rather than a
//! skip, so adding an arm for a new platform turns this red until somebody
//! records which triple proves it.
//!
//! # Cost, and which subset it belongs in
//!
//! One `cargo check` per declared triple over the three crates every backend and
//! the runtime link. That is a real build per triple with nothing shared between
//! them, minutes rather than seconds, and it cannot be narrowed by a diff: a
//! dependency edit anywhere in the graph changes the answer. It belongs in the
//! nightly and pre-release subsets, not on every push. Nothing on the push path
//! can catch this class, which is the honest cost of the class: a per-push gate
//! would have to be the manifest proxy, and the manifest proxy is wrong.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

/// The crates every backend and the runtime link. Compiling these compiles the
/// whole product dependency graph; the benchmark and evidence crates link
/// SQLite, OpenSSL and PCRE2 and are host tools, so a cross target is not a
/// claim they make.
const PRODUCT_CRATES: &[&str] = &["vyre-driver", "vyre-runtime", "vyre-megakernel"];

/// The triple that proves each `target_os` a `cfg` arm may name.
///
/// One triple per platform, not every triple that platform has: the failure this
/// catches is a build script branching on the target, and one triple per
/// `target_os` plus one per architecture family is what distinguishes those
/// branches. `macos` is pinned to the x86_64 triple on purpose - the aarch64 one
/// was the arm that got silenced, so the x86_64 one is the arm that was blind.
const TRIPLE_FOR_OS: &[(&str, &str)] = &[
    ("android", "aarch64-linux-android"),
    ("ios", "aarch64-apple-ios"),
    ("linux", "x86_64-unknown-linux-gnu"),
    ("macos", "x86_64-apple-darwin"),
    ("windows", "x86_64-pc-windows-msvc"),
];

/// Cap on one source file read while scanning for `cfg` arms.
const MAX_SOURCE_BYTES: u64 = 2_097_152;

/// Compiles the product crates for every platform the source declares an arm for.
pub struct CrossTarget;

impl Gate for CrossTarget {
    fn name(&self) -> &'static str {
        "cross-target"
    }

    fn help(&self) -> &'static str {
        "Compile the product crates for every target_os the source declares a cfg arm for"
    }

    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let root = xtask::checkout::checkout_root();
        let declared = declared_target_oses(&root);
        if declared.len() < 2 {
            return Err(GateError::new(
                format!(
                    "cross-target found {} target_os arm(s) in the workspace, which means the scan \
                     found nothing rather than a tree with one platform",
                    declared.len()
                ),
                "run it inside a checkout of the workspace",
            ));
        }

        let recorded: BTreeMap<&str, &str> = TRIPLE_FOR_OS.iter().copied().collect();
        let mut report = Report::clean();
        let mut checked = 0usize;

        for (os, origin) in &declared {
            let Some(triple) = recorded.get(os.as_str()) else {
                report.find(located(
                    origin.as_ref(),
                    format!(
                        "the source branches on target_os = \"{os}\" and no triple proves that arm"
                    ),
                    format!(
                        "add (\"{os}\", \"<triple>\") to TRIPLE_FOR_OS in \
                         xtask-registry/src/gates/cross_target.rs, or drop the cfg arm; an \
                         unrecorded platform is a claim nothing compiles"
                    ),
                ));
                continue;
            };
            checked += 1;
            match check_triple(&root, triple) {
                TripleResult::Clean => {}
                TripleResult::NotInstalled => {
                    return Err(GateError::new(
                        format!(
                            "cross-target cannot check {triple}, so target_os = \"{os}\" went \
                             unjudged; a gate that silently checks fewer triples than the tree \
                             declares is the defect it exists to remove"
                        ),
                        format!("rustup target add {triple}"),
                    ))
                }
                TripleResult::Failed(detail) => report.find(located(
                    origin.as_ref(),
                    format!("target {triple} does not compile: {detail}"),
                    "a build script that picks a C or assembly path off the target triple needs \
                     that target's toolchain; take the dependency's pure-Rust feature, or move it \
                     behind a host-only crate"
                        .to_string(),
                )),
            }
        }

        report.note(format!(
            "{} declared platform(s), {checked} triple(s) compiled over {} crate(s)",
            declared.len(),
            PRODUCT_CRATES.len()
        ));
        Ok(report)
    }
}

/// The `cfg` arm that claimed one platform.
struct Origin {
    file: PathBuf,
    line: u32,
}

/// A finding that names the arm when the platform came from one, and the tree
/// as a whole when it is the host platform no arm mentions.
fn located(origin: Option<&Origin>, message: String, fix: String) -> Finding {
    match origin {
        Some(origin) => Finding::at(origin.file.clone(), origin.line, message, fix),
        None => Finding::new(message, fix),
    }
}

enum TripleResult {
    Clean,
    NotInstalled,
    Failed(String),
}

/// Compile [`PRODUCT_CRATES`] for one triple.
///
/// No build-affecting flag or environment variable is set: the target is the one
/// thing that must differ, and everything else has to be the build the tree
/// declares in its own configuration or the answer is about a build nobody runs.
fn check_triple(root: &Path, triple: &str) -> TripleResult {
    let mut command = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()));
    command.current_dir(root).arg("check").arg("--target").arg(triple);
    for crate_name in PRODUCT_CRATES {
        command.arg("-p").arg(crate_name);
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(error) => return TripleResult::Failed(format!("cargo could not be run: {error}")),
    };
    if output.status.success() {
        return TripleResult::Clean;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("may not be installed") || stderr.contains("rustup target add") {
        return TripleResult::NotInstalled;
    }
    TripleResult::Failed(first_error_line(&stderr))
}

/// The first real error cargo printed, so the finding names the cause rather
/// than the last line of a build log.
fn first_error_line(stderr: &str) -> String {
    stderr
        .lines()
        .find(|line| line.starts_with("error"))
        .unwrap_or("the target check reported no error line")
        .trim()
        .to_string()
}

/// Every `target_os` the workspace source branches on, with the arm that claimed
/// it.
///
/// The host platform is always included: a tree that compiles nowhere else still
/// has to compile here, and its absence from every `cfg` arm is not a reason to
/// leave it unchecked. It carries no origin, because no line claimed it.
fn declared_target_oses(root: &Path) -> BTreeMap<String, Option<Origin>> {
    let mut found = BTreeMap::new();
    found.insert(std::env::consts::OS.to_string(), None);
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let path: PathBuf = entry.path();
            if path.is_dir() {
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
            } else if name.ends_with(".rs") {
                if let Ok(source) =
                    xtask::output_arg::read_text_bounded(&path, MAX_SOURCE_BYTES, "cross target")
                {
                    let relative = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
                    collect_target_oses(&source, &relative, &mut found);
                }
            }
        }
    }
    found
}

/// Pull every `target_os` literal a compiled `cfg` arm names out of one file.
///
/// Comment text is cut off at `//` before the scan, because a comment naming a
/// platform is prose and compiles nothing. That also drops the placeholder
/// spellings this file's own documentation uses, which is why the gate is not
/// red on itself. The literal has to read as a platform token as well: rustc
/// accepts only `[a-z0-9_]` there, so anything else is a placeholder rather than
/// a claim.
fn collect_target_oses(source: &str, file: &Path, found: &mut BTreeMap<String, Option<Origin>>) {
    const MARKER: &str = "target_os = \"";
    for (index, line) in source.lines().enumerate() {
        let code = match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        };
        let mut rest = code;
        while let Some(at) = rest.find(MARKER) {
            rest = &rest[at + MARKER.len()..];
            let Some(end) = rest.find('"') else { break };
            let os = &rest[..end];
            rest = &rest[end..];
            if os.is_empty() || !os.bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_') {
                continue;
            }
            found.entry(os.to_string()).or_insert_with(|| {
                Some(Origin {
                    file: file.to_path_buf(),
                    line: u32::try_from(index + 1).unwrap_or(u32::MAX),
                })
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_recorded_platform_has_a_distinct_triple() {
        let mut triples = BTreeSet::new();
        for (os, triple) in TRIPLE_FOR_OS {
            assert!(
                triples.insert(*triple),
                "{triple} proves two platforms; {os} needs its own"
            );
            assert!(
                triple.contains('-'),
                "{triple} recorded for {os} is not a target triple"
            );
        }
    }

    #[test]
    fn a_cfg_arm_is_read_and_a_comment_is_not() {
        let mut found = BTreeMap::new();
        collect_target_oses(
            "// supported on target_os = \"plan9\" one day\n\
             #[cfg(any(target_os = \"macos\", target_os = \"ios\"))]\nmod native {}\n",
            Path::new("a/b.rs"),
            &mut found,
        );
        assert_eq!(found.keys().cloned().collect::<Vec<String>>(), ["ios", "macos"]);
        let origin = found["macos"].as_ref().expect("the arm is located");
        assert_eq!(origin.file, Path::new("a/b.rs"));
        assert_eq!(origin.line, 2);
    }

    #[test]
    fn a_placeholder_spelling_is_not_a_platform() {
        let mut found = BTreeMap::new();
        collect_target_oses(
            "#[cfg(target_os = \"...\")]\n#[cfg(target_os = \"MacOS\")]\n#[cfg(target_os = \"\")]\n",
            Path::new("a/b.rs"),
            &mut found,
        );
        assert!(found.is_empty(), "got {:?}", found.keys().collect::<Vec<_>>());
    }

    #[test]
    fn a_file_with_no_cfg_arm_contributes_nothing() {
        let mut found = BTreeMap::new();
        collect_target_oses(
            "fn main() { println!(\"target_os\"); }\n",
            Path::new("a/b.rs"),
            &mut found,
        );
        assert!(found.is_empty(), "got {:?}", found.keys().collect::<Vec<_>>());
    }

    #[test]
    fn an_unterminated_literal_does_not_hang_or_panic() {
        let mut found = BTreeMap::new();
        collect_target_oses("#[cfg(target_os = \"macos", Path::new("a/b.rs"), &mut found);
        assert!(found.is_empty(), "got {:?}", found.keys().collect::<Vec<_>>());
    }

    #[test]
    fn the_first_error_line_is_the_one_reported() {
        let stderr = "   Compiling blake3 v1.8.5\n\
                      error: failed to run custom build command for `blake3 v1.8.5`\n\
                      error: could not compile `blake3`\n";
        assert_eq!(
            first_error_line(stderr),
            "error: failed to run custom build command for `blake3 v1.8.5`"
        );
    }

    #[test]
    fn a_failure_with_no_error_line_still_names_itself() {
        assert_eq!(
            first_error_line("warning: something\n"),
            "cargo check failed with no error line"
        );
    }
}
