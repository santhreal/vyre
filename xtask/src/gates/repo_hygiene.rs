//! What the checkout itself must and must not contain.
//!
//! Two rules live here. The repository carries a fixed set of contribution and
//! licensing files and no build output, developer cache, binary artifact or
//! silent-skip language. And there is one execution queue, so a second committed
//! plan surface is a finding.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Files every checkout carries.
const REQUIRED: &[&str] = &[
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "CODEOWNERS",
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/gpu-parity.yml",
    ".github/workflows/architectural-invariants.yml",
];

/// Instruction files that redirect to `AGENTS.md` instead of holding policy.
///
/// A redirect is required only when the checkout carries `AGENTS.md`. A published
/// tree that tracks no instruction file has nothing to redirect to, and demanding
/// one there reported every clean checkout, including CI, as broken. Present but
/// holding policy is still a finding whether or not `AGENTS.md` is tracked: that
/// is the second policy surface the rule exists to catch.
const REDIRECTS: &[&str] = &["CLAUDE.md", "GEMINI.md"];

/// Directory names that are developer cache or build output.
const CACHE_DIRECTORIES: &[&str] = &[".pytest_cache", ".cursor", "__pycache__"];
const OUTPUT_DIRECTORIES: &[&str] = &["node_modules", ".venv", ".next", "dist"];

/// Extensions a source repository does not carry.
const FORBIDDEN_EXTENSIONS: &[&str] = &[
    "rlib", "so", "dylib", "exe", "o", "a", "bin", "dll", "lib", "pdb", "pyd", "whl", "tgz", "zip",
    "old", "backup", "orig", "bak",
];

/// Names that end a path regardless of extension parsing.
const FORBIDDEN_SUFFIXES: &[&str] = &[".tar.gz"];

/// Escape hatches that would let the GPU requirement be turned off.
const GPU_ESCAPES: &[&str] = &["no-gpu", "gpu-feature"];

/// Language that describes skipping a test because no device was found.
const SILENT_SKIPS: &[(&str, &str)] = &[("no GPU", "skipp"), ("adapter missing", "skipp")];

/// The files that own a silent-skip rule and therefore spell what it forbids.
///
/// The pattern the rule looks for is a string literal, and so is the rule's own
/// table, so a gate that scans its own source reports itself and can never reach
/// zero. Masking string literals is not the answer: a real silent skip prints its
/// excuse from a literal, which is exactly what the rule is for. The test below
/// requires each row to exist and to carry the language, so a row that stops
/// spelling the rule stops being an exemption.
const SILENT_SKIP_RULE_SOURCES: &[&str] = &[
    "xtask/src/gates/gpu_loudness.rs",
    "xtask/src/gates/repo_hygiene.rs",
];

/// Instruction files, redirects and backlog files hold their agreed shape.
pub struct RepoHygiene;

impl Gate for RepoHygiene {
    fn name(&self) -> &'static str {
        "repo-hygiene"
    }

    fn help(&self) -> &'static str {
        "required repository files, and artifacts a source tree must not carry"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        for required in REQUIRED {
            if !tree.exists(required) {
                report.find(Finding::in_file(
                    *required,
                    "required repository file is missing",
                    "restore the file; it is part of the contribution and licensing surface",
                ));
            }
        }

        let templates = tree
            .paths()
            .iter()
            .filter(|path| {
                path.parent() == Some(Path::new(".github/ISSUE_TEMPLATE"))
                    && path.extension().and_then(|value| value.to_str()) == Some("md")
            })
            .count();
        if templates < 3 {
            report.find(Finding::in_file(
                ".github/ISSUE_TEMPLATE",
                format!("{templates} issue template(s), fewer than the three required"),
                "add the missing templates so a reporter is routed rather than guessing",
            ));
        }

        let policy_is_tracked = tree.exists("AGENTS.md");
        report.note(if policy_is_tracked {
            "AGENTS.md is in the checkout, so each instruction redirect is required".to_string()
        } else {
            "no AGENTS.md in the checkout, so a redirect is judged only where one exists"
                .to_string()
        });
        for redirect in REDIRECTS {
            if !tree.exists(redirect) {
                if policy_is_tracked {
                    report.find(Finding::in_file(
                        *redirect,
                        "instruction redirect is missing",
                        "restore the file as a short redirect to AGENTS.md",
                    ));
                }
                continue;
            }
            let text = tree.read(redirect)?;
            let lowered = text.to_ascii_lowercase();
            if !lowered.contains("compatibility redirect") || !text.contains("AGENTS.md") {
                report.find(Finding::in_file(
                    *redirect,
                    "instruction file is not a redirect to AGENTS.md",
                    "make the file a compatibility redirect naming AGENTS.md; \
                     AGENTS.md is the only place policy lives",
                ));
            } else if text.lines().count() > 8 {
                report.find(Finding::in_file(
                    *redirect,
                    format!(
                        "redirect is {} lines, above the eight-line bound",
                        text.lines().count()
                    ),
                    "cut the file back to a redirect; policy belongs in AGENTS.md",
                ));
            }
        }

        for directory in walk_directories(&ctx.root) {
            let name = directory
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if CACHE_DIRECTORIES.contains(&name) {
                report.find(Finding::in_file(
                    relative(&ctx.root, &directory),
                    "developer cache directory present in the repository tree",
                    "delete the cache and keep it covered by .gitignore",
                ));
            }
            if OUTPUT_DIRECTORIES.contains(&name) {
                report.find(Finding::in_file(
                    relative(&ctx.root, &directory),
                    "build-output directory present in the repository tree",
                    "delete the output directory and keep it covered by .gitignore",
                ));
            }
        }

        for file in walk_files(&ctx.root) {
            let relative_path = relative(&ctx.root, &file);
            let text = relative_path.to_string_lossy();
            if text.contains("/tests/corpus/") || text.contains("/tests/fixtures/") {
                continue;
            }
            let forbidden = FORBIDDEN_SUFFIXES
                .iter()
                .any(|suffix| text.ends_with(suffix))
                || file
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| FORBIDDEN_EXTENSIONS.contains(&extension));
            if forbidden {
                report.find(Finding::in_file(
                    relative_path,
                    "binary or backup artifact present in the repository tree",
                    "delete the artifact; a source tree carries sources",
                ));
            }
        }

        let mut manifests: Vec<PathBuf> = tree.scope(&[".github"], &["yml", "yaml"])?;
        manifests.push(PathBuf::from("vyre-driver-wgpu/Cargo.toml"));
        for file in &manifests {
            let text = tree.read(file)?;
            for (number, line) in scan::numbered(&text) {
                if let Some(escape) = scan::first_of(line, GPU_ESCAPES) {
                    report.find(Finding::at(
                        file.clone(),
                        number,
                        format!("GPU escape hatch `{escape}`"),
                        "assume a device exists; a probe failure is a configuration failure and \
                         is reported loudly rather than compiled away",
                    ));
                }
            }
        }

        if tree.exists(".github/workflows-paused/gpu-parity.yml") {
            report.find(Finding::in_file(
                ".github/workflows-paused/gpu-parity.yml",
                "the GPU parity workflow is paused",
                "move the workflow back under .github/workflows/ so it runs",
            ));
        }

        for file in tree.all_rust() {
            if SILENT_SKIP_RULE_SOURCES
                .iter()
                .any(|source| file == Path::new(source))
            {
                continue;
            }
            let text = tree.read(&file)?;
            for (number, line) in scan::numbered(&text) {
                if SILENT_SKIPS
                    .iter()
                    .any(|(reason, action)| line.contains(reason) && line.contains(action))
                {
                    report.find(Finding::at(
                        file.clone(),
                        number,
                        format!("silent GPU skip language: {}", line.trim()),
                        "fail loudly when no device is present; a skipped GPU test reads as \
                         coverage and proves nothing",
                    ));
                }
            }
        }

        Ok(report)
    }
}

/// One execution queue, and no committed parallel plan surface.
pub struct SingleBacklog;

impl Gate for SingleBacklog {
    fn name(&self) -> &'static str {
        "single-backlog"
    }

    fn help(&self) -> &'static str {
        "committed parallel execution-plan documents beside the one backlog"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        /// Names that mark a document as a second execution queue.
        const PLAN_MARKERS: &[&str] = &[
            "PLAN",
            "ROADMAP",
            "BACKLOG",
            "STATUS",
            "HANDOFF",
            "TASKS",
            "BUILDOUT",
            "PRD",
            "BRIEF",
            "TRAJECTORY",
            "SEGMENTATION",
            "GENERALIZATION",
        ];
        const FOUR_COLUMN: &str = "| number | affected files | problem | acceptance criteria |";
        const SEVEN_COLUMN: &str =
            "| ID | Axis | Local evidence | Research basis | Work | Proof gate | Dedup seam |";

        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        if !tree.exists("CHANGELOG.md") {
            report.find(Finding::in_file(
                "CHANGELOG.md",
                "the changelog is missing",
                "restore CHANGELOG.md; it is the one generated release artifact",
            ));
        }

        // The queue is a gitignored local file, so its absence is not a
        // violation: a clean checkout legitimately has none. The rule is an
        // upper bound over tracked files, because an untracked local plan
        // confuses nobody.
        if tree.exists("BACKLOG.md") {
            let text = tree.read("BACKLOG.md")?;
            let lowered = text.to_lowercase();
            if !lowered.contains(FOUR_COLUMN) {
                report.find(Finding::in_file(
                    "BACKLOG.md",
                    "the backlog does not use the four-column contract",
                    "give the table the columns number, affected files, problem, \
                     acceptance criteria",
                ));
            }
            if text.contains(SEVEN_COLUMN) {
                report.find(Finding::in_file(
                    "BACKLOG.md",
                    "the backlog still carries the superseded seven-column plan table",
                    "migrate the rows into the four-column contract and delete the old table",
                ));
            }
        }

        for path in tree.paths() {
            if path.extension().and_then(|value| value.to_str()) != Some("md") {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if PLAN_MARKERS.iter().any(|marker| name.contains(marker)) {
                report.find(Finding::in_file(
                    path.clone(),
                    "committed parallel execution-plan surface",
                    "migrate the rows into the backlog and delete the document; \
                     published documents are contracts, procedures or evidence",
                ));
            }
        }

        Ok(report)
    }
}

/// Directories in the tree, with build and metadata roots pruned.
fn walk_directories(root: &Path) -> Vec<PathBuf> {
    walk(root, true)
}

/// Files in the tree, with build and metadata roots pruned.
fn walk_files(root: &Path) -> Vec<PathBuf> {
    walk(root, false)
}

fn walk(root: &Path, directories: bool) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            !(name == ".git"
                || name == "target"
                || name.starts_with("target-")
                || name.starts_with(".cargo-target"))
        })
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_dir() == directories)
        .map(|entry| entry.path().to_path_buf())
        .collect()
}

fn relative(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the artifact rule is about extensions, and `.tar.gz` is two of them.
    /// A check that only read the final extension would let a source archive
    /// through as a `.gz`.
    #[test]
    fn a_double_extension_archive_is_still_an_artifact() {
        assert!(FORBIDDEN_SUFFIXES
            .iter()
            .any(|suffix| "release/x.tar.gz".ends_with(suffix)));
        assert!(FORBIDDEN_EXTENSIONS.contains(&"zip"));
        assert!(!FORBIDDEN_EXTENSIONS.contains(&"rs"));
    }

    /// WHY: an exemption keyed on a path is dead the moment the rule text moves,
    /// and a dead row reads as a decision while doing nothing. Each exempt file
    /// must exist and must still carry the language the rule forbids, so a row
    /// that stops spelling the rule turns this red instead of quietly widening
    /// the scan by one file.
    #[test]
    fn every_silent_skip_exemption_still_spells_the_rule() {
        let root = structure_gate::workspace_root();
        for source in SILENT_SKIP_RULE_SOURCES {
            let path = root.join(source);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("Fix: {source} is exempt but unreadable: {error}"));
            assert!(
                SILENT_SKIPS
                    .iter()
                    .any(|(reason, action)| text.contains(reason) && text.contains(action)),
                "Fix: {source} is exempt from the silent-skip scan but no longer spells the \
                 rule; delete the row"
            );
        }
    }

    /// WHY: the rule required CLAUDE.md and GEMINI.md in every checkout, and this
    /// repository tracks no instruction file at all, so the gate was red on a clean
    /// tree and in CI for a defect no commit could fix. It stays red for the defect
    /// it exists for: a second instruction file holding policy of its own.
    #[test]
    fn a_redirect_is_required_only_where_policy_is_tracked_and_a_policy_copy_is_always_reported() {
        let judge = |files: &[(&str, &str)]| {
            let (_directory, root) = crate::gates::fixture_checkout::checkout(files);
            let tree = Tree::open(&root).expect("Fix: the fixture checkout must list");
            let mut report = Report::clean();
            let policy_is_tracked = tree.exists("AGENTS.md");
            for redirect in REDIRECTS {
                if !tree.exists(redirect) {
                    if policy_is_tracked {
                        report.find(Finding::in_file(*redirect, "missing", "restore it"));
                    }
                    continue;
                }
                let text = tree
                    .read(redirect)
                    .expect("Fix: the fixture file must read");
                if !text.to_ascii_lowercase().contains("compatibility redirect")
                    || !text.contains("AGENTS.md")
                {
                    report.find(Finding::in_file(*redirect, "holds policy", "redirect it"));
                }
            }
            report.count()
        };

        assert_eq!(
            judge(&[("README.md", "# vyre\n")]),
            0,
            "no tracked policy, no redirect to demand"
        );
        assert_eq!(
            judge(&[("AGENTS.md", "# policy\n")]),
            REDIRECTS.len(),
            "tracked policy with no redirect beside it is the finding"
        );
        assert_eq!(
            judge(&[("CLAUDE.md", "# rules of my own\n")]),
            1,
            "a second policy surface is reported with no AGENTS.md in sight"
        );
        assert_eq!(
            judge(&[(
                "CLAUDE.md",
                "This is a compatibility redirect. Read AGENTS.md.\n"
            )]),
            0,
            "a real redirect is clean on its own"
        );
    }
}
