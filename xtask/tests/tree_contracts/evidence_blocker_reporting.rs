//! No xtask command exits on a blocker list without naming the blockers.
//!
//! WHY: `release-evidence` spawned the wrong binary for twelve of its thirteen
//! children, every one answered "not implemented", and the whole batch read as
//! a normal failing gate for a merge cycle. It read that way because a failing
//! evidence command printed its artifact path and exited 1, and so did a
//! command whose gate genuinely failed. An exit code with no cause is the same
//! output for a broken harness and a broken tree.
//!
//! The fix is one owner, `xtask::output_arg::report_evidence_artifact`, which
//! now writes every blocker to stderr before exiting. This gate covers what an
//! owner cannot: a command that grows its own copy of the epilogue. Any
//! function in the xtask crates that decides an exit from something named
//! `blocker` must also put those blockers in front of the reader.
//!
//! The member set is every function in the three xtask crates, parsed at run
//! time, so a command added tomorrow is judged tomorrow.
//!
//! What it does not catch: a function that prints a blocker list unrelated to
//! the one it exits on. It matches names, not dataflow.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

use super::common::workspace_root;

/// Below this many functions the walk has stopped seeing the xtask crates and
/// an empty offender set would mean nothing. The three crates hold thousands.
const MINIMUM_FUNCTIONS_SCANNED: usize = 500;

/// Below this many exit-on-blocker functions the SIGNATURE has stopped matching
/// and the gate would pass by judging nothing.
const MINIMUM_MEMBERS: usize = 2;

/// One function whose exit is decided by a blocker list.
struct ExitOnBlockers {
    function: String,
    reports: bool,
}

#[derive(Default)]
struct Collector {
    scanned: usize,
    members: Vec<ExitOnBlockers>,
}

impl Collector {
    fn push(&mut self, name: String, block: &syn::Block) {
        self.scanned += 1;
        if !exits_on_a_blocker_guard(block) {
            return;
        }
        let mut tokens = block.to_token_stream().to_string();
        tokens.retain(|character| !character.is_whitespace());
        // Either the shared epilogue does the reporting, or this function
        // writes the blockers to stderr itself.
        let reports =
            tokens.contains("report_evidence_artifact") || stderr_calls_mention_blockers(block);
        self.members.push(ExitOnBlockers {
            function: name,
            reports,
        });
    }
}

/// Whether `block` holds an `if` whose condition names a blocker and whose
/// branch exits the process.
///
/// The guard is the signature, not the mere presence of both words. A writer
/// that exits because a directory could not be created, in a function that also
/// builds a blocker list, decides nothing from those blockers and is not what
/// this rule is about.
fn exits_on_a_blocker_guard(block: &syn::Block) -> bool {
    struct Guards {
        found: bool,
    }
    impl<'ast> Visit<'ast> for Guards {
        fn visit_expr_if(&mut self, item: &'ast syn::ExprIf) {
            let mut condition = item.cond.to_token_stream().to_string();
            condition.retain(|character| !character.is_whitespace());
            if condition.contains("blocker") {
                let mut branches = item.then_branch.to_token_stream().to_string();
                if let Some((_, alternative)) = &item.else_branch {
                    branches.push_str(&alternative.to_token_stream().to_string());
                }
                branches.retain(|character| !character.is_whitespace());
                self.found |= branches.contains("exit(");
            }
            syn::visit::visit_expr_if(self, item);
        }
    }
    let mut guards = Guards { found: false };
    guards.visit_block(block);
    guards.found
}

impl<'ast> Visit<'ast> for Collector {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.push(item.sig.ident.to_string(), &item.block);
        syn::visit::visit_block(self, &item.block);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.push(item.sig.ident.to_string(), &item.block);
        syn::visit::visit_block(self, &item.block);
    }
}

/// Whether any `eprintln!`/`eprint!` in `block` names a blocker.
///
/// Checked per macro invocation rather than over the whole function, so a
/// function that prints something unrelated and exits on blockers is still a
/// violation.
fn stderr_calls_mention_blockers(block: &syn::Block) -> bool {
    struct Macros {
        found: bool,
    }
    impl<'ast> Visit<'ast> for Macros {
        fn visit_macro(&mut self, item: &'ast syn::Macro) {
            let name = item
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_default();
            if name != "eprintln" && name != "eprint" {
                return;
            }
            let mut tokens = item.tokens.to_string();
            tokens.retain(|character| !character.is_whitespace());
            self.found |= tokens.contains("blocker");
        }
    }
    let mut macros = Macros { found: false };
    macros.visit_block(block);
    macros.found
}

/// The `src` directory of every xtask crate, read from the workspace manifest.
///
/// Derived rather than listed: the tooling is split across `xtask` and the
/// `xtask-*` crates that link vyre, and which crate a command ended up in is a
/// dependency-weight decision this rule has no stake in.
fn xtask_source_roots(root: &Path) -> Vec<PathBuf> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: the workspace Cargo.toml must be readable.");
    let mut roots = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim().trim_end_matches(',').trim_matches('"');
        if trimmed != "xtask" && !trimmed.starts_with("xtask-") {
            continue;
        }
        let source = root.join(trimmed).join("src");
        if source.is_dir() {
            roots.push(source);
        }
    }
    roots.sort();
    roots.dedup();
    assert!(
        roots.len() >= 3,
        "Fix: only {} xtask source root(s) were derived from the workspace manifest; the walk is wrong, so this gate would judge nothing.",
        roots.len()
    );
    roots
}

fn rust_files(directory: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", directory.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_xtask_command_exits_on_blockers_without_naming_them() {
    let root = workspace_root();
    let mut files = Vec::new();
    for source in xtask_source_roots(&root) {
        rust_files(&source, &mut files);
    }
    files.sort();

    let mut collector = Collector::default();
    for file in &files {
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", file.display()));
        let Ok(parsed) = syn::parse_file(&text) else {
            continue;
        };
        let before = collector.members.len();
        collector.visit_file(&parsed);
        for member in &mut collector.members[before..] {
            member.function = format!(
                "{}::{}",
                file.strip_prefix(&root).unwrap_or(file).display(),
                member.function
            );
        }
    }

    assert!(
        collector.scanned >= MINIMUM_FUNCTIONS_SCANNED,
        "Fix: only {} function(s) were parsed across the xtask crates; the walk is wrong, so this gate would pass by judging nothing.",
        collector.scanned
    );
    assert!(
        collector.members.len() >= MINIMUM_MEMBERS,
        "Fix: only {} exit-on-blocker function(s) were derived; the signature no longer matches, so this gate would pass by judging nothing.",
        collector.members.len()
    );

    let silent: BTreeSet<&str> = collector
        .members
        .iter()
        .filter(|member| !member.reports)
        .map(|member| member.function.as_str())
        .collect();
    assert!(
        silent.is_empty(),
        "Fix: these functions exit on a blocker list without putting it in front of the reader. Route the epilogue through `xtask::output_arg::report_evidence_artifact`, or print each blocker to stderr first; an exit code with no cause reads the same as a broken harness:\n  {}",
        silent.into_iter().collect::<Vec<_>>().join("\n  ")
    );
}
