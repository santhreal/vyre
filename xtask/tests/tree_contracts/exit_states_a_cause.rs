//! No xtask command exits nonzero without saying why.
//!
//! WHY: `release-evidence` spawned the wrong binary for twelve of its thirteen
//! children, every one answered "not implemented", and the whole batch read as
//! a normal failing gate for a merge cycle. It read that way because a failing
//! evidence command printed its artifact path and exited 1, and so did a
//! command whose gate genuinely failed. An exit code with no cause is the same
//! output for a broken harness and a broken tree.
//!
//! This gate used to match one shape of that defect: an `if` whose condition
//! named a blocker and whose branch exited. The gate architecture removed every
//! member of that set, because a gate now returns findings and only the
//! dispatcher exits, so the rule matched nothing and passed by judging nothing.
//! The subject is the exit itself: every nonzero `process::exit` in the xtask
//! crates must be reachable only after the process has written a cause, on
//! either stream. Findings go to stdout and a gate error to stderr, so the rule
//! accepts either.
//!
//! `exit(0)` is a success path, such as `--help`, and states nothing.
//!
//! The member set is every exit site in the three xtask crates, parsed at run
//! time, so a command added tomorrow is judged tomorrow.
//!
//! What it does not catch: a cause that is written but says nothing useful, and
//! an exit whose cause is printed by a callee rather than in an enclosing block.
//! It matches output position, not wording.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

use super::workspace_sources::workspace_root;

/// Below this many exit sites the walk or the shape match has stopped seeing the
/// xtask crates and an empty offender set would mean nothing.
const MINIMUM_EXIT_SITES: usize = 20;

/// Macros that put something in front of a reader.
const OUTPUT_MACROS: [&str; 4] = ["eprintln", "eprint", "println", "print"];

/// The shared epilogue that renders a blocker report before exiting.
const SHARED_EPILOGUE: &str = "report_evidence_artifact";

/// One `process::exit(N)` with `N != 0`.
struct ExitSite {
    /// `path::to/file.rs::function`, for the offender list.
    function: String,
    /// Whether an enclosing block writes a cause first.
    states_a_cause: bool,
}

/// Collects every exit site, remembering which enclosing blocks speak.
#[derive(Default)]
struct Collector {
    /// One entry per enclosing block: whether that block writes any output.
    speaking: Vec<bool>,
    /// One entry per enclosing function, innermost last.
    functions: Vec<String>,
    /// Every exit site found so far.
    sites: Vec<ExitSite>,
}

impl Collector {
    /// Whether any enclosing block writes a cause.
    fn enclosing_speaks(&self) -> bool {
        self.speaking.iter().any(|speaks| *speaks)
    }

    /// The innermost function name, or the file scope.
    fn function(&self) -> String {
        self.functions
            .last()
            .cloned()
            .unwrap_or_else(|| "<file scope>".to_string())
    }
}

impl<'ast> Visit<'ast> for Collector {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        self.speaking.push(writes_output(&block.to_token_stream()));
        syn::visit::visit_block(self, block);
        self.speaking.pop();
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.functions.push(item.sig.ident.to_string());
        syn::visit::visit_item_fn(self, item);
        self.functions.pop();
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.functions.push(item.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, item);
        self.functions.pop();
    }

    fn visit_expr_call(&mut self, item: &'ast syn::ExprCall) {
        if let Some(code) = nonzero_exit_code(item) {
            let _ = code;
            self.sites.push(ExitSite {
                function: self.function(),
                states_a_cause: self.enclosing_speaks(),
            });
        }
        syn::visit::visit_expr_call(self, item);
    }
}

/// The status of a `process::exit(N)` call with a nonzero literal `N`.
///
/// The path is matched on its last segment so `exit`, `process::exit` and
/// `std::process::exit` are one shape, and a nonliteral argument is treated as
/// nonzero: a computed status is never a success path.
fn nonzero_exit_code(call: &syn::ExprCall) -> Option<i64> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.path.segments.last()?.ident != "exit" {
        return None;
    }
    let argument = call.args.first()?;
    let literal = match argument {
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(value) => value.base10_parse::<i64>().ok(),
            _ => None,
        },
        _ => None,
    };
    match literal {
        Some(0) => None,
        Some(code) => Some(code),
        None => Some(1),
    }
}

/// Whether `tokens` invoke a macro that writes to stdout or stderr, or the
/// shared blocker epilogue.
fn writes_output(tokens: &proc_macro2::TokenStream) -> bool {
    let text = tokens.to_string();
    if text.contains(SHARED_EPILOGUE) {
        return true;
    }
    OUTPUT_MACROS
        .iter()
        .any(|macro_name| text.contains(&format!("{macro_name} !")))
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

/// Every `.rs` file under `directory`, recursively.
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
fn no_xtask_command_exits_nonzero_without_stating_a_cause() {
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
        let before = collector.sites.len();
        collector.visit_file(&parsed);
        for site in &mut collector.sites[before..] {
            site.function = format!(
                "{}::{}",
                file.strip_prefix(&root).unwrap_or(file).display(),
                site.function
            );
        }
    }

    assert!(
        collector.sites.len() >= MINIMUM_EXIT_SITES,
        "Fix: only {} nonzero exit site(s) were derived across the xtask crates; the walk or the shape match is wrong, so this gate would pass by judging nothing.",
        collector.sites.len()
    );

    let silent: BTreeSet<&str> = collector
        .sites
        .iter()
        .filter(|site| !site.states_a_cause)
        .map(|site| site.function.as_str())
        .collect();
    assert!(
        silent.is_empty(),
        "Fix: these functions exit nonzero without writing a cause first. Print the finding or the error, or route the epilogue through `xtask::output_arg::report_evidence_artifact`; an exit code with no cause reads the same as a broken harness:\n  {}",
        silent.into_iter().collect::<Vec<_>>().join("\n  ")
    );
}
