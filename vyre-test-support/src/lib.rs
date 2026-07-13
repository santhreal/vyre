//! Shared test-only harness helpers for the vyre workspace.
//!
//! # The registry/coverage CLOSURE gate. ONE definitional home
//!
//! Every vyre crate that ships `pub fn ... -> Program` builders needs the same contract:
//! each builder must be reachable from its `inventory::submit!` registry OR pinned by a
//! parity/behavioral test, otherwise a builder can rot, diverge from its GPU/reference
//! arm, or silently lose coverage with nothing red. Historically each crate shipped its
//! own ~230-line copy of the enumerator (and 22 crates shipped a *tautology stub* whose doc
//! claimed adversarial closure coverage while asserting `bytes[0] == bytes[0]`: a Law 6 /
//! Law 9 evasion). That duplication is itself a ONE-PLACE violation: 26 copies drift.
//!
//! [`assert_registry_closure`] is the single canonical enumerator. Each crate's
//! `tests/adversarial_registry_closure.rs` becomes a thin wrapper:
//!
//! ```ignore
//! const COVERAGE_WAIVER: &[&str] = &[ /* builder, reason */ ];
//! #[test]
//! fn every_program_builder_is_tested_registered_or_explicitly_waived() {
//!     vyre_test_support::assert_registry_closure(env!("CARGO_MANIFEST_DIR"), COVERAGE_WAIVER, 4);
//! }
//! ```
//!
//! The enumeration is **feature-independent** (it reads source files as TEXT, never compiling
//! them), so the gate is green under any feature set, it matches CI regardless of which
//! `--features` the runner selects. See `BACKLOG.md` WIRING-tautology-closure-25crates.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Assert the registry-closure contract for the crate rooted at `manifest_dir`.
///
/// Source-enumerates every `pub fn NAME(...) -> Program` builder under `<manifest_dir>/src`,
/// EXCLUDING `impl`-block methods (`&self` receiver) and IR-transform passes (first parameter
/// is `Program`/`&Program`/`&mut Program`: a pass rewrites an existing Program rather than
/// constructing one from source inputs, so it submits no `OpEntry` and is exercised by
/// optimizer/pass tests, not the source-builder registry contract).
///
/// A builder is COVERED iff its name appears (word-boundary) in (a) an `inventory::submit!`
/// block, (b) any file under `<manifest_dir>/tests` (except the closure gate itself), or
/// (c) an inline `#[cfg(test)]` / `#[test]` / `mod tests` region of a source file.
///
/// Every UNCOVERED builder must be listed in `waiver` with a trailing `//` reason. Three
/// guards keep the waiver honest and only-shrinkable:
/// * **stale**: a waiver entry that is no longer a builder (renamed/removed/now a transform);
/// * **now-covered**: a waiver entry that has since gained a test/registry footprint;
/// * **unwaived**: an uncovered builder missing from the waiver (the real finding to fix).
///
/// `floor` is the minimum expected builder count; it fails loudly if the source enumeration
/// silently breaks (e.g. a parser regression that finds zero builders).
///
/// # Panics
/// Panics (i.e. fails the test) on any guard violation, or if a source/test file is unreadable.
pub fn assert_registry_closure(manifest_dir: &str, waiver: &[&str], floor: usize) {
    let crate_name = Path::new(manifest_dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<crate>");
    let src = Path::new(manifest_dir).join("src");
    let tests = Path::new(manifest_dir).join("tests");

    let mut src_files = Vec::new();
    collect_rust_files(&src, &mut src_files);
    let mut test_files = Vec::new();
    collect_rust_files(&tests, &mut test_files);

    let mut builders: BTreeSet<String> = BTreeSet::new();
    let mut corpus = String::new();
    for path in &src_files {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{crate_name} source file {path:?} must be readable: {e}"));
        for name in program_builders_in(&text) {
            builders.insert(name);
        }
        for block in inventory_submit_blocks(&text) {
            corpus.push_str(&block);
            corpus.push('\n');
        }
        if let Some(pos) = ["#[cfg(test)]", "#[test]", "mod tests"]
            .iter()
            .filter_map(|marker| text.find(marker))
            .min()
        {
            corpus.push_str(&text[pos..]);
            corpus.push('\n');
        }
    }
    for path in &test_files {
        if path.file_name().and_then(|n| n.to_str()) == Some("adversarial_registry_closure.rs") {
            continue;
        }
        corpus.push_str(
            &fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("{crate_name} test file {path:?} must be readable: {e}")),
        );
        corpus.push('\n');
    }

    let uncovered: BTreeSet<String> = builders
        .iter()
        .filter(|b| !corpus_contains_word(&corpus, b))
        .cloned()
        .collect();

    eprintln!(
        "[{crate_name}] registry closure: {} public `pub fn -> Program` builders enumerated, {} uncovered",
        builders.len(),
        uncovered.len()
    );
    let waiver_set: BTreeSet<String> = waiver.iter().map(|s| (*s).to_string()).collect();

    let stale: BTreeSet<&String> = waiver_set.iter().filter(|w| !builders.contains(*w)).collect();
    let now_covered: BTreeSet<&String> =
        waiver_set.iter().filter(|w| !uncovered.contains(*w)).collect();
    let unwaived: BTreeSet<&String> = uncovered.difference(&waiver_set).collect();

    if !stale.is_empty() || !now_covered.is_empty() || !unwaived.is_empty() {
        eprintln!("== [{crate_name}] registry closure diagnostic ==");
        eprintln!("builders={} uncovered={}", builders.len(), uncovered.len());
        eprintln!("UNCOVERED (ground truth for the waiver): {uncovered:?}");
        eprintln!("STALE waiver (not a builder): {stale:?}");
        eprintln!("NOW-COVERED waiver (remove): {now_covered:?}");
        eprintln!("UNWAIVED (untested+unregistered, must fix): {unwaived:?}");
    }
    assert!(
        stale.is_empty(),
        "[{crate_name}] COVERAGE_WAIVER has stale entries (no such `pub fn -> Program` builder. \
         renamed, removed, or now a transform pass): {stale:?}. Fix: remove them."
    );
    assert!(
        now_covered.is_empty(),
        "[{crate_name}] these builders are now COVERED but still in COVERAGE_WAIVER: {now_covered:?}. \
         Fix: remove them (the waiver must shrink)."
    );
    assert!(
        unwaived.is_empty(),
        "[{crate_name}] {} Program builder(s) have NO parity/behavioral test AND are NOT registered \
         in the inventory: {unwaived:?}. Fix: add a reference_eval parity test, register an OpEntry, \
         or add to COVERAGE_WAIVER with a reason. See BACKLOG.md WIRING-tautology-closure-25crates.",
        unwaived.len()
    );

    assert!(
        builders.len() >= floor,
        "[{crate_name}] expected >= {floor} source `pub fn -> Program` builders (excluding `&self` \
         methods and IR-transform passes), found {}, the source enumeration is broken.",
        builders.len()
    );
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("source entry must be readable").path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Names of every `pub fn NAME(...) -> Program` whose return type is exactly `Program`,
/// excluding `&self` methods and IR-transform passes (see [`assert_registry_closure`]).
fn program_builders_in(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut search = 0;
    while let Some(rel) = text[search..].find("fn ") {
        let pos = search + rel;
        search = pos + 3;
        let before = text[..pos].trim_end();
        if !before.ends_with("pub") {
            continue;
        }
        let rest = &text[pos + 3..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let after_name = &rest[name.len()..];
        let window = after_name.split('{').next().unwrap_or("");
        // Skip `impl`-block methods (`pub fn m(&self, ...) -> Program`).
        if takes_self_receiver(after_name) {
            continue;
        }
        // Skip IR-transform passes (`pub fn pass(program: Program, ...) -> Program` /
        // `pub fn pass(&Program) -> Program`): a pass rewrites an existing Program, it does
        // not CONSTRUCT one from source inputs, so it submits no OpEntry and is exercised by
        // optimizer/pass tests, not the source-builder registry contract.
        if first_param_is_program(after_name) {
            continue;
        }
        if returns_program(window) {
            names.push(name);
        }
    }
    names
}

/// True iff the parameter list beginning in `after_name` has a `self` receiver.
fn takes_self_receiver(after_name: &str) -> bool {
    let Some(open) = after_name.find('(') else {
        return false;
    };
    let mut s = after_name[open + 1..].trim_start();
    if let Some(rest) = s.strip_prefix('&') {
        s = rest.trim_start();
        if s.starts_with('\'') {
            s = s[1..]
                .trim_start_matches(|c: char| c.is_alphanumeric() || c == '_')
                .trim_start();
        }
    }
    if let Some(rest) = s.strip_prefix("mut ") {
        s = rest.trim_start();
    }
    if let Some(after_self) = s.strip_prefix("self") {
        matches!(
            after_self.chars().next(),
            None | Some(',') | Some(')') | Some(':') | Some(' ') | Some('\n') | Some('\r')
        )
    } else {
        false
    }
}

/// True iff the FIRST parameter's declared type is `Program` / `&Program` / `&mut Program`
/// (a signal that this `pub fn` is an IR-transform pass, not a source builder).
fn first_param_is_program(after_name: &str) -> bool {
    let Some(open) = after_name.find('(') else {
        return false;
    };
    let params = &after_name[open + 1..];
    let first = params.split(|c| c == ',' || c == ')').next().unwrap_or("");
    let Some(colon) = first.find(':') else {
        return false;
    };
    let mut ty = first[colon + 1..].trim_start();
    ty = ty.strip_prefix('&').unwrap_or(ty).trim_start();
    if ty.starts_with('\'') {
        ty = ty[1..]
            .trim_start_matches(|c: char| c.is_alphanumeric() || c == '_')
            .trim_start();
    }
    ty = ty.strip_prefix("mut ").unwrap_or(ty).trim_start();
    ty.strip_prefix("Program").is_some_and(|rest| {
        rest.chars()
            .next()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
    })
}

fn returns_program(window: &str) -> bool {
    for arrow in ["-> Program", "->Program"] {
        if let Some(i) = window.find(arrow) {
            let next = window[i + arrow.len()..].chars().next();
            match next {
                None => return true,
                Some(c) if !(c.is_alphanumeric() || c == '_') => return true,
                _ => {}
            }
        }
    }
    false
}

/// Extract the brace-balanced body of every `inventory::submit! { ... }` block.
fn inventory_submit_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut search = 0;
    while let Some(rel) = text[search..].find("inventory::submit!") {
        let start = search + rel;
        let Some(brace_rel) = text[start..].find('{') else {
            break;
        };
        let open = start + brace_rel;
        let mut depth = 0i32;
        let mut end = open;
        for (i, ch) in text[open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        blocks.push(text[open..=end].to_string());
        search = end + 1;
    }
    blocks
}

/// True iff `name` appears in `corpus` bounded by non-identifier characters.
fn corpus_contains_word(corpus: &str, name: &str) -> bool {
    let bytes = corpus.as_bytes();
    let nb = name.as_bytes();
    let mut i = 0;
    while let Some(rel) = corpus[i..].find(name) {
        let pos = i + rel;
        let before_ok = pos == 0 || !is_ident_byte(bytes[pos - 1]);
        let after = pos + nb.len();
        let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            return true;
        }
        i = pos + 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_plain_program_builder() {
        assert_eq!(
            program_builders_in("pub fn make_thing(n: u32) -> Program { todo }"),
            vec!["make_thing".to_string()]
        );
    }

    #[test]
    fn excludes_self_methods() {
        assert!(program_builders_in("pub fn build(&self, n: u32) -> Program { x }").is_empty());
        assert!(program_builders_in("pub fn build(&self) -> Program { x }").is_empty());
        assert!(
            program_builders_in("pub fn build(&'a mut self, n: u32) -> Program { x }").is_empty()
        );
    }

    #[test]
    fn excludes_transform_passes() {
        assert!(program_builders_in("pub fn cse(program: Program) -> Program { p }").is_empty());
        assert!(program_builders_in("pub fn opt(p: &Program) -> Program { p }").is_empty());
        assert!(
            program_builders_in("pub fn run(p: &'a mut Program, x: u32) -> Program { p }")
                .is_empty()
        );
    }

    #[test]
    fn keeps_builder_with_non_program_first_param() {
        assert_eq!(
            program_builders_in("pub fn lower(ast: &Ast, cfg: Config) -> Program { p }"),
            vec!["lower".to_string()]
        );
    }

    #[test]
    fn requires_exact_program_return() {
        assert!(program_builders_in("pub fn f(n: u32) -> ProgramGraph { x }").is_empty());
        assert!(program_builders_in("pub fn f(n: u32) -> Result<Program> { x }").is_empty());
    }

    #[test]
    fn non_pub_is_ignored() {
        assert!(program_builders_in("fn f(n: u32) -> Program { x }").is_empty());
    }

    #[test]
    fn word_boundary_matching() {
        assert!(corpus_contains_word("register(make_thing);", "make_thing"));
        assert!(!corpus_contains_word("register(make_thing_ext);", "make_thing"));
        assert!(!corpus_contains_word("premake_thing", "make_thing"));
    }

    #[test]
    fn inventory_block_is_balanced() {
        let blocks = inventory_submit_blocks("inventory::submit! { OpEntry { a: b(), c: {1} } }");
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("OpEntry"));
        assert!(blocks[0].ends_with('}'));
    }
}
