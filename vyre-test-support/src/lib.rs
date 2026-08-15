//! Shared test-only harness helpers for the vyre workspace.
//!
//! # The registry/coverage closure gate, one definitional home
//!
//! Every vyre crate that ships `pub fn ... -> Program` builders owes the same
//! contract: each builder is reachable from that crate's `inventory::submit!`
//! registry, or it is pinned by a parity/behavioral test. A builder that is
//! neither still compiles, still appears in the catalogs generated from source,
//! and still diverges from its reference arm with nothing red.
//!
//! [`assert_registry_closure`] is the one enumerator, and
//! [`registry_closure_gate!`] is how a crate declares its gate. The crate's
//! `tests/registry_closure.rs` carries only what is crate-specific, the floor
//! and the waived builders, because the test name, the manifest-directory
//! argument and the call itself are the same in every crate and were being
//! copied verbatim:
//!
//! ```ignore
//! vyre_test_support::registry_closure_gate! {
//!     floor: 4,
//!     waiver: ["uncovered_builder_with_its_reason_above"],
//! }
//! ```
//!
//! The candidate set is derived from the crate's tree on each run rather than
//! listed in the caller, so a builder added tomorrow is judged tomorrow. That
//! derivation's failure mode is finding nothing: zero builders are trivially
//! all covered, so `BUILDER_FLOOR` is what makes a broken scan fail instead of
//! reporting a clean sweep of a nearly empty set.
//!
//! The enumeration is feature-independent: it reads source files as TEXT and
//! never compiles them, so it reports the same builder set whichever features
//! the runner selects.
#![forbid(unsafe_code)]

/// Declare this crate's registry/coverage closure gate.
///
/// `floor` is the minimum builder count the source enumeration must find, and
/// `waiver` lists builders that are knowingly uncovered. Both are the only
/// crate-specific parts of the gate, so they are the only arguments; the test
/// name and the crate directory are derived here. The directory is the run-time
/// checkout root joined to `CARGO_PKG_NAME`, which expands at the call site and
/// so names the crate that declares the gate. A compiled-in manifest directory
/// would name whichever checkout built the binary, and every checkout here
/// shares one target directory.
#[macro_export]
macro_rules! registry_closure_gate {
    (floor: $floor:expr, waiver: [$($waived:expr),* $(,)?] $(,)?) => {
        #[test]
        fn every_program_builder_is_tested_registered_or_explicitly_waived() {
            $crate::assert_registry_closure(
                $crate::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")),
                &[$($waived),*],
                $floor,
            );
        }
    };
}

/// Declare a test-only `Expr::Opaque` payload type.
///
/// An extension payload is six trait methods of which five are the same in
/// every test that needs one: report `Ok(())` from validation, hand back
/// `self` for downcasting, and answer the two identity questions from a
/// literal. Only the kind string, the debug identity, the result type, the
/// CSE answer and the fingerprint byte differ, so those are the arguments.
///
/// A test that needs a payload with reachable structure, a wire body, or a
/// validation failure writes the impl out: this macro is for the inert leaf.
///
/// `ExprNode` and `DataType` are named unqualified, so the caller must have
/// both in scope. `vyre-foundation` implements these traits on its own types
/// from inside itself, where a path through this crate's dependency on it
/// names a different crate instance and does not compile.
#[macro_export]
macro_rules! test_expr_extension {
    (
        $name:ident,
        kind: $kind:expr,
        identity: $identity:expr,
        result_type: $result_type:expr,
        cse_safe: $cse_safe:expr,
        fingerprint: $fingerprint:expr $(,)?
    ) => {
        #[derive(Debug)]
        struct $name;

        impl ExprNode for $name {
            fn extension_kind(&self) -> &'static str {
                $kind
            }
            fn debug_identity(&self) -> &str {
                $identity
            }
            fn result_type(&self) -> Option<DataType> {
                $result_type
            }
            fn cse_safe(&self) -> bool {
                $cse_safe
            }
            fn stable_fingerprint(&self) -> [u8; 32] {
                [$fingerprint; 32]
            }
            fn validate_extension(&self) -> ::core::result::Result<(), ::std::string::String> {
                Ok(())
            }
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
        }
    };
}

/// Declare a test-only `Node::Opaque` payload type.
///
/// The statement form of [`test_expr_extension!`]: a statement extension has
/// no result type and no CSE answer, so only the kind string, the debug
/// identity and the fingerprint byte differ between tests. `NodeExtension` is
/// named unqualified, so the caller must have it in scope.
#[macro_export]
macro_rules! test_node_extension {
    (
        $name:ident,
        kind: $kind:expr,
        identity: $identity:expr,
        fingerprint: $fingerprint:expr $(,)?
    ) => {
        #[derive(Debug)]
        struct $name;

        impl NodeExtension for $name {
            fn extension_kind(&self) -> &'static str {
                $kind
            }
            fn debug_identity(&self) -> &str {
                $identity
            }
            fn stable_fingerprint(&self) -> [u8; 32] {
                [$fingerprint; 32]
            }
            fn validate_extension(&self) -> ::core::result::Result<(), ::std::string::String> {
                Ok(())
            }
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
        }
    };
}

#[cfg(feature = "ir-fixtures")]
pub mod binop_parity;
pub mod case_table;
pub mod cast_parity;
pub mod consumer_boundary;
pub mod exploded_ifds_cases;
#[cfg(feature = "ir-fixtures")]
pub mod ir_variants;
pub mod monorepo;
#[cfg(feature = "ir-fixtures")]
pub mod pass_programs;

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Per-file read cap for the source enumeration.
///
/// The enumerator reads every `.rs` file under `src/` and `tests/` as text. An
/// unbounded `read_to_string` would let one pathological generated file exhaust
/// memory during a test run, so each file is capped and an over-cap file is a
/// loud failure rather than a silent truncation (a truncated file would drop
/// builders from the enumeration and quietly weaken the closure gate).
const MAX_SOURCE_FILE_BYTES: u64 = 4_194_304;

/// Read one source file as text, bounded by [`MAX_SOURCE_FILE_BYTES`].
pub(crate) fn read_source_file_bounded(path: &Path) -> std::io::Result<String> {
    read_source_file_with_cap(path, MAX_SOURCE_FILE_BYTES)
}

fn read_source_file_with_cap(path: &Path, max_bytes: u64) -> std::io::Result<String> {
    let mut text = String::new();
    fs::File::open(path)?
        .take(max_bytes.saturating_add(1))
        .read_to_string(&mut text)?;
    if text.len() as u64 > max_bytes {
        return Err(std::io::Error::other(format!(
            "{} exceeds the {max_bytes} byte source read cap; truncating it \
             would silently drop builders from the closure enumeration. Fix: split the \
             file or raise MAX_SOURCE_FILE_BYTES deliberately.",
            path.display()
        )));
    }
    Ok(text)
}

/// Assert the registry-closure contract for the crate rooted at `crate_dir`.
///
/// Resolve `crate_dir` from the working directory, with
/// [`monorepo::vyre_workspace_root`] joined to the crate's directory name. A
/// compiled-in `CARGO_MANIFEST_DIR` answers for whichever checkout built the
/// test binary, which is not the checkout the command ran in whenever a target
/// directory is shared.
///
/// Source-enumerates every `pub fn NAME(...) -> Program` builder under `<crate_dir>/src`,
/// EXCLUDING `impl`-block methods (`&self` receiver) and IR-transform passes (first parameter
/// is `Program`/`&Program`/`&mut Program`: a pass rewrites an existing Program rather than
/// constructing one from source inputs, so it submits no `OperationRegistration` and is
/// exercised by optimizer/pass tests, not the source-builder registry contract).
///
/// A builder is COVERED iff its name appears (word-boundary) in (a) an `inventory::submit!`
/// block, (b) any file under `<crate_dir>/tests` (except the closure gate itself), or
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

pub fn assert_registry_closure(crate_dir: impl AsRef<Path>, waiver: &[&str], floor: usize) {
    let crate_dir = crate_dir.as_ref();
    let crate_name = crate_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<crate>");
    let src = crate_dir.join("src");
    let tests = crate_dir.join("tests");

    let mut src_files = Vec::new();
    collect_rust_files(&src, &mut src_files);
    let mut test_files = Vec::new();
    collect_rust_files(&tests, &mut test_files);

    let mut builders: BTreeSet<String> = BTreeSet::new();
    let mut corpus = String::new();
    for path in &src_files {
        let text = read_source_file_bounded(path)
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
        let text = read_source_file_bounded(path)
            .unwrap_or_else(|e| panic!("{crate_name} test file {path:?} must be readable: {e}"));
        // The gate's own caller is not coverage. Excluding it by name would
        // only exclude one spelling of the file; excluding every file that
        // calls the enumerator keeps a waiver entry from covering itself
        // whatever the caller is named.
        if text.contains("assert_registry_closure(") {
            continue;
        }
        corpus.push_str(&text);
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

    let stale: BTreeSet<&String> = waiver_set
        .iter()
        .filter(|w| !builders.contains(*w))
        .collect();
    let now_covered: BTreeSet<&String> = waiver_set
        .iter()
        .filter(|w| !uncovered.contains(*w))
        .collect();
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
         in the inventory: {unwaived:?}. Fix: add a reference_eval parity test, submit an OperationRegistration, \
         or add to COVERAGE_WAIVER with the reason it is permanently uncoverable.",
        unwaived.len()
    );

    assert!(
        builders.len() >= floor,
        "[{crate_name}] expected >= {floor} source `pub fn -> Program` builders (excluding `&self` \
         methods and IR-transform passes), found {}, the source enumeration is broken.",
        builders.len()
    );
}

/// Collect every `.rs` file under `dir` into `out`.
///
/// # Panics
/// Panics when a directory entry is unreadable. The closure gate enumerates source
/// text, so skipping an unreadable file would quietly shrink the builder set and
/// weaken the gate.
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
        // not CONSTRUCT one from source inputs, so it submits no OperationRegistration and
        // is exercised by optimizer/pass tests, not the source-builder registry contract.
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
    let first = params.split([',', ')']).next().unwrap_or("");
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
        assert!(!corpus_contains_word(
            "register(make_thing_ext);",
            "make_thing"
        ));
        assert!(!corpus_contains_word("premake_thing", "make_thing"));
    }

    #[test]
    fn inventory_block_is_balanced() {
        let blocks = inventory_submit_blocks(
            "inventory::submit! { OperationRegistration::primitive(OP_ID, build, None, None) }",
        );
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("OperationRegistration"));
        assert!(blocks[0].ends_with('}'));
    }

    /// Source readers accept input exactly at the configured cap.
    #[test]
    fn bounded_source_reader_accepts_exact_cap() {
        let dir = tempfile::tempdir().expect("Fix: source-reader fixture directory must exist");
        let path = dir.path().join("source.rs");
        fs::write(&path, "12345678").expect("Fix: source-reader fixture must be writable");

        assert_eq!(
            read_source_file_with_cap(&path, 8).expect("Fix: exact-cap input must be readable"),
            "12345678"
        );
    }

    /// Source readers reject oversized input rather than silently truncating coverage.
    #[test]
    fn bounded_source_reader_rejects_oversized_input() {
        let dir = tempfile::tempdir().expect("Fix: source-reader fixture directory must exist");
        let path = dir.path().join("source.rs");
        fs::write(&path, "123456789").expect("Fix: source-reader fixture must be writable");

        let error = read_source_file_with_cap(&path, 8)
            .expect_err("oversized source evidence must fail closed");
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert!(error
            .to_string()
            .contains("exceeds the 8 byte source read cap"));
    }

    /// Missing source evidence preserves the filesystem error for the calling contract.
    #[test]
    fn bounded_source_reader_reports_missing_files() {
        let dir = tempfile::tempdir().expect("Fix: source-reader fixture directory must exist");
        let error = read_source_file_with_cap(&dir.path().join("missing.rs"), 8)
            .expect_err("missing source evidence must fail");

        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}
