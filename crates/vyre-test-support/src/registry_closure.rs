//! Source-enumerated registry/coverage closure for one crate.
//!
//! Split out of `lib.rs`: the enumerator, its builder-signature parser, and the
//! `inventory::submit!` corpus scan are one concern and were the larger half of
//! that file. Both entry points are re-exported at the crate root, which stays
//! their one public path.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::read_source_file_bounded;

/// Assert the registry-closure contract for the crate rooted at `crate_dir`.
///
/// Resolve `crate_dir` from the working directory, with
/// [`crate::monorepo::vyre_workspace_root`] joined to the crate's directory name. A
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
    assert_registry_closure_crates(&[crate_dir.as_ref().to_path_buf()], waiver, floor);
}

/// Assert the registry-closure contract across a set of crate directories.
pub fn assert_registry_closure_crates(crate_dirs: &[PathBuf], waiver: &[&str], floor: usize) {
    let crate_name = if crate_dirs.len() == 1 {
        crate_dirs[0]
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<crate>")
            .to_string()
    } else {
        "vyre-libs".to_string()
    };

    let mut src_files = Vec::new();
    let mut test_files = Vec::new();
    for dir in crate_dirs {
        let src = dir.join("src");
        let tests = dir.join("tests");
        collect_rust_files(&src, &mut src_files);
        collect_rust_files(&tests, &mut test_files);
    }

    let mut src_texts: Vec<(&Path, String)> = Vec::with_capacity(src_files.len());
    for path in &src_files {
        let text = read_source_file_bounded(path)
            .unwrap_or_else(|e| panic!("{crate_name} source file {path:?} must be readable: {e}"));
        src_texts.push((path.as_path(), text));
    }

    // A test module written in its own file carries no attribute of its own, so
    // the gate is read where it is written: the `#[cfg(test)] mod name;` in the
    // declaring file. Everything that resolves to is test text, whole.
    let mut gated_paths: BTreeSet<PathBuf> = BTreeSet::new();
    for (path, text) in &src_texts {
        let module_dir = module_directory(path);
        for name in structure_gate::cfg_test::cfg_test_module_declarations(text) {
            gated_paths.insert(module_dir.join(format!("{name}.rs")));
            gated_paths.insert(module_dir.join(&name));
        }
    }

    let mut builders: BTreeSet<String> = BTreeSet::new();
    let mut corpus = String::new();
    for (path, text) in &src_texts {
        if path.ancestors().any(|a| gated_paths.contains(a)) {
            corpus.push_str(text);
            corpus.push('\n');
            continue;
        }
        for name in program_builders_in(text) {
            builders.insert(name);
        }
        for block in inventory_submit_blocks(text) {
            corpus.push_str(&block);
            corpus.push('\n');
        }
        // Only the test-gated items count as in-crate coverage. Taking every
        // byte after the first `#[cfg(test)]` marker counted production code as
        // test text, and a crate whose first marker precedes its re-export list
        // had 174 symbols "covered" by a `pub use` block that merely names them.
        let gated = structure_gate::cfg_test::cfg_test_items(text);
        if !gated.is_empty() {
            corpus.push_str(&gated);
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
        if text.contains("assert_registry_closure(")
            || text.contains("assert_registry_closure_crates(")
        {
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

    let production_files = src_texts
        .iter()
        .filter(|(path, _)| !path.ancestors().any(|a| gated_paths.contains(a)))
        .count();
    assert!(
        production_files > 0,
        "[{crate_name}] the source walk found no production `.rs` file under the crate source roots, so the \
         enumeration proves nothing. Fix: the walk or the crate layout, not this gate. A crate \
         whose only builders are test-gated fixtures reports zero builders and still scans files."
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
/// Public because a source-derived gate outside this crate needs the same walk:
/// a second copy is what a `tests/support` coverage gate reached for, and it
/// duplicated this one exactly.
///
/// # Panics
/// Panics when `dir` exists but cannot be read, or when an entry under it cannot
/// be read. A source-derived gate enumerates this set to decide what it proves,
/// so a skipped file would shrink that set in silence and weaken the gate instead
/// of failing it. A directory that does not exist is not a shrink: a crate with no
/// `tests/` directory contributes no test sources, and that walk yields nothing.
pub fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
        Err(error) => panic!(
            "cannot enumerate `{}`: {error}. Fix: restore read permission on the directory so the gate proves every source file under it.",
            dir.display()
        ),
    };
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| {
                panic!(
                    "cannot read an entry under `{}`: {error}. Fix: restore read permission on that entry so the gate proves every source file under it.",
                    dir.display()
                )
            })
            .path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Directory the submodules declared by one source file live in.
///
/// `mod.rs`, `lib.rs` and `main.rs` declare siblings; every other file declares
/// children in a directory named for it, which is where a `mod tests;` beside
/// `foo.rs` resolves to `foo/tests.rs`.
fn module_directory(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    match path.file_stem().and_then(|s| s.to_str()) {
        Some("mod" | "lib" | "main") | None => parent,
        Some(stem) => parent.join(stem),
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
        if returns_program(after_name) {
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

/// The return-type text of a `fn` signature: everything between the parameter
/// list's closing paren and the body's opening brace, `where` clause included.
///
/// Reading the whole signature instead is what misclassified
/// `vyre_driver::self_optimizer_bench::report_scaling`, which returns `()` and
/// takes a `fn(Program, &dyn SemanticExecutor) -> Program` callback: the arrow
/// in the parameter type was read as the function's own return type.
fn return_type_window(after_name: &str) -> Option<&str> {
    let mut rest = after_name;
    // A generic parameter list carries parens and arrows of its own
    // (`<F: Fn(u32) -> Program>`), so the parameter list does not start at the
    // first `(`. `>` preceded by `-` is an arrow, not a closing bracket.
    if rest.starts_with('<') {
        let mut depth = 0i32;
        let mut previous = ' ';
        let mut end = None;
        for (i, ch) in rest.char_indices() {
            match ch {
                '<' => depth += 1,
                '>' if previous != '-' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
            previous = ch;
        }
        rest = &rest[end? + 1..];
    }
    let open = rest.find('(')?;
    let mut depth = 0i32;
    let mut close = None;
    for (i, ch) in rest[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let tail = &rest[close? + 1..];
    Some(tail.split('{').next().unwrap_or(tail))
}

/// True iff the signature's declared return type is exactly `Program`.
///
/// Anchored at the arrow: a `where` clause bound such as
/// `where F: Fn() -> Program` sits in the same window and must not count, and
/// neither must `-> Result<Program, E>` or `-> Arc<Program>`.
fn returns_program(after_name: &str) -> bool {
    let window = match return_type_window(after_name) {
        Some(window) => window.trim_start(),
        None => return false,
    };
    let Some(after_arrow) = window.strip_prefix("->") else {
        return false;
    };
    after_arrow
        .trim_start()
        .strip_prefix("Program")
        .is_some_and(|rest| {
            rest.chars()
                .next()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
        })
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

    /// WHY: the enumerator used to search the WHOLE signature for `-> Program`,
    /// so a function returning `()` was reported as an unregistered builder as
    /// soon as one of its parameters was a callback returning a `Program`. That
    /// misfire had `vyre-driver`'s closure gate red on a function that builds
    /// nothing. The arrow only counts in return position.
    ///
    /// Not caught: a builder hidden behind a type alias (`-> MyProgram` where
    /// `type MyProgram = Program`). The enumerator reads text, not types.
    #[test]
    fn ignores_program_arrows_outside_return_position() {
        assert!(program_builders_in(
            "pub fn report(dispatch: fn(Program, &dyn D) -> Program) { x }"
        )
        .is_empty());
        assert!(program_builders_in(
            "pub fn report(\n    backend: &str,\n    build: fn(u32) -> Program,\n) {\n    x\n}"
        )
        .is_empty());
        assert!(
            program_builders_in("pub fn run<F: Fn(u32) -> Program>(f: F) -> u32 { 0 }").is_empty()
        );
        assert!(
            program_builders_in("pub fn run<F>(f: F) -> u32 where F: Fn() -> Program { 0 }")
                .is_empty()
        );
    }

    /// The same anchoring must not lose a real builder whose signature carries a
    /// generic list, a `where` clause, or a nested-paren parameter type.
    #[test]
    fn keeps_builder_through_generics_and_where_clauses() {
        assert_eq!(
            program_builders_in("pub fn build<T: Shape>(shape: T) -> Program { p }"),
            vec!["build".to_string()]
        );
        assert_eq!(
            program_builders_in(
                "pub fn build<T>(shape: T) -> Program\nwhere\n    T: Shape,\n{\n    p\n}"
            ),
            vec!["build".to_string()]
        );
        assert_eq!(
            program_builders_in("pub fn build(pairs: Vec<(u32, u32)>) -> Program { p }"),
            vec!["build".to_string()]
        );
        assert_eq!(
            program_builders_in("pub fn build<F: Fn() -> u32>(f: F) -> Program { p }"),
            vec!["build".to_string()]
        );
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
            "inventory::submit! { OperationRegistration::primitive_unconstrained(OP_ID, build, None, None) }",
        );
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("OperationRegistration"));
        assert!(blocks[0].ends_with('}'));
    }
}
