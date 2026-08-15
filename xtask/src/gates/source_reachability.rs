//! Whether every tracked Rust file is compiled by something, and parses.
//!
//! Two halves of one class. A file no cargo target reaches is never compiled, so
//! it can hold anything and every build stays green:
//! `vyre-libs/src/visual/glyph_grid/mod.rs` shipped an op registration and eight
//! contracts that no `mod` declaration named, while three generated documents
//! listed the op as supported. Reachability answers whether a file is compiled.
//! Parsing answers the other half, because the answer for an orphan and the
//! answer for a scaffolding template are both "no target", and one of those is
//! allowed to not be Rust.
//!
//! Neither gate runs cargo. `cargo build` cannot see an orphan at all: a file
//! outside every target is exactly what a green build produces.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Edition the parser reads files under.
const EDITION: &str = "2021";

/// Files per rustfmt invocation. A batch is one process, and a batch that
/// reports an error is re-run per file so the finding names the file.
const BATCH: usize = 200;

/// Every tracked `.rs` file is reached from a declared cargo target.
pub struct SourceReachability;

impl Gate for SourceReachability {
    fn name(&self) -> &'static str {
        "source-reachability"
    }

    fn help(&self) -> &'static str {
        "every tracked Rust file is compiled by a declared cargo target"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let sources: BTreeSet<String> = tree
            .paths()
            .iter()
            .filter(|path| extension_is(path, "rs"))
            .map(as_key)
            .collect();
        // A `#[path]` may name Rust that is not called `.rs`: vyre-aot compiles a
        // shipped template directly, so module resolution runs against every
        // tracked file rather than only the Rust-suffixed ones.
        let everything: BTreeSet<String> = tree.paths().iter().map(as_key).collect();
        let manifests: Vec<String> = tree
            .paths()
            .iter()
            .filter(|path| file_name(path) == Some("Cargo.toml"))
            .map(as_key)
            .collect();
        let templates: Vec<String> = tree
            .paths()
            .iter()
            .filter(|path| {
                file_name(path).is_some_and(|name| {
                    name.starts_with("Cargo.toml.") && name.len() > "Cargo.toml.".len()
                })
            })
            .map(as_key)
            .collect();
        if sources.is_empty() {
            return Err(GateError::new(
                "no tracked .rs file found",
                "run this gate inside the workspace checkout; a reachability scan over an \
                 empty source set reports success forever",
            ));
        }
        if manifests.is_empty() {
            return Err(GateError::new(
                "no tracked Cargo.toml found",
                "run this gate inside the workspace checkout; target roots are derived \
                 from the manifests",
            ));
        }

        let mut roots: Vec<(String, String)> = Vec::new();
        for entry in &manifests {
            let table = match tree.read_toml(entry) {
                Ok(table) => table,
                Err(error) => {
                    report.find(Finding::in_file(
                        entry,
                        format!("manifest is not readable as TOML: {}", error.message),
                        "repair the manifest; its targets are unknown while it does not parse",
                    ));
                    continue;
                }
            };
            collect_roots(entry, &table, &sources, &tree, &mut roots, &mut report);
        }
        if roots.is_empty() {
            return Err(GateError::new(
                "no cargo target root resolved",
                "check that the manifests declare targets; a reachability scan with no \
                 roots reports every file orphaned or nothing at all",
            ));
        }

        let mut reached: BTreeMap<String, String> = BTreeMap::new();
        let mut queue: Vec<(String, String, String)> = Vec::new();
        for (path, label) in &roots {
            if !reached.contains_key(path) {
                reached.insert(path.clone(), label.clone());
                queue.push((path.clone(), parent_of(path), label.clone()));
            }
        }
        let mut missing_mods: Vec<Finding> = Vec::new();
        while let Some((rel, mod_base, label)) = queue.pop() {
            let text = match tree.read(&rel) {
                Ok(text) => text,
                Err(error) => {
                    report.find(Finding::in_file(
                        &rel,
                        format!("file is reached from {label} but unreadable: {}", error.message),
                        "restore the file as UTF-8 text, or delete the declaration reaching it",
                    ));
                    continue;
                }
            };
            let found = scan_source(&text);
            for declaration in &found.mods {
                let mut base = mod_base.clone();
                for part in &declaration.stack {
                    base = join(&base, part);
                }
                let candidates = match &declaration.path_attr {
                    // A `#[path]` on a file-scope module resolves against the
                    // file's own directory; inside an inline `mod` block it
                    // resolves against the enclosing module directory.
                    Some(attr) => {
                        let anchor = if declaration.stack.is_empty() {
                            parent_of(&rel)
                        } else {
                            base.clone()
                        };
                        vec![normalize(&join(&anchor, attr))]
                    }
                    None => vec![
                        normalize(&join(&base, &format!("{}.rs", declaration.name))),
                        normalize(&join(&join(&base, &declaration.name), "mod.rs")),
                    ],
                };
                let hit = candidates
                    .iter()
                    .find(|candidate| everything.contains(*candidate))
                    .cloned();
                let Some(hit) = hit else {
                    missing_mods.push(Finding::in_file(
                        &rel,
                        format!(
                            "declares `mod {};` but none of {} is tracked",
                            declaration.name,
                            candidates.join(", ")
                        ),
                        "add the module file, or delete the declaration",
                    ));
                    continue;
                };
                if !reached.contains_key(&hit) {
                    let trace = format!("{label} -> {rel}");
                    let child_base = if file_name(Path::new(&hit)) == Some("mod.rs") {
                        parent_of(&hit)
                    } else {
                        join(&parent_of(&hit), stem_of(&hit))
                    };
                    reached.insert(hit.clone(), trace.clone());
                    queue.push((hit, child_base, trace));
                }
            }
            for raw in &found.includes {
                let target = normalize(&join(&parent_of(&rel), raw));
                if !everything.contains(&target) || reached.contains_key(&target) {
                    continue;
                }
                let trace = format!("{label} -> {rel} include!");
                reached.insert(target.clone(), trace.clone());
                queue.push((target, mod_base.clone(), trace));
            }
        }

        let mut exempt: BTreeMap<String, String> = BTreeMap::new();
        for entry in &templates {
            let template_root = parent_of(entry);
            let owned: Vec<&String> = sources
                .iter()
                .filter(|path| path.starts_with(&format!("{template_root}/")))
                .collect();
            if owned.is_empty() {
                report.find(Finding::in_file(
                    entry,
                    "template manifest covers no tracked .rs file",
                    "delete the template, or delete this exemption; an exemption matching \
                     nothing reserves an allowance nothing uses",
                ));
                continue;
            }
            for path in owned {
                exempt.insert(
                    path.clone(),
                    format!("scaffolding template `{entry}`, which no package compiles"),
                );
            }
        }

        for path in reached.keys() {
            let Ok(text) = tree.read(path) else {
                continue;
            };
            let found = scan_source(&text);
            for pattern in &found.trybuild {
                let owner = manifests
                    .iter()
                    .map(|entry| parent_of(entry))
                    .find(|directory| {
                        !directory.is_empty() && path.starts_with(&format!("{directory}/"))
                    })
                    .unwrap_or_else(|| parent_of(path));
                let prefix = normalize(&join(&owner, pattern));
                let named: Vec<&String> = if pattern.contains('*') || pattern.contains('?') {
                    sources
                        .iter()
                        .filter(|candidate| glob_match(&prefix, candidate))
                        .collect()
                } else {
                    sources.iter().filter(|candidate| **candidate == prefix).collect()
                };
                if named.is_empty() {
                    report.find(Finding::in_file(
                        path,
                        format!("runs trybuild over `{pattern}`, which matches no tracked file"),
                        "restore the fixture, or delete the case; a trybuild path naming \
                         nothing is a compile-fail test that asserts nothing",
                    ));
                    continue;
                }
                for candidate in named {
                    exempt.insert(
                        candidate.clone(),
                        format!("trybuild fixture compiled at run time by `{path}`"),
                    );
                }
            }
        }

        for path in &sources {
            if reached.contains_key(path) || exempt.contains_key(path) {
                continue;
            }
            report.find(Finding::in_file(
                path,
                "file is compiled by no cargo target",
                "declare it with `mod`, `#[path]`, `include!` or a target entry in the \
                 owning Cargo.toml, or delete it; a file nothing compiles reads as \
                 coverage and provides none",
            ));
        }
        for finding in missing_mods {
            report.find(finding);
        }

        report.note(format!(
            "{} tracked .rs file(s) reachable from {} declared target(s) across {} \
             manifest(s), {} exempt",
            reached.keys().filter(|path| sources.contains(*path)).count(),
            roots.len(),
            manifests.len(),
            exempt.len()
        ));
        Ok(report)
    }
}

/// Every tracked `.rs` file is valid Rust syntax.
pub struct SourceParses;

impl Gate for SourceParses {
    fn name(&self) -> &'static str {
        "source-parses"
    }

    fn help(&self) -> &'static str {
        "every tracked Rust file parses, scaffolding templates excepted"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let sources: Vec<String> = tree
            .paths()
            .iter()
            .filter(|path| extension_is(path, "rs"))
            .map(as_key)
            .collect();
        if sources.is_empty() {
            return Err(GateError::new(
                "no tracked .rs file found",
                "run this gate inside the workspace checkout; a parse scan over an empty \
                 source set reports success forever",
            ));
        }

        // The exemption comes from the tree: a directory holding a tracked
        // template manifest is scaffolding, so its sources are rendered before
        // they are Rust. Nothing is matched on file contents.
        let mut exempt: BTreeMap<String, String> = BTreeMap::new();
        let template_roots: BTreeSet<String> = tree
            .paths()
            .iter()
            .filter(|path| {
                file_name(path).is_some_and(|name| {
                    name.starts_with("Cargo.toml.") && name.len() > "Cargo.toml.".len()
                })
            })
            .map(|path| parent_of(&as_key(&path.clone())))
            .collect();
        for template_root in &template_roots {
            let owned: Vec<&String> = sources
                .iter()
                .filter(|path| path.starts_with(&format!("{template_root}/")))
                .collect();
            if owned.is_empty() {
                report.find(Finding::in_file(
                    template_root,
                    "template root covers no tracked .rs file",
                    "delete the template, or delete this exemption; an exemption matching \
                     nothing reserves an allowance nothing uses",
                ));
            }
            for path in owned {
                exempt.insert(path.clone(), template_root.clone());
            }
        }

        let scanned: Vec<&String> = sources
            .iter()
            .filter(|path| !exempt.contains_key(*path))
            .collect();
        if scanned.is_empty() {
            return Err(GateError::new(
                "every tracked .rs file is exempt",
                "narrow the template roots; a parse scan that reads nothing reports \
                 success forever",
            ));
        }

        // The exemption has to bite. A template whose sources all parse needs no
        // exemption, and slack is where a real parse failure hides.
        let mut parsing_templates: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (path, template_root) in &exempt {
            if parse_errors(&tree, std::slice::from_ref(path))?.is_empty() {
                parsing_templates
                    .entry(template_root.clone())
                    .or_default()
                    .push(path.clone());
            }
        }
        for (template_root, parsing) in &parsing_templates {
            let owned = exempt.values().filter(|owner| *owner == template_root).count();
            if parsing.len() == owned {
                report.find(Finding::in_file(
                    template_root,
                    format!(
                        "every tracked .rs file under this template root parses as Rust: {}",
                        parsing.join(", ")
                    ),
                    "delete the template exemption and scan those files with the rest; an \
                     exemption whose files no longer need it is slack a real parse failure \
                     hides in",
                ));
            }
        }

        let owned: Vec<String> = scanned.iter().map(|path| (*path).clone()).collect();
        for batch in owned.chunks(BATCH) {
            if parse_errors(&tree, batch)?.is_empty() {
                continue;
            }
            // A batch names the offender in a `-->` line. A per-file run names it
            // without reading rustfmt's diagnostic layout.
            for path in batch {
                if let Some(error) = parse_errors(&tree, std::slice::from_ref(path))?
                    .into_iter()
                    .next()
                {
                    report.find(Finding::in_file(
                        path,
                        format!("file is not valid Rust: {error}"),
                        "make the file parse, or move it under a scaffolding template root \
                         if it is not Rust; a tracked .rs file nothing can parse is not code",
                    ));
                }
            }
        }

        report.note(format!(
            "{} tracked .rs file(s) parse under edition {EDITION}, {} exempt across {} \
             template root(s)",
            scanned.len(),
            exempt.len(),
            template_roots.len()
        ));
        Ok(report)
    }
}

/// Every `error` line rustfmt reports for these files.
///
/// rustfmt is the parser because it is the only one the toolchain ships that
/// reads a file without a crate around it. Formatting differences print as
/// `Diff in <path>` and are ignored. Nothing is written: `--check` is read-only
/// and no `--emit` is passed. A missing rustfmt is fatal, because a missing
/// parser and a clean tree must not be the same result.
fn parse_errors(tree: &Tree, paths: &[String]) -> Result<Vec<String>, GateError> {
    let output = Command::new("rustfmt")
        .current_dir(tree.root())
        .args(["--edition", EDITION, "--check", "--color", "never"])
        .args(paths)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot run rustfmt: {error}"),
                "install the rustfmt component with `rustup component add rustfmt`; this \
                 gate parses every tracked Rust file with it, and a missing parser is not \
                 a clean tree",
            )
        })?;
    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    let errors: Vec<String> = text
        .lines()
        .filter(|line| line.starts_with("error"))
        .map(str::to_string)
        .collect();
    let code = output.status.code().unwrap_or(-1);
    if !(code == 0 || code == 1) && errors.is_empty() {
        return Err(GateError::new(
            format!("rustfmt exited {code} without reporting an error line"),
            format!(
                "run `rustfmt --edition {EDITION} --check {}` by hand and read the output; \
                 a parser that fails for an unknown reason must not read as a clean tree",
                paths.first().map_or("<file>", String::as_str)
            ),
        ));
    }
    Ok(errors)
}

/// A module declaration and where it was written.
struct Declaration {
    /// Module name as declared.
    name: String,
    /// The `#[path]` value attached to it, when there was one.
    path_attr: Option<String>,
    /// Enclosing inline `mod` blocks, outermost first.
    stack: Vec<String>,
}

/// What one file declares about other files.
struct Declared {
    mods: Vec<Declaration>,
    includes: Vec<String>,
    trybuild: Vec<String>,
}

/// One lexical token, enough to read declarations without parsing Rust.
enum Token {
    Ident(String),
    Str(String),
    Punct(char),
}

/// Tokenise source, dropping comments and keeping literal bodies.
///
/// Comments and literals are skipped through `structure_gate::opaque_span`, the
/// one owner of what is not code, so a `mod` inside a doc comment or a string
/// never reads as a declaration.
fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut at = 0usize;
    while at < text.len() {
        if !text.is_char_boundary(at) {
            at += 1;
            continue;
        }
        if let Some(span) = structure_gate::opaque_span(text, at) {
            if span > 0 {
                let mut end = (at + span).min(text.len());
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }
                let piece = &text[at..end];
                if let Some(body) = literal_body(piece) {
                    tokens.push(Token::Str(body));
                }
                at = end;
                continue;
            }
        }
        let byte = bytes[at];
        if byte == b'_' || byte.is_ascii_alphabetic() {
            let start = at;
            while at < bytes.len() && (bytes[at] == b'_' || bytes[at].is_ascii_alphanumeric()) {
                at += 1;
            }
            tokens.push(Token::Ident(text[start..at].to_string()));
            continue;
        }
        if byte.is_ascii_digit() {
            while at < bytes.len() && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'.') {
                at += 1;
            }
            continue;
        }
        let character = text[at..].chars().next().unwrap_or(' ');
        if !character.is_whitespace() {
            tokens.push(Token::Punct(character));
        }
        at += character.len_utf8();
    }
    tokens
}

/// The body of a string literal span, or `None` when the span is a comment or a
/// char literal.
fn literal_body(piece: &str) -> Option<String> {
    if piece.starts_with("//") || piece.starts_with("/*") || piece.starts_with('\'') {
        return None;
    }
    let after_prefix = piece
        .trim_start_matches('b')
        .trim_start_matches('r')
        .trim_start_matches('b');
    let hashes = after_prefix.len() - after_prefix.trim_start_matches('#').len();
    let raw = piece.contains('r') && piece.find('r') < piece.find('"');
    let opening = piece.find('"')? + 1;
    let closing = if hashes > 0 {
        piece.rfind(&format!("\"{}", "#".repeat(hashes)))?
    } else {
        piece.rfind('"')?
    };
    if closing < opening {
        return None;
    }
    let body = &piece[opening..closing];
    if raw {
        return Some(body.to_string());
    }
    let mut unescaped = String::with_capacity(body.len());
    let mut characters = body.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            if let Some(escaped) = characters.next() {
                unescaped.push(escaped);
            }
            continue;
        }
        unescaped.push(character);
    }
    Some(unescaped)
}

/// Read module declarations, `include!` arguments and trybuild patterns.
fn scan_source(text: &str) -> Declared {
    let tokens = tokenize(text);
    let mut mods = Vec::new();
    let mut includes = Vec::new();
    let mut trybuild = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut opened: Vec<usize> = Vec::new();
    let mut depth = 0usize;
    let mut pending: Option<String> = None;
    let mut at = 0usize;
    while at < tokens.len() {
        match &tokens[at] {
            Token::Punct('#') => {
                if let Some((path, next)) = path_attribute(&tokens, at) {
                    pending = Some(path);
                    at = next;
                    continue;
                }
                at += 1;
            }
            Token::Punct('{') => {
                depth += 1;
                at += 1;
            }
            Token::Punct('}') => {
                depth = depth.saturating_sub(1);
                if opened.last() == Some(&depth) {
                    opened.pop();
                    stack.pop();
                }
                at += 1;
            }
            Token::Ident(word) if word == "mod" => {
                let Some(Token::Ident(name)) = tokens.get(at + 1) else {
                    at += 1;
                    continue;
                };
                match tokens.get(at + 2) {
                    Some(Token::Punct('{')) => {
                        stack.push(name.clone());
                        opened.push(depth);
                        depth += 1;
                        at += 3;
                    }
                    Some(Token::Punct(';')) => {
                        mods.push(Declaration {
                            name: name.clone(),
                            path_attr: pending.take(),
                            stack: stack.clone(),
                        });
                        at += 3;
                    }
                    _ => at += 2,
                }
                pending = None;
            }
            Token::Ident(word) if word == "include" => {
                if let (Some(Token::Punct('!')), Some(Token::Punct('(')), Some(Token::Str(body))) =
                    (tokens.get(at + 1), tokens.get(at + 2), tokens.get(at + 3))
                {
                    includes.push(body.clone());
                    at += 4;
                    continue;
                }
                at += 1;
            }
            Token::Ident(word) if word == "compile_fail" || word == "pass" => {
                if matches!(tokens.get(at.wrapping_sub(1)), Some(Token::Punct('.')))
                    && at > 0
                    && matches!(tokens.get(at + 1), Some(Token::Punct('(')))
                {
                    if let Some(Token::Str(body)) = tokens.get(at + 2) {
                        trybuild.push(body.clone());
                        at += 3;
                        continue;
                    }
                }
                at += 1;
            }
            _ => at += 1,
        }
    }
    Declared {
        mods,
        includes,
        trybuild,
    }
}

/// The `#[path = "..."]` value starting at a `#` token, and the token after it.
fn path_attribute(tokens: &[Token], at: usize) -> Option<(String, usize)> {
    if !matches!(tokens.get(at + 1), Some(Token::Punct('['))) {
        return None;
    }
    let mut depth = 0usize;
    let mut index = at + 1;
    let mut saw_path = false;
    let mut value = None;
    while index < tokens.len() {
        match &tokens[index] {
            Token::Punct('[') => depth += 1,
            Token::Punct(']') => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            Token::Ident(word) if word == "path" => saw_path = true,
            Token::Str(body) if saw_path => value = Some(body.clone()),
            _ => {}
        }
        index += 1;
    }
    value.map(|body| (body, index + 1))
}

/// What a declared target path resolves against.
struct Targets<'t> {
    entry: &'t str,
    sources: &'t BTreeSet<String>,
    tree: &'t Tree,
}

impl Targets<'_> {
    /// Record a declared target root, or report that it names no tracked file.
    fn declare(
        &self,
        label: String,
        path: String,
        roots: &mut Vec<(String, String)>,
        report: &mut Report,
    ) {
        let path = normalize(&path);
        if self.sources.contains(&path) {
            roots.push((path, label));
            return;
        }
        let state = if self.tree.exists(&path) {
            "exists but is untracked"
        } else {
            "does not exist"
        };
        report.find(Finding::in_file(
            self.entry,
            format!("declared target {label} names `{path}`, which {state}"),
            "restore the file, or delete the target entry; a target path naming nothing \
             is a target that never runs and never says so",
        ));
    }
}

/// Every declared and autodiscovered target root of one manifest.
fn collect_roots(
    entry: &str,
    table: &toml::Table,
    sources: &BTreeSet<String>,
    tree: &Tree,
    roots: &mut Vec<(String, String)>,
    report: &mut Report,
) {
    let Some(package) = table.get("package").and_then(toml::Value::as_table) else {
        return;
    };
    let directory = parent_of(entry);
    let name = package
        .get("name")
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
        .to_string();

    let targets = Targets {
        entry,
        sources,
        tree,
    };

    // A `[lib]` table without a `path` is still the default root, which is how
    // every proc-macro crate declares itself: `proc-macro = true` and nothing
    // else. Reading the table's presence as a declaration loses the whole crate.
    let declared_lib = table
        .get("lib")
        .and_then(toml::Value::as_table)
        .and_then(|lib| lib.get("path"))
        .and_then(toml::Value::as_str);
    if let Some(path) = declared_lib {
        targets.declare(
            format!("[lib] of `{entry}`"),
            join(&directory, path),
            roots,
            report,
        );
    } else if sources.contains(&normalize(&join(&directory, "src/lib.rs"))) {
        roots.push((
            normalize(&join(&directory, "src/lib.rs")),
            format!("src/lib.rs of `{entry}`"),
        ));
    }

    match package.get("build") {
        Some(toml::Value::String(build)) => targets.declare(
            format!("package.build of `{entry}`"),
            join(&directory, build),
            roots,
            report,
        ),
        Some(toml::Value::Boolean(false)) => {}
        _ => {
            let candidate = normalize(&join(&directory, "build.rs"));
            if sources.contains(&candidate) {
                roots.push((candidate, format!("build.rs of `{entry}`")));
            }
        }
    }

    for spec in array_of(table, "bin") {
        let bin_name = spec
            .get("name")
            .and_then(toml::Value::as_str)
            .unwrap_or(&name)
            .to_string();
        if let Some(path) = spec.get("path").and_then(toml::Value::as_str) {
            targets.declare(
                format!("[[bin]] {bin_name} of `{entry}`"),
                join(&directory, path),
                roots,
                report,
            );
            continue;
        }
        let mut guesses = Vec::new();
        if bin_name == name {
            guesses.push(normalize(&join(&directory, "src/main.rs")));
        }
        guesses.push(normalize(&join(
            &directory,
            &format!("src/bin/{bin_name}.rs"),
        )));
        guesses.push(normalize(&join(
            &directory,
            &format!("src/bin/{bin_name}/main.rs"),
        )));
        match guesses.iter().find(|guess| sources.contains(*guess)) {
            Some(found) => roots.push((found.clone(), format!("[[bin]] {bin_name} of `{entry}`"))),
            None => report.find(Finding::in_file(
                entry,
                format!(
                    "declared target [[bin]] {bin_name} has no `path` and none of {} is tracked",
                    guesses.join(", ")
                ),
                "add the source file, or delete the [[bin]] entry",
            )),
        }
    }

    for (kind, folder, auto) in [
        ("test", "tests", "autotests"),
        ("bench", "benches", "autobenches"),
        ("example", "examples", "autoexamples"),
    ] {
        for spec in array_of(table, kind) {
            let spec_name = spec
                .get("name")
                .and_then(toml::Value::as_str)
                .unwrap_or("<unnamed>")
                .to_string();
            if let Some(path) = spec.get("path").and_then(toml::Value::as_str) {
                targets.declare(
                    format!("[[{kind}]] {spec_name} of `{entry}`"),
                    join(&directory, path),
                    roots,
                    report,
                );
                continue;
            }
            let guesses = vec![
                normalize(&join(&directory, &format!("{folder}/{spec_name}.rs"))),
                normalize(&join(&directory, &format!("{folder}/{spec_name}/main.rs"))),
            ];
            match guesses.iter().find(|guess| sources.contains(*guess)) {
                Some(found) => roots.push((
                    found.clone(),
                    format!("[[{kind}]] {spec_name} of `{entry}`"),
                )),
                None => report.find(Finding::in_file(
                    entry,
                    format!(
                        "declared target [[{kind}]] {spec_name} has no `path` and none of {} \
                         is tracked",
                        guesses.join(", ")
                    ),
                    format!("add the source file, or delete the [[{kind}]] entry"),
                )),
            }
        }
        if package.get(auto).and_then(toml::Value::as_bool) != Some(false) {
            for path in auto_roots(sources, &directory, folder) {
                roots.push((path, format!("autodiscovered {kind} of `{entry}`")));
            }
        }
    }

    if package.get("autobins").and_then(toml::Value::as_bool) != Some(false) {
        let main = normalize(&join(&directory, "src/main.rs"));
        if sources.contains(&main) {
            roots.push((main, format!("src/main.rs of `{entry}`")));
        }
        for path in auto_roots(sources, &directory, "src/bin") {
            roots.push((path, format!("autodiscovered bin of `{entry}`")));
        }
    }
}

/// Cargo's own autodiscovery: `<kind>/*.rs` plus `<kind>/*/main.rs`.
fn auto_roots(sources: &BTreeSet<String>, directory: &str, kind: &str) -> Vec<String> {
    let base = normalize(&join(directory, kind));
    let mut found: Vec<String> = sources
        .iter()
        .filter(|path| parent_of(path) == base)
        .cloned()
        .collect();
    found.extend(
        sources
            .iter()
            .filter(|path| {
                file_name(Path::new(path)) == Some("main.rs") && {
                    let parent = parent_of(path);
                    parent != base && parent_of(&parent) == base
                }
            })
            .cloned(),
    );
    found
}

/// Every table of an array-of-tables entry, or nothing when the key is absent.
fn array_of(table: &toml::Table, key: &str) -> Vec<toml::Table> {
    table
        .get(key)
        .and_then(toml::Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_table().cloned())
                .collect()
        })
        .unwrap_or_default()
}

/// Whether a path carries the given extension.
fn extension_is(path: &Path, extension: &str) -> bool {
    path.extension().and_then(|value| value.to_str()) == Some(extension)
}

/// The final component of a path.
fn file_name(path: &Path) -> Option<&str> {
    path.file_name().and_then(|name| name.to_str())
}

/// A repository-relative path as a slash-separated key.
fn as_key(path: &PathBuf) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// The directory part of a slash-separated key, empty at the repository root.
fn parent_of(path: &str) -> String {
    match path.rfind('/') {
        Some(at) => path[..at].to_string(),
        None => String::new(),
    }
}

/// The file name without its extension.
fn stem_of(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rfind('.') {
        Some(at) => &name[..at],
        None => name,
    }
}

/// Join a directory key and a relative path, keeping slash separators.
fn join(directory: &str, relative: &str) -> String {
    if directory.is_empty() {
        return relative.to_string();
    }
    format!("{directory}/{relative}")
}

/// Collapse `.` and `..` so a `#[path]` that climbs compares equal to a tracked
/// entry.
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// Whether a path matches a pattern whose `*` and `?` stay inside one component.
fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    if pattern.len() != path.len() {
        return false;
    }
    pattern
        .iter()
        .zip(path.iter())
        .all(|(pattern, part)| component_match(pattern, part))
}

/// Whether one path component matches one pattern component.
fn component_match(pattern: &str, part: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let part: Vec<char> = part.chars().collect();
    let mut memo = vec![vec![false; part.len() + 1]; pattern.len() + 1];
    memo[0][0] = true;
    for at in 1..=pattern.len() {
        if pattern[at - 1] == '*' {
            memo[at][0] = memo[at - 1][0];
        }
    }
    for at in 1..=pattern.len() {
        for index in 1..=part.len() {
            memo[at][index] = match pattern[at - 1] {
                '*' => memo[at - 1][index] || memo[at][index - 1],
                '?' => memo[at - 1][index - 1],
                literal => memo[at - 1][index - 1] && literal == part[index - 1],
            };
        }
    }
    memo[pattern.len()][part.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a `mod` inside a comment or a string is not a declaration, and a
    /// declaration inside an inline module resolves under that module's
    /// directory. Both were live cases in the tree the shell original walked.
    #[test]
    fn declarations_come_from_code_only() {
        let source = "// mod commented;\nmod real;\nmod inner { mod nested; }\n\
                      let text = \"mod quoted;\";\n";
        let found = scan_source(source);
        let names: Vec<&str> = found
            .mods
            .iter()
            .map(|declaration| declaration.name.as_str())
            .collect();
        assert_eq!(names, vec!["real", "nested"]);
        let nested = found
            .mods
            .iter()
            .find(|declaration| declaration.name == "nested")
            .expect("nested is declared");
        assert_eq!(nested.stack, vec!["inner".to_string()]);
    }

    /// WHY: `#[path]` is how a file outside the module directory joins a crate,
    /// including a file that is not called `.rs`. Losing the attribute makes the
    /// gate report a real module as an orphan.
    #[test]
    fn a_path_attribute_names_its_own_file() {
        let found = scan_source("#[path = \"../templates/artifact.rs.tmpl\"]\nmod artifact;\n");
        assert_eq!(found.mods.len(), 1);
        assert_eq!(
            found.mods[0].path_attr.as_deref(),
            Some("../templates/artifact.rs.tmpl")
        );
    }

    /// WHY: `include!` and trybuild are the two ways a tracked file is compiled
    /// without a `mod` naming it. A gate blind to either reports a live fixture
    /// as an orphan, and the fix would be to delete a test that works.
    #[test]
    fn includes_and_trybuild_patterns_are_read() {
        let found = scan_source(
            "include!(\"generated/table.rs\");\n\
             let cases = trybuild::TestCases::new();\n\
             cases.compile_fail(\"tests/ui/*.rs\");\n\
             cases.pass(\"tests/ui/good.rs\");\n",
        );
        assert_eq!(found.includes, vec!["generated/table.rs".to_string()]);
        assert_eq!(
            found.trybuild,
            vec!["tests/ui/*.rs".to_string(), "tests/ui/good.rs".to_string()]
        );
    }

    /// WHY: a `#[path]` that climbs out of the module directory must compare
    /// equal to the tracked entry it names, or the file it points at reads as
    /// both a missing module and an orphan.
    #[test]
    fn relative_paths_collapse() {
        assert_eq!(normalize("a/b/../c/./d.rs"), "a/b/c/d.rs");
        assert_eq!(normalize("./a.rs"), "a.rs");
    }

    /// WHY: the trybuild exemption is granted by pattern, and a pattern that
    /// crossed a directory separator would exempt files nothing compiles.
    #[test]
    fn a_star_stays_inside_one_component() {
        assert!(glob_match("vyre-macros/tests/ui/*.rs", "vyre-macros/tests/ui/bad.rs"));
        assert!(!glob_match(
            "vyre-macros/tests/ui/*.rs",
            "vyre-macros/tests/ui/deep/bad.rs"
        ));
        assert!(glob_match("a/b?.rs", "a/bc.rs"));
        assert!(!glob_match("a/b?.rs", "a/bcd.rs"));
    }

    /// WHY: a raw string keeps its backslashes and a plain string does not, and
    /// both forms appear in `#[path]` attributes on Windows-style paths.
    #[test]
    fn literal_bodies_are_unescaped_once() {
        assert_eq!(literal_body("\"a\\\"b\""), Some("a\"b".to_string()));
        assert_eq!(literal_body("r#\"a\\b\"#"), Some("a\\b".to_string()));
        assert_eq!(literal_body("// comment"), None);
    }
}
