//! Reading workspace Rust sources as text, for the contracts that judge the
//! checked-in tree.
//!
//! A contract that judges the tree walks every source file, masks what is not
//! code, and finds the end of a brace-delimited block. Each scanner carried its
//! own copy of all three, so a fix to the masker reached only the file its
//! author had open, and two scanners could disagree about which files the
//! workspace contains.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use walkdir::{DirEntry, WalkDir};

use crate::opaque_span;

/// Every `.rs` file under `root`, sorted, with a path relative to `root` and
/// its contents.
///
/// The path is slash-separated on every platform so a reported location reads
/// the same in CI and on a workstation. Files that cannot be read are skipped:
/// a scanner reports what the tree says, and a file it cannot open says
/// nothing.
///
/// Lazy on purpose. The workspace holds thousands of sources and a scanner
/// looks at one at a time, so collecting every file's text first would hold the
/// whole tree in memory to answer a question about one file.
pub fn rust_sources_with_text(root: &Path) -> impl Iterator<Item = (String, String)> + '_ {
    rust_sources(root).into_iter().filter_map(move |file| {
        let text = read_source(&file)?;
        let relative = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        Some((relative, text))
    })
}

/// The most bytes one source may hold before a scanner refuses to judge it.
///
/// The per-file line cap holds every tracked source far below this, so a file
/// over the bound is not a large source: it is generated output or a binary
/// that reached the walk. Reading it in full would let one file decide how much
/// memory a tree scan takes.
const MAX_SOURCE_BYTES: u64 = 4 * 1024 * 1024;

/// The text of one source, read under [`MAX_SOURCE_BYTES`].
///
/// A file that cannot be opened, is not UTF-8, or holds more than the cap
/// yields nothing. Truncating it would let a scanner judge a source it read
/// only part of and report the tree as clean past the cut. Aborting is the
/// other wrong answer: one generated file dropped into a source tree took down
/// every rule over the whole tree, where a reader that has to have the file can
/// name it and a reader that only needs the shape of the module tree carries
/// on.
fn read_source(path: &Path) -> Option<String> {
    let mut text = String::new();
    fs::File::open(path)
        .ok()?
        .take(MAX_SOURCE_BYTES + 1)
        .read_to_string(&mut text)
        .ok()?;
    (text.len() as u64 <= MAX_SOURCE_BYTES).then_some(text)
}

/// Every `.rs` file under `root`, sorted.
///
/// Build outputs and hidden directories are skipped, which is what makes the
/// set the checked-in tree rather than whatever the last build left behind.
fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| !is_pruned(entry))
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(DirEntry::into_path)
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect();
    found.sort();
    found
}

/// Whether `directory` holds a Rust source, at any depth below it.
///
/// The question every placement answer has to ask. Git tracks files, not
/// directories, so a domain deleted in one commit leaves its directory behind
/// in every checkout that pulled the deletion, and asking whether the directory
/// is there names an empty shell as the owner of code that lives somewhere
/// else. A directory holding a `.rs` file holds code.
#[must_use]
pub fn carries_rust_source(directory: &Path) -> bool {
    WalkDir::new(directory)
        .into_iter()
        .filter_entry(|entry| !is_pruned(entry))
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "rs")
        })
}

/// The directory named `name` under `root` that carries the code, shallowest
/// match first.
///
/// A composition move nests a domain under another one: the optimizer ops live
/// in `vyre-libs/src/nn/optim` and `vyre-libs/src/optim` never existed. A name
/// derived from an operation id therefore answers with a directory that holds
/// no code, and a table of moved names is a snapshot that goes stale on the
/// next move, so the tree is asked instead. The shallowest match wins because a
/// domain re-declared deeper is a submodule of the one above it, and ties break
/// on path order so two checkouts of one commit answer the same.
#[must_use]
pub fn source_directory_named(root: &Path, name: &str) -> Option<PathBuf> {
    let mut matches: Vec<PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| !is_pruned(entry))
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_dir() && entry.file_name() == name)
        .map(DirEntry::into_path)
        .filter(|path| carries_rust_source(path))
        .collect();
    matches.sort_by(|left, right| {
        left.components()
            .count()
            .cmp(&right.components().count())
            .then_with(|| left.cmp(right))
    });
    matches.into_iter().next()
}

/// Whether the walk should refuse to descend into `entry`.
///
/// The root itself is never pruned: the checkout may sit in a hidden directory
/// on a workstation, and pruning it would report an empty workspace.
fn is_pruned(entry: &DirEntry) -> bool {
    if entry.depth() == 0 || !entry.file_type().is_dir() {
        return false;
    }
    let name = entry.file_name().to_string_lossy();
    name == "target" || name.starts_with('.')
}

/// Index of the `}` closing the `{` at `open`.
pub fn matching_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate().skip(open) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `byte` can sit inside a Rust identifier.
#[must_use]
pub const fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// `text` with comments and literals blanked out.
///
/// Blanked rather than removed so every reported byte offset still maps to the
/// line it came from, and newlines survive so a reported line number is the
/// line the reader will open. Without this a brace inside a doc comment
/// desynchronises the brace matcher for the rest of the file, and the scanner
/// reports whatever happens to follow.
///
/// Which spans are not code is [`opaque_span`]'s answer, the same one the
/// registration parser reads, so a masker and a parser cannot disagree about
/// whether a raw string or a nested block comment holds code.
///
/// Built one character at a time rather than by overwriting bytes, so the result
/// is UTF-8 by construction and the walk never lands inside a character.
#[must_use]
pub fn mask_comments_and_strings(text: &str) -> String {
    let mut masked = String::with_capacity(text.len());
    let mut at = 0usize;
    while at < text.len() {
        let rest = &text[at..];
        if let Some(span) = opaque_span(text, at) {
            let mut end = (at + span.max(1)).min(text.len());
            while end < text.len() && !text.is_char_boundary(end) {
                end += 1;
            }
            for ch in text[at..end].chars() {
                if ch == '\n' {
                    masked.push('\n');
                } else {
                    // One space per byte, so a reported offset still maps to its line.
                    for _ in 0..ch.len_utf8() {
                        masked.push(' ');
                    }
                }
            }
            at = end;
            continue;
        }
        let Some(ch) = rest.chars().next() else {
            break;
        };
        masked.push(ch);
        at += ch.len_utf8();
    }
    masked
}

/// Interior text of every string literal in `text`, in source order.
///
/// A reader that tracks only the double quote desynchronises on a char literal
/// that holds one: `b'"'` opens a string that never closes, and every literal
/// after it is read as the gap between two later quotes. A lexer is full of
/// them, which is how the registration id in
/// `vyre-libs/src/parsing/python/lex.rs` went unread and the operation it
/// defines was reported as having no definition site.
///
/// Which spans are not code is [`opaque_span`](crate::opaque_span)'s answer,
/// the same one the masker and the registration parser read, so a raw string,
/// a byte string and a nested block comment are all one decision.
#[must_use]
pub fn string_literals(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut at = 0usize;
    while at < text.len() {
        let Some(span) = opaque_span(text, at) else {
            at += text[at..].chars().next().map_or(1, char::len_utf8);
            continue;
        };
        let mut end = (at + span.max(1)).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        if let Some(interior) = string_interior(&text[at..end]) {
            found.push(interior);
        }
        at = end;
    }
    found
}

/// Text between the delimiters of one string literal, or `None` when the span
/// is a comment, a char literal, or a literal nobody closed.
fn string_interior(literal: &str) -> Option<&str> {
    let bytes = literal.as_bytes();
    let mut open = 0usize;
    while matches!(bytes.get(open), Some(b'r' | b'b' | b'c' | b'#')) {
        open += 1;
    }
    if bytes.get(open) != Some(&b'"') {
        return None;
    }
    let hashes = literal[..open].bytes().filter(|byte| *byte == b'#').count();
    let close = literal.len().checked_sub(1 + hashes)?;
    if bytes.get(close) != Some(&b'"') {
        return None;
    }
    literal.get(open + 1..close)
}

/// One module file a crate root reaches, and the features on the way to it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleRoute {
    /// Module file, as an absolute path.
    pub path: PathBuf,
    /// Features a build must enable to compile the file, in declaration order.
    /// Empty means the crate compiles it unconditionally.
    pub features: Vec<String>,
}

/// Every module file `<crate_src>/lib.rs` reaches, with the features on the way.
///
/// The `mod` declarations are the route, not the directory listing: a file no
/// declaration names is not compiled, and a directory whose declaration is gone
/// is not a module in a checkout that pulled the deletion. Modules reachable
/// only in a test build are left out, so a fixture is never read as production.
///
/// Two readers asked this question and answered it differently. One took a
/// dialect's features from the crate root alone, which reported 13 imports in a
/// file that `encoding/mod.rs` declares behind the neural-network gates as
/// unreachable coupling, and read a `#[cfg(test)]` module file as production
/// source for 6 more. The other walked the declarations. The walk is here now,
/// once.
#[must_use]
pub fn module_routes(crate_src: &Path) -> Vec<ModuleRoute> {
    let mut found = Vec::new();
    let mut visited = BTreeSet::new();
    let mut pending = vec![ModuleRoute {
        path: crate_src.join("lib.rs"),
        features: Vec::new(),
    }];
    while let Some(module) = pending.pop() {
        if !visited.insert(identity(&module.path)) {
            continue;
        }
        // A module the walk cannot read is still on this route. Only the
        // descent stops, because an unread file declares no children, and the
        // reader that has to read the file is the one that can name why.
        if let Some(text) = read_source(&module.path) {
            let directory = module_directory(&module.path);
            for (name, attributes) in module_declarations(&text) {
                let Some(gates) = reachable_features(&attributes) else {
                    continue;
                };
                let mut features = module.features.clone();
                for gate in gates {
                    if !features.contains(&gate) {
                        features.push(gate);
                    }
                }
                let file = directory.join(format!("{name}.rs"));
                let path = if file.is_file() {
                    file
                } else {
                    directory.join(&name).join("mod.rs")
                };
                if path.is_file() {
                    pending.push(ModuleRoute { path, features });
                }
            }
        }
        found.push(module);
    }
    found
}

/// The identity a walk compares two module paths by.
///
/// A `mod` reachable twice, and a symlink that points back into the tree it
/// came from, both name a file the walk has already read. Comparing the
/// canonical path catches the second, where comparing the written path would
/// walk the loop until the process died.
fn identity(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Features gating the item declared on line `at`, or `None` when only a test
/// build reaches it.
#[must_use]
pub fn gating_features(text: &str, at: usize) -> Option<Vec<String>> {
    let lines: Vec<&str> = text.lines().collect();
    let blocks = attribute_blocks(&lines);
    reachable_features(&gating_attributes(&lines, &blocks, at))
}

/// Directory the modules a file declares live in.
fn module_directory(file: &Path) -> PathBuf {
    let parent = file.parent().unwrap_or(Path::new("")).to_path_buf();
    match file.file_name().and_then(|name| name.to_str()) {
        Some("lib.rs" | "mod.rs") | None => parent,
        Some(name) => parent.join(name.trim_end_matches(".rs")),
    }
}

/// `(module name, attributes above it)` for every out-of-line `mod` in a file.
fn module_declarations(text: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = text.lines().collect();
    let attributes = attribute_blocks(&lines);
    let mut declarations = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(name) = module_name(line) else {
            continue;
        };
        declarations.push((name, gating_attributes(&lines, &attributes, index)));
    }
    declarations
}

/// Module name declared by an out-of-line `mod` statement on one line.
fn module_name(line: &str) -> Option<String> {
    let rest = line.trim();
    let rest = rest.strip_prefix("pub ").unwrap_or(rest).trim_start();
    let rest = match rest.strip_prefix("pub(") {
        Some(tail) => tail.split_once(')')?.1.trim_start(),
        None => rest,
    };
    let name = rest.strip_prefix("mod ")?.strip_suffix(';')?.trim();
    (!name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .then(|| name.to_string())
}

/// `(last line, (first line, joined text))` for every attribute in a file.
///
/// An attribute is joined across lines because the tree writes
/// `#[cfg(any(\n    feature = "a",\n    feature = "b"\n))]`, and reading only
/// the last line of that spelling records no feature at all.
fn attribute_blocks(lines: &[&str]) -> BTreeMap<usize, (usize, String)> {
    let mut blocks = BTreeMap::new();
    let mut index = 0;
    while index < lines.len() {
        let trimmed = lines[index].trim();
        if !trimmed.starts_with("#[") {
            index += 1;
            continue;
        }
        let mut joined = trimmed.to_string();
        let mut last = index;
        while joined.matches('(').count() > joined.matches(')').count() && last + 1 < lines.len() {
            last += 1;
            joined.push(' ');
            joined.push_str(lines[last].trim());
        }
        blocks.insert(last, (index, joined));
        index = last + 1;
    }
    blocks
}

/// Attribute texts that gate the item on line `at`.
fn gating_attributes(
    lines: &[&str],
    blocks: &BTreeMap<usize, (usize, String)>,
    at: usize,
) -> Vec<String> {
    let mut found = Vec::new();
    let mut index = at;
    while index > 0 {
        index -= 1;
        let trimmed = lines[index].trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        let Some((first, text)) = blocks.get(&index) else {
            break;
        };
        if text.starts_with("#[cfg") {
            found.push(text.clone());
        }
        index = *first;
    }
    found
}

/// Features that reach an item, or `None` when only a test build reaches it.
///
/// A `cfg` naming `test` beside a feature, such as
/// `any(test, feature = "cpu-parity")`, still compiles in a feature build, so
/// only a `cfg` that requires `test` and names no feature at all is test-only.
fn reachable_features(attributes: &[String]) -> Option<Vec<String>> {
    let mut features = Vec::new();
    for attribute in attributes {
        let named = cfg_feature_names(attribute);
        if named.is_empty() && requires_test(attribute) {
            return None;
        }
        for feature in named {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    Some(features)
}

/// Every feature named by one `cfg` attribute.
///
/// `any`, `all` and `not` flatten to the names they mention: the question is
/// which features can reach the item, not the predicate that admits it.
#[must_use]
pub fn cfg_feature_names(attribute: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = attribute;
    while let Some(start) = rest.find("feature = \"") {
        rest = &rest[start + "feature = \"".len()..];
        let Some(end) = rest.find('"') else {
            break;
        };
        let feature = rest[..end].to_string();
        if !feature.is_empty() && !found.contains(&feature) {
            found.push(feature);
        }
        rest = &rest[end + 1..];
    }
    found
}

/// Whether one `cfg` attribute requires the `test` predicate to hold.
///
/// Polarity is the whole question. `#[cfg(test)]` admits an item only in a test
/// build; `#[cfg(not(test))]` admits it in every build except that one, so the
/// item is production source. Reading the bare word made the second look like
/// the first and dropped such a module from every route it belonged to.
fn requires_test(attribute: &str) -> bool {
    let bytes = attribute.as_bytes();
    let mut negations: Vec<bool> = Vec::new();
    let mut last_word: Option<(usize, usize)> = None;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if is_word_byte(byte) {
            let start = index;
            while index < bytes.len() && is_word_byte(bytes[index]) {
                index += 1;
            }
            if &attribute[start..index] == "test"
                && negations.iter().filter(|negated| **negated).count() % 2 == 0
            {
                return true;
            }
            last_word = Some((start, index));
            continue;
        }
        if byte == b'(' {
            let opener = last_word.is_some_and(|(start, end)| &attribute[start..end] == "not");
            negations.push(opener);
            last_word = None;
        } else if byte == b')' {
            negations.pop();
            last_word = None;
        }
        index += 1;
    }
    false
}
