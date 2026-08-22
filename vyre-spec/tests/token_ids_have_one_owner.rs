//! Token numbering has exactly one declaration site: this crate.
//!
//! WHY. A token id is a wire contract. `vyre-grammar-gen` bakes the C11 ids
//! into the DFA and LR blobs it emits and `vyre-libs::parsing::c` decodes those
//! blobs on the GPU, so the two sides must agree on every value. They used to
//! agree by hand: each crate declared its own `TOK_*` table and a parity suite
//! paired the names. That catches a value that moves, and nothing else. It does
//! not catch a third copy arriving, a copy that adds a name the other side
//! never learns, or a copy in a crate the parity suite does not import. All
//! three fail silently, because a lexer that decodes every token as something
//! else still produces a parse.
//!
//! This gate closes the class instead of the incident: no file outside
//! `vyre-spec` may declare a `TOK_`-prefixed constant at all, so a new copy
//! fails on arrival rather than on drift. The file set is walked from the
//! checkout at run time, so a copy in a crate nobody thought of is still seen,
//! and the assertion is an exact equality against an empty set, so the failure
//! names the offending files rather than a count.
//!
//! What it does not catch: a numbering copied under a different prefix, and a
//! value that disagrees with a table baked into a committed blob. The first is
//! a naming convention this workspace keeps; the second is what the generator
//! round-trip suites cover.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Crate directory allowed to declare token ids.
const OWNER: &str = "vyre-spec";

/// Below this many owned declarations the walk has stopped seeing the tree and
/// an empty offender set would mean nothing. The owner declares several hundred
/// C11 ids alone, so this floor cannot be met by an accident of the scanner.
const MINIMUM_OWNED_DECLARATIONS: usize = 200;

/// Distinct owner files that must carry declarations, for the same reason: one
/// file's worth of hits would pass the count floor while the walk missed the
/// rest of the crate.
const MINIMUM_OWNED_FILES: usize = 4;

#[test]
fn no_file_outside_the_owner_declares_a_token_id() {
    let root = checkout_root();
    let owned = Path::new(OWNER);

    let offenders: BTreeMap<String, Vec<String>> = declarations(&root)
        .into_iter()
        .filter(|(path, _)| !Path::new(path).starts_with(owned))
        .collect();

    let offending_files: BTreeSet<&String> = offenders.keys().collect();
    assert_eq!(
        offending_files,
        BTreeSet::new(),
        "Fix: token ids are declared outside `{OWNER}`. Move the numbering into \
         a `vyre-spec` module and re-export it from the consumer so the published \
         path keeps resolving. Offenders: {offenders:?}"
    );
}

#[test]
fn the_walk_actually_reaches_the_owner() {
    let root = checkout_root();
    let owned = Path::new(OWNER);

    let found: BTreeMap<String, Vec<String>> = declarations(&root)
        .into_iter()
        .filter(|(path, _)| Path::new(path).starts_with(owned))
        .collect();

    let total: usize = found.values().map(Vec::len).sum();
    assert!(
        total >= MINIMUM_OWNED_DECLARATIONS && found.len() >= MINIMUM_OWNED_FILES,
        "Fix: the source walk found {total} token declarations across {} files in \
         `{OWNER}`, below the floor of {MINIMUM_OWNED_DECLARATIONS} across \
         {MINIMUM_OWNED_FILES}. The walk is not reading the tree, so an empty \
         offender set proves nothing. Files seen: {:?}",
        found.len(),
        found.keys().collect::<Vec<_>>()
    );
}

#[test]
fn the_scanner_reads_module_scope_and_ignores_function_locals() {
    let source = r##"
pub const TOK_TOP: u32 = 1;
mod inner {
    pub(crate) static TOK_NESTED: u32 = 2;
    fn body() {
        const TOK_LOCAL: u32 = 3;
        let brace = "} } }";
        let raw = r#"} const TOK_IN_RAW: u32 = 4; {"#;
        let ch = '}';
        let esc = '\u{7d}';
    }
}
// pub const TOK_COMMENTED: u32 = 5;
/* pub const TOK_BLOCK: u32 = 6; /* } */ */
impl Thing {
    const TOK_ASSOCIATED: u32 = 7;
    const fn made(&self) -> u32 {
        const TOK_IN_CONST_FN: u32 = 8;
        TOK_IN_CONST_FN
    }
}
trait Decl {
    fn no_body(&self) -> u32;
}
pub const TOK_TAIL: u32 = 9;
"##;

    assert_eq!(
        token_declarations(source),
        vec![
            "TOK_TOP".to_owned(),
            "TOK_NESTED".to_owned(),
            "TOK_ASSOCIATED".to_owned(),
            "TOK_TAIL".to_owned(),
        ]
    );
}

/// Absolute root of the checkout this test runs in.
///
/// Resolved from the working directory rather than `CARGO_MANIFEST_DIR`: a
/// target directory shared by several checkouts computes the same unit hash for
/// a member in each of them, so a compiled-in path can name a different tree
/// than the one under test.
fn checkout_root() -> PathBuf {
    let start = std::env::current_dir().expect("Fix: the working directory must be readable");
    for candidate in start.ancestors() {
        let manifest = candidate.join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        if text.lines().any(|line| line.trim_start() == "[workspace]") {
            return candidate.to_path_buf();
        }
    }
    panic!(
        "Fix: no ancestor of `{}` declares a `[workspace]`; this gate reports on \
         a workspace and has nothing to measure outside one",
        start.display()
    );
}

/// Every `TOK_`-prefixed declaration in the checkout, keyed by root-relative path.
fn declarations(root: &Path) -> BTreeMap<String, Vec<String>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("Fix: cannot read `{}`: {error}", directory.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic!(
                    "Fix: cannot read an entry of `{}`: {error}",
                    directory.display()
                )
            });
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                // `target` holds build output including vendored sources, and a
                // dot directory holds VCS and tool state. Neither is workspace
                // source, and walking them would report generated copies.
                if name != "target" && !name.starts_with('.') {
                    pending.push(path);
                }
                continue;
            }
            if !name.ends_with(".rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("Fix: cannot read `{}`: {error}", path.display()));
            let names = token_declarations(&source);
            if names.is_empty() {
                continue;
            }
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            found.insert(relative, names);
        }
    }
    found
}

/// Names of the `TOK_`-prefixed `const` and `static` items declared outside any
/// function body, in source order.
///
/// Brace depth is tracked over a source stripped of comments, string literals
/// and character literals as it goes, because a brace inside `"{}"` or `'}'`
/// would otherwise move the depth and hide a declaration behind a phantom
/// function body.
fn token_declarations(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut names = Vec::new();
    let mut depth: i32 = 0;
    let mut function_bodies: Vec<i32> = Vec::new();
    let mut pending_signature = false;
    let mut index = 0usize;

    while index < bytes.len() {
        if let Some(next) = skip_comment(bytes, index) {
            index = next;
            continue;
        }
        if let Some(next) = skip_literal(bytes, index) {
            index = next;
            continue;
        }
        match bytes[index] {
            b'{' => {
                depth += 1;
                if pending_signature {
                    function_bodies.push(depth);
                    pending_signature = false;
                }
                index += 1;
            }
            b'}' => {
                if function_bodies.last() == Some(&depth) {
                    function_bodies.pop();
                }
                depth -= 1;
                index += 1;
            }
            // A signature with no body: `fn f(&self);` in a trait or extern block.
            b';' => {
                pending_signature = false;
                index += 1;
            }
            byte if is_identifier_start(byte) => {
                let end = identifier_end(bytes, index);
                let word = &source[index..end];
                if word == "fn" {
                    pending_signature = true;
                } else if (word == "const" || word == "static") && function_bodies.is_empty() {
                    if let Some(name) = declared_name(source, end) {
                        names.push(name);
                    }
                }
                index = end;
            }
            _ => index += 1,
        }
    }

    names
}

/// End index of the comment starting at `index`, or `None` when none starts there.
fn skip_comment(bytes: &[u8], index: usize) -> Option<usize> {
    if bytes[index] != b'/' {
        return None;
    }
    match bytes.get(index + 1) {
        Some(b'/') => {
            let mut cursor = index + 2;
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            Some(cursor)
        }
        Some(b'*') => {
            let mut cursor = index + 2;
            let mut nesting = 1u32;
            while cursor < bytes.len() && nesting > 0 {
                if bytes[cursor] == b'/' && bytes.get(cursor + 1) == Some(&b'*') {
                    nesting += 1;
                    cursor += 2;
                } else if bytes[cursor] == b'*' && bytes.get(cursor + 1) == Some(&b'/') {
                    nesting -= 1;
                    cursor += 2;
                } else {
                    cursor += 1;
                }
            }
            Some(cursor)
        }
        _ => None,
    }
}

/// End index of the string, raw string, byte string or character literal
/// starting at `index`, or `None` when none starts there.
fn skip_literal(bytes: &[u8], index: usize) -> Option<usize> {
    let mut cursor = index;
    if bytes[cursor] == b'b' {
        cursor += 1;
    }
    if bytes.get(cursor) == Some(&b'r') {
        let mut hashes = cursor + 1;
        while bytes.get(hashes) == Some(&b'#') {
            hashes += 1;
        }
        if bytes.get(hashes) == Some(&b'"') {
            return Some(end_of_raw_string(bytes, hashes + 1, hashes - cursor - 1));
        }
    }
    if bytes.get(cursor) == Some(&b'"') {
        return Some(end_of_quoted(bytes, cursor + 1, b'"'));
    }
    if bytes[index] != b'\'' {
        return None;
    }
    // `'a` opening a lifetime, not a character literal: an identifier byte that
    // is not followed by the closing quote.
    let after = bytes.get(index + 1).copied();
    if after.is_some_and(is_identifier_start) && bytes.get(index + 2) != Some(&b'\'') {
        return None;
    }
    Some(end_of_quoted(bytes, index + 1, b'\''))
}

/// End index of a quoted run opened at `start`, honouring backslash escapes.
fn end_of_quoted(bytes: &[u8], start: usize, quote: u8) -> usize {
    let mut cursor = start;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'\\' => cursor += 2,
            byte if byte == quote => return cursor + 1,
            _ => cursor += 1,
        }
    }
    cursor
}

/// End index of a raw string opened at `start` and closed by `"` plus `hashes`
/// hash marks. Raw strings honour no escapes.
fn end_of_raw_string(bytes: &[u8], start: usize, hashes: usize) -> usize {
    let mut cursor = start;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"' {
            let closing = cursor + 1 + hashes;
            if bytes[cursor + 1..closing.min(bytes.len())]
                .iter()
                .all(|&byte| byte == b'#')
                && closing <= bytes.len()
            {
                return closing;
            }
        }
        cursor += 1;
    }
    cursor
}

/// Name declared after a `const` or `static` keyword ending at `from`, when it
/// carries the token prefix. `mut` between the keyword and the name is skipped.
fn declared_name(source: &str, from: usize) -> Option<String> {
    let bytes = source.as_bytes();
    let mut cursor = from;
    for _ in 0..2 {
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || !is_identifier_start(bytes[cursor]) {
            return None;
        }
        let end = identifier_end(bytes, cursor);
        let word = &source[cursor..end];
        if word == "mut" {
            cursor = end;
            continue;
        }
        return word.starts_with("TOK_").then(|| word.to_owned());
    }
    None
}

fn is_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn identifier_end(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start;
    while cursor < bytes.len() && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_') {
        cursor += 1;
    }
    cursor
}
