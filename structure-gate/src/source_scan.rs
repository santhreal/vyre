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
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use walkdir::{DirEntry, WalkDir};

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
pub fn rust_sources_with_text(root: &Path) -> impl Iterator<Item = SourceText> + '_ {
    rust_sources(root).into_iter().map(move |file| {
        let path = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        match read_source(&file) {
            Ok(text) => SourceText::Read { path, text },
            Err(reason) => SourceText::Unread { path, reason },
        }
    })
}

/// One tracked source, read or refused.
///
/// A refusal is carried to the caller rather than dropped. A scanner that skips
/// a file it could not read reports the tree as clean for a file nothing
/// judged, and the file most likely to be refused is the generated or oversized
/// one that a rule most wants to see.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceText {
    /// The file, read in full.
    Read {
        /// Path relative to the scanned root, with forward slashes.
        path: String,
        /// The whole file.
        text: String,
    },
    /// The file was not read, and why.
    Unread {
        /// Path relative to the scanned root, with forward slashes.
        path: String,
        /// What stopped the read.
        reason: String,
    },
}

impl SourceText {
    /// The path, whichever way the read went.
    #[must_use]
    pub fn path(&self) -> &str {
        match self {
            Self::Read { path, .. } | Self::Unread { path, .. } => path,
        }
    }
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
/// yields the reason instead of the text. Truncating it would let a scanner
/// judge a source it read only part of and report the tree as clean past the
/// cut. Aborting is the other wrong answer: one generated file dropped into a
/// source tree took down every rule over the whole tree. The reason travels
/// with the path so a reader that has to have the file can name it, and a
/// reader that only needs the shape of the module tree can carry on.
fn read_source(path: &Path) -> Result<String, String> {
    let mut text = String::new();
    let file = fs::File::open(path).map_err(|error| format!("cannot be opened: {error}"))?;
    file.take(MAX_SOURCE_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| format!("cannot be read as UTF-8: {error}"))?;
    if text.len() as u64 > MAX_SOURCE_BYTES {
        return Err(format!(
            "holds more than {MAX_SOURCE_BYTES} bytes, which is generated output or a binary rather than a source"
        ));
    }
    Ok(text)
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
    if open >= bytes.len() || bytes[open] != b'{' {
        return None;
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        let mut depth = 0usize;
        let mut offset = open;
        while offset < bytes.len() {
            if let Some(span) = opaque_span(text, offset) {
                offset += span.get();
                continue;
            }
            match bytes[offset] {
                b'{' => depth += 1,
                b'}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some(offset);
                    }
                }
                _ => {}
            }
            offset += 1;
        }
        None
    } else {
        let mut depth = 0usize;
        for (index, byte) in bytes.iter().enumerate().skip(open) {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some(index);
                    }
                }
                _ => {}
            }
        }
        None
    }
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
            let mut end = (at + span.get()).min(text.len());
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

/// Byte length of the span starting at `at` whose interior is not code: a line
/// or block comment, a string, a char literal, or any prefixed or raw form of
/// those. `None` when ordinary code starts there.
///
/// The gate reads source text without compiling it, so nothing else
/// distinguishes a comma inside `", "` from an argument separator.
///
/// The length is non-zero by type. Every caller skips the span to keep
/// scanning, so a zero-length answer would leave the cursor where it was and
/// hang the scan. Each caller used to defend itself against that, or not, and
/// the ones that did not were correct only because no arm here returns zero
/// today.
pub fn opaque_span(text: &str, at: usize) -> Option<NonZeroUsize> {
    let rest = text.get(at..)?;
    if let Some(body) = rest.strip_prefix("//") {
        return at_least_one_byte(2 + body.find('\n').map_or(body.len(), |end| end + 1));
    }
    if rest.starts_with("/*") {
        return at_least_one_byte(block_comment_len(rest));
    }
    if rest.starts_with('"') {
        return at_least_one_byte(escaped_string_len(rest));
    }
    if rest.starts_with('\'') {
        return char_literal_len(rest).and_then(at_least_one_byte);
    }
    prefixed_literal_len(text, at).and_then(at_least_one_byte)
}

/// The one place the non-zero guarantee is made.
///
/// Every arm of [`opaque_span`] measures at least one byte: a line comment is
/// at least `//`, a block comment at least `/*`, a string at least its opening
/// quote, a char literal at least three bytes, a prefixed literal at least its
/// prefix. The clamp is what keeps a future arm that measures zero from
/// turning every scan in the gate into a hang instead of a wrong answer.
fn at_least_one_byte(length: usize) -> Option<NonZeroUsize> {
    Some(NonZeroUsize::new(length).unwrap_or(NonZeroUsize::MIN))
}

/// Byte length of the block comment starting at `rest`, which nests in Rust.
fn block_comment_len(rest: &str) -> usize {
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut offset = 0usize;
    while offset + 1 < bytes.len() {
        if bytes[offset] == b'/' && bytes[offset + 1] == b'*' {
            depth += 1;
            offset += 2;
        } else if bytes[offset] == b'*' && bytes[offset + 1] == b'/' {
            depth -= 1;
            offset += 2;
            if depth == 0 {
                return offset;
            }
        } else {
            offset += 1;
        }
    }
    rest.len()
}

/// Byte length of the backslash-escaped string starting at `rest`.
///
/// An unterminated literal consumes the remaining text: the alternative is to
/// resume scanning inside a string, where every delimiter is misread.
fn escaped_string_len(rest: &str) -> usize {
    let bytes = rest.as_bytes();
    let mut offset = 1usize;
    while offset < bytes.len() {
        match bytes[offset] {
            b'\\' => offset += 2,
            b'"' => return offset + 1,
            _ => offset += 1,
        }
    }
    rest.len()
}

/// Byte length of the char literal starting at `rest`, or `None` when the quote
/// opens a lifetime or a loop label instead.
fn char_literal_len(rest: &str) -> Option<usize> {
    let body = &rest[1..];
    if let Some(escape) = body.strip_prefix('\\') {
        let escaped = if escape.starts_with('u') {
            escape.find('}')? + 1
        } else {
            escape.chars().next()?.len_utf8()
        };
        return Some(2 + escaped + escape[escaped..].find('\'')? + 1);
    }
    let literal = body.chars().next()?.len_utf8();
    body[literal..].starts_with('\'').then_some(literal + 2)
}

/// Byte length of a literal carrying a `r`, `b` or `c` prefix, including every
/// raw form. `None` when the bytes are an ordinary identifier such as `bytes`
/// or `crc32`.
fn prefixed_literal_len(text: &str, at: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_') {
        return None;
    }
    let rest = &text[at..];
    let prefix = rest
        .bytes()
        .take(2)
        .take_while(|byte| matches!(*byte, b'r' | b'b' | b'c'))
        .count();
    if prefix == 0 {
        return None;
    }
    let body = &rest[prefix..];
    if rest[..prefix].contains('r') {
        let hashes = body.bytes().take_while(|byte| *byte == b'#').count();
        let quoted = &body[hashes..];
        if !quoted.starts_with('"') {
            return None;
        }
        return Some(prefix + hashes + raw_string_len(quoted, hashes));
    }
    if body.starts_with('"') {
        return Some(prefix + escaped_string_len(body));
    }
    if body.starts_with('\'') {
        return char_literal_len(body).map(|len| prefix + len);
    }
    None
}

/// Byte length of the raw string opening at `quoted`, closed by a quote
/// followed by `hashes` hash marks. Raw strings honour no escape.
fn raw_string_len(quoted: &str, hashes: usize) -> usize {
    let bytes = quoted.as_bytes();
    let mut offset = 1usize;
    while offset < bytes.len() {
        if bytes[offset] == b'"'
            && quoted[offset + 1..]
                .bytes()
                .take_while(|byte| *byte == b'#')
                .count()
                >= hashes
        {
            return offset + 1 + hashes;
        }
        offset += 1;
    }
    quoted.len()
}

/// Byte offsets of code in `text`, with comments and literals skipped.
pub fn code_offsets(text: &str) -> impl Iterator<Item = usize> + '_ {
    let mut skip_to = 0usize;
    text.char_indices().filter_map(move |(at, _)| {
        if at < skip_to {
            return None;
        }
        if let Some(span) = opaque_span(text, at) {
            skip_to = at + span.get();
            return None;
        }
        Some(at)
    })
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
        let mut end = (at + span.get()).min(text.len());
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
        if let Ok(text) = read_source(&module.path) {
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
        let trimmed = attribute.trim();
        let predicate = if let Some(after) = trimmed.strip_prefix("#[cfg(") {
            let inner = after.strip_suffix(']').unwrap_or(after).trim();
            inner.strip_suffix(')').unwrap_or(inner).trim()
        } else {
            trimmed
        };
        let named = cfg_feature_names(attribute);
        if named.is_empty() && crate::cfg_test::requires_test(predicate) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_brace_ignores_braces_in_strings_and_comments() {
        let source = r##"{
            let s = "}";
            let raw = r#"{"#;
            // line comment with }
            /* block comment with { and } */
            let x = 1;
        }"##;
        let open = source.find('{').unwrap();
        let close = matching_brace(source.as_bytes(), open).unwrap();
        assert_eq!(close, source.len() - 1);
    }

    #[test]
    fn matching_brace_returns_none_for_non_brace_or_unclosed() {
        assert_eq!(matching_brace(b"not a brace", 0), None);
        assert_eq!(matching_brace(b"{ unclosed", 0), None);
    }

    #[test]
    fn opaque_span_refuses_non_utf8_boundary_offsets() {
        let source = "λ";
        assert!(!source.is_char_boundary(1));
        assert_eq!(opaque_span(source, 1), None);
    }
}
