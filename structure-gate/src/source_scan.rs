//! Reading workspace Rust sources as text, for the contracts that judge the
//! checked-in tree.
//!
//! A contract that judges the tree walks every source file, masks what is not
//! code, and finds the end of a brace-delimited block. Each scanner carried its
//! own copy of all three, so a fix to the masker reached only the file its
//! author had open, and two scanners could disagree about which files the
//! workspace contains.

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
/// A file that cannot be opened or is not UTF-8 yields nothing, because a
/// scanner reports what the tree says and an unreadable file says nothing.
///
/// # Panics
///
/// When the file holds more than [`MAX_SOURCE_BYTES`]. Truncating it would let
/// a scanner judge a source it read only part of and report the tree as clean
/// past the cut, which is the one answer worse than refusing.
fn read_source(path: &Path) -> Option<String> {
    let mut text = String::new();
    fs::File::open(path)
        .ok()?
        .take(MAX_SOURCE_BYTES + 1)
        .read_to_string(&mut text)
        .ok()?;
    assert!(
        text.len() as u64 <= MAX_SOURCE_BYTES,
        "Fix: {} holds more than {MAX_SOURCE_BYTES} bytes, so a tree scan cannot read it whole; \
         split the file or keep generated output out of the source tree",
        path.display()
    );
    Some(text)
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
    let rest = &text[at..];
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
