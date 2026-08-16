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
