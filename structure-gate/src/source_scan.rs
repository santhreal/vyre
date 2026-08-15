//! Reading workspace Rust sources as text, for the contracts that judge the
//! checked-in tree.
//!
//! A contract that judges the tree walks every source file, masks what is not
//! code, and finds the end of a brace-delimited block. Each scanner carried its
//! own copy of all three, so a fix to the masker reached only the file its
//! author had open, and two scanners could disagree about which files the
//! workspace contains.

use std::fs;
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
        let text = fs::read_to_string(&file).ok()?;
        let relative = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        Some((relative, text))
    })
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
#[must_use]
pub fn mask_comments_and_strings(text: &str) -> String {
    let mut masked = text.as_bytes().to_vec();
    let mut at = 0usize;
    while at < text.len() {
        if !text.is_char_boundary(at) {
            at += 1;
            continue;
        }
        let Some(span) = opaque_span(text, at) else {
            at += 1;
            continue;
        };
        let mut end = (at + span.max(1)).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        for byte in &mut masked[at..end] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
        at = end;
    }
    String::from_utf8(masked)
        .expect("masking replaces whole opaque spans with ASCII, so the rest stays valid UTF-8")
}
