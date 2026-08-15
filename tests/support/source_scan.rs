//! Reading workspace Rust sources as text, for the contract scanners.
//!
//! A contract that judges the checked-in tree walks every source file, masks
//! what is not code, and finds the end of a brace-delimited block. Each scanner
//! carried its own copy of all three, so a fix to the masker reached only the
//! file its author had open, and two scanners could disagree about which files
//! the workspace contains.

use std::fs;
use std::path::{Path, PathBuf};

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
    let mut found = Vec::new();
    collect(root, &mut found);
    found.sort();
    found
}

fn collect(directory: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name.starts_with('.') {
                continue;
            }
            collect(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
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
pub const fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// `text` with line comments, block comments and string literals blanked out.
///
/// Blanked rather than removed so every reported byte offset still maps to the
/// line it came from. Without this a brace inside a doc comment desynchronises
/// the brace matcher for the rest of the file, and the scanner reports whatever
/// happens to follow.
pub fn mask_comments_and_strings(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < bytes.len() {
        let rest = &text[index..];
        if rest.starts_with("//") {
            let end = rest.find('\n').map_or(bytes.len(), |offset| index + offset);
            out.push_str(&" ".repeat(end - index));
            index = end;
        } else if rest.starts_with("/*") {
            let end = rest.find("*/").map_or(bytes.len(), |offset| index + offset + 2);
            for byte in &bytes[index..end] {
                out.push(if *byte == b'\n' { '\n' } else { ' ' });
            }
            index = end;
        } else if bytes[index] == b'"' {
            let mut cursor = index + 1;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'\\' => cursor += 2,
                    b'"' => break,
                    _ => cursor += 1,
                }
            }
            let end = (cursor + 1).min(bytes.len());
            for byte in &bytes[index..end] {
                out.push(if *byte == b'\n' { '\n' } else { ' ' });
            }
            index = end;
        } else {
            let character = text[index..].chars().next().unwrap_or(' ');
            out.push(character);
            index += character.len_utf8();
        }
    }
    out
}
