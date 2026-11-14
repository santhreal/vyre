//! Zero-copy byte-level scanner over Rust source files.

use std::num::NonZeroUsize;
use walkdir::DirEntry;

/// Directories skipped during workspace source walks.
pub fn is_pruned(entry: &DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    if entry.depth() == 0 {
        return false;
    }
    let Some(name) = entry.file_name().to_str() else {
        return true;
    };
    name.starts_with('.') || matches!(name, "target" | "node_modules")
}

/// If `index` starts a comment, string literal, raw string literal, or char literal in `text`,
/// returns the non-zero byte length of that opaque span. Operates directly on `&[u8]`.
#[must_use]
pub fn opaque_span(text: &str, index: usize) -> Option<NonZeroUsize> {
    let bytes = text.as_bytes();
    if index >= bytes.len() {
        return None;
    }
    let end = match bytes[index] {
        b'/' if bytes.get(index + 1) == Some(&b'/') => {
            let rel = bytes[index + 2..].iter().position(|&b| b == b'\n');
            Some(rel.map_or(bytes.len(), |p| index + 2 + p))
        }
        b'/' if bytes.get(index + 1) == Some(&b'*') => {
            let mut depth = 1usize;
            let mut i = index + 2;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            Some(if depth == 0 { i } else { bytes.len() })
        }
        b'"' => Some(scan_quoted_string(bytes, index + 1)),
        b'b' if bytes.get(index + 1) == Some(&b'"') => Some(scan_quoted_string(bytes, index + 2)),
        b'r' | b'b' => scan_raw_string(bytes, index),
        b'\'' => scan_char_literal(bytes, index),
        _ => None,
    }?;
    NonZeroUsize::new(end.saturating_sub(index))
}

fn scan_quoted_string(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i = (i + 2).min(bytes.len()),
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    bytes.len()
}

fn scan_raw_string(bytes: &[u8], index: usize) -> Option<usize> {
    let mut i = index;
    if bytes.get(i) == Some(&b'b') {
        i += 1;
    }
    if bytes.get(i) != Some(&b'r') {
        return None;
    }
    i += 1;
    let mut hashes = 0usize;
    while bytes.get(i) == Some(&b'#') {
        hashes += 1;
        i += 1;
    }
    if bytes.get(i) != Some(&b'"') {
        return None;
    }
    i += 1;
    while i < bytes.len() {
        if bytes[i] == b'"'
            && i + 1 + hashes <= bytes.len()
            && bytes[i + 1..i + 1 + hashes].iter().all(|&b| b == b'#')
        {
            return Some(i + 1 + hashes);
        }
        i += 1;
    }
    Some(bytes.len())
}

fn scan_char_literal(bytes: &[u8], index: usize) -> Option<usize> {
    let after = index + 1;
    if after >= bytes.len() {
        return None;
    }
    if bytes[after] == b'\\' {
        let mut i = after + 2;
        while i < bytes.len() && i <= after + 10 {
            if bytes[i] == b'\'' {
                return Some(i + 1);
            }
            i += 1;
        }
        return None;
    }
    if bytes.get(after + 1) == Some(&b'\'') {
        return Some(after + 2);
    }
    let rest = &bytes[after..];
    if let Ok(s) = std::str::from_utf8(rest) {
        if let Some(ch) = s.chars().next() {
            let close = after + ch.len_utf8();
            if bytes.get(close) == Some(&b'\'') {
                return Some(close + 1);
            }
        }
    }
    None
}

/// Iterator over byte offsets in `text` that lie outside comments and string/char literals.
pub fn code_offsets(text: &str) -> impl Iterator<Item = usize> + '_ {
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        while pos < bytes.len() {
            if let Some(len) = opaque_span(text, pos) {
                pos += len.get();
                continue;
            }
            let current = pos;
            let ch_len = text[pos..].chars().next().map_or(1, char::len_utf8);
            pos += ch_len;
            return Some(current);
        }
        None
    })
}
