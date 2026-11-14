//! Zero-copy 8-line FNV-1a window duplication scanner across workspace Rust sources.

use std::collections::HashMap;
use std::path::Path;

const WINDOW_LINES: usize = 8;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Summary of normalized 8-line duplicate windows across non-test Rust sources.
#[derive(Clone, Copy, Debug)]
pub struct DupReport {
    /// Total non-trivial normalized source lines scanned.
    pub total_lines: usize,
    /// Lines participating in a window duplicated in another file.
    pub duplicate_lines: usize,
    /// Percentage `100 * duplicate_lines / total_lines`.
    pub ratio_pct: f64,
}

fn hash_line(trimmed: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in trimmed {
        if !b.is_ascii_whitespace() {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

fn is_trivial(trimmed: &[u8]) -> bool {
    trimmed.len() < 6
        || trimmed.starts_with(b"//")
        || trimmed.starts_with(b"use ")
        || trimmed.starts_with(b"pub use ")
        || trimmed.starts_with(b"mod ")
        || trimmed.starts_with(b"pub mod ")
        || trimmed.starts_with(b"#[")
        || matches!(trimmed, b"{" | b"}" | b"};" | b");" | b"else {")
}

/// Compute duplication statistics across production `src/` Rust files.
#[must_use]
pub fn scan(root: &Path) -> DupReport {
    let roster = crate::workspace_manifest::source_file_roster(root);
    let mut window_owner: HashMap<u64, u32> = HashMap::with_capacity(65_536);
    let mut dup_windows: HashMap<u64, ()> = HashMap::new();
    let mut per_file_hashes: Vec<(u32, Vec<u64>)> = Vec::new();
    let mut total_lines = 0usize;

    for (file_idx, rel) in roster.iter().enumerate() {
        if !rel.contains("/src/") || rel.ends_with("/tests.rs") {
            continue;
        }
        let Ok(text) = crate::workspace_manifest::read_source_bounded(&root.join(rel)) else {
            continue;
        };
        let mut line_hashes = Vec::new();
        for line in text.as_bytes().split(|&b| b == b'\n') {
            let trimmed = trim_ascii(line);
            if is_trivial(trimmed) {
                continue;
            }
            line_hashes.push(hash_line(trimmed));
        }
        total_lines += line_hashes.len();
        if line_hashes.len() >= WINDOW_LINES {
            let fid = file_idx as u32;
            let mut win_hashes = Vec::with_capacity(line_hashes.len() - WINDOW_LINES + 1);
            for win in line_hashes.windows(WINDOW_LINES) {
                let mut wh = FNV_OFFSET;
                for &lh in win {
                    wh ^= lh;
                    wh = wh.wrapping_mul(FNV_PRIME);
                }
                win_hashes.push(wh);
                match window_owner.get(&wh) {
                    None => {
                        window_owner.insert(wh, fid);
                    }
                    Some(&prev_fid) if prev_fid != fid => {
                        dup_windows.insert(wh, ());
                    }
                    Some(_) => {}
                }
            }
            per_file_hashes.push((fid, win_hashes));
        }
    }

    let mut duplicate_lines = 0usize;
    for (_fid, win_hashes) in &per_file_hashes {
        let len = win_hashes.len() + WINDOW_LINES - 1;
        let mut mask = vec![false; len];
        for (i, wh) in win_hashes.iter().enumerate() {
            if dup_windows.contains_key(wh) {
                for slot in &mut mask[i..i + WINDOW_LINES] {
                    *slot = true;
                }
            }
        }
        duplicate_lines += mask.iter().filter(|&&b| b).count();
    }

    let ratio_pct = if total_lines == 0 {
        0.0
    } else {
        (duplicate_lines as f64) * 100.0 / (total_lines as f64)
    };
    DupReport {
        total_lines,
        duplicate_lines,
        ratio_pct,
    }
}

fn trim_ascii(mut s: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = s.split_first() {
        if first.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    while let Some((&last, rest)) = s.split_last() {
        if last.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    s
}

/// Gate entry point enforcing cross-file duplication ceiling (< 6.0%).
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let report = scan(root);
    if report.ratio_pct > 6.0 {
        Err(vec![format!(
            "cross-file duplicate line ratio {:.2}% ({}/{}) exceeds 6.00% ceiling",
            report.ratio_pct, report.duplicate_lines, report.total_lines
        )])
    } else {
        Ok(())
    }
}
