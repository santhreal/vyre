//! Cross-file duplication budget for the two operation families that were
//! de-duplicated together: the neural-network dialect under `src/nn` (attention,
//! norm, linear, activation) and the classic Aho-Corasick dialect under
//! `src/scan/classic_ac`.
//!
//! # What this gate owns
//!
//! It owns the *class*: "an eight-line-or-longer run of normalized source lines
//! exists verbatim in two different files of these families". Every member is
//! discovered by walking the two directory roots at run time, so a new sibling
//! module, a new attention variant, or a new prefilter shape is measured the
//! moment it lands, with nobody editing a list here. The floors below fail the
//! suite when the walk stops finding the tree, so a moved directory cannot turn
//! this into a vacuous pass.
//!
//! # What it deliberately does not own
//!
//! It does not read the source for any particular spelling: no function name,
//! type name, or helper name appears in it, so an equivalent copy under a
//! different set of identifiers is caught exactly as well as a literal one.
//! It says nothing about duplication *inside* one file, nor about duplication
//! against files outside these two roots, and it does not judge whether a given
//! block ought to be shared. A differential test's two independent arms, a
//! per-shape capability constant, and a public ABI parameter list are all
//! legitimately repeated shapes; the budgets are pinned at what the tree
//! measures with those left in place, so the gate fires on growth rather than on
//! their existence.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Shingle width, matching `xtask/src/dup_scan.rs`: eight consecutive
/// normalized lines are the smallest run this treats as a copy.
const SHINGLE: usize = 8;

/// Directory roots, relative to the crate manifest, whose files are compared
/// against each other.
const ROOTS: [&str; 2] = ["src/nn", "src/scan/classic_ac"];

/// Anti-vacuity floor on files discovered by the walk. Well below the current
/// count so ordinary growth or a merged deletion does not trip it, and far
/// enough above zero that a broken root path fails instead of passing.
const FILE_FLOOR: usize = 90;

/// Anti-vacuity floor on shingles indexed, so a walk that finds the files but
/// reads them as empty also fails.
const SHINGLE_FLOOR: usize = 15_000;

/// Longest permitted contiguous cross-file block, in normalized lines.
///
/// Pinned at the longest block that survives on purpose: the 21-line
/// `GatedDeltaSpec` field destructure, which is a restatement of that type's own
/// field list and has no owner short of a macro.
const BLOCK_CAP: usize = 21;

/// Total permitted normalized lines that participate in any cross-file block,
/// summed over both roots.
const DUPLICATED_LINE_BUDGET: usize = 1_673;

/// Permitted number of distinct file pairs that share at least one eight-line
/// run. A file that starts copying from a new partner adds a pair, so this fires
/// even when the copy is short enough to clear [`BLOCK_CAP`].
const PAIR_BUDGET: usize = 248;

/// One source file reduced to the lines `xtask`'s duplication scanner counts:
/// trimmed, with blank lines and comment lines dropped.
struct Normalized {
    path: PathBuf,
    lines: Vec<String>,
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => panic!("Fix: duplication gate cannot read {}: {err}", dir.display()),
    };
    for entry in entries {
        let path = entry.expect("Fix: directory entry must be readable").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn normalize(path: PathBuf) -> Normalized {
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {}: {err}", path.display()));
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .map(str::to_owned)
        .collect();
    Normalized { path, lines }
}

fn families() -> Vec<Normalized> {
    let root = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"));
    let mut paths = Vec::new();
    for relative in ROOTS {
        collect_rs(&root.join(relative), &mut paths);
    }
    paths.sort();
    paths.into_iter().map(normalize).collect()
}

/// Which files, other than `file`, contain the shingle starting at each index.
type SharedStarts = HashMap<usize, Vec<usize>>;

/// Cross-file shingle occupancy over the walked files.
struct Occupancy {
    files: Vec<Normalized>,
    shingles: usize,
    /// `shared[file] [start]` lists the peer file indices sharing that shingle.
    shared: Vec<SharedStarts>,
}

fn occupancy() -> Occupancy {
    let files = families();
    let mut index: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    let mut shingles = 0;
    for (f, file) in files.iter().enumerate() {
        for start in 0..file.lines.len().saturating_sub(SHINGLE - 1) {
            let key = file.lines[start..start + SHINGLE].join("\n");
            index.entry(key).or_default().push((f, start));
            shingles += 1;
        }
    }
    let mut shared: Vec<SharedStarts> = vec![SharedStarts::new(); files.len()];
    for occurrences in index.values() {
        let mut distinct: Vec<usize> = occurrences.iter().map(|&(f, _)| f).collect();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < 2 {
            continue;
        }
        for &(f, start) in occurrences {
            let peers: Vec<usize> = distinct.iter().copied().filter(|&p| p != f).collect();
            shared[f].insert(start, peers);
        }
    }
    Occupancy {
        files,
        shingles,
        shared,
    }
}

/// The longest maximal run of consecutive shared shingles, as
/// `(normalized_line_count, offender_description)`.
fn longest_block(occupancy: &Occupancy) -> (usize, String) {
    let mut best = (0usize, String::from("none"));
    for (f, starts) in occupancy.shared.iter().enumerate() {
        let mut sorted: Vec<usize> = starts.keys().copied().collect();
        sorted.sort_unstable();
        let mut i = 0;
        while i < sorted.len() {
            let mut j = i;
            while j + 1 < sorted.len() && sorted[j + 1] == sorted[j] + 1 {
                j += 1;
            }
            let length = sorted[j] - sorted[i] + SHINGLE;
            if length > best.0 {
                let peers: Vec<&str> = starts[&sorted[i]]
                    .iter()
                    .map(|&p| {
                        occupancy.files[p]
                            .path
                            .to_str()
                            .expect("Fix: path must be UTF-8")
                    })
                    .collect();
                best = (
                    length,
                    format!(
                        "{} (normalized line {}) shared with {}",
                        occupancy.files[f].path.display(),
                        sorted[i] + 1,
                        peers.join(", ")
                    ),
                );
            }
            i = j + 1;
        }
    }
    best
}

/// The file pairs contributing the most duplicated normalized lines, worst
/// first, so a budget failure names the copy that grew rather than whichever
/// pre-existing block happens to be the longest.
fn worst_pairs(occupancy: &Occupancy, keep: usize) -> String {
    let mut totals: HashMap<(usize, usize), usize> = HashMap::new();
    for (f, starts) in occupancy.shared.iter().enumerate() {
        let mut per_peer: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&start, peers) in starts {
            for &peer in peers {
                per_peer
                    .entry(peer)
                    .or_default()
                    .extend(start..start + SHINGLE);
            }
        }
        for (peer, mut covered) in per_peer {
            covered.sort_unstable();
            covered.dedup();
            let key = if f < peer { (f, peer) } else { (peer, f) };
            *totals.entry(key).or_default() += covered.len();
        }
    }
    let mut ranked: Vec<((usize, usize), usize)> = totals.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked
        .into_iter()
        .take(keep)
        .map(|((a, b), lines)| {
            format!(
                "{lines} lines: {} <-> {}",
                occupancy.files[a].path.display(),
                occupancy.files[b].path.display()
            )
        })
        .collect::<Vec<String>>()
        .join("; ")
}

fn duplicated_lines(occupancy: &Occupancy) -> usize {
    occupancy
        .shared
        .iter()
        .map(|starts| {
            let mut covered: Vec<usize> = starts
                .keys()
                .flat_map(|&start| start..start + SHINGLE)
                .collect();
            covered.sort_unstable();
            covered.dedup();
            covered.len()
        })
        .sum()
}

#[test]
fn family_walk_finds_the_tree() {
    let occupancy = occupancy();
    assert!(
        occupancy.files.len() >= FILE_FLOOR,
        "duplication gate walked only {} files under {ROOTS:?}; below the {FILE_FLOOR} floor, so \
         it can no longer see the families it is supposed to guard",
        occupancy.files.len()
    );
    assert!(
        occupancy.shingles >= SHINGLE_FLOOR,
        "duplication gate indexed only {} shingles; below the {SHINGLE_FLOOR} floor, so the walk \
         found files but not their contents",
        occupancy.shingles
    );
}

#[test]
fn no_cross_file_block_exceeds_the_cap() {
    let occupancy = occupancy();
    let (length, offender) = longest_block(&occupancy);
    assert!(
        length <= BLOCK_CAP,
        "cross-file duplicated block of {length} normalized lines exceeds the {BLOCK_CAP}-line \
         cap: {offender}. Give the block one owner, or state why the two positions must differ."
    );
}

#[test]
fn duplicated_line_budget_holds() {
    let occupancy = occupancy();
    let total = duplicated_lines(&occupancy);
    assert!(
        total <= DUPLICATED_LINE_BUDGET,
        "{total} normalized lines under {ROOTS:?} participate in a cross-file duplicated block, \
         over the {DUPLICATED_LINE_BUDGET}-line budget. Worst pairs: {}",
        worst_pairs(&occupancy, 5)
    );
}

/// Every distinct pair of files sharing at least one shingle, crate-relative and
/// sorted, so a newly duplicating file is named by the failure itself.
fn duplicated_pairs(occupancy: &Occupancy) -> Vec<String> {
    let root = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"));
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (f, starts) in occupancy.shared.iter().enumerate() {
        for peers in starts.values() {
            for &peer in peers {
                if f < peer {
                    pairs.push((f, peer));
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    let name = |i: usize| {
        occupancy.files[i]
            .path
            .strip_prefix(&root)
            .unwrap_or(&occupancy.files[i].path)
            .display()
            .to_string()
    };
    pairs
        .into_iter()
        .map(|(a, b)| format!("{} <-> {}", name(a), name(b)))
        .collect()
}

#[test]
fn no_new_file_pair_starts_duplicating() {
    let occupancy = occupancy();
    let pairs = duplicated_pairs(&occupancy);
    assert!(
        pairs.len() <= PAIR_BUDGET,
        "{} file pairs under {ROOTS:?} share at least one eight-line run, over the \
         {PAIR_BUDGET}-pair budget. Full list, newest offender included: {}",
        pairs.len(),
        pairs.join("; ")
    );
}
