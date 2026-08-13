//! Cross-file duplicate source measurement, pinned per crate.
//!
//! `whats-similar` and `lego-audit` compare REGISTERED operations by IR
//! fingerprint. That misses the way duplication actually arrives here: someone
//! copies a file, edits three lines, and never registers a second op. The
//! campaign baseline found 45,582 duplicated lines across 4,187 files that way,
//! and no gate could have caught a single one of them.
//!
//! The measure is deliberately crude and therefore stable: normalize away
//! blank lines and comments, cut every file into 8-line shingles, and count a
//! line as duplicated when a shingle covering it also appears in another file.
//! Eight lines is long enough that shared boilerplate such as a use block or a
//! derive list does not trip it, and short enough to catch a copied function.
//!
//! Per-crate counts are pinned in `xtask/dup-baseline.toml`. More duplication
//! than the pin fails. Less is reported so the owning PR can lower it, which is
//! how each dedup PR records what it actually removed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use serde::Deserialize;
use walkdir::WalkDir;

use crate::gates::workspace_root;

/// Shingle length in normalized lines.
const SHINGLE: usize = 8;

#[derive(Debug, Deserialize)]
struct Pin {
    name: String,
    duplicate_lines: usize,
}

#[derive(Debug, Default, Deserialize)]
struct BaselineFile {
    #[serde(default, rename = "crate")]
    crates: Vec<Pin>,
}

fn baseline_path(root: &Path) -> PathBuf {
    root.join("xtask/dup-baseline.toml")
}

/// One measured crate.
#[derive(Debug, Default, Clone)]
pub(crate) struct CrateCount {
    /// Normalized non-comment lines covered by a shingle seen in another file.
    pub(crate) duplicate_lines: usize,
    /// Normalized non-comment lines in the crate.
    pub(crate) total_lines: usize,
}

/// Strip comments and blank lines, leaving comparable code text.
///
/// Indentation is normalized away so that moving a block into a loop does not
/// read as new code, which would let a pure re-indent hide a copy.
fn normalize(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .map(str::to_string)
        .collect()
}

fn crate_of(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    relative
        .components()
        .next()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
}

fn source_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            name != "target" && name != ".git" && !name.starts_with(".cargo")
        })
        .flatten()
    {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path.to_path_buf());
        }
    }
    files.sort();
    files
}

/// Measure duplicated lines per crate across the workspace.
#[must_use]
pub(crate) fn measure(root: &Path) -> BTreeMap<String, CrateCount> {
    let files = source_files(root);
    let mut normalized: Vec<(usize, String, Vec<String>)> = Vec::new();
    for (index, path) in files.iter().enumerate() {
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        let Some(owner) = crate_of(root, path) else {
            continue;
        };
        normalized.push((index, owner, normalize(&text)));
    }

    // Shingle -> the distinct files it appears in, capped at two because that
    // is all the rule needs and it keeps the map small on a tree this size.
    let mut seen: HashMap<u64, (usize, bool)> = HashMap::new();
    for (index, _, lines) in &normalized {
        for window in lines.windows(SHINGLE) {
            let key = hash(window);
            seen.entry(key)
                .and_modify(|(first, shared)| {
                    if *first != *index {
                        *shared = true;
                    }
                })
                .or_insert((*index, false));
        }
    }

    let mut counts: BTreeMap<String, CrateCount> = BTreeMap::new();
    for (_, owner, lines) in &normalized {
        let entry = counts.entry(owner.clone()).or_default();
        entry.total_lines += lines.len();
        let mut duplicated: HashSet<usize> = HashSet::new();
        for (start, window) in lines.windows(SHINGLE).enumerate() {
            if seen.get(&hash(window)).is_some_and(|(_, shared)| *shared) {
                for offset in 0..SHINGLE {
                    duplicated.insert(start + offset);
                }
            }
        }
        entry.duplicate_lines += duplicated.len();
    }
    counts
}

fn hash(window: &[String]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for line in window {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    let bytes = hasher.finalize();
    u64::from_le_bytes(bytes.as_bytes()[..8].try_into().unwrap_or([0; 8]))
}

fn render(counts: &BTreeMap<String, CrateCount>) -> String {
    let mut text = String::from(
        "# Duplicated source lines per crate, written by `xtask dup-scan --write-baseline`.\n\
         #\n\
         # A line counts as duplicated when an 8-line normalized shingle covering it\n\
         # also appears in another file. More than the pin fails; less is reported so\n\
         # the owning dedup PR can lower it here as part of its own diff.\n",
    );
    for (name, count) in counts {
        if count.duplicate_lines == 0 {
            continue;
        }
        text.push_str("\n[[crate]]\n");
        text.push_str(&format!("name = \"{name}\"\n"));
        text.push_str(&format!("duplicate_lines = {}\n", count.duplicate_lines));
        text.push_str(&format!("total_lines = {}\n", count.total_lines));
    }
    text
}

/// Run the duplicate scan.
pub(crate) fn run(args: &[String]) {
    let root = workspace_root();
    let counts = measure(&root);

    if args.iter().any(|argument| argument == "--write-baseline") {
        let path = baseline_path(&root);
        fs::write(&path, render(&counts)).unwrap_or_else(|error| {
            eprintln!("Fix: cannot write {}: {error}", path.display());
            process::exit(1);
        });
        let total: usize = counts.values().map(|count| count.duplicate_lines).sum();
        println!("dup-scan: wrote {} ({total} duplicated lines)", path.display());
        return;
    }

    let path = baseline_path(&root);
    let text = fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!(
            "Fix: cannot read {}: {error}. Regenerate it with `xtask dup-scan --write-baseline`.",
            path.display()
        );
        process::exit(1);
    });
    let baseline: BaselineFile = toml::from_str(&text).unwrap_or_else(|error| {
        eprintln!("Fix: cannot parse {}: {error}", path.display());
        process::exit(1);
    });

    let mut failures = Vec::new();
    for pin in &baseline.crates {
        let Some(count) = counts.get(&pin.name) else {
            failures.push(format!(
                "xtask/dup-baseline.toml pins `{}`, which is not a directory in the workspace",
                pin.name
            ));
            continue;
        };
        if count.duplicate_lines > pin.duplicate_lines {
            failures.push(format!(
                "`{}` has {} duplicated lines against a pinned {}; collapse the new copy",
                pin.name, count.duplicate_lines, pin.duplicate_lines
            ));
        } else if count.duplicate_lines < pin.duplicate_lines {
            println!(
                "{}: {} duplicated lines, improved from {}; lower the pin",
                pin.name, count.duplicate_lines, pin.duplicate_lines
            );
        }
    }
    for (name, count) in &counts {
        if count.duplicate_lines > 0 && !baseline.crates.iter().any(|pin| &pin.name == name) {
            failures.push(format!(
                "`{name}` has {} duplicated lines and no pin; add one to xtask/dup-baseline.toml",
                count.duplicate_lines
            ));
        }
    }

    if !failures.is_empty() {
        eprintln!("dup-scan: {} failure(s):", failures.len());
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        process::exit(1);
    }
    let total: usize = counts.values().map(|count| count.duplicate_lines).sum();
    println!("dup-scan: {total} duplicated lines, all within their pins");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: comments and indentation are exactly what a copy-paste edits first,
    /// so normalization must see through both or the scan misses real clones.
    #[test]
    fn normalization_drops_comments_blanks_and_indentation() {
        let lines = normalize("    let a = 1;\n\n// explanation\n        let a = 1;\n");
        assert_eq!(lines, vec!["let a = 1;", "let a = 1;"]);
    }

    /// WHY: the whole measure rests on a shingle matching across files and not
    /// within one, or every long file would report itself as duplicated.
    #[test]
    fn a_block_repeated_inside_one_file_is_not_cross_file_duplication() {
        let dir = std::env::temp_dir().join("vyre-dup-scan-single");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        let block: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), format!("{block}{block}")).expect("write");

        let counts = measure(&dir);
        assert_eq!(
            counts["crate-a"].duplicate_lines, 0,
            "a repeat inside one file is not a cross-file clone"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// WHY: this is the case the campaign exists for, a copied file in a second
    /// crate. It must be counted in both crates.
    #[test]
    fn the_same_block_in_two_crates_is_counted_in_both() {
        let dir = std::env::temp_dir().join("vyre-dup-scan-pair");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let block: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), &block).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &block).expect("write");

        let counts = measure(&dir);
        assert_eq!(counts["crate-a"].duplicate_lines, SHINGLE);
        assert_eq!(counts["crate-b"].duplicate_lines, SHINGLE);
        let _ = fs::remove_dir_all(&dir);
    }

    /// WHY: a shorter shared run is boilerplate, not a copied implementation.
    /// Counting it would drown the real findings.
    #[test]
    fn a_shared_run_shorter_than_the_shingle_is_not_duplication() {
        let dir = std::env::temp_dir().join("vyre-dup-scan-short");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let block: String = (0..SHINGLE - 1).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), &block).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &block).expect("write");

        let counts = measure(&dir);
        assert_eq!(counts["crate-a"].duplicate_lines, 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
