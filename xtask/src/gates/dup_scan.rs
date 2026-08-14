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

use crate::gates::gates::workspace_root;

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

/// Files a shingle was seen in, held inline so the index allocates nothing.
///
/// Four is enough to name a copy's partners in a report. Truncation cannot
/// change any count: a file missing from a full list is a file the list already
/// proved shares with four others.
const OCCUPANT_CAP: usize = 4;

#[derive(Clone, Copy)]
struct Occupants {
    files: [u32; OCCUPANT_CAP],
    len: u8,
}

impl Occupants {
    fn new(file: u32) -> Self {
        let mut files = [0; OCCUPANT_CAP];
        files[0] = file;
        Self { files, len: 1 }
    }

    fn insert(&mut self, file: u32) {
        let len = usize::from(self.len);
        if len == OCCUPANT_CAP || self.files[..len].contains(&file) {
            return;
        }
        self.files[len] = file;
        self.len += 1;
    }

    fn files(&self) -> &[u32] {
        &self.files[..usize::from(self.len)]
    }
}

/// One file's share of the duplication, and what it duplicates against.
#[derive(Debug, Clone)]
pub(crate) struct FileReport {
    /// Workspace-relative path.
    pub(crate) path: String,
    /// Normalized lines covered by a shingle that also appears in another file.
    pub(crate) duplicate_lines: usize,
    /// Normalized lines in the file.
    pub(crate) total_lines: usize,
    /// Files sharing the most shingles with this one, highest first.
    pub(crate) partners: Vec<(String, usize)>,
}

/// Attribute a crate's duplication to individual files and their partners.
///
/// `measure` answers whether a crate regressed. It cannot answer which copy to
/// collapse, which is the only question a failing pin actually raises. The
/// per-file totals here sum to the same crate figure `measure` reports.
#[must_use]
pub(crate) fn report(root: &Path, only: Option<&str>) -> Vec<FileReport> {
    let mut normalized: Vec<(String, String, Vec<String>)> = Vec::new();
    for path in source_files(root) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Some(owner) = crate_of(root, &path) else {
            continue;
        };
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        normalized.push((owner, relative, normalize(&text)));
    }

    let mut index: HashMap<u64, Occupants> = HashMap::new();
    for (position, (_, _, lines)) in normalized.iter().enumerate() {
        let id = u32::try_from(position).unwrap_or(u32::MAX);
        for window in lines.windows(SHINGLE) {
            index
                .entry(hash(window))
                .and_modify(|occupants| occupants.insert(id))
                .or_insert_with(|| Occupants::new(id));
        }
    }

    let mut reports: Vec<FileReport> = Vec::new();
    for (position, (owner, relative, lines)) in normalized.iter().enumerate() {
        if only.is_some_and(|name| name != owner) {
            continue;
        }
        let id = u32::try_from(position).unwrap_or(u32::MAX);
        let mut duplicated: HashSet<usize> = HashSet::new();
        let mut partners: HashMap<u32, usize> = HashMap::new();
        for (start, window) in lines.windows(SHINGLE).enumerate() {
            let Some(occupants) = index.get(&hash(window)) else {
                continue;
            };
            let mut shared = false;
            for other in occupants.files() {
                if *other != id {
                    shared = true;
                    *partners.entry(*other).or_default() += 1;
                }
            }
            if shared {
                for offset in 0..SHINGLE {
                    duplicated.insert(start + offset);
                }
            }
        }
        if duplicated.is_empty() {
            continue;
        }
        let mut ranked: Vec<(String, usize)> = partners
            .into_iter()
            .map(|(other, shared)| {
                let path = normalized
                    .get(other as usize)
                    .map_or_else(String::new, |entry| entry.1.clone());
                (path, shared)
            })
            .collect();
        ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        ranked.truncate(3);
        reports.push(FileReport {
            path: relative.clone(),
            duplicate_lines: duplicated.len(),
            total_lines: lines.len(),
            partners: ranked,
        });
    }
    reports.sort_by(|left, right| {
        right
            .duplicate_lines
            .cmp(&left.duplicate_lines)
            .then_with(|| left.path.cmp(&right.path))
    });
    reports
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

    if let Some(position) = args.iter().position(|argument| argument == "--report") {
        let only = args
            .get(position + 1)
            .filter(|value| !value.starts_with("--"))
            .map(String::as_str);
        let reports = report(&root, only);
        let scope = only.unwrap_or("the workspace");
        let total: usize = reports.iter().map(|entry| entry.duplicate_lines).sum();
        println!(
            "dup-scan report: {total} duplicated lines across {} file(s) in {scope}",
            reports.len()
        );
        for entry in reports.iter().take(40) {
            println!(
                "  {:>6} of {:>6} lines  {}",
                entry.duplicate_lines, entry.total_lines, entry.path
            );
            for (partner, shared) in &entry.partners {
                println!("           shares {shared} shingle(s) with {partner}");
            }
        }
        return;
    }

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

    /// WHY: a failing pin is only actionable if the report names the other file,
    /// so the partner path is the contract, not the count beside it.
    #[test]
    fn the_report_names_the_file_a_copy_was_made_from() {
        let dir = std::env::temp_dir().join("vyre-dup-scan-report");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let block: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), &block).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &block).expect("write");

        let reports = report(&dir, Some("crate-a"));
        assert_eq!(reports.len(), 1, "only the filtered crate is reported");
        assert_eq!(reports[0].path, "crate-a/src/lib.rs");
        assert_eq!(reports[0].duplicate_lines, SHINGLE);
        assert_eq!(
            reports[0].partners,
            vec![("crate-b/src/lib.rs".to_string(), 1)],
            "the report must name where the copy lives"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// WHY: the report exists to explain a `measure` failure. If the two
    /// disagree it sends the owner at the wrong file, so they are pinned to each
    /// other rather than each being checked alone.
    #[test]
    fn per_file_duplication_sums_to_the_crate_measure() {
        let dir = std::env::temp_dir().join("vyre-dup-scan-agree");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let shared: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        let unique: String = (0..SHINGLE).map(|n| format!("let u{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/one.rs"), &shared).expect("write");
        fs::write(dir.join("crate-a/src/two.rs"), format!("{unique}{shared}")).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &shared).expect("write");

        let measured = measure(&dir)["crate-a"].duplicate_lines;
        let reported: usize = report(&dir, Some("crate-a"))
            .iter()
            .map(|entry| entry.duplicate_lines)
            .sum();
        assert_eq!(reported, measured, "report must explain the measured figure");
        let _ = fs::remove_dir_all(&dir);
    }
}
