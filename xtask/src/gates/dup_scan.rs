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

use serde::Deserialize;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan;

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

/// The crate a repository-relative path belongs to.
fn crate_of(path: &Path) -> Option<String> {
    path.components()
        .next()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
}

/// Measure duplicated lines per crate across the workspace.
///
/// The file set is what git would commit: tracked files plus untracked files no
/// ignore rule excludes, which is [`scan::Tree`]. A working tree also holds
/// scratch that git ignores, and counting it made the measurement local.
/// Twenty-two `.rs` files ignored by one rule were once counted into their
/// crates' totals here, so a pin recorded on a workstation described that
/// workstation while CI measured a smaller tree, which is the direction that
/// lets a gate pass by accident.
#[must_use]
pub(crate) fn measure(root: &Path) -> Result<BTreeMap<String, CrateCount>, GateError> {
    let tree = scan::Tree::open(root)?;
    let files = tree.all_rust();
    let mut normalized: Vec<(usize, String, Vec<String>)> = Vec::new();
    for (index, path) in files.iter().enumerate() {
        let Ok(text) = fs::read_to_string(tree.absolute(path)) else {
            continue;
        };
        let Some(owner) = crate_of(path) else {
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
    Ok(counts)
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
pub(crate) fn report_for(root: &Path, only: Option<&str>) -> Result<Vec<FileReport>, GateError> {
    let mut normalized: Vec<(String, String, Vec<String>)> = Vec::new();
    let tree = scan::Tree::open(root)?;
    for path in tree.all_rust() {
        let Ok(text) = fs::read_to_string(tree.absolute(&path)) else {
            continue;
        };
        let Some(owner) = crate_of(&path) else {
            continue;
        };
        let relative = path.to_string_lossy().into_owned();
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
    Ok(reports)
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

/// Rewrite one crate's pin in place, leaving every other byte of the file alone.
///
/// Editing the text rather than serializing the parsed rows is the whole point:
/// the file's leading comment block records which pins are deliberately tight
/// and why neither side of a cross-crate pair can own the shape yet, and a
/// serializer has no field to put that in. A writer that reproduced the rows
/// and dropped the block destroyed the only record of the reasoning, which made
/// the one job this file has, recording progress, corrupt the record instead.
///
/// `None` when the file pins no such crate.
fn rewrite_pin(text: &str, name: &str, count: &CrateCount) -> Option<String> {
    let mut out = Vec::new();
    let mut in_block = false;
    let mut found = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "[[crate]]" {
            in_block = false;
        } else if let Some(value) = trimmed.strip_prefix("name = ") {
            in_block = value.trim_matches('"') == name;
            found |= in_block;
        }
        if in_block && trimmed.starts_with("duplicate_lines = ") {
            out.push(format!("duplicate_lines = {}", count.duplicate_lines));
            continue;
        }
        if in_block && trimmed.starts_with("total_lines = ") {
            out.push(format!("total_lines = {}", count.total_lines));
            continue;
        }
        out.push(line.to_string());
    }
    if !found {
        return None;
    }
    let mut text = out.join("\n");
    text.push('\n');
    Some(text)
}

/// Insert a pin for a crate the file does not measure yet, in name order.
///
/// Rows are sorted by name so a new crate lands where a reader looks for it
/// instead of at the end, and the file stays a diffable record.
fn insert_pin(text: &str, name: &str, count: &CrateCount) -> String {
    let row = format!(
        "[[crate]]\nname = \"{name}\"\nduplicate_lines = {}\ntotal_lines = {}\n",
        count.duplicate_lines, count.total_lines
    );
    let lines: Vec<&str> = text.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        if line.trim() != "[[crate]]" {
            continue;
        }
        let existing = lines
            .get(index + 1)
            .and_then(|next| next.trim().strip_prefix("name = "))
            .map(|value| value.trim_matches('"'))
            .unwrap_or("");
        if existing > name {
            let mut out = lines[..index].join("\n");
            out.push('\n');
            out.push_str(&row);
            out.push('\n');
            out.push_str(&lines[index..].join("\n"));
            out.push('\n');
            return out;
        }
    }
    let mut out = text.to_string();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(&row);
    out
}

/// Read the pinned baseline, naming the file that could not be read.
fn read_baseline(path: &Path) -> Result<(String, BaselineFile), GateError> {
    let text = fs::read_to_string(path).map_err(|error| {
        GateError::new(
            format!("cannot read {}: {error}", path.display()),
            "restore the duplication baseline, or record one with `xtask dup-scan --write`",
        )
    })?;
    let baseline: BaselineFile = toml::from_str(&text).map_err(|error| {
        GateError::new(
            format!("cannot parse {}: {error}", path.display()),
            "repair the baseline syntax",
        )
    })?;
    Ok((text, baseline))
}

/// Record every crate the baseline does not pin yet, and nothing else.
///
/// No bulk operation lowers a pin. A pin below its measurement is closing work
/// some other checkout owns, and rewriting all 34 rows from one tree erased the
/// intentional-red state those pins exist to hold. Lowering is `--lower-pin`,
/// one crate at a time, in the diff that removed the duplication.
fn write_baseline(
    root: &Path,
    counts: &BTreeMap<String, CrateCount>,
    report: &mut Report,
) -> Result<(), GateError> {
    let path = baseline_path(root);
    let (mut text, baseline) = read_baseline(&path)?;
    let mut added = Vec::new();
    for (name, count) in counts {
        if count.duplicate_lines == 0 {
            continue;
        }
        match baseline.crates.iter().find(|pin| &pin.name == name) {
            None => {
                text = insert_pin(&text, name, count);
                added.push(name.clone());
            }
            Some(pin) if count.duplicate_lines < pin.duplicate_lines => {
                report.note(format!(
                    "{name}: {} measured against a pinned {}; lower it with `dup-scan --lower-pin {name}`",
                    count.duplicate_lines, pin.duplicate_lines
                ));
            }
            Some(_) => {}
        }
    }
    fs::write(&path, &text).map_err(|error| {
        GateError::new(
            format!("cannot write {}: {error}", path.display()),
            "make the duplication baseline writable",
        )
    })?;
    report.note(format!("pinned {} newly measured crate(s)", added.len()));
    for name in &added {
        report.note(format!("pinned {name}"));
    }
    Ok(())
}

/// Lower one crate's pin to what this tree measures.
///
/// Refuses to raise: a measurement above the pin is a regression to collapse,
/// and the one thing this file must never do is move a pin up to make a
/// regression pass.
fn lower_pin(
    root: &Path,
    name: &str,
    counts: &BTreeMap<String, CrateCount>,
    report: &mut Report,
) -> Result<(), GateError> {
    let path = baseline_path(root);
    let (text, baseline) = read_baseline(&path)?;
    let Some(pin) = baseline.crates.iter().find(|pin| pin.name == name) else {
        return Err(GateError::new(
            format!("{} pins no crate named `{name}`", path.display()),
            "name a crate the baseline pins",
        ));
    };
    let Some(count) = counts.get(name) else {
        return Err(GateError::new(
            format!("`{name}` is not a measured directory in this workspace"),
            "name a directory the scan measures",
        ));
    };
    if count.duplicate_lines > pin.duplicate_lines {
        report.find(Finding::new(
            format!(
                "`{name}` measures {} duplicated lines against a pinned {}",
                count.duplicate_lines, pin.duplicate_lines
            ),
            "collapse the new copy; a pin never moves up",
        ));
        return Ok(());
    }
    if count.duplicate_lines == pin.duplicate_lines {
        report.note(format!(
            "`{name}` is already pinned at {}",
            pin.duplicate_lines
        ));
        return Ok(());
    }
    let Some(updated) = rewrite_pin(&text, name, count) else {
        return Err(GateError::new(
            format!("{} has no editable row for `{name}`", path.display()),
            "restore the row this baseline should carry for that crate",
        ));
    };
    fs::write(&path, updated).map_err(|error| {
        GateError::new(
            format!("cannot write {}: {error}", path.display()),
            "make the duplication baseline writable",
        )
    })?;
    report.note(format!(
        "lowered `{name}` from {} to {}",
        pin.duplicate_lines, count.duplicate_lines
    ));
    Ok(())
}

/// Measures cross-file duplicate source blocks against the pinned per-crate baseline.
pub struct DupScan;

impl crate::gate::GateBehavior for DupScan {
    fn usage(&self) -> &'static [&'static str] {
        &[
            "--lower-pin CRATE lowers one pinned row to what the crate measures now",
            "--report [CRATE] lists the duplicated files instead of the pinned counts",
        ]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = &ctx.root;
        let mut report = Report::clean();
        report.produced("xtask/dup-baseline.toml");

        if let Some(position) = ctx.args.iter().position(|argument| argument == "--report") {
            let only = ctx
                .args
                .get(position + 1)
                .filter(|value| !value.starts_with("--"))
                .map(String::as_str);
            let reports = report_for(root, only)?;
            report.cover_complete("source files with duplication measurements", reports.len());
            let scope = only.unwrap_or("the workspace");
            let total: usize = reports.iter().map(|entry| entry.duplicate_lines).sum();
            report.note(format!(
                "{total} duplicated lines across {} file(s) in {scope}",
                reports.len()
            ));
            for entry in reports.iter().take(40) {
                report.note(format!(
                    "{:>6} of {:>6} lines  {}",
                    entry.duplicate_lines, entry.total_lines, entry.path
                ));
                for (partner, shared) in &entry.partners {
                    report.note(format!("  shares {shared} shingle(s) with {partner}"));
                }
            }
            return Ok(report);
        }

        let counts = measure(root)?;
        report.cover_complete("workspace crates", counts.len());

        if let Some(position) = ctx
            .args
            .iter()
            .position(|argument| argument == "--lower-pin")
        {
            let Some(name) = ctx
                .args
                .get(position + 1)
                .filter(|value| !value.starts_with("--"))
            else {
                return Err(GateError::new(
                    "`--lower-pin` was passed without a crate",
                    "name the crate whose row it lowers",
                ));
            };
            lower_pin(root, name, &counts, &mut report)?;
            return Ok(report);
        }

        if ctx.write {
            write_baseline(root, &counts, &mut report)?;
            return Ok(report);
        }

        let path = baseline_path(root);
        let (_, baseline) = read_baseline(&path)?;

        for pin in &baseline.crates {
            let Some(count) = counts.get(&pin.name) else {
                report.find(Finding::in_file(
                    "xtask/dup-baseline.toml",
                    format!(
                        "pins `{}`, which is not a directory in the workspace",
                        pin.name
                    ),
                    "delete the row, or restore the directory it pins",
                ));
                continue;
            };
            if count.duplicate_lines > pin.duplicate_lines {
                report.find(Finding::new(
                    format!(
                        "`{}` has {} duplicated lines against a pinned {}",
                        pin.name, count.duplicate_lines, pin.duplicate_lines
                    ),
                    "collapse the new copy into the primitive it duplicates",
                ));
            } else if count.duplicate_lines < pin.duplicate_lines {
                report.note(format!(
                    "{}: {} duplicated lines, improved from {}; record it with `dup-scan --lower-pin {}`",
                    pin.name, count.duplicate_lines, pin.duplicate_lines, pin.name
                ));
            }
        }
        for (name, count) in &counts {
            if count.duplicate_lines > 0 && !baseline.crates.iter().any(|pin| &pin.name == name) {
                report.find(Finding::new(
                    format!(
                        "`{name}` has {} duplicated lines and no pin",
                        count.duplicate_lines
                    ),
                    "record it with `xtask dup-scan --write`",
                ));
            }
        }
        let total: usize = counts.values().map(|count| count.duplicate_lines).sum();
        report.note(format!("{total} duplicated lines across the workspace"));
        Ok(report)
    }
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
        let dir = std::env::temp_dir().join(format!("vyre-dup-scan-single-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        crate::fixture_checkout::empty(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        let block: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), format!("{block}{block}")).expect("write");

        let counts = measure(&dir).expect("the fixture checkout is measurable");
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
        let dir = std::env::temp_dir().join(format!("vyre-dup-scan-pair-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        crate::fixture_checkout::empty(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let block: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), &block).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &block).expect("write");

        let counts = measure(&dir).expect("the fixture checkout is measurable");
        assert_eq!(counts["crate-a"].duplicate_lines, SHINGLE);
        assert_eq!(counts["crate-b"].duplicate_lines, SHINGLE);
        let _ = fs::remove_dir_all(&dir);
    }

    /// WHY: a shorter shared run is boilerplate, not a copied implementation.
    /// Counting it would drown the real findings.
    #[test]
    fn a_shared_run_shorter_than_the_shingle_is_not_duplication() {
        let dir = std::env::temp_dir().join(format!("vyre-dup-scan-short-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        crate::fixture_checkout::empty(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let block: String = (0..SHINGLE - 1)
            .map(|n| format!("let v{n} = {n};\n"))
            .collect();
        fs::write(dir.join("crate-a/src/lib.rs"), &block).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &block).expect("write");

        let counts = measure(&dir).expect("the fixture checkout is measurable");
        assert_eq!(counts["crate-a"].duplicate_lines, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// WHY: a failing pin is only actionable if the report names the other file,
    /// so the partner path is the contract, not the count beside it.
    #[test]
    fn the_report_names_the_file_a_copy_was_made_from() {
        let dir = std::env::temp_dir().join(format!("vyre-dup-scan-report-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        crate::fixture_checkout::empty(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let block: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), &block).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &block).expect("write");

        let reports =
            report_for(&dir, Some("crate-a")).expect("the fixture checkout is reportable");
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
        let dir = std::env::temp_dir().join(format!("vyre-dup-scan-agree-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        crate::fixture_checkout::empty(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/src")).expect("temp dir");
        let shared: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        let unique: String = (0..SHINGLE).map(|n| format!("let u{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/one.rs"), &shared).expect("write");
        fs::write(dir.join("crate-a/src/two.rs"), format!("{unique}{shared}")).expect("write");
        fs::write(dir.join("crate-b/src/lib.rs"), &shared).expect("write");

        let measured =
            measure(&dir).expect("the fixture checkout is measurable")["crate-a"].duplicate_lines;
        let reported: usize = report_for(&dir, Some("crate-a"))
            .expect("the fixture checkout is measurable")
            .iter()
            .map(|entry| entry.duplicate_lines)
            .sum();
        assert_eq!(
            reported, measured,
            "report must explain the measured figure"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// WHY: a pin has to mean the same thing on a workstation and in CI. A
    /// working tree carries scratch that git ignores, and counting it made the
    /// local figure larger than the one CI can measure, which is the direction
    /// that lets a pin pass by accident. This tree once carried twenty-two
    /// ignored `.rs` files that were counted into their crates' totals.
    #[test]
    fn a_file_the_repository_ignores_is_not_measured() {
        let dir =
            std::env::temp_dir().join(format!("vyre-dup-scan-ignored-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        crate::fixture_checkout::empty(&dir);
        fs::create_dir_all(dir.join("crate-a/src")).expect("temp dir");
        fs::create_dir_all(dir.join("crate-b/tests")).expect("temp dir");
        fs::write(dir.join(".gitignore"), "**/tests/scratch.rs\n").expect("write");
        let block: String = (0..SHINGLE).map(|n| format!("let v{n} = {n};\n")).collect();
        fs::write(dir.join("crate-a/src/lib.rs"), &block).expect("write");
        fs::write(dir.join("crate-b/tests/scratch.rs"), &block).expect("write");

        let counts = measure(&dir).expect("the fixture checkout is measurable");
        assert_eq!(
            counts["crate-a"].duplicate_lines, 0,
            "an ignored copy is not part of the repository, so it makes nothing duplicated"
        );
        assert!(
            !counts.contains_key("crate-b"),
            "a crate whose only source is ignored is not measured at all"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The baseline file as it is actually shaped: a justifying comment block
    /// over name-ordered rows. Built here rather than read from disk so the
    /// contract holds for the file's shape and not for today's contents.
    fn baseline_fixture() -> String {
        "# Duplicated source lines per crate.\n\
         #\n\
         # vyre-libs sits at its measured value but not at a floor.\n\
         \n\
         [[crate]]\n\
         name = \"alpha\"\n\
         duplicate_lines = 100\n\
         total_lines = 1000\n\
         \n\
         [[crate]]\n\
         name = \"gamma\"\n\
         duplicate_lines = 50\n\
         total_lines = 500\n"
            .to_string()
    }

    fn count(duplicate_lines: usize, total_lines: usize) -> CrateCount {
        CrateCount {
            duplicate_lines,
            total_lines,
        }
    }

    /// WHY: the header is the only record of which pins are deliberately tight
    /// and why neither side of a cross-crate pair can own the shape yet. The
    /// serializing writer dropped it, so the one job this file has, recording
    /// progress, corrupted the record. This is the regression that closes.
    #[test]
    fn lowering_a_pin_preserves_the_justifying_header() {
        let updated = rewrite_pin(&baseline_fixture(), "alpha", &count(90, 990))
            .expect("the fixture pins alpha");
        assert!(
            updated.contains("# vyre-libs sits at its measured value but not at a floor."),
            "the comment block survives a pin edit: {updated}"
        );
        assert!(updated.contains("duplicate_lines = 90"));
        assert!(updated.contains("total_lines = 990"));
    }

    /// WHY: a bulk rewrite from one checkout is how the intentional-red state of
    /// another worktree's pins was erased. Editing one row must leave every
    /// other row byte-identical, not re-derive it.
    #[test]
    fn lowering_one_pin_leaves_every_other_row_untouched() {
        let updated = rewrite_pin(&baseline_fixture(), "alpha", &count(90, 990))
            .expect("the fixture pins alpha");
        assert!(
            updated.contains("name = \"gamma\"\nduplicate_lines = 50\ntotal_lines = 500"),
            "an unnamed crate's row is not rewritten: {updated}"
        );
    }

    /// WHY: a name that is a prefix or suffix of another crate's name must not
    /// match it. `vyre-libs` and `vyre-libs-extra` would both be edited by a
    /// substring test, and the wrong row is worse than no edit.
    #[test]
    fn a_crate_the_file_does_not_pin_is_reported_rather_than_guessed() {
        assert!(
            rewrite_pin(&baseline_fixture(), "alph", &count(1, 2)).is_none(),
            "a prefix of a pinned name is not that name"
        );
        assert!(
            rewrite_pin(&baseline_fixture(), "beta", &count(1, 2)).is_none(),
            "an absent crate has no row to rewrite"
        );
    }

    /// WHY: a new crate must arrive measured, and it must land where a reader
    /// looks for it. Appending at the end makes the file stop being ordered,
    /// which is what makes its diffs readable.
    #[test]
    fn a_new_crate_is_inserted_in_name_order_under_the_header() {
        let updated = insert_pin(&baseline_fixture(), "beta", &count(7, 70));
        let alpha = updated.find("name = \"alpha\"").expect("alpha row");
        let beta = updated.find("name = \"beta\"").expect("beta row");
        let gamma = updated.find("name = \"gamma\"").expect("gamma row");
        assert!(
            alpha < beta && beta < gamma,
            "rows stay name-ordered: {updated}"
        );
        assert!(updated.starts_with("# Duplicated source lines per crate."));
        let parsed: BaselineFile =
            toml::from_str(&updated).expect("the written file must still parse");
        assert_eq!(parsed.crates.len(), 3, "insertion adds exactly one row");
    }

    /// WHY: a crate sorting after every pinned name has no row to insert before,
    /// and the tail case is where a hand-written splice loses the last row.
    #[test]
    fn a_crate_sorting_last_is_appended_and_still_parses() {
        let updated = insert_pin(&baseline_fixture(), "zeta", &count(3, 30));
        let parsed: BaselineFile =
            toml::from_str(&updated).expect("the written file must still parse");
        let names: Vec<&str> = parsed.crates.iter().map(|pin| pin.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "gamma", "zeta"]);
        assert_eq!(parsed.crates[2].duplicate_lines, 3);
    }
}
