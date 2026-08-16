//! `cargo xtask hot-path-scan`  -  ROADMAP S11 enforcement.
//!
//! Reads `docs/optimization/HOT_PATHS.toml` and scans every listed file
//! for allocation, clone, lock, sleep, panic, and string-construction patterns that
//! are usually evidence of hot-path waste:
//!
//! - `.clone()`  -  almost always hidden allocation; scratch reuse or
//!   `Cow` / `Arc` is cheaper.
//! - `.to_owned()` / `.to_string()`  -  allocates on every call.
//! - `Vec::new()` / `Vec::with_capacity(N)` (in non-init code)  -
//!   per-call vector; consider scratch reuse.
//! - `HashMap::new()` / `BTreeMap::new()`  -  per-call map.
//! - `String::new()` / `String::from(...)`  -  per-call string.
//! - `Mutex::new(...)` / `RwLock::new(...)`  -  per-call lock primitive
//!   in code that runs many times per dispatch.
//! - `std::thread::sleep(...)` / `tokio::time::sleep(...)`  -  fixed
//!   wait on a measured path.
//! - `panic!(...)` / `todo!(...)` / `unimplemented!(...)`  -  fail-open
//!   runtime behavior where hot paths need structured errors.
//!
//! Each finding prints `file:line | pattern | line content`. Exit 0
//! when the scan is informational (passed `--report` or default), exit
//! 1 when `--strict` is set and any finding fires.
//!
//! The scanner is line-oriented and regex-free to keep it deterministic across
//! rust-fmt rewrites; no AST parsing. A `#[cfg(test)]` item's whole body is
//! masked, because a hot-path file's own suite allocates on purpose and a
//! budget that counts those lines reports the test rather than the path. A
//! pattern matches only at a path boundary, so `SmallVec::new()` is not a
//! `Vec::new()` and `FxHashMap::new()` is one finding rather than two.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::ownership::{load_ownership_lanes, owner_lane_for_file, OwnershipLaneRule};
use crate::gates::scan::{cfg_test_lines, error_construction_lines, scan_code};

const MAX_HOT_PATH_SCAN_FILE_BYTES: u64 = 2_097_152;

#[derive(Debug, Deserialize)]
struct HotPathsConfig {
    #[serde(default)]
    schema: u32,
    #[serde(default)]
    hot_path: Vec<HotPathEntry>,
}

#[derive(Debug, Deserialize)]
struct HotPathEntry {
    file: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    max_findings: Option<usize>,
    #[serde(default)]
    max_allocation_findings: Option<usize>,
    #[serde(default)]
    max_clone_findings: Option<usize>,
    #[serde(default)]
    max_lock_findings: Option<usize>,
    #[serde(default)]
    max_sleep_findings: Option<usize>,
    #[serde(default)]
    max_panic_findings: Option<usize>,
}

#[derive(Debug)]
struct Hit {
    file: String,
    line: u32,
    pattern: &'static str,
    kind: PatternKind,
    content: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PatternKind {
    Allocation,
    Clone,
    Lock,
    Sleep,
    Panic,
}

#[derive(Clone, Copy, Debug)]
struct PatternSpec {
    name: &'static str,
    text: &'static str,
    kind: PatternKind,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FindingCounts {
    total: usize,
    allocations: usize,
    clones: usize,
    locks: usize,
    sleeps: usize,
    panics: usize,
    formats: usize,
}

impl FindingCounts {
    fn add(&mut self, finding: &Hit) {
        self.total = self.total.saturating_add(1);
        match finding.kind {
            PatternKind::Allocation => self.allocations = self.allocations.saturating_add(1),
            PatternKind::Clone => self.clones = self.clones.saturating_add(1),
            PatternKind::Lock => self.locks = self.locks.saturating_add(1),
            PatternKind::Sleep => self.sleeps = self.sleeps.saturating_add(1),
            PatternKind::Panic => self.panics = self.panics.saturating_add(1),
        }
        if finding.pattern == "format!" {
            self.formats = self.formats.saturating_add(1);
        }
    }
}

#[derive(Debug)]
struct BudgetDelta {
    file: String,
    budget: &'static str,
    actual: usize,
    limit: usize,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
struct BudgetVxCandidate {
    file: String,
    line: u32,
    owner_lane: String,
    budget: String,
    actual: usize,
    limit: usize,
    delta: usize,
    gate: String,
    suggested_vx: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct HotPathHeatmapRow {
    owner_lane: String,
    file: String,
    code_lines: usize,
    score: u64,
    findings_per_kloc: u64,
    allocations_per_kloc: u64,
    clones_per_kloc: u64,
    locks_per_kloc: u64,
    formats_per_kloc: u64,
    panics_per_kloc: u64,
}

const PATTERNS: &[PatternSpec] = &[
    PatternSpec {
        name: "clone",
        text: ".clone()",
        kind: PatternKind::Clone,
    },
    PatternSpec {
        name: "to_owned",
        text: ".to_owned()",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "to_string",
        text: ".to_string()",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "Vec::new",
        text: "Vec::new()",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "Vec::with_capacity",
        text: "Vec::with_capacity",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "HashMap::new",
        text: "HashMap::new()",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "BTreeMap::new",
        text: "BTreeMap::new()",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "FxHashMap::new",
        text: "FxHashMap::new()",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "String::new",
        text: "String::new()",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "String::from",
        text: "String::from(",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "Mutex::new",
        text: "Mutex::new(",
        kind: PatternKind::Lock,
    },
    PatternSpec {
        name: "RwLock::new",
        text: "RwLock::new(",
        kind: PatternKind::Lock,
    },
    PatternSpec {
        name: "format!",
        text: "format!(",
        kind: PatternKind::Allocation,
    },
    PatternSpec {
        name: "std_thread_sleep",
        text: "std::thread::sleep(",
        kind: PatternKind::Sleep,
    },
    PatternSpec {
        name: "tokio_sleep",
        text: "tokio::time::sleep(",
        kind: PatternKind::Sleep,
    },
    PatternSpec {
        name: "panic!",
        text: "panic!(",
        kind: PatternKind::Panic,
    },
    PatternSpec {
        name: "todo!",
        text: "todo!(",
        kind: PatternKind::Panic,
    },
    PatternSpec {
        name: "unimplemented!",
        text: "unimplemented!(",
        kind: PatternKind::Panic,
    },
];

/// What an overspent hot-path budget costs, and how to close it.
const BUDGET_FIX: &str = "reuse a scratch buffer, borrow instead of cloning, or hoist the allocation out of the measured path; raise the budget in docs/optimization/HOT_PATHS.toml only with a measurement that says the pattern is not the cost";

/// Scans every file docs/optimization/HOT_PATHS.toml lists against its budget.
pub struct HotPathScan;

impl Gate for HotPathScan {
    fn name(&self) -> &'static str {
        "hot-path-scan"
    }

    fn help(&self) -> &'static str {
        "Hold every file listed in docs/optimization/HOT_PATHS.toml to its allocation, clone, lock, sleep and panic budget; --budget-vx-json PATH writes the overage candidates"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = &ctx.root;
        let mut report = Report::clean();
        let budget_vx_json = parse_budget_vx_json(&ctx.args)
            .map_err(|error| GateError::new(error, "pass a path after --budget-vx-json"))?;
        let config_path = root
            .join("docs")
            .join("optimization")
            .join("HOT_PATHS.toml");
        let entries = load_config(&config_path).map_err(|error| {
            GateError::new(
                format!("cannot load {}: {error}", config_path.display()),
                "repair docs/optimization/HOT_PATHS.toml",
            )
        })?;
        let ownership_path = root
            .join("docs")
            .join("optimization")
            .join("OWNERSHIP.toml");
        let ownership_lanes = load_ownership_lanes(&ownership_path).map_err(|error| {
            GateError::new(
                format!("cannot load {}: {error}", ownership_path.display()),
                "repair docs/optimization/OWNERSHIP.toml",
            )
        })?;

        let mut hits: Vec<Hit> = Vec::new();
        let mut code_lines_by_file: BTreeMap<String, usize> = BTreeMap::new();
        let mut error_path_by_file: BTreeMap<String, usize> = BTreeMap::new();
        let mut scanned = 0usize;
        let mut missing: Vec<String> = Vec::new();
        for entry in &entries {
            let path = root.join(&entry.file);
            if !path.exists() {
                missing.push(entry.file.clone());
                continue;
            }
            scanned += 1;
            match read_text_bounded(&path) {
                Ok(text) => {
                    code_lines_by_file.insert(entry.file.clone(), count_code_lines(&text));
                    let on_error_paths = collect_findings(&entry.file, &text, &mut hits);
                    error_path_by_file.insert(entry.file.clone(), on_error_paths);
                }
                // An unreadable file used to warn and go unscanned, which is a
                // budget nobody measured reported as a budget held.
                Err(error) => report.find(Finding::in_file(
                    &entry.file,
                    format!("listed as a hot path and cannot be read: {error}"),
                    "make the file readable, or drop its row from docs/optimization/HOT_PATHS.toml",
                )),
            }
        }
        hits.sort_by(|a, b| {
            (a.file.as_str(), a.line, a.pattern).cmp(&(b.file.as_str(), b.line, b.pattern))
        });
        let mut by_file: BTreeMap<&str, FindingCounts> = BTreeMap::new();
        for hit in &hits {
            by_file.entry(hit.file.as_str()).or_default().add(hit);
        }
        let budget_deltas = collect_budget_deltas(&entries, &by_file);
        let unowned_hot_paths = unowned_hot_path_files(&entries, &ownership_lanes);
        let candidates = budget_vx_candidates(&budget_deltas, &hits, &ownership_lanes);
        let heatmap =
            build_hot_path_heatmap(&entries, &by_file, &code_lines_by_file, &ownership_lanes);

        for path in &missing {
            report.find(Finding::in_file(
                path,
                "listed in docs/optimization/HOT_PATHS.toml and not on disk",
                "point the row at where the code moved, or delete the row with the code",
            ));
        }
        for path in &unowned_hot_paths {
            report.find(Finding::in_file(
                path,
                "is a hot path with no owner lane",
                "give the file a lane in docs/optimization/OWNERSHIP.toml",
            ));
        }
        for delta in &budget_deltas {
            report.find(Finding::at(
                &delta.file,
                first_budget_finding_line(delta, &hits),
                format!(
                    "{} is {} against a budget of {}",
                    delta.budget, delta.actual, delta.limit
                ),
                BUDGET_FIX,
            ));
        }

        report.note(format!(
            "listed {} | scanned {scanned} | pattern hits {}",
            entries.len(),
            hits.len()
        ));
        let reason_by_file: BTreeMap<&str, &str> = entries
            .iter()
            .map(|entry| (entry.file.as_str(), entry.reason.as_str()))
            .collect();
        for (file, counts) in &by_file {
            let reason = reason_by_file.get(file).copied().unwrap_or("");
            let on_error_paths = error_path_by_file.get(*file).copied().unwrap_or(0);
            report.note(format!(
                "{file}: total={} allocations={} clones={} locks={} sleeps={} panics={} formats={} error-path={on_error_paths}{}{reason}",
                counts.total,
                counts.allocations,
                counts.clones,
                counts.locks,
                counts.sleeps,
                counts.panics,
                counts.formats,
                if reason.is_empty() { "" } else { " | " }
            ));
        }
        for row in &heatmap {
            report.note(format!(
                "owner={} | file={} | score={} code_loc={} findings/kLOC={} allocations/kLOC={} clones/kLOC={} locks/kLOC={} formats/kLOC={} panics/kLOC={}",
                row.owner_lane,
                row.file,
                row.score,
                row.code_lines,
                row.findings_per_kloc,
                row.allocations_per_kloc,
                row.clones_per_kloc,
                row.locks_per_kloc,
                row.formats_per_kloc,
                row.panics_per_kloc
            ));
        }
        for hit in &hits {
            report.note(format!(
                "{}:{} | {} | {}",
                hit.file,
                hit.line,
                hit.pattern,
                hit.content.trim()
            ));
        }

        if let Some(path) = budget_vx_json {
            write_budget_vx_candidates(&path, &candidates).map_err(|error| {
                GateError::new(
                    format!(
                        "cannot write hot-path budget candidates to {}: {error}",
                        path.display()
                    ),
                    "name a writable path after --budget-vx-json",
                )
            })?;
            report.note(format!(
                "wrote {} budget candidate(s) to {}",
                candidates.len(),
                path.display()
            ));
        }
        Ok(report)
    }
}

fn parse_budget_vx_json(args: &[String]) -> Result<Option<PathBuf>, String> {
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--budget-vx-json" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("--budget-vx-json requires a path".to_string());
                };
                return Ok(Some(PathBuf::from(path)));
            }
            _ => index += 1,
        }
    }
    Ok(None)
}

fn load_config(path: &Path) -> Result<Vec<HotPathEntry>, String> {
    let text = read_text_bounded(path).map_err(|e| e.to_string())?;
    let cfg: HotPathsConfig = toml::from_str(&text).map_err(|e| e.to_string())?;
    if cfg.schema != 1 {
        return Err(format!(
            "expected schema = 1, got {}  -  update the loader before changing the schema",
            cfg.schema
        ));
    }
    Ok(cfg.hot_path)
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(path, MAX_HOT_PATH_SCAN_FILE_BYTES, "hot-path scan")
}

fn write_budget_vx_candidates(path: &Path, candidates: &[BudgetVxCandidate]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(candidates).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("budget VX candidate JSON serialization failed: {error}"),
        )
    })?;
    std::fs::write(path, format!("{text}\n"))
}

/// Whether `text` occurs in `line` as its own path segment.
///
/// `contains` alone reads `SmallVec::new()` as `Vec::new()`, so a stack
/// allocation entered a heap-allocation budget, and it reads
/// `FxHashMap::new()` as both `FxHashMap::new` and `HashMap::new`, so one map
/// construction spent two findings out of the file's ceiling. A pattern that
/// begins with an identifier byte therefore has to begin at a boundary. `:` is
/// not an identifier byte, so `crate::Vec::new()` still matches, and a pattern
/// that begins with `.` is unaffected because the receiver before it is the
/// whole point.
fn occurs_as_path(line: &str, text: &str) -> bool {
    let anchored = text
        .as_bytes()
        .first()
        .is_some_and(|byte| is_identifier_byte(*byte));
    if !anchored {
        return line.contains(text);
    }
    let bytes = line.as_bytes();
    line.match_indices(text)
        .any(|(at, _)| at == 0 || !is_identifier_byte(bytes[at - 1]))
}

/// Bytes that can spell part of a Rust identifier.
fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Record every measured-path pattern hit in `text`, and return how many hits
/// were left out because they build an error rather than a result.
fn collect_findings(file: &str, text: &str, out: &mut Vec<Hit>) -> usize {
    let lines: Vec<&str> = text.lines().collect();
    let test_only = cfg_test_lines(&lines);
    let on_error_path = error_construction_lines(&lines);
    let mut excluded = 0usize;
    for (line_no, line) in lines.iter().enumerate() {
        if test_only[line_no] {
            continue;
        }
        let scan_line = scan_code(line).code;
        if scan_line.trim_start().is_empty() {
            continue;
        }
        for spec in PATTERNS {
            if !occurs_as_path(scan_line, spec.text) {
                continue;
            }
            if on_error_path[line_no] {
                excluded += 1;
                continue;
            }
            out.push(Hit {
                file: file.to_string(),
                line: (line_no + 1) as u32,
                pattern: spec.name,
                kind: spec.kind,
                content: (*line).to_string(),
            });
        }
    }
    excluded
}

fn count_code_lines(text: &str) -> usize {
    let lines: Vec<&str> = text.lines().collect();
    let test_only = cfg_test_lines(&lines);
    lines
        .iter()
        .enumerate()
        .filter(|(line_no, line)| {
            !test_only[*line_no] && !scan_code(line).code.trim_start().is_empty()
        })
        .count()
}

fn collect_budget_deltas(
    entries: &[HotPathEntry],
    by_file: &BTreeMap<&str, FindingCounts>,
) -> Vec<BudgetDelta> {
    let mut deltas = Vec::new();
    for entry in entries {
        let counts = by_file
            .get(entry.file.as_str())
            .copied()
            .unwrap_or_default();
        push_budget_delta(
            &mut deltas,
            &entry.file,
            "max_findings",
            counts.total,
            entry.max_findings,
        );
        push_budget_delta(
            &mut deltas,
            &entry.file,
            "max_allocation_findings",
            counts.allocations,
            entry.max_allocation_findings,
        );
        push_budget_delta(
            &mut deltas,
            &entry.file,
            "max_clone_findings",
            counts.clones,
            entry.max_clone_findings,
        );
        push_budget_delta(
            &mut deltas,
            &entry.file,
            "max_lock_findings",
            counts.locks,
            entry.max_lock_findings,
        );
        push_budget_delta(
            &mut deltas,
            &entry.file,
            "max_sleep_findings",
            counts.sleeps,
            entry.max_sleep_findings,
        );
        push_budget_delta(
            &mut deltas,
            &entry.file,
            "max_panic_findings",
            counts.panics,
            entry.max_panic_findings,
        );
    }
    deltas
}

fn unowned_hot_path_files(
    entries: &[HotPathEntry],
    ownership_lanes: &[OwnershipLaneRule],
) -> Vec<String> {
    entries
        .iter()
        .filter(|entry| owner_lane_for_file(&entry.file, ownership_lanes) == "unowned")
        .map(|entry| entry.file.clone())
        .collect()
}

fn budget_vx_candidates(
    deltas: &[BudgetDelta],
    findings: &[Hit],
    ownership_lanes: &[OwnershipLaneRule],
) -> Vec<BudgetVxCandidate> {
    deltas
        .iter()
        .map(|delta| {
            let line = first_budget_finding_line(delta, findings);
            let owner_lane = owner_lane_for_file(&delta.file, ownership_lanes).to_string();
            BudgetVxCandidate {
                file: delta.file.clone(),
                line,
                owner_lane,
                budget: delta.budget.to_string(),
                actual: delta.actual,
                limit: delta.limit,
                delta: delta.actual.saturating_sub(delta.limit),
                gate: "cargo_full run -p xtask --bin xtask -- hot-path-scan --strict".to_string(),
                suggested_vx: budget_candidate_id(&delta.file, delta.budget),
            }
        })
        .collect()
}

fn first_budget_finding_line(delta: &BudgetDelta, findings: &[Hit]) -> u32 {
    findings
        .iter()
        .find(|finding| finding.file == delta.file && finding_matches_budget(finding, delta.budget))
        .map(|finding| finding.line)
        .unwrap_or(0)
}

fn finding_matches_budget(finding: &Hit, budget: &str) -> bool {
    match budget {
        "max_findings" => true,
        "max_allocation_findings" => finding.kind == PatternKind::Allocation,
        "max_clone_findings" => finding.kind == PatternKind::Clone,
        "max_lock_findings" => finding.kind == PatternKind::Lock,
        "max_sleep_findings" => finding.kind == PatternKind::Sleep,
        "max_panic_findings" => finding.kind == PatternKind::Panic,
        _ => false,
    }
}

fn budget_candidate_id(file: &str, budget: &str) -> String {
    let mut out = String::from("HOTPATH-");
    for byte in file.bytes().chain([b'-']).chain(budget.bytes()) {
        if byte.is_ascii_alphanumeric() {
            out.push((byte as char).to_ascii_uppercase());
        } else {
            out.push('-');
        }
    }
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    out.trim_end_matches('-').to_string()
}

fn build_hot_path_heatmap(
    entries: &[HotPathEntry],
    by_file: &BTreeMap<&str, FindingCounts>,
    code_lines_by_file: &BTreeMap<String, usize>,
    ownership_lanes: &[OwnershipLaneRule],
) -> Vec<HotPathHeatmapRow> {
    let mut rows = Vec::new();
    for entry in entries {
        let Some(code_lines) = code_lines_by_file.get(&entry.file).copied() else {
            continue;
        };
        let counts = by_file
            .get(entry.file.as_str())
            .copied()
            .unwrap_or_default();
        let findings_per_kloc = per_kloc(counts.total, code_lines);
        let allocations_per_kloc = per_kloc(counts.allocations, code_lines);
        let clones_per_kloc = per_kloc(counts.clones, code_lines);
        let locks_per_kloc = per_kloc(counts.locks, code_lines);
        let formats_per_kloc = per_kloc(counts.formats, code_lines);
        let panics_per_kloc = per_kloc(counts.panics, code_lines);
        let score = code_lines as u64
            + findings_per_kloc
            + allocations_per_kloc.saturating_mul(4)
            + clones_per_kloc.saturating_mul(3)
            + locks_per_kloc.saturating_mul(8)
            + formats_per_kloc.saturating_mul(5)
            + panics_per_kloc.saturating_mul(12);
        rows.push(HotPathHeatmapRow {
            owner_lane: owner_lane_for_file(&entry.file, ownership_lanes).to_string(),
            file: entry.file.clone(),
            code_lines,
            score,
            findings_per_kloc,
            allocations_per_kloc,
            clones_per_kloc,
            locks_per_kloc,
            formats_per_kloc,
            panics_per_kloc,
        });
    }
    rows.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.owner_lane.cmp(&b.owner_lane))
            .then_with(|| a.file.cmp(&b.file))
    });
    rows
}

fn per_kloc(count: usize, code_lines: usize) -> u64 {
    if code_lines == 0 {
        0
    } else {
        (count as u64).saturating_mul(1000) / code_lines as u64
    }
}

fn push_budget_delta(
    deltas: &mut Vec<BudgetDelta>,
    file: &str,
    budget: &'static str,
    actual: usize,
    limit: Option<usize>,
) {
    let Some(limit) = limit else {
        return;
    };
    if actual > limit {
        deltas.push(BudgetDelta {
            file: file.to_string(),
            budget,
            actual,
            limit,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_findings_picks_up_clone() {
        let mut out = Vec::new();
        let _ = collect_findings("x.rs", "let y = x.clone();\n", &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pattern, "clone");
        assert_eq!(out[0].kind, PatternKind::Clone);
        assert_eq!(out[0].line, 1);
    }

    #[test]
    fn collect_findings_skips_comments() {
        let mut out = Vec::new();
        let _ = collect_findings("x.rs", "// uses x.clone() in docs\n", &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_findings_picks_up_multiple_patterns() {
        let mut out = Vec::new();
        let _ = collect_findings(
            "x.rs",
            "let v: Vec<u32> = Vec::new();\nlet s = String::from(\"a\");\nlet l = Mutex::new(0);\n",
            &mut out,
        );
        let pats: Vec<&str> = out.iter().map(|f| f.pattern).collect();
        assert!(pats.contains(&"Vec::new"));
        assert!(pats.contains(&"String::from"));
        assert!(pats.contains(&"Mutex::new"));
        let mut counts = FindingCounts::default();
        for finding in &out {
            counts.add(finding);
        }
        assert_eq!(counts.total, 3);
        assert_eq!(counts.allocations, 2);
        assert_eq!(counts.locks, 1);
    }

    #[test]
    fn collect_findings_picks_up_sleep_and_panic_patterns() {
        let mut out = Vec::new();
        let _ = collect_findings(
            "x.rs",
            "std::thread::sleep(d);\ntokio::time::sleep(d);\npanic!(\"bad\");\ntodo!();\nunimplemented!();\n",
            &mut out,
        );
        let pats: Vec<&str> = out.iter().map(|f| f.pattern).collect();
        assert!(pats.contains(&"std_thread_sleep"));
        assert!(pats.contains(&"tokio_sleep"));
        assert!(pats.contains(&"panic!"));
        assert!(pats.contains(&"todo!"));
        assert!(pats.contains(&"unimplemented!"));
        let mut counts = FindingCounts::default();
        for finding in &out {
            counts.add(finding);
        }
        assert_eq!(counts.sleeps, 2);
        assert_eq!(counts.panics, 3);
    }

    #[test]
    fn collect_findings_picks_up_format_macro() {
        let mut out = Vec::new();
        let _ = collect_findings("x.rs", "let s = format!(\"{}\", 5);\n", &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pattern, "format!");
    }

    #[test]
    fn load_config_rejects_wrong_schema() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("hp.toml");
        std::fs::write(&path, "schema = 99\nhot_path = []\n").unwrap();
        let err = load_config(&path).unwrap_err();
        assert!(err.contains("schema = 1"));
    }

    #[test]
    fn load_config_parses_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("hp.toml");
        std::fs::write(
            &path,
            "schema = 1\n[[hot_path]]\nfile = \"a.rs\"\nreason = \"x\"\nmax_findings = 2\nmax_allocation_findings = 1\nmax_clone_findings = 1\nmax_lock_findings = 0\nmax_sleep_findings = 0\nmax_panic_findings = 0\n",
        )
        .unwrap();
        let entries = load_config(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file, "a.rs");
        assert_eq!(entries[0].max_findings, Some(2));
        assert_eq!(entries[0].max_allocation_findings, Some(1));
        assert_eq!(entries[0].max_clone_findings, Some(1));
        assert_eq!(entries[0].max_lock_findings, Some(0));
        assert_eq!(entries[0].max_sleep_findings, Some(0));
        assert_eq!(entries[0].max_panic_findings, Some(0));
    }

    #[test]
    fn unowned_hot_paths_report_exact_missing_owner_files() {
        let entries = vec![
            HotPathEntry {
                file: "vyre-lower/src/pre_emit.rs".to_string(),
                reason: String::new(),
                max_findings: None,
                max_allocation_findings: None,
                max_clone_findings: None,
                max_lock_findings: None,
                max_sleep_findings: None,
                max_panic_findings: None,
            },
            HotPathEntry {
                file: "vyre-driver/src/launch_fusion.rs".to_string(),
                reason: String::new(),
                max_findings: None,
                max_allocation_findings: None,
                max_clone_findings: None,
                max_lock_findings: None,
                max_sleep_findings: None,
                max_panic_findings: None,
            },
        ];
        let ownership = crate::gates::ownership::parse_ownership_lane_rules(
            r#"
[lane.driver_shared]
write = ["vyre-driver/src/**"]
"#,
        )
        .unwrap();

        let unowned = unowned_hot_path_files(&entries, &ownership);

        assert_eq!(unowned, vec!["vyre-lower/src/pre_emit.rs"]);
    }

    #[test]
    fn budget_vx_candidates_serialize_exact_budget_owner_line_and_gate() {
        let deltas = vec![
            BudgetDelta {
                file: "vyre-emit-naga/src/lib.rs".to_string(),
                budget: "max_findings",
                actual: 10,
                limit: 9,
            },
            BudgetDelta {
                file: "vyre-runtime/src/resident_work_queue/telemetry/mod.rs".to_string(),
                budget: "max_panic_findings",
                actual: 10,
                limit: 0,
            },
        ];
        let findings = vec![
            Hit {
                file: "vyre-emit-naga/src/lib.rs".to_string(),
                line: 86,
                pattern: "clone",
                kind: PatternKind::Clone,
                content: "Some(cached.module.clone())".to_string(),
            },
            Hit {
                file: "vyre-runtime/src/resident_work_queue/telemetry/mod.rs".to_string(),
                line: 69,
                pattern: "panic!",
                kind: PatternKind::Panic,
                content: "panic!(\"bad\")".to_string(),
            },
        ];
        let ownership = crate::gates::ownership::parse_ownership_lane_rules(
            r#"
[lane.lower_emit]
write = ["vyre-emit-naga/src/**"]

[lane.runtime_resident_work_queue]
write = ["vyre-runtime/src/resident_work_queue/**"]
"#,
        )
        .unwrap();

        let candidates = budget_vx_candidates(&deltas, &findings, &ownership);
        let json = serde_json::to_value(&candidates).unwrap();

        assert_eq!(json[0]["file"], "vyre-emit-naga/src/lib.rs");
        assert_eq!(json[0]["line"], 86);
        assert_eq!(json[0]["owner_lane"], "lower_emit");
        assert_eq!(json[0]["budget"], "max_findings");
        assert_eq!(json[0]["actual"], 10);
        assert_eq!(json[0]["limit"], 9);
        assert_eq!(json[0]["delta"], 1);
        assert_eq!(
            json[0]["gate"],
            "cargo_full run -p xtask --bin xtask -- hot-path-scan --strict"
        );
        assert_eq!(
            json[0]["suggested_vx"],
            "HOTPATH-VYRE-EMIT-NAGA-SRC-LIB-RS-MAX-FINDINGS"
        );
        assert_eq!(
            json[1]["file"],
            "vyre-runtime/src/resident_work_queue/telemetry/mod.rs"
        );
        assert_eq!(json[1]["line"], 69);
        assert_eq!(json[1]["owner_lane"], "runtime_resident_work_queue");
        assert_eq!(json[1]["budget"], "max_panic_findings");
    }

    #[test]
    fn collect_budget_deltas_reports_only_over_budget_categories() {
        let entries = vec![HotPathEntry {
            file: "x.rs".to_string(),
            reason: "hot".to_string(),
            max_findings: Some(4),
            max_allocation_findings: Some(1),
            max_clone_findings: Some(1),
            max_lock_findings: Some(0),
            max_sleep_findings: Some(0),
            max_panic_findings: Some(0),
        }];
        let mut by_file = std::collections::BTreeMap::new();
        by_file.insert(
            "x.rs",
            FindingCounts {
                total: 4,
                allocations: 2,
                clones: 1,
                locks: 0,
                sleeps: 1,
                panics: 1,
                formats: 0,
            },
        );
        let deltas = collect_budget_deltas(&entries, &by_file);
        assert_eq!(deltas.len(), 3);
        assert_eq!(deltas[0].file, "x.rs");
        assert_eq!(deltas[0].budget, "max_allocation_findings");
        assert_eq!(deltas[0].actual, 2);
        assert_eq!(deltas[0].limit, 1);
        assert_eq!(deltas[1].budget, "max_sleep_findings");
        assert_eq!(deltas[1].actual, 1);
        assert_eq!(deltas[1].limit, 0);
        assert_eq!(deltas[2].budget, "max_panic_findings");
        assert_eq!(deltas[2].actual, 1);
        assert_eq!(deltas[2].limit, 0);
    }

    #[test]
    fn collect_findings_ignores_inline_comment_patterns_and_counts_code_lines() {
        let mut out = Vec::new();
        let text = "// format!(\"{}\", x)\nlet keep = 1; // panic!(\"comment\")\nlet msg = format!(\"{}\", keep);\n\n";

        let _ = collect_findings("x.rs", text, &mut out);

        assert_eq!(count_code_lines(text), 2);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pattern, "format!");
    }

    /// WHY: an attribute annotates the item after it. Skipping the line the
    /// attribute sits on excluded one line and left the whole `mod tests` body
    /// counted as runtime cost, so a fixture's `panic!` was reported as a
    /// hot-path panic and weighted twelve times per kLOC. The body must be
    /// excluded from BOTH the findings and the per-kLOC denominator.
    #[test]
    fn a_cfg_test_module_body_is_not_runtime_cost() {
        let text = concat!(
            "pub fn run() {\n",
            "    let s = format!(\"{}\", 1);\n",
            "}\n",
            "\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    #[test]\n",
            "    fn t() {\n",
            "        panic!(\"fixture\");\n",
            "        let x = String::new();\n",
            "    }\n",
            "}\n",
        );
        let mut out = Vec::new();
        let _ = collect_findings("x.rs", text, &mut out);

        assert_eq!(
            out.iter().map(|f| f.pattern).collect::<Vec<_>>(),
            vec!["format!"],
            "only the runtime format! may be reported"
        );
        assert_eq!(
            count_code_lines(text),
            3,
            "the three runtime lines, not the eight test lines"
        );
    }

    /// WHY: the exclusion must end where the item does. A test module that
    /// swallowed the rest of the file would silence every later runtime
    /// finding, which is the same defect with the sign flipped.
    #[test]
    fn runtime_code_after_a_cfg_test_module_is_still_scanned() {
        let text = concat!(
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    fn t() {\n",
            "        panic!(\"fixture\");\n",
            "    }\n",
            "}\n",
            "pub fn later() {\n",
            "    let s = String::new();\n",
            "}\n",
        );
        let mut out = Vec::new();
        let _ = collect_findings("x.rs", text, &mut out);

        assert_eq!(
            out.iter().map(|f| f.pattern).collect::<Vec<_>>(),
            vec!["String::new"],
            "the runtime allocation after the module must still be reported"
        );
        assert_eq!(out[0].line, 8);
    }

    /// WHY: a `#[cfg(test)]` item with no body ends at its semicolon. Treating
    /// it as an unterminated block would mask every line after it.
    #[test]
    fn a_cfg_test_item_without_a_body_ends_at_its_semicolon() {
        let text = concat!(
            "#[cfg(test)]\n",
            "use std::string::String;\n",
            "pub fn later() {\n",
            "    let s = String::new();\n",
            "}\n",
        );
        let mut out = Vec::new();
        let _ = collect_findings("x.rs", text, &mut out);

        assert_eq!(out.len(), 1, "got {out:?}");
        assert_eq!(out[0].line, 4);
    }

    /// WHY: the exclusion that took the CUDA dispatch surface from nineteen
    /// measured allocations to the ones a successful launch pays. Every message
    /// in this fixture is built after the call has already failed, and this
    /// workspace requires those messages to carry context.
    #[test]
    fn an_allocation_that_builds_an_error_is_not_measured_but_is_counted() {
        let text = concat!(
            "pub fn f(n: usize) -> Result<usize, String> {\n",
            "    if n == 0 {\n",
            "        return Err(format!(\n",
            "            \"Fix: n must exceed {n}.\"\n",
            "        ));\n",
            "    }\n",
            "    Ok(n)\n",
            "}\n",
        );
        let mut out = Vec::new();
        let excluded = collect_findings("x.rs", text, &mut out);

        assert!(out.is_empty(), "got {out:?}");
        assert_eq!(excluded, 1);
    }

    /// WHY: `contains` matching made `SmallVec::new()` a `Vec::new()` finding
    /// and `FxHashMap::new()` two findings, so a file's ceiling was spent on a
    /// stack allocation and on one construction counted twice. That is the whole
    /// distance between a budget that measures heap traffic and one that
    /// measures spelling. A qualified path still matches, because a `Vec` reached
    /// through `alloc::vec::Vec` is the same allocation.
    #[test]
    fn a_pattern_matches_a_whole_path_segment_and_not_a_longer_name() {
        let text = concat!(
            "pub fn f() {\n",
            "    let a: SmallVec<[u8; 4]> = SmallVec::new();\n",
            "    let b = FxHashMap::new();\n",
            "    let c = alloc::vec::Vec::new();\n",
            "}\n",
        );
        let mut out = Vec::new();
        let _ = collect_findings("x.rs", text, &mut out);

        let reported: Vec<(&str, u32)> = out
            .iter()
            .map(|hit| (hit.pattern, hit.line))
            .collect();
        assert_eq!(
            reported,
            vec![("FxHashMap::new", 3), ("Vec::new", 4)],
            "SmallVec is not a Vec and FxHashMap is one finding: {out:?}"
        );
    }

    /// WHY: the exclusion must end where the error expression ends, or the first
    /// `map_err` in a function hides every allocation after it.
    #[test]
    fn an_allocation_after_an_error_expression_is_still_measured() {
        let text = concat!(
            "pub fn f(input: &str) -> Result<String, String> {\n",
            "    let n: usize = input\n",
            "        .parse()\n",
            "        .map_err(|error| format!(\"Fix: {error}\"))?;\n",
            "    let label = format!(\"{n} items\");\n",
            "    Ok(label)\n",
            "}\n",
        );
        let mut out = Vec::new();
        let excluded = collect_findings("x.rs", text, &mut out);

        assert_eq!(out.len(), 1, "got {out:?}");
        assert_eq!(out[0].line, 5);
        assert_eq!(excluded, 1);
    }

    /// WHY: a brace inside a string literal is not a block. `panic!("{")` in a
    /// test body used to close the module early, and every later runtime line
    /// then read as test-only or the reverse depending on parity.
    #[test]
    fn a_brace_in_a_string_literal_does_not_move_block_depth() {
        assert_eq!(scan_code("    panic!(\"{\");").brace_delta, 0);
        assert_eq!(scan_code("    let x = '}';").brace_delta, 0);
        assert_eq!(scan_code("fn f() {").brace_delta, 1);
    }

    /// WHY: `'` opens a lifetime as often as a character literal. Reading
    /// `&'a str` as the start of a literal swallowed the rest of the line, so a
    /// trailing `//` comment read as code and a following `{` never counted.
    #[test]
    fn a_lifetime_is_not_a_character_literal() {
        let scan = scan_code("fn f<'a>(x: &'a str) { // format!(\"{}\", x)");
        assert_eq!(scan.brace_delta, 1);
        assert!(
            !scan.code.contains("format!"),
            "the trailing comment must still be stripped, got {:?}",
            scan.code
        );
    }

    #[test]
    fn hot_path_heatmap_ranks_mega_file_and_assigns_owner_lane() {
        let entries = vec![
            HotPathEntry {
                file: "vyre-foundation/src/optimizer/big.rs".to_string(),
                reason: "mega optimizer".to_string(),
                max_findings: None,
                max_allocation_findings: None,
                max_clone_findings: None,
                max_lock_findings: None,
                max_sleep_findings: None,
                max_panic_findings: None,
            },
            HotPathEntry {
                file: "vyre-driver/src/small.rs".to_string(),
                reason: "small driver".to_string(),
                max_findings: None,
                max_allocation_findings: None,
                max_clone_findings: None,
                max_lock_findings: None,
                max_sleep_findings: None,
                max_panic_findings: None,
            },
        ];
        let mut by_file = BTreeMap::new();
        by_file.insert(
            "vyre-foundation/src/optimizer/big.rs",
            FindingCounts {
                total: 80,
                allocations: 30,
                clones: 20,
                locks: 2,
                sleeps: 0,
                panics: 1,
                formats: 10,
            },
        );
        by_file.insert(
            "vyre-driver/src/small.rs",
            FindingCounts {
                total: 1,
                allocations: 1,
                clones: 0,
                locks: 0,
                sleeps: 0,
                panics: 0,
                formats: 0,
            },
        );
        let mut code_lines = BTreeMap::new();
        code_lines.insert("vyre-foundation/src/optimizer/big.rs".to_string(), 5862);
        code_lines.insert("vyre-driver/src/small.rs".to_string(), 40);
        let lanes = vec![
            OwnershipLaneRule {
                lane: "foundation_optimizer".to_string(),
                write_patterns: vec!["vyre-foundation/src/optimizer/**".to_string()],
            },
            OwnershipLaneRule {
                lane: "driver_shared".to_string(),
                write_patterns: vec!["vyre-driver/src/**".to_string()],
            },
        ];

        let rows = build_hot_path_heatmap(&entries, &by_file, &code_lines, &lanes);

        assert_eq!(rows[0].file, "vyre-foundation/src/optimizer/big.rs");
        assert_eq!(rows[0].owner_lane, "foundation_optimizer");
        assert!(rows[0].formats_per_kloc > 0);
    }
}
