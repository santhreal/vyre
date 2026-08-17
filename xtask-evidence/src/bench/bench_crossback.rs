//! Project the committed release benchmark evidence into one cross-backend table.
//!
//! The comparison is derived, never measured here. Every millisecond in the
//! table is a wall-clock reading a release benchmark suite already recorded
//! under `release/evidence/benchmarks/`, carried with the commit, source-tree
//! fingerprint and device signature it was taken under. So the gate regenerates
//! the table from that evidence and compares it byte for byte: two runs over the
//! same committed artifacts render the same table, and a divergence is a stale
//! table rather than the clock.
//!
//! This replaces a harness that timed an XOR loop inside this crate, recorded
//! `n/a` in every backend column, and wrote the file under `docs/perf/`, which
//! `.gitignore` excluded. A fresh checkout was red, one local `--write` turned
//! it green, and no reviewer could read the number. A table that carries no
//! measured backend row, and a measurement that arrives without provenance, are
//! findings now, and the table is committed evidence like every other artifact
//! a reviewer is expected to read.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;
use xtask::gate::{Finding, GateCtx, GateError, Report};
use xtask::output_arg::read_text_bounded;

/// The committed release benchmark evidence, and the table derived from it.
const EVIDENCE_DIR: &str = "release/evidence/benchmarks";
const TABLE: &str = "release/evidence/benchmarks/cross-backend-comparison.md";

/// Only the per-case result artifacts carry a backend measurement. The suite
/// aggregates and the multi-device artifacts in the same directory carry other
/// schemas, and are read by their own gates.
const RESULT_SCHEMA: &str = "vyre-bench.result.v1";

/// A result artifact is a few hundred kilobytes of percentile tables.
const MAX_ARTIFACT_BYTES: u64 = 8_388_608;

/// The rendered table is small. Anything larger than this is not one of ours.
const MAX_TABLE_BYTES: u64 = 262_144;

/// Fingerprints are printed to a fixed width so the table stays legible. The
/// full value stays in the artifact the row cites.
const FINGERPRINT_WIDTH: usize = 12;

const BANNER: &str = "# cross-backend comparison\n\n\
    Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a\n\
    wall-clock reading a release benchmark suite recorded under\n\
    `release/evidence/benchmarks/`, with the commit, source-tree fingerprint and\n\
    device signature it was taken under. `ratio` is the case wall time over the\n\
    fastest backend measured for that case.\n\n";

const MEASURED_COLUMNS: &str =
    "| case | backend | ms | ratio | commit | source tree | device | artifact |\n\
    |------|---------|----|-------|--------|-------------|--------|----------|\n";

const GAP_HEADING: &str = "\n## declared without a measurement\n\n";

const GAP_COLUMNS: &str = "| case | backend | declared by |\n\
    |------|---------|-------------|\n";

const NO_GAPS: &str = "Every backend a case contract declares carries a measurement.\n";

const REGENERATE: &str =
    "Run `./cargo_full run --bin xtask -- bench-crossback --write` and commit \
                          the table. It is derived from the committed evidence, so the two agree \
                          or one of them is stale.";

pub(crate) struct BenchCrossbackGate;

impl xtask::gate::GateBehavior for BenchCrossbackGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let cases = collect(&ctx.root, &mut report)?;
        report.cover_complete("benchmark crossback cases", cases.len());
        let measured: usize = cases.values().map(|case| case.measured.len()).sum();
        if measured == 0 {
            report.find(Finding::in_file(
                PathBuf::from(EVIDENCE_DIR),
                format!(
                    "the committed benchmark evidence carries no measured backend row, so there \
                 is no cross-backend comparison to record ({} artifact(s) read as \
                 `{RESULT_SCHEMA}`)",
                    cases.len()
                ),
                "Run a release benchmark suite on hardware and commit its result artifact. A \
             comparison table with no measurement in it records nothing.",
            ));
            return Ok(report);
        }

        let rendered = render(&cases);
        let absolute = ctx.root.join(TABLE);
        if ctx.write {
            xtask::output_arg::create_parent_dir(&absolute);
            std::fs::write(&absolute, &rendered).map_err(|error| {
                GateError::new(
                    format!("failed to write `{}`: {error}", absolute.display()),
                    "Check that the release/evidence/benchmarks directory is writable.",
                )
            })?;
            report.note(format!("wrote {TABLE}"));
        } else {
            audit_table(&absolute, &rendered, &mut report);
        }

        let backends: BTreeSet<&str> = cases
            .values()
            .flat_map(|case| case.measured.keys().map(String::as_str))
            .collect();
        report.note(format!(
            "{measured} measurement(s) across {} case(s) and {} backend(s)",
            cases.len(),
            backends.len()
        ));
        let gaps = gaps(&cases);
        if !gaps.is_empty() {
            report.note(format!(
                "{} case-backend pair(s) declared without a measurement, recorded in the table",
                gaps.len()
            ));
        }
        Ok(report)
    }
}

/// One backend's reading of one case, with the provenance it was taken under.
struct Measurement {
    wall_ns: f64,
    commit: String,
    source_tree: String,
    device: String,
    artifact: String,
}

/// One case, the backends its own contract declares, and what was measured.
#[derive(Default)]
struct Case {
    declared: BTreeMap<String, String>,
    measured: BTreeMap<String, Measurement>,
}

/// Read every result artifact in the committed evidence directory.
///
/// A measurement missing its wall-clock reading, its commit, its source-tree
/// fingerprint or its device signature is a finding against the artifact that
/// carries it. An unprovenanced number is not evidence, and the table must not
/// launder one into a comparison.
fn collect(root: &Path, report: &mut Report) -> Result<BTreeMap<String, Case>, GateError> {
    let directory = root.join(EVIDENCE_DIR);
    let mut artifacts: Vec<PathBuf> = std::fs::read_dir(&directory)
        .map_err(|error| {
            GateError::new(
                format!("failed to read `{EVIDENCE_DIR}`: {error}"),
                "The committed release benchmark evidence is the input to this gate. Run it from \
                 the workspace root.",
            )
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    artifacts.sort();

    let mut cases: BTreeMap<String, Case> = BTreeMap::new();
    for path in artifacts {
        let relative = format!(
            "{EVIDENCE_DIR}/{}",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        let text = read_text_bounded(&path, MAX_ARTIFACT_BYTES, "benchmark result artifact")
            .map_err(|error| {
                GateError::new(
                    format!("failed to read `{relative}`: {error}"),
                    "The committed evidence has to be readable to be compared against.",
                )
            })?;
        let document: Value = match serde_json::from_str(&text) {
            Ok(document) => document,
            Err(error) => {
                report.find(Finding::in_file(
                    PathBuf::from(&relative),
                    format!("benchmark evidence artifact is not valid JSON: {error}"),
                    "Regenerate the artifact from the suite that owns it.",
                ));
                continue;
            }
        };
        if document.get("schema").and_then(Value::as_str) != Some(RESULT_SCHEMA) {
            continue;
        }
        read_artifact(&relative, &document, &mut cases, report);
    }
    Ok(cases)
}

fn read_artifact(
    relative: &str,
    document: &Value,
    cases: &mut BTreeMap<String, Case>,
    report: &mut Report,
) {
    let commit = string_at(document, &["git", "commit"]);
    let source_tree = string_at(document, &["source_tree_fingerprint"]);
    let selected = string_at(document, &["selected_backend"]);
    let Some(entries) = document.get("cases").and_then(Value::as_array) else {
        report.find(Finding::in_file(
            PathBuf::from(relative),
            format!("artifact declares schema `{RESULT_SCHEMA}` and carries no cases array"),
            "Regenerate the artifact from the suite that owns it.",
        ));
        return;
    };
    for entry in entries {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            report.find(Finding::in_file(
                PathBuf::from(relative),
                "a recorded case carries no id, so no row can name it".to_string(),
                "Regenerate the artifact from the suite that owns it.",
            ));
            continue;
        };
        let case = cases.entry(id.to_string()).or_default();
        for backend in declared_backends(entry) {
            case.declared.insert(backend, relative.to_string());
        }
        let backend = string_at(entry, &["backend_id"])
            .or_else(|| selected.clone())
            .unwrap_or_default();
        let device = string_at(entry, &["device_signature"]);
        let wall_ns = entry.get("wall_ns").and_then(Value::as_f64);
        let missing = missing_provenance(&backend, wall_ns, &commit, &source_tree, &device);
        if !missing.is_empty() {
            report.find(Finding::in_file(
                PathBuf::from(relative),
                format!(
                    "case `{id}` records a measurement without {}",
                    missing.join(", ")
                ),
                "Regenerate the artifact from the suite that owns it. A number without the \
                 commit, tree and device it was taken on cannot be compared against another \
                 backend.",
            ));
            continue;
        }
        case.measured.insert(
            backend,
            Measurement {
                wall_ns: wall_ns.unwrap_or_default(),
                commit: commit.clone().unwrap_or_default(),
                source_tree: source_tree.clone().unwrap_or_default(),
                device: device.unwrap_or_default(),
                artifact: relative.to_string(),
            },
        );
    }
}

/// Which required parts of a measurement the artifact left out.
fn missing_provenance(
    backend: &str,
    wall_ns: Option<f64>,
    commit: &Option<String>,
    source_tree: &Option<String>,
    device: &Option<String>,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if backend.is_empty() {
        missing.push("a backend");
    }
    match wall_ns {
        Some(value) if value > 0.0 && value.is_finite() => {}
        _ => missing.push("a positive wall-clock reading"),
    }
    if commit.is_none() {
        missing.push("a commit");
    }
    if source_tree.is_none() {
        missing.push("a source-tree fingerprint");
    }
    if device.is_none() {
        missing.push("a device signature");
    }
    missing
}

/// The backends a case asserts its performance contract against.
///
/// The set is read from the case, so a workload added for a third backend is
/// covered without anyone editing this gate.
fn declared_backends(entry: &Value) -> Vec<String> {
    entry
        .get("contract")
        .and_then(|contract| contract.get("baselines"))
        .and_then(Value::as_array)
        .map(|baselines| {
            baselines
                .iter()
                .filter_map(|baseline| baseline.get("backend_ids"))
                .filter_map(Value::as_array)
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// A non-empty string at a path of object keys.
fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(key)?;
    }
    let text = cursor.as_str()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// Every case-backend pair a contract declares and no artifact measured.
fn gaps(cases: &BTreeMap<String, Case>) -> Vec<(&str, &str, &str)> {
    let mut gaps = Vec::new();
    for (id, case) in cases {
        for (backend, artifact) in &case.declared {
            if !case.measured.contains_key(backend) {
                gaps.push((id.as_str(), backend.as_str(), artifact.as_str()));
            }
        }
    }
    gaps
}

fn short(fingerprint: &str) -> String {
    match fingerprint.rsplit_once(':') {
        Some((prefix, hex)) => {
            let width = hex.len().min(FINGERPRINT_WIDTH);
            format!("{prefix}:{}", &hex[..width])
        }
        None => fingerprint
            .chars()
            .take(FINGERPRINT_WIDTH)
            .collect::<String>(),
    }
}

fn render(cases: &BTreeMap<String, Case>) -> String {
    let mut out = String::with_capacity(BANNER.len() + MEASURED_COLUMNS.len() + cases.len() * 256);
    out.push_str(BANNER);
    out.push_str(MEASURED_COLUMNS);
    for (id, case) in cases {
        let fastest = case
            .measured
            .values()
            .map(|measurement| measurement.wall_ns)
            .fold(f64::INFINITY, f64::min);
        for (backend, measurement) in &case.measured {
            out.push_str(&format!(
                "| `{id}` | {backend} | {:.3} | {:.3} | {} | {} | {} | `{}` |\n",
                measurement.wall_ns / 1_000_000.0,
                measurement.wall_ns / fastest,
                short(&measurement.commit),
                short(&measurement.source_tree),
                short(&measurement.device),
                measurement.artifact,
            ));
        }
    }
    out.push_str(GAP_HEADING);
    let gaps = gaps(cases);
    if gaps.is_empty() {
        out.push_str(NO_GAPS);
    } else {
        out.push_str(GAP_COLUMNS);
        for (id, backend, artifact) in gaps {
            out.push_str(&format!("| `{id}` | {backend} | `{artifact}` |\n"));
        }
    }
    out
}

/// Compare the committed table against the one the evidence renders.
fn audit_table(absolute: &Path, rendered: &str, report: &mut Report) {
    let text = match read_text_bounded(absolute, MAX_TABLE_BYTES, "cross-backend table") {
        Ok(text) => text,
        Err(error) => {
            report.find(Finding::in_file(
                PathBuf::from(TABLE),
                format!("cross-backend comparison table is missing or unreadable: {error}"),
                REGENERATE,
            ));
            return;
        }
    };
    let text = text.replace("\r\n", "\n");
    if text == rendered {
        return;
    }
    let line = first_divergence(&text, rendered);
    report.find(Finding::at(
        PathBuf::from(TABLE),
        line,
        format!(
            "cross-backend comparison table diverges from the committed evidence at line {line}"
        ),
        REGENERATE,
    ));
}

/// The one-based line the recorded and rendered tables first disagree on.
fn first_divergence(recorded: &str, rendered: &str) -> u32 {
    let mut recorded = recorded.lines();
    let mut rendered = rendered.lines();
    let mut line = 1u32;
    loop {
        match (recorded.next(), rendered.next()) {
            (None, None) => return line.saturating_sub(1).max(1),
            (Some(left), Some(right)) if left == right => line += 1,
            _ => return line,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn artifact(backend: &str, id: &str, wall_ns: f64, declared: &[&str]) -> Value {
        json!({
            "schema": RESULT_SCHEMA,
            "selected_backend": backend,
            "git": { "commit": format!("{backend}commit0000deadbeef") },
            "source_tree_fingerprint": "source-tree-v1:aaaabbbbccccdddd",
            "cases": [{
                "id": id,
                "backend_id": backend,
                "device_signature": format!("device-profile-v1:{backend}0000111122"),
                "wall_ns": wall_ns,
                "contract": { "baselines": [{ "backend_ids": declared }] },
            }],
        })
    }

    fn read(artifacts: &[(&str, Value)]) -> (BTreeMap<String, Case>, Report) {
        let mut cases = BTreeMap::new();
        let mut report = Report::clean();
        report.cover_complete("benchmark crossback cases", cases.len());
        for (name, document) in artifacts {
            read_artifact(name, document, &mut cases, &mut report);
        }
        (cases, report)
    }

    /// WHY: the whole point of the table is comparing two backends on one case.
    /// The rows have to land under one case id whichever artifact carried them,
    /// and the ratio column is the comparison, so it is 1 for the fastest.
    #[test]
    fn two_backends_measuring_one_case_render_one_comparison() {
        let (cases, report) = read(&[
            (
                "a.json",
                artifact("cuda", "case.one", 200_000.0, &["cuda", "wgpu"]),
            ),
            (
                "b.json",
                artifact("wgpu", "case.one", 800_000.0, &["cuda", "wgpu"]),
            ),
        ]);
        assert_eq!(report.findings, Vec::new());
        assert_eq!(cases.len(), 1);
        let table = render(&cases);
        assert!(
            table.contains("| `case.one` | cuda | 0.200 | 1.000 |"),
            "{table}"
        );
        assert!(
            table.contains("| `case.one` | wgpu | 0.800 | 4.000 |"),
            "{table}"
        );
        assert!(table.contains(NO_GAPS), "{table}");
    }

    /// WHY: this is the defect the gate shipped with. An empty evidence set, or
    /// one that carries no measurement, rendered a table of placeholders and
    /// reported nothing, so a fresh checkout went green on a file recording no
    /// measurement at all.
    #[test]
    fn evidence_with_no_measurement_cannot_render_a_clean_table() {
        let (cases, _) = read(&[]);
        assert_eq!(
            cases
                .values()
                .map(|case| case.measured.len())
                .sum::<usize>(),
            0
        );
        let mut named = artifact("cuda", "case.one", 200_000.0, &["cuda"]);
        named["cases"][0]["wall_ns"] = json!(null);
        let (cases, report) = read(&[("a.json", named)]);
        assert_eq!(
            cases
                .values()
                .map(|case| case.measured.len())
                .sum::<usize>(),
            0
        );
        assert_eq!(report.findings.len(), 1);
    }

    /// WHY: a number without the commit, tree and device it was taken on is not
    /// evidence, and laundering one into a comparison table is how an
    /// unreproducible measurement acquires authority. Every required part is
    /// covered, not one representative, because the missing one is always the
    /// one nobody tested.
    #[test]
    fn a_measurement_missing_any_provenance_field_is_a_finding() {
        let removals: &[(&[&str], &str)] = &[
            (&["git"], "commit"),
            (&["source_tree_fingerprint"], "source-tree fingerprint"),
        ];
        for (path, expected) in removals {
            let mut document = artifact("cuda", "case.one", 200_000.0, &["cuda"]);
            document
                .as_object_mut()
                .expect("artifact is an object")
                .remove(path[0]);
            let (cases, report) = read(&[("a.json", document)]);
            assert_eq!(
                report.findings.len(),
                1,
                "removing {path:?} reported nothing"
            );
            assert!(
                report.findings[0].message.contains(expected),
                "{}",
                report.findings[0].message
            );
            assert_eq!(cases["case.one"].measured.len(), 0);
        }
        for (field, expected) in [
            ("device_signature", "device signature"),
            ("wall_ns", "positive wall-clock reading"),
            ("backend_id", "a backend"),
        ] {
            let mut document = artifact("cuda", "case.one", 200_000.0, &["cuda"]);
            document["cases"][0]
                .as_object_mut()
                .expect("case is an object")
                .remove(field);
            if field == "backend_id" {
                document
                    .as_object_mut()
                    .expect("artifact is an object")
                    .remove("selected_backend");
            }
            let (_, report) = read(&[("a.json", document)]);
            assert_eq!(
                report.findings.len(),
                1,
                "removing {field} reported nothing"
            );
            assert!(
                report.findings[0].message.contains(expected),
                "{}",
                report.findings[0].message
            );
        }
    }

    /// WHY: a zero or negative wall time is the shape a harness writes when it
    /// measured nothing, and it used to pass as a number.
    #[test]
    fn a_non_positive_or_infinite_reading_is_not_a_measurement() {
        for wall_ns in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let (_, report) = read(&[("a.json", artifact("cuda", "case.one", wall_ns, &["cuda"]))]);
            assert_eq!(
                report.findings.len(),
                1,
                "wall_ns {wall_ns} reported nothing"
            );
        }
    }

    /// WHY: a backend a case contract declares and no suite measured is the
    /// parity gap the release gate owns, and it has to be visible in the table
    /// rather than dropping out of it. The declared set is read from the case,
    /// so a case declaring a third backend fails the recorded table until
    /// someone regenerates it.
    #[test]
    fn a_declared_backend_with_no_measurement_is_recorded_as_a_gap() {
        let (cases, report) = read(&[(
            "a.json",
            artifact("cuda", "case.one", 200_000.0, &["cuda", "wgpu", "metal"]),
        )]);
        assert_eq!(report.findings, Vec::new());
        assert_eq!(
            gaps(&cases),
            vec![
                ("case.one", "metal", "a.json"),
                ("case.one", "wgpu", "a.json")
            ]
        );
        let table = render(&cases);
        assert!(
            table.contains("| `case.one` | metal | `a.json` |"),
            "{table}"
        );
        assert!(!table.contains(NO_GAPS), "{table}");
    }

    /// WHY: the gate compares a committed file against this renderer, so the
    /// renderer's own output must audit clean or every run reports a divergence
    /// that is nobody's defect. A single edited cell must report, and it must
    /// name the line, because a table this wide is unreadable without one.
    #[test]
    fn a_rendered_table_audits_clean_and_one_edited_cell_does_not() {
        let (cases, _) = read(&[
            (
                "a.json",
                artifact("cuda", "case.one", 200_000.0, &["cuda", "wgpu"]),
            ),
            (
                "b.json",
                artifact("wgpu", "case.one", 800_000.0, &["cuda", "wgpu"]),
            ),
        ]);
        let rendered = render(&cases);
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("cross-backend-comparison.md");
        std::fs::write(&path, &rendered).expect("write table");
        let mut report = Report::clean();
        report.cover_complete("benchmark crossback cases", cases.len());
        audit_table(&path, &rendered, &mut report);
        assert_eq!(report.findings, Vec::new());

        let tampered = rendered.replace("0.800", "0.001");
        std::fs::write(&path, &tampered).expect("write tampered table");
        let mut report = Report::clean();
        report.cover_complete("benchmark crossback cases", cases.len());
        audit_table(&path, &rendered, &mut report);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(
            report.findings[0].line,
            Some(first_divergence(&tampered, &rendered))
        );
    }

    /// WHY: the previous table lived in a gitignored directory, so the absent
    /// case was the normal case and the gate was red on every fresh checkout
    /// until someone ran it locally. Absence has to report, and the fix text has
    /// to name the command that ends it.
    #[test]
    fn an_absent_table_is_a_finding_that_names_the_regeneration_command() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let mut report = Report::clean();
        audit_table(&directory.path().join("absent.md"), "table", &mut report);
        assert_eq!(report.findings.len(), 1);
        assert!(report.findings[0].fix.contains("bench-crossback --write"));
    }

    /// WHY: a fingerprint column exists to be compared by eye across rows, and
    /// the prefix is what says which fingerprint it is. Truncating past the
    /// prefix, or panicking on a short one, both defeat that.
    #[test]
    fn a_shortened_fingerprint_keeps_its_prefix_and_survives_a_short_value() {
        assert_eq!(
            short("source-tree-v1:0123456789abcdef"),
            "source-tree-v1:0123456789ab"
        );
        assert_eq!(short("source-tree-v1:abc"), "source-tree-v1:abc");
        assert_eq!(short("0123456789abcdef"), "0123456789ab");
        assert_eq!(short(""), "");
    }

    /// WHY: The authoritative descriptor and cross-backend comparison producer must agree on
    /// the exact output path so comparison is immutable and write mutations
    /// are never undeclared.
    #[test]
    fn authoritative_descriptor_declares_exact_bench_crossback_artifact() {
        let descriptor = xtask::gate_metadata::descriptor_by_name("bench-crossback");
        let mut expected: Vec<&str> = vec![super::TABLE];
        expected.sort_unstable();
        let mut actual: Vec<&str> = descriptor.artifacts.to_vec();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "Fix: bench-crossback gate descriptor must declare exactly the canonical cross-backend table artifact (`{}`)",
            super::TABLE
        );
    }
}
