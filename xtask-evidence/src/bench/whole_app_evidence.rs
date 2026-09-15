//! Own the whole-application evidence artifacts.
//!
//! The four records under `release/evidence/benchmarks` that state what one
//! complete application did on a device had no registered owner. `vyre-bench
//! whole-app-evidence` wrote them straight from the CLI, so no gate declared
//! them, the attribution ledger reported each one as an artifact nothing in the
//! tree produces, and the committed bodies predated the provenance stamp
//! entirely. This gate is their single owner: `--write` measures the suite on
//! this host's device and records all four, and without it the committed
//! records are held to the contract their own types state.

use std::path::Path;

use serde::de::DeserializeOwned;
use vyre_bench::workloads::whole_app::{
    write_whole_application_evidence_artifacts, ApplicationDomain,
    WholeApplicationDomainMatrixRecord, WholeApplicationRecord, MIN_MEASURED_SAMPLES,
    WHOLE_APPLICATION_RECORD_SCHEMA_V2,
};
use xtask::gate::{Finding, GateCtx, GateError, Report};

/// Directory the suite writes every record into.
const EVIDENCE_DIR: &str = "release/evidence/benchmarks";

/// Every artifact this gate owns, matrix first, in the order it records them.
const OWNED: &[&str] = xtask::artifact_paths::WHOLE_APP_EVIDENCE_ARTIFACTS;

/// The matrix holding every per-domain record.
const MATRIX_ARTIFACT: &str = OWNED[0];

/// Flags this gate answers `--help` with.
const USAGE: &[&str] = &[
    "usage: whole-app-evidence [--write] [--backend ID] [--measured-samples N]",
    "  --write             measure every whole-application workload on this host's device",
    "                      and record all four artifacts under their provenance",
    "  --backend ID        acquire this dispatch backend instead of the first that admits",
    "  --measured-samples N samples per latency distribution (default 30, the stated minimum)",
];

/// One committed record and the domain whose file holds it.
///
/// The producer selects a file from the domain of the record it is writing, so
/// the pairing is the same one on both sides and is stated once here.
const DOMAIN_ARTIFACTS: [(ApplicationDomain, &str); 3] = [
    (ApplicationDomain::DenseNumerical, OWNED[1]),
    (ApplicationDomain::IrregularStateful, OWNED[2]),
    (ApplicationDomain::LatencySensitiveInteractive, OWNED[3]),
];

pub(crate) struct WholeAppEvidenceGate;

impl xtask::gate::GateBehavior for WholeAppEvidenceGate {
    fn usage(&self) -> &'static [&'static str] {
        USAGE
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(xtask::artifact_gate::settle_measured(
            ctx,
            OWNED,
            "whole-application evidence records",
            USAGE,
            parse_args,
            measure,
            audit,
        ))
    }
}

/// How the suite is measured when this gate writes.
struct Config {
    /// Dispatch backend to acquire, or the first that admits.
    backend: Option<String>,
    /// Samples behind each recorded latency distribution.
    measured_samples: usize,
}

/// The configuration to measure with, `None` for the option list.
fn parse_args(args: &[String]) -> Result<Option<Config>, String> {
    let mut backend = None;
    let mut measured_samples = MIN_MEASURED_SAMPLES;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--backend" => {
                let value = rest
                    .next()
                    .ok_or_else(|| "`--backend` names no backend id".to_string())?;
                backend = Some(value.clone());
            }
            "--measured-samples" => {
                let value = rest
                    .next()
                    .ok_or_else(|| "`--measured-samples` names no count".to_string())?;
                measured_samples = value.parse::<usize>().map_err(|error| {
                    format!("`--measured-samples {value}` is not a sample count: {error}")
                })?;
                if measured_samples < MIN_MEASURED_SAMPLES {
                    return Err(format!(
                        "`--measured-samples {measured_samples}` is below the {MIN_MEASURED_SAMPLES} \
                         samples a recorded latency distribution states"
                    ));
                }
            }
            other => return Err(format!("`{other}` is not a flag this gate reads")),
        }
    }
    Ok(Some(Config {
        backend,
        measured_samples,
    }))
}

/// Measure every workload on this host's device and record all four artifacts.
///
/// The producer probes the device once before any workload runs and refuses the
/// whole suite when none acquires, so a host without a device writes nothing
/// rather than a partial matrix.
fn measure(root: &Path, config: &Config, report: &mut Report) {
    let directory = root.join(EVIDENCE_DIR);
    match write_whole_application_evidence_artifacts(
        &directory,
        config.backend.as_deref(),
        config.measured_samples,
    ) {
        Ok(written) => {
            for path in written {
                report.note(format!("recorded {}", path.display()));
            }
        }
        Err(message) => report.find(Finding::in_file(
            std::path::PathBuf::from(EVIDENCE_DIR),
            message,
            "Run this gate's `--write` on a host with a linked dispatch backend and a working \
             driver, or name a backend that acquires with `--backend`.",
        )),
    }
}

/// Hold every committed record to the contract its own type states.
///
/// This runs no workload and needs no device. The producer used to be the only
/// reader of these files, which means the check that a committed record is
/// complete ran only on a release host with a GPU, and therefore never. The
/// flags describe a measurement, so an audit of the recorded set reads none of
/// them.
fn audit(root: &Path, _config: &Config, report: &mut Report) {
    let matrix: WholeApplicationDomainMatrixRecord = match read_record(root, MATRIX_ARTIFACT) {
        Ok(matrix) => matrix,
        Err(finding) => {
            report.find(finding);
            return;
        }
    };
    audit_matrix(&matrix, report);
    for (domain, path) in DOMAIN_ARTIFACTS {
        let record: WholeApplicationRecord = match read_record(root, path) {
            Ok(record) => record,
            Err(finding) => {
                report.find(finding);
                continue;
            }
        };
        audit_record(path, domain, &record, report);
        if !matrix.records.iter().any(|held| *held == record) {
            report.find(Finding::in_file(
                std::path::PathBuf::from(path),
                format!(
                    "`{path}` holds a record the domain matrix does not, so the two were not \
                     recorded by one run"
                ),
                "Rerun this gate's `--write`, which records the matrix and every per-domain file \
                 from the same measurement.",
            ));
        }
    }
}

/// Read one committed artifact body, past the provenance head it carries.
fn read_record<T: DeserializeOwned>(root: &Path, path: &str) -> Result<T, Finding> {
    let committed = std::fs::read_to_string(root.join(path)).map_err(|error| {
        Finding::in_file(
            std::path::PathBuf::from(path),
            format!("`{path}` cannot be read: {error}"),
            "Record the whole-application evidence with this gate's `--write` on a release host.",
        )
    })?;
    let (_, body) = xtask::artifact_gate::split_provenance(&committed);
    serde_json::from_str(&body).map_err(|error| {
        Finding::in_file(
            std::path::PathBuf::from(path),
            format!("`{path}` is not a whole-application record this gate judges: {error}"),
            "Rerun this gate's `--write` so the committed body is the one the producer renders.",
        )
    })
}

/// What the matrix has to state about the suite that produced it.
///
/// A readiness gap is not a defect here. Three of the pinned native
/// comparators are unvendored, so every record states the absence and names
/// the baseline, and the matrix carries that gap by construction. What has to
/// hold is that the gaps and the status are the ones its own records derive: a
/// matrix that claims completeness its records do not support, or carries a
/// gap no record states, was assembled from something other than one run.
fn audit_matrix(matrix: &WholeApplicationDomainMatrixRecord, report: &mut Report) {
    if matrix.schema_version != WHOLE_APPLICATION_RECORD_SCHEMA_V2 {
        report.find(Finding::in_file(
            std::path::PathBuf::from(MATRIX_ARTIFACT),
            format!(
                "the domain matrix records schema `{}`, and this gate judges \
                 `{WHOLE_APPLICATION_RECORD_SCHEMA_V2}`",
                matrix.schema_version
            ),
            "Rerun this gate's `--write` on a release host so the committed matrix carries the \
             current schema.",
        ));
    }
    if matrix.records.len() != ApplicationDomain::ALL.len() {
        report.find(Finding::in_file(
            std::path::PathBuf::from(MATRIX_ARTIFACT),
            format!(
                "the domain matrix holds {} record(s) and the workload catalog declares {} \
                 application domain(s)",
                matrix.records.len(),
                ApplicationDomain::ALL.len()
            ),
            "Rerun this gate's `--write`, which measures every registered domain in one run.",
        ));
    }
    let derived = derived_gaps(&matrix.records);
    if matrix.readiness_gaps != derived {
        report.find(Finding::in_file(
            std::path::PathBuf::from(MATRIX_ARTIFACT),
            format!(
                "the domain matrix records readiness gaps {:?} and its own records state {derived:?}",
                matrix.readiness_gaps
            ),
            "Rerun this gate's `--write`, which derives the matrix gaps from the records it \
             measured in the same run.",
        ));
    }
    let status =
        WholeApplicationDomainMatrixRecord::derive_status(&matrix.records, &matrix.readiness_gaps);
    if matrix.status != status {
        report.find(Finding::in_file(
            std::path::PathBuf::from(MATRIX_ARTIFACT),
            format!(
                "the domain matrix records status `{}` and the records and gaps it carries derive \
                 `{status}`",
                matrix.status
            ),
            "Rerun this gate's `--write`, which derives the status from the same records.",
        ));
    }
}

/// Every readiness gap the records themselves state, as the writer records it.
///
/// The writer unions the per-record gaps through a set, so the recorded list is
/// sorted and carries each gap once. Comparing against a first-seen ordering
/// would report a difference in ordering as a difference in content.
fn derived_gaps(records: &[WholeApplicationRecord]) -> Vec<String> {
    let gaps: std::collections::BTreeSet<String> = records
        .iter()
        .flat_map(WholeApplicationRecord::readiness_gaps)
        .map(|gap| gap.as_str().to_string())
        .collect();
    gaps.into_iter().collect()
}

/// What one per-domain record has to state about the run behind it.
fn audit_record(
    path: &str,
    domain: ApplicationDomain,
    record: &WholeApplicationRecord,
    report: &mut Report,
) {
    if record.domain != domain {
        report.find(Finding::in_file(
            std::path::PathBuf::from(path),
            format!(
                "`{path}` holds the `{}` domain, and its name states `{}`",
                record.domain.as_str(),
                domain.as_str()
            ),
            "Rerun this gate's `--write`, which selects each file from the domain of the record \
             it holds.",
        ));
    }
    if let Err(error) = record.validate_required_fields() {
        report.find(Finding::in_file(
            std::path::PathBuf::from(path),
            error.to_string(),
            "Rerun this gate's `--write` on a release host so every stated field comes from the \
             measurement.",
        ));
    }
    if let Err(refusal) = record.fail_closed_if_stale_or_partial() {
        report.find(Finding::in_file(
            std::path::PathBuf::from(path),
            refusal.to_string(),
            "Rerun this gate's `--write` on a release host so the record is neither stale nor \
             partial.",
        ));
    }
    if !record.executed_via_production_route() {
        report.find(Finding::in_file(
            std::path::PathBuf::from(path),
            format!(
                "`{path}` records workload `{}` without the production-route identities a device \
                 run passes through",
                record.workload_id
            ),
            "Rerun this gate's `--write` on a host whose dispatch backend compiles, admits and \
             submits the workload graph.",
        ));
    }
    if record.native_baseline_comparison.is_none() && record.native_baseline_unmeasured.is_none() {
        report.find(Finding::in_file(
            std::path::PathBuf::from(path),
            format!(
                "`{path}` records workload `{}` with neither a native baseline comparison nor the \
                 reason it has none",
                record.workload_id
            ),
            "Vendor and measure the pinned native comparator, or rerun this gate's `--write` so \
             the record names the baseline it could not resolve and why.",
        ));
    }
    if record.native_baseline_comparison.is_some() && record.native_baseline_unmeasured.is_some() {
        report.find(Finding::in_file(
            std::path::PathBuf::from(path),
            format!(
                "`{path}` records workload `{}` with both a native baseline comparison and a \
                 reason there is none",
                record.workload_id
            ),
            "Rerun this gate's `--write`; the producer records one or the other, so a record \
             carrying both was assembled from two runs.",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the gate exists because these four paths had no declared owner, so
    /// the descriptor's set and the set this gate reads have to be one list. A
    /// descriptor pointed at another list makes a written artifact an unowned
    /// write and a committed one unattributed, both in silence.
    #[test]
    fn the_descriptor_declares_exactly_the_artifacts_this_gate_records() {
        let descriptor = xtask::gate_metadata::descriptor("whole-app-evidence")
            .expect("Fix: `whole-app-evidence` must be a registered gate descriptor");
        assert_eq!(descriptor.artifacts, OWNED);
    }

    /// WHY: every owned path is read by exactly one arm of the audit. A path
    /// appended to the list with no domain beside it would be written by the
    /// producer, compared by nothing, and reported as covered.
    #[test]
    fn every_owned_artifact_has_exactly_one_reader() {
        for path in OWNED {
            let readers = usize::from(*path == MATRIX_ARTIFACT)
                + DOMAIN_ARTIFACTS
                    .iter()
                    .filter(|(_, owned)| owned == path)
                    .count();
            assert_eq!(
                readers, 1,
                "Fix: `{path}` must be read by exactly one audit arm, and {readers} read it"
            );
        }
    }

    /// WHY: one file per domain, and the producer selects the file from the
    /// domain. A domain added to the catalog with no file here would be
    /// measured into the matrix and never recorded on its own.
    #[test]
    fn every_application_domain_has_exactly_one_artifact() {
        let mut declared: Vec<&'static str> = DOMAIN_ARTIFACTS
            .iter()
            .map(|(domain, _)| domain.as_str())
            .collect();
        declared.sort_unstable();
        let mut catalog: Vec<&'static str> =
            ApplicationDomain::ALL.iter().map(|d| d.as_str()).collect();
        catalog.sort_unstable();
        assert_eq!(declared, catalog);
    }

    /// WHY: a sample count below the stated minimum makes a recorded percentile
    /// a number no distribution supports, and the flag was the only way in.
    #[test]
    fn a_sample_count_below_the_stated_minimum_is_refused() {
        let args = vec!["--measured-samples".to_string(), "4".to_string()];
        let message = match parse_args(&args) {
            Err(message) => message,
            Ok(_) => panic!("Fix: `--measured-samples 4` must be refused"),
        };
        assert!(
            message.contains("below the 30 samples"),
            "Fix: the refusal must name the minimum, got {message}"
        );
    }

    /// WHY: the minimum itself is a legal request, and an off-by-one in the
    /// bound would reject the default the producer runs with.
    #[test]
    fn the_stated_minimum_is_an_accepted_sample_count() {
        let args = vec![
            "--measured-samples".to_string(),
            MIN_MEASURED_SAMPLES.to_string(),
        ];
        match parse_args(&args) {
            Ok(Some(config)) => assert_eq!(config.measured_samples, MIN_MEASURED_SAMPLES),
            _ => panic!("Fix: `--measured-samples 30` must be accepted"),
        }
    }

    /// WHY: an unknown flag used to be ignored by every hand-rolled parser in
    /// this crate, so `--backend-id cuda` measured on whatever admitted first.
    #[test]
    fn an_unknown_flag_is_refused_rather_than_ignored() {
        let args = vec!["--backend-id".to_string(), "cuda".to_string()];
        assert!(parse_args(&args).is_err());
    }

    /// WHY: a gap the matrix carries and no record states means the matrix and
    /// the records came from different runs. With no record at all, the only
    /// honest gap list is empty, so a non-empty one has to be reported.
    #[test]
    fn a_gap_no_record_states_is_a_finding() {
        let matrix = WholeApplicationDomainMatrixRecord {
            schema_version: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
            total_workloads: 3,
            covered_domains: 3,
            records: Vec::new(),
            backend_id: "cuda".to_string(),
            status: "measured_with_unstated_fields".to_string(),
            readiness_gaps: vec!["p99_latency_ns".to_string()],
            generated_at_utc: "1970-01-01T00:00:00Z".to_string(),
        };
        let mut report = Report::clean();
        audit_matrix(&matrix, &mut report);
        let rendered = format!("{report:?}");
        assert!(
            rendered.contains("p99_latency_ns"),
            "Fix: the finding must name the gap the matrix carries, got {rendered}"
        );
    }

    /// WHY: the status is the field a reader trusts, and it is derived from the
    /// records and gaps. `no_records` with a `complete` status is the exact
    /// shape a matrix assembled by hand takes, and every other check passes on
    /// it.
    #[test]
    fn a_status_the_records_do_not_derive_is_a_finding() {
        let matrix = WholeApplicationDomainMatrixRecord {
            schema_version: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
            total_workloads: 3,
            covered_domains: 3,
            records: Vec::new(),
            backend_id: "cuda".to_string(),
            status: "complete".to_string(),
            readiness_gaps: Vec::new(),
            generated_at_utc: "1970-01-01T00:00:00Z".to_string(),
        };
        let mut report = Report::clean();
        audit_matrix(&matrix, &mut report);
        let rendered = format!("{report:?}");
        assert!(
            rendered.contains("derive `no_records`"),
            "Fix: the finding must name the status the records derive, got {rendered}"
        );
    }

    /// WHY: three of the pinned native comparators are unvendored, so the
    /// recorded gap list is not empty on any host. A gate that demanded an
    /// empty one could never pass and would be deleted rather than read.
    #[test]
    fn the_committed_matrix_gaps_are_the_ones_its_records_state() {
        let root = xtask::checkout::checkout_root();
        let matrix: WholeApplicationDomainMatrixRecord = match read_record(&root, MATRIX_ARTIFACT) {
            Ok(matrix) => matrix,
            Err(_) => return,
        };
        assert_eq!(matrix.readiness_gaps, derived_gaps(&matrix.records));
    }

    /// WHY: a body written against an older schema deserializes into the
    /// current type and would otherwise pass every other check in the audit.
    #[test]
    fn a_matrix_recording_a_foreign_schema_is_a_finding() {
        let matrix = WholeApplicationDomainMatrixRecord {
            schema_version: "vyre.whole-application-record.v1".to_string(),
            total_workloads: 3,
            covered_domains: 3,
            records: Vec::new(),
            backend_id: "cuda".to_string(),
            status: "complete".to_string(),
            readiness_gaps: Vec::new(),
            generated_at_utc: "1970-01-01T00:00:00Z".to_string(),
        };
        let mut report = Report::clean();
        audit_matrix(&matrix, &mut report);
        let rendered = format!("{:?}", report);
        assert!(
            rendered.contains("vyre.whole-application-record.v1"),
            "Fix: the finding must name the schema the body records, got {rendered}"
        );
    }
}
