//! Running every whole-application workload and recording what it did.
//!
//! The device is probed once, before any workload runs, so a host with no
//! dispatch device produces a refusal instead of a partial matrix.

use super::*;

/// Release evidence domain matrix holding every whole-application record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeApplicationDomainMatrixRecord {
    /// Schema version.
    pub schema_version: String,
    /// Total registered whole-application cases.
    pub total_workloads: usize,
    /// Total distinct application domains covered.
    pub covered_domains: usize,
    /// Individual workload measurement records.
    pub records: Vec<WholeApplicationRecord>,
    /// Backend that executed every record in this matrix.
    pub backend_id: String,
    /// Outcome derived from the records the matrix holds.
    pub status: String,
    /// Required fields no record in this matrix states.
    pub readiness_gaps: Vec<String>,
    /// UTC timestamp of generation.
    pub generated_at_utc: String,
}

impl WholeApplicationDomainMatrixRecord {
    /// Status a matrix with these records and gaps states.
    fn derive_status(records: &[WholeApplicationRecord], readiness_gaps: &[String]) -> String {
        if records.is_empty() {
            "no_records".to_string()
        } else if readiness_gaps.is_empty() {
            "complete".to_string()
        } else {
            "measured_with_unstated_fields".to_string()
        }
    }
}

/// Execute every whole-application workload on one acquired device.
///
/// The device is probed once, before any workload runs, so a host with no
/// dispatch device produces the refusal instead of a partial matrix.
///
/// # Errors
///
/// Returns the refusal text when no device acquires, and the workload
/// diagnostic when a compile, submission, parity comparison, or record
/// validation fails.
pub fn generate_whole_application_evidence_suite(
    backend_id: Option<&str>,
    measured_samples: usize,
) -> Result<WholeApplicationDomainMatrixRecord, String> {
    let device = WholeApplicationDevice::probe(backend_id).map_err(|err| err.to_string())?;
    let workloads = all_whole_application_workloads();
    let mut records = Vec::with_capacity(workloads.len());
    let mut domain_set = BTreeSet::new();
    let mut readiness_gaps = BTreeSet::new();

    for workload in &workloads {
        domain_set.insert(workload.domain);
        let record = workload.execute_and_measure(&device, measured_samples)?;
        for gap in record.readiness_gaps() {
            readiness_gaps.insert(gap.as_str().to_string());
        }
        records.push(record);
    }

    let readiness_gaps: Vec<String> = readiness_gaps.into_iter().collect();
    let status = WholeApplicationDomainMatrixRecord::derive_status(&records, &readiness_gaps);
    Ok(WholeApplicationDomainMatrixRecord {
        schema_version: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
        total_workloads: workloads.len(),
        covered_domains: domain_set.len(),
        records,
        backend_id: device.backend_id().to_string(),
        status,
        readiness_gaps,
        generated_at_utc: crate::workloads::provenance::utc_timestamp(std::time::SystemTime::now())?,
    })
}

/// Write every whole-application evidence artifact into one directory.
///
/// # Errors
///
/// Returns a diagnostic when the directory cannot be created, when the suite
/// cannot be measured, or when a file cannot be serialized or written.
pub fn write_whole_application_evidence_artifacts(
    artifacts_dir: &Path,
    backend_id: Option<&str>,
    measured_samples: usize,
) -> Result<Vec<PathBuf>, String> {
    let matrix = generate_whole_application_evidence_suite(backend_id, measured_samples)?;

    let root = xtask::checkout::checkout_root();
    let mut written_paths = Vec::new();

    let matrix_path = artifacts_dir.join("whole-application-domain-matrix.json");
    record_whole_application_artifact(&root, &matrix_path, &matrix)?;
    written_paths.push(matrix_path);

    for record in &matrix.records {
        let file_name = match record.domain {
            ApplicationDomain::DenseNumerical => "whole-app-dense-numerical-pipeline.json",
            ApplicationDomain::IrregularStateful => "whole-app-irregular-stateful-traversal.json",
            ApplicationDomain::LatencySensitiveInteractive => {
                "whole-app-interactive-event-pipeline.json"
            }
        };
        let record_path = artifacts_dir.join(file_name);
        record_whole_application_artifact(&root, &record_path, record)?;
        written_paths.push(record_path);
    }

    Ok(written_paths)
}

/// Record one whole-application artifact under the devices that measured it.
///
/// These records carry wall times a device produced, so the measurement class
/// is `Device` and the stamp names every device the run could see. The suite
/// refuses to run without one, so a host-only record here would be false.
fn record_whole_application_artifact(
    root: &Path,
    path: &Path,
    body: &impl serde::Serialize,
) -> Result<(), String> {
    xtask::artifact_gate::write_recorded(
        root,
        path,
        xtask::evidence_record::MeasurementRecord::device(),
        body,
    )
}
