use crate::api::suite::SuiteKind;
use crate::report::json::ReportSchema;
use crate::runner::{execute_suite, refuse_unoptimized_release_measurement, RunConfig};

pub(super) fn execute_run_matrix(
    registry: &crate::registry::BenchRegistry,
    suite: &SuiteKind,
    config: &RunConfig,
) -> anyhow::Result<Vec<ReportSchema>> {
    refuse_unoptimized_release_measurement(suite)?;
    match suite {
        SuiteKind::CrossBackend if config.backend_id.is_none() => {
            let mut reports = Vec::new();
            for backend in dispatch_backend_ids()? {
                let mut cfg = config.clone();
                cfg.backend_id = Some(backend.to_string());
                reports.push(execute_suite(registry, suite, &cfg));
            }
            Ok(reports)
        }
        SuiteKind::Sweep if config.workgroup_override.is_none() => {
            let mut reports = Vec::new();
            for size in [32, 64, 128, 256] {
                let mut cfg = config.clone();
                cfg.workgroup_override = Some([size, 1, 1]);
                reports.push(execute_suite(registry, suite, &cfg));
            }
            Ok(reports)
        }
        _ => Ok(vec![execute_suite(registry, suite, config)]),
    }
}

fn dispatch_backend_ids() -> anyhow::Result<Vec<&'static str>> {
    let registered = vyre_registry_link::backend::live_backend_registry_by_precedence()?;
    let mut backends = Vec::new();
    for backend in registered {
        if vyre_driver::backend_dispatches(backend.id)? {
            backends.push(backend.id);
        }
    }
    Ok(backends)
}

pub(super) fn write_run_reports(reports: &[ReportSchema], output: &str) -> anyhow::Result<()> {
    let output = std::path::Path::new(output);
    let root = xtask::checkout::checkout_root();
    if reports.len() == 1 {
        return record_run_report(&root, output, &reports[0]);
    }
    std::fs::create_dir_all(output)?;
    for (index, report) in reports.iter().enumerate() {
        let suite = sanitize_path_component(&report.suite);
        let backend = report
            .selected_backend
            .as_deref()
            .map(sanitize_path_component)
            .unwrap_or_else(|| "unknown-backend".to_string());
        let path = output.join(format!("{suite}-{backend}-{index:03}.json"));
        record_run_report(&root, &path, report)?;
    }
    Ok(())
}

/// Write one report, recorded under its provenance when it is evidence.
///
/// A report under `release/evidence` is read by someone who no longer has the
/// tree, the host or the device, so it goes through the recorded writer and
/// names all three. A report written anywhere else is a scratch rendering of
/// the same run and carries no head, which is the same split
/// `json_document::write` enforces for every other generator.
fn record_run_report(
    root: &std::path::Path,
    path: &std::path::Path,
    report: &ReportSchema,
) -> anyhow::Result<()> {
    if xtask::artifact_gate::records_provenance(path) {
        return xtask::artifact_gate::write_recorded(
            root,
            path,
            xtask::evidence_record::MeasurementRecord::device(),
            report,
        )
        .map_err(|error| anyhow::anyhow!("{error}"));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        path,
        format!("{}\n", crate::report::json::generate_json_report(report)?),
    )?;
    Ok(())
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
