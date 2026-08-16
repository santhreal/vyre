use vyre_bench::probes::environment::build_profile;

use crate::api::suite::SuiteKind;
use crate::report::json::ReportSchema;
use crate::runner::{execute_suite, RunConfig};

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

/// Refuse to measure the release suite with a build that carries debug checks.
///
/// The release suite is the one whose numbers are published, and an unoptimized
/// harness inflates every speedup it reports: the CPU baseline runs the scan
/// without optimization while device time is set by the device. A run that cannot
/// be published must not produce a document that looks publishable, so it fails
/// here rather than writing one.
fn refuse_unoptimized_release_measurement(suite: &SuiteKind) -> anyhow::Result<()> {
    if matches!(suite, SuiteKind::Release) && build_profile() != "release" {
        anyhow::bail!(
            "the release suite measures published numbers and this harness is a {} build. Fix: \
             rerun with `--release`.",
            build_profile()
        );
    }
    Ok(())
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
    if reports.len() == 1 {
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            output,
            format!(
                "{}\n",
                crate::report::json::generate_json_report(&reports[0])?
            ),
        )?;
        return Ok(());
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
        std::fs::write(
            path,
            format!("{}\n", crate::report::json::generate_json_report(report)?),
        )?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The release suite is refused unless the harness is optimized, and every
    /// other suite is unaffected.
    ///
    /// The generator that writes `release/evidence/benchmarks` spawned this
    /// harness without `--release`, so it published a CPU baseline that was
    /// mostly missing optimization. One assertion per branch: the refusal fires
    /// exactly when the profile is not `release`, and never for another suite.
    #[test]
    fn only_an_optimized_build_may_measure_the_release_suite() {
        let release = refuse_unoptimized_release_measurement(&SuiteKind::Release);
        if build_profile() == "release" {
            assert!(
                release.is_ok(),
                "Fix: an optimized build must be allowed to measure the release suite."
            );
        } else {
            let error = release.expect_err(
                "Fix: a debug build must be refused before it writes release evidence.",
            );
            assert!(
                error.to_string().contains("--release"),
                "Fix: the refusal must name the flag that repairs it, got `{error}`."
            );
        }
        for suite in [
            SuiteKind::Smoke,
            SuiteKind::Deep,
            SuiteKind::Gpu,
            SuiteKind::Sweep,
        ] {
            assert!(
                refuse_unoptimized_release_measurement(&suite).is_ok(),
                "Fix: only the release suite publishes numbers; `{suite:?}` must run under any \
                 profile."
            );
        }
    }
}
