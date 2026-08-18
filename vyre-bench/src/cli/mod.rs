use clap::{Parser, Subcommand};

use crate::api::suite::SuiteKind;
use crate::runner::RunConfig;

mod bundle;
mod compare;
mod dashboard;
mod evolve_server;
mod registry;
mod report_io;
mod run;
#[cfg(test)]
mod tests;
#[cfg(not(test))]
use bundle::validate_benchmark_bundle;
#[cfg(test)]
use bundle::*;
#[cfg(test)]
use compare::{build_comparison_artifact, write_comparison_artifact};
use compare::{compare_reports, load_comparison_artifact, validate_comparison_expectations};
use dashboard::generate_dashboard;
#[cfg(test)]
use dashboard::{generate_index_html, generate_scorecard_md};
use registry::{explain_case, list_cases};
use report_io::{load_report, validate_report_expectations};
use run::{execute_run_matrix, write_run_reports};

#[derive(Parser)]
#[command(name = "vyre-bench")]
#[command(about = "Canonical performance and evolution harness for Vyre", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run {
        #[arg(long)]
        suite: String,
        #[arg(long, default_value = "table")]
        format: String,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        enforce_budgets: bool,
        #[arg(long = "case")]
        case_ids: Vec<String>,
        #[arg(long, default_value_t = 3)]
        warmup_samples: usize,
        #[arg(long)]
        measured_samples: Option<usize>,
        #[arg(long, default_value_t = 30)]
        sample_timeout_secs: u64,
        #[arg(long)]
        snapshot_on_pass: bool,
        #[arg(long, default_value_t = 1)]
        determinism_runs: usize,
        #[arg(long)]
        workgroup_size: Option<u32>,
        #[arg(long)]
        roofline_only: bool,
        #[arg(long)]
        output: Option<String>,
    },
    Compare {
        #[arg(long)]
        baseline: String,
        #[arg(long)]
        candidate: String,
        #[arg(long)]
        output: Option<String>,
    },
    ValidateReport {
        #[arg(long)]
        path: String,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        total_cases: Option<usize>,
        #[arg(long)]
        failed: Option<usize>,
    },
    ValidateComparison {
        #[arg(long)]
        path: String,
        #[arg(long)]
        baseline_backend: String,
        #[arg(long)]
        candidate_backend: String,
        #[arg(long = "case")]
        case_ids: Vec<String>,
    },
    ValidateBenchmarkBundle {
        #[arg(long)]
        dir: String,
        #[arg(long)]
        manifest_output: Option<String>,
        #[arg(long)]
        manifest_input: Option<String>,
    },
    SnapshotDiff {
        #[arg(long)]
        base: String,
    },
    List {
        #[arg(long, default_value = "table")]
        format: String,
    },
    Explain {
        id: String,
    },
    Dashboard {
        #[arg(long, default_value = "dashboard")]
        output: String,
    },
    /// Print the release workload matrix.
    ///
    /// This command prints and never writes. The committed
    /// `release/evidence/benchmarks/release-workload-matrix.json` has one
    /// producer, the `release-workload-matrix` gate, which stamps a provenance
    /// head this serialization does not carry.
    ReleaseMatrix {
        #[arg(long, default_value = "table")]
        format: String,
        #[arg(long)]
        enforce: bool,
    },
    EvolveServer,
}

pub(super) fn run_cli() -> anyhow::Result<()> {
    env_logger::init();
    run_cli_with(std::env::args_os())
}

fn run_cli_with<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = Cli::parse_from(args);
    match &cli.command {
        Commands::Run {
            suite,
            format,
            backend,
            enforce_budgets,
            case_ids,
            warmup_samples,
            measured_samples,
            sample_timeout_secs,
            snapshot_on_pass,
            determinism_runs,
            workgroup_size,
            roofline_only,
            output,
        } => {
            let suite_kind: SuiteKind = suite
                .parse()
                .map_err(|error: String| anyhow::anyhow!("{error}"))?;
            let registry = crate::registry::collect_all();
            let config = RunConfig {
                backend_id: backend.clone(),
                enforce_budgets: *enforce_budgets,
                case_ids: case_ids.clone(),
                warmup_samples: *warmup_samples,
                measured_samples: *measured_samples,
                sample_timeout: std::time::Duration::from_secs(*sample_timeout_secs),
                determinism_runs: *determinism_runs,
                workgroup_override: workgroup_size.map(|size| [size, 1, 1]),
                baseline_warmup_runs: 0,
                snapshot_on_pass: *snapshot_on_pass,
            };
            let reports = execute_run_matrix(&registry, &suite_kind, &config)?;
            if let Some(output) = output {
                write_run_reports(&reports, output)?;
            }
            for report in &reports {
                crate::runner::print_report(report, format, *roofline_only)?;
            }

            let failed: usize = reports.iter().map(|report| report.summary.failed).sum();
            if failed > 0 {
                anyhow::bail!("{failed} benchmark case(s) failed");
            }
        }
        Commands::Compare {
            baseline,
            candidate,
            output,
        } => {
            let baseline_report = load_report(baseline)?;
            let candidate_report = load_report(candidate)?;
            compare_reports(&baseline_report, &candidate_report, output.as_deref())?;
        }
        Commands::ValidateReport {
            path,
            backend,
            total_cases,
            failed,
        } => {
            let report = load_report(path)?;
            validate_report_expectations(&report, backend.as_deref(), *total_cases, *failed)?;
            let selected = report.selected_backend.as_deref().unwrap_or("unknown");
            let timing_quality = report
                .backend_profile
                .as_ref()
                .map(|profile| profile.timing_quality.as_str())
                .unwrap_or("unknown");
            println!(
                "report_valid path={} selected_backend={} timing_quality={}",
                path, selected, timing_quality
            );
        }
        Commands::ValidateComparison {
            path,
            baseline_backend,
            candidate_backend,
            case_ids,
        } => {
            let comparison = load_comparison_artifact(path)?;
            validate_comparison_expectations(
                &comparison,
                baseline_backend,
                candidate_backend,
                case_ids,
            )?;
            println!(
                "comparison_valid path={} baseline_backend={} candidate_backend={} cases={}",
                path,
                comparison.baseline.profile_backend,
                comparison.candidate.profile_backend,
                comparison.cases.len()
            );
        }
        Commands::ValidateBenchmarkBundle {
            dir,
            manifest_output,
            manifest_input,
        } => {
            let manifest = validate_benchmark_bundle(
                dir,
                manifest_output.as_deref(),
                manifest_input.as_deref(),
            )?;
            println!(
                "benchmark_bundle_valid dir={} artifacts={} bundle_blake3={}",
                dir, manifest.artifact_count, manifest.bundle_blake3
            );
        }
        Commands::SnapshotDiff { base } => {
            let snapshots_dir = std::path::Path::new("snapshots");
            let path = snapshots_dir.join(format!("{}.json", base));
            if !path.exists() {
                anyhow::bail!("snapshot for commit `{}` not found in snapshots/", base);
            }
            let baseline_report = load_report(&path.to_string_lossy())?;
            let registry = crate::registry::collect_all();
            let config = RunConfig::default();
            let mut current_reports = execute_run_matrix(&registry, &SuiteKind::Release, &config)?;
            let current_report = current_reports.pop().ok_or_else(|| {
                anyhow::anyhow!("the release suite returned no report to compare against")
            })?;
            compare_reports(&baseline_report, &current_report, None)?;
        }
        Commands::List { format } => list_cases(format)?,
        Commands::Explain { id } => explain_case(id)?,
        Commands::Dashboard { output } => generate_dashboard(output)?,
        Commands::ReleaseMatrix { format, enforce } => {
            let registry = crate::registry::collect_all();
            let matrix = crate::release_matrix::build_release_matrix(&registry);
            print!(
                "{}",
                crate::release_matrix::render_release_matrix(&matrix, format)?
            );
            if *enforce {
                crate::release_matrix::enforce_release_matrix(&matrix)?;
            }
        }
        Commands::EvolveServer => evolve_server::run_evolve_server()?,
    }
    Ok(())
}
