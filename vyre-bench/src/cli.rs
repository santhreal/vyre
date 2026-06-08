#![allow(missing_docs)]

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::api::case::BenchId;
use crate::api::suite::SuiteKind;
use crate::report::json::ReportSchema;
use crate::runner::{execute_suite, RunConfig};

const MAX_REPORT_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const BENCHMARK_BUNDLE_SCHEMA: &str = "vyre-bench.bundle.v1";
const MAC_BENCHMARK_BUNDLE_CASE_ID: &str = "foundation.elementwise.add.1m";
const MAC_BENCHMARK_BUNDLE_BASELINE_BACKEND: &str = "wgpu";
const MAC_BENCHMARK_BUNDLE_CANDIDATE_BACKEND: &str = "metal";
const MAC_BENCHMARK_BUNDLE_COMPARISONS: &[(&str, &str, &str, &str)] = &[
    ("wgpu-vs-metal.json", "wgpu-vs-metal.txt", "wgpu", "metal"),
    (
        "cpu-ref-vs-metal.json",
        "cpu-ref-vs-metal.txt",
        "cpu-ref",
        "metal",
    ),
];
const BENCHMARK_BUNDLE_REQUIRED_ARTIFACTS: &[(&str, &str)] = &[
    ("cpu-ref.json", "backend_report"),
    ("wgpu.json", "backend_report"),
    ("metal.json", "backend_report"),
    ("wgpu-vs-metal.json", "comparison_json"),
    ("wgpu-vs-metal.txt", "comparison_text"),
    ("cpu-ref-vs-metal.json", "comparison_json"),
    ("cpu-ref-vs-metal.txt", "comparison_text"),
];

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
    ReleaseMatrix {
        #[arg(long, default_value = "table")]
        format: String,
        #[arg(long)]
        output: Option<String>,
        #[arg(long)]
        enforce: bool,
    },
    EvolveServer,
}

pub fn run_cli() -> anyhow::Result<()> {
    env_logger::init();
    run_cli_with(std::env::args_os())
}

pub fn run_cli_with<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    crate::link_benchmark_backend_registrations();
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
            let reports = execute_run_matrix(&registry, suite_kind, &config)?;
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
            validate_report_expectations(
                &report,
                backend.as_deref(),
                *total_cases,
                *failed,
            )?;
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
            let current_report = execute_suite(&registry, SuiteKind::Release, &config);
            compare_reports(&baseline_report, &current_report, None)?;
        }
        Commands::List { format } => list_cases(format)?,
        Commands::Explain { id } => explain_case(id)?,
        Commands::Dashboard { output } => generate_dashboard(output)?,
        Commands::ReleaseMatrix {
            format,
            output,
            enforce,
        } => {
            let registry = crate::registry::collect_all();
            let matrix = crate::release_matrix::build_release_matrix(&registry);
            crate::release_matrix::emit_release_matrix(&matrix, format, output.as_deref())?;
            if *enforce {
                crate::release_matrix::enforce_release_matrix(&matrix)?;
            }
        }
        Commands::EvolveServer => crate::evolve::server::run_evolve_server()?,
    }
    Ok(())
}

fn execute_run_matrix(
    registry: &crate::registry::BenchRegistry,
    suite: SuiteKind,
    config: &RunConfig,
) -> anyhow::Result<Vec<ReportSchema>> {
    match suite {
        SuiteKind::CrossBackend if config.backend_id.is_none() => {
            let mut reports = Vec::new();
            for backend in dispatch_backend_ids() {
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

fn dispatch_backend_ids() -> Vec<&'static str> {
    vyre_driver::backend::registered_backends_by_precedence_slice()
        .iter()
        .filter(|backend| vyre_driver::backend::backend_dispatches(backend.id))
        .map(|backend| backend.id)
        .collect()
}

fn write_run_reports(reports: &[ReportSchema], output: &str) -> anyhow::Result<()> {
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

fn list_cases(format: &str) -> anyhow::Result<()> {
    let registry = crate::registry::collect_all();
    let metadata: Vec<_> = registry.iter().map(|case| case.metadata()).collect();
    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&metadata)?);
        return Ok(());
    }
    for meta in metadata {
        println!("{} ({}) {}", meta.id.0, meta.name, meta.description);
    }
    Ok(())
}

fn explain_case(id: &str) -> anyhow::Result<()> {
    let registry = crate::registry::collect_all();
    let case = registry
        .get(&BenchId(id.to_string()))
        .ok_or_else(|| anyhow::anyhow!("unknown benchmark `{id}`"))?;
    let mut details = BTreeMap::new();
    details.insert("metadata", serde_json::to_value(case.metadata())?);
    details.insert("requirements", serde_json::to_value(case.requirements())?);
    details.insert(
        "performance_contract",
        serde_json::to_value(case.performance_contract())?,
    );
    println!("{}", serde_json::to_string_pretty(&details)?);
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ComparisonArtifact {
    schema: String,
    baseline: ComparisonSide,
    candidate: ComparisonSide,
    cases: Vec<ComparisonCase>,
    regressed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ComparisonSide {
    run_id: String,
    suite: String,
    selected_backend: String,
    profile_backend: String,
    timing_quality: String,
    source_fingerprint: String,
    source_tree_fingerprint: String,
    total_cases: usize,
    failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ComparisonCase {
    id: String,
    baseline_p50_ns: u64,
    candidate_p50_ns: u64,
    baseline_mean_ns: f64,
    candidate_mean_ns: f64,
    delta_fraction: Option<f64>,
    delta_percent: Option<f64>,
    p_value: Option<f64>,
    verdict: String,
    regressed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkBundleManifest {
    schema: String,
    provenance: BenchmarkBundleProvenance,
    artifact_count: usize,
    bundle_blake3: String,
    artifacts: Vec<BundleArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct BenchmarkBundleProvenance {
    validator: String,
    validator_version: String,
    suite: String,
    case_id: String,
    report_backends: Vec<String>,
    baseline_backend: String,
    candidate_backend: String,
    comparison_pairs: Vec<String>,
    source_fingerprint: String,
    source_tree_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BundleArtifact {
    path: String,
    kind: String,
    bytes: u64,
    blake3: String,
}

#[derive(Serialize)]
struct BundleHashMaterial<'a> {
    schema: &'static str,
    provenance: &'a BenchmarkBundleProvenance,
    artifacts: &'a [BundleArtifact],
}

fn compare_reports(
    baseline: &ReportSchema,
    candidate: &ReportSchema,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let comparison = build_comparison_artifact(baseline, candidate)?;
    print_comparison_artifact(&comparison);
    if let Some(output) = output {
        write_comparison_artifact(&comparison, output)?;
    }
    if comparison.regressed {
        anyhow::bail!("One or more cases regressed by >1σ");
    }
    Ok(())
}

fn build_comparison_artifact(
    baseline: &ReportSchema,
    candidate: &ReportSchema,
) -> anyhow::Result<ComparisonArtifact> {
    let baseline_cases: BTreeMap<_, _> = baseline
        .cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect();
    let mut cases = Vec::with_capacity(candidate.cases.len());
    for case in &candidate.cases {
        let baseline_case = baseline_cases
            .get(case.id.as_str())
            .ok_or_else(|| anyhow::anyhow!("candidate case `{}` has no baseline", case.id))?;
        let baseline_stats = baseline_case
            .metrics
            .get("wall_ns")
            .ok_or_else(|| anyhow::anyhow!("baseline case `{}` lacks wall_ns", case.id))?;
        let candidate_stats = case
            .metrics
            .get("wall_ns")
            .ok_or_else(|| anyhow::anyhow!("candidate case `{}` lacks wall_ns", case.id))?;
        let baseline_p50 = baseline_stats.p50;
        let candidate_p50 = candidate_stats.p50;
        let delta_fraction = if baseline_p50 == 0 {
            None
        } else {
            Some((candidate_p50 as f64 - baseline_p50 as f64) / baseline_p50 as f64)
        };
        let p_value = welch_p_value(baseline_stats, candidate_stats);
        let verdict = compare_verdict(delta_fraction, p_value);
        let regressed = candidate_stats.mean > baseline_stats.mean + baseline_stats.stddev;
        cases.push(ComparisonCase {
            id: case.id.clone(),
            baseline_p50_ns: baseline_p50,
            candidate_p50_ns: candidate_p50,
            baseline_mean_ns: baseline_stats.mean,
            candidate_mean_ns: candidate_stats.mean,
            delta_fraction,
            delta_percent: delta_fraction.map(|delta| delta * 100.0),
            p_value,
            verdict: verdict.to_string(),
            regressed,
        });
    }
    let regressed = cases.iter().any(|case| case.regressed);
    Ok(ComparisonArtifact {
        schema: "vyre-bench.compare.v1".to_string(),
        baseline: comparison_side(baseline),
        candidate: comparison_side(candidate),
        cases,
        regressed,
    })
}

fn comparison_side(report: &ReportSchema) -> ComparisonSide {
    let (profile_backend, timing_quality) = report
        .backend_profile
        .as_ref()
        .map(|profile| (profile.backend.as_str(), profile.timing_quality.as_str()))
        .unwrap_or(("unknown", "unknown"));
    ComparisonSide {
        run_id: report.run_id.clone(),
        suite: report.suite.clone(),
        selected_backend: report
            .selected_backend
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        profile_backend: profile_backend.to_string(),
        timing_quality: timing_quality.to_string(),
        source_fingerprint: report.source_fingerprint.clone(),
        source_tree_fingerprint: report.source_tree_fingerprint.clone(),
        total_cases: report.summary.total_cases,
        failed: report.summary.failed,
    }
}

fn print_comparison_artifact(comparison: &ComparisonArtifact) {
    print_compare_profile("baseline", &comparison.baseline);
    print_compare_profile("candidate", &comparison.candidate);
    println!(
        "{:<30} | {:<12} | {:<12} | {:<10} | {:<12} | {:<10}",
        "Benchmark", "Baseline", "Candidate", "Delta", "p-value", "Verdict"
    );
    println!(
        "------------------------------------------------------------------------------------------------"
    );
    for case in &comparison.cases {
        let delta = case
            .delta_percent
            .map(|value| format!("{value:+.2}%"))
            .unwrap_or_else(|| "n/a".to_string());
        let p_value = case
            .p_value
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{:<30} | {:<12} | {:<12} | {:<10} | {:<12} | {:<10}",
            case.id,
            case.baseline_p50_ns,
            case.candidate_p50_ns,
            delta,
            p_value,
            case.verdict
        );
    }
}

fn print_compare_profile(label: &str, side: &ComparisonSide) {
    println!(
        "{label}_selected_backend={} {label}_profile_backend={} {label}_timing_quality={}",
        side.selected_backend, side.profile_backend, side.timing_quality
    );
}

fn write_comparison_artifact(comparison: &ComparisonArtifact, path: &str) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, comparison)?;
    Ok(())
}

fn load_comparison_artifact(path: &str) -> anyhow::Result<ComparisonArtifact> {
    let bytes = read_report_bounded(std::path::Path::new(path))?;
    parse_comparison_artifact(&bytes)
}

fn parse_comparison_artifact(bytes: &[u8]) -> anyhow::Result<ComparisonArtifact> {
    Ok(serde_json::from_slice(bytes)?)
}

fn validate_comparison_expectations(
    comparison: &ComparisonArtifact,
    baseline_backend: &str,
    candidate_backend: &str,
    case_ids: &[String],
) -> anyhow::Result<()> {
    if comparison.schema != "vyre-bench.compare.v1" {
        anyhow::bail!(
            "comparison schema `{}` is not `vyre-bench.compare.v1`. Fix: regenerate comparison with current vyre-bench compare.",
            comparison.schema
        );
    }
    if comparison.baseline.profile_backend != baseline_backend {
        anyhow::bail!(
            "comparison baseline profile backend `{}` does not match expected `{baseline_backend}`. Fix: compare the intended baseline report.",
            comparison.baseline.profile_backend
        );
    }
    if comparison.candidate.profile_backend != candidate_backend {
        anyhow::bail!(
            "comparison candidate profile backend `{}` does not match expected `{candidate_backend}`. Fix: compare the intended candidate report.",
            comparison.candidate.profile_backend
        );
    }
    for (label, side) in [
        ("baseline", &comparison.baseline),
        ("candidate", &comparison.candidate),
    ] {
        if !matches!(
            side.timing_quality.as_str(),
            "host_only" | "host_enqueue_wait" | "device_timestamps" | "hardware_counters"
        ) {
            anyhow::bail!(
                "{label} timing quality `{}` is invalid. Fix: regenerate comparison from reports with DeviceTimingQuality::as_str() values.",
                side.timing_quality
            );
        }
    }
    if comparison.cases.is_empty() {
        anyhow::bail!("comparison contains zero cases. Fix: compare reports with overlapping benchmark cases.");
    }
    for case_id in case_ids {
        if !comparison.cases.iter().any(|case| case.id == *case_id) {
            anyhow::bail!(
                "comparison artifact lacks case `{case_id}`. Fix: compare reports generated with the intended --case selection."
            );
        }
    }
    let derived_regressed = comparison.cases.iter().any(|case| case.regressed);
    if comparison.regressed != derived_regressed {
        anyhow::bail!(
            "comparison regressed={} contradicts case-derived regressed={derived_regressed}. Fix: regenerate comparison from case evidence.",
            comparison.regressed
        );
    }
    Ok(())
}

fn validate_benchmark_bundle(
    dir: &str,
    manifest_output: Option<&str>,
    manifest_input: Option<&str>,
) -> anyhow::Result<BenchmarkBundleManifest> {
    let dir = std::path::Path::new(dir);
    if !dir.is_dir() {
        anyhow::bail!(
            "benchmark bundle dir `{}` is not a directory. Fix: pass the VYRE_MACBOOK_BENCH_OUTPUT_DIR directory produced by scripts/check_metal_macbook.sh benchmark.",
            dir.display()
        );
    }
    let mut artifacts = Vec::new();
    let mut reports = Vec::new();
    for backend in ["cpu-ref", "wgpu", "metal"] {
        let path = dir.join(format!("{backend}.json"));
        let bytes = read_report_bounded(&path).map_err(|error| {
            anyhow::anyhow!(
                "failed to load benchmark report `{}`: {error}. Fix: rerun the benchmark gate so {backend}.json is produced.",
                path.display()
            )
        })?;
        let report = parse_report(&bytes, &path.to_string_lossy())?;
        validate_report_expectations(&report, Some(backend), Some(1), Some(0))?;
        reports.push(report);
        artifacts.push(bundle_artifact(dir, &path, "backend_report", &bytes)?);
    }
    let mut comparisons = Vec::new();
    for (json_name, text_name, baseline_backend, candidate_backend) in
        MAC_BENCHMARK_BUNDLE_COMPARISONS
    {
        let comparison_json = dir.join(json_name);
        let comparison_bytes = read_report_bounded(&comparison_json).map_err(|error| {
            anyhow::anyhow!(
                "failed to load comparison artifact `{}`: {error}. Fix: rerun vyre-bench compare --output from the benchmark gate.",
                comparison_json.display()
            )
        })?;
        let comparison = parse_comparison_artifact(&comparison_bytes)?;
        validate_comparison_expectations(
            &comparison,
            baseline_backend,
            candidate_backend,
            &[MAC_BENCHMARK_BUNDLE_CASE_ID.to_string()],
        )?;
        validate_comparison_matches_bundle_reports(&comparison, &reports)?;
        artifacts.push(bundle_artifact(
            dir,
            &comparison_json,
            "comparison_json",
            &comparison_bytes,
        )?);

        let comparison_text_path = dir.join(text_name);
        let comparison_text_bytes =
            read_report_bounded(&comparison_text_path).map_err(|error| {
                anyhow::anyhow!(
                    "failed to load comparison text artifact `{}`: {error}. Fix: rerun vyre-bench compare from the benchmark gate.",
                    comparison_text_path.display()
                )
            })?;
        if comparison_text_bytes.is_empty() {
            anyhow::bail!(
                "comparison text artifact `{}` is empty. Fix: rerun vyre-bench compare from the MacBook benchmark gate.",
                comparison_text_path.display()
            );
        }
        let comparison_text = std::str::from_utf8(&comparison_text_bytes)?;
        validate_comparison_text_evidence(
            comparison_text,
            &comparison,
            baseline_backend,
            candidate_backend,
        )?;
        artifacts.push(bundle_artifact(
            dir,
            &comparison_text_path,
            "comparison_text",
            &comparison_text_bytes,
        )?);
        comparisons.push(comparison);
    }

    let provenance = derive_benchmark_bundle_provenance(&reports, &comparisons)?;
    let manifest = build_benchmark_bundle_manifest(artifacts, provenance)?;
    if let Some(path) = manifest_input {
        let expected = load_benchmark_bundle_manifest(std::path::Path::new(path))?;
        validate_benchmark_bundle_manifest_matches(&expected, &manifest, path)?;
    }
    if let Some(path) = manifest_output {
        write_benchmark_bundle_manifest(&manifest, std::path::Path::new(path))?;
    }
    Ok(manifest)
}

fn bundle_artifact(
    dir: &std::path::Path,
    path: &std::path::Path,
    kind: &str,
    bytes: &[u8],
) -> anyhow::Result<BundleArtifact> {
    let relative = path.strip_prefix(dir).map_err(|error| {
        anyhow::anyhow!(
            "bundle artifact `{}` is not under bundle dir `{}`: {error}. Fix: validate artifacts from one benchmark output directory.",
            path.display(),
            dir.display()
        )
    })?;
    Ok(BundleArtifact {
        path: relative.to_string_lossy().replace('\\', "/"),
        kind: kind.to_string(),
        bytes: bytes.len() as u64,
        blake3: blake3::hash(bytes).to_hex().to_string(),
    })
}

fn build_benchmark_bundle_manifest(
    mut artifacts: Vec<BundleArtifact>,
    provenance: BenchmarkBundleProvenance,
) -> anyhow::Result<BenchmarkBundleManifest> {
    artifacts.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then_with(|| left.kind.cmp(&right.kind))
    });
    let material = BundleHashMaterial {
        schema: BENCHMARK_BUNDLE_SCHEMA,
        provenance: &provenance,
        artifacts: &artifacts,
    };
    let canonical = serde_json::to_vec(&material)?;
    Ok(BenchmarkBundleManifest {
        schema: BENCHMARK_BUNDLE_SCHEMA.to_string(),
        provenance,
        artifact_count: artifacts.len(),
        bundle_blake3: blake3::hash(&canonical).to_hex().to_string(),
        artifacts,
    })
}

fn derive_benchmark_bundle_provenance(
    reports: &[ReportSchema],
    comparisons: &[ComparisonArtifact],
) -> anyhow::Result<BenchmarkBundleProvenance> {
    if reports.is_empty() {
        anyhow::bail!(
            "benchmark bundle has no backend reports. Fix: rerun the benchmark gate so cpu-ref, wgpu, and metal reports are present."
        );
    }
    if comparisons.is_empty() {
        anyhow::bail!(
            "benchmark bundle has no comparison artifacts. Fix: rerun the benchmark gate so comparison JSON artifacts are present."
        );
    }
    let mut suites = reports
        .iter()
        .map(|report| report.suite.as_str())
        .collect::<Vec<_>>();
    suites.sort_unstable();
    suites.dedup();
    if suites.len() != 1 {
        anyhow::bail!(
            "benchmark bundle reports disagree on suite {:?}. Fix: regenerate the bundle from one vyre-bench run configuration.",
            suites
        );
    }
    let mut case_ids = comparisons
        .iter()
        .flat_map(|comparison| comparison.cases.iter())
        .map(|case| case.id.as_str())
        .collect::<Vec<_>>();
    case_ids.sort_unstable();
    case_ids.dedup();
    if case_ids.len() != 1 {
        anyhow::bail!(
            "benchmark bundle comparison must contain exactly one case for the Mac smoke bundle, got {:?}. Fix: rerun scripts/check_metal_macbook.sh benchmark.",
            case_ids
        );
    }
    let mut report_backends = reports
        .iter()
        .map(|report| {
            report
                .selected_backend
                .clone()
                .unwrap_or_else(|| "unknown".to_string())
        })
        .collect::<Vec<_>>();
    report_backends.sort();
    let mut source_fingerprints = reports
        .iter()
        .map(|report| report.source_fingerprint.clone())
        .collect::<Vec<_>>();
    for comparison in comparisons {
        source_fingerprints.push(comparison.baseline.source_fingerprint.clone());
        source_fingerprints.push(comparison.candidate.source_fingerprint.clone());
    }
    source_fingerprints.sort();
    source_fingerprints.dedup();
    if source_fingerprints.len() != 1 {
        anyhow::bail!(
            "benchmark bundle reports disagree on source_fingerprint {:?}. Fix: regenerate all benchmark artifacts from the same source checkout.",
            source_fingerprints
        );
    }
    let mut source_tree_fingerprints = reports
        .iter()
        .map(|report| report.source_tree_fingerprint.clone())
        .collect::<Vec<_>>();
    for comparison in comparisons {
        source_tree_fingerprints.push(comparison.baseline.source_tree_fingerprint.clone());
        source_tree_fingerprints.push(comparison.candidate.source_tree_fingerprint.clone());
    }
    source_tree_fingerprints.sort();
    source_tree_fingerprints.dedup();
    if source_tree_fingerprints.len() != 1 {
        anyhow::bail!(
            "benchmark bundle reports disagree on source_tree_fingerprint {:?}. Fix: regenerate all benchmark artifacts from the same source checkout.",
            source_tree_fingerprints
        );
    }
    let mut comparison_pairs = comparisons
        .iter()
        .map(|comparison| {
            format!(
                "{}->{}",
                comparison.baseline.profile_backend, comparison.candidate.profile_backend
            )
        })
        .collect::<Vec<_>>();
    comparison_pairs.sort();
    comparison_pairs.dedup();
    Ok(BenchmarkBundleProvenance {
        validator: "vyre-bench validate-benchmark-bundle".to_string(),
        validator_version: env!("CARGO_PKG_VERSION").to_string(),
        suite: suites[0].to_string(),
        case_id: case_ids[0].to_string(),
        report_backends,
        baseline_backend: comparisons[0].baseline.profile_backend.clone(),
        candidate_backend: comparisons[0].candidate.profile_backend.clone(),
        comparison_pairs,
        source_fingerprint: source_fingerprints[0].clone(),
        source_tree_fingerprint: source_tree_fingerprints[0].clone(),
    })
}

fn write_benchmark_bundle_manifest(
    manifest: &BenchmarkBundleManifest,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(path, format!("{json}\n"))?;
    Ok(())
}

fn load_benchmark_bundle_manifest(path: &std::path::Path) -> anyhow::Result<BenchmarkBundleManifest> {
    let bytes = read_report_bounded(path).map_err(|error| {
        anyhow::anyhow!(
            "failed to load benchmark bundle manifest `{}`: {error}. Fix: pass the bundle-manifest.json produced by validate-benchmark-bundle --manifest-output.",
            path.display()
        )
    })?;
    let manifest: BenchmarkBundleManifest = serde_json::from_slice(&bytes).map_err(|error| {
        anyhow::anyhow!(
            "benchmark bundle manifest `{}` is invalid JSON: {error}. Fix: regenerate it with validate-benchmark-bundle --manifest-output.",
            path.display()
        )
    })?;
    validate_benchmark_bundle_manifest_integrity(&manifest, &path.display().to_string())?;
    Ok(manifest)
}

fn validate_benchmark_bundle_manifest_integrity(
    manifest: &BenchmarkBundleManifest,
    label: &str,
) -> anyhow::Result<()> {
    if manifest.schema != BENCHMARK_BUNDLE_SCHEMA {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` schema `{}` is not `{BENCHMARK_BUNDLE_SCHEMA}`. Fix: regenerate the manifest with current vyre-bench.",
            manifest.schema
        );
    }
    validate_benchmark_bundle_provenance_shape(&manifest.provenance, label)?;
    if manifest.artifact_count != manifest.artifacts.len() {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` artifact_count={} contradicts artifacts.len()={}. Fix: regenerate the manifest from the benchmark bundle directory.",
            manifest.artifact_count,
            manifest.artifacts.len()
        );
    }
    for artifact in &manifest.artifacts {
        if artifact.path.is_empty()
            || artifact.path.starts_with('/')
            || artifact.path.contains("..")
            || artifact.path.contains('\\')
        {
            anyhow::bail!(
                "benchmark bundle manifest `{label}` has invalid relative artifact path `{}`. Fix: regenerate the manifest from one benchmark output directory.",
                artifact.path
            );
        }
        if artifact.kind.is_empty() {
            anyhow::bail!(
                "benchmark bundle manifest `{label}` artifact `{}` has an empty kind. Fix: regenerate the manifest with current vyre-bench.",
                artifact.path
            );
        }
        if !is_hex_64(&artifact.blake3) {
            anyhow::bail!(
                "benchmark bundle manifest `{label}` artifact `{}` has invalid blake3 `{}`. Fix: regenerate the manifest from the benchmark artifacts.",
                artifact.path,
                artifact.blake3
            );
        }
    }
    validate_benchmark_bundle_manifest_artifact_set(manifest, label)?;
    let normalized =
        build_benchmark_bundle_manifest(manifest.artifacts.clone(), manifest.provenance.clone())?;
    if normalized.artifact_count != manifest.artifact_count {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` normalized artifact_count={} contradicts recorded artifact_count={}. Fix: regenerate the manifest from the benchmark bundle directory.",
            normalized.artifact_count,
            manifest.artifact_count
        );
    }
    if normalized.bundle_blake3 != manifest.bundle_blake3 {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` bundle_blake3 `{}` does not match normalized artifact metadata hash `{}`. Fix: regenerate the manifest from the benchmark bundle directory.",
            manifest.bundle_blake3,
            normalized.bundle_blake3
        );
    }
    Ok(())
}

fn validate_benchmark_bundle_manifest_artifact_set(
    manifest: &BenchmarkBundleManifest,
    label: &str,
) -> anyhow::Result<()> {
    let mut observed = BTreeMap::<(String, String), usize>::new();
    for artifact in &manifest.artifacts {
        *observed
            .entry((artifact.path.clone(), artifact.kind.clone()))
            .or_default() += 1;
    }
    for ((path, kind), count) in &observed {
        if *count != 1 {
            anyhow::bail!(
                "benchmark bundle manifest `{label}` repeats artifact path `{path}` kind `{kind}` {count} times. Fix: regenerate the manifest from the benchmark bundle directory."
            );
        }
        if !BENCHMARK_BUNDLE_REQUIRED_ARTIFACTS
            .iter()
            .any(|(expected_path, expected_kind)| expected_path == path && expected_kind == kind)
        {
            anyhow::bail!(
                "benchmark bundle manifest `{label}` has unexpected artifact path `{path}` kind `{kind}`. Fix: regenerate the manifest with current vyre-bench."
            );
        }
    }
    for (path, kind) in BENCHMARK_BUNDLE_REQUIRED_ARTIFACTS {
        if !observed.contains_key(&((*path).to_string(), (*kind).to_string())) {
            anyhow::bail!(
                "benchmark bundle manifest `{label}` is missing required artifact path `{path}` kind `{kind}`. Fix: regenerate the manifest from the benchmark bundle directory."
            );
        }
    }
    Ok(())
}

fn validate_comparison_text_evidence(
    comparison_text: &str,
    comparison: &ComparisonArtifact,
    baseline_backend: &str,
    candidate_backend: &str,
) -> anyhow::Result<()> {
    for required in [
        format!("baseline_backend={baseline_backend}"),
        format!("candidate_backend={candidate_backend}"),
        format!("baseline_profile_backend={baseline_backend}"),
        format!("candidate_profile_backend={candidate_backend}"),
        "baseline_timing_quality=".to_string(),
        "candidate_timing_quality=".to_string(),
        "compare_exit_code=".to_string(),
        MAC_BENCHMARK_BUNDLE_CASE_ID.to_string(),
    ] {
        if !comparison_text.contains(&required) {
            anyhow::bail!(
                "comparison text artifact lacks `{required}`. Fix: regenerate the text comparison with current vyre-bench compare output."
            );
        }
    }
    validate_comparison_text_exit_code(comparison_text, comparison)
}

fn validate_comparison_text_exit_code(
    comparison_text: &str,
    comparison: &ComparisonArtifact,
) -> anyhow::Result<()> {
    let exit_code = comparison_text
        .lines()
        .find_map(|line| line.strip_prefix("compare_exit_code="))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "comparison text artifact lacks `compare_exit_code=`. Fix: regenerate the comparison text with scripts/check_metal_macbook.sh benchmark."
            )
        })?;
    let exit_code: i32 = exit_code.parse().map_err(|error| {
        anyhow::anyhow!(
            "comparison text artifact has invalid compare_exit_code `{exit_code}`: {error}. Fix: regenerate the comparison text with scripts/check_metal_macbook.sh benchmark."
        )
    })?;
    if exit_code < 0 {
        anyhow::bail!(
            "comparison text artifact has negative compare_exit_code={exit_code}. Fix: regenerate the comparison text with scripts/check_metal_macbook.sh benchmark."
        );
    }
    if comparison.regressed && exit_code == 0 {
        anyhow::bail!(
            "comparison text compare_exit_code=0 contradicts structured comparison regressed=true. Fix: rerun vyre-bench compare and capture its exit code."
        );
    }
    if !comparison.regressed && exit_code != 0 {
        anyhow::bail!(
            "comparison text compare_exit_code={exit_code} contradicts structured comparison regressed=false. Fix: rerun vyre-bench compare and capture its exit code."
        );
    }
    Ok(())
}

fn validate_comparison_matches_bundle_reports(
    comparison: &ComparisonArtifact,
    reports: &[ReportSchema],
) -> anyhow::Result<()> {
    let baseline_report = report_for_profile_backend(reports, &comparison.baseline.profile_backend)?;
    let candidate_report = report_for_profile_backend(reports, &comparison.candidate.profile_backend)?;
    let expected = build_comparison_artifact(baseline_report, candidate_report)?;
    if expected.schema != comparison.schema {
        anyhow::bail!(
            "comparison artifact schema `{}` does not match recomputed schema `{}`. Fix: rerun vyre-bench compare against the reports in the same benchmark bundle directory.",
            comparison.schema,
            expected.schema
        );
    }
    if expected.baseline != comparison.baseline {
        anyhow::bail!(
            "comparison artifact does not match bundled baseline `{}` report. Fix: rerun vyre-bench compare against the reports in the same benchmark bundle directory.",
            comparison.baseline.profile_backend
        );
    }
    if expected.candidate != comparison.candidate {
        anyhow::bail!(
            "comparison artifact does not match bundled candidate `{}` report. Fix: rerun vyre-bench compare against the reports in the same benchmark bundle directory.",
            comparison.candidate.profile_backend
        );
    }
    if expected.regressed != comparison.regressed {
        anyhow::bail!(
            "comparison artifact regressed={} does not match recomputed regressed={}. Fix: rerun vyre-bench compare against the reports in the same benchmark bundle directory.",
            comparison.regressed,
            expected.regressed
        );
    }
    if expected.cases.len() != comparison.cases.len() {
        anyhow::bail!(
            "comparison artifact case count {} does not match recomputed case count {}. Fix: rerun vyre-bench compare against the reports in the same benchmark bundle directory.",
            comparison.cases.len(),
            expected.cases.len()
        );
    }
    for (actual, expected) in comparison.cases.iter().zip(expected.cases.iter()) {
        validate_comparison_case_matches(actual, expected)?;
    }
    Ok(())
}

fn validate_comparison_case_matches(
    actual: &ComparisonCase,
    expected: &ComparisonCase,
) -> anyhow::Result<()> {
    if actual.id != expected.id
        || actual.baseline_p50_ns != expected.baseline_p50_ns
        || actual.candidate_p50_ns != expected.candidate_p50_ns
        || actual.verdict != expected.verdict
        || actual.regressed != expected.regressed
    {
        anyhow::bail!(
            "comparison artifact case `{}` does not match recomputed case evidence. Fix: rerun vyre-bench compare against the reports in the same benchmark bundle directory.",
            actual.id
        );
    }
    for (label, actual_value, expected_value) in [
        (
            "baseline_mean_ns",
            Some(actual.baseline_mean_ns),
            Some(expected.baseline_mean_ns),
        ),
        (
            "candidate_mean_ns",
            Some(actual.candidate_mean_ns),
            Some(expected.candidate_mean_ns),
        ),
        ("delta_fraction", actual.delta_fraction, expected.delta_fraction),
        ("delta_percent", actual.delta_percent, expected.delta_percent),
        ("p_value", actual.p_value, expected.p_value),
    ] {
        if !float_option_close(actual_value, expected_value) {
            anyhow::bail!(
                "comparison artifact case `{}` field `{label}` does not match recomputed floating evidence. Fix: rerun vyre-bench compare against the reports in the same benchmark bundle directory.",
                actual.id
            );
        }
    }
    Ok(())
}

fn float_option_close(actual: Option<f64>, expected: Option<f64>) -> bool {
    match (actual, expected) {
        (None, None) => true,
        (Some(actual), Some(expected)) => {
            if actual == expected {
                true
            } else {
                let scale = actual.abs().max(expected.abs()).max(1.0);
                (actual - expected).abs() <= scale * 1.0e-9
            }
        }
        _ => false,
    }
}

fn report_for_profile_backend<'a>(
    reports: &'a [ReportSchema],
    backend: &str,
) -> anyhow::Result<&'a ReportSchema> {
    reports
        .iter()
        .find(|report| {
            report
                .backend_profile
                .as_ref()
                .is_some_and(|profile| profile.backend == backend)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "comparison references backend `{backend}` but the benchmark bundle has no matching backend report. Fix: rerun the benchmark gate so the comparison and reports come from one bundle."
            )
        })
}

fn validate_benchmark_bundle_provenance_shape(
    provenance: &BenchmarkBundleProvenance,
    label: &str,
) -> anyhow::Result<()> {
    if provenance.validator != "vyre-bench validate-benchmark-bundle" {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` validator `{}` is not `vyre-bench validate-benchmark-bundle`. Fix: regenerate the manifest with current vyre-bench.",
            provenance.validator
        );
    }
    if provenance.validator_version != env!("CARGO_PKG_VERSION") {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` validator_version `{}` does not match current vyre-bench `{}`. Fix: regenerate the manifest with the same validator binary used for replay.",
            provenance.validator_version,
            env!("CARGO_PKG_VERSION")
        );
    }
    for (field, value) in [
        ("suite", provenance.suite.as_str()),
        ("case_id", provenance.case_id.as_str()),
        ("baseline_backend", provenance.baseline_backend.as_str()),
        ("candidate_backend", provenance.candidate_backend.as_str()),
        ("source_fingerprint", provenance.source_fingerprint.as_str()),
        (
            "source_tree_fingerprint",
            provenance.source_tree_fingerprint.as_str(),
        ),
    ] {
        if value.is_empty() {
            anyhow::bail!(
                "benchmark bundle manifest `{label}` provenance field `{field}` is empty. Fix: regenerate the manifest from validated benchmark reports."
            );
        }
    }
    if provenance.report_backends.is_empty() {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` provenance has no report_backends. Fix: regenerate the manifest from benchmark reports."
        );
    }
    if provenance.comparison_pairs.is_empty() {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` provenance has no comparison_pairs. Fix: regenerate the manifest from comparison artifacts."
        );
    }
    let mut sorted = provenance.report_backends.clone();
    sorted.sort();
    if sorted != provenance.report_backends {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` report_backends are not sorted. Fix: regenerate the manifest with current vyre-bench."
        );
    }
    if provenance
        .report_backends
        .iter()
        .any(|backend| backend.is_empty())
    {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` report_backends contains an empty backend id. Fix: regenerate the manifest from validated benchmark reports."
        );
    }
    let mut sorted_pairs = provenance.comparison_pairs.clone();
    sorted_pairs.sort();
    if sorted_pairs != provenance.comparison_pairs {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` comparison_pairs are not sorted. Fix: regenerate the manifest with current vyre-bench."
        );
    }
    if provenance
        .comparison_pairs
        .iter()
        .any(|pair| !pair.contains("->"))
    {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` comparison_pairs contains an invalid pair. Fix: regenerate the manifest from comparison artifacts."
        );
    }
    Ok(())
}

fn validate_benchmark_bundle_manifest_matches(
    expected: &BenchmarkBundleManifest,
    observed: &BenchmarkBundleManifest,
    label: &str,
) -> anyhow::Result<()> {
    validate_benchmark_bundle_manifest_integrity(observed, "fresh benchmark bundle")?;
    if expected.bundle_blake3 != observed.bundle_blake3 {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` bundle_blake3 `{}` does not match current artifacts `{}`. Fix: rerun the benchmark gate or investigate artifact drift.",
            expected.bundle_blake3,
            observed.bundle_blake3
        );
    }
    let expected_json = serde_json::to_value(expected)?;
    let observed_json = serde_json::to_value(observed)?;
    if expected_json != observed_json {
        anyhow::bail!(
            "benchmark bundle manifest `{label}` metadata does not match current artifacts. Fix: rerun validate-benchmark-bundle --manifest-output after checking for artifact drift."
        );
    }
    Ok(())
}

fn is_hex_64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_report_expectations(
    report: &ReportSchema,
    backend: Option<&str>,
    total_cases: Option<usize>,
    failed: Option<usize>,
) -> anyhow::Result<()> {
    report
        .validate_backend_profile_evidence(backend)
        .map_err(|error| anyhow::anyhow!("invalid benchmark report backend profile: {error}"))?;
    if let Some(expected_backend) = backend {
        if report.selected_backend.as_deref() != Some(expected_backend) {
            anyhow::bail!(
                "selected_backend {:?} does not match expected backend `{expected_backend}`. Fix: rerun the benchmark with --backend {expected_backend}.",
                report.selected_backend
            );
        }
    }
    if let Some(total_cases) = total_cases {
        if report.summary.total_cases != total_cases {
            anyhow::bail!(
                "summary.total_cases={} does not match expected total_cases={total_cases}. Fix: rerun the benchmark with the intended --case selection.",
                report.summary.total_cases
            );
        }
    }
    if let Some(failed) = failed {
        if report.summary.failed != failed {
            anyhow::bail!(
                "summary.failed={} does not match expected failed={failed}. Fix: inspect blockers and rerun after fixing failing benchmark cases.",
                report.summary.failed
            );
        }
    }
    Ok(())
}

fn compare_verdict(delta_fraction: Option<f64>, p_value: Option<f64>) -> &'static str {
    match (delta_fraction, p_value) {
        (Some(delta), Some(p)) if delta > 0.05 && p < 0.05 => "regress",
        (Some(delta), Some(p)) if delta < -0.05 && p < 0.05 => "improve",
        (Some(delta), _) if delta.abs() <= 0.05 => "flat",
        _ => "noisy",
    }
}

fn welch_p_value(
    baseline: &crate::api::metric::MetricStats,
    candidate: &crate::api::metric::MetricStats,
) -> Option<f64> {
    if baseline.samples < 2 || candidate.samples < 2 {
        return None;
    }
    let n1 = f64::from(baseline.samples);
    let n2 = f64::from(candidate.samples);
    let variance = baseline.stddev.powi(2) / n1 + candidate.stddev.powi(2) / n2;
    if variance <= f64::EPSILON {
        return (baseline.mean != candidate.mean)
            .then_some(0.0)
            .or(Some(1.0));
    }
    let t = (candidate.mean - baseline.mean).abs() / variance.sqrt();
    Some((2.0 * (1.0 - normal_cdf(t))).clamp(0.0, 1.0))
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn load_report(path: &str) -> anyhow::Result<ReportSchema> {
    let bytes = read_report_bounded(std::path::Path::new(path))?;
    parse_report(&bytes, path)
}

fn parse_report(bytes: &[u8], path: &str) -> anyhow::Result<ReportSchema> {
    let report: ReportSchema = serde_json::from_slice(&bytes)?;
    report
        .validate_summary_evidence()
        .map_err(|error| anyhow::anyhow!("invalid benchmark report `{}`: {error}", path))?;
    report
        .validate_blocker_evidence()
        .map_err(|error| anyhow::anyhow!("invalid benchmark report `{}`: {error}", path))?;
    Ok(report)
}

fn read_report_bounded(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read as _;

    let mut file = std::fs::File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > MAX_REPORT_INPUT_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("benchmark report exceeds {MAX_REPORT_INPUT_BYTES} byte limit"),
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.by_ref()
        .take(MAX_REPORT_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_REPORT_INPUT_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "benchmark report exceeded bounded read limit",
        ));
    }
    Ok(bytes)
}

fn generate_dashboard(output_dir: impl AsRef<str>) -> anyhow::Result<()> {
    let output = std::path::Path::new(output_dir.as_ref());
    std::fs::create_dir_all(output)?;
    std::fs::create_dir_all(output.join("data"))?;
    std::fs::create_dir_all(output.join("history"))?;

    // Find latest snapshot
    let snapshots_dir = std::path::Path::new("snapshots");
    let latest = find_latest_snapshot(snapshots_dir)?;
    let report: ReportSchema = load_report(&latest.to_string_lossy())?;

    // Copy raw data
    std::fs::copy(&latest, output.join("data/results.json"))?;

    // Generate scorecard markdown
    let scorecard_md = generate_scorecard_md(&report);
    std::fs::write(output.join("scorecard.md"), &scorecard_md)?;

    // Generate per-case SVG bar charts
    for case in &report.cases {
        let svg = generate_case_svg(case);
        let filename = case.id.replace('.', "_") + ".svg";
        std::fs::write(output.join(&filename), &svg)?;
    }

    // Generate cross-backend SVG
    let cross_svg = generate_cross_backend_svg(&report);
    std::fs::write(output.join("cross-backend.svg"), &cross_svg)?;

    // Generate index.html
    let html = generate_index_html(&report, &scorecard_md);
    std::fs::write(output.join("index.html"), &html)?;

    println!(
        "Dashboard generated: {} ({} cases, {} files)",
        output.display(),
        report.cases.len(),
        4 + report.cases.len() // index.html + scorecard.md + data/results.json + cross-backend.svg + per-case SVGs
    );
    Ok(())
}

fn find_latest_snapshot(dir: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    if !dir.exists() {
        anyhow::bail!("snapshots directory does not exist: {}", dir.display());
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| {
        e.metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    entries
        .last()
        .map(|e| e.path())
        .ok_or_else(|| anyhow::anyhow!("no snapshot files found in {}", dir.display()))
}

fn generate_scorecard_md(report: &ReportSchema) -> String {
    let mut md = String::new();
    md.push_str("# vyre-bench Scorecard\n\n");
    md.push_str(&format!(
        "Suite: **{}** | Cases: {}/{} passed\n\n",
        report.suite,
        report
            .cases
            .iter()
            .filter(|case| case.passes_summary_evidence())
            .count(),
        report.cases.len(),
    ));
    md.push_str("| Case | Status | p50 (ns) | p99 (ns) | Speedup | CV |\n");
    md.push_str("|------|--------|----------|----------|---------|----|\n");
    for case in &report.cases {
        let wall = case.metrics.get("wall_ns");
        let p50 = wall.map(|s| s.p50).unwrap_or(0);
        let p99 = wall.map(|s| s.p99).unwrap_or(0);
        let cv = wall
            .map(|s| {
                if s.mean > 0.0 {
                    format!("{:.3}", s.stddev / s.mean)
                } else {
                    " - ".into()
                }
            })
            .unwrap_or_else(|| " - ".into());
        let speedup = case
            .performance
            .as_ref()
            .and_then(|p| p.speedup_x)
            .map(|s| format!("{:.1}×", s))
            .unwrap_or_else(|| " - ".into());
        let status_emoji = if case.passes_summary_evidence() {
            "✅"
        } else {
            match case.status.as_str() {
                "failed" => "❌",
                "unstable" | "thermal_unstable" => "⚠️",
                _ => "❓",
            }
        };
        md.push_str(&format!(
            "| {} | {} {} | {:>10} | {:>10} | {:>7} | {} |\n",
            case.id, status_emoji, case.status, p50, p99, speedup, cv
        ));
    }
    md
}

fn generate_case_svg(case: &crate::report::json::CaseReport) -> String {
    let wall = case.metrics.get("wall_ns");
    let p50 = wall.map(|s| s.p50).unwrap_or(1) as f64;
    let p99 = wall.map(|s| s.p99).unwrap_or(1) as f64;
    let max = wall.map(|s| s.max).unwrap_or(1) as f64;
    let scale = 300.0 / max.max(1.0);

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80" viewBox="0 0 400 80">
  <style>
    text {{ font-family: 'Inter', sans-serif; font-size: 11px; fill: #e0e0e0; }}
    .title {{ font-size: 12px; font-weight: 600; }}
    .bar {{ rx: 3; }}
  </style>
  <rect width="400" height="80" fill="#1a1a2e" rx="6"/>
  <text x="10" y="16" class="title">{id}</text>
  <rect class="bar" x="10" y="28" width="{w50}" height="14" fill="#00d2ff"/>
  <text x="{tw50}" y="39" fill="#fff">p50: {p50_ns}ns</text>
  <rect class="bar" x="10" y="48" width="{w99}" height="14" fill="#7b2ff7"/>
  <text x="{tw99}" y="59" fill="#fff">p99: {p99_ns}ns</text>
  <rect class="bar" x="10" y="68" width="{wmax}" height="8" fill="#ff6b6b" opacity="0.5"/>
</svg>"##,
        id = case.id,
        w50 = (p50 * scale) as u32,
        w99 = (p99 * scale) as u32,
        wmax = (max * scale) as u32,
        tw50 = (p50 * scale) as u32 + 14,
        tw99 = (p99 * scale) as u32 + 14,
        p50_ns = p50 as u64,
        p99_ns = p99 as u64,
    )
}

fn generate_cross_backend_svg(report: &ReportSchema) -> String {
    let case_count = report.cases.len();
    let height = 40 + case_count * 30;
    let mut bars = String::new();

    for (i, case) in report.cases.iter().enumerate() {
        let wall = case.metrics.get("wall_ns");
        let p50 = wall.map(|s| s.p50).unwrap_or(0);
        let y = 30 + i * 30;
        let width = (p50 as f64 / 1_000_000.0).clamp(5.0, 350.0) as u32; // scale to ms

        bars.push_str(&format!(
            r##"  <rect x="10" y="{y}" width="{w}" height="20" fill="#00d2ff" rx="3"/>
  <text x="{tx}" y="{ty}" fill="#e0e0e0" font-size="10">{id} ({p50_us}μs)</text>
"##,
            y = y,
            w = width,
            tx = width + 14,
            ty = y + 14,
            id = case.id,
            p50_us = p50 / 1000,
        ));
    }

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="600" height="{h}" viewBox="0 0 600 {h}">
  <style>text {{ font-family: 'Inter', sans-serif; }}</style>
  <rect width="600" height="{h}" fill="#1a1a2e" rx="6"/>
  <text x="10" y="20" fill="#e0e0e0" font-size="14" font-weight="600">Cross-Backend: {suite}</text>
{bars}</svg>"##,
        h = height,
        suite = report.suite,
        bars = bars,
    )
}

fn generate_index_html(report: &ReportSchema, _scorecard_md: &str) -> String {
    let cases_count = report.cases.len();
    let passed = report
        .cases
        .iter()
        .filter(|case| case.passes_summary_evidence())
        .count();

    let mut rows = String::new();
    for case in &report.cases {
        let wall = case.metrics.get("wall_ns");
        let p50 = wall.map(|s| s.p50).unwrap_or(0);
        let p99 = wall.map(|s| s.p99).unwrap_or(0);
        let cv = wall
            .map(|s| {
                if s.mean > 0.0 {
                    format!("{:.3}", s.stddev / s.mean)
                } else {
                    " - ".into()
                }
            })
            .unwrap_or_else(|| " - ".into());
        let speedup = case
            .performance
            .as_ref()
            .and_then(|p| p.speedup_x)
            .map(|s| format!("{:.1}×", s))
            .unwrap_or_else(|| " - ".into());
        let status_class = if case.passes_summary_evidence() {
            "status-pass"
        } else {
            match case.status.as_str() {
                "failed" => "status-fail",
                _ => "status-warn",
            }
        };
        let svg_file = case.id.replace('.', "_") + ".svg";

        rows.push_str(&format!(
            r#"        <tr>
          <td><a href="{svg}">{id}</a></td>
          <td class="{cls}">{status}</td>
          <td class="num">{p50}</td>
          <td class="num">{p99}</td>
          <td class="num">{speedup}</td>
          <td class="num">{cv}</td>
        </tr>
"#,
            svg = svg_file,
            id = case.id,
            cls = status_class,
            status = case.status,
            p50 = p50,
            p99 = p99,
            speedup = speedup,
            cv = cv,
        ));
    }

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vyre-bench Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0f0f23;
      --surface: #1a1a2e;
      --accent: #00d2ff;
      --accent2: #7b2ff7;
      --text: #e0e0e0;
      --pass: #00e676;
      --fail: #ff5252;
      --warn: #ffab40;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem;
    }}
    h1 {{
      font-size: 2rem;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }}
    .summary {{
      display: flex;
      gap: 2rem;
      margin: 1rem 0 2rem;
    }}
    .stat {{
      background: var(--surface);
      border-radius: 12px;
      padding: 1.5rem 2rem;
      min-width: 140px;
      text-align: center;
    }}
    .stat-value {{
      font-size: 2.5rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .stat-label {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      opacity: 0.7;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border-radius: 12px;
      overflow: hidden;
    }}
    th {{
      text-align: left;
      padding: 0.8rem 1rem;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      background: rgba(0,0,0,0.2);
    }}
    td {{
      padding: 0.6rem 1rem;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    td a {{
      color: var(--accent);
      text-decoration: none;
    }}
    td a:hover {{ text-decoration: underline; }}
    .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .status-pass {{ color: var(--pass); font-weight: 600; }}
    .status-fail {{ color: var(--fail); font-weight: 600; }}
    .status-warn {{ color: var(--warn); font-weight: 600; }}
    .footer {{
      margin-top: 2rem;
      font-size: 0.8rem;
      opacity: 0.5;
    }}
    tr:hover {{ background: rgba(0,210,255,0.05); }}
  </style>
</head>
<body>
  <h1>vyre-bench Dashboard</h1>
  <p>Suite: <strong>{suite}</strong> &mdash; Generated {timestamp}</p>

  <div class="summary">
    <div class="stat">
      <div class="stat-value">{passed}</div>
      <div class="stat-label">Passed</div>
    </div>
    <div class="stat">
      <div class="stat-value">{total}</div>
      <div class="stat-label">Total Cases</div>
    </div>
    <div class="stat">
      <div class="stat-value">{pass_rate}%</div>
      <div class="stat-label">Pass Rate</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>Status</th>
        <th>p50 (ns)</th>
        <th>p99 (ns)</th>
        <th>Speedup</th>
        <th>CV</th>
      </tr>
    </thead>
    <tbody>
{rows}    </tbody>
  </table>

  <div class="footer">
    <p>Data: <a href="data/results.json">results.json</a> |
       Cross-backend: <a href="cross-backend.svg">cross-backend.svg</a> |
       Scorecard: <a href="scorecard.md">scorecard.md</a></p>
  </div>
</body>
</html>"##,
        suite = report.suite,
        timestamp = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            format!("{now} (unix)")
        },
        passed = passed,
        total = cases_count,
        pass_rate = if cases_count > 0 {
            passed * 100 / cases_count
        } else {
            0
        },
        rows = rows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::case::{Correctness, PerformanceEvaluation};
    use crate::api::metric::MetricStats;
    use crate::probes::environment::EnvironmentData;
    use crate::report::json::{CaseReport, ReportBackendProfile, ReportSummary};

    fn report(cases: Vec<CaseReport>, passed: usize, failed: usize) -> ReportSchema {
        ReportSchema {
            schema: "vyre-bench.result.v1".to_string(),
            run_id: "vyre-bench.release".to_string(),
            suite: "release".to_string(),
            selected_backend: Some("cuda".to_string()),
            backend_profile: None,
            git: BTreeMap::new(),
            source_fingerprint: "source:unit".to_string(),
            source_tree_fingerprint: "tree:unit".to_string(),
            environment: EnvironmentData {
                os: "linux".to_string(),
                architecture: "x86_64".to_string(),
                cpu_model: Some("unit".to_string()),
                cpu_cores: 1,
                has_gpu: true,
                gpu_devices: Vec::new(),
                nvidia_driver_version: Some("unit".to_string()),
                nvidia_cuda_version: Some("unit".to_string()),
                features: vec!["backend.usable.cuda".to_string()],
            },
            features: vec!["backend:cuda".to_string()],
            summary: ReportSummary {
                total_cases: cases.len(),
                passed,
                failed,
                total_time_ns: 1,
                cache_hit_rate: None,
            },
            cases,
            blockers: Vec::new(),
        }
    }

    fn backend_profile(backend: &str, timing_quality: &str) -> ReportBackendProfile {
        ReportBackendProfile {
            backend: backend.to_string(),
            timing_quality: timing_quality.to_string(),
            supports_device_timestamps: timing_quality == "device_timestamps",
            supports_hardware_counters: timing_quality == "hardware_counters",
            supports_subgroup_ops: false,
            supports_indirect_dispatch: false,
            max_workgroup_size: [1, 1, 1],
            max_invocations_per_workgroup: 1,
            max_shared_memory_bytes: 0,
            max_storage_buffer_binding_size: 0,
            subgroup_size: 0,
            compute_units: 0,
            mem_bw_gbps: 0,
        }
    }

    fn case_report(id: &str, status: &str, contract_passed: bool) -> CaseReport {
        CaseReport {
            id: id.to_string(),
            workload_fingerprint: format!("bench-case:{id}"),
            name: id.to_string(),
            owner_crate: "vyre-bench".to_string(),
            workload_class: "Release".to_string(),
            tags: Vec::new(),
            backend_id: Some("cuda".to_string()),
            needs_gpu: true,
            min_vram_bytes: None,
            min_input_bytes: None,
            required_features: Vec::new(),
            status: status.to_string(),
            wall_ns: Some(1.0),
            correctness: Correctness::Exact,
            contract: None,
            performance: Some(PerformanceEvaluation {
                speedup_x: Some(100.0),
                contract_passed,
                violations: if contract_passed {
                    Vec::new()
                } else {
                    vec!["speedup below release floor".to_string()]
                },
            }),
            metrics: BTreeMap::new(),
            optimization_passes_applied: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    fn wall_stats(p50: u64, mean: f64, stddev: f64, samples: u32) -> MetricStats {
        MetricStats {
            min: p50,
            p50,
            p90: p50,
            p95: p50,
            p99: p50,
            p999: p50,
            p9999: p50,
            max: p50,
            mean,
            stddev,
            samples,
            determinism_cv: None,
        }
    }

    fn case_report_with_wall(id: &str, p50: u64, mean: f64) -> CaseReport {
        let mut case = case_report(id, "pass", true);
        case.metrics
            .insert("wall_ns".to_string(), wall_stats(p50, mean, 1.0, 3));
        case
    }

    fn comparison_reports() -> (ReportSchema, ReportSchema) {
        let case_id = "foundation.elementwise.add.1m";
        let mut baseline = report(vec![case_report_with_wall(case_id, 100, 100.0)], 1, 0);
        baseline.suite = "smoke".to_string();
        baseline.selected_backend = Some("wgpu".to_string());
        baseline.backend_profile = Some(backend_profile("wgpu", "host_enqueue_wait"));
        let mut candidate = report(vec![case_report_with_wall(case_id, 90, 90.0)], 1, 0);
        candidate.suite = "smoke".to_string();
        candidate.selected_backend = Some("metal".to_string());
        candidate.backend_profile = Some(backend_profile("metal", "host_enqueue_wait"));
        (baseline, candidate)
    }

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vyre-bench-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock must be after UNIX_EPOCH")
                .as_nanos()
        ))
    }

    fn write_report(path: &std::path::Path, report: &ReportSchema) {
        std::fs::write(
            path,
            serde_json::to_vec(report).expect("test report should serialize"),
        )
        .expect("test report should be writable");
    }

    fn write_complete_benchmark_bundle(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).expect("test bundle dir should be creatable");
        let case_id = "foundation.elementwise.add.1m";
        let mut cpu_ref = report(vec![case_report_with_wall(case_id, 120, 120.0)], 1, 0);
        cpu_ref.suite = "smoke".to_string();
        cpu_ref.selected_backend = Some("cpu-ref".to_string());
        cpu_ref.backend_profile = Some(backend_profile("cpu-ref", "host_only"));
        write_report(&dir.join("cpu-ref.json"), &cpu_ref);

        let (wgpu, metal) = comparison_reports();
        write_report(&dir.join("wgpu.json"), &wgpu);
        write_report(&dir.join("metal.json"), &metal);

        let comparison = build_comparison_artifact(&wgpu, &metal)
            .expect("test comparison artifact should build");
        write_comparison_artifact(&comparison, &dir.join("wgpu-vs-metal.json").to_string_lossy())
            .expect("test comparison artifact should be writable");
        std::fs::write(
            dir.join("wgpu-vs-metal.txt"),
            "baseline_backend=wgpu\ncandidate_backend=metal\nbaseline_selected_backend=wgpu baseline_profile_backend=wgpu baseline_timing_quality=host_enqueue_wait\ncandidate_selected_backend=metal candidate_profile_backend=metal candidate_timing_quality=host_enqueue_wait\nfoundation.elementwise.add.1m\ncompare_exit_code=0\n",
        )
        .expect("test comparison text should be writable");
        let ref_comparison = build_comparison_artifact(&cpu_ref, &metal)
            .expect("test reference comparison artifact should build");
        write_comparison_artifact(
            &ref_comparison,
            &dir.join("cpu-ref-vs-metal.json").to_string_lossy(),
        )
        .expect("test reference comparison artifact should be writable");
        std::fs::write(
            dir.join("cpu-ref-vs-metal.txt"),
            "baseline_backend=cpu-ref\ncandidate_backend=metal\nbaseline_selected_backend=cpu-ref baseline_profile_backend=cpu-ref baseline_timing_quality=host_only\ncandidate_selected_backend=metal candidate_profile_backend=metal candidate_timing_quality=host_enqueue_wait\nfoundation.elementwise.add.1m\ncompare_exit_code=0\n",
        )
        .expect("test reference comparison text should be writable");
    }

    fn write_manifest_variant<F>(
        dir: &std::path::Path,
        label: &str,
        mutate: F,
    ) -> std::path::PathBuf
    where
        F: FnOnce(&mut BenchmarkBundleManifest),
    {
        let mut manifest = validate_benchmark_bundle(&dir.to_string_lossy(), None, None)
            .expect("Fix: complete benchmark bundle should produce a manifest model.");
        mutate(&mut manifest);
        let manifest = build_benchmark_bundle_manifest(
            manifest.artifacts.clone(),
            manifest.provenance.clone(),
        )
        .expect("Fix: mutated manifest should be hashable for schema-negative tests.");
        let path = dir.join(format!("{label}.json"));
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("Fix: manifest should serialize"),
        )
        .expect("Fix: manifest variant should be writable");
        path
    }

    fn expect_manifest_artifact_set_error<F>(label: &str, mutate: F, expected: &str)
    where
        F: FnOnce(&mut BenchmarkBundleManifest),
    {
        let dir = unique_temp_dir(label);
        write_complete_benchmark_bundle(&dir);
        let manifest_path = write_manifest_variant(&dir, "bundle-manifest-mutated", mutate);
        let error = validate_benchmark_bundle(
            &dir.to_string_lossy(),
            None,
            Some(&manifest_path.to_string_lossy()),
        )
        .expect_err("Fix: mutated manifest artifact set should be rejected.");
        let _ = std::fs::remove_dir_all(&dir);
        let error = error.to_string();
        assert!(
            error.contains(expected),
            "Fix: manifest artifact-set error should contain `{expected}`: {error}"
        );
    }

    #[test]
    fn compare_writes_structured_profile_artifact() {
        let (baseline, candidate) = comparison_reports();
        let path = std::env::temp_dir().join(format!(
            "vyre-bench-compare-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock must be after UNIX_EPOCH")
                .as_nanos()
        ));
        let path_arg = path.to_string_lossy().into_owned();

        compare_reports(&baseline, &candidate, Some(&path_arg))
            .expect("Fix: non-regressed comparison must write an artifact and succeed.");

        let artifact = load_comparison_artifact(&path_arg)
            .expect("Fix: comparison artifact must deserialize through the benchmark loader.");
        let _ = std::fs::remove_file(&path);
        validate_comparison_expectations(
            &artifact,
            "wgpu",
            "metal",
            &["foundation.elementwise.add.1m".to_string()],
        )
        .expect("Fix: comparison artifact must validate profile backend and case evidence.");
        assert_eq!(artifact.schema, "vyre-bench.compare.v1");
        assert_eq!(artifact.baseline.profile_backend, "wgpu");
        assert_eq!(artifact.candidate.profile_backend, "metal");
        assert_eq!(artifact.cases[0].baseline_p50_ns, 100);
        assert_eq!(artifact.cases[0].candidate_p50_ns, 90);
        assert!(!artifact.regressed);
    }

    #[test]
    fn validate_benchmark_bundle_accepts_complete_mac_gate_artifacts() {
        let dir = unique_temp_dir("bundle-ok");
        write_complete_benchmark_bundle(&dir);
        let manifest_path = dir.join("bundle-manifest.json");

        let manifest = validate_benchmark_bundle(
            &dir.to_string_lossy(),
            Some(&manifest_path.to_string_lossy()),
            None,
        )
            .expect("Fix: complete benchmark bundle should validate as one artifact set.");
        validate_benchmark_bundle(
            &dir.to_string_lossy(),
            None,
            Some(&manifest_path.to_string_lossy()),
        )
        .expect("Fix: freshly written bundle manifest should replay against current artifacts.");
        let manifest_bytes =
            std::fs::read(&manifest_path).expect("Fix: bundle manifest should be written.");
        let manifest_from_disk: BenchmarkBundleManifest = serde_json::from_slice(&manifest_bytes)
            .expect("Fix: bundle manifest should deserialize.");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            manifest.artifact_count, 7,
            "Fix: bundle validation should cover three backend reports and two comparison JSON/text pairs."
        );
        assert_eq!(manifest.schema, BENCHMARK_BUNDLE_SCHEMA);
        assert_eq!(manifest.provenance.validator, "vyre-bench validate-benchmark-bundle");
        assert_eq!(manifest.provenance.validator_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(manifest.provenance.suite, "smoke");
        assert_eq!(manifest.provenance.case_id, MAC_BENCHMARK_BUNDLE_CASE_ID);
        assert_eq!(
            manifest.provenance.report_backends,
            vec![
                "cpu-ref".to_string(),
                "metal".to_string(),
                "wgpu".to_string()
            ]
        );
        assert_eq!(
            manifest.provenance.baseline_backend,
            MAC_BENCHMARK_BUNDLE_BASELINE_BACKEND
        );
        assert_eq!(
            manifest.provenance.candidate_backend,
            MAC_BENCHMARK_BUNDLE_CANDIDATE_BACKEND
        );
        assert_eq!(
            manifest.provenance.comparison_pairs,
            vec!["cpu-ref->metal".to_string(), "wgpu->metal".to_string()]
        );
        assert_eq!(manifest.provenance.source_fingerprint, "source:unit");
        assert_eq!(manifest.provenance.source_tree_fingerprint, "tree:unit");
        assert_eq!(manifest.bundle_blake3.len(), 64);
        assert_eq!(manifest_from_disk.bundle_blake3, manifest.bundle_blake3);
        assert!(
            manifest
                .artifacts
                .iter()
                .any(|artifact| artifact.path == "metal.json"
                    && artifact.kind == "backend_report"
                    && artifact.blake3.len() == 64),
            "Fix: bundle manifest must content-address the Metal backend report."
        );
    }

    #[test]
    fn validate_benchmark_bundle_cli_writes_and_replays_manifest() {
        let dir = unique_temp_dir("bundle-cli-ok");
        write_complete_benchmark_bundle(&dir);
        let manifest_path = dir.join("bundle-manifest.json");
        let dir_arg = dir.to_string_lossy().into_owned();
        let manifest_arg = manifest_path.to_string_lossy().into_owned();

        run_cli_with(vec![
            "vyre-bench".to_string(),
            "validate-benchmark-bundle".to_string(),
            "--dir".to_string(),
            dir_arg.clone(),
            "--manifest-output".to_string(),
            manifest_arg.clone(),
        ])
        .expect("Fix: CLI should write a manifest for a complete benchmark bundle.");
        assert!(
            manifest_path.exists(),
            "Fix: CLI --manifest-output must create the bundle manifest."
        );
        run_cli_with(vec![
            "vyre-bench".to_string(),
            "validate-benchmark-bundle".to_string(),
            "--dir".to_string(),
            dir_arg,
            "--manifest-input".to_string(),
            manifest_arg,
        ])
        .expect("Fix: CLI should replay a freshly written benchmark bundle manifest.");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_benchmark_bundle_cli_rejects_compare_exit_code_drift() {
        let dir = unique_temp_dir("bundle-cli-exit-code-drift");
        write_complete_benchmark_bundle(&dir);
        std::fs::write(
            dir.join("wgpu-vs-metal.txt"),
            "baseline_backend=wgpu\ncandidate_backend=metal\nbaseline_selected_backend=wgpu baseline_profile_backend=wgpu baseline_timing_quality=host_enqueue_wait\ncandidate_selected_backend=metal candidate_profile_backend=metal candidate_timing_quality=host_enqueue_wait\nfoundation.elementwise.add.1m\ncompare_exit_code=9\n",
        )
        .expect("test comparison text should be writable");
        let error = run_cli_with(vec![
            "vyre-bench".to_string(),
            "validate-benchmark-bundle".to_string(),
            "--dir".to_string(),
            dir.to_string_lossy().into_owned(),
        ])
        .expect_err("Fix: CLI must reject contradictory compare exit-code evidence.");
        let _ = std::fs::remove_dir_all(&dir);
        let error = error.to_string();
        assert!(
            error.contains("compare_exit_code=9") && error.contains("regressed=false"),
            "Fix: CLI error should surface compare exit-code contradiction: {error}"
        );
    }

    #[test]
    fn validate_benchmark_bundle_rejects_manifest_provenance_drift() {
        let dir = unique_temp_dir("bundle-provenance-drift");
        write_complete_benchmark_bundle(&dir);
        let manifest_path = dir.join("bundle-manifest.json");
        validate_benchmark_bundle(
            &dir.to_string_lossy(),
            Some(&manifest_path.to_string_lossy()),
            None,
        )
        .expect("Fix: complete benchmark bundle should write a replay manifest.");
        let manifest_bytes =
            std::fs::read(&manifest_path).expect("Fix: bundle manifest should be readable.");
        let mut manifest: BenchmarkBundleManifest = serde_json::from_slice(&manifest_bytes)
            .expect("Fix: bundle manifest should deserialize.");
        manifest.provenance.case_id = "wrong.case".to_string();
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("Fix: mutated manifest should serialize"),
        )
        .expect("Fix: mutated manifest should be writable.");

        let error = validate_benchmark_bundle(
            &dir.to_string_lossy(),
            None,
            Some(&manifest_path.to_string_lossy()),
        )
        .expect_err("Fix: bundle validation must reject edited manifest provenance.");
        let _ = std::fs::remove_dir_all(&dir);
        let error = error.to_string();
        assert!(
            error.contains("bundle_blake3") && error.contains("normalized artifact metadata hash"),
            "Fix: manifest provenance drift should invalidate the normalized bundle hash: {error}"
        );
    }

    #[test]
    fn validate_benchmark_bundle_rejects_manifest_artifact_set_drift() {
        expect_manifest_artifact_set_error(
            "bundle-manifest-missing-artifact",
            |manifest| {
                manifest
                    .artifacts
                    .retain(|artifact| artifact.path != "metal.json");
            },
            "missing required artifact",
        );
        expect_manifest_artifact_set_error(
            "bundle-manifest-duplicate-artifact",
            |manifest| {
                let duplicate = manifest
                    .artifacts
                    .iter()
                    .find(|artifact| artifact.path == "metal.json")
                    .expect("Fix: base manifest should contain metal.json")
                    .clone();
                manifest.artifacts.push(duplicate);
            },
            "repeats artifact path",
        );
        expect_manifest_artifact_set_error(
            "bundle-manifest-unknown-artifact",
            |manifest| {
                let mut extra = manifest
                    .artifacts
                    .iter()
                    .find(|artifact| artifact.path == "metal.json")
                    .expect("Fix: base manifest should contain metal.json")
                    .clone();
                extra.path = "extra.json".to_string();
                manifest.artifacts.push(extra);
            },
            "unexpected artifact path",
        );
        expect_manifest_artifact_set_error(
            "bundle-manifest-mislabeled-artifact",
            |manifest| {
                let artifact = manifest
                    .artifacts
                    .iter_mut()
                    .find(|artifact| artifact.path == "metal.json")
                    .expect("Fix: base manifest should contain metal.json");
                artifact.kind = "comparison_json".to_string();
            },
            "unexpected artifact path",
        );
    }

    #[test]
    fn benchmark_bundle_provenance_is_derived_from_report_evidence() {
        let case_id = "custom.case";
        let mut cpu_ref = report(vec![case_report(case_id, "pass", true)], 1, 0);
        cpu_ref.suite = "custom-suite".to_string();
        cpu_ref.selected_backend = Some("cpu-ref".to_string());
        cpu_ref.backend_profile = Some(backend_profile("cpu-ref", "host_only"));
        let mut baseline = report(vec![case_report_with_wall(case_id, 100, 100.0)], 1, 0);
        baseline.suite = "custom-suite".to_string();
        baseline.selected_backend = Some("alpha".to_string());
        baseline.backend_profile = Some(backend_profile("alpha", "host_enqueue_wait"));
        let mut candidate = report(vec![case_report_with_wall(case_id, 90, 90.0)], 1, 0);
        candidate.suite = "custom-suite".to_string();
        candidate.selected_backend = Some("beta".to_string());
        candidate.backend_profile = Some(backend_profile("beta", "host_enqueue_wait"));
        let comparison = build_comparison_artifact(&baseline, &candidate)
            .expect("Fix: comparison artifact should build from matching custom cases.");

        let provenance = derive_benchmark_bundle_provenance(
            &[cpu_ref, baseline, candidate],
            &[comparison],
        )
        .expect("Fix: provenance should derive from valid report and comparison evidence.");

        assert_eq!(provenance.suite, "custom-suite");
        assert_eq!(provenance.case_id, case_id);
        assert_eq!(
            provenance.report_backends,
            vec![
                "alpha".to_string(),
                "beta".to_string(),
                "cpu-ref".to_string()
            ]
        );
        assert_eq!(provenance.baseline_backend, "alpha");
        assert_eq!(provenance.candidate_backend, "beta");
        assert_eq!(provenance.comparison_pairs, vec!["alpha->beta".to_string()]);
        assert_eq!(provenance.source_fingerprint, "source:unit");
        assert_eq!(provenance.source_tree_fingerprint, "tree:unit");
    }

    #[test]
    fn benchmark_bundle_provenance_rejects_mixed_source_reports() {
        let case_id = "custom.case";
        let mut cpu_ref = report(vec![case_report(case_id, "pass", true)], 1, 0);
        cpu_ref.suite = "custom-suite".to_string();
        cpu_ref.selected_backend = Some("cpu-ref".to_string());
        cpu_ref.backend_profile = Some(backend_profile("cpu-ref", "host_only"));
        let mut baseline = report(vec![case_report_with_wall(case_id, 100, 100.0)], 1, 0);
        baseline.suite = "custom-suite".to_string();
        baseline.selected_backend = Some("alpha".to_string());
        baseline.backend_profile = Some(backend_profile("alpha", "host_enqueue_wait"));
        let mut candidate = report(vec![case_report_with_wall(case_id, 90, 90.0)], 1, 0);
        candidate.suite = "custom-suite".to_string();
        candidate.source_tree_fingerprint = "tree:other".to_string();
        candidate.selected_backend = Some("beta".to_string());
        candidate.backend_profile = Some(backend_profile("beta", "host_enqueue_wait"));
        let comparison = build_comparison_artifact(&baseline, &candidate)
            .expect("Fix: comparison artifact should build from matching custom cases.");

        let error = derive_benchmark_bundle_provenance(
            &[cpu_ref, baseline, candidate],
            &[comparison],
        )
        .expect_err("Fix: bundle provenance must reject mixed source-tree evidence.");
        let error = error.to_string();
        assert!(
            error.contains("source_tree_fingerprint"),
            "Fix: mixed source-tree rejection must name the drifting field: {error}"
        );
    }

    #[test]
    fn validate_benchmark_bundle_rejects_comparison_report_drift() {
        let dir = unique_temp_dir("bundle-comparison-drift");
        write_complete_benchmark_bundle(&dir);
        let wgpu_path = dir.join("wgpu.json");
        let mut wgpu_report = load_report(&wgpu_path.to_string_lossy())
            .expect("Fix: synthetic WGPU report should load before mutation.");
        wgpu_report.run_id = "mutated-after-comparison".to_string();
        write_report(&wgpu_path, &wgpu_report);

        let error = validate_benchmark_bundle(&dir.to_string_lossy(), None, None)
            .expect_err("Fix: bundle validation must reject stale comparison JSON.");
        let _ = std::fs::remove_dir_all(&dir);
        let error = error.to_string();
        assert!(
            error.contains("comparison artifact does not match bundled"),
            "Fix: comparison/report drift must explain the stale comparison artifact: {error}"
        );
    }

    #[test]
    fn validate_benchmark_bundle_rejects_compare_exit_code_drift() {
        let dir = unique_temp_dir("bundle-exit-code-drift");
        write_complete_benchmark_bundle(&dir);
        std::fs::write(
            dir.join("wgpu-vs-metal.txt"),
            "baseline_backend=wgpu\ncandidate_backend=metal\nbaseline_selected_backend=wgpu baseline_profile_backend=wgpu baseline_timing_quality=host_enqueue_wait\ncandidate_selected_backend=metal candidate_profile_backend=metal candidate_timing_quality=host_enqueue_wait\nfoundation.elementwise.add.1m\ncompare_exit_code=7\n",
        )
        .expect("test comparison text should be writable");

        let error = validate_benchmark_bundle(&dir.to_string_lossy(), None, None)
            .expect_err("Fix: bundle validation must reject contradictory compare exit code.");
        let _ = std::fs::remove_dir_all(&dir);
        let error = error.to_string();
        assert!(
            error.contains("compare_exit_code=7") && error.contains("regressed=false"),
            "Fix: compare exit-code drift must explain the JSON/text contradiction: {error}"
        );
    }

    #[test]
    fn validate_benchmark_bundle_rejects_missing_comparison_json() {
        let dir = unique_temp_dir("bundle-missing-comparison");
        write_complete_benchmark_bundle(&dir);
        std::fs::remove_file(dir.join("wgpu-vs-metal.json"))
            .expect("test comparison JSON should be removable");

        let error = validate_benchmark_bundle(&dir.to_string_lossy(), None, None)
            .expect_err("Fix: bundle validation must reject missing comparison JSON.");
        let _ = std::fs::remove_dir_all(&dir);
        let error = error.to_string();
        assert!(
            error.contains("wgpu-vs-metal.json"),
            "Fix: missing comparison JSON should be named in the validation error: {error}"
        );
    }

    #[test]
    fn validate_benchmark_bundle_rejects_manifest_artifact_drift() {
        let dir = unique_temp_dir("bundle-manifest-drift");
        write_complete_benchmark_bundle(&dir);
        let manifest_path = dir.join("bundle-manifest.json");
        validate_benchmark_bundle(
            &dir.to_string_lossy(),
            Some(&manifest_path.to_string_lossy()),
            None,
        )
        .expect("Fix: complete benchmark bundle should write a replay manifest.");
        std::fs::write(
            dir.join("wgpu-vs-metal.txt"),
            "baseline_backend=wgpu\ncandidate_backend=metal\nbaseline_selected_backend=wgpu baseline_profile_backend=wgpu baseline_timing_quality=host_enqueue_wait\ncandidate_selected_backend=metal candidate_profile_backend=metal candidate_timing_quality=host_enqueue_wait\nfoundation.elementwise.add.1m\ncompare_exit_code=0\nmutated_after_manifest=1\n",
        )
        .expect("test comparison text should be writable");

        let error = validate_benchmark_bundle(
            &dir.to_string_lossy(),
            None,
            Some(&manifest_path.to_string_lossy()),
        )
        .expect_err("Fix: bundle validation must reject artifacts drifted after manifest creation.");
        let _ = std::fs::remove_dir_all(&dir);
        let error = error.to_string();
        assert!(
            error.contains("bundle_blake3") && error.contains("does not match current artifacts"),
            "Fix: manifest drift error should name the bundle hash mismatch: {error}"
        );
    }

    #[test]
    fn validate_comparison_rejects_candidate_backend_drift() {
        let (baseline, candidate) = comparison_reports();
        let artifact = build_comparison_artifact(&baseline, &candidate)
            .expect("Fix: comparison artifact should build from matching cases.");
        let error = validate_comparison_expectations(
            &artifact,
            "wgpu",
            "cuda",
            &["foundation.elementwise.add.1m".to_string()],
        )
        .expect_err("Fix: comparison validation must reject wrong candidate backend.");
        let error = error.to_string();
        assert!(
            error.contains("candidate profile backend"),
            "Fix: candidate backend drift must be explained: {error}"
        );
    }

    #[test]
    fn validate_report_command_accepts_backend_profile_contract() {
        let mut valid = report(
            vec![case_report("foundation.elementwise.add.1m", "pass", true)],
            1,
            0,
        );
        valid.backend_profile = Some(backend_profile("cuda", "device_timestamps"));
        let path = std::env::temp_dir().join(format!(
            "vyre-bench-valid-report-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock must be after UNIX_EPOCH")
                .as_nanos()
        ));
        std::fs::write(
            &path,
            serde_json::to_vec(&valid).expect("test report should serialize"),
        )
        .expect("test report should be writable");
        let path_arg = path.to_string_lossy().into_owned();

        let result = run_cli_with(vec![
            "vyre-bench".to_string(),
            "validate-report".to_string(),
            "--path".to_string(),
            path_arg,
            "--backend".to_string(),
            "cuda".to_string(),
            "--total-cases".to_string(),
            "1".to_string(),
            "--failed".to_string(),
            "0".to_string(),
        ]);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "Fix: validate-report must accept matching backend profile evidence: {result:?}"
        );
    }

    #[test]
    fn validate_report_expectations_rejects_missing_backend_profile() {
        let forged = report(
            vec![case_report("foundation.elementwise.add.1m", "pass", true)],
            1,
            0,
        );
        let error = validate_report_expectations(&forged, Some("cuda"), Some(1), Some(0))
            .expect_err("Fix: expected-backend validation must reject missing backend_profile");
        let error = error.to_string();
        assert!(
            error.contains("lacks backend_profile"),
            "Fix: missing backend_profile errors should explain the report must be regenerated: {error}"
        );
    }

    #[test]
    fn validate_report_expectations_rejects_profile_backend_drift() {
        let mut forged = report(
            vec![case_report("foundation.elementwise.add.1m", "pass", true)],
            1,
            0,
        );
        forged.backend_profile = Some(backend_profile("wgpu", "host_enqueue_wait"));
        let error = validate_report_expectations(&forged, Some("cuda"), Some(1), Some(0))
            .expect_err("Fix: expected-backend validation must reject mismatched backend_profile");
        let error = error.to_string();
        assert!(
            error.contains("contradicts expected backend"),
            "Fix: backend drift errors should name the profile mismatch: {error}"
        );
    }

    #[test]
    fn load_report_rejects_summary_that_hides_contract_failed_case() {
        let forged = report(
            vec![case_report("release.condition_eval.1m", "pass", false)],
            1,
            0,
        );
        let path = std::env::temp_dir().join(format!(
            "vyre-bench-forged-summary-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock must be after UNIX_EPOCH")
                .as_nanos()
        ));
        std::fs::write(
            &path,
            serde_json::to_vec(&forged).expect("test report should serialize"),
        )
        .expect("test report should be writable");

        let error = load_report(&path.to_string_lossy())
            .expect_err("Fix: loaded benchmark evidence must reject hidden contract failures");
        let _ = std::fs::remove_file(&path);
        let error = error.to_string();
        assert!(
            error.contains("invalid benchmark report") && error.contains("contradicts case evidence"),
            "Fix: report loader should explain that summary counts disagree with case evidence: {error}"
        );
    }

    #[test]
    fn load_report_rejects_blockers_that_hide_contract_failed_case() {
        let mut forged = report(
            vec![case_report("release.condition_eval.1m", "pass", false)],
            0,
            1,
        );
        forged.blockers.clear();
        let path = std::env::temp_dir().join(format!(
            "vyre-bench-forged-blockers-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock must be after UNIX_EPOCH")
                .as_nanos()
        ));
        std::fs::write(
            &path,
            serde_json::to_vec(&forged).expect("test report should serialize"),
        )
        .expect("test report should be writable");

        let error = load_report(&path.to_string_lossy())
            .expect_err("Fix: loaded benchmark evidence must reject hidden blockers");
        let _ = std::fs::remove_file(&path);
        let error = error.to_string();
        assert!(
            error.contains("invalid benchmark report")
                && error.contains("top-level blockers")
                && error.contains("contradict case-derived blockers"),
            "Fix: report loader should explain that top-level blockers disagree with case evidence: {error}"
        );
    }

    #[test]
    fn dashboard_counts_pass_status_evidence_not_legacy_passed_string() {
        let report = report(
            vec![
                case_report("release.condition_eval.1m", "pass", true),
                case_report("release.scan_ac_irregular.1m", "failed", true),
            ],
            1,
            1,
        );

        let scorecard = generate_scorecard_md(&report);
        assert!(
            scorecard.contains("Cases: 1/2 passed"),
            "Fix: dashboard scorecard must count generated `pass` status as pass evidence: {scorecard}"
        );

        let html = generate_index_html(&report, &scorecard);
        assert!(
            html.contains("<div class=\"stat-value\">1</div>")
                && html.contains("<td class=\"status-pass\">pass</td>"),
            "Fix: dashboard HTML must render generated `pass` status with pass styling: {html}"
        );
    }
}
