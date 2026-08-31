/// What the caller asked for: a measurement run, or the option list.
pub(super) enum Parsed {
    Run(Config),
    Usage,
}

/// The option list, one line per note.
///
/// A gate never prints. `--help` used to write these lines on stdout and exit,
/// and stdout is where a delegated gate's parent reads one JSON report: the
/// caller got usage text followed by a protocol error, and a gate that exits has
/// judged nothing. They travel back as report notes, which the runner renders.
pub(super) const USAGE: &[&str] = &[
    "usage: release-benchmarks [--write] [--backend cuda|wgpu] [--only FAMILY] \
     [--measured-samples N] [--sample-timeout-secs N] [--include-wgpu-comparison] \
     [--reuse-existing] [--refresh-suites-only] [--workload-suite-only]",
    "--write re-measures the suite on a release host; without it the recorded artifacts are \
     audited and no benchmark runs.",
    "--backend selects the measured backend and defaults to cuda.",
    "--only names one release workload family from the release workload matrix.",
    "--measured-samples sets the per-case sample count and must be 30 or more for release \
     evidence.",
    "--sample-timeout-secs bounds one sample.",
    "--include-wgpu-comparison measures the wgpu comparison suite as well, which the cuda \
     release path does not need.",
    "--reuse-existing keeps artifacts that already validate and re-measures only the missing or \
     invalid cases.",
    "--refresh-suites-only rewrites the suite and proof summaries from recorded artifact JSON \
     without measuring.",
    "--workload-suite-only writes workload artifacts and suite summaries without the auxiliary \
     optimization artifacts.",
];

pub(super) struct Config {
    pub(super) backend: String,
    pub(super) only: Option<String>,
    pub(super) measured_samples: Option<usize>,
    pub(super) sample_timeout_secs: u64,
    pub(super) include_wgpu_comparison: bool,
    pub(super) reuse_existing: bool,
    pub(super) refresh_suites_only: bool,
    pub(super) workload_suite_only: bool,
}

/// Parse the caller flags a `GateCtx` carries.
///
/// `ctx.args` holds the flags after the subcommand name, so parsing starts at the
/// first element. This used to start at index 2, which skipped two flags: every
/// invocation that passed one option reported the third token as unknown, so
/// `--write --backend cuda` rejected `cuda` and no option was reachable at all. A
/// bare `--write` still measured, which is why the break stayed hidden.
pub(super) fn parse_args(args: &[String]) -> Result<Parsed, String> {
    let mut backend = "cuda".to_string();
    let mut only = None;
    let mut measured_samples = Some(30usize);
    let mut sample_timeout_secs = 120u64;
    let mut include_wgpu_comparison = false;
    let mut reuse_existing = false;
    let mut refresh_suites_only = false;
    let mut workload_suite_only = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--backend" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --backend requires a backend id.".to_string());
                };
                if value != "cuda" && value != "wgpu" {
                    return Err(
                        "Fix: release-benchmarks only accepts `cuda` or `wgpu` backends."
                            .to_string(),
                    );
                }
                backend = value.clone();
                index += 2;
            }
            "--only" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --only requires a release workload family id.".to_string());
                };
                only = Some(value.clone());
                index += 2;
            }
            "--measured-samples" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --measured-samples requires a positive integer.".to_string());
                };
                let parsed = value.parse::<usize>().map_err(|error| {
                    format!("Fix: --measured-samples must be a positive integer: {error}")
                })?;
                if parsed == 0 {
                    return Err("Fix: --measured-samples must be greater than zero.".to_string());
                }
                if parsed < 30 {
                    return Err(
                        "Fix: release-benchmarks requires --measured-samples >= 30 for release evidence."
                            .to_string(),
                    );
                }
                measured_samples = Some(parsed);
                index += 2;
            }
            "--sample-timeout-secs" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --sample-timeout-secs requires seconds.".to_string());
                };
                sample_timeout_secs = value.parse::<u64>().map_err(|error| {
                    format!("Fix: --sample-timeout-secs must be a positive integer: {error}")
                })?;
                if sample_timeout_secs == 0 {
                    return Err("Fix: --sample-timeout-secs must be greater than zero.".to_string());
                }
                index += 2;
            }
            "--include-wgpu-comparison" => {
                include_wgpu_comparison = true;
                index += 1;
            }
            "--reuse-existing" => {
                reuse_existing = true;
                index += 1;
            }
            "--refresh-suites-only" => {
                refresh_suites_only = true;
                index += 1;
            }
            "--workload-suite-only" => {
                workload_suite_only = true;
                index += 1;
            }
            "--write" => {
                index += 1;
            }
            "--help" | "-h" => return Ok(Parsed::Usage),
            other => return Err(format!("Fix: unknown release-benchmarks option `{other}`.")),
        }
    }
    Ok(Parsed::Run(Config {
        backend,
        only,
        measured_samples,
        sample_timeout_secs,
        include_wgpu_comparison,
        reuse_existing,
        refresh_suites_only,
        workload_suite_only,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flags a `GateCtx` carries: everything after the subcommand name, which
    /// includes `--write` when the caller asked for a re-measure.
    fn args(extra: &[&str]) -> Vec<String> {
        extra.iter().map(|arg| (*arg).to_string()).collect()
    }

    /// The configuration a flag list parses to, or a panic naming the flags.
    fn config(extra: &[&str]) -> Config {
        match parse_args(&args(extra)) {
            Ok(Parsed::Run(config)) => config,
            Ok(Parsed::Usage) => panic!("Fix: {extra:?} must parse as a run, not as usage."),
            Err(error) => panic!("Fix: {extra:?} must parse: {error}"),
        }
    }

    #[test]
    fn refresh_suites_only_parses_without_forcing_benchmark_reuse() {
        let config = config(&[
            "--backend",
            "cuda",
            "--include-wgpu-comparison",
            "--refresh-suites-only",
        ]);

        assert_eq!(config.backend, "cuda");
        assert!(config.include_wgpu_comparison);
        assert!(config.refresh_suites_only);
        assert!(
            !config.reuse_existing,
            "Fix: suite summary refresh must be distinct from freshness-based benchmark reuse."
        );
    }

    #[test]
    fn workload_suite_only_parses_as_auxiliary_skip() {
        let config = config(&["--backend", "wgpu", "--workload-suite-only"]);

        assert_eq!(config.backend, "wgpu");
        assert!(config.workload_suite_only);
        assert!(
            !config.refresh_suites_only,
            "Fix: workload-suite execution must still run benchmark artifacts unless refresh-only is also explicit."
        );
    }

    /// Every option must be reachable from the first flag onward.
    ///
    /// The parser used to start at index 2 while `GateCtx.args` starts at the
    /// first flag, so the two leading flags were skipped and their values were
    /// read as commands. `--write --backend cuda` reported `cuda` as an unknown
    /// option; a lone `--write` fell off the end and silently took the default
    /// backend, which is why the gate looked usable.
    #[test]
    fn every_flag_is_reachable_from_the_first_argument() {
        let written = config(&["--write", "--backend", "wgpu"]);
        assert_eq!(written.backend, "wgpu");

        let defaulted = config(&["--write"]);
        assert_eq!(defaulted.backend, "cuda");

        let selected = config(&["--only", "condition-eval", "--measured-samples", "30"]);
        assert_eq!(selected.only.as_deref(), Some("condition-eval"));
        assert_eq!(selected.measured_samples, Some(30));

        let Err(error) = parse_args(&args(&["--nonsense"])) else {
            panic!("Fix: an unknown first flag must still be rejected.");
        };
        assert!(
            error.contains("--nonsense"),
            "Fix: the error must name the option the caller typed, got `{error}`."
        );
    }

    /// `--help` is an answer, not an exit. A gate that printed usage on stdout
    /// broke the report protocol its parent reads, so the option list comes back
    /// as an outcome the caller renders.
    #[test]
    fn help_parses_as_usage_rather_than_printing() {
        for flag in ["--help", "-h"] {
            assert!(
                matches!(parse_args(&args(&[flag])), Ok(Parsed::Usage)),
                "Fix: `{flag}` must return the usage outcome."
            );
        }
        assert!(
            USAGE.iter().any(|line| line.contains("--measured-samples")),
            "Fix: the usage lines must name every option the parser accepts."
        );
    }
}
