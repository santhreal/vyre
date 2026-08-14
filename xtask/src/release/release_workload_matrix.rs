//! Generate the cheap release workload matrix without running benchmarks.

use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn run(args: &[String]) {
    let (output, enforce) = crate::output_arg::parsed_or_exit(parse_args(args));
    let workspace_root = crate::checkout::checkout_root();
    let runner = cargo_runner(&workspace_root);
    let mut command_args = vec![
        "run".to_string(),
        "-p".to_string(),
        "vyre-bench".to_string(),
        "--quiet".to_string(),
        "--".to_string(),
        "release-matrix".to_string(),
        "--format".to_string(),
        "json".to_string(),
        "--output".to_string(),
        output.display().to_string(),
    ];
    if enforce {
        command_args.push("--enforce".to_string());
    }
    let status = Command::new(&runner)
        .args(&command_args)
        .current_dir(&workspace_root)
        .status();
    match status {
        Ok(status) if status.success() => {
            println!("release-workload-matrix: wrote {}", output.display());
        }
        Ok(status) => {
            eprintln!(
                "Fix: `{}` exited with {status}. Workload matrix blockers must be resolved before release.",
                display_command(&runner, &command_args)
            );
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!(
                "Fix: failed to run `{}`: {error}. Set VYRE_CARGO_RUNNER to the bounded workspace cargo wrapper if it is not named `cargo_full`.",
                display_command(&runner, &command_args)
            );
            std::process::exit(1);
        }
    }
}

fn parse_args(args: &[String]) -> Result<(PathBuf, bool), String> {
    crate::output_arg::parse_output_and_flag_arg(
        args,
        "release-workload-matrix",
        "--enforce",
        "USAGE:\n  cargo xtask release-workload-matrix [--output PATH] [--enforce]\n\n\
         Writes the release workload family matrix without running benchmark cases.",
        || PathBuf::from("release/evidence/benchmarks/release-workload-matrix.json"),
    )
}

fn cargo_runner(workspace_root: &Path) -> PathBuf {
    crate::output_arg::cargo_runner(workspace_root)
}

fn display_command(runner: &Path, args: &[String]) -> String {
    format!("{} {}", runner.display(), args.join(" "))
}
