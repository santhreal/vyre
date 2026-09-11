//! The selection `whats-similar` accepts on the command line.

use std::path::{Path, PathBuf};

use xtask::artifact_paths::REGISTERED_OP_DUPLICATES_ARTIFACT;
use xtask::gates::dedup_report::duplicate_report_json_path;

pub(super) const DEFAULT_TOP_N: usize = 5;
const DEFAULT_MIN_SCORE: f64 = 0.20;
pub(super) const DEFAULT_ALL_MIN_SCORE: f64 = 0.80;

#[derive(Debug)]
pub(super) struct Cli {
    pub(super) mode: Mode,
    pub(super) top_n: usize,
    pub(super) min_score: f64,
    pub(super) duplicate_report_json: PathBuf,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) enum Mode {
    Target(String),
    All,
}

pub(super) fn parse_args(args: &[String]) -> Result<Cli, String> {
    let mut op_id: Option<String> = None;
    let mut all = false;
    let mut top_n = DEFAULT_TOP_N;
    let mut min_score = None;
    let mut duplicate_report_json = PathBuf::from(REGISTERED_OP_DUPLICATES_ARTIFACT);
    let mut iter = args.iter().skip(2);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--write" => {}
            "--all" => {
                all = true;
            }
            "--op-id" => {
                op_id = Some(
                    iter.next()
                        .cloned()
                        .ok_or_else(|| "--op-id needs a value".to_string())?,
                );
            }
            "--top" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--top needs a value".to_string())?;
                top_n = v
                    .parse::<usize>()
                    .map_err(|e| format!("--top must be a positive integer ({e})"))?;
                if top_n == 0 {
                    return Err("--top must be > 0".to_string());
                }
            }
            "--min" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--min needs a value".to_string())?;
                let parsed_min_score = v
                    .parse::<f64>()
                    .map_err(|e| format!("--min must be a float in [0,1] ({e})"))?;
                if !(0.0..=1.0).contains(&parsed_min_score) {
                    return Err("--min must be in [0,1]".to_string());
                }
                min_score = Some(parsed_min_score);
            }
            "--duplicate-report-json" => {
                let path = duplicate_report_json_path(
                    "--duplicate-report-json",
                    iter.next().map(String::as_str),
                    "--duplicate-report-json needs a value",
                )?;
                if path != Path::new(REGISTERED_OP_DUPLICATES_ARTIFACT) {
                    return Err(format!(
                        "--duplicate-report-json is fixed at `{REGISTERED_OP_DUPLICATES_ARTIFACT}`"
                    ));
                }
                duplicate_report_json = path;
            }
            "--file" => {
                return Err(
                    "Fix: whats-similar compares canonical SemanticOperation programs; submit the candidate and pass its id with --op-id <id>"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown arg `{other}`")),
        }
    }
    if all && op_id.is_some() {
        return Err("--all and --op-id are mutually exclusive".to_string());
    }
    let mode = if all {
        Mode::All
    } else {
        Mode::Target(op_id.ok_or_else(|| "--op-id is required unless --all is set".to_string())?)
    };
    let min_score = min_score.unwrap_or(match &mode {
        Mode::All => DEFAULT_ALL_MIN_SCORE,
        Mode::Target(_) => DEFAULT_MIN_SCORE,
    });
    Ok(Cli {
        mode,
        top_n,
        min_score,
        duplicate_report_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_requires_op_id() {
        let args = vec!["xtask".to_string(), "whats-similar".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn parse_with_op_id_and_defaults() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--op-id".to_string(),
            "vyre-libs::math::matmul".to_string(),
        ];
        let cli = parse_args(&args).unwrap();
        assert_eq!(
            cli.mode,
            Mode::Target("vyre-libs::math::matmul".to_string())
        );
        assert_eq!(cli.top_n, DEFAULT_TOP_N);
        assert!((cli.min_score - DEFAULT_MIN_SCORE).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_top_and_min_overrides() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--op-id".to_string(),
            "x".to_string(),
            "--top".to_string(),
            "10".to_string(),
            "--min".to_string(),
            "0.05".to_string(),
        ];
        let cli = parse_args(&args).unwrap();
        assert_eq!(cli.mode, Mode::Target("x".to_string()));
        assert_eq!(cli.top_n, 10);
        assert!((cli.min_score - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_duplicate_report_json_path() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--all".to_string(),
            "--duplicate-report-json".to_string(),
            "release/evidence/dedup/registered-op-duplicates.json".to_string(),
        ];
        let cli = parse_args(&args).unwrap();
        assert_eq!(cli.mode, Mode::All);
        assert_eq!(
            cli.duplicate_report_json,
            PathBuf::from("release/evidence/dedup/registered-op-duplicates.json")
        );
    }

    #[test]
    fn parse_accepts_write_authority() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--all".to_string(),
            "--write".to_string(),
        ];
        let cli = parse_args(&args).unwrap();
        assert_eq!(cli.mode, Mode::All);
    }

    #[test]
    fn parse_rejects_caller_selected_report_path() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--all".to_string(),
            "--duplicate-report-json".to_string(),
            "other.json".to_string(),
        ];
        let error = parse_args(&args).unwrap_err();
        assert!(error.contains(REGISTERED_OP_DUPLICATES_ARTIFACT));
    }
    #[test]
    fn parse_all_sets_duplicate_floor() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--all".to_string(),
        ];
        let cli = parse_args(&args).unwrap();
        assert_eq!(cli.mode, Mode::All);
        assert_eq!(cli.top_n, DEFAULT_TOP_N);
        assert!((cli.min_score - DEFAULT_ALL_MIN_SCORE).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_rejects_all_with_op_id() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--all".to_string(),
            "--op-id".to_string(),
            "x".to_string(),
        ];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn parse_rejects_top_zero() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--op-id".to_string(),
            "x".to_string(),
            "--top".to_string(),
            "0".to_string(),
        ];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn parse_rejects_min_out_of_range() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--op-id".to_string(),
            "x".to_string(),
            "--min".to_string(),
            "1.5".to_string(),
        ];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn parse_rejects_unknown_arg() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--op-id".to_string(),
            "x".to_string(),
            "--bogus".to_string(),
        ];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn parse_file_arg_returns_helpful_error() {
        let args = vec![
            "xtask".to_string(),
            "whats-similar".to_string(),
            "--file".to_string(),
            "x.rs".to_string(),
        ];
        let err = parse_args(&args).unwrap_err();
        assert!(err.contains("submit the candidate"));
    }
}
