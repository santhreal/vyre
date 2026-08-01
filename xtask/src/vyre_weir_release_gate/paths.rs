use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use super::types::{GateMode, MAX_RELEASE_GATE_TEXT_BYTES};

pub(super) struct GateOptions {
    pub(super) manifest_path: PathBuf,
    pub(super) mode: GateMode,
}

pub(super) fn options_from_args(args: &[String]) -> Result<GateOptions, String> {
    let mut manifest_path = None;
    let mut mode = GateMode::Final;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--manifest" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --manifest requires a path.".to_string());
                };
                manifest_path = Some(PathBuf::from(path));
                index += 2;
            }
            "--prepublish" => {
                mode = GateMode::Prepublish;
                index += 1;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo xtask vyre-release-gate [--prepublish] [--manifest PATH]\n\n\
                     Checks the Vyre release evidence manifest. Final mode requires \
                     completed publication, repository verification, and pushes. \
                     --prepublish accepts only those explicit approval-gated actions \
                     as pending and rejects every internal blocker."
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!("Fix: unknown vyre-release-gate option `{other}`."));
            }
        }
    }

    Ok(GateOptions {
        manifest_path: manifest_path.unwrap_or_else(default_manifest_path),
        mode,
    })
}
pub(super) fn default_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/vyre-release-evidence.toml"))
        .unwrap_or_else(|| PathBuf::from("release/vyre-release-evidence.toml"))
}
pub(super) fn resolve_manifest_path(base_dir: &Path, path: &str) -> PathBuf {
    crate::output_arg::resolve_path(base_dir, path)
}
pub(super) fn resolve_artifact_path(base_dir: &Path, path: &str) -> PathBuf {
    crate::output_arg::resolve_release_artifact_path(base_dir, path)
}
pub(super) fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_RELEASE_GATE_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_RELEASE_GATE_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_RELEASE_GATE_TEXT_BYTES} byte release gate read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prepublication mode must remain an explicit opt-in while preserving a
    /// caller-supplied evidence manifest.
    #[test]
    fn parses_prepublish_mode_with_manifest() {
        let args = vec![
            "xtask".to_string(),
            "vyre-release-gate".to_string(),
            "--prepublish".to_string(),
            "--manifest".to_string(),
            "release/custom.toml".to_string(),
        ];

        let options = options_from_args(&args).expect("valid prepublish arguments");

        assert_eq!(options.mode, GateMode::Prepublish);
        assert_eq!(options.manifest_path, PathBuf::from("release/custom.toml"));
    }

    /// Omitting the prepublication flag must retain the final-launch gate so
    /// an ordinary invocation cannot silently weaken release closure.
    #[test]
    fn defaults_to_final_launch_mode() {
        let args = vec!["xtask".to_string(), "vyre-release-gate".to_string()];

        let options = options_from_args(&args).expect("valid final-gate arguments");

        assert_eq!(options.mode, GateMode::Final);
        assert_eq!(options.manifest_path, default_manifest_path());
    }
}
