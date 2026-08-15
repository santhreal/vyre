use std::io;
use std::path::{Path, PathBuf};

use super::gate_inputs::{GateMode, MAX_RELEASE_GATE_TEXT_BYTES};

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
    xtask::checkout::checkout_root().join("release/vyre-release-evidence.toml")
}
pub(super) fn resolve_manifest_path(base_dir: &Path, path: &str) -> PathBuf {
    xtask::output_arg::resolve_path(base_dir, path)
}
pub(super) fn resolve_artifact_path(base_dir: &Path, path: &str) -> PathBuf {
    xtask::output_arg::resolve_release_artifact_path(base_dir, path)
}
pub(super) fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_RELEASE_GATE_TEXT_BYTES, "release gate")
}

/// Whether a manifest evidence entry resolves outside the repository.
///
/// Evidence paths are written relative to the manifest's directory, which is
/// one level below the repository root, so a relative entry may pop exactly
/// one segment. Anything absolute, or with more `..` than it has earned, names
/// a file no clone of this repository is guaranteed to have.
pub(super) fn escapes_repository(evidence: &str) -> bool {
    xtask::output_arg::escapes_root(Path::new(evidence), 1)
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

    /// Evidence must name a file the repository actually carries. Entries that
    /// climb past the repository root resolve into whatever tree the checkout
    /// happens to sit in, so the gate passes or fails on the contents of a
    /// directory that is not under version control here and that no clone,
    /// worktree, or CI runner will reproduce.
    ///
    /// The class it closes: two requirements cited
    /// `../../../../../.github/CI_REQUIRED.md` and four siblings, all five of
    /// which exist inside this repository. Three of them also happened to
    /// exist five levels up on the machine the manifest was written on, which
    /// is why only the fourth ever reported missing. It does not catch an
    /// in-repository path that names the wrong file.
    #[test]
    fn evidence_paths_that_climb_past_the_repository_root_are_rejected() {
        assert!(escapes_repository("../../../../../.github/CI_REQUIRED.md"));
        assert!(escapes_repository("../.././.github/CI_REQUIRED.md"));
        assert!(escapes_repository("/etc/passwd"));

        assert!(!escapes_repository("../.github/CI_REQUIRED.md"));
        assert!(!escapes_repository("evidence/hygiene/hygiene-matrix.json"));
        assert!(!escapes_repository("../scripts/../scripts/apply.sh"));
    }

    /// The shipped manifest is the artifact the rule exists for, so assert it
    /// directly rather than trusting that a future edit stays inside the tree.
    #[test]
    fn the_shipped_manifest_cites_only_repository_paths() {
        let manifest = default_manifest_path();
        let text = read_text_bounded(&manifest).expect("release evidence manifest is readable");
        let manifest: super::super::gate_inputs::EvidenceManifest =
            toml::from_str(&text).expect("release evidence manifest is valid TOML");

        let escaping = manifest
            .requirements
            .iter()
            .flat_map(|requirement| requirement.evidence.iter())
            .filter(|evidence| !super::super::is_manifest_command_evidence(evidence))
            .filter(|evidence| escapes_repository(evidence))
            .cloned()
            .collect::<Vec<_>>();

        assert_eq!(escaping, Vec::<String>::new());
    }
}
