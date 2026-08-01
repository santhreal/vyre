//! Shared `--output` parsing for evidence-producing xtask commands.

use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub(crate) fn read_text_bounded(path: &Path, max_bytes: u64, context: &str) -> io::Result<String> {
    let mut reader = std::fs::File::open(path)?.take(max_bytes.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > max_bytes {
        let cap = if context.is_empty() {
            "read cap".to_string()
        } else {
            format!("{context} read cap")
        };
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} exceeds {max_bytes} byte {cap}", path.display()),
        ));
    }
    Ok(text)
}

pub(crate) fn parse_define(value: &str) -> (String, Option<String>) {
    match value.split_once('=') {
        Some((name, body)) => (name.to_string(), Some(body.to_string())),
        None => (value.to_string(), None),
    }
}

pub(crate) fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

pub(crate) fn resolve_path(base_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
    }
}

pub(crate) fn resolve_release_artifact_path(base_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate;
    }
    if path.starts_with("release/") {
        return base_dir
            .parent()
            .map(|workspace| workspace.join(candidate))
            .unwrap_or_else(|| base_dir.join(path));
    }
    base_dir.join(candidate)
}

pub(crate) fn parse_output_arg(
    args: &[String],
    command: &str,
    description: &str,
    default_output: impl FnOnce() -> PathBuf,
) -> Result<PathBuf, String> {
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!("USAGE:\n  cargo xtask {command} [--output PATH]\n\n  {description}");
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown {command} option `{other}`.")),
        }
    }
    Ok(output.unwrap_or_else(default_output))
}

pub(crate) fn write_json(path: &Path, value: &impl serde::Serialize) {
    let json = match serde_json::to_string_pretty(value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize `{}`: {error}", path.display());
            std::process::exit(1);
        }
    };
    let vyre_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let santh_root = vyre_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| vyre_root.clone());
    let json = normalize_serialized_workspace_paths(&json, &vyre_root, &santh_root);
    if let Err(error) = std::fs::write(path, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn normalize_serialized_workspace_paths(json: &str, vyre_root: &Path, santh_root: &Path) -> String {
    let json = replace_serialized_root(json, vyre_root, "", ".");
    replace_serialized_root(&json, santh_root, "../../../../", "../../../..")
}

fn replace_serialized_root(
    json: &str,
    root: &Path,
    descendant_prefix: &str,
    exact_replacement: &str,
) -> String {
    let Ok(encoded) = serde_json::to_string(root.to_string_lossy().as_ref()) else {
        return json.to_string();
    };
    let Some(fragment) = encoded
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
    else {
        return json.to_string();
    };
    json.replace(&format!("{fragment}/"), descendant_prefix)
        .replace(&format!("{fragment}\\\\\\"), descendant_prefix)
        .replace(fragment, exact_replacement)
}

pub(crate) fn cargo_runner(workspace_root: &Path) -> PathBuf {
    if let Some(runner) = std::env::var_os("VYRE_CARGO_RUNNER") {
        return PathBuf::from(runner);
    }
    let local = workspace_root.join("cargo_full");
    if local.is_file() {
        return local;
    }
    PathBuf::from("cargo_full")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the shared default path contract when no override is supplied.
    #[test]
    fn no_override_uses_command_default() {
        let args = vec!["xtask".to_string(), "docs-matrix".to_string()];
        assert_eq!(
            parse_output_arg(&args, "docs-matrix", "description", || {
                PathBuf::from("default.json")
            }),
            Ok(PathBuf::from("default.json"))
        );
    }

    /// Locks explicit output precedence so every evidence command writes to the requested artifact.
    #[test]
    fn explicit_output_overrides_command_default() {
        let args = vec![
            "xtask".to_string(),
            "docs-matrix".to_string(),
            "--output".to_string(),
            "custom.json".to_string(),
        ];
        assert_eq!(
            parse_output_arg(&args, "docs-matrix", "description", || {
                PathBuf::from("default.json")
            }),
            Ok(PathBuf::from("custom.json"))
        );
    }

    /// Locks the actionable missing-value diagnostic shared by all output-producing commands.
    #[test]
    fn missing_output_value_fails_with_fix() {
        let args = vec![
            "xtask".to_string(),
            "docs-matrix".to_string(),
            "--output".to_string(),
        ];
        assert_eq!(
            parse_output_arg(&args, "docs-matrix", "description", PathBuf::new),
            Err("Fix: --output requires a path.".to_string())
        );
    }

    /// Locks command-specific unknown-option diagnostics after parser centralization.
    #[test]
    fn unknown_option_names_owning_command() {
        let args = vec![
            "xtask".to_string(),
            "docs-matrix".to_string(),
            "--wat".to_string(),
        ];
        assert_eq!(
            parse_output_arg(&args, "docs-matrix", "description", PathBuf::new),
            Err("Fix: unknown docs-matrix option `--wat`.".to_string())
        );
    }

    /// Public evidence uses repository-relative paths instead of host-private Vyre paths.
    #[test]
    fn serialized_vyre_paths_are_repository_relative() {
        let json = r#"{"path":"/srv/Santh/libs/performance/matching/vyre/docs/RELEASE.md","message":"read /srv/Santh/libs/performance/matching/vyre/README.md"}"#;
        let normalized = normalize_serialized_workspace_paths(
            json,
            Path::new("/srv/Santh/libs/performance/matching/vyre"),
            Path::new("/srv/Santh"),
        );

        assert_eq!(
            normalized,
            r#"{"path":"docs/RELEASE.md","message":"read README.md"}"#
        );
    }

    /// Sibling release components retain a stable path from the public Vyre repository.
    #[test]
    fn serialized_santh_sibling_paths_use_public_relative_locations() {
        let json = r#"{"path":"/srv/Santh/tools/vyrec/README.md","root":"/srv/Santh"}"#;
        let normalized = normalize_serialized_workspace_paths(
            json,
            Path::new("/srv/Santh/libs/performance/matching/vyre"),
            Path::new("/srv/Santh"),
        );

        assert_eq!(
            normalized,
            r#"{"path":"../../../../tools/vyrec/README.md","root":"../../../.."}"#
        );
    }
}
