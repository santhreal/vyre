//! Path resolution and bounded reads for release completion audit evidence.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use super::MAX_RELEASE_AUDIT_TEXT_BYTES;

pub(crate) fn markdown_line_is_release_rule_text(lowered: &str) -> bool {
    crate::output_arg::contains_any(
        lowered,
        &[
            "no-stub",
            "no shipped source",
            "must not",
            "not only",
            "not optional",
            "not a ",
            "no todo",
            "todo/fixme",
        ],
    )
}

pub(crate) fn paths_equal(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

pub(crate) fn resolve_manifest_path(base_dir: &Path, path: &str) -> PathBuf {
    crate::output_arg::resolve_path(base_dir, path)
}

pub(crate) fn is_checklist_artifact(entry: &str) -> bool {
    !entry.starts_with("cargo_full ")
        && (entry.ends_with(".json")
            || entry.ends_with(".md")
            || entry.ends_with(".yml")
            || entry.ends_with(".yaml")
            || entry.ends_with(".toml")
            || entry.starts_with("release/"))
}

pub(crate) fn is_manifest_command_evidence(evidence: &str) -> bool {
    evidence.starts_with("cargo_full ")
}

pub(crate) fn resolve_checklist_artifact_path(base_dir: &Path, path: &str) -> PathBuf {
    crate::output_arg::resolve_release_artifact_path(base_dir, path)
}

pub(crate) fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_RELEASE_AUDIT_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_RELEASE_AUDIT_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_RELEASE_AUDIT_TEXT_BYTES} byte release audit read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
