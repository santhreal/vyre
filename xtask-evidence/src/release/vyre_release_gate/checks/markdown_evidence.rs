use super::*;

pub(crate) fn check_markdown_evidence_path_ready(
    requirement: &Requirement,
    path: &Path,
    manifest_path: &str,
    failures: &mut Vec<String>,
) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read markdown evidence `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return;
        }
    };
    if text.trim().is_empty() {
        failures.push(format!(
            "requirement `{}` markdown evidence `{manifest_path}` is empty",
            requirement.id
        ));
    }
    for marker in [
        "status: blocked",
        "status: open",
        "status: pending",
        "todo",
        "fixme",
        "placeholder",
        "stub",
        "tbd",
        "to be filled",
    ] {
        for line in text.lines() {
            let lowered = line.to_ascii_lowercase();
            if xtask::text_markers::markdown_line_is_release_rule_text(&lowered) {
                continue;
            }
            if lowered.contains(marker) {
                failures.push(format!(
                    "requirement `{}` markdown evidence `{manifest_path}` contains unresolved marker `{marker}`",
                    requirement.id
                ));
                break;
            }
        }
    }
    if manifest_path.starts_with("evidence/docs/") && !text.contains("Evidence sources:") {
        failures.push(format!(
            "requirement `{}` markdown evidence `{manifest_path}` does not list evidence sources",
            requirement.id
        ));
    }
}
