use std::path::Path;

use super::super::checks::*;
use super::super::gate_inputs::Requirement;
use xtask::release::release_train;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    if !requirement
        .evidence
        .iter()
        .any(|evidence| evidence.contains("cargo_full") && evidence.contains("version-matrix"))
    {
        failures.push(
            "requirement `version-story` must include the cargo_full version-matrix evidence command"
                .to_string(),
        );
    }
    let Some(matrix) = first_json_evidence(requirement, base_dir, "version-matrix.json", failures)
    else {
        return;
    };
    let blockers = array_len(&matrix, "blockers");
    if blockers != 0 {
        failures.push(format!(
            "requirement `version-story` matrix still reports {blockers} blocker(s)"
        ));
    }
    let vyre_release = matrix
        .get("requested_vyre_release")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if vyre_release != release_train::vyre_version() {
        failures.push(format!(
            "requirement `version-story` requested_vyre_release is `{vyre_release}`, expected `{}`",
            release_train::vyre_version()
        ));
    }
    if matrix
        .get("release_doc_tag_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        failures.push(format!(
            "requirement `version-story` release docs must not contain bare v{} tag commands",
            release_train::vyre_version()
        ));
    }
    if matrix
        .get("release_note_token_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        failures.push(
            "requirement `version-story` release-note docs must include every required version and product-scoped tag token"
                .to_string(),
        );
    }
    if matrix
        .get("missing_required_release_packages")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|packages| !packages.is_empty())
    {
        failures.push(
            "requirement `version-story` missing_required_release_packages must be empty"
                .to_string(),
        );
    }
    let required_release_packages = matrix
        .get("required_release_packages")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for (package, version, group) in release_train::required_release_packages() {
        if group != "vyre" {
            continue;
        }
        let required_package = format!("{package}@{version}");
        if !required_release_packages
            .iter()
            .any(|package| package.as_str() == Some(required_package.as_str()))
        {
            failures.push(format!(
                "requirement `version-story` required_release_packages must include `{required_package}`"
            ));
        }
    }
    let Some(tag_story) = matrix
        .get("tag_story")
        .and_then(serde_json::Value::as_object)
    else {
        failures
            .push("requirement `version-story` version matrix is missing `tag_story`".to_string());
        return;
    };
    for (field, expected) in [
        ("vyre_rc_tag", release_train::vyre_rc_tag()),
        ("vyre_tag", release_train::vyre_tag()),
    ] {
        let actual = tag_story
            .get(field)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if actual != expected {
            failures.push(format!(
                "requirement `version-story` tag_story.{field} is `{actual}`, expected `{expected}`"
            ));
        }
    }
    for required in release_train::required_release_note_tokens() {
        let present = tag_story
            .get("required_in_release_notes")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|entries| entries.iter().any(|entry| entry.as_str() == Some(required)));
        if !present {
            failures.push(format!(
                "requirement `version-story` tag_story.required_in_release_notes is missing `{required}`"
            ));
        }
    }
    check_tag_plan(requirement, base_dir, failures);
}

fn check_tag_plan(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(tag_plan) =
        first_json_evidence(requirement, base_dir, "release-tag-plan.json", failures)
    else {
        return;
    };
    let tag_plan_blockers = array_len(&tag_plan, "blockers");
    if tag_plan_blockers != 0 {
        failures.push(format!(
            "requirement `version-story` release-tag-plan reports {tag_plan_blockers} blocker(s)"
        ));
    }
    for (field, expected) in release_train::tag_story_fields() {
        if tag_plan.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
            failures.push(format!(
                "requirement `version-story` release-tag-plan {field} must be `{expected}`"
            ));
        }
    }
    if !tag_plan
        .get("required_gate_before_rc_tag")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|command| {
            command.contains("version-matrix")
                && command.contains("vyre-release-gate")
                && command.contains("scripts/apply-branch-protection.sh")
                && command.contains("cargo_full")
        })
    {
        failures.push(
            "requirement `version-story` release-tag-plan must require version matrix regeneration, release gate, and branch-protection application before RC tag creation"
                .to_string(),
        );
    }
    let order = tag_plan
        .get("tag_creation_order")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let ordered_tags = order
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect::<Vec<_>>();
    let (rc, final_tag) = (release_train::vyre_rc_tag(), release_train::vyre_tag());
    let rc_index = ordered_tags.iter().position(|tag| *tag == rc);
    let final_index = ordered_tags.iter().position(|tag| *tag == final_tag);
    if !matches!((rc_index, final_index), (Some(left), Some(right)) if left < right) {
        failures.push(format!(
            "requirement `version-story` release-tag-plan must list `{rc}` before `{final_tag}`"
        ));
    }
    if !tag_plan
        .get("required_gate_before_tag")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|command| {
            command.contains("version-matrix")
                && command.contains("vyre-release-gate")
                && command.contains("scripts/apply-branch-protection.sh")
                && command.contains("cargo_full")
        })
    {
        failures.push(
            "requirement `version-story` release-tag-plan must require version matrix regeneration, release gate, and branch-protection application before tag creation"
                .to_string(),
        );
    }
    let version_blockers = u64_field(&tag_plan, "version_matrix_blocker_count", u64::MAX);
    if version_blockers != 0 {
        failures.push(format!(
            "requirement `version-story` release-tag-plan carries {version_blockers} version matrix blocker(s)"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_story_requires_version_matrix_command() {
        let mut failures = Vec::new();
        let requirement = Requirement {
            id: "version-story".to_string(),
            title: "version story".to_string(),
            status: "closed".to_string(),
            evidence: vec!["evidence/version/version-matrix.json".to_string()],
        };
        check(&requirement, Path::new("."), &mut failures);
        assert!(
            failures
                .iter()
                .any(|f| f.contains("must include the cargo_full version-matrix evidence command")),
            "Fix: version-story requirement must fail when version-matrix command is absent: {failures:?}"
        );
    }

    #[test]
    fn version_story_accepts_write_mode_evidence_command() {
        let mut failures = Vec::new();
        let requirement = Requirement {
            id: "version-story".to_string(),
            title: "version story".to_string(),
            status: "closed".to_string(),
            evidence: vec!["cargo_full run --bin xtask -- version-matrix --write".to_string()],
        };
        check(&requirement, Path::new("."), &mut failures);
        assert!(
            !failures
                .iter()
                .any(|f| f.contains("must include the cargo_full version-matrix evidence command")),
            "Fix: version-story requirement must accept standard --write command: {failures:?}"
        );
    }
}
