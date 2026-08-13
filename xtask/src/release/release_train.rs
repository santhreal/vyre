use std::sync::OnceLock;

use serde::Deserialize;

pub(crate) const RELEASE_TRAIN_TOML_PATH: &str = "release/release-train.toml";
const RELEASE_TRAIN_TOML: &str = include_str!("../../../release/release-train.toml");

#[derive(Debug, Deserialize)]
struct ReleaseTrainData {
    versions: Versions,
    tags: Tags,
    required_release_note_tokens: Vec<String>,
    required_packaging_steps: Vec<String>,
    package_verify_passed: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Versions {
    vyre: String,
    vyre_frontend_c: String,
}

#[derive(Debug, Deserialize)]
struct Tags {
    vyre_rc: String,
    vyre: String,
    policy: String,
}

static RELEASE_TRAIN: OnceLock<Result<ReleaseTrainData, String>> = OnceLock::new();

fn data() -> &'static ReleaseTrainData {
    crate::toml_config::data_or_exit(RELEASE_TRAIN.get_or_init(|| {
        crate::toml_config::parse_embedded_toml(RELEASE_TRAIN_TOML_PATH, RELEASE_TRAIN_TOML)
    }))
}

pub(crate) fn vyre_version() -> &'static str {
    data().versions.vyre.as_str()
}

pub(crate) fn vyre_frontend_c_version() -> &'static str {
    data().versions.vyre_frontend_c.as_str()
}

pub(crate) fn vyre_rc_tag() -> &'static str {
    data().tags.vyre_rc.as_str()
}

pub(crate) fn vyre_tag() -> &'static str {
    data().tags.vyre.as_str()
}

pub(crate) fn tag_story_fields() -> [(&'static str, &'static str); 2] {
    [("vyre_rc_tag", vyre_rc_tag()), ("vyre_tag", vyre_tag())]
}

pub(crate) fn tag_creation_order() -> [&'static str; 2] {
    [vyre_rc_tag(), vyre_tag()]
}

pub(crate) fn rc_to_final_tags() -> [(&'static str, &'static str); 1] {
    [(vyre_rc_tag(), vyre_tag())]
}

pub(crate) fn tag_policy() -> &'static str {
    data().tags.policy.as_str()
}

pub(crate) fn required_release_note_tokens() -> Vec<&'static str> {
    data()
        .required_release_note_tokens
        .iter()
        .map(String::as_str)
        .collect()
}

pub(crate) fn required_packaging_steps() -> Vec<&'static str> {
    data()
        .required_packaging_steps
        .iter()
        .map(String::as_str)
        .collect()
}

pub(crate) fn package_verify_passed() -> Vec<&'static str> {
    data()
        .package_verify_passed
        .iter()
        .map(String::as_str)
        .collect()
}

pub(crate) fn required_release_packages() -> [(&'static str, &'static str, &'static str); 4] {
    [
        ("vyre", vyre_version(), "vyre"),
        ("vyre-driver-cuda", vyre_version(), "vyre"),
        ("vyre-driver-wgpu", vyre_version(), "vyre"),
        ("vyre-frontend-c", vyre_frontend_c_version(), "vyre"),
    ]
}

pub(crate) fn release_group_version(group: &str) -> Option<&'static str> {
    match group {
        "vyre" => Some(vyre_version()),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use std::{path::Path, process::Command};

    /// The shell launch path must load the same RC and final tags as the Rust release contract.
    #[test]
    fn shell_release_loader_matches_canonical_tag_creation_order() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("Fix: xtask must live directly under the Vyre workspace.");
        let output = Command::new("bash")
            .arg("-c")
            .arg(
                r#"source scripts/lib/release_train.sh; vyre_load_release_train; printf '%s\n' "${VYRE_RELEASE_TAGS[@]}""#,
            )
            .current_dir(workspace)
            .output()
            .expect("Fix: bash must be available to validate the release launcher contract.");

        assert!(
            output.status.success(),
            "shell release loader failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout =
            String::from_utf8(output.stdout).expect("Fix: shell release tags must be valid UTF-8.");
        assert_eq!(
            stdout.lines().collect::<Vec<_>>(),
            super::tag_creation_order()
        );
    }
}
