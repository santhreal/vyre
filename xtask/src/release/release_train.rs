//! The release train manifest: versions, tags and packaging obligations.
//!
//! `release/release-train.toml` is embedded at build time and is the single
//! source for the version every gate and document is checked against.

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

/// The version this release train ships.
pub fn vyre_version() -> &'static str {
    data().versions.vyre.as_str()
}

/// Release-candidate tag for this train.
pub fn vyre_rc_tag() -> &'static str {
    data().tags.vyre_rc.as_str()
}

/// Final release tag for this train.
pub fn vyre_tag() -> &'static str {
    data().tags.vyre.as_str()
}

/// The tag fields a release note must state, and their expected values.
pub fn tag_story_fields() -> [(&'static str, &'static str); 2] {
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

/// Tokens a release note must contain.
pub fn required_release_note_tokens() -> Vec<&'static str> {
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

/// Every package this train publishes, with its version and group.
pub fn required_release_packages() -> [(&'static str, &'static str, &'static str); 3] {
    [
        ("vyre", vyre_version(), "vyre"),
        ("vyre-driver-cuda", vyre_version(), "vyre"),
        ("vyre-driver-wgpu", vyre_version(), "vyre"),
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
    use std::process::Command;

    /// The shell launch path must load the same RC and final tags as the Rust release contract.
    ///
    /// Unix only, because the launcher is: `scripts/lib/toml_reader.sh` reads
    /// the manifest through `python3`, and on Windows that name resolves to the
    /// Store execution alias, which exits nonzero and prints nothing. The
    /// release hosts run the script; Windows never does.
    #[cfg(unix)]
    #[test]
    fn shell_release_loader_matches_canonical_tag_creation_order() {
        let workspace = crate::checkout::checkout_root();
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
