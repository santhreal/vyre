//! What the release gate is given: the evidence manifest it reads, the mode it
//! was invoked in, and the cap on any text it will read.

use serde::Deserialize;

pub(super) const MAX_RELEASE_GATE_TEXT_BYTES: u64 = 16_777_216;

/// Which release question the gate answers.
///
/// `Prepublish` is the default because it is the only mode that judges the tree
/// in front of it: it asks whether this checkout is ready to publish, and a
/// clean tree can answer yes. `LaunchComplete` asks whether the release has
/// already been published, verified and pushed, which no pre-release tree can
/// satisfy. Running it by default made the gate red by construction and kept it
/// red forever, so it is now reached only through `--launch-complete`, from the
/// one caller that runs after the publish.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum GateMode {
    #[default]
    Prepublish,
    LaunchComplete,
}

#[derive(Debug, Deserialize)]
pub(super) struct EvidenceManifest {
    pub(super) schema_version: u32,
    pub(super) release_contract_path: String,
    pub(super) release: ReleaseNames,
    pub(super) requirements: Vec<Requirement>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ReleaseNames {
    pub(super) vyre: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct Requirement {
    pub(super) id: String,
    pub(super) title: String,
    pub(super) status: String,
    pub(super) evidence: Vec<String>,
}

impl Requirement {
    /// A required requirement carrying no evidence paths, for a check that is
    /// handed the evidence directly.
    #[cfg(test)]
    pub(super) fn required(id: &str, title: &str) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            status: "required".to_string(),
            evidence: Vec::new(),
        }
    }
}
