//! What the release gate is given: the evidence manifest it reads, the mode it
//! was invoked in, and the cap on any text it will read.

use serde::Deserialize;

pub(super) const MAX_RELEASE_GATE_TEXT_BYTES: u64 = 16_777_216;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GateMode {
    Final,
    Prepublish,
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
    pub(super) minimum_evidence: usize,
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
            minimum_evidence: 0,
        }
    }
}
