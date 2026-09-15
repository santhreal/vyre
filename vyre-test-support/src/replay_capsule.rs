//! Replay capsules for persisting and minimizing differential failure counterexamples.
//!
//! Per Section 184.5:
//! - Records source, binary, device, driver, feature, seed, Program wire, input,
//!   tolerance, and mismatch identity in a replay capsule.
//! - Can be fed to minimizers and retained as deterministic regressions.

/// Capsule capturing the complete environment and inputs for a reproducible differential failure.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct ReplayCapsule {
    /// Test or benchmark source identity.
    pub source: String,
    /// Target binary identifier.
    pub binary: String,
    /// Hardware device string.
    pub device: String,
    /// Driver version or platform description.
    pub driver: String,
    /// Active feature set.
    pub feature: String,
    /// Random seed used for generation.
    pub seed: u64,
    /// Binary encoded VIR0 program bytes.
    pub program_wire: Vec<u8>,
    /// Input buffer payloads.
    pub input_bytes: Vec<Vec<u8>>,
    /// Registered ULP tolerance for the operation under test.
    pub tolerance_ulp: u32,
    /// Summary of the observed mismatch.
    pub mismatch_identity: String,
}

impl ReplayCapsule {
    /// Construct a new replay capsule.
    #[must_use]
    pub fn new(
        source: impl Into<String>,
        binary: impl Into<String>,
        device: impl Into<String>,
        driver: impl Into<String>,
        feature: impl Into<String>,
        seed: u64,
        program_wire: Vec<u8>,
        input_bytes: Vec<Vec<u8>>,
        tolerance_ulp: u32,
        mismatch_identity: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            binary: binary.into(),
            device: device.into(),
            driver: driver.into(),
            feature: feature.into(),
            seed,
            program_wire,
            input_bytes,
            tolerance_ulp,
            mismatch_identity: mismatch_identity.into(),
        }
    }

    /// Serialize capsule to a JSON string.
    ///
    /// # Errors
    /// Returns `Err` if JSON serialization fails.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("failed to serialize capsule: {e}"))
    }

    /// Deserialize capsule from a JSON string.
    ///
    /// # Errors
    /// Returns `Err` if JSON deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("failed to deserialize capsule: {e}"))
    }

    /// Attempt naive shrinking of input payload lengths while preserving mismatch structure.
    #[must_use]
    pub fn minimize_inputs(&self) -> Self {
        let mut shrunk = self.clone();
        for input in &mut shrunk.input_bytes {
            if input.len() > 16 {
                input.truncate(16);
            }
        }
        shrunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_capsule_json_roundtrip() {
        let capsule = ReplayCapsule {
            source: "tests/matrix_diff.rs".to_string(),
            binary: "vyre-driver-cuda".to_string(),
            device: "RTX 4090".to_string(),
            driver: "550.54.14".to_string(),
            feature: "full,cuda".to_string(),
            seed: 123456789,
            program_wire: vec![0x56, 0x49, 0x52, 0x30, 0x01, 0x02],
            input_bytes: vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            tolerance_ulp: 4,
            mismatch_identity: "ULP distance 8 > 4 at element 0".to_string(),
        };

        let json = capsule.to_json().expect("serialization must succeed");
        let roundtripped = ReplayCapsule::from_json(&json).expect("deserialization must succeed");

        assert_eq!(capsule, roundtripped);
    }
}
