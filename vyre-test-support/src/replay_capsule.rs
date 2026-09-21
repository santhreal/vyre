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

    /// Shorten input prefixes while replay reproduces the recorded mismatch.
    ///
    /// `replay` returns the observed mismatch identity, or `None` for a passing
    /// case. A different failure is not a reproducer. At most `max_attempts`
    /// replays run, including the initial case. Budget exhaustion returns the
    /// smallest reproducer found, not a claim of global minimality. The replay
    /// implementation must enforce its own per-attempt execution deadline.
    ///
    /// # Errors
    /// Returns an error for a zero budget, an unreproducible initial mismatch,
    /// or a replay execution error.
    pub fn minimize_inputs(
        &self,
        max_attempts: usize,
        mut replay: impl FnMut(&Self) -> Result<Option<String>, String>,
    ) -> Result<Self, String> {
        if max_attempts == 0 {
            return Err("replay minimization requires at least one attempt".to_string());
        }
        if replay(self)?.as_deref() != Some(self.mismatch_identity.as_str()) {
            return Err("the initial replay did not reproduce the recorded mismatch".to_string());
        }
        let mut attempts = 1;
        let mut shrunk = self.clone();
        for index in 0..shrunk.input_bytes.len() {
            let mut step = shrunk.input_bytes[index].len();
            while step > 0 {
                while shrunk.input_bytes[index].len() >= step {
                    if attempts == max_attempts {
                        return Ok(shrunk);
                    }
                    let previous_len = shrunk.input_bytes[index].len();
                    let candidate_len = previous_len - step;
                    shrunk.input_bytes[index].truncate(candidate_len);
                    attempts += 1;
                    if replay(&shrunk)?.as_deref() != Some(self.mismatch_identity.as_str()) {
                        // Every candidate is a prefix of the original input.
                        // Restore rejected bytes without another allocation.
                        shrunk.input_bytes[index].extend_from_slice(
                            &self.input_bytes[index][candidate_len..previous_len],
                        );
                        break;
                    }
                }
                step /= 2;
            }
        }
        Ok(shrunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capsule(input_bytes: Vec<Vec<u8>>) -> ReplayCapsule {
        ReplayCapsule::new(
            "synthetic-source",
            "synthetic-binary",
            "host",
            "synthetic-driver",
            "default",
            7,
            vec![1, 2, 3],
            input_bytes,
            0,
            "recorded-mismatch",
        )
    }

    /// WHY: shrinking must preserve the observed failure, its bytes and its
    /// metadata, rather than truncating to a fixed width. This covers prefix
    /// reduction; it does not claim arbitrary subsequence minimality.
    #[test]
    fn prefix_reduction_preserves_every_observed_failure_boundary() {
        for length in 0u8..=65 {
            for required in 0..=usize::from(length) {
                let original = capsule(vec![(0..length).collect()]);
                let mut attempts = 0;
                let shrunk = original
                    .minimize_inputs(128, |candidate| {
                        attempts += 1;
                        Ok((candidate.input_bytes[0].len() >= required)
                            .then(|| candidate.mismatch_identity.clone()))
                    })
                    .expect("replay succeeds");
                let mut expected = original.clone();
                expected.input_bytes[0].truncate(required);
                assert_eq!(shrunk, expected, "length={length}, required={required}");
                assert!(attempts <= 128);
                assert_eq!(original.input_bytes[0], (0..length).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn different_failures_do_not_replace_the_recorded_mismatch() {
        let original = capsule(vec![(0..33).collect(), (100..109).collect()]);
        let shrunk = original
            .minimize_inputs(64, |candidate| {
                Ok(Some(
                    if candidate.input_bytes[0].len() >= 17 && candidate.input_bytes[1].len() >= 2 {
                        candidate.mismatch_identity.clone()
                    } else {
                        "different-failure".to_string()
                    },
                ))
            })
            .expect("replay succeeds");
        let mut expected = original;
        expected.input_bytes[0].truncate(17);
        expected.input_bytes[1].truncate(2);
        assert_eq!(shrunk, expected);
    }

    #[test]
    fn replay_budget_bounds_all_candidates_and_includes_the_baseline() {
        let original = capsule(vec![(0..64).collect(), vec![1; 32]]);
        for budget in 0..=20 {
            let mut attempts = 0;
            let result = original.minimize_inputs(budget, |candidate| {
                attempts += 1;
                Ok(Some(if candidate == &original {
                    candidate.mismatch_identity.clone()
                } else {
                    "different-failure".to_string()
                }))
            });
            assert!(attempts <= budget);
            if budget == 0 {
                assert_eq!(
                    result.unwrap_err(),
                    "replay minimization requires at least one attempt"
                );
            } else {
                assert_eq!(result.unwrap(), original);
                assert!(attempts > 0);
            }
        }
    }

    #[test]
    fn baseline_failure_and_replay_errors_stop_without_a_reproducer() {
        let original = capsule(vec![vec![1; 32]]);
        for observed in [None, Some("different-failure".to_string())] {
            let mut attempts = 0;
            let error = original
                .minimize_inputs(32, |_| {
                    attempts += 1;
                    Ok(observed.clone())
                })
                .unwrap_err();
            assert_eq!(
                error,
                "the initial replay did not reproduce the recorded mismatch"
            );
            assert_eq!(attempts, 1);
        }
        for failure_at in [1, 2] {
            let mut attempts = 0;
            let error = original
                .minimize_inputs(32, |candidate| {
                    attempts += 1;
                    if attempts == failure_at {
                        Err("replay execution failed".to_string())
                    } else {
                        Ok(Some(candidate.mismatch_identity.clone()))
                    }
                })
                .unwrap_err();
            assert_eq!(error, "replay execution failed");
            assert_eq!(attempts, failure_at);
        }
    }

    #[test]
    fn empty_inputs_require_only_the_baseline_replay() {
        for inputs in [vec![], vec![vec![], vec![]]] {
            let original = capsule(inputs);
            let mut attempts = 0;
            let shrunk = original
                .minimize_inputs(2, |candidate| {
                    attempts += 1;
                    Ok(Some(candidate.mismatch_identity.clone()))
                })
                .unwrap();
            assert_eq!(shrunk, original);
            assert_eq!(attempts, 1);
        }
    }

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
