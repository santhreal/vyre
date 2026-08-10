//! GPU probe and no-skip test contract validation.

/// Observed GPU test/probe outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuProbeRecord<'a> {
    /// Test or gate name.
    pub gate: &'a str,
    /// Probe command or API used.
    pub probe: &'a str,
    /// Probe output or failure detail.
    pub detail: &'a str,
    /// Whether GPU was discovered.
    pub gpu_discovered: bool,
    /// Whether the gate skipped execution.
    pub skipped: bool,
}

/// GPU probe contract proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuProbeContractProof {
    /// Number of probe records.
    pub record_count: usize,
    /// Number of successful GPU discoveries.
    pub discovered_count: usize,
}

/// GPU probe contract validation errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuProbeContractError {
    /// No probe records supplied.
    EmptyRecords,
    /// Required metadata is empty.
    EmptyMetadata {
        /// Gate name.
        gate: String,
        /// Field.
        field: &'static str,
    },
    /// GPU test skipped instead of failing loudly.
    SkippedGpuGate {
        /// Gate name.
        gate: String,
    },
    /// Failed probe lacks adapter/device detail.
    MissingProbeFailureDetail {
        /// Gate name.
        gate: String,
    },
}

impl std::fmt::Display for GpuProbeContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRecords => write!(
                f,
                "GPU probe contract has no records. Fix: every GPU gate must record adapter/device probe evidence."
            ),
            Self::EmptyMetadata { gate, field } => write!(
                f,
                "GPU probe record `{gate}` has empty {field}. Fix: record gate, probe, and discovery detail."
            ),
            Self::SkippedGpuGate { gate } => write!(
                f,
                "GPU gate `{gate}` skipped. Fix: fail loudly with probe detail instead of treating GPU absence as normal."
            ),
            Self::MissingProbeFailureDetail { gate } => write!(
                f,
                "GPU gate `{gate}` failed discovery without adapter/device detail. Fix: report nvidia-smi, CUDA device count, or adapter enumeration output."
            ),
        }
    }
}

impl std::error::Error for GpuProbeContractError {}

/// Validate GPU probes and reject skip-on-no-GPU behavior.
pub fn validate_gpu_probe_contract(
    records: &[GpuProbeRecord<'_>],
) -> Result<GpuProbeContractProof, GpuProbeContractError> {
    if records.is_empty() {
        return Err(GpuProbeContractError::EmptyRecords);
    }
    let mut discovered_count = 0_usize;

    for record in records {
        for (field, value) in [
            ("gate", record.gate),
            ("probe", record.probe),
            ("detail", record.detail),
        ] {
            if value.trim().is_empty() {
                return Err(GpuProbeContractError::EmptyMetadata {
                    gate: record.gate.to_owned(),
                    field,
                });
            }
        }
        if record.skipped {
            return Err(GpuProbeContractError::SkippedGpuGate {
                gate: record.gate.to_owned(),
            });
        }
        if record.gpu_discovered {
            discovered_count += 1;
        } else if !has_probe_detail(record.detail) {
            return Err(GpuProbeContractError::MissingProbeFailureDetail {
                gate: record.gate.to_owned(),
            });
        }
    }

    Ok(GpuProbeContractProof {
        record_count: records.len(),
        discovered_count,
    })
}

fn has_probe_detail(detail: &str) -> bool {
    let lower = detail.to_ascii_lowercase();
    lower.contains("nvidia-smi")
        || lower.contains("cuda")
        || lower.contains("adapter")
        || lower.contains("device")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_probe_contract_accepts_discovered_gpu_records() {
        let proof = validate_gpu_probe_contract(&[GpuProbeRecord {
            gate: "cuda parity",
            probe: "nvidia-smi",
            detail: "NVIDIA GeForce RTX 5090 CUDA device 0",
            gpu_discovered: true,
            skipped: false,
        }])
        .expect("Fix: discovered GPU record should pass");

        assert_eq!(proof.record_count, 1);
        assert_eq!(proof.discovered_count, 1);
    }

    #[test]
    fn gpu_probe_contract_rejects_skipped_gpu_tests() {
        assert_eq!(
            validate_gpu_probe_contract(&[GpuProbeRecord {
                gate: "cuda parity",
                probe: "nvidia-smi",
                detail: "disabled: no GPU",
                gpu_discovered: false,
                skipped: true,
            }])
            .expect_err("skip-on-no-GPU must fail"),
            GpuProbeContractError::SkippedGpuGate {
                gate: "cuda parity".to_owned(),
            }
        );
    }

    #[test]
    fn gpu_probe_contract_requires_failure_detail() {
        assert_eq!(
            validate_gpu_probe_contract(&[GpuProbeRecord {
                gate: "cuda parity",
                probe: "nvidia-smi",
                detail: "not available",
                gpu_discovered: false,
                skipped: false,
            }])
            .expect_err("missing device detail should fail"),
            GpuProbeContractError::MissingProbeFailureDetail {
                gate: "cuda parity".to_owned(),
            }
        );
    }
}
