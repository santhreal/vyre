//! Runtime evidence records for held-out math and NN release kernels and for scan
//! competitor corpora, with the validators that gate them.

/// Runtime evidence required for held-out math and NN release kernels.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReleaseMathNnKernelEvidence {
    /// Stable benchmark case id.
    pub case_id: &'static str,
    /// CPU oracle output digest.
    pub cpu_digest: u64,
    /// GPU/backend output digest.
    pub gpu_digest: u64,
    /// Absolute tolerance scaled by 1e9.
    pub tolerance_abs_e9: u64,
    /// Active kernel/device time in nanoseconds, or wall time when unavailable.
    pub active_time_ns: u64,
    /// Accounted transfer bytes for this run.
    pub transfer_bytes: u64,
    /// Selected planner/kernel path.
    pub selected_kernel_path: &'static str,
}

/// Validate held-out math and NN release-kernel evidence.
///
/// # Errors
/// Returns `Err` when digests, tolerance, active time, transfer accounting, or
/// planner path evidence is missing.
pub fn validate_release_math_nn_kernel_evidence(
    evidence: &ReleaseMathNnKernelEvidence,
) -> Result<(), String> {
    if evidence.cpu_digest == 0 {
        return Err(format!(
            "Fix: release math/NN evidence `{}` is missing cpu_digest.",
            evidence.case_id
        ));
    }
    if evidence.gpu_digest == 0 {
        return Err(format!(
            "Fix: release math/NN evidence `{}` is missing gpu_digest.",
            evidence.case_id
        ));
    }
    if evidence.tolerance_abs_e9 == 0 {
        return Err(format!(
            "Fix: release math/NN evidence `{}` is missing tolerance_abs_e9.",
            evidence.case_id
        ));
    }
    if evidence.active_time_ns == 0 {
        return Err(format!(
            "Fix: release math/NN evidence `{}` is missing active_time_ns.",
            evidence.case_id
        ));
    }
    if evidence.transfer_bytes == 0 {
        return Err(format!(
            "Fix: release math/NN evidence `{}` is missing transfer_bytes.",
            evidence.case_id
        ));
    }
    if evidence.selected_kernel_path.is_empty() {
        return Err(format!(
            "Fix: release math/NN evidence `{}` is missing selected_kernel_path.",
            evidence.case_id
        ));
    }
    Ok(())
}

/// Scan competitor corpus metadata required for release workload evidence.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReleaseScanCompetitorCorpusMetadata {
    /// Stable benchmark case id.
    pub case_id: &'static str,
    /// Rule family, for example literal, regex, mixed, or secret-token.
    pub rule_family: &'static str,
    /// Number of patterns in the rule set.
    pub pattern_count: u32,
    /// Literal density in basis points across the pattern set.
    pub literal_density_bps: u16,
    /// Construct classes represented by this scan corpus.
    pub construct_classes: &'static [&'static str],
    /// Haystack corpus id.
    pub haystack_corpus_id: &'static str,
    /// Competitor baseline engine.
    pub baseline_engine: &'static str,
    /// Exact semantic exclusions for unsupported competitor constructs.
    pub unsupported_construct_reasons: &'static [&'static str],
}

/// Validate scan competitor corpus metadata.
///
/// # Errors
/// Returns `Err` when required corpus, pattern, construct, baseline, or
/// unsupported-construct fields are missing or malformed.
pub fn validate_release_scan_competitor_corpus_metadata(
    metadata: &ReleaseScanCompetitorCorpusMetadata,
) -> Result<(), String> {
    if metadata.case_id.is_empty() {
        return Err("Fix: scan competitor metadata case_id must be non-empty.".to_string());
    }
    if metadata.rule_family.is_empty() {
        return Err(format!(
            "Fix: scan competitor metadata `{}` is missing rule_family.",
            metadata.case_id
        ));
    }
    if metadata.pattern_count == 0 {
        return Err(format!(
            "Fix: scan competitor metadata `{}` must record a positive pattern_count.",
            metadata.case_id
        ));
    }
    if metadata.literal_density_bps > 10_000 {
        return Err(format!(
            "Fix: scan competitor metadata `{}` literal_density_bps must be <= 10000.",
            metadata.case_id
        ));
    }
    if metadata.construct_classes.is_empty()
        || metadata
            .construct_classes
            .iter()
            .any(|construct| construct.is_empty())
    {
        return Err(format!(
            "Fix: scan competitor metadata `{}` must record non-empty construct_classes.",
            metadata.case_id
        ));
    }
    if metadata.haystack_corpus_id.is_empty() {
        return Err(format!(
            "Fix: scan competitor metadata `{}` is missing haystack_corpus_id.",
            metadata.case_id
        ));
    }
    if metadata.baseline_engine.is_empty() {
        return Err(format!(
            "Fix: scan competitor metadata `{}` is missing baseline_engine.",
            metadata.case_id
        ));
    }
    for reason in metadata.unsupported_construct_reasons {
        if reason.is_empty() || !reason.contains("Fix:") {
            return Err(format!(
                "Fix: scan competitor metadata `{}` unsupported construct reasons must be exact and actionable.",
                metadata.case_id
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_math_nn_kernel_evidence_accepts_complete_digest_and_planner_record() {
        let evidence = ReleaseMathNnKernelEvidence {
            case_id: "nn.linear_4bit_affine_grouped.1m",
            cpu_digest: 1,
            gpu_digest: 1,
            tolerance_abs_e9: 100_000,
            active_time_ns: 42,
            transfer_bytes: 64,
            selected_kernel_path: "cooperative",
        };

        validate_release_math_nn_kernel_evidence(&evidence)
            .expect("Fix: complete math/NN release evidence must pass.");
    }

    #[test]
    fn release_math_nn_kernel_evidence_rejects_missing_digest_or_planner_path() {
        let missing_digest = ReleaseMathNnKernelEvidence {
            case_id: "nn.linear_4bit_affine_grouped.1m",
            cpu_digest: 0,
            gpu_digest: 1,
            tolerance_abs_e9: 100_000,
            active_time_ns: 42,
            transfer_bytes: 64,
            selected_kernel_path: "cooperative",
        };
        let error = validate_release_math_nn_kernel_evidence(&missing_digest)
            .expect_err("Fix: missing CPU digest must reject.");
        assert!(error.contains("cpu_digest"));

        let missing_path = ReleaseMathNnKernelEvidence {
            selected_kernel_path: "",
            cpu_digest: 1,
            ..missing_digest
        };
        let error = validate_release_math_nn_kernel_evidence(&missing_path)
            .expect_err("Fix: missing planner path must reject.");
        assert!(error.contains("selected_kernel_path"));
    }

    #[test]
    fn scan_competitor_corpus_metadata_accepts_complete_scan_baseline_record() {
        let metadata = ReleaseScanCompetitorCorpusMetadata {
            case_id: "release.scan_ac_irregular.1m",
            rule_family: "mixed-literal-regex",
            pattern_count: 128,
            literal_density_bps: 6_250,
            construct_classes: &["literal", "bounded-repeat", "ascii-class"],
            haystack_corpus_id: "heldout:scan:irregular:1m",
            baseline_engine: "hyperscan-compatible",
            unsupported_construct_reasons: &[
                "look-around excluded from Hyperscan-compatible baseline. Fix: compare this fixture against regex-automata or route unsupported constructs to verifier-only evidence.",
            ],
        };

        validate_release_scan_competitor_corpus_metadata(&metadata)
            .expect("Fix: complete scan competitor metadata must pass");
    }

    #[test]
    fn scan_competitor_corpus_metadata_rejects_missing_baseline_or_weak_exclusion() {
        let missing_baseline = ReleaseScanCompetitorCorpusMetadata {
            case_id: "release.scan_ac_irregular.1m",
            rule_family: "mixed-literal-regex",
            pattern_count: 128,
            literal_density_bps: 6_250,
            construct_classes: &["literal"],
            haystack_corpus_id: "heldout:scan:irregular:1m",
            baseline_engine: "",
            unsupported_construct_reasons: &[],
        };
        let error = validate_release_scan_competitor_corpus_metadata(&missing_baseline)
            .expect_err("Fix: missing scan baseline engine must reject");
        assert!(error.contains("baseline_engine"));

        let weak_exclusion = ReleaseScanCompetitorCorpusMetadata {
            baseline_engine: "hyperscan-compatible",
            unsupported_construct_reasons: &["look-around"],
            ..missing_baseline
        };
        let error = validate_release_scan_competitor_corpus_metadata(&weak_exclusion)
            .expect_err("Fix: weak unsupported construct reason must reject");
        assert!(error.contains("unsupported construct"));
    }
}
