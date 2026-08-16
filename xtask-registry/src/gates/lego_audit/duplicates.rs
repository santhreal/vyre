//! The duplicate-family report the audit writes for the dedup evidence path.
//!
//! The report is the machine-readable form of what checks 1 and 10 print, so a
//! consumer reads one artifact rather than parsing gate output.

use super::*;

pub(super) fn lego_duplicate_report(
    ops: &[OpInfo],
    generator_command: &str,
) -> DuplicateFamilyReport {
    let mut families = Vec::new();
    families.extend(
        no_reinvention_pairs(ops)
            .into_iter()
            .map(|(score, left, right)| {
                lego_duplicate_family("lego-audit:no-reinvention", score, left, right)
            }),
    );
    families.extend(
        operand_shape_duplicate_pairs(ops)
            .into_iter()
            .map(|(score, left, right)| {
                lego_duplicate_family("lego-audit:operand-shape", score, left, right)
            }),
    );
    duplicate_family_report(generator_command, "registered-op-lego-audit", families)
}

pub(super) fn lego_duplicate_family(
    detector: &str,
    score: f64,
    left: &OpInfo,
    right: &OpInfo,
) -> DuplicateFamilyFinding {
    DuplicateFamilyFinding {
        family_id: registered_op_duplicate_family_id(&left.id, &right.id),
        detector: detector.to_string(),
        severity: duplicate_severity(score),
        score,
        left: lego_duplicate_subject(left),
        right: lego_duplicate_subject(right),
        import_owner: if left.tier <= right.tier {
            registered_op_owner_lane(&left.id).to_string()
        } else {
            registered_op_owner_lane(&right.id).to_string()
        },
        import_target: if left.tier <= right.tier {
            left.id.clone()
        } else {
            right.id.clone()
        },
        evidence: DuplicateEvidence {
            similarity_metric: "lego-ir-structural-similarity",
            left_metric: format!(
                "tier={:?}:own_nodes={}:composed_nodes={}:fingerprint_bytes={}",
                left.tier,
                left.own_nodes,
                left.composed_nodes,
                left.fingerprint.len()
            ),
            right_metric: format!(
                "tier={:?}:own_nodes={}:composed_nodes={}:fingerprint_bytes={}",
                right.tier,
                right.own_nodes,
                right.composed_nodes,
                right.fingerprint.len()
            ),
            dedup_action: "extract_shared_tier_2_5_primitive_or_compose_existing_op",
        },
    }
}

pub(super) fn lego_duplicate_subject(op: &OpInfo) -> DuplicateSubject {
    registered_op_duplicate_subject(&op.id, &op.fingerprint, op.own_nodes + op.composed_nodes)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use xtask::gates::dedup_report::duplicate_report_json_path;

    /// WHY: this preserves the explicit duplicate-report output path contract.
    /// The gate reads the flag off `GateCtx` and resolves it through the shared
    /// helper, so the test exercises both halves rather than a parser that no
    /// longer exists.
    #[test]
    fn duplicate_report_json_arg_accepts_path() {
        let ctx = xtask::gate::GateCtx::new(
            PathBuf::from("."),
            vec![
                "--with-repo".to_string(),
                "--duplicate-report-json".to_string(),
                "release/evidence/dedup/lego-duplicates.json".to_string(),
            ],
        );
        let resolved = duplicate_report_json_path(
            "--duplicate-report-json",
            ctx.flag("--duplicate-report-json"),
            "--duplicate-report-json requires a path",
        );
        assert_eq!(
            resolved.ok(),
            Some(PathBuf::from("release/evidence/dedup/lego-duplicates.json"))
        );
    }
}
