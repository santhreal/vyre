//! The duplicate-family report the audit writes for the dedup evidence path.
//!
//! The report is the machine-readable form of what checks 1 and 10 print, so a
//! consumer reads one artifact rather than parsing gate output. The two
//! detectors both feed it, so it is owned here rather than by either check, and
//! the gate holds the committed artifact to what the live registry produces.
//!
//! It was reachable only through a `--duplicate-report-json` path on a
//! `lego-audit` command that no longer parsed one, so the committed evidence
//! named a regeneration command nothing implemented. `--write` on this gate is
//! that command.

use std::path::Path;

use xtask::artifact_gate::{settle, Generated};
use xtask::artifact_paths::LEGO_AUDIT_DUPLICATES_ARTIFACT;

use super::*;

/// The command line recorded inside the artifact, and the one that rebuilds it.
///
/// Release evidence derives the expected string from the argument vector it
/// spawns, so this must stay the literal `xtask <gate> --write` form rather than
/// the `--duplicate-report-json` shape the shared helper produces for the
/// `whats-similar` command, which parses that flag.
const GENERATOR_COMMAND: &str = "xtask lego-duplicate-report --write";

/// Holds the lego duplicate-family evidence to the live registry.
pub struct LegoDuplicateReport;

impl xtask::gate::GateBehavior for LegoDuplicateReport {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.cover(Coverage::complete("registered operations", ops.len()));
        let path = Path::new(LEGO_AUDIT_DUPLICATES_ARTIFACT);
        let duplicates = lego_duplicate_report(&ops, GENERATOR_COMMAND);
        report.note(format!(
            "{} duplicate family(ies) across the no-reinvention and operand-shape detectors",
            duplicates.families.len()
        ));
        let recorded = xtask::evidence_record::EvidenceArtifact::new(
            xtask::evidence_record::MeasurementRecord::HostOnly,
            &duplicates,
        );
        match Generated::evidence(path, &recorded) {
            Ok(generated) => {
                report.produced(path);
                for finding in settle(&ctx.root, "lego-duplicate-report", &[generated], ctx.write) {
                    report.find(finding);
                }
            }
            Err(finding) => report.find(finding),
        }
        Ok(report)
    }
}

fn lego_duplicate_report(ops: &[OpInfo], generator_command: &str) -> DuplicateFamilyReport {
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

fn lego_duplicate_family(
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

fn lego_duplicate_subject(op: &OpInfo) -> DuplicateSubject {
    registered_op_duplicate_subject(&op.id, &op.fingerprint, op.own_nodes + op.composed_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::lego_audit::test_ops::op_with_fingerprint;

    /// WHY: release evidence derives the expected generator command from the
    /// argument vector it spawns, and the previous artifact named a `lego-audit
    /// --duplicate-report-json` path nothing parsed, so the committed evidence
    /// could not be reproduced. The recorded command is spelled out here so an
    /// edit to the const has to be a deliberate one made on both sides.
    #[test]
    fn the_report_records_the_gate_that_regenerates_it() {
        let report = lego_duplicate_report(&[], GENERATOR_COMMAND);
        assert_eq!(
            report.generator_command,
            "xtask lego-duplicate-report --write"
        );
        assert_eq!(report.detector_family, "registered-op-lego-audit");
        assert!(
            report.families.is_empty(),
            "no registered operation, no duplicate family: {:?}",
            report.families
        );
    }

    /// WHY: the report unions two detectors, and a version that collected one of
    /// them would look correct on a tree where the other found nothing. Two ops
    /// sharing a fingerprint are a pair for both, so both detector labels must
    /// appear. One pair is one family whatever found it, and the report credits
    /// every detector that reached it in the family's own label.
    #[test]
    fn both_detectors_reach_the_report() {
        let shared = vec![7u8; 128];
        let ops = [
            op_with_fingerprint("vyre-libs::a::twin", shared.clone()),
            op_with_fingerprint("vyre-libs::b::twin", shared),
        ];
        let report = lego_duplicate_report(&ops, "xtask lego-duplicate-report");
        let detectors: BTreeSet<&str> = report
            .families
            .iter()
            .flat_map(|family| family.detector.split('+'))
            .collect();
        assert_eq!(
            detectors,
            BTreeSet::from(["lego-audit:no-reinvention", "lego-audit:operand-shape"]),
            "both detectors must contribute: {detectors:?}"
        );
    }
}
