//! The two scans this gate runs: one operation against the registry, or every
//! registered pair against each other.

use std::path::{Path, PathBuf};

use xtask::artifact_gate::{settle, Generated};
use xtask::gate::{Finding, GateError, Report};
use xtask::gates::dedup_report::{
    duplicate_report_generator_command, structural_similarity, DuplicateFamilyReport,
};

use crate::gates::lego_audit::{OpInfo, MIN_COMPARABLE_FINGERPRINT_BYTES};

use super::pair_facts::{
    implementation_family, known_distinct_implementation_family, pair_verdict,
    same_buffer_contract, same_centralized_family, same_semantics, tier_label,
};
use super::report::{all_pairs_duplicate_report, target_duplicate_report};

/// Score at or above which two operations are the same operation twice.
const DUPLICATE_SCORE: f64 = 0.95;

/// Pairs skipped because the two bodies match and the buffer contracts differ.
///
/// Semantic identity qualification currently leaves no skipped pair. A gate
/// that reports zero findings while stepping over a pair has described the
/// surface rather than judged it, so every class it steps over carries a zero
/// ceiling. A new skipped pair is a duplicate nobody looked at and fails here.
const CONTRACT_VARIANT_CEILING: usize = 0;

/// Pairs skipped because both are routed through one centralized builder.
///
/// Semantic identity qualification currently leaves no skipped pair.
const CENTRALIZED_FAMILY_CEILING: usize = 0;

/// Pairs skipped because their families are recorded as distinct.
///
/// Semantic identity qualification currently leaves no skipped pair.
const KNOWN_DISTINCT_FAMILY_CEILING: usize = 0;

/// One skipped class, its measured ceiling and what the skip claims.
struct SkipClass {
    /// What the class is called in the report.
    label: &'static str,
    /// Pairs the scan stepped over for this reason.
    counted: usize,
    /// The measured ceiling the class is held to.
    ceiling: usize,
    /// What the class asserts about the pairs in it.
    claim: &'static str,
    /// What to do when the class grew.
    fix: &'static str,
}

/// Judge each skipped class against its ceiling.
///
/// A class over its ceiling is one finding naming the class, because the pairs
/// it hides are exactly the ones the scan was run to find. A class under its
/// ceiling is reported the way the sweep reports an improved gate, so the
/// number here follows the registry down instead of holding slack a later
/// duplicate can grow into.
fn skip_class_findings(report: &mut Report, classes: &[SkipClass]) {
    for class in classes {
        if class.counted > class.ceiling {
            report.find(Finding::new(
                format!(
                    "`{}` skipped {} pair(s) against a measured {}; {}",
                    class.label, class.counted, class.ceiling, class.claim
                ),
                class.fix,
            ));
        } else {
            report.note(format!(
                "  skipped {} pair(s) as {} ({}); {}",
                class.counted,
                class.label,
                class.claim,
                if class.counted < class.ceiling {
                    format!(
                        "lower the ceiling from {} to {}",
                        class.ceiling, class.counted
                    )
                } else {
                    format!("at the measured ceiling of {}", class.ceiling)
                }
            ));
        }
    }
}

/// One finding for a pair that is the same shape twice.
fn duplicate_finding(left: &str, right: &str, score: f64) -> Finding {
    Finding::new(
        format!(
            "`{left}` and `{right}` are {:.0}% structurally identical, so the registry carries the same operation twice",
            score * 100.0
        ),
        "extract the shared body into a registered primitive and compose both operations from it, or record the pair as a known-distinct implementation family",
    )
}

fn settle_duplicate_report(
    report: &mut Report,
    root: &Path,
    write: bool,
    path: &Path,
    duplicates: &DuplicateFamilyReport,
) {
    match Generated::json(path, duplicates) {
        Ok(generated) => {
            report.produced(path);
            for finding in settle(root, "whats-similar", &[generated], write) {
                report.find(finding);
            }
        }
        Err(finding) => report.find(finding),
    }
}

pub(super) fn run_target_query(
    report: &mut Report,
    ops: &[OpInfo],
    op_id: &str,
    top_n: usize,
    min_score: f64,
    duplicate_report_json: Option<&PathBuf>,
    root: &Path,
    write: bool,
) -> Result<(), GateError> {
    let target = match ops.iter().find(|o| o.id == op_id) {
        Some(op) => op,
        None => {
            return Err(GateError::new(
                format!("operation id `{op_id}` is not registered"),
                "submit one OperationRegistration with a neutral builder for it, or name a registered id",
            ));
        }
    };

    let mut scored: Vec<(f64, bool, bool, &OpInfo)> = ops
        .iter()
        .filter(|o| o.id != target.id)
        .filter(|o| o.fingerprint.len() >= MIN_COMPARABLE_FINGERPRINT_BYTES)
        .filter(|o| same_semantics(target, o))
        .map(|o| {
            (
                structural_similarity(&target.fingerprint, &o.fingerprint),
                same_buffer_contract(target, o),
                same_centralized_family(target, o),
                o,
            )
        })
        .filter(|(s, _, _, _)| *s >= min_score)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (score, same_contract, same_family, op) in &scored {
        if *score >= DUPLICATE_SCORE
            && *same_contract
            && !*same_family
            && !known_distinct_implementation_family(target, op)
        {
            report.find(duplicate_finding(op_id, &op.id, *score));
        }
    }
    scored.truncate(top_n);
    if let Some(path) = duplicate_report_json {
        let generator_command = duplicate_report_generator_command(
            &format!("whats-similar --op-id {}", target.id),
            path,
        );
        let duplicates = target_duplicate_report(target, &scored, &generator_command);
        settle_duplicate_report(report, root, write, path, &duplicates);
    }

    report.note(format!("whats-similar: target `{}` (tier={}, own_nodes={}, composed_nodes={}, fingerprint={} bytes)",
        target.id,
        tier_label(target.tier),
        target.own_nodes,
        target.composed_nodes,
        target.fingerprint.len()));

    if scored.is_empty() {
        report.note(format!("  ✓ no neighbors at score ≥ {:.2}. The op shape is novel (or your fingerprint is too short).",
            min_score));
        return Ok(());
    }

    report.note(format!(
        "  Top {} matches by bigram-cosine structural similarity:",
        scored.len()
    ));
    for (i, (score, same_contract, same_family, op)) in scored.iter().enumerate() {
        let verdict = pair_verdict(*score, *same_contract, *same_family);
        report.note(format!(
            "    {:>2}. {:>5.1}%  {}  ({})",
            i + 1,
            score * 100.0,
            op.id,
            verdict
        ));
        report.note(format!(
            "         tier={} own={} composed={} children={}",
            tier_label(op.tier),
            op.own_nodes,
            op.composed_nodes,
            op.children.len()
        ));
        if !same_contract {
            report.note(format!(
                "         contract=DIFFERENT target_buffers={} match_buffers={}",
                target.buffer_signature.len(),
                op.buffer_signature.len()
            ));
        }
        if *same_family {
            report.note(format!(
                "         implementation=CENTRALIZED family={}",
                implementation_family(target).unwrap_or("unknown")
            ));
        }
    }
    report.note("  Bar: ≥ 0.95 = duplicate, ≥ 0.80 = very similar, ≥ 0.50 = same family, < 0.20 = unrelated.".to_string());
    Ok(())
}

pub(super) fn run_all_pairs_query(
    report: &mut Report,
    ops: &[OpInfo],
    top_n: usize,
    min_score: f64,
    duplicate_report_json: Option<&PathBuf>,
    root: &Path,
    write: bool,
) -> Result<(), GateError> {
    let eligible: Vec<&OpInfo> = ops
        .iter()
        .filter(|op| op.fingerprint.len() >= MIN_COMPARABLE_FINGERPRINT_BYTES)
        .collect();
    let mut pairs: Vec<(f64, &OpInfo, &OpInfo)> = Vec::new();
    let mut contract_variants = 0usize;
    let mut centralized_family_variants = 0usize;
    let mut distinct_family_variants = 0usize;
    for left_index in 0..eligible.len() {
        for right in eligible.iter().skip(left_index + 1) {
            let left = eligible[left_index];
            let right = *right;
            let score = structural_similarity(&left.fingerprint, &right.fingerprint);
            if score >= min_score {
                if !same_semantics(left, right) {
                    continue;
                }
                if same_centralized_family(left, right) {
                    centralized_family_variants += 1;
                    continue;
                }
                if known_distinct_implementation_family(left, right) {
                    distinct_family_variants += 1;
                    continue;
                }
                if !same_buffer_contract(left, right) {
                    contract_variants += 1;
                    continue;
                }
                pairs.push((score, left, right));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (score, left, right) in &pairs {
        if *score >= DUPLICATE_SCORE {
            report.find(duplicate_finding(&left.id, &right.id, *score));
        }
    }
    pairs.truncate(top_n);
    if let Some(path) = duplicate_report_json {
        let generator_command = duplicate_report_generator_command("whats-similar --all", path);
        let duplicates = all_pairs_duplicate_report(&pairs, &generator_command);
        settle_duplicate_report(report, root, write, path, &duplicates);
    }

    report.note(format!("whats-similar: scanned {} registered ops for all-pairs duplicate candidates (min={:.2}, top={})",
        eligible.len(),
        min_score,
        top_n));
    skip_class_findings(
        report,
        &[
            SkipClass {
                label: "same-body pairs with different buffer contracts",
                counted: contract_variants,
                ceiling: CONTRACT_VARIANT_CEILING,
                claim: "each is a wrapper or variant of the other, not the same operation twice",
                fix: "compose the variant from the operation it wraps, or record why the two buffer contracts describe different operations; a pair added to this class is a duplicate the scan was told to ignore",
            },
            SkipClass {
                label: "same-family pairs routed through a centralized builder",
                counted: centralized_family_variants,
                ceiling: CENTRALIZED_FAMILY_CEILING,
                claim: "both bodies come from one shared builder, so the shape they share is the builder's",
                fix: "route the new pair through the builder its family already uses, or take it out of the family; a family that grows on its own is a shape nobody attributed",
            },
            SkipClass {
                label: "known-distinct implementation-family pairs",
                counted: distinct_family_variants,
                ceiling: KNOWN_DISTINCT_FAMILY_CEILING,
                claim: "the two families share scaffolding and mean different things",
                fix: "read the new pair and either merge it or record the distinction with its reason; a known-distinct list that grows silently is an exemption surface",
            },
        ],
    );
    if pairs.is_empty() {
        report.note("no registered-op pair crossed the duplicate or similarity floor");
        return Ok(());
    }
    for (index, (score, left, right)) in pairs.iter().enumerate() {
        let verdict = match *score {
            s if s >= 0.95 => "DUPLICATE",
            s if s >= 0.80 => "VERY SIMILAR",
            s if s >= 0.50 => "SIMILAR",
            _ => "RELATED",
        };
        report.note(format!(
            "  {:>2}. {:>5.1}%  {}",
            index + 1,
            score * 100.0,
            verdict
        ));
        report.note(format!(
            "      A: {} tier={} own={} composed={}",
            left.id,
            tier_label(left.tier),
            left.own_nodes,
            left.composed_nodes
        ));
        report.note(format!(
            "      B: {} tier={} own={} composed={}",
            right.id,
            tier_label(right.tier),
            right.own_nodes,
            right.composed_nodes
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{settle_duplicate_report, skip_class_findings, SkipClass};
    use crate::gates::lego_audit::test_ops::op_with_fingerprint;
    use xtask::gate::Report;
    use xtask::gates::dedup_report::duplicate_family_report;

    /// `SkipClass` and `skip_class_findings` are private to this module and the
    /// all-pairs scan that calls them needs the whole registered operation set,
    /// so no integration test can hand the rule a class and read the verdict.
    fn judge(counted: usize, ceiling: usize) -> Report {
        let mut report = Report::clean();
        skip_class_findings(
            &mut report,
            &[SkipClass {
                label: "same-body pairs with different buffer contracts",
                counted,
                ceiling,
                claim: "each is a wrapper or variant of the other",
                fix: "compose the variant from the operation it wraps",
            }],
        );
        report
    }

    /// Comparison mode must derive and compare the report without writing it;
    /// write mode must install those exact derived bytes.
    #[test]
    fn duplicate_report_respects_write_authority() {
        let root = tempfile::tempdir().unwrap();
        let path = std::path::Path::new("report.json");
        let duplicates = duplicate_family_report(
            "xtask whats-similar --all --duplicate-report-json report.json",
            "registered-op-ir-shape",
            Vec::new(),
        );

        let mut comparison = Report::clean();
        settle_duplicate_report(&mut comparison, root.path(), false, path, &duplicates);
        assert_eq!(comparison.count(), 1);
        assert!(!root.path().join(path).exists());

        let mut writer = Report::clean();
        settle_duplicate_report(&mut writer, root.path(), true, path, &duplicates);
        assert_eq!(writer.count(), 0);
        assert!(fs::read_to_string(root.path().join(path))
            .unwrap()
            .contains("\"family_count\": 0"));
    }

    /// Shape similarity is only a candidate generator. Different canonical
    /// semantic fingerprints prove that two similar programs compute different
    /// operations and must not be reported as duplicate implementations.
    #[test]
    fn shape_similarity_does_not_override_semantic_identity() {
        let fingerprint = (0..64_u8).collect::<Vec<_>>();
        let left = op_with_fingerprint("vyre-libs::reduce::sum", fingerprint.clone());
        let mut right = op_with_fingerprint("vyre-libs::reduce::max", fingerprint);
        right.semantic_fingerprint = [1; 32];
        let root = tempfile::tempdir().unwrap();
        let mut report = Report::clean();

        super::run_all_pairs_query(
            &mut report,
            &[left, right],
            5,
            0.8,
            None,
            root.path(),
            false,
        )
        .unwrap();

        assert_eq!(report.count(), 0);
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("no registered-op pair crossed")));
    }

    /// Equal canonical semantics and equal structure remain a hard duplicate,
    /// so the semantic discriminator cannot make the detector vacuous.
    #[test]
    fn equal_semantics_and_shape_remain_a_duplicate() {
        let fingerprint = (0..64_u8).collect::<Vec<_>>();
        let left = op_with_fingerprint("vyre-libs::alpha::left", fingerprint.clone());
        let right = op_with_fingerprint("vyre-libs::beta::right", fingerprint);
        let root = tempfile::tempdir().unwrap();
        let mut report = Report::clean();

        super::run_all_pairs_query(
            &mut report,
            &[left, right],
            5,
            0.8,
            None,
            root.path(),
            false,
        )
        .unwrap();

        assert_eq!(report.count(), 1);
        assert!(report.findings[0].message.contains("same operation twice"));
    }

    /// A class that grew past its measured ceiling is a finding naming both
    /// numbers, because the pairs it hides are the ones the scan was run to
    /// find.
    #[test]
    fn a_skip_class_over_its_ceiling_is_a_finding() {
        let report = judge(324, 323);
        assert_eq!(report.count(), 1);
        assert!(
            report.findings[0]
                .message
                .contains("skipped 324 pair(s) against a measured 323"),
            "{}",
            report.findings[0].message
        );
    }

    /// A class at its ceiling reports nothing and says so, so the sweep reads
    /// the same zero it reads for a gate with nothing to say.
    #[test]
    fn a_skip_class_at_its_ceiling_is_silent() {
        let report = judge(323, 323);
        assert_eq!(report.count(), 0);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("at the measured ceiling of 323")),
            "{:?}",
            report.notes
        );
    }

    /// A class that shrank asks for the ceiling to follow it down, which is how
    /// the sweep reports an improved gate. A ceiling left above the measurement
    /// is slack a later duplicate grows into without ever being reported.
    #[test]
    fn a_skip_class_under_its_ceiling_asks_for_the_lower_number() {
        let report = judge(300, 323);
        assert_eq!(report.count(), 0);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("lower the ceiling from 323 to 300")),
            "{:?}",
            report.notes
        );
    }
}
