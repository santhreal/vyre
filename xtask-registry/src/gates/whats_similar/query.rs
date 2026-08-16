//! The two scans this gate runs: one operation against the registry, or every
//! registered pair against each other.

use std::path::PathBuf;

use xtask::gate::{Finding, GateError, Report};
use xtask::gates::dedup_report::{
    duplicate_report_generator_command, structural_similarity, write_duplicate_report_json,
};

use crate::gates::lego_audit::{OpInfo, MIN_COMPARABLE_FINGERPRINT_BYTES};

use super::pair_facts::{
    implementation_family, known_distinct_implementation_family, pair_verdict,
    same_buffer_contract, same_centralized_family, tier_label,
};
use super::report::{all_pairs_duplicate_report, target_duplicate_report};

/// Score at or above which two operations are the same operation twice.
const DUPLICATE_SCORE: f64 = 0.95;

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

pub(super) fn run_target_query(
    report: &mut Report,
    ops: &[OpInfo],
    op_id: &str,
    top_n: usize,
    min_score: f64,
    duplicate_report_json: Option<&PathBuf>,
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
    for (score, _, _, op) in &scored {
        if *score >= DUPLICATE_SCORE {
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
        if let Err(error) = write_duplicate_report_json(path, &duplicates) {
            return Err(GateError::new(
                format!(
                    "could not write the duplicate family report `{}`: {error}",
                    path.display()
                ),
                "pass a writable path after --duplicate-report-json",
            ));
        }
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
        if let Err(error) = write_duplicate_report_json(path, &duplicates) {
            return Err(GateError::new(
                format!(
                    "could not write the duplicate family report `{}`: {error}",
                    path.display()
                ),
                "pass a writable path after --duplicate-report-json",
            ));
        }
    }

    report.note(format!("whats-similar: scanned {} registered ops for all-pairs duplicate candidates (min={:.2}, top={})",
        eligible.len(),
        min_score,
        top_n));
    if contract_variants > 0 {
        report.note(format!("  skipped {contract_variants} same-body pairs with different buffer contracts; these are wrapper/variant candidates, not raw duplicate ops."));
    }
    if centralized_family_variants > 0 {
        report.note(format!("  skipped {centralized_family_variants} same-family pairs already routed through a centralized builder."));
    }
    if distinct_family_variants > 0 {
        report.note(format!("  skipped {distinct_family_variants} known-distinct implementation-family pairs with shared scaffolding but different semantics."));
    }
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
