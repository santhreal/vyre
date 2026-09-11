//! The duplicate family report this gate can write as JSON.

use xtask::gates::dedup_report::{
    duplicate_family_report, duplicate_severity, registered_op_duplicate_family_id,
    registered_op_duplicate_subject, registered_op_owner_lane, DuplicateEvidence,
    DuplicateFamilyFinding, DuplicateFamilyReport, DuplicateSubject,
};

use crate::gates::lego_audit::OpInfo;

use super::pair_facts::{implementation_family, tier_label};

pub(super) fn target_duplicate_report(
    target: &OpInfo,
    scored: &[(f64, bool, bool, &OpInfo)],
    generator_command: &str,
) -> DuplicateFamilyReport {
    let families = scored
        .iter()
        .map(|(score, same_contract, same_family, op)| {
            registered_op_duplicate_family(*score, target, op, *same_contract, *same_family)
        })
        .collect();
    duplicate_family_report(generator_command, "registered-op-ir-shape", families)
}

pub(super) fn all_pairs_duplicate_report(
    pairs: &[(f64, &OpInfo, &OpInfo)],
    generator_command: &str,
) -> DuplicateFamilyReport {
    let families = pairs
        .iter()
        .map(|(score, left, right)| {
            registered_op_duplicate_family(*score, left, right, true, false)
        })
        .collect();
    duplicate_family_report(generator_command, "registered-op-ir-shape", families)
}

fn registered_op_duplicate_family(
    score: f64,
    left: &OpInfo,
    right: &OpInfo,
    same_contract: bool,
    same_family: bool,
) -> DuplicateFamilyFinding {
    DuplicateFamilyFinding {
        family_id: registered_op_duplicate_family_id(&left.id, &right.id),
        detector: "whats-similar".to_string(),
        severity: duplicate_severity(score),
        score,
        left: registered_op_subject(left),
        right: registered_op_subject(right),
        import_owner: registered_op_import_owner(left, right, same_family),
        import_target: registered_op_import_target(left, right, same_contract, same_family),
        evidence: DuplicateEvidence {
            similarity_metric: "ir-shape-bigram-cosine",
            left_metric: format!(
                "tier={}:own_nodes={}:composed_nodes={}:fingerprint_bytes={}",
                tier_label(left.tier),
                left.own_nodes,
                left.composed_nodes,
                left.fingerprint.len()
            ),
            right_metric: format!(
                "tier={}:own_nodes={}:composed_nodes={}:fingerprint_bytes={}",
                tier_label(right.tier),
                right.own_nodes,
                right.composed_nodes,
                right.fingerprint.len()
            ),
            dedup_action: if same_family {
                "keep_shared_builder_family_and_remove_duplicate_registration"
            } else if same_contract {
                "extract_shared_primitive_or_reuse_existing_op"
            } else {
                "share_helper_without_merging_distinct_contracts"
            },
        },
    }
}

fn registered_op_subject(op: &OpInfo) -> DuplicateSubject {
    registered_op_duplicate_subject(&op.id, &op.fingerprint, op.own_nodes + op.composed_nodes)
}

fn registered_op_import_owner(left: &OpInfo, right: &OpInfo, same_family: bool) -> String {
    if same_family {
        return implementation_family(left)
            .or_else(|| implementation_family(right))
            .map(ToString::to_string)
            .unwrap_or_else(|| registered_op_owner_lane(&left.id).to_string());
    }
    if left.tier <= right.tier {
        registered_op_owner_lane(&left.id).to_string()
    } else {
        registered_op_owner_lane(&right.id).to_string()
    }
}

fn registered_op_import_target(
    left: &OpInfo,
    right: &OpInfo,
    same_contract: bool,
    same_family: bool,
) -> String {
    if same_family {
        return implementation_family(left)
            .or_else(|| implementation_family(right))
            .map(ToString::to_string)
            .unwrap_or_else(|| "shared_registered_op_family".to_string());
    }
    if !same_contract {
        return "shared_helper_for_contract_variants".to_string();
    }
    if left.tier <= right.tier {
        left.id.clone()
    } else {
        right.id.clone()
    }
}
