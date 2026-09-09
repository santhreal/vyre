//! Structural comparison between compiled Artifacts and SelectedPlans.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use vyre::compiler::{Artifact, SelectedPlan};

/// Structural difference between two compiled Artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDiff {
    /// Fusion group count delta.
    pub fusion_groups_delta: i64,
    /// Barrier count delta.
    pub barrier_count_delta: i64,
    /// Resource count delta.
    pub resource_count_delta: i64,
    /// Nodes added by ID.
    pub nodes_added: Vec<u32>,
    /// Nodes removed by ID.
    pub nodes_removed: Vec<u32>,
    /// Whether artifacts have identical content digests.
    pub digest_matches: bool,
    /// Whether artifacts are structurally identical.
    pub is_identical: bool,
}

/// Compare two compiled Artifact representations structurally.
#[must_use]
pub fn diff_artifacts(before: &Artifact, after: &Artifact) -> ArtifactDiff {
    let p1 = before.selected_plan();
    let p2 = after.selected_plan();

    let fusion_groups_delta = (p2.fusion.len() as i64) - (p1.fusion.len() as i64);
    let barrier_count_delta = (p2.barriers.len() as i64) - (p1.barriers.len() as i64);
    let resource_count_delta = (after.resources().len() as i64) - (before.resources().len() as i64);

    let before_nodes: BTreeMap<u32, _> = before.nodes().iter().map(|n| (n.id.0, n)).collect();
    let after_nodes: BTreeMap<u32, _> = after.nodes().iter().map(|n| (n.id.0, n)).collect();

    let mut nodes_added = Vec::new();
    let mut nodes_removed = Vec::new();

    for id in after_nodes.keys() {
        if !before_nodes.contains_key(id) {
            nodes_added.push(*id);
        }
    }
    for id in before_nodes.keys() {
        if !after_nodes.contains_key(id) {
            nodes_removed.push(*id);
        }
    }

    let digest_matches = before.digest() == after.digest();
    let is_identical = digest_matches
        && fusion_groups_delta == 0
        && barrier_count_delta == 0
        && resource_count_delta == 0
        && nodes_added.is_empty()
        && nodes_removed.is_empty();

    ArtifactDiff {
        fusion_groups_delta,
        barrier_count_delta,
        resource_count_delta,
        nodes_added,
        nodes_removed,
        digest_matches,
        is_identical,
    }
}

/// Structural difference between two SelectedPlans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDiff {
    /// Fusion groups count delta.
    pub fusion_groups_delta: i64,
    /// Barriers count delta.
    pub barrier_count_delta: i64,
    /// Search candidate count delta.
    pub candidate_count_delta: i64,
    /// Whether plans are identical.
    pub is_identical: bool,
}

/// Compare two SelectedPlan representations structurally.
#[must_use]
pub fn diff_selected_plans(before: &SelectedPlan, after: &SelectedPlan) -> PlanDiff {
    let fusion_groups_delta = (after.fusion.len() as i64) - (before.fusion.len() as i64);
    let barrier_count_delta = (after.barriers.len() as i64) - (before.barriers.len() as i64);
    let before_candidates: i64 = before
        .certificate
        .derived
        .iter()
        .map(|d| d.derived as i64)
        .sum();
    let after_candidates: i64 = after
        .certificate
        .derived
        .iter()
        .map(|d| d.derived as i64)
        .sum();
    let candidate_count_delta = after_candidates - before_candidates;

    let is_identical = fusion_groups_delta == 0
        && barrier_count_delta == 0
        && candidate_count_delta == 0
        && before == after;

    PlanDiff {
        fusion_groups_delta,
        barrier_count_delta,
        candidate_count_delta,
        is_identical,
    }
}
