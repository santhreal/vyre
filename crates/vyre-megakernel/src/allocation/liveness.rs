//! The one live range and the one peak every stage of the compiler reads.
//!
//! Ranking, resource records and the placement plan all ask the same question:
//! which stages does a value occupy, and how many bytes are held at once. Three
//! answers to that used to exist, and the figure the objective ordered was not
//! the figure the artifact recorded. Both now come from here.

use std::collections::BTreeMap;

use crate::identity::{ArtifactNodeId, ArtifactValueId, FusionGroupId};
use crate::schema::ResourceLifetime;

/// What one value contributes to the resident byte total, per candidate.
///
/// Derived from the graph, never from a candidate, so one derivation serves
/// every candidate the search scores. The candidate supplies only the grouping
/// and the stage of each group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ValueLiveness {
    /// Canonical value identity.
    pub(crate) value: ArtifactValueId,
    /// Packed bytes the value occupies.
    pub(crate) bytes: u64,
    /// Node that writes the value, when the graph produces it.
    pub(crate) producer: Option<ArtifactNodeId>,
    /// Nodes that read the value.
    pub(crate) consumers: Vec<ArtifactNodeId>,
    /// Whether a caller or a later submission reads the value after the last
    /// stage, which holds its storage to the end.
    pub(crate) survives_to_end: bool,
    /// Storage class the resource records state for the value.
    pub(crate) lifetime: ResourceLifetime,
    /// Prior retained value whose storage this value advances.
    pub(crate) retained_predecessor: Option<ArtifactValueId>,
}

/// Dependency stage the group holding `node` runs in.
pub(crate) fn stage_of(node: ArtifactNodeId, node_groups: &[FusionGroupId], stages: &[u32]) -> u32 {
    node_groups
        .get(node.0 as usize)
        .and_then(|group| stages.get(group.0 as usize))
        .copied()
        .unwrap_or(0)
}

/// Live range of one value, in dependency stages.
///
/// A value survives to the final stage when a caller reads it after the last
/// entry point or a later submission advances it, because its storage is held
/// until then whatever stage last touched it.
pub(crate) fn span(
    producer: Option<ArtifactNodeId>,
    consumers: &[ArtifactNodeId],
    survives_to_end: bool,
    node_groups: &[FusionGroupId],
    stages: &[u32],
    final_stage: u32,
) -> (u32, u32) {
    let first = producer.map_or(0, |node| stage_of(node, node_groups, stages));
    let mut last = consumers
        .iter()
        .copied()
        .map(|node| stage_of(node, node_groups, stages))
        .max()
        .unwrap_or(first);
    if survives_to_end {
        last = last.max(final_stage);
    }
    (first, last.max(first))
}

/// Largest byte total held at once under this grouping.
///
/// This is the figure candidate ranking prices as peak memory and the figure the
/// placement plan records, so a plan that disagrees with the ranking it won is a
/// refused compile rather than a silent difference.
///
/// A retained successor the artifact owns advances the storage of the value it
/// replaces instead of taking a second allocation, so a chain of them is charged
/// for the widest value in it once, for every stage any of them is live.
pub(crate) fn peak(values: &[ValueLiveness], node_groups: &[FusionGroupId], stages: &[u32]) -> u64 {
    let final_stage = stages.iter().copied().max().unwrap_or(0);
    let spans: Vec<(u32, u32)> = values
        .iter()
        .map(|value| {
            span(
                value.producer,
                &value.consumers,
                value.survives_to_end,
                node_groups,
                stages,
                final_stage,
            )
        })
        .collect();
    let storage = storage_groups(values);
    let last_stage = spans.iter().map(|span| span.1).max().unwrap_or(0);
    let mut peak = 0u64;
    let mut live: BTreeMap<u32, u64> = BTreeMap::new();
    for stage in 0..=last_stage {
        live.clear();
        for ((value, span), group) in values.iter().zip(&spans).zip(&storage) {
            if span.0 > stage || stage > span.1 {
                continue;
            }
            let held = live.entry(*group).or_default();
            *held = (*held).max(value.bytes);
        }
        let total = live
            .values()
            .fold(0u64, |total, bytes| total.saturating_add(*bytes));
        peak = peak.max(total);
    }
    peak
}

/// One storage group per retained chain the artifact owns, identified by the
/// first value in the chain, and one group per value for everything else.
///
/// A chain that leaves artifact-owned storage ends there: the caller binds the
/// successor, so the two hold different bytes and are charged separately.
fn storage_groups(values: &[ValueLiveness]) -> Vec<u32> {
    let owned: BTreeMap<u32, bool> = values
        .iter()
        .map(|value| {
            (
                value.value.0,
                super::owned_by_artifact(value.producer.is_some(), value.lifetime),
            )
        })
        .collect();
    let mut root: BTreeMap<u32, u32> = BTreeMap::new();
    for value in values {
        let joins = value
            .retained_predecessor
            .map(|prior| prior.0)
            .filter(|prior| {
                owned.get(prior).copied().unwrap_or(false)
                    && owned.get(&value.value.0).copied().unwrap_or(false)
            });
        let group = joins
            .and_then(|prior| root.get(&prior).copied())
            .or(joins)
            .unwrap_or(value.value.0);
        root.insert(value.value.0, group);
    }
    values
        .iter()
        .map(|value| root.get(&value.value.0).copied().unwrap_or(value.value.0))
        .collect()
}
