//! Structural comparison for frontend Program and ProgramGraph representations.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

use vyre::ir::{Program, ProgramGraph};

/// Structural difference between two Program instances.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramDiff {
    /// Difference in operation count.
    pub op_count_delta: i64,
    /// Buffer declarations added.
    pub buffers_added: Vec<String>,
    /// Buffer declarations dropped.
    pub buffers_dropped: Vec<String>,
    /// Whether launch geometry changed.
    pub geometry_changed: bool,
    /// Whether programs are structurally identical.
    pub is_identical: bool,
}

/// Compare two Program representations structurally.
#[must_use]
pub fn diff_programs(before: &Program, after: &Program) -> ProgramDiff {
    let op_count_delta = (after.entry.len() as i64) - (before.entry.len() as i64);
    let mut buffers_added = Vec::new();
    let mut buffers_dropped = Vec::new();

    let before_bufs: BTreeMap<String, _> = before.buffers.iter().map(|b| (b.name.to_string(), b)).collect();
    let after_bufs: BTreeMap<String, _> = after.buffers.iter().map(|b| (b.name.to_string(), b)).collect();

    for name in after_bufs.keys() {
        if !before_bufs.contains_key(name) {
            buffers_added.push(name.clone());
        }
    }
    for name in before_bufs.keys() {
        if !after_bufs.contains_key(name) {
            buffers_dropped.push(name.clone());
        }
    }

    let geometry_changed = before.workgroup_size != after.workgroup_size;
    let is_identical = op_count_delta == 0
        && buffers_added.is_empty()
        && buffers_dropped.is_empty()
        && !geometry_changed;

    ProgramDiff {
        op_count_delta,
        buffers_added,
        buffers_dropped,
        geometry_changed,
        is_identical,
    }
}

/// Structural difference between two ProgramGraph instances.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphDiff {
    /// Node count difference.
    pub node_count_delta: i64,
    /// Nodes added by name.
    pub nodes_added: Vec<String>,
    /// Nodes removed by name.
    pub nodes_removed: Vec<String>,
    /// Whether graphs are structurally identical.
    pub is_identical: bool,
}

/// Compare two ProgramGraph representations structurally.
#[must_use]
pub fn diff_program_graphs(before: &ProgramGraph, after: &ProgramGraph) -> GraphDiff {
    let node_count_delta = (after.nodes().len() as i64) - (before.nodes().len() as i64);
    let mut nodes_added = Vec::new();
    let mut nodes_removed = Vec::new();

    let before_nodes: BTreeMap<&str, _> = before.nodes().iter().map(|n| (n.name.as_str(), n)).collect();
    let after_nodes: BTreeMap<&str, _> = after.nodes().iter().map(|n| (n.name.as_str(), n)).collect();

    for name in after_nodes.keys() {
        if !before_nodes.contains_key(name) {
            nodes_added.push((*name).to_string());
        }
    }
    for name in before_nodes.keys() {
        if !after_nodes.contains_key(name) {
            nodes_removed.push((*name).to_string());
        }
    }

    let is_identical = node_count_delta == 0
        && nodes_added.is_empty()
        && nodes_removed.is_empty();

    GraphDiff {
        node_count_delta,
        nodes_added,
        nodes_removed,
        is_identical,
    }
}
