//! Structural and parametric schedule diffing.
//!
//! Compares two [`SchedulePlan`] or [`ScheduleTree`] structures and produces
//! structured change records.

use super::tree::{ScheduleOp, SchedulePlan, ScheduleTree};
use serde::{Deserialize, Serialize};

/// One discrete difference between two schedules.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleDiffItem {
    /// Operation kind or parameter changed at a specific tree path.
    OpModified {
        /// Breadth-first / path index.
        path: Vec<usize>,
        /// Previous schedule operation.
        before: ScheduleOp,
        /// New schedule operation.
        after: ScheduleOp,
    },
    /// Subtree inserted in the new schedule.
    SubtreeAdded {
        /// Insertion path.
        path: Vec<usize>,
        /// Node count of inserted subtree.
        nodes: usize,
    },
    /// Subtree removed in the new schedule.
    SubtreeRemoved {
        /// Removal path.
        path: Vec<usize>,
        /// Node count of removed subtree.
        nodes: usize,
    },
    /// Resource bound changed.
    ResourceBoundChanged {
        /// Resource name.
        name: String,
        /// Previous bound.
        before: u64,
        /// New bound.
        after: u64,
    },
}

/// A collection of differences between two schedule plans.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleDiff {
    /// Differences in schedule tree structure or parameters.
    pub items: Vec<ScheduleDiffItem>,
}

impl ScheduleDiff {
    /// Compute differences between two schedule plans.
    #[must_use]
    pub fn diff_plans(before: &SchedulePlan, after: &SchedulePlan) -> Self {
        let mut diff = Self::default();

        // Check resource bounds differences
        if before.resource_bounds.shared_bytes != after.resource_bounds.shared_bytes {
            diff.items.push(ScheduleDiffItem::ResourceBoundChanged {
                name: "shared_bytes".into(),
                before: before.resource_bounds.shared_bytes,
                after: after.resource_bounds.shared_bytes,
            });
        }
        if before.resource_bounds.logical_points != after.resource_bounds.logical_points {
            diff.items.push(ScheduleDiffItem::ResourceBoundChanged {
                name: "logical_points".into(),
                before: before.resource_bounds.logical_points,
                after: after.resource_bounds.logical_points,
            });
        }
        if before.resource_bounds.registers_per_invocation
            != after.resource_bounds.registers_per_invocation
        {
            diff.items.push(ScheduleDiffItem::ResourceBoundChanged {
                name: "registers_per_invocation".into(),
                before: before.resource_bounds.registers_per_invocation as u64,
                after: after.resource_bounds.registers_per_invocation as u64,
            });
        }

        // Tree diffing
        diff_tree_recursive(&before.root, &after.root, Vec::new(), &mut diff.items);

        diff
    }

    /// Return true if the two schedules are identical.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

fn diff_tree_recursive(
    before: &ScheduleTree,
    after: &ScheduleTree,
    path: Vec<usize>,
    out: &mut Vec<ScheduleDiffItem>,
) {
    match (before, after) {
        (ScheduleTree::Leaf(op_a), ScheduleTree::Leaf(op_b)) => {
            if op_a != op_b {
                out.push(ScheduleDiffItem::OpModified {
                    path,
                    before: op_a.clone(),
                    after: op_b.clone(),
                });
            }
        }
        (
            ScheduleTree::Node {
                op: op_a,
                child: child_a,
                ..
            },
            ScheduleTree::Node {
                op: op_b,
                child: child_b,
                ..
            },
        ) => {
            if op_a != op_b {
                out.push(ScheduleDiffItem::OpModified {
                    path: path.clone(),
                    before: op_a.clone(),
                    after: op_b.clone(),
                });
            }
            let mut next_path = path;
            next_path.push(0);
            diff_tree_recursive(child_a, child_b, next_path, out);
        }
        (ScheduleTree::Sequence(seq_a), ScheduleTree::Sequence(seq_b)) => {
            let min_len = seq_a.len().min(seq_b.len());
            for i in 0..min_len {
                let mut next_path = path.clone();
                next_path.push(i);
                diff_tree_recursive(&seq_a[i], &seq_b[i], next_path, out);
            }
            if seq_b.len() > seq_a.len() {
                for i in min_len..seq_b.len() {
                    let mut next_path = path.clone();
                    next_path.push(i);
                    out.push(ScheduleDiffItem::SubtreeAdded {
                        path: next_path,
                        nodes: seq_b[i].node_count(),
                    });
                }
            } else if seq_a.len() > seq_b.len() {
                for i in min_len..seq_a.len() {
                    let mut next_path = path.clone();
                    next_path.push(i);
                    out.push(ScheduleDiffItem::SubtreeRemoved {
                        path: next_path,
                        nodes: seq_a[i].node_count(),
                    });
                }
            }
        }
        _ => {
            // Structural change
            out.push(ScheduleDiffItem::SubtreeRemoved {
                path: path.clone(),
                nodes: before.node_count(),
            });
            out.push(ScheduleDiffItem::SubtreeAdded {
                path,
                nodes: after.node_count(),
            });
        }
    }
}
