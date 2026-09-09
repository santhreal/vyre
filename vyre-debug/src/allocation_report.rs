//! Physical allocation, memory layout, and resource lifetime inspection.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use vyre::compiler::Artifact;

/// Report on physical allocation decisions and memory maps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationReport {
    /// Resource allocations: slot -> (name, byte_count, lifetime).
    pub resources: BTreeMap<u32, ResourceAllocationInfo>,
    /// Global memory bytes allocated.
    pub total_global_bytes: u64,
    /// Workgroup shared memory bytes allocated.
    pub workgroup_shared_bytes: u32,
    /// Dynamic shared memory bytes allocated.
    pub dynamic_shared_bytes: u32,
}

/// Allocation details for one canonical resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAllocationInfo {
    /// Resource name.
    pub name: String,
    /// Resource byte count.
    pub byte_count: u64,
    /// Stated value lifetime.
    pub lifetime: String,
}

impl AllocationReport {
    /// Project allocation report from an authenticated Artifact.
    #[must_use]
    pub fn from_artifact(artifact: &Artifact) -> Self {
        let mut resources = BTreeMap::new();
        let mut total_global = 0;

        for res in artifact.resources() {
            total_global += res.byte_count;
            resources.insert(
                res.value.0,
                ResourceAllocationInfo {
                    name: res.name.clone(),
                    byte_count: res.byte_count,
                    lifetime: format!("{:?}", res.lifetime),
                },
            );
        }

        let mut workgroup_shared = 0;
        let mut dynamic_shared = 0;
        for geom in artifact.geometry() {
            dynamic_shared = dynamic_shared.max(geom.dynamic_shared_bytes);
            workgroup_shared = workgroup_shared
                .max(geom.workgroup_size[0] * geom.workgroup_size[1] * geom.workgroup_size[2]);
        }

        Self {
            resources,
            total_global_bytes: total_global,
            workgroup_shared_bytes: workgroup_shared,
            dynamic_shared_bytes: dynamic_shared,
        }
    }
}

/// Structural difference between two allocation reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationDiff {
    /// Difference in total global bytes.
    pub global_bytes_delta: i64,
    /// Resources added.
    pub resources_added: Vec<String>,
    /// Resources removed.
    pub resources_removed: Vec<String>,
    /// Whether allocations are identical.
    pub is_identical: bool,
}

/// Compare two allocation plans structurally.
#[must_use]
pub fn diff_allocations(before: &AllocationReport, after: &AllocationReport) -> AllocationDiff {
    let global_bytes_delta = (after.total_global_bytes as i64) - (before.total_global_bytes as i64);
    let mut resources_added = Vec::new();
    let mut resources_removed = Vec::new();

    for (slot, res) in &after.resources {
        if !before.resources.contains_key(slot) {
            resources_added.push(res.name.clone());
        }
    }
    for (slot, res) in &before.resources {
        if !after.resources.contains_key(slot) {
            resources_removed.push(res.name.clone());
        }
    }

    let is_identical =
        global_bytes_delta == 0 && resources_added.is_empty() && resources_removed.is_empty();

    AllocationDiff {
        global_bytes_delta,
        resources_added,
        resources_removed,
        is_identical,
    }
}
