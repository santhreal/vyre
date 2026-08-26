//! Selected launch geometry and the workspace plan a runtime allocates.
//!
//! Every record here is a projection of the schedule phase the search selected.
//! Validation is field-level and owned beside the records, so a consumer that
//! decodes an artifact and one that assembles a fresh plan are held to the same
//! rule.

use serde::{Deserialize, Serialize};
use vyre_foundation::schedule::{PipelineRoleGroup, SchedulePhaseId, SynchronizationScope};

use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::identity::{ArtifactNodeId, ArtifactValueId};

use super::records::ResourceLifetime;

/// Byte alignment every workspace region starts on.
///
/// One value per region is addressed as a typed buffer on every target, and the
/// widest alignment any of them requires is 256 bytes. Packing regions tighter
/// would make an offset legal in the artifact and unbindable on the device.
pub const WORKSPACE_REGION_ALIGNMENT: u64 = 256;

/// Whether one entry point runs once per submission or drains a bounded queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryPersistence {
    /// One launch per submission.
    Static,
    /// A persistent entry draining a bounded device-side work queue.
    Persistent {
        /// Nonzero queue entries the selected schedule reserved.
        queue_capacity: u32,
    },
}

/// Resources one launch of an entry point intends to hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LaunchResourceIntent {
    /// Invocation-private bytes the selected phase requires.
    pub private_bytes: u64,
    /// Scalar register slots one invocation requires.
    pub registers_per_invocation: u32,
}

/// One synchronization boundary the selected schedule placed on an entry point.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BarrierPhaseRecord {
    /// Scope crossed by the boundary.
    pub scope: SynchronizationScope,
    /// Selected phases the boundary synchronizes.
    pub phases: Vec<SchedulePhaseId>,
}

/// Complete compiler-selected launch geometry of one entry point.
///
/// Every field is a projection of the selected schedule phase that covers the
/// node, so nothing downstream has a geometry question left to answer. A
/// consumer that computed one of these instead of reading it would launch a
/// shape the emitted module does not have.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryRecord {
    /// Node this geometry launches.
    pub node: ArtifactNodeId,
    /// Selected schedule phase this geometry projects.
    pub phase: SchedulePhaseId,
    /// Entry points that must complete before this one launches.
    pub predecessors: Vec<ArtifactNodeId>,
    /// Exact logical points the phase covers on each axis.
    pub logical_coverage: [u64; 3],
    /// Workgroup count launched on each axis.
    pub grid: [u32; 3],
    /// Workgroup dimensions the search selected for this node.
    ///
    /// Every consumer launches this geometry. The workgroup the node program
    /// declares is an input to the search, not its result, so a consumer that
    /// reads the program instead of this record can launch a shape the emitted
    /// module does not have.
    pub workgroup_size: [u32; 3],
    /// Selected vector width.
    pub vector_width: u32,
    /// Producer and consumer role groups of a bounded pipeline, empty when the
    /// schedule formed none.
    pub roles: Vec<PipelineRoleGroup>,
    /// Ring slots of that pipeline, zero when the schedule formed none.
    pub ring_slots: u32,
    /// Synchronization boundaries covering this entry point.
    pub barrier_phases: Vec<BarrierPhaseRecord>,
    /// Workgroup-shared bytes the launch reserves.
    pub dynamic_shared_bytes: u32,
    /// Resources one launch intends to hold.
    pub launch_intent: LaunchResourceIntent,
    /// Whether the entry point is launched once or drains a queue.
    pub persistence: EntryPersistence,
}

impl GeometryRecord {
    /// Launch grid that exactly covers `coverage` at `workgroup`.
    ///
    /// # Errors
    ///
    /// Returns when an axis is zero or the covering grid exceeds `u32`.
    pub fn covering_grid(
        coverage: [u64; 3],
        workgroup: [u32; 3],
    ) -> Result<[u32; 3], CompileError> {
        let mut grid = [0u32; 3];
        for (axis, slot) in grid.iter_mut().enumerate() {
            let points = coverage[axis];
            let lanes = u64::from(workgroup[axis]);
            if points == 0 || lanes == 0 {
                return Err(failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("artifact.geometry.axis[{axis}]"),
                    "selected geometry covers zero points",
                    "select a schedule phase whose coverage and workgroup are positive on every axis",
                ));
            }
            *slot = u32::try_from(points.div_ceil(lanes)).map_err(|_| {
                failure(
                    CompilerFailureKind::ResourceOverflow,
                    format!("artifact.geometry.grid[{axis}]"),
                    "covering launch grid exceeds the u32 launch limit",
                    "tile the phase so its covering grid fits one launch",
                )
            })?;
        }
        Ok(grid)
    }

    /// Reject a record whose geometry no consumer could launch.
    ///
    /// # Errors
    ///
    /// Returns when an extent is zero, the grid does not exactly cover the
    /// logical points, a pipeline role group is empty, or a persistent entry
    /// reserves no queue.
    pub fn validate(&self) -> Result<(), CompileError> {
        let path = format!("artifact.geometry[{}]", self.node.0);
        if self.vector_width == 0 {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("{path}.vector_width"),
                "selected vector width is zero",
                "record the vector width the selected phase carries",
            ));
        }
        if self.grid != Self::covering_grid(self.logical_coverage, self.workgroup_size)? {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("{path}.grid"),
                "recorded launch grid does not exactly cover the recorded logical points",
                "record the grid the selected coverage and workgroup imply",
            ));
        }
        if self.ring_slots == 0 && !self.roles.is_empty() {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("{path}.ring_slots"),
                "pipeline roles are assigned with no ring slots",
                "record the ring depth the selected pipeline reserved",
            ));
        }
        if self.ring_slots != 0 && self.roles.is_empty() {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("{path}.roles"),
                "ring slots are reserved with no pipeline role assignment",
                "record the producer and consumer groups the selected pipeline assigned",
            ));
        }
        if let Some(empty) = self.roles.iter().position(|group| group.workers == 0) {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("{path}.roles[{empty}].workers"),
                "pipeline role group assigns no worker",
                "record the worker count the selected pipeline assigned",
            ));
        }
        if self.persistence == (EntryPersistence::Persistent { queue_capacity: 0 }) {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("{path}.persistence"),
                "persistent entry reserves no queue entry",
                "record the queue capacity the selected schedule reserved",
            ));
        }
        Ok(())
    }
}

/// One cross-entry workspace region with its exact offset and live range.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRegion {
    /// Typed graph value stored in the region.
    pub value: ArtifactValueId,
    /// Byte offset of the region inside the workspace allocation.
    pub offset: u64,
    /// Bytes reserved for the value.
    pub bytes: u64,
    /// Semantic lifetime of the value.
    pub lifetime: ResourceLifetime,
    /// First entry point that reads or writes the region.
    pub first_entry: ArtifactNodeId,
    /// Last entry point that reads or writes the region.
    pub last_entry: ArtifactNodeId,
}

/// Exact workspace the runtime allocates before submitting an artifact.
///
/// The compiler assigns every offset. A runtime that packed the regions itself
/// would decide storage the selected schedule already decided, and two packers
/// disagreeing is a wrong-buffer bind rather than a slower one.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspacePlan {
    /// Bytes the whole workspace allocation requires.
    pub total_bytes: u64,
    /// Regions in ascending offset order.
    pub regions: Vec<WorkspaceRegion>,
}

impl WorkspacePlan {
    /// Reject a plan a runtime could not allocate or bind exactly.
    ///
    /// # Errors
    ///
    /// Returns when a region is empty, misaligned, overlaps its predecessor, or
    /// runs past the recorded allocation.
    pub fn validate(&self) -> Result<(), CompileError> {
        let mut next = 0u64;
        for (index, region) in self.regions.iter().enumerate() {
            let path = format!("artifact.workspace.regions[{index}]");
            if region.bytes == 0 {
                return Err(failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("{path}.bytes"),
                    "workspace region reserves no byte",
                    "record the packed byte count of the value",
                ));
            }
            if region.offset % WORKSPACE_REGION_ALIGNMENT != 0 {
                return Err(failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("{path}.offset"),
                    format!("workspace offset is not a multiple of {WORKSPACE_REGION_ALIGNMENT}"),
                    "align every region to the workspace alignment",
                ));
            }
            if region.offset < next {
                return Err(failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("{path}.offset"),
                    "workspace region overlaps the region before it",
                    "assign one disjoint offset per retained value",
                ));
            }
            let end = region.offset.checked_add(region.bytes).ok_or_else(|| {
                failure(
                    CompilerFailureKind::ResourceOverflow,
                    format!("{path}.bytes"),
                    "workspace region end exceeds the addressable range",
                    "reduce the retained working set",
                )
            })?;
            if end > self.total_bytes {
                return Err(failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("{path}.bytes"),
                    "workspace region runs past the recorded allocation",
                    "record the allocation the assigned offsets require",
                ));
            }
            next = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::schedule::PipelineRole;

    fn record() -> GeometryRecord {
        GeometryRecord {
            node: ArtifactNodeId(0),
            phase: SchedulePhaseId(0),
            predecessors: Vec::new(),
            logical_coverage: [64, 1, 1],
            grid: [2, 1, 1],
            workgroup_size: [32, 1, 1],
            vector_width: 1,
            roles: Vec::new(),
            ring_slots: 0,
            barrier_phases: Vec::new(),
            dynamic_shared_bytes: 0,
            launch_intent: LaunchResourceIntent::default(),
            persistence: EntryPersistence::Static,
        }
    }

    fn region(offset: u64, bytes: u64) -> WorkspaceRegion {
        WorkspaceRegion {
            value: ArtifactValueId(1),
            offset,
            bytes,
            lifetime: ResourceLifetime::Invocation,
            first_entry: ArtifactNodeId(0),
            last_entry: ArtifactNodeId(0),
        }
    }

    fn path_of(error: &CompileError) -> String {
        error
            .diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.clone())
            .unwrap_or_else(|| panic!("diagnostic carries no path: {error}"))
    }

    /// WHY: every field of a record is launched verbatim, so a record that
    /// contradicts itself is a wrong launch rather than a slow one. Each case is
    /// one contradiction a consumer would otherwise resolve by recomputing the
    /// field for itself, which is the disagreement this record exists to end.
    #[test]
    fn an_unlaunchable_record_is_rejected_at_the_field_that_states_it() {
        let cases: Vec<(&str, fn(&mut GeometryRecord), &str)> = vec![
            (
                "zero vector width",
                |record| record.vector_width = 0,
                "artifact.geometry[0].vector_width",
            ),
            (
                "grid too small to cover the recorded points",
                |record| record.grid = [1, 1, 1],
                "artifact.geometry[0].grid",
            ),
            (
                "grid larger than the recorded points require",
                |record| record.grid = [3, 1, 1],
                "artifact.geometry[0].grid",
            ),
            (
                "zero coverage axis",
                |record| record.logical_coverage[1] = 0,
                "artifact.geometry.axis[1]",
            ),
            (
                "zero workgroup axis",
                |record| record.workgroup_size[2] = 0,
                "artifact.geometry.axis[2]",
            ),
            (
                "pipeline roles with no ring depth",
                |record| {
                    record.roles = vec![PipelineRoleGroup {
                        role: PipelineRole::Producer,
                        workers: 1,
                    }];
                },
                "artifact.geometry[0].ring_slots",
            ),
            (
                "ring depth with no pipeline roles",
                |record| record.ring_slots = 2,
                "artifact.geometry[0].roles",
            ),
            (
                "pipeline role group with no worker",
                |record| {
                    record.ring_slots = 2;
                    record.roles = vec![PipelineRoleGroup {
                        role: PipelineRole::Consumer,
                        workers: 0,
                    }];
                },
                "artifact.geometry[0].roles[0].workers",
            ),
            (
                "persistent entry with no queue",
                |record| {
                    record.persistence = EntryPersistence::Persistent { queue_capacity: 0 };
                },
                "artifact.geometry[0].persistence",
            ),
        ];
        for (case, mutate, expected) in cases {
            let mut invalid = record();
            mutate(&mut invalid);
            let error = invalid
                .validate()
                .expect_err(&format!("{case} must not be launchable"));
            assert_eq!(path_of(&error), expected, "{case}");
        }
        record().validate().expect("the fixture record launches");
    }

    /// WHY: a covering grid is the one arithmetic every consumer used to do for
    /// itself, and an off-by-one there leaves the tail of the domain uncomputed.
    #[test]
    fn a_covering_grid_launches_every_logical_point_once() {
        for (coverage, workgroup, expected) in [
            ([100u64, 1, 1], [32u32, 1, 1], [4u32, 1, 1]),
            ([64, 8, 2], [32, 4, 2], [2, 2, 1]),
            ([1, 1, 1], [64, 1, 1], [1, 1, 1]),
            ([96, 1, 1], [32, 1, 1], [3, 1, 1]),
        ] {
            assert_eq!(
                GeometryRecord::covering_grid(coverage, workgroup).expect("positive extents"),
                expected
            );
        }
        for (coverage, workgroup) in [([0u64, 1, 1], [1u32, 1, 1]), ([1, 1, 1], [1, 0, 1])] {
            GeometryRecord::covering_grid(coverage, workgroup)
                .expect_err("a zero extent covers no point");
        }
        let error = GeometryRecord::covering_grid([u64::from(u32::MAX) * 2, 1, 1], [1, 1, 1])
            .expect_err("a grid past the launch limit is not launchable");
        assert_eq!(path_of(&error), "artifact.geometry.grid[0]");
    }

    /// WHY: the runtime allocates this plan and binds these offsets verbatim, so
    /// a plan that overlaps or overruns is a wrong-buffer bind rather than a
    /// slower one.
    #[test]
    fn a_workspace_plan_no_runtime_could_bind_is_rejected() {
        let cases: Vec<(&str, WorkspacePlan, &str)> = vec![
            (
                "region reserving no byte",
                WorkspacePlan {
                    total_bytes: 256,
                    regions: vec![region(0, 0)],
                },
                "artifact.workspace.regions[0].bytes",
            ),
            (
                "misaligned region",
                WorkspacePlan {
                    total_bytes: 512,
                    regions: vec![region(8, 256)],
                },
                "artifact.workspace.regions[0].offset",
            ),
            (
                "region overlapping the region before it",
                WorkspacePlan {
                    total_bytes: 1024,
                    regions: vec![region(0, 512), region(256, 256)],
                },
                "artifact.workspace.regions[1].offset",
            ),
            (
                "region running past the recorded allocation",
                WorkspacePlan {
                    total_bytes: 128,
                    regions: vec![region(0, 256)],
                },
                "artifact.workspace.regions[0].bytes",
            ),
            (
                "region end past the addressable range",
                WorkspacePlan {
                    total_bytes: u64::MAX,
                    regions: vec![region(WORKSPACE_REGION_ALIGNMENT, u64::MAX)],
                },
                "artifact.workspace.regions[0].bytes",
            ),
        ];
        for (case, plan, expected) in cases {
            let error = plan
                .validate()
                .expect_err(&format!("{case} must not be bindable"));
            assert_eq!(path_of(&error), expected, "{case}");
        }
        WorkspacePlan {
            total_bytes: 768,
            regions: vec![region(0, 256), region(512, 256)],
        }
        .validate()
        .expect("aligned disjoint regions inside the allocation bind");
        WorkspacePlan::default()
            .validate()
            .expect("an artifact with no workspace allocates nothing");
    }
}
