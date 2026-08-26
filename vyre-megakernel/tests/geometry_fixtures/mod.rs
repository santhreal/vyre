//! The geometry record the megakernel suites assert against.
//!
//! A geometry record names every launch fact one entry point is frozen with, so
//! a suite that wants one writes thirteen fields it does not care about. Three
//! modules had written that literal out, identical but for the node, phase and
//! workgroup they were asking about. One owner means a field added to the record
//! is added once.

// Every test binary compiles this module on its own, so a fixture a given suite
// does not ask for is unused in that binary.
#![allow(dead_code)]

use vyre_foundation::schedule::SchedulePhaseId;

use crate::identity::ArtifactNodeId;
use crate::schema::{EntryPersistence, GeometryRecord, LaunchResourceIntent};

/// Logical extents every fixture record covers.
pub(crate) const COVERAGE: [u64; 3] = [64, 1, 1];

/// One static entry point covering [`COVERAGE`] at `workgroup`, with no
/// predecessors, roles, ring slots, barrier phases or dynamic shared bytes.
pub(crate) fn geometry(node: u32, phase: u32, workgroup: [u32; 3]) -> GeometryRecord {
    GeometryRecord {
        node: ArtifactNodeId(node),
        phase: SchedulePhaseId(phase),
        predecessors: Vec::new(),
        logical_coverage: COVERAGE,
        grid: GeometryRecord::covering_grid(COVERAGE, workgroup).expect("positive extents"),
        workgroup_size: workgroup,
        vector_width: 1,
        roles: Vec::new(),
        ring_slots: 0,
        barrier_phases: Vec::new(),
        dynamic_shared_bytes: 0,
        launch_intent: LaunchResourceIntent::default(),
        persistence: EntryPersistence::Static,
    }
}
