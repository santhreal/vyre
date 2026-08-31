//! The unscheduled starting point every candidate is derived from.
//!
//! A logical graph states what the program computes and what extents each
//! region spans. Turning that into phases with a launch shape is the first
//! schedule decision of a compile, so it belongs to the crate that owns
//! selection. Foundation supplies the regions, extents, and dependencies as
//! facts; the baseline width below is the one shape the search starts from and
//! every production narrows or widens.

use vyre_foundation::logical::{LogicalExtent, LogicalProgramGraph};
use vyre_foundation::schedule::{
    ScheduleAxis, SchedulePhase, SchedulePhaseId, ScheduleResourceBounds, SelectedSchedule,
    SCHEDULE_IR_VERSION,
};

/// One thread per phase before any production widens it.
///
/// The unfused baseline stays in the candidate set, so this width is a real
/// candidate rather than a placeholder: it is the schedule every ranking is
/// measured against.
const BASELINE_WORKGROUP: [u32; 3] = [1, 1, 1];

/// One lane per invocation before any vectorizing production widens it.
const BASELINE_VECTOR_WIDTH: u32 = 1;

/// Construct the unfused, unmapped baseline schedule for a validated graph.
#[must_use]
pub fn baseline_schedule(logical: &LogicalProgramGraph<'_>) -> SelectedSchedule {
    let logical_identity = *blake3::hash(logical.semantic_wire()).as_bytes();
    let phases = logical
        .regions()
        .iter()
        .enumerate()
        .map(|(index, region)| {
            let axes = region
                .extents
                .iter()
                .enumerate()
                .map(|(axis, extent)| ScheduleAxis {
                    region: region.node.0,
                    axis: u32::try_from(axis).unwrap_or(u32::MAX),
                    extent: match extent {
                        LogicalExtent::Static(value)
                        | LogicalExtent::GraphValue { bound: value, .. } => *value,
                    },
                })
                .collect();
            SchedulePhase {
                id: SchedulePhaseId(u32::try_from(index).unwrap_or(u32::MAX)),
                source_regions: vec![region.node.0],
                axes,
                grid: [region.max_points, 1, 1],
                workgroup: BASELINE_WORKGROUP,
                vector_width: BASELINE_VECTOR_WIDTH,
                mappings: Vec::new(),
                predecessors: region
                    .dependencies
                    .iter()
                    .map(|dependency| SchedulePhaseId(dependency.predecessor.0))
                    .collect(),
                resources: ScheduleResourceBounds {
                    logical_points: region.max_points,
                    ..ScheduleResourceBounds::default()
                },
            }
        })
        .collect::<Vec<_>>();
    let resources = phases
        .iter()
        .fold(ScheduleResourceBounds::default(), |total, phase| {
            ScheduleResourceBounds {
                logical_points: total
                    .logical_points
                    .saturating_add(phase.resources.logical_points),
                ..total
            }
        });
    SelectedSchedule {
        version: SCHEDULE_IR_VERSION,
        logical_identity,
        source_phases: phases.clone(),
        source_resources: resources,
        phases,
        transforms: Vec::new(),
        resources,
    }
}
