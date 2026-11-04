//! The complete launch one dispatch runs.
//!
//! A compiled artifact records the launch its search selected, and a caller that
//! dispatches a program the compiler never saw states its own. Both arrive here
//! as one value, so a submission carries a whole launch or none: a shape with a
//! grid and no workgroup, or a workgroup a shared loop drops on the way to the
//! device, cannot be expressed.

use vyre_megakernel::GeometryRecord;

use crate::backend::{BackendError, DispatchConfig};

/// Workgroup shape, workgroup count, logical coverage and shared bytes of one
/// launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LaunchDirective {
    workgroup: [u32; 3],
    grid: [u32; 3],
    logical_coverage: [u64; 3],
    dynamic_shared_bytes: u32,
}

impl LaunchDirective {
    /// Read the launch `record` states for one admitted entry point.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] when an extent is zero. A zero
    /// extent records no launch, and defaulting it to one submits a shape the
    /// compiler never selected.
    pub fn from_record(record: &GeometryRecord, backend: &str) -> Result<Self, BackendError> {
        for (field, extents) in [
            ("workgroup_size", record.workgroup_size),
            ("grid", record.grid),
        ] {
            if extents.contains(&0) {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: backend `{backend}` received an artifact whose geometry record states {field} {extents:?}. \
                         Recompile the artifact with a compiler that records a launchable geometry for every node."
                    ),
                });
            }
        }
        if record.logical_coverage.contains(&0) {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: backend `{backend}` received an artifact whose geometry record covers {:?} logical points. \
                     Recompile the artifact with a compiler that records positive coverage for every node.",
                    record.logical_coverage
                ),
            });
        }
        Ok(Self {
            workgroup: record.workgroup_size,
            grid: record.grid,
            logical_coverage: record.logical_coverage,
            dynamic_shared_bytes: record.dynamic_shared_bytes,
        })
    }

    /// The launch a caller states for a program no artifact governs.
    ///
    /// Coverage is the whole launch, so a backend that infers its dispatch from
    /// buffer shapes runs every invocation the device would.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] when an extent is zero.
    pub fn stated(
        workgroup: [u32; 3],
        grid: [u32; 3],
        dynamic_shared_bytes: u32,
    ) -> Result<Self, BackendError> {
        if workgroup.contains(&0) || grid.contains(&0) {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: a stated launch has workgroup {workgroup:?} and grid {grid:?}; \
                     every extent must be positive."
                ),
            });
        }
        let mut logical_coverage = [0u64; 3];
        for (axis, points) in logical_coverage.iter_mut().enumerate() {
            *points = u64::from(workgroup[axis]) * u64::from(grid[axis]);
        }
        Ok(Self {
            workgroup,
            grid,
            logical_coverage,
            dynamic_shared_bytes,
        })
    }

    /// The launch a caller states for `program` over `grid` workgroups.
    ///
    /// The workgroup is the one `program` declares, which is the shape a caller
    /// that sized `grid` as `ceil(work / workgroup)` already assumed.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] when an extent is zero.
    pub fn stated_for(
        program: &vyre_foundation::ir::Program,
        grid: [u32; 3],
    ) -> Result<Self, BackendError> {
        Self::stated(program.workgroup_size(), grid, 0)
    }

    /// Workgroup dimensions the launch runs.
    #[must_use]
    pub const fn workgroup(&self) -> [u32; 3] {
        self.workgroup
    }

    /// Workgroup count the launch runs on each axis.
    #[must_use]
    pub const fn grid(&self) -> [u32; 3] {
        self.grid
    }

    /// Logical points the launch covers on each axis.
    #[must_use]
    pub const fn logical_coverage(&self) -> [u64; 3] {
        self.logical_coverage
    }

    /// Workgroup-shared bytes the launch reserves.
    #[must_use]
    pub const fn dynamic_shared_bytes(&self) -> u32 {
        self.dynamic_shared_bytes
    }

    /// The dispatch policy that submits this launch and states nothing else.
    #[must_use]
    pub fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig {
            launch: Some(*self),
            ..DispatchConfig::default()
        }
    }
}
