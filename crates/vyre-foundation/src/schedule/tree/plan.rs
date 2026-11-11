//! A schedule under construction and the plan it becomes.
//!
//! A partial schedule carries symbolic parameters that are not yet bound to
//! values. Kept apart from the tree itself so the op taxonomy does not change
//! when the way a schedule is assembled does.

use serde::{Deserialize, Serialize};

use super::ScheduleTree;
use crate::schedule::error::ScheduleLegalityError;
use crate::schedule::{
    SchedulePhaseId, ScheduleResourceBounds, ScheduleTransformRecord, SCHEDULE_IR_VERSION,
};

/// Symbolic parameter representing a bounded tunable parameter in partial schedules.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolicParameter {
    /// Parameter name.
    pub name: String,
    /// Minimum legal value.
    pub min_value: u64,
    /// Maximum legal value.
    pub max_value: u64,
    /// Optional default value.
    pub default_value: Option<u64>,
}

/// A partial schedule tree with unassigned regions and symbolic parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartialSchedule {
    /// Root schedule tree.
    pub root: ScheduleTree,
    /// Unassigned logical regions awaiting lowering/placement.
    pub unassigned_regions: Vec<u32>,
    /// Symbolic parameters pending instantiation.
    pub symbolic_parameters: Vec<SymbolicParameter>,
}

impl PartialSchedule {
    /// Create a new partial schedule.
    #[must_use]
    pub fn new(
        root: ScheduleTree,
        unassigned_regions: Vec<u32>,
        symbolic_parameters: Vec<SymbolicParameter>,
    ) -> Self {
        Self {
            root,
            unassigned_regions,
            symbolic_parameters,
        }
    }

    /// Return true if all regions and symbolic parameters have been assigned.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.unassigned_regions.is_empty() && self.symbolic_parameters.is_empty()
    }

    /// Instantiate symbolic parameters with concrete values to produce a
    /// complete validated [`SchedulePlan`] under the bounds the caller
    /// selected.
    ///
    /// The bounds arrive as an argument rather than as a default. Defaulting
    /// them here made the foundation choose the resource envelope a schedule
    /// runs under, which is a second selection route: the plan would validate
    /// against an envelope no selector had ranked.
    pub fn instantiate(
        &self,
        bindings: &std::collections::HashMap<String, u64>,
        bounds: ScheduleResourceBounds,
    ) -> Result<SchedulePlan, ScheduleLegalityError> {
        for param in &self.symbolic_parameters {
            if let Some(&val) = bindings.get(&param.name) {
                if val < param.min_value || val > param.max_value {
                    return Err(ScheduleLegalityError::ResourceOverflow(
                        "symbolic_parameter_out_of_bounds",
                    ));
                }
            } else if param.default_value.is_none() {
                return Err(ScheduleLegalityError::MissingPhase(SchedulePhaseId(0)));
            }
        }
        let plan = SchedulePlan::new(self.root.canonicalize(), bounds);
        plan.validate()?;
        Ok(plan)
    }
}

/// A complete validated schedule plan over logical regions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchedulePlan {
    /// Schema version.
    pub version: u16,
    /// Root schedule tree.
    pub root: ScheduleTree,
    /// Resource bounds checked for this plan.
    pub resource_bounds: ScheduleResourceBounds,
    /// Applied transformation records with provenance.
    pub history: Vec<ScheduleTransformRecord>,
}

impl SchedulePlan {
    /// Create a new schedule plan.
    #[must_use]
    pub fn new(root: ScheduleTree, resource_bounds: ScheduleResourceBounds) -> Self {
        Self {
            version: SCHEDULE_IR_VERSION,
            root,
            resource_bounds,
            history: Vec::new(),
        }
    }

    /// Add a transformation record to the history.
    pub fn record_transform(&mut self, record: ScheduleTransformRecord) {
        self.history.push(record);
    }

    /// Validate the entire schedule plan.
    pub fn validate(&self) -> Result<(), ScheduleLegalityError> {
        self.root.validate()?;
        if self.resource_bounds.shared_bytes > 64 * 1024 * 1024 {
            return Err(ScheduleLegalityError::ResourceOverflow("shared_bytes"));
        }
        Ok(())
    }
}
