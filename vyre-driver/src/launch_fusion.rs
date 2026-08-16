//! Backend-neutral adjacent-stage launch fusion planning.
//!
//! Backends that dispatch adjacent stages with compatible memory layouts can
//! fuse them into fewer launches when the fused memory envelope fits an
//! explicit budget. This module owns the pure planning algorithm so no backend
//! carries divergent launch-fusion logic.

use rustc_hash::FxHashSet;

use crate::reservation_policy::ReservationPolicy;

const LAUNCH_FUSION_RESERVATION: ReservationPolicy = ReservationPolicy::new(
    "adjacent launch fusion",
    "shard adjacent stages before fusion planning",
);

/// One adjacent backend stage considered for launch fusion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchFusionStage {
    /// Stable stage id.
    pub id: u32,
    /// Memory-layout compatibility hash.
    pub layout_hash: u64,
    /// Input bytes consumed by this stage.
    pub input_bytes: u64,
    /// Output bytes produced by this stage.
    pub output_bytes: u64,
    /// Scratch bytes required by this stage.
    pub scratch_bytes: u64,
    /// Whether this stage boundary requires host-visible materialization.
    pub requires_host_materialization: bool,
}

/// One fused adjacent-stage launch group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchFusionGroup {
    /// Stage ids included in the fused group.
    pub stage_ids: Vec<u32>,
    /// Shared layout hash for the group.
    pub layout_hash: u64,
    /// Peak bytes required by the fused group.
    pub required_bytes: u64,
    /// Host-visible intermediate bytes avoided by fusion.
    pub avoided_intermediate_bytes: u64,
}

/// Complete adjacent-stage launch fusion plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchFusionPlan {
    /// Fused or singleton groups in original stage order.
    pub groups: Vec<LaunchFusionGroup>,
    /// Number of backend launches after fusion.
    pub launch_count: u32,
    /// Number of launches removed by fusion.
    pub avoided_launches: u32,
    /// Total host-visible intermediate bytes avoided.
    pub avoided_intermediate_bytes: u64,
}

/// Caller-owned scratch for repeated launch-fusion planning.
#[derive(Debug, Default)]
pub struct LaunchFusionScratch {
    ids: FxHashSet<u32>,
}

impl LaunchFusionScratch {
    /// Create empty reusable launch-fusion scratch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ids: FxHashSet::default(),
        }
    }

    /// Allocate reusable launch-fusion scratch for a known stage count.
    ///
    /// # Errors
    ///
    /// Returns [`LaunchFusionError`] when duplicate-detection storage cannot
    /// be reserved.
    pub fn try_with_capacity(stage_count: usize) -> Result<Self, LaunchFusionError> {
        let mut scratch = Self::new();
        scratch.try_reserve_ids(stage_count)?;
        Ok(scratch)
    }

    fn try_reserve_ids(&mut self, stage_count: usize) -> Result<(), LaunchFusionError> {
        LAUNCH_FUSION_RESERVATION
            .reserve_hash_set_to_capacity(&mut self.ids, stage_count, "duplicate stage ids")
            .map_err(|error| LaunchFusionError::StorageReserveFailed {
                field: "duplicate stage ids",
                requested: stage_count,
                message: error.to_string(),
            })
    }

    /// Retained duplicate-detection capacity.
    #[must_use]
    pub fn id_capacity(&self) -> usize {
        self.ids.capacity()
    }
}

/// Launch fusion planning errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LaunchFusionError {
    /// Duplicate stage id.
    DuplicateStage {
        /// Duplicate id.
        id: u32,
    },
    /// Explicit fusion budget cannot be zero.
    ZeroBudget,
    /// Byte arithmetic overflowed.
    ByteCountOverflow {
        /// Field being computed.
        field: &'static str,
    },
    /// One stage cannot fit the explicit fusion budget even without fusion.
    StageOverBudget {
        /// Stage id.
        id: u32,
        /// Required bytes for the singleton stage.
        required_bytes: u64,
        /// Caller-provided budget.
        budget_bytes: u64,
    },
    /// Planner storage could not be reserved.
    StorageReserveFailed {
        /// Field being reserved.
        field: &'static str,
        /// Number of entries requested.
        requested: usize,
        /// Allocator error text.
        message: String,
    },
}

impl std::fmt::Display for LaunchFusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStage { id } => write!(
                f,
                "Launch fusion received duplicate stage id {id}. Fix: emit unique stage ids before fusion planning."
            ),
            Self::ZeroBudget => write!(
                f,
                "Launch fusion received a zero byte budget. Fix: pass an explicit device-memory budget before planning fusion."
            ),
            Self::ByteCountOverflow { field } => write!(
                f,
                "Launch fusion overflowed while computing {field}. Fix: shard adjacent stages before launch fusion planning."
            ),
            Self::StageOverBudget {
                id,
                required_bytes,
                budget_bytes,
            } => write!(
                f,
                "Launch fusion stage {id} requires {required_bytes} bytes but budget allows {budget_bytes}. Fix: shard the stage or raise the explicit fusion budget."
            ),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "Launch fusion could not reserve {requested} {field} entries: {message}. Fix: shard adjacent stages before fusion planning."
            ),
        }
    }
}

impl std::error::Error for LaunchFusionError {}

/// Plan adjacent launch fusion under layout and memory constraints.
///
/// # Errors
///
/// Returns [`LaunchFusionError`] when inputs are invalid, byte arithmetic
/// overflows, staging allocation fails, or any singleton stage exceeds the
/// explicit budget.
pub fn plan_launch_fusion(
    stages: &[LaunchFusionStage],
    max_group_bytes: u64,
) -> Result<LaunchFusionPlan, LaunchFusionError> {
    let mut scratch = LaunchFusionScratch::try_with_capacity(stages.len())?;
    plan_launch_fusion_with_scratch(stages, max_group_bytes, &mut scratch)
}

/// Plan adjacent launch fusion using caller-owned temporary storage.
///
/// # Errors
///
/// Returns [`LaunchFusionError`] when inputs are invalid, byte arithmetic
/// overflows, staging allocation fails, or any singleton stage exceeds the
/// explicit budget.
pub fn plan_launch_fusion_with_scratch(
    stages: &[LaunchFusionStage],
    max_group_bytes: u64,
    scratch: &mut LaunchFusionScratch,
) -> Result<LaunchFusionPlan, LaunchFusionError> {
    if max_group_bytes == 0 {
        return Err(LaunchFusionError::ZeroBudget);
    }
    if stages.is_empty() {
        return Ok(LaunchFusionPlan {
            groups: Vec::new(),
            launch_count: 0,
            avoided_launches: 0,
            avoided_intermediate_bytes: 0,
        });
    }
    if stages.len() == 1 {
        let group = singleton_group_with_capacity(stages[0], 1)?;
        if group.required_bytes > max_group_bytes {
            return Err(LaunchFusionError::StageOverBudget {
                id: stages[0].id,
                required_bytes: group.required_bytes,
                budget_bytes: max_group_bytes,
            });
        }
        let mut groups = reserved_vec(1, "fusion groups")?;
        groups.push(group);
        return Ok(LaunchFusionPlan {
            groups,
            launch_count: 1,
            avoided_launches: 0,
            avoided_intermediate_bytes: 0,
        });
    }

    scratch.ids.clear();
    if stages.len() <= 8 {
        for i in 0..stages.len() {
            let current = stages[i].id;
            if stages[..i].iter().any(|prev| prev.id == current) {
                return Err(LaunchFusionError::DuplicateStage { id: current });
            }
        }
    } else {
        scratch.try_reserve_ids(stages.len())?;
        for stage in stages {
            if !scratch.ids.insert(stage.id) {
                return Err(LaunchFusionError::DuplicateStage { id: stage.id });
            }
        }
    }

    let mut groups = reserved_vec(stages.len(), "fusion groups")?;
    let mut index = 0;
    while index < stages.len() {
        let remaining_stage_count = stages.len() - index;
        let mut group = singleton_group_with_capacity(stages[index], remaining_stage_count)?;
        if group.required_bytes > max_group_bytes {
            return Err(LaunchFusionError::StageOverBudget {
                id: stages[index].id,
                required_bytes: group.required_bytes,
                budget_bytes: max_group_bytes,
            });
        }
        let mut cursor = index + 1;
        while cursor < stages.len() && can_append_to_group(&group, stages[cursor], max_group_bytes)?
        {
            let previous_output = stages[cursor - 1].output_bytes;
            group.required_bytes = fused_required_bytes(&group, stages[cursor])?;
            group.avoided_intermediate_bytes = checked_add_u64(
                group.avoided_intermediate_bytes,
                previous_output,
                "avoided intermediate bytes",
            )?;
            group.stage_ids.push(stages[cursor].id);
            cursor += 1;
        }
        groups.push(group);
        index = cursor;
    }

    let launch_count =
        u32::try_from(groups.len()).map_err(|_| LaunchFusionError::ByteCountOverflow {
            field: "launch count",
        })?;
    let avoided_launches = u32::try_from(stages.len() - groups.len()).map_err(|_| {
        LaunchFusionError::ByteCountOverflow {
            field: "avoided launches",
        }
    })?;
    let mut avoided_intermediate_bytes = 0_u64;
    for group in &groups {
        avoided_intermediate_bytes = checked_add_u64(
            avoided_intermediate_bytes,
            group.avoided_intermediate_bytes,
            "total avoided intermediate bytes",
        )?;
    }

    Ok(LaunchFusionPlan {
        groups,
        launch_count,
        avoided_launches,
        avoided_intermediate_bytes,
    })
}

fn reserved_vec<T>(capacity: usize, field: &'static str) -> Result<Vec<T>, LaunchFusionError> {
    LAUNCH_FUSION_RESERVATION
        .reserved_vec(capacity, field)
        .map_err(|error| LaunchFusionError::StorageReserveFailed {
            field,
            requested: capacity,
            message: error.to_string(),
        })
}

fn singleton_group_with_capacity(
    stage: LaunchFusionStage,
    stage_id_capacity: usize,
) -> Result<LaunchFusionGroup, LaunchFusionError> {
    let mut stage_ids = reserved_vec(stage_id_capacity.max(1), "fusion group stage ids")?;
    stage_ids.push(stage.id);
    Ok(LaunchFusionGroup {
        stage_ids,
        layout_hash: stage.layout_hash,
        required_bytes: stage_required_bytes(stage)?,
        avoided_intermediate_bytes: 0,
    })
}

fn can_append_to_group(
    group: &LaunchFusionGroup,
    stage: LaunchFusionStage,
    max_group_bytes: u64,
) -> Result<bool, LaunchFusionError> {
    if stage.requires_host_materialization || stage.layout_hash != group.layout_hash {
        return Ok(false);
    }
    Ok(fused_required_bytes(group, stage)? <= max_group_bytes)
}

fn fused_required_bytes(
    group: &LaunchFusionGroup,
    stage: LaunchFusionStage,
) -> Result<u64, LaunchFusionError> {
    checked_add_u64(
        group.required_bytes,
        stage.scratch_bytes,
        "fused scratch bytes",
    )
    .and_then(|bytes| checked_add_u64(bytes, stage.output_bytes, "fused output bytes"))
}

fn stage_required_bytes(stage: LaunchFusionStage) -> Result<u64, LaunchFusionError> {
    let input_plus_output =
        checked_add_u64(stage.input_bytes, stage.output_bytes, "stage io bytes")?;
    checked_add_u64(
        input_plus_output,
        stage.scratch_bytes,
        "stage required bytes",
    )
}

fn checked_add_u64(left: u64, right: u64, field: &'static str) -> Result<u64, LaunchFusionError> {
    left.checked_add(right)
        .ok_or(LaunchFusionError::ByteCountOverflow { field })
}
