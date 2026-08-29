//! Backend-neutral frontier planning for dependency-aware megakernels.
//!
//! Backends can choose different execution topologies, but the memory envelope
//! of dependency-layered frontier waves is a backend-neutral contract. This
//! module plans that envelope once, including dependency barriers, fused-group
//! splitting under an explicit byte budget, peak byte accounting, and readback
//! pressure amortization, and then drives topology selection from it.
//!
//! Composing those two halves used to live in one concrete driver, which is why the
//! device-wide-barrier rule in [`crate::megakernel_execution`] could sit on one
//! backend without the neutral policy knowing it. The composition is decided
//! entirely by graph shape, wave bytes, and budgets, so it belongs here; a
//! backend supplies only its telemetry and, through
//! [`crate::megakernel_execution::MegakernelExecutionPlanner`], its own plan cache.

use crate::accounting::{
    checked_add_u64_count as checked_add, checked_mul_u64_count as checked_mul,
};
use crate::megakernel_barrier::{
    plan_megakernel_barriers_with_scratch, MegakernelBarrierGroup, MegakernelBarrierPlan,
    MegakernelBarrierPlanError, MegakernelBarrierScratch, MegakernelWaveDependency,
};
use crate::megakernel_execution::{
    megakernel_resident_graph_bytes, MegakernelByteLayout, MegakernelDeviceCapabilities,
    MegakernelExecutionPlan, MegakernelExecutionPlanner, MegakernelExecutionRequest,
    MegakernelExecutionSample, MegakernelGraphShape, MegakernelMemoryError,
};
use crate::reservation_policy::{
    reserve_typed_vec_to_capacity as reserve_vec_to_capacity, storage_reserve_failure_adapter,
    ReservationPolicy,
};

const MEGAKERNEL_FRONTIER_RESERVATION: ReservationPolicy = ReservationPolicy::new(
    "megakernel frontier memory planner",
    "shard the frontier wave group or split the fused phase",
);

/// Frontier-typed megakernel wave memory envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MegakernelFrontierWave {
    /// Resident frontier bytes touched by this wave.
    pub frontier_bytes: u64,
    /// Temporary scratch bytes required by this wave before topology scaling.
    pub scratch_bytes: u64,
    /// Output bytes produced by this wave.
    pub output_bytes: u64,
}

/// Dependency-aware megakernel frontier memory plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MegakernelFrontierMemoryPlan {
    /// Minimum global-barrier grouping after memory-budget splitting.
    pub barriers: MegakernelBarrierPlan,
    /// Peak frontier bytes across any fused barrier-free group.
    pub peak_frontier_bytes: u64,
    /// Peak scratch bytes across any fused barrier-free group.
    pub peak_scratch_bytes: u64,
    /// Peak output bytes across any fused barrier-free group.
    pub peak_output_bytes: u64,
    /// Readback pressure after combining runtime telemetry with static
    /// fused-wave output volume.
    pub amortized_readback_bytes: u64,
    /// Widest barrier-free group in wave count.
    pub max_group_width: usize,
}

/// Frontier memory planning failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MegakernelFrontierMemoryPlanError {
    /// Dependency graph cannot be barrier-planned.
    Barrier(MegakernelBarrierPlanError),
    /// Peak wave bytes overflowed while grouping a barrier-free phase.
    ByteCountOverflow {
        /// Field being accumulated.
        field: &'static str,
    },
    /// Static graph or fused frontier bytes exceed the caller-approved budget.
    GroupOverBudget {
        /// Required bytes before topology selection.
        required_bytes: u64,
        /// Caller-provided budget.
        budget_bytes: u64,
        /// Budget region being checked.
        field: &'static str,
    },
    /// Frontier planning result storage could not be reserved.
    StorageReserveFailed {
        /// Field being reserved.
        field: &'static str,
        /// Number of elements requested.
        requested: usize,
        /// Allocator error text.
        message: String,
    },
}

impl crate::accounting::ArithmeticOverflow for MegakernelFrontierMemoryPlanError {
    fn arithmetic_overflow(field: &'static str) -> Self {
        Self::ByteCountOverflow { field }
    }
}

impl std::fmt::Display for MegakernelFrontierMemoryPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Barrier(error) => error.fmt(f),
            Self::ByteCountOverflow { field } => write!(
                f,
                "megakernel frontier memory planner overflowed while accumulating {field}. Fix: shard the frontier wave group or split the fused phase."
            ),
            Self::GroupOverBudget {
                required_bytes,
                budget_bytes,
                field,
            } => write!(
                f,
                "megakernel frontier memory planner requires {required_bytes} bytes for {field} but budget allows {budget_bytes}. Fix: shard the graph/frontier waves or raise the explicit megakernel budget."
            ),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "megakernel frontier memory planner could not reserve {requested} {field} entries: {message}. Fix: shard the frontier waves before planning."
            ),
        }
    }
}

impl std::error::Error for MegakernelFrontierMemoryPlanError {}

impl From<MegakernelBarrierPlanError> for MegakernelFrontierMemoryPlanError {
    fn from(error: MegakernelBarrierPlanError) -> Self {
        Self::Barrier(error)
    }
}

/// Dependency-aware megakernel execution plan for frontier waves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MegakernelFrontierExecutionPlan {
    /// Topology and memory-budget plan for the peak barrier-free group.
    pub execution: MegakernelExecutionPlan,
    /// Minimum global-barrier grouping for the wave dependencies.
    pub barriers: MegakernelBarrierPlan,
    /// Peak frontier bytes across any fused barrier-free group.
    pub peak_frontier_bytes: u64,
    /// Peak scratch bytes across any fused barrier-free group.
    pub peak_scratch_bytes: u64,
    /// Peak output bytes across any fused barrier-free group.
    pub peak_output_bytes: u64,
    /// Readback pressure fed into topology selection after combining runtime
    /// telemetry with static fused-wave output volume.
    pub amortized_readback_bytes: u64,
    /// Widest barrier-free group in wave count.
    pub max_group_width: usize,
}

/// Dependency-aware frontier execution planning failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MegakernelFrontierExecutionPlanError {
    /// Dependency graph cannot be barrier-planned.
    Barrier(MegakernelBarrierPlanError),
    /// Peak wave bytes overflowed while grouping a barrier-free phase.
    ByteCountOverflow {
        /// Field being accumulated.
        field: &'static str,
    },
    /// Static graph or fused frontier bytes exceed the caller-approved budget.
    GroupOverBudget {
        /// Required bytes before topology selection.
        required_bytes: u64,
        /// Caller-provided budget.
        budget_bytes: u64,
        /// Budget region being checked.
        field: &'static str,
    },
    /// Topology-validated execution memory planning failed.
    Memory(MegakernelMemoryError),
    /// Frontier planning result storage could not be reserved.
    StorageReserveFailed {
        /// Field being reserved.
        field: &'static str,
        /// Number of elements requested.
        requested: usize,
        /// Allocator error text.
        message: String,
    },
}

impl crate::accounting::ArithmeticOverflow for MegakernelFrontierExecutionPlanError {
    fn arithmetic_overflow(field: &'static str) -> Self {
        Self::ByteCountOverflow { field }
    }
}

impl std::fmt::Display for MegakernelFrontierExecutionPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Barrier(error) => error.fmt(f),
            Self::ByteCountOverflow { field } => write!(
                f,
                "megakernel frontier execution planner overflowed while accumulating {field}. Fix: shard the frontier wave group or split the fused phase."
            ),
            Self::GroupOverBudget {
                required_bytes,
                budget_bytes,
                field,
            } => write!(
                f,
                "megakernel frontier execution planner requires {required_bytes} bytes for {field} but budget allows {budget_bytes}. Fix: shard the graph/frontier waves or raise the explicit megakernel budget."
            ),
            Self::Memory(error) => error.fmt(f),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "megakernel frontier execution planner could not reserve {requested} {field} entries: {message}. Fix: shard the frontier waves before planning."
            ),
        }
    }
}

impl std::error::Error for MegakernelFrontierExecutionPlanError {}

impl From<MegakernelBarrierPlanError> for MegakernelFrontierExecutionPlanError {
    fn from(error: MegakernelBarrierPlanError) -> Self {
        Self::Barrier(error)
    }
}

impl From<MegakernelMemoryError> for MegakernelFrontierExecutionPlanError {
    fn from(error: MegakernelMemoryError) -> Self {
        Self::Memory(error)
    }
}

impl From<MegakernelFrontierMemoryPlanError> for MegakernelFrontierExecutionPlanError {
    fn from(error: MegakernelFrontierMemoryPlanError) -> Self {
        match error {
            MegakernelFrontierMemoryPlanError::Barrier(error) => Self::Barrier(error),
            MegakernelFrontierMemoryPlanError::ByteCountOverflow { field } => {
                Self::ByteCountOverflow { field }
            }
            MegakernelFrontierMemoryPlanError::GroupOverBudget {
                required_bytes,
                budget_bytes,
                field,
            } => Self::GroupOverBudget {
                required_bytes,
                budget_bytes,
                field,
            },
            MegakernelFrontierMemoryPlanError::StorageReserveFailed {
                field,
                requested,
                message,
            } => Self::StorageReserveFailed {
                field,
                requested,
                message,
            },
        }
    }
}

/// Plan dependency-aware megakernel execution for frontier-typed waves.
///
/// The planner minimizes global barriers from the wave dependencies, computes
/// the peak memory envelope of any barrier-free fused group, and asks `planner`
/// for a memory-validated topology for that envelope.
///
/// # Errors
///
/// Returns [`MegakernelFrontierExecutionPlanError`] when dependencies are
/// invalid, counters overflow, storage cannot be reserved, or the envelope does
/// not fit the explicit budget.
pub fn plan_megakernel_frontier_execution(
    planner: &mut impl MegakernelExecutionPlanner,
    sample: MegakernelExecutionSample,
    graph: MegakernelGraphShape,
    bytes_per_node: u64,
    bytes_per_edge: u64,
    waves: &[MegakernelFrontierWave],
    dependencies: &[MegakernelWaveDependency],
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    capabilities: MegakernelDeviceCapabilities,
) -> Result<MegakernelFrontierExecutionPlan, MegakernelFrontierExecutionPlanError> {
    let mut scratch = MegakernelBarrierScratch::try_with_capacity(waves.len(), dependencies.len())?;
    plan_megakernel_frontier_execution_with_scratch(
        planner,
        sample,
        graph,
        bytes_per_node,
        bytes_per_edge,
        waves,
        dependencies,
        budget_bytes,
        launch_overhead_ns,
        fusion_pressure,
        capabilities,
        &mut scratch,
    )
}

/// Plan dependency-aware megakernel execution using caller-owned scratch.
///
/// # Errors
///
/// Same rejections as [`plan_megakernel_frontier_execution`].
#[allow(clippy::too_many_arguments)]
pub fn plan_megakernel_frontier_execution_with_scratch(
    planner: &mut impl MegakernelExecutionPlanner,
    sample: MegakernelExecutionSample,
    graph: MegakernelGraphShape,
    bytes_per_node: u64,
    bytes_per_edge: u64,
    waves: &[MegakernelFrontierWave],
    dependencies: &[MegakernelWaveDependency],
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    capabilities: MegakernelDeviceCapabilities,
    scratch: &mut MegakernelBarrierScratch,
) -> Result<MegakernelFrontierExecutionPlan, MegakernelFrontierExecutionPlanError> {
    let graph_bytes = megakernel_resident_graph_bytes(graph, bytes_per_node, bytes_per_edge)?;
    let memory = plan_megakernel_frontier_memory_with_scratch(
        waves,
        dependencies,
        graph_bytes,
        budget_bytes,
        sample.readback_bytes,
        scratch,
    )?;
    let execution = planner.plan_execution(MegakernelExecutionRequest {
        sample: MegakernelExecutionSample {
            readback_bytes: memory.amortized_readback_bytes,
            ..sample
        },
        graph,
        bytes: MegakernelByteLayout {
            bytes_per_node,
            bytes_per_edge,
            frontier_bytes: memory.peak_frontier_bytes,
            scratch_bytes: memory.peak_scratch_bytes,
            output_bytes: memory.peak_output_bytes,
            budget_bytes,
        },
        launch_overhead_ns,
        fusion_pressure: capabilities.admissible_fusion_pressure(fusion_pressure),
        capabilities,
    })?;
    Ok(MegakernelFrontierExecutionPlan {
        execution,
        barriers: memory.barriers,
        peak_frontier_bytes: memory.peak_frontier_bytes,
        peak_scratch_bytes: memory.peak_scratch_bytes,
        peak_output_bytes: memory.peak_output_bytes,
        amortized_readback_bytes: memory.amortized_readback_bytes,
        max_group_width: memory.max_group_width,
    })
}

/// Plan dependency-aware frontier memory using caller-owned barrier scratch.
///
/// # Errors
///
/// Returns [`MegakernelFrontierMemoryPlanError`] when dependencies are invalid,
/// counters overflow, or the requested graph/frontier envelope cannot fit the
/// explicit budget.
pub fn plan_megakernel_frontier_memory_with_scratch(
    waves: &[MegakernelFrontierWave],
    dependencies: &[MegakernelWaveDependency],
    resident_graph_bytes: u64,
    budget_bytes: u64,
    readback_bytes: u64,
    scratch: &mut MegakernelBarrierScratch,
) -> Result<MegakernelFrontierMemoryPlan, MegakernelFrontierMemoryPlanError> {
    let barriers = plan_megakernel_barriers_with_scratch(waves.len(), dependencies, scratch)?;
    let group_budget_bytes = budget_bytes.checked_sub(resident_graph_bytes).ok_or(
        MegakernelFrontierMemoryPlanError::GroupOverBudget {
            required_bytes: resident_graph_bytes,
            budget_bytes,
            field: "resident graph bytes",
        },
    )?;
    let barriers = split_barrier_groups_to_memory_budget(barriers, waves, group_budget_bytes)?;
    let mut peak_frontier_bytes = 0u64;
    let mut peak_scratch_bytes = 0u64;
    let mut peak_output_bytes = 0u64;
    let mut max_group_width = 0usize;
    for group in &barriers.groups {
        let mut group_frontier_bytes = 0u64;
        let mut group_scratch_bytes = 0u64;
        let mut group_output_bytes = 0u64;
        max_group_width = max_group_width.max(group.waves.len());
        for &wave_index in &group.waves {
            let wave = waves[wave_index];
            group_frontier_bytes = checked_add::<MegakernelFrontierMemoryPlanError>(
                group_frontier_bytes,
                wave.frontier_bytes,
                "frontier wave bytes",
            )?;
            group_scratch_bytes = checked_add::<MegakernelFrontierMemoryPlanError>(
                group_scratch_bytes,
                wave.scratch_bytes,
                "scratch wave bytes",
            )?;
            group_output_bytes = checked_add::<MegakernelFrontierMemoryPlanError>(
                group_output_bytes,
                wave.output_bytes,
                "output wave bytes",
            )?;
        }
        peak_frontier_bytes = peak_frontier_bytes.max(group_frontier_bytes);
        peak_scratch_bytes = peak_scratch_bytes.max(group_scratch_bytes);
        peak_output_bytes = peak_output_bytes.max(group_output_bytes);
    }

    Ok(MegakernelFrontierMemoryPlan {
        barriers,
        peak_frontier_bytes,
        peak_scratch_bytes,
        peak_output_bytes,
        amortized_readback_bytes: readback_bytes.max(peak_output_bytes),
        max_group_width,
    })
}

fn split_barrier_groups_to_memory_budget(
    barriers: MegakernelBarrierPlan,
    waves: &[MegakernelFrontierWave],
    group_budget_bytes: u64,
) -> Result<MegakernelBarrierPlan, MegakernelFrontierMemoryPlanError> {
    let mut groups = Vec::new();
    reserve_vec::<MegakernelBarrierGroup>(
        &mut groups,
        barriers.groups.len(),
        "split barrier groups",
    )?;
    for group in barriers.groups {
        split_one_barrier_group_to_memory_budget(group, waves, group_budget_bytes, &mut groups)?;
    }
    Ok(MegakernelBarrierPlan::from_groups(groups))
}

fn split_one_barrier_group_to_memory_budget(
    group: MegakernelBarrierGroup,
    waves: &[MegakernelFrontierWave],
    group_budget_bytes: u64,
    groups: &mut Vec<MegakernelBarrierGroup>,
) -> Result<(), MegakernelFrontierMemoryPlanError> {
    let mut current = Vec::new();
    reserve_vec::<usize>(
        &mut current,
        group.waves.len().min(8),
        "current split barrier group",
    )?;
    let mut current_bytes = 0u64;
    for wave_index in group.waves {
        let wave_bytes = megakernel_frontier_fused_wave_budget_bytes(waves[wave_index])?;
        let combined = checked_add::<MegakernelFrontierMemoryPlanError>(
            current_bytes,
            wave_bytes,
            "barrier group fused wave budget bytes",
        )?;
        if current.is_empty() && wave_bytes > group_budget_bytes {
            return Err(MegakernelFrontierMemoryPlanError::GroupOverBudget {
                required_bytes: wave_bytes,
                budget_bytes: group_budget_bytes,
                field: "single fused frontier wave bytes",
            });
        }
        if !current.is_empty() && combined > group_budget_bytes {
            groups.push(MegakernelBarrierGroup {
                waves: std::mem::take(&mut current),
            });
            current_bytes = 0;
        }
        current.push(wave_index);
        current_bytes = checked_add::<MegakernelFrontierMemoryPlanError>(
            current_bytes,
            wave_bytes,
            "barrier group fused wave budget bytes",
        )?;
    }
    if !current.is_empty() {
        groups.push(MegakernelBarrierGroup { waves: current });
    }
    Ok(())
}

/// Compute the byte budget used to decide whether one frontier wave can fit in
/// a fused barrier-free resident group.
pub fn megakernel_frontier_fused_wave_budget_bytes(
    wave: MegakernelFrontierWave,
) -> Result<u64, MegakernelFrontierMemoryPlanError> {
    let fused_scratch_bytes = checked_mul::<MegakernelFrontierMemoryPlanError>(
        wave.scratch_bytes,
        4,
        "fused wave scratch bytes",
    )?;
    let bytes = checked_add::<MegakernelFrontierMemoryPlanError>(
        wave.frontier_bytes,
        fused_scratch_bytes,
        "fused wave bytes",
    )?;
    checked_add::<MegakernelFrontierMemoryPlanError>(bytes, wave.output_bytes, "fused wave bytes")
}

fn reserve_vec<T>(
    vec: &mut Vec<T>,
    target_capacity: usize,
    item: &'static str,
) -> Result<(), MegakernelFrontierMemoryPlanError> {
    reserve_vec_to_capacity(
        MEGAKERNEL_FRONTIER_RESERVATION,
        vec,
        target_capacity,
        item,
        storage_reserve_failed,
    )
}

storage_reserve_failure_adapter!(MegakernelFrontierMemoryPlanError);

// Inline: covers `from`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::{
        megakernel_frontier_fused_wave_budget_bytes, plan_megakernel_frontier_execution,
        plan_megakernel_frontier_memory_with_scratch, MegakernelFrontierExecutionPlanError,
        MegakernelFrontierMemoryPlanError, MegakernelFrontierWave,
    };
    use crate::megakernel_barrier::{MegakernelBarrierScratch, MegakernelWaveDependency};
    use crate::megakernel_execution::{
        FrontierTopology, MegakernelDeviceCapabilities, MegakernelExecutionSample,
        MegakernelGraphShape, NeutralMegakernelExecutionPlanner,
    };
    use crate::megakernel_fixtures::{
        layered_dag_dependencies, DIAMOND_DEPENDENCIES, DIAMOND_WAVES, GROWING_PAIR_WAVES,
        OUTPUT_HEAVY_WAVES, OVERFLOW_WAVES, THREE_SMALL_WAVES,
    };

    #[test]
    fn frontier_memory_plan_uses_peak_barrier_group_memory() {
        let mut scratch = MegakernelBarrierScratch::default();
        let plan = plan_megakernel_frontier_memory_with_scratch(
            DIAMOND_WAVES,
            DIAMOND_DEPENDENCIES,
            16_000,
            128 * 1024,
            1 << 20,
            &mut scratch,
        )
        .expect("Fix: frontier-typed megakernel memory plan should fit the budget.");

        assert_eq!(plan.barriers.global_barriers, 2);
        assert_eq!(plan.barriers.groups[1].waves, vec![1, 2]);
        assert_eq!(plan.peak_frontier_bytes, 8_192);
        assert_eq!(plan.peak_scratch_bytes, 4_096);
        assert_eq!(plan.peak_output_bytes, 2_048);
        assert_eq!(plan.amortized_readback_bytes, 1 << 20);
        assert_eq!(plan.max_group_width, 2);
    }

    #[test]
    fn frontier_memory_uses_static_group_output_to_amortize_readback() {
        let mut scratch = MegakernelBarrierScratch::default();
        let plan = plan_megakernel_frontier_memory_with_scratch(
            OUTPUT_HEAVY_WAVES,
            &[],
            16_000,
            128 * 1024,
            0,
            &mut scratch,
        )
        .expect("Fix: static output-amortized frontier memory plan should fit the budget.");

        assert_eq!(plan.peak_output_bytes, 6_144);
        assert_eq!(plan.amortized_readback_bytes, 6_144);
    }

    #[test]
    fn frontier_memory_splits_independent_layers_to_fit_fused_budget() {
        let mut scratch = MegakernelBarrierScratch::default();
        let waves = THREE_SMALL_WAVES;
        let plan =
            plan_megakernel_frontier_memory_with_scratch(waves, &[], 0, 100, 4_096, &mut scratch)
                .expect("Fix: independent frontier waves should split into budget-fit chunks.");

        assert_eq!(plan.barriers.groups.len(), 3);
        assert_eq!(plan.barriers.global_barriers, 2);
        assert_eq!(plan.max_group_width, 1);
        assert_eq!(plan.peak_frontier_bytes, 10);
        assert_eq!(plan.peak_scratch_bytes, 10);
        assert_eq!(plan.peak_output_bytes, 10);
    }

    #[test]
    fn frontier_memory_rejects_graph_and_single_wave_over_budget() {
        let mut scratch = MegakernelBarrierScratch::default();
        let graph_error = plan_megakernel_frontier_memory_with_scratch(
            &[MegakernelFrontierWave {
                frontier_bytes: 1,
                scratch_bytes: 1,
                output_bytes: 1,
            }],
            &[],
            1_600,
            1_000,
            0,
            &mut scratch,
        )
        .expect_err("resident graph bytes above budget must fail before split planning");
        assert_eq!(
            graph_error,
            MegakernelFrontierMemoryPlanError::GroupOverBudget {
                required_bytes: 1_600,
                budget_bytes: 1_000,
                field: "resident graph bytes",
            }
        );

        let wave_error = plan_megakernel_frontier_memory_with_scratch(
            &[MegakernelFrontierWave {
                frontier_bytes: 100,
                scratch_bytes: 100,
                output_bytes: 100,
            }],
            &[],
            0,
            500,
            0,
            &mut scratch,
        )
        .expect_err("single fused wave above group budget must fail before topology planning");
        assert_eq!(
            wave_error,
            MegakernelFrontierMemoryPlanError::GroupOverBudget {
                required_bytes: 600,
                budget_bytes: 500,
                field: "single fused frontier wave bytes",
            }
        );
    }

    #[test]
    fn frontier_fused_wave_budget_uses_topology_scratch_multiplier() {
        assert_eq!(
            megakernel_frontier_fused_wave_budget_bytes(MegakernelFrontierWave {
                frontier_bytes: 16,
                scratch_bytes: 16,
                output_bytes: 16,
            })
            .expect("Fix: fused frontier wave budget should fit"),
            96
        );
    }

    #[test]
    fn frontier_memory_fails_loudly_on_wave_byte_overflow() {
        let mut scratch = MegakernelBarrierScratch::default();
        let error = plan_megakernel_frontier_memory_with_scratch(
            OVERFLOW_WAVES,
            &[],
            2,
            u64::MAX,
            0,
            &mut scratch,
        )
        .expect_err("Fix: overflowed frontier wave bytes must fail before launch planning.");

        assert_eq!(
            error,
            MegakernelFrontierMemoryPlanError::ByteCountOverflow {
                field: "fused wave bytes"
            }
        );
    }

    #[test]
    fn generated_frontier_memory_profiles_preserve_peak_and_budget_for_1024_shapes() {
        let mut scratch = MegakernelBarrierScratch::default();
        for width in 1u64..=32 {
            for depth in 1u64..=32 {
                let dependencies = layered_dag_dependencies(width as usize, depth as usize);
                let mut waves = Vec::new();
                for layer in 0..depth {
                    for slot in 0..width {
                        waves.push(MegakernelFrontierWave {
                            frontier_bytes: width,
                            scratch_bytes: slot + 1,
                            output_bytes: layer + 1,
                        });
                    }
                }

                let plan = plan_megakernel_frontier_memory_with_scratch(
                    &waves,
                    &dependencies,
                    256,
                    u64::MAX / 2,
                    7,
                    &mut scratch,
                )
                .expect("Fix: generated frontier memory DAG should plan under large budget.");

                assert_eq!(plan.barriers.groups.len(), depth as usize);
                assert_eq!(plan.max_group_width, width as usize);
                assert_eq!(plan.peak_frontier_bytes, width * width);
                assert_eq!(plan.peak_scratch_bytes, width * (width + 1) / 2);
                assert_eq!(plan.peak_output_bytes, width * depth);
                assert_eq!(plan.amortized_readback_bytes, 7.max(width * depth));
            }
        }
    }

    fn fused_pressure_plan(
        capabilities: MegakernelDeviceCapabilities,
    ) -> Result<super::MegakernelFrontierExecutionPlan, MegakernelFrontierExecutionPlanError> {
        plan_megakernel_frontier_execution(
            &mut NeutralMegakernelExecutionPlanner,
            MegakernelExecutionSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.50,
                readback_bytes: 1 << 20,
            },
            MegakernelGraphShape {
                node_count: 1_000,
                edge_count: 4_000,
            },
            16,
            8,
            GROWING_PAIR_WAVES,
            &[MegakernelWaveDependency {
                before: 0,
                after: 1,
            }],
            128 * 1024,
            250.0,
            0.95,
            capabilities,
        )
    }

    #[test]
    fn frontier_execution_plans_barriers_and_topology_from_one_envelope() {
        let plan = fused_pressure_plan(MegakernelDeviceCapabilities::FUSION_CAPABLE)
            .expect("Fix: dependency-layered frontier waves should fit the budget.");

        assert_eq!(plan.barriers.global_barriers, 1);
        assert_eq!(plan.barriers.groups[0].waves, vec![0]);
        assert_eq!(plan.barriers.groups[1].waves, vec![1]);
        assert_eq!(plan.peak_frontier_bytes, 2_048);
        assert_eq!(plan.peak_scratch_bytes, 1_024);
        assert_eq!(plan.peak_output_bytes, 512);
        assert_eq!(plan.amortized_readback_bytes, 1 << 20);
        assert_eq!(plan.max_group_width, 1);
        assert_eq!(
            plan.execution.topology,
            FrontierTopology::FusedWave
        );
        assert_eq!(plan.execution.memory.frontier_bytes, 2_048);
        assert_eq!(plan.execution.memory.scratch_bytes, 4_096);
    }

    #[test]
    fn frontier_execution_refuses_a_fused_wave_without_a_device_wide_barrier() {
        let capable = fused_pressure_plan(MegakernelDeviceCapabilities::FUSION_CAPABLE)
            .expect("Fix: capable device should plan.");
        let incapable = fused_pressure_plan(MegakernelDeviceCapabilities::FUSION_INCAPABLE)
            .expect("Fix: a device without a device-wide barrier still gets a plan.");

        assert_eq!(
            capable.barriers, incapable.barriers,
            "Fix: device capability changes the topology, never the dependency grouping."
        );
        assert_ne!(
            incapable.execution.topology,
            FrontierTopology::FusedWave,
            "Fix: a fused wave crosses wave boundaries inside one launch and needs a barrier \
             across every resident block; a device without one cannot run the plan."
        );
        assert!(
            incapable.execution.memory.required_bytes < capable.execution.memory.required_bytes,
            "Fix: refusing fusion must also drop the fused scratch multiplier from the envelope."
        );
    }

    #[test]
    fn frontier_execution_rejects_a_graph_that_leaves_no_wave_headroom() {
        let error = plan_megakernel_frontier_execution(
            &mut NeutralMegakernelExecutionPlanner,
            MegakernelExecutionSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.50,
                readback_bytes: 4_096,
            },
            MegakernelGraphShape {
                node_count: 100,
                edge_count: 100,
            },
            8,
            8,
            &[MegakernelFrontierWave {
                frontier_bytes: 1,
                scratch_bytes: 1,
                output_bytes: 1,
            }],
            &[],
            1_000,
            250.0,
            0.95,
            MegakernelDeviceCapabilities::FUSION_CAPABLE,
        )
        .expect_err("Fix: resident graph bytes above budget must fail before split planning.");

        assert_eq!(
            error,
            MegakernelFrontierExecutionPlanError::GroupOverBudget {
                required_bytes: 1_600,
                budget_bytes: 1_000,
                field: "resident graph bytes",
            }
        );
    }

    #[test]
    fn frontier_execution_fails_loudly_on_wave_byte_overflow() {
        let error = plan_megakernel_frontier_execution(
            &mut NeutralMegakernelExecutionPlanner,
            MegakernelExecutionSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.90,
                readback_bytes: 1 << 20,
            },
            MegakernelGraphShape {
                node_count: 1,
                edge_count: 1,
            },
            1,
            1,
            OVERFLOW_WAVES,
            &[],
            u64::MAX,
            250.0,
            0.95,
            MegakernelDeviceCapabilities::FUSION_CAPABLE,
        )
        .expect_err("Fix: overflowed frontier wave bytes must fail before launch planning.");

        assert_eq!(
            error,
            MegakernelFrontierExecutionPlanError::ByteCountOverflow {
                field: "fused wave bytes"
            }
        );
    }
}
