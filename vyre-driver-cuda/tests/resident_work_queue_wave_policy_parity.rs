//! Cross-entry-point equality guard for the megakernel wave policy family.
//!
//! Barrier placement, wave/topology selection, and frontier scheduling are one
//! policy. `vyre-driver` owns it; `vyre-driver-cuda` is only allowed to map
//! CUDA telemetry onto it. This suite drives both entry points with the same
//! inputs and asserts they agree decision for decision, so a CUDA-local fork of
//! any of the three decisions fails here instead of silently diverging on one
//! backend.

#![cfg(feature = "device-tests")]

use vyre_driver::megakernel_barrier::MegakernelWaveDependency;
use vyre_driver::megakernel_execution::{
    plan_megakernel_execution, select_frontier_topology, select_frontier_topology_stable,
    FrontierExecutionSample, FrontierGraphShape, FrontierMemoryBudget, FrontierTopology,
    MegakernelByteLayout, MegakernelDeviceCapabilities, MegakernelExecutionSample,
    MegakernelGraphShape, MegakernelMemoryBudget, MegakernelMemoryError,
    NeutralMegakernelExecutionPlanner,
};
// The wave and dependency corpora are the neutral policy's own definitions,
// imported rather than restated: a table copied into this suite would let the
// two entry points be driven with different inputs and still agree, which is
// exactly the divergence this gate exists to catch.
use vyre_driver::megakernel_fixtures::{
    CHAIN_DEPENDENCIES as CHAIN, CYCLE_DEPENDENCIES as CYCLE, DIAMOND_DEPENDENCIES as DIAMOND,
    DIAMOND_WAVES, ONE_WAVE, OVERFLOW_WAVES, THREE_EQUAL_WAVES,
};
use vyre_driver::megakernel_frontier::{
    plan_megakernel_frontier_execution, MegakernelFrontierExecutionPlan,
    MegakernelFrontierExecutionPlanError, MegakernelFrontierWave,
};
use vyre_driver_cuda::{
    plan_cuda_frontier_megakernel_execution, CudaMegakernelAnalysisKind, CudaMegakernelDeviceKey,
    CudaMegakernelPlanCache, CudaMegakernelScheduleSample,
};
use vyre_driver_cuda::{
    plan_cuda_megakernel_execution, select_cuda_megakernel_topology,
    select_cuda_megakernel_topology_stable,
};

/// One frontier-scheduling decision, normalized so both entry points compare.
#[derive(Debug, PartialEq, Eq)]
struct FrontierDecision {
    groups: Vec<Vec<usize>>,
    global_barriers: usize,
    peak_frontier_bytes: u64,
    peak_scratch_bytes: u64,
    peak_output_bytes: u64,
    amortized_readback_bytes: u64,
    max_group_width: usize,
    topology: FrontierTopology,
    downgraded_to_sparse: bool,
    graph_bytes: u64,
    scratch_bytes: u64,
    required_bytes: u64,
    memory_pressure_bps: u32,
}

/// Planner rejection, normalized so both entry points compare.
#[derive(Debug, PartialEq, Eq)]
enum FrontierRejection {
    Barrier(String),
    ByteCountOverflow(&'static str),
    GroupOverBudget {
        required_bytes: u64,
        budget_bytes: u64,
        field: &'static str,
    },
    OverBudget {
        topology: FrontierTopology,
        required_bytes: u64,
        budget_bytes: u64,
    },
    StorageReserveFailed(&'static str),
}

type FrontierOutcome = Result<FrontierDecision, FrontierRejection>;

/// One line naming every decision the scenario is allowed to reach.
///
/// Both entry points must produce this, so the suite fails on a fork between
/// them and on a change to the shared policy they both call.
fn signature(outcome: &FrontierOutcome) -> String {
    match outcome {
        Ok(decision) => {
            let groups = decision
                .groups
                .iter()
                .map(|group| {
                    let waves = group
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("[{waves}]")
                })
                .collect::<String>();
            format!(
                "groups={groups} barriers={} peak={}/{}/{} readback={} width={} topology={:?} downgraded={} graph={} scratch={} required={} pressure={}",
                decision.global_barriers,
                decision.peak_frontier_bytes,
                decision.peak_scratch_bytes,
                decision.peak_output_bytes,
                decision.amortized_readback_bytes,
                decision.max_group_width,
                decision.topology,
                decision.downgraded_to_sparse,
                decision.graph_bytes,
                decision.scratch_bytes,
                decision.required_bytes,
                decision.memory_pressure_bps,
            )
        }
        Err(FrontierRejection::Barrier(message)) => format!("rejected=Barrier {message}"),
        Err(FrontierRejection::ByteCountOverflow(field)) => {
            format!("rejected=ByteCountOverflow field={field}")
        }
        Err(FrontierRejection::GroupOverBudget {
            required_bytes,
            budget_bytes,
            field,
        }) => format!(
            "rejected=GroupOverBudget required={required_bytes} budget={budget_bytes} field={field}"
        ),
        Err(FrontierRejection::OverBudget {
            topology,
            required_bytes,
            budget_bytes,
        }) => format!(
            "rejected=OverBudget topology={topology:?} required={required_bytes} budget={budget_bytes}"
        ),
        Err(FrontierRejection::StorageReserveFailed(field)) => {
            format!("rejected=StorageReserveFailed field={field}")
        }
    }
}

struct Scenario {
    name: &'static str,
    expects: &'static str,
    dispatch_cost_ns: f64,
    frontier_density: f64,
    readback_bytes: u64,
    graph: MegakernelGraphShape,
    bytes_per_node: u64,
    bytes_per_edge: u64,
    waves: &'static [MegakernelFrontierWave],
    dependencies: &'static [MegakernelWaveDependency],
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    supports_grid_sync: bool,
}

impl Scenario {
    fn sample(&self) -> MegakernelExecutionSample {
        MegakernelExecutionSample {
            dispatch_cost_ns: self.dispatch_cost_ns,
            frontier_density: self.frontier_density,
            readback_bytes: self.readback_bytes,
        }
    }

    fn cuda_sample(&self) -> CudaMegakernelScheduleSample {
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: self.dispatch_cost_ns,
            frontier_density: self.frontier_density,
            readback_bytes: self.readback_bytes,
        }
    }

    fn capabilities(&self) -> MegakernelDeviceCapabilities {
        MegakernelDeviceCapabilities {
            supports_device_wide_barrier: self.supports_grid_sync,
        }
    }

    fn device(&self) -> CudaMegakernelDeviceKey {
        CudaMegakernelDeviceKey {
            sm_major: 12,
            sm_minor: 0,
            warp_size: 32,
            supports_grid_sync: self.supports_grid_sync,
            supports_tensor_cores: true,
            max_workgroup_size: 1024,
        }
    }
}

impl From<MegakernelFrontierExecutionPlanError> for FrontierRejection {
    fn from(error: MegakernelFrontierExecutionPlanError) -> Self {
        match error {
            MegakernelFrontierExecutionPlanError::Barrier(error) => {
                Self::Barrier(error.to_string())
            }
            MegakernelFrontierExecutionPlanError::ByteCountOverflow { field } => {
                Self::ByteCountOverflow(field)
            }
            MegakernelFrontierExecutionPlanError::GroupOverBudget {
                required_bytes,
                budget_bytes,
                field,
            } => Self::GroupOverBudget {
                required_bytes,
                budget_bytes,
                field,
            },
            MegakernelFrontierExecutionPlanError::Memory(error) => error.into(),
            MegakernelFrontierExecutionPlanError::StorageReserveFailed { field, .. } => {
                Self::StorageReserveFailed(field)
            }
        }
    }
}

impl From<MegakernelMemoryError> for FrontierRejection {
    fn from(error: MegakernelMemoryError) -> Self {
        match error {
            MegakernelMemoryError::ByteCountOverflow { field } => Self::ByteCountOverflow(field),
            MegakernelMemoryError::OverBudget {
                topology,
                required_bytes,
                budget_bytes,
                ..
            } => Self::OverBudget {
                topology,
                required_bytes,
                budget_bytes,
            },
            MegakernelMemoryError::InvalidSample { field } => {
                panic!("fixture supplied an unrepresentable {field}")
            }
        }
    }
}

fn decision(plan: &MegakernelFrontierExecutionPlan) -> FrontierDecision {
    FrontierDecision {
        groups: plan
            .barriers
            .groups
            .iter()
            .map(|group| group.waves.clone())
            .collect(),
        global_barriers: plan.barriers.global_barriers,
        peak_frontier_bytes: plan.peak_frontier_bytes,
        peak_scratch_bytes: plan.peak_scratch_bytes,
        peak_output_bytes: plan.peak_output_bytes,
        amortized_readback_bytes: plan.amortized_readback_bytes,
        max_group_width: plan.max_group_width,
        topology: plan.execution.topology,
        downgraded_to_sparse: plan.execution.downgraded_to_sparse,
        graph_bytes: plan.execution.memory.graph_bytes,
        scratch_bytes: plan.execution.memory.scratch_bytes,
        required_bytes: plan.execution.memory.required_bytes,
        memory_pressure_bps: plan.execution.memory.memory_pressure_bps,
    }
}

/// Neutral answer: `vyre-driver` alone, no CUDA types anywhere on this path.
fn neutral_frontier_outcome(scenario: &Scenario) -> FrontierOutcome {
    plan_megakernel_frontier_execution(
        &mut NeutralMegakernelExecutionPlanner,
        scenario.sample(),
        scenario.graph,
        scenario.bytes_per_node,
        scenario.bytes_per_edge,
        scenario.waves,
        scenario.dependencies,
        scenario.budget_bytes,
        scenario.launch_overhead_ns,
        scenario.fusion_pressure,
        scenario.capabilities(),
    )
    .map(|plan| decision(&plan))
    .map_err(FrontierRejection::from)
}

/// CUDA answer: the concrete driver's own frontier entry point, cold cache.
fn cuda_frontier_outcome(scenario: &Scenario) -> FrontierOutcome {
    let mut cache = CudaMegakernelPlanCache::new();
    plan_cuda_frontier_megakernel_execution(
        &mut cache,
        0xF00D_BEEF,
        CudaMegakernelAnalysisKind::Dataflow,
        scenario.device(),
        scenario.cuda_sample(),
        scenario.graph,
        scenario.bytes_per_node,
        scenario.bytes_per_edge,
        scenario.waves,
        scenario.dependencies,
        scenario.budget_bytes,
        scenario.launch_overhead_ns,
        scenario.fusion_pressure,
    )
    .map(|plan| decision(&plan))
    .map_err(FrontierRejection::from)
}

fn scenarios() -> Vec<Scenario> {
    let base = Scenario {
        name: "",
        expects: "",
        dispatch_cost_ns: 1_000.0,
        frontier_density: 0.5,
        readback_bytes: 4_096,
        graph: MegakernelGraphShape {
            node_count: 1_000,
            edge_count: 4_000,
        },
        bytes_per_node: 16,
        bytes_per_edge: 8,
        waves: DIAMOND_WAVES,
        dependencies: DIAMOND,
        budget_bytes: 128 * 1024,
        launch_overhead_ns: 250.0,
        fusion_pressure: 0.95,
        supports_grid_sync: true,
    };
    vec![
        Scenario {
            name: "single wave, no dependencies",
            expects: "groups=[0] barriers=0 peak=40/15/0 readback=4096 width=1 topology=FusedWave downgraded=false graph=0 scratch=60 required=100 pressure=244",
            waves: ONE_WAVE,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 1,
                edge_count: 0,
            },
            bytes_per_node: 0,
            bytes_per_edge: 0,
            budget_bytes: 4_096,
            ..base
        },
        Scenario {
            name: "exactly-full fused group",
            expects: "groups=[0,1,2] barriers=0 peak=120/45/0 readback=4096 width=3 topology=FusedWave downgraded=false graph=0 scratch=180 required=300 pressure=10000",
            waves: THREE_EQUAL_WAVES,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 1,
                edge_count: 0,
            },
            bytes_per_node: 0,
            bytes_per_edge: 0,
            budget_bytes: 300,
            ..base
        },
        Scenario {
            name: "ragged final fused group",
            expects: "groups=[0,1][2] barriers=1 peak=80/30/0 readback=4096 width=2 topology=FusedWave downgraded=false graph=0 scratch=120 required=200 pressure=8000",
            waves: THREE_EQUAL_WAVES,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 1,
                edge_count: 0,
            },
            bytes_per_node: 0,
            bytes_per_edge: 0,
            budget_bytes: 250,
            ..base
        },
        Scenario {
            name: "one wave per split group",
            expects: "groups=[0][1][2] barriers=2 peak=40/15/0 readback=4096 width=1 topology=FusedWave downgraded=false graph=0 scratch=60 required=100 pressure=10000",
            waves: THREE_EQUAL_WAVES,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 1,
                edge_count: 0,
            },
            bytes_per_node: 0,
            bytes_per_edge: 0,
            budget_bytes: 100,
            ..base
        },
        Scenario {
            name: "empty frontier",
            expects: "groups= barriers=0 peak=0/0/0 readback=4096 width=0 topology=FusedWave downgraded=false graph=48000 scratch=0 required=48000 pressure=0",
            waves: &[],
            dependencies: &[],
            budget_bytes: 1 << 30,
            ..base
        },
        Scenario {
            name: "empty frontier, zero budget headroom",
            expects: "groups= barriers=0 peak=0/0/0 readback=4096 width=0 topology=SparseFrontier downgraded=false graph=32 scratch=0 required=32 pressure=10000",
            waves: &[],
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 4,
                edge_count: 0,
            },
            bytes_per_node: 8,
            bytes_per_edge: 0,
            budget_bytes: 32,
            ..base
        },
        Scenario {
            name: "dependency chain",
            expects: "groups=[0][1][2] barriers=2 peak=40/15/0 readback=4096 width=1 topology=FusedWave downgraded=false graph=48000 scratch=60 required=48100 pressure=458",
            waves: THREE_EQUAL_WAVES,
            dependencies: CHAIN,
            budget_bytes: 1 << 20,
            ..base
        },
        Scenario {
            name: "diamond dependencies, dense telemetry",
            expects: "groups=[0][1,2][3] barriers=2 peak=8192/4096/2048 readback=1048576 width=2 topology=FusedWave downgraded=false graph=48000 scratch=16384 required=74624 pressure=5693",
            frontier_density: 0.90,
            readback_bytes: 1 << 20,
            ..base
        },
        Scenario {
            name: "diamond dependencies, no grid sync",
            expects: "groups=[0][1,2][3] barriers=2 peak=8192/4096/2048 readback=1048576 width=2 topology=BlockDenseFrontier downgraded=false graph=48000 scratch=8192 required=66432 pressure=5068",
            frontier_density: 0.90,
            readback_bytes: 1 << 20,
            supports_grid_sync: false,
            ..base
        },
        Scenario {
            name: "static output pressure drives fusion",
            expects: "groups=[0,1] barriers=0 peak=2048/1024/6144 readback=6144 width=2 topology=FusedWave downgraded=false graph=48000 scratch=4096 required=60288 pressure=4599",
            frontier_density: 0.50,
            readback_bytes: 0,
            waves: &[
                MegakernelFrontierWave {
                    frontier_bytes: 1_024,
                    scratch_bytes: 512,
                    output_bytes: 3_072,
                },
                MegakernelFrontierWave {
                    frontier_bytes: 1_024,
                    scratch_bytes: 512,
                    output_bytes: 3_072,
                },
            ],
            dependencies: &[],
            ..base
        },
        Scenario {
            name: "static output pressure, no grid sync",
            expects: "groups=[0,1] barriers=0 peak=2048/1024/6144 readback=6144 width=2 topology=HybridFrontier downgraded=false graph=48000 scratch=3072 required=59264 pressure=4521",
            frontier_density: 0.50,
            readback_bytes: 0,
            supports_grid_sync: false,
            waves: &[
                MegakernelFrontierWave {
                    frontier_bytes: 1_024,
                    scratch_bytes: 512,
                    output_bytes: 3_072,
                },
                MegakernelFrontierWave {
                    frontier_bytes: 1_024,
                    scratch_bytes: 512,
                    output_bytes: 3_072,
                },
            ],
            dependencies: &[],
            ..base
        },
        Scenario {
            name: "ultra sparse frontier",
            expects: "groups=[0][1,2][3] barriers=2 peak=8192/4096/2048 readback=4096 width=2 topology=WarpSparseFrontier downgraded=false graph=98304 scratch=4096 required=112640 pressure=8593",
            frontier_density: 0.01,
            graph: MegakernelGraphShape {
                node_count: 4_096,
                edge_count: 4_096,
            },
            fusion_pressure: 0.0,
            ..base
        },
        Scenario {
            name: "cyclic dependencies",
            expects: "rejected=Barrier megakernel wave dependency graph contains a cycle with 2 unscheduled waves. Fix: break the cyclic dataflow edge or insert an explicit iterative fixed-point kernel.",
            waves: THREE_EQUAL_WAVES,
            dependencies: CYCLE,
            budget_bytes: 1 << 20,
            ..base
        },
        Scenario {
            name: "resident graph over budget",
            expects: "rejected=GroupOverBudget required=1600 budget=1000 field=resident graph bytes",
            waves: ONE_WAVE,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 100,
                edge_count: 100,
            },
            bytes_per_node: 8,
            bytes_per_edge: 8,
            budget_bytes: 1_000,
            ..base
        },
        Scenario {
            name: "single wave over group budget",
            expects: "rejected=GroupOverBudget required=100 budget=50 field=single fused frontier wave bytes",
            waves: ONE_WAVE,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 1,
                edge_count: 0,
            },
            bytes_per_node: 0,
            bytes_per_edge: 0,
            budget_bytes: 50,
            ..base
        },
        Scenario {
            name: "frontier wave byte overflow",
            expects: "rejected=ByteCountOverflow field=fused wave bytes",
            waves: OVERFLOW_WAVES,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: 1,
                edge_count: 1,
            },
            bytes_per_node: 1,
            bytes_per_edge: 1,
            budget_bytes: u64::MAX,
            ..base
        },
        Scenario {
            name: "largest supported grid",
            expects: "rejected=GroupOverBudget required=100 budget=0 field=single fused frontier wave bytes",
            waves: ONE_WAVE,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: u64::MAX,
                edge_count: 0,
            },
            bytes_per_node: 1,
            bytes_per_edge: 0,
            budget_bytes: u64::MAX,
            ..base
        },
        Scenario {
            name: "grid layout bytes overflow",
            expects: "rejected=ByteCountOverflow field=graph layout bytes",
            waves: ONE_WAVE,
            dependencies: &[],
            graph: MegakernelGraphShape {
                node_count: u64::MAX,
                edge_count: u64::MAX,
            },
            bytes_per_node: 1,
            bytes_per_edge: 1,
            budget_bytes: u64::MAX,
            ..base
        },
    ]
}

#[test]
fn cuda_and_neutral_frontier_scheduling_agree_decision_for_decision() {
    let mut drift = Vec::new();
    for scenario in scenarios() {
        let cuda = signature(&cuda_frontier_outcome(&scenario));
        let neutral = signature(&neutral_frontier_outcome(&scenario));
        if cuda != neutral || cuda != scenario.expects {
            drift.push(format!(
                "\n  scenario `{}`\n    expected: {}\n    neutral:  {neutral}\n    cuda:     {cuda}",
                scenario.name, scenario.expects
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "Fix: barrier placement, wave topology, and frontier scheduling are one policy owned by \
         vyre-driver. A `neutral` line that differs from `cuda` is a backend fork; a line that \
         differs from `expected` is a change to the shared policy that no caller asked for.{}",
        drift.join("")
    );
}

const TOPOLOGIES: [FrontierTopology; 6] = [
    FrontierTopology::WarpSparseFrontier,
    FrontierTopology::SparseFrontier,
    FrontierTopology::BlockDenseFrontier,
    FrontierTopology::DenseFrontier,
    FrontierTopology::HybridFrontier,
    FrontierTopology::FusedWave,
];

/// Density spread: every threshold, both sides of every threshold, and the
/// non-finite inputs the policy is required to clamp.
const DENSITIES: [f64; 16] = [
    0.0,
    0.03125,
    0.031_26,
    0.05,
    0.125,
    0.125_1,
    0.35,
    0.6999,
    0.70,
    0.7001,
    0.8499,
    0.85,
    1.0,
    -1.0,
    2.0,
    f64::NAN,
];

/// Degree spread straddling the dense (2.0) and warp-sparse (8.0) proxies.
const GRAPHS: [MegakernelGraphShape; 5] = [
    MegakernelGraphShape {
        node_count: 1,
        edge_count: 0,
    },
    MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 1_999,
    },
    MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 2_000,
    },
    MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 8_000,
    },
    MegakernelGraphShape {
        node_count: u64::MAX,
        edge_count: u64::MAX,
    },
];

const MEMORY_BUDGETS: [MegakernelMemoryBudget; 4] = [
    MegakernelMemoryBudget {
        required_bytes: 0,
        budget_bytes: 1 << 20,
    },
    MegakernelMemoryBudget {
        required_bytes: 890_000,
        budget_bytes: 1_000_000,
    },
    MegakernelMemoryBudget {
        required_bytes: 900_000,
        budget_bytes: 1_000_000,
    },
    MegakernelMemoryBudget {
        required_bytes: 1,
        budget_bytes: 0,
    },
];

#[test]
fn cuda_and_neutral_topology_selection_agree_decision_for_decision() {
    for density in DENSITIES {
        for graph in GRAPHS {
            for memory in MEMORY_BUDGETS {
                for (readback_bytes, fusion_pressure, launch_overhead_ns) in [
                    (0u64, 0.0f64, 0.0f64),
                    (4_095, 0.70, 1_500.0),
                    (4_096, 0.6999, 1_500.0),
                    (4_096, 0.70, 1_499.0),
                    (4_096, 0.70, 1_500.0),
                    (1 << 20, 0.95, 250.0),
                ] {
                    let sample = MegakernelExecutionSample {
                        dispatch_cost_ns: 1_000.0,
                        frontier_density: density,
                        readback_bytes,
                    };
                    let cuda_sample = CudaMegakernelScheduleSample {
                        dispatch_cost_ns: sample.dispatch_cost_ns,
                        frontier_density: sample.frontier_density,
                        readback_bytes: sample.readback_bytes,
                    };
                    // Both entry points receive the same corpus facts, converted
                    // the way each wrapper converts them, so a divergence is a
                    // policy difference and never a difference in the inputs.
                    let frontier_sample = FrontierExecutionSample {
                        dispatch_cost_ns: sample.dispatch_cost_ns,
                        frontier_density: sample.frontier_density,
                        readback_bytes: sample.readback_bytes,
                    };
                    let frontier_graph = FrontierGraphShape {
                        node_count: graph.node_count,
                        edge_count: graph.edge_count,
                    };
                    let frontier_memory = FrontierMemoryBudget {
                        required_bytes: memory.required_bytes,
                        budget_bytes: memory.budget_bytes,
                    };
                    let label = format!(
                        "density={density} graph={graph:?} memory={memory:?} readback={readback_bytes} fusion={fusion_pressure} launch={launch_overhead_ns}"
                    );
                    assert_eq!(
                        select_cuda_megakernel_topology(
                            cuda_sample,
                            graph,
                            memory,
                            launch_overhead_ns,
                            fusion_pressure,
                        ),
                        select_frontier_topology(
                            frontier_sample,
                            frontier_graph,
                            frontier_memory,
                            launch_overhead_ns,
                            fusion_pressure,
                            MegakernelDeviceCapabilities::FUSION_CAPABLE
                                .supports_device_wide_barrier,
                        ),
                        "Fix: CUDA topology selection diverged from the neutral policy for {label}."
                    );
                    for previous in TOPOLOGIES {
                        assert_eq!(
                            select_cuda_megakernel_topology_stable(
                                cuda_sample,
                                graph,
                                memory,
                                launch_overhead_ns,
                                fusion_pressure,
                                previous,
                            ),
                            select_frontier_topology_stable(
                                frontier_sample,
                                frontier_graph,
                                frontier_memory,
                                launch_overhead_ns,
                                fusion_pressure,
                                previous,
                                MegakernelDeviceCapabilities::FUSION_CAPABLE
                                    .supports_device_wide_barrier,
                            ),
                            "Fix: CUDA topology hysteresis diverged from the neutral policy for \
                             {label} previous={previous:?}."
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cuda_and_neutral_execution_planning_agree_decision_for_decision() {
    for density in DENSITIES {
        for graph in GRAPHS {
            for bytes in [
                MegakernelByteLayout::default(),
                MegakernelByteLayout {
                    bytes_per_node: 16,
                    bytes_per_edge: 8,
                    frontier_bytes: 8_192,
                    scratch_bytes: 4_096,
                    output_bytes: 2_048,
                    budget_bytes: 128 * 1024,
                },
                MegakernelByteLayout {
                    bytes_per_node: 16,
                    bytes_per_edge: 8,
                    frontier_bytes: 8_192,
                    scratch_bytes: 4_096,
                    output_bytes: 2_048,
                    budget_bytes: 40_000,
                },
                MegakernelByteLayout {
                    bytes_per_node: 1,
                    bytes_per_edge: 1,
                    frontier_bytes: 1,
                    scratch_bytes: 1,
                    output_bytes: 1,
                    budget_bytes: u64::MAX,
                },
                MegakernelByteLayout {
                    bytes_per_node: u64::MAX,
                    bytes_per_edge: u64::MAX,
                    frontier_bytes: 1,
                    scratch_bytes: 1,
                    output_bytes: 1,
                    budget_bytes: u64::MAX,
                },
            ] {
                let sample = MegakernelExecutionSample {
                    dispatch_cost_ns: 1_000.0,
                    frontier_density: density,
                    readback_bytes: 1 << 20,
                };
                let cuda_sample = CudaMegakernelScheduleSample {
                    dispatch_cost_ns: sample.dispatch_cost_ns,
                    frontier_density: sample.frontier_density,
                    readback_bytes: sample.readback_bytes,
                };
                assert_eq!(
                    plan_cuda_megakernel_execution(cuda_sample, graph, bytes, 250.0, 0.95),
                    plan_megakernel_execution(
                        sample,
                        graph,
                        bytes,
                        250.0,
                        0.95,
                        MegakernelDeviceCapabilities::FUSION_CAPABLE,
                    ),
                    "Fix: CUDA execution planning diverged from the neutral policy for \
                     density={density} graph={graph:?} bytes={bytes:?}."
                );
            }
        }
    }
}
