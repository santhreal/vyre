use std::time::Instant;

use crate::api::case::{BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, WorkloadClass};
use crate::api::metric::{elapsed_ns, BenchMetrics};
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, transfer_accounting_with_resident_reset,
    ResidentInputSet, TransferAccounting,
};
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::reference_sample::timed_reference;
use vyre_driver::{ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs::graph::csr_forward_or_changed::csr_forward_or_changed_parallel;
use vyre_libs::graph::program_graph::ProgramGraphShape;

use super::fixture::{
    build_ifds_skewed_fixture, ifds_closure_inputs, ifds_closure_resident_inputs,
    ifds_skewed_closure_oracle, ifds_skewed_launch_wave_iterations, IfdsSkewedStats,
    IFDS_REACH_MASK, NODE_COUNT,
};
use super::metrics::{ifds_closure_baseline_metric_points, ifds_closure_metric_points};
use super::SUITES;

pub(super) const CLOSURE_MAX_ITERS: u32 = 64;
const CLOSURE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const SEED_FRONTIER_RESOURCE_INDEX: usize = 5;
const FRONTIER_RESOURCE_INDEX: usize = 6;
const CHANGED_RESOURCE_INDEX: usize = 7;
const CLOSURE_RESOURCE_INDICES: [usize; 7] = [
    0,
    1,
    2,
    3,
    4,
    FRONTIER_RESOURCE_INDEX,
    CHANGED_RESOURCE_INDEX,
];
const RESET_RESOURCE_INDICES: [usize; 3] = [
    SEED_FRONTIER_RESOURCE_INDEX,
    FRONTIER_RESOURCE_INDEX,
    CHANGED_RESOURCE_INDEX,
];

pub(super) struct DataflowIfdsSkewedClosurePrepared {
    pub(super) program: Program,
    pub(super) reset_program: Program,
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) input_bytes_total: u64,
    pub(super) baseline_outputs: Vec<Vec<u8>>,
    pub(super) baseline_wall_ns: u64,
    pub(super) stats: IfdsSkewedStats,
    pub(super) closure_iterations: u32,
    pub(super) dispatch_iterations: u32,
    pub(super) closure_changed: u32,
    pub(super) resident: Option<ResidentInputSet>,
}

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "dataflow.ifds.skewed.closure.1m",
    name: "Dataflow IFDS Skewed Closure 1M",
    summary: "Bounded IFDS reachability closure over a million-node skewed exploded-supergraph CSR with resident frontier accumulation",
    tags: &[
        "dataflow",
        "ifds",
        "graph",
        "csr",
        "bitset",
        "closure",
        "skewed-degree",
        "irregular",
        "resident",
        "release",
    ],
    layer: BenchLayer::Libs,
    workload: WorkloadClass::Macro,
    owner_crate: "vyre-primitives",
    suites: SUITES,
    min_vram_bytes: Some(96 * 1024 * 1024),
    min_input_bytes: Some(NODE_COUNT as u64 * 20),
    feature_set: &[
        "dataflow",
        "ifds",
        "skewed-csr",
        "resident-frontier",
    ],
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<DataflowIfdsSkewedClosurePrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: closure_program,
    fingerprint: None,
    bytes_touched,
};

static CASE: HarnessCase<DataflowIfdsSkewedClosurePrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn build_case(ctx: &mut BenchContext) -> Result<DataflowIfdsSkewedClosurePrepared, BenchError> {
    prepare_ifds_skewed_closure(Some(ctx))
}

fn closure_program(prepared: &DataflowIfdsSkewedClosurePrepared) -> Option<&Program> {
    Some(&prepared.program)
}

fn bytes_touched(prepared: &DataflowIfdsSkewedClosurePrepared) -> (u64, u64) {
    (
        prepared.input_bytes_total,
        prepared
            .baseline_outputs
            .iter()
            .map(Vec::len)
            .sum::<usize>() as u64,
    )
}

/// Replay the bounded closure, resident when the backend keeps buffers on the
/// device and as one fixpoint dispatch otherwise.
fn measure(
    ctx: &mut BenchContext,
    prepared: &mut DataflowIfdsSkewedClosurePrepared,
) -> Result<BenchRun, BenchError> {
    let workgroup = prepared.program.workgroup_size();
    if workgroup.contains(&0) {
        return Err(BenchError::ExecutionFailed(format!(
                "IFDS closure benchmark received invalid workgroup {:?}. Fix: use positive dispatch dimensions.",
                workgroup
            )));
    }

    let mut reported_workgroup_x = workgroup[0];
    let (outputs, wall_ns, dispatch_ns, resident_used, resident_reset_bytes, device_reset_sequence) =
        if let Some(resident) = prepared.resident.as_ref() {
            let sequence = dispatch_resident_closure_sequence(ctx, prepared, resident, workgroup)?;
            (sequence.outputs, sequence.wall_ns, None, true, 0, true)
        } else {
            let mut dispatch_config = ctx.dispatch_config.clone();
            dispatch_config.fixpoint_iterations = Some(prepared.dispatch_iterations);
            dispatch_config
                .workgroup_override
                .get_or_insert(CLOSURE_WORKGROUP_SIZE);
            let dispatch_workgroup = dispatch_config
                .workgroup_override
                .unwrap_or_else(|| prepared.program.workgroup_size());
            if dispatch_workgroup.contains(&0) {
                return Err(BenchError::ExecutionFailed(format!(
                    "IFDS closure benchmark received invalid workgroup {:?}. Fix: use positive dispatch dimensions.",
                    dispatch_workgroup
                )));
            }
            reported_workgroup_x = dispatch_workgroup[0];
            dispatch_config.grid_override.get_or_insert([
                prepared.stats.nodes.div_ceil(dispatch_workgroup[0]),
                1,
                1,
            ]);
            let dispatch = dispatch_program_timed(
                ctx,
                &prepared.program,
                None,
                &prepared.inputs,
                &dispatch_config,
            )?;
            (
                dispatch.timed.outputs,
                dispatch.timed.wall_ns,
                dispatch.timed.device_ns,
                dispatch.resident_used,
                0,
                false,
            )
        };
    let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;
    let accounting = closure_transfer_accounting(
        prepared.input_bytes_total,
        output_bytes,
        resident_used,
        resident_reset_bytes,
    );

    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(wall_ns),
            dispatch_ns,
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(output_bytes),
            bytes_read: Some(accounting.bytes_read),
            bytes_written: Some(accounting.bytes_written),
            bytes_touched: Some(accounting.bytes_touched),
            custom: ifds_closure_metric_points(
                prepared.stats,
                prepared.closure_iterations,
                prepared.closure_changed,
                prepared.baseline_wall_ns,
                wall_ns,
                resident_used,
                resident_reset_bytes,
                device_reset_sequence,
                prepared.dispatch_iterations,
                CLOSURE_MAX_ITERS,
                reported_workgroup_x,
            ),
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(prepared.baseline_wall_ns),
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(
                prepared
                    .baseline_outputs
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>() as u64,
            ),
            custom: ifds_closure_baseline_metric_points(
                prepared.stats,
                prepared.closure_iterations,
                prepared.closure_changed,
                prepared.dispatch_iterations,
                CLOSURE_MAX_ITERS,
            ),
            ..Default::default()
        }),
        outputs,
        baseline_outputs: Some(prepared.baseline_outputs.clone()),
    })
}

pub(super) fn prepare_ifds_skewed_closure(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedClosurePrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let shape = ProgramGraphShape::new(fixture.stats.nodes, fixture.stats.edges);
    let mut program =
        csr_forward_or_changed_parallel(shape, "frontier_accumulator", "changed", IFDS_REACH_MASK);
    program.set_workgroup_size(CLOSURE_WORKGROUP_SIZE);
    let reset_program = ifds_closure_reset_program(fixture.stats.frontier_words);

    let (oracle, baseline_wall_ns) =
        timed_reference(|| ifds_skewed_closure_oracle(&fixture, CLOSURE_MAX_ITERS));
    let mut stats = fixture.stats;
    stats.output_words_set = oracle.output_words_set;
    let dispatch_iterations = ifds_skewed_launch_wave_iterations(&fixture, CLOSURE_MAX_ITERS);

    let inputs = ifds_closure_inputs(&fixture);
    let resident_inputs = ifds_closure_resident_inputs(&fixture);
    let input_bytes_total = input_bytes_total(&inputs);
    let baseline_outputs = vec![
        vyre_primitives::wire::pack_u32_slice(&oracle.output),
        vyre_primitives::wire::pack_u32_slice(&[oracle.changed]),
    ];
    let resident = ctx
        .map(|ctx| {
            ResidentInputSet::upload_optional(ctx, &resident_inputs, "dataflow IFDS closure")
        })
        .transpose()?
        .flatten();

    Ok(DataflowIfdsSkewedClosurePrepared {
        program,
        reset_program,
        inputs,
        input_bytes_total,
        baseline_outputs,
        baseline_wall_ns,
        stats,
        closure_iterations: oracle.iterations,
        dispatch_iterations,
        closure_changed: oracle.changed,
        resident,
    })
}

fn closure_transfer_accounting(
    input_bytes_total: u64,
    output_bytes_total: u64,
    resident_used: bool,
    resident_reset_bytes: u64,
) -> TransferAccounting {
    transfer_accounting_with_resident_reset(
        input_bytes_total,
        output_bytes_total,
        resident_used,
        resident_reset_bytes,
    )
}

struct ClosureSequenceRun {
    outputs: Vec<Vec<u8>>,
    wall_ns: u64,
}

fn dispatch_resident_closure_sequence(
    ctx: &BenchContext,
    prepared: &DataflowIfdsSkewedClosurePrepared,
    resident: &ResidentInputSet,
    workgroup: [u32; 3],
) -> Result<ClosureSequenceRun, BenchError> {
    if let Some(override_workgroup) = ctx.dispatch_config.workgroup_override {
        if override_workgroup != workgroup {
            return Err(BenchError::ExecutionFailed(format!(
                "IFDS closure resident sequence uses program workgroup {:?}, but received override {:?}. Fix: run the resident closure sequence without a workgroup override or rebuild the resident sequence program.",
                workgroup, override_workgroup
            )));
        }
    }

    let reset_resources =
        resident.resources_for_indices(&RESET_RESOURCE_INDICES, "IFDS closure reset sequence")?;
    let closure_resources = resident
        .resources_for_indices(&CLOSURE_RESOURCE_INDICES, "IFDS closure traversal sequence")?;
    let reset_step = ResidentDispatchStep {
        program: &prepared.reset_program,
        resources: &reset_resources,
        grid_override: Some([
            prepared.stats.frontier_words.div_ceil(workgroup[0]).max(1),
            1,
            1,
        ]),
        workgroup_override: None,
    };
    let closure_step = ResidentDispatchStep {
        program: &prepared.program,
        resources: &closure_resources,
        grid_override: Some([prepared.stats.nodes.div_ceil(workgroup[0]).max(1), 1, 1]),
        workgroup_override: None,
    };
    let read_ranges = [
        ResidentReadRange {
            resource: &reset_resources[1],
            byte_offset: 0,
            byte_len: prepared.baseline_outputs[0].len(),
        },
        ResidentReadRange {
            resource: &reset_resources[2],
            byte_offset: 0,
            byte_len: prepared.baseline_outputs[1].len(),
        },
    ];

    let mut frontier_output = Vec::with_capacity(prepared.baseline_outputs[0].len());
    let mut changed_output = Vec::with_capacity(prepared.baseline_outputs[1].len());
    let started = Instant::now();
    ctx.dispatch_resident_repeated_sequence_read_ranges_into(
        &[reset_step],
        &[closure_step],
        prepared.dispatch_iterations,
        &read_ranges,
        &mut [&mut frontier_output, &mut changed_output],
    )
    .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    let wall_ns = elapsed_ns(started);

    Ok(ClosureSequenceRun {
        outputs: vec![frontier_output, changed_output],
        wall_ns,
    })
}

fn ifds_closure_reset_program(frontier_words: u32) -> Program {
    let idx = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("frontier_seed", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(frontier_words.max(1)),
            BufferDecl::storage(
                "frontier_accumulator",
                1,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(frontier_words.max(1)),
            BufferDecl::storage("changed", 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        CLOSURE_WORKGROUP_SIZE,
        vec![
            Node::if_then(
                Expr::lt(idx.clone(), Expr::u32(frontier_words)),
                vec![Node::store(
                    "frontier_accumulator",
                    idx.clone(),
                    Expr::load("frontier_seed", idx.clone()),
                )],
            ),
            Node::if_then(
                Expr::eq(idx, Expr::u32(0)),
                vec![Node::store("changed", Expr::u32(0), Expr::u32(0))],
            ),
        ],
    )
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
