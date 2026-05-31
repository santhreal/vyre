use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, ResidentInputSet, TransferAccounting,
};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_forward_or_changed::csr_forward_or_changed_parallel;
use vyre_primitives::graph::program_graph::ProgramGraphShape;

use super::fixture::{
    build_ifds_skewed_fixture, ifds_closure_inputs, ifds_skewed_closure_oracle, IfdsSkewedStats,
    IFDS_REACH_MASK, NODE_COUNT,
};
use super::metrics::{ifds_closure_baseline_metric_points, ifds_closure_metric_points};
use super::SUITES;

pub(super) const CLOSURE_MAX_ITERS: u32 = 64;
const CLOSURE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const FRONTIER_RESOURCE_INDEX: usize = 5;
const CHANGED_RESOURCE_INDEX: usize = 6;
const ZERO_U32_BYTES: &[u8] = &[0, 0, 0, 0];

pub(super) struct DataflowIfdsSkewedClosurePrepared {
    pub(super) program: Program,
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) input_bytes_total: u64,
    pub(super) seed_frontier_bytes: Vec<u8>,
    pub(super) baseline_outputs: Vec<Vec<u8>>,
    pub(super) baseline_wall_ns: u64,
    pub(super) stats: IfdsSkewedStats,
    pub(super) closure_iterations: u32,
    pub(super) closure_changed: u32,
    pub(super) resident: Option<ResidentInputSet>,
}

/// Fixed-replay IFDS closure over a resident skewed exploded-supergraph.
struct DataflowIfdsSkewedClosure;

impl BenchCase for DataflowIfdsSkewedClosure {
    fn id(&self) -> BenchId {
        BenchId("dataflow.ifds.skewed.closure.1m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Dataflow IFDS Skewed Closure 1M".to_string(),
            description: "Bounded IFDS reachability closure over a million-node skewed exploded-supergraph CSR with resident frontier accumulation".to_string(),
            tags: vec![
                "dataflow".to_string(),
                "ifds".to_string(),
                "graph".to_string(),
                "csr".to_string(),
                "bitset".to_string(),
                "closure".to_string(),
                "skewed-degree".to_string(),
                "irregular".to_string(),
                "resident".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Libs,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-primitives".to_string(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: Some(96 * 1024 * 1024),
            min_input_bytes: Some(u64::from(NODE_COUNT) * 20),
            feature_set: vec![
                "dataflow".to_string(),
                "ifds".to_string(),
                "skewed-csr".to_string(),
                "resident-frontier".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<DataflowIfdsSkewedClosurePrepared>()
            .map(|prepared| {
                (
                    prepared.input_bytes_total,
                    prepared
                        .baseline_outputs
                        .iter()
                        .map(Vec::len)
                        .sum::<usize>() as u64,
                )
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(prepare_ifds_skewed_closure(Some(ctx))?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<DataflowIfdsSkewedClosurePrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<DataflowIfdsSkewedClosurePrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared IFDS closure payload had the wrong type".to_string(),
                )
            })?;

        let mut dispatch_config = ctx.dispatch_config.clone();
        dispatch_config.fixpoint_iterations = Some(CLOSURE_MAX_ITERS);
        dispatch_config
            .workgroup_override
            .get_or_insert(CLOSURE_WORKGROUP_SIZE);
        let workgroup = dispatch_config
            .workgroup_override
            .unwrap_or_else(|| prepared.program.workgroup_size());
        if workgroup.contains(&0) {
            return Err(BenchError::ExecutionFailed(format!(
                "IFDS closure benchmark received invalid workgroup {:?}. Fix: use positive dispatch dimensions.",
                workgroup
            )));
        }
        dispatch_config.grid_override.get_or_insert([
            prepared.stats.nodes.div_ceil(workgroup[0]),
            1,
            1,
        ]);

        let resident_reset_bytes = prepared
            .resident
            .as_ref()
            .map(|resident| reset_resident_closure_inputs(resident, &prepared.seed_frontier_bytes))
            .transpose()?
            .unwrap_or(0);
        let dispatch = dispatch_program_timed(
            ctx,
            &prepared.program,
            prepared.resident.as_ref(),
            &prepared.inputs,
            &dispatch_config,
        )?;
        let resident_used = dispatch.resident_used;
        let timed = dispatch.timed;
        let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let accounting = closure_transfer_accounting(
            prepared.input_bytes_total,
            output_bytes,
            resident_used,
            resident_reset_bytes,
        );

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(timed.wall_ns),
                dispatch_ns: timed.device_ns,
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
                    timed.wall_ns,
                    resident_used,
                    resident_reset_bytes,
                    CLOSURE_MAX_ITERS,
                    workgroup[0],
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
                    CLOSURE_MAX_ITERS,
                ),
                ..Default::default()
            }),
            outputs: timed.outputs,
            baseline_outputs: Some(prepared.baseline_outputs.clone()),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

pub(super) fn prepare_ifds_skewed_closure(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedClosurePrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let shape = ProgramGraphShape::new(fixture.stats.nodes, fixture.stats.edges);
    let program =
        csr_forward_or_changed_parallel(shape, "frontier_accumulator", "changed", IFDS_REACH_MASK);

    let baseline_start = std::time::Instant::now();
    let oracle = ifds_skewed_closure_oracle(&fixture, CLOSURE_MAX_ITERS);
    let baseline_wall_ns = baseline_start
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let mut stats = fixture.stats;
    stats.output_words_set = oracle.output_words_set;

    let inputs = ifds_closure_inputs(&fixture);
    let seed_frontier_bytes = inputs[FRONTIER_RESOURCE_INDEX].clone();
    let input_bytes_total = input_bytes_total(&inputs);
    let baseline_outputs = vec![
        vyre_primitives::wire::pack_u32_slice(&oracle.output),
        vyre_primitives::wire::pack_u32_slice(&[oracle.changed]),
    ];
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, "dataflow IFDS closure"))
        .transpose()?
        .flatten();

    Ok(DataflowIfdsSkewedClosurePrepared {
        program,
        inputs,
        input_bytes_total,
        seed_frontier_bytes,
        baseline_outputs,
        baseline_wall_ns,
        stats,
        closure_iterations: oracle.iterations,
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
    let bytes_read = if resident_used {
        resident_reset_bytes
    } else {
        input_bytes_total
    };
    TransferAccounting {
        bytes_touched: bytes_read.saturating_add(output_bytes_total),
        bytes_read,
        bytes_written: output_bytes_total,
    }
}

fn reset_resident_closure_inputs(
    resident: &ResidentInputSet,
    seed_frontier_bytes: &[u8],
) -> Result<u64, BenchError> {
    resident.upload_resource(
        FRONTIER_RESOURCE_INDEX,
        seed_frontier_bytes,
        "IFDS closure frontier reset",
    )?;
    resident.upload_resource(
        CHANGED_RESOURCE_INDEX,
        ZERO_U32_BYTES,
        "IFDS closure changed reset",
    )?;
    Ok(seed_frontier_bytes.len() as u64 + ZERO_U32_BYTES.len() as u64)
}

inventory::submit! {
    &DataflowIfdsSkewedClosure as &'static dyn BenchCase
}
