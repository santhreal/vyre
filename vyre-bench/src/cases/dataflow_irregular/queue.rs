use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, transfer_accounting, ResidentInputSet,
};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_frontier_queue::csr_queue_forward_traverse;

use super::fixture::{
    build_ifds_skewed_fixture, ifds_active_queue_inputs, ifds_skewed_cpu_oracle, IfdsSkewedStats,
    IFDS_REACH_MASK, NODE_COUNT,
};
use super::metrics::{ifds_queue_baseline_metric_points, ifds_queue_metric_points};
use super::SUITES;

mod materialize;
#[cfg(test)]
pub(super) use materialize::{
    ifds_queue_reset_program, prepare_ifds_skewed_queue_materialize_step, QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_FRONTIER_IN_INDEX, QUEUE_FRONTIER_OUT_INDEX, QUEUE_LEN_INDEX,
};

pub(super) const ACTIVE_QUEUE_ACTIVE_QUEUE_INDEX: usize = 0;
pub(super) const ACTIVE_QUEUE_LEN_INDEX: usize = 1;
pub(super) const ACTIVE_QUEUE_EDGE_OFFSETS_INDEX: usize = 2;
pub(super) const ACTIVE_QUEUE_EDGE_TARGETS_INDEX: usize = 3;
pub(super) const ACTIVE_QUEUE_EDGE_KIND_INDEX: usize = 4;
pub(super) const ACTIVE_QUEUE_FRONTIER_OUT_INDEX: usize = 5;

pub(super) struct DataflowIfdsSkewedActiveQueuePrepared {
    pub(super) traverse_program: Program,
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) input_bytes_total: u64,
    pub(super) baseline_output: Vec<u8>,
    pub(super) baseline_wall_ns: u64,
    pub(super) stats: IfdsSkewedStats,
    pub(super) queue_capacity: u32,
    pub(super) resident: Option<ResidentInputSet>,
}

pub(super) fn ifds_sparse_queue_capacity(active_sources: u64) -> Result<u32, BenchError> {
    if active_sources == 0 {
        return Err(BenchError::EnvironmentInvalid(
            "IFDS queue benchmark requires at least one active source. Fix: seed the frontier before queue sizing."
                .to_string(),
        ));
    }
    u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "IFDS queue active source count {active_sources} exceeds u32 indexing. Fix: split the frontier."
        ))
    })
}

/// Queue-driven IFDS step when the active frontier queue is already resident.
struct DataflowIfdsSkewedActiveQueueStep;

impl BenchCase for DataflowIfdsSkewedActiveQueueStep {
    fn id(&self) -> BenchId {
        BenchId("dataflow.ifds.skewed.queue_step.1m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Dataflow IFDS Skewed Active Queue Step 1M".to_string(),
            description: "One IFDS propagation step over a million-node skewed exploded-supergraph CSR from a pre-materialized GPU-resident active frontier queue".to_string(),
            tags: vec![
                "dataflow".to_string(),
                "ifds".to_string(),
                "graph".to_string(),
                "csr".to_string(),
                "frontier-queue".to_string(),
                "active-queue".to_string(),
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
            min_input_bytes: Some(u64::from(NODE_COUNT) * 12),
            feature_set: vec![
                "dataflow".to_string(),
                "ifds".to_string(),
                "skewed-csr".to_string(),
                "frontier-queue".to_string(),
                "active-queue".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<DataflowIfdsSkewedActiveQueuePrepared>()
            .map(|prepared| {
                (
                    prepared.input_bytes_total,
                    prepared.baseline_output.len() as u64,
                )
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(prepare_ifds_skewed_active_queue_step(Some(ctx))?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<DataflowIfdsSkewedActiveQueuePrepared>()
            .map(|prepared| &prepared.traverse_program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<DataflowIfdsSkewedActiveQueuePrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared IFDS active-queue payload had the wrong type".to_string(),
                )
            })?;

        let mut dispatch_config = ctx.dispatch_config.clone();
        let workgroup = dispatch_config
            .workgroup_override
            .unwrap_or_else(|| prepared.traverse_program.workgroup_size());
        if workgroup.contains(&0) {
            return Err(BenchError::ExecutionFailed(format!(
                "IFDS active-queue benchmark received invalid workgroup {:?}. Fix: use positive dispatch dimensions.",
                workgroup
            )));
        }
        dispatch_config.grid_override.get_or_insert([
            prepared.queue_capacity.div_ceil(workgroup[0]).max(1),
            1,
            1,
        ]);

        let dispatch = dispatch_program_timed(
            ctx,
            &prepared.traverse_program,
            prepared.resident.as_ref(),
            &prepared.inputs,
            &dispatch_config,
        )?;
        let resident_used = dispatch.resident_used;
        let timed = dispatch.timed;
        let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let accounting =
            transfer_accounting(prepared.input_bytes_total, output_bytes, resident_used);

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(timed.wall_ns),
                dispatch_ns: timed.device_ns,
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(output_bytes),
                bytes_read: Some(accounting.bytes_read),
                bytes_written: Some(accounting.bytes_written),
                bytes_touched: Some(accounting.bytes_touched),
                custom: ifds_queue_metric_points(
                    prepared.stats,
                    prepared.queue_capacity,
                    prepared.baseline_wall_ns,
                    timed.wall_ns,
                    resident_used,
                    workgroup[0],
                    false,
                ),
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(prepared.baseline_wall_ns),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(prepared.baseline_output.len() as u64),
                custom: ifds_queue_baseline_metric_points(prepared.stats, prepared.queue_capacity),
                ..Default::default()
            }),
            outputs: timed.outputs,
            baseline_outputs: Some(vec![prepared.baseline_output.clone()]),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

pub(super) fn prepare_ifds_skewed_active_queue_step(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedActiveQueuePrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let queue_capacity = ifds_sparse_queue_capacity(fixture.stats.active_sources)?;
    let traverse_program = csr_queue_forward_traverse(
        "active_queue",
        "queue_len",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "frontier_out",
        fixture.stats.nodes,
        fixture.stats.edges,
        queue_capacity,
        IFDS_REACH_MASK,
    );

    let baseline_start = Instant::now();
    let oracle = ifds_skewed_cpu_oracle(&fixture);
    let baseline_wall_ns = baseline_start
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let mut stats = fixture.stats;
    stats.allowed_edges_from_active = oracle.allowed_edges_from_active;
    stats.filtered_edges_from_active = oracle.filtered_edges_from_active;
    stats.output_words_set = oracle.output_words_set;

    let inputs = ifds_active_queue_inputs(&fixture, queue_capacity)?;
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, "dataflow IFDS active queue"))
        .transpose()?
        .flatten();

    Ok(DataflowIfdsSkewedActiveQueuePrepared {
        traverse_program,
        inputs,
        input_bytes_total,
        baseline_output: vyre_primitives::wire::pack_u32_slice(&oracle.output),
        baseline_wall_ns,
        stats,
        queue_capacity,
        resident,
    })
}

inventory::submit! {
    &DataflowIfdsSkewedActiveQueueStep as &'static dyn BenchCase
}
