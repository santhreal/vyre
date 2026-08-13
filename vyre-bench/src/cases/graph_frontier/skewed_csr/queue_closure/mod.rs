use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, PreparedCase, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::resident::{input_bytes_total, ResidentInputSet};
use crate::cases::queue_closure_profile::validate_queue_closure_wave_profile;
use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_frontier_queue::frontier_queue_len_init;
use vyre_primitives::graph::csr_queue_delta::{
    csr_queue_delta_enqueue, csr_queue_delta_strided_dispatch_grid,
    csr_queue_delta_strided_enqueue, CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE,
};

use super::queue_materialize::graph_queue_should_use_row_strided;
use super::support::{
    build_skewed_csr_fixture, skewed_csr_queue_closure_inputs, skewed_csr_queue_closure_oracle,
    SkewedCsrStats, CSR_ALLOW_MASK, CSR_NODE_COUNT, SUITES,
};

mod metrics;

use metrics::{queue_closure_baseline_metric_points, queue_closure_metric_points};
use crate::cases::queue_stage::{QueueClosureSequenceRun, ResidentQueueClosureSpec};

pub(super) const GRAPH_QUEUE_CLOSURE_MAX_ITERS: u32 = 128;
pub(super) const GRAPH_QUEUE_CLOSURE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];


pub(super) struct GraphCsrSkewedQueueClosurePrepared {
    pub(super) reset_program: Program,
    pub(super) clear_len_program: Program,
    pub(super) delta_program: Program,
    pub(super) delta_grid: [u32; 3],
    pub(super) row_strided_delta: bool,
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) input_bytes_total: u64,
    pub(super) baseline_output: Vec<u8>,
    pub(super) baseline_wall_ns: u64,
    pub(super) stats: SkewedCsrStats,
    pub(super) queue_capacity: u32,
    pub(super) seed_queue_len: u32,
    pub(super) closure_iterations: u32,
    pub(super) closure_changed: u32,
    pub(super) total_queue_pops: u64,
    pub(super) max_wave_queue_len: u32,
    pub(super) wave_queue_lengths: Vec<u32>,
    pub(super) resident: Option<ResidentInputSet>,
}

pub(super) fn graph_queue_closure_delta_lanes_per_source(row_strided_delta: bool) -> u32 {
    if row_strided_delta {
        CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE
    } else {
        1
    }
}

struct GraphCsrSkewedQueueClosure;

impl BenchCase for GraphCsrSkewedQueueClosure {
    fn id(&self) -> BenchId {
        BenchId("primitives.graph.csr_skewed_queue_closure.1m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Skewed CSR Queue Closure 1M".to_string(),
            description: "Sparse-delta reachability closure over a million-node skewed CSR graph using GPU-resident ping-pong active queues".to_string(),
            tags: vec![
                "graph".to_string(),
                "frontier".to_string(),
                "csr".to_string(),
                "frontier-queue".to_string(),
                "delta-queue".to_string(),
                "closure".to_string(),
                "skewed-degree".to_string(),
                "irregular".to_string(),
                "resident".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Foundation,
            workload: WorkloadClass::Macro,
            determinism: crate::api::case::DeterminismClass::Deterministic,
            owner_crate: "vyre-primitives".to_string(),
        }
    }

    fn suites(&self) -> &'static [crate::api::suite::SuiteKind] {
        SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: Some(128 * 1024 * 1024),
            min_input_bytes: Some(u64::from(CSR_NODE_COUNT) * 16),
            feature_set: vec![
                "graph.csr".to_string(),
                "graph.frontier.queue".to_string(),
                "graph.delta-queue".to_string(),
                "graph.skewed-degree".to_string(),
                "resident-repeated-sequence".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<GraphCsrSkewedQueueClosurePrepared>()
            .map(|prepared| {
                (
                    prepared.input_bytes_total,
                    prepared.baseline_output.len() as u64,
                )
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(prepare_skewed_csr_queue_closure(Some(ctx))?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<GraphCsrSkewedQueueClosurePrepared>()
            .map(|prepared| &prepared.delta_program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<GraphCsrSkewedQueueClosurePrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared skewed CSR queue-closure payload had the wrong type".to_string(),
                )
            })?;
        if ctx.dispatch_config.workgroup_override.is_some() {
            return Err(BenchError::ExecutionFailed(
                "skewed CSR queue closure uses mixed workgroups across reset, clear, and delta kernels. Fix: run without a workgroup override."
                    .to_string(),
            ));
        }
        let resident = prepared.resident.as_ref().ok_or_else(|| {
            BenchError::EnvironmentInvalid(
                "skewed CSR queue closure requires resident GPU buffers. Fix: run on a backend with resident repeated-sequence support."
                    .to_string(),
            )
        })?;

        let sequence = dispatch_resident_queue_closure_sequence(ctx, prepared, resident)?;
        let output_bytes = sequence.outputs.iter().map(Vec::len).sum::<usize>() as u64;

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(sequence.wall_ns),
                dispatch_ns: None,
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(output_bytes),
                bytes_read: Some(0),
                bytes_written: Some(output_bytes),
                bytes_touched: Some(output_bytes),
                custom: queue_closure_metric_points(prepared, sequence.wall_ns, true),
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(prepared.baseline_wall_ns),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(prepared.baseline_output.len() as u64),
                custom: queue_closure_baseline_metric_points(prepared),
                ..Default::default()
            }),
            outputs: sequence.outputs,
            baseline_outputs: Some(vec![prepared.baseline_output.clone()]),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

pub(super) fn prepare_skewed_csr_queue_closure(
    ctx: Option<&BenchContext>,
) -> Result<GraphCsrSkewedQueueClosurePrepared, BenchError> {
    let fixture = build_skewed_csr_fixture(CSR_NODE_COUNT)?;
    let seed_queue_len = u32::try_from(fixture.stats.active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "skewed CSR queue closure active source count {} exceeds u32 indexing. Fix: split the seed queue.",
            fixture.stats.active_sources
        ))
    })?;
    let baseline_start = Instant::now();
    let oracle = skewed_csr_queue_closure_oracle(
        &fixture,
        GRAPH_QUEUE_CLOSURE_MAX_ITERS,
        fixture.stats.node_count,
    )?;
    let baseline_wall_ns = baseline_start
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let queue_capacity = oracle.max_wave_queue_len.max(seed_queue_len).max(1);
    validate_queue_closure_wave_profile(
        "skewed CSR",
        &oracle.wave_queue_lengths,
        oracle.iterations,
        oracle.total_queue_pops,
        oracle.max_wave_queue_len,
        queue_capacity,
    )?;
    let reset_program = graph_queue_closure_reset_program(
        fixture.stats.frontier_words,
        seed_queue_len,
        queue_capacity,
    );
    let clear_len_program = frontier_queue_len_init("queue_len");
    let row_strided_delta = graph_queue_should_use_row_strided(fixture.stats.max_degree);
    let (delta_program, delta_grid) = if row_strided_delta {
        (
            csr_queue_delta_strided_enqueue(
                "active_queue",
                "active_len",
                "edge_offsets",
                "edge_targets",
                "edge_kind_mask",
                "accumulator",
                "next_queue",
                "next_len",
                fixture.stats.node_count,
                fixture.stats.edge_count,
                queue_capacity,
                queue_capacity,
                CSR_ALLOW_MASK,
            ),
            csr_queue_delta_strided_dispatch_grid(queue_capacity),
        )
    } else {
        (
            csr_queue_delta_enqueue(
                "active_queue",
                "active_len",
                "edge_offsets",
                "edge_targets",
                "edge_kind_mask",
                "accumulator",
                "next_queue",
                "next_len",
                fixture.stats.node_count,
                fixture.stats.edge_count,
                queue_capacity,
                queue_capacity,
                CSR_ALLOW_MASK,
            ),
            [
                queue_capacity
                    .div_ceil(GRAPH_QUEUE_CLOSURE_WORKGROUP_SIZE[0])
                    .max(1),
                1,
                1,
            ],
        )
    };

    let mut stats = fixture.stats;
    stats.output_words_set = oracle.output.iter().filter(|word| **word != 0).count() as u64;
    let inputs = skewed_csr_queue_closure_inputs(&fixture, queue_capacity)?;
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, "skewed CSR queue closure"))
        .transpose()?
        .flatten();

    Ok(GraphCsrSkewedQueueClosurePrepared {
        reset_program,
        clear_len_program,
        delta_program,
        delta_grid,
        row_strided_delta,
        inputs,
        input_bytes_total,
        baseline_output: vyre_primitives::wire::pack_u32_slice(&oracle.output),
        baseline_wall_ns,
        stats,
        queue_capacity,
        seed_queue_len,
        closure_iterations: oracle.iterations,
        closure_changed: oracle.changed,
        total_queue_pops: oracle.total_queue_pops,
        max_wave_queue_len: oracle.max_wave_queue_len,
        wave_queue_lengths: oracle.wave_queue_lengths,
        resident,
    })
}

fn graph_queue_closure_reset_program(
    frontier_words: u32,
    seed_queue_len: u32,
    queue_capacity: u32,
) -> Program {
    crate::cases::queue_stage::build_queue_closure_reset_program(
        frontier_words,
        seed_queue_len,
        queue_capacity,
        GRAPH_QUEUE_CLOSURE_WORKGROUP_SIZE,
    )
}

inventory::submit! {
    &GraphCsrSkewedQueueClosure as &'static dyn BenchCase
}

fn dispatch_resident_queue_closure_sequence(
    ctx: &BenchContext,
    prepared: &GraphCsrSkewedQueueClosurePrepared,
    resident: &ResidentInputSet,
) -> Result<QueueClosureSequenceRun, BenchError> {
    crate::cases::queue_stage::dispatch_resident_queue_closure_sequence(
        ctx,
        ResidentQueueClosureSpec {
            reset_program: &prepared.reset_program,
            clear_len_program: &prepared.clear_len_program,
            delta_program: &prepared.delta_program,
            frontier_words: prepared.stats.frontier_words,
            seed_queue_len: prepared.seed_queue_len,
            baseline_output_len: prepared.baseline_output.len(),
            closure_iterations: prepared.closure_iterations,
            delta_grid: prepared.delta_grid,
            workgroup: GRAPH_QUEUE_CLOSURE_WORKGROUP_SIZE,
            context: "skewed CSR queue closure",
        },
        resident,
    )
}
