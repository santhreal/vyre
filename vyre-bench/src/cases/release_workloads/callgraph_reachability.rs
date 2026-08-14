//! Callgraph reachability step release case over a CSR graph, with the linear graph
//! generator, CPU forward baseline, and witness digest.

use super::registration::{gpu_requirements, RELEASE_SUITES};
use super::run_assembly::{
    bench_run_from_timed_with_accounting, encode_u32_words, resident_reset_transfer_accounting,
};
use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::metric::MetricPoint;
use crate::api::resident::{input_bytes_total, ResidentInputPool};
use vyre::ir::Program;
use vyre_primitives::graph::program_graph::ProgramGraphShape;

pub struct CallgraphReachabilityStep;

struct CallgraphPrepared {
    program: Program,
    graph: GraphInputs,
    input_bytes_total: u64,
    output_resource_index: usize,
    resident_batch: Option<ResidentInputPool>,
}

const CALLGRAPH_NODES: u32 = 262_144;

const CALLGRAPH_RESIDENT_BATCH_SIZE: usize = 16;

const CALLGRAPH_EDGES: u32 = CALLGRAPH_NODES - 1;

const CALLGRAPH_WORDS: usize = CALLGRAPH_NODES.div_ceil(32) as usize;

impl BenchCase for CallgraphReachabilityStep {
    fn id(&self) -> BenchId {
        BenchId("callgraph.reachability.step.262k".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Callgraph Reachability Step 262K".to_string(),
            description: "Graph reachability step over a callgraph-shaped CSR workload".to_string(),
            tags: vec![
                "callgraph".to_string(),
                "reachability".to_string(),
                "graph".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Libs,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-primitives".to_string(),
        }
    }

    fn suites(&self) -> &'static [crate::api::suite::SuiteKind] {
        RELEASE_SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        gpu_requirements(graph_input_bytes().saturating_add((CALLGRAPH_WORDS * 4) as u64))
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_min_speedup(
            "callgraph reachability CSR step",
            "vyre-primitives",
            "optimized CPU graph reachability and witness extraction",
            25.0,
        ))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let shape = ProgramGraphShape::new(CALLGRAPH_NODES, CALLGRAPH_EDGES);
        let program = vyre_primitives::graph::csr_forward_traverse::csr_forward_traverse(
            shape,
            "frontier_in",
            "frontier_out",
            1,
        );
        let graph = linear_graph_inputs();
        let input_bytes_total = input_bytes_total(&graph.inputs);
        let output_resource_index = program
            .buffers()
            .iter()
            .position(|buffer| buffer.name() == "frontier_out")
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "callgraph traversal program is missing frontier_out binding".to_string(),
                )
            })?;
        let resident_batch = ResidentInputPool::upload_optional(
            ctx,
            &graph.inputs,
            CALLGRAPH_RESIDENT_BATCH_SIZE,
            "callgraph reachability batch",
        )?;

        Ok(Box::new(CallgraphPrepared {
            program,
            graph,
            input_bytes_total,
            output_resource_index,
            resident_batch,
        }))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<CallgraphPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<CallgraphPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed("callgraph prepared payload type mismatch".to_string())
            })?;
        let reset_payload = prepared
            .graph
            .inputs
            .get(prepared.output_resource_index)
            .ok_or_else(|| {
                BenchError::ExecutionFailed(format!(
                    "callgraph output resource index {} is outside {} input payloads",
                    prepared.output_resource_index,
                    prepared.graph.inputs.len()
                ))
            })?;
        let mut batch_wall_ns = None;
        let mut batch_len = None;
        let (timed, resident_used, resident_reset_bytes) =
            if let Some(resident_batch) = prepared.resident_batch.as_ref() {
                resident_batch.upload_resource_to_all_sets(
                    prepared.output_resource_index,
                    reset_payload,
                    "callgraph resident batch frontier reset",
                )?;
                let config = crate::api::case::dispatch_config_with_inferred_grid(
                    &prepared.program,
                    &prepared.graph.inputs,
                    &ctx.dispatch_config,
                )
                .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
                match resident_batch.dispatch_artifact_batch_timed(
                    ctx,
                    &prepared.program,
                    CALLGRAPH_RESIDENT_BATCH_SIZE,
                    &config,
                ) {
                    Ok(batch) => {
                        if batch.outputs.len() != CALLGRAPH_RESIDENT_BATCH_SIZE {
                            return Err(BenchError::ExecutionFailed(format!(
                                "callgraph resident batch returned {} output row(s), expected {}",
                                batch.outputs.len(),
                                CALLGRAPH_RESIDENT_BATCH_SIZE
                            )));
                        }
                        let first_outputs = batch.outputs.first().cloned().ok_or_else(|| {
                            BenchError::ExecutionFailed(
                                "callgraph resident batch returned no output rows".to_string(),
                            )
                        })?;
                        if let Some((index, _)) = batch
                            .outputs
                            .iter()
                            .enumerate()
                            .find(|(_, outputs)| **outputs != first_outputs)
                        {
                            return Err(BenchError::CorrectnessViolation(format!(
                                "callgraph resident batch output row {index} disagreed with row 0"
                            )));
                        }
                        batch_wall_ns = Some(batch.wall_ns_total);
                        batch_len = Some(batch.batch_len as u64);
                        (
                            vyre_driver::TimedDispatchResult {
                                outputs: first_outputs,
                                wall_ns: batch.per_item_wall_ns(),
                                device_ns: batch.per_item_device_ns(),
                                enqueue_ns: None,
                                wait_ns: None,
                            },
                            true,
                            reset_payload.len() as u64,
                        )
                    }
                    Err(vyre_driver::BackendError::UnsupportedFeature { .. }) => {
                        let timed = ctx
                            .dispatch_timed(
                                &prepared.program,
                                &prepared.graph.inputs,
                                &ctx.dispatch_config,
                            )
                            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
                        (timed, false, 0)
                    }
                    Err(error) => return Err(BenchError::BackendFailed(error.to_string())),
                }
            } else {
                let timed = ctx
                    .dispatch_timed(
                        &prepared.program,
                        &prepared.graph.inputs,
                        &ctx.dispatch_config,
                    )
                    .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
                (timed, false, 0)
            };

        let graph = &prepared.graph;
        let baseline_start = std::time::Instant::now();
        let mut expected = release_benchmark_csr_forward_baseline(
            CALLGRAPH_NODES,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
            &graph.frontier_in,
            1,
        );
        let witness_digest = callgraph_witness_digest(
            CALLGRAPH_NODES,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
            &graph.frontier_in,
            1,
        );
        for (out, seed) in expected.iter_mut().zip(graph.frontier_out_seed.iter()) {
            *out |= *seed;
        }
        let baseline_outputs = vec![encode_u32_words(&expected)];
        let baseline_wall = baseline_start.elapsed().as_nanos() as u64;
        let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let logical_bytes_touched = prepared.input_bytes_total.saturating_add(output_bytes);
        let accounting = resident_reset_transfer_accounting(
            prepared.input_bytes_total,
            output_bytes,
            resident_used,
            resident_reset_bytes,
        );
        let mut run = bench_run_from_timed_with_accounting(
            timed,
            prepared.input_bytes_total,
            baseline_outputs,
            baseline_wall,
            "callgraph_nodes",
            CALLGRAPH_NODES,
            logical_bytes_touched,
            accounting,
        )?;
        run.metrics.custom.push(MetricPoint {
            name: "callgraph_witness_digest".to_string(),
            value: u64::from(witness_digest),
        });
        run.metrics.custom.push(MetricPoint {
            name: "callgraph_resident_buffers".to_string(),
            value: u64::from(resident_used),
        });
        run.metrics.custom.push(MetricPoint {
            name: "callgraph_resident_reset_bytes".to_string(),
            value: resident_reset_bytes,
        });
        if let Some(wall_ns) = batch_wall_ns {
            run.metrics.custom.push(MetricPoint {
                name: "callgraph_resident_batch_wall_ns".to_string(),
                value: wall_ns,
            });
        }
        if let Some(len) = batch_len {
            run.metrics.custom.push(MetricPoint {
                name: "callgraph_resident_batch_len".to_string(),
                value: len,
            });
        }
        Ok(run)
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

struct GraphInputs {
    inputs: Vec<Vec<u8>>,
    edge_offsets: Vec<u32>,
    edge_targets: Vec<u32>,
    edge_kind_mask: Vec<u32>,
    frontier_in: Vec<u32>,
    frontier_out_seed: Vec<u32>,
}

fn linear_graph_inputs() -> GraphInputs {
    let nodes = vec![0; CALLGRAPH_NODES as usize];
    let mut edge_offsets = Vec::with_capacity(CALLGRAPH_NODES as usize + 1);
    for node in 0..CALLGRAPH_NODES {
        edge_offsets.push(node.min(CALLGRAPH_EDGES));
    }
    edge_offsets.push(CALLGRAPH_EDGES);
    let edge_targets: Vec<u32> = (1..CALLGRAPH_NODES).collect();
    let edge_kind_mask = vec![1; CALLGRAPH_EDGES as usize];
    let node_tags = vec![0; CALLGRAPH_NODES as usize];
    let mut frontier_in = vec![u32::MAX; CALLGRAPH_WORDS];
    let extra_bits = (CALLGRAPH_WORDS as u32 * 32).saturating_sub(CALLGRAPH_NODES);
    if extra_bits > 0 {
        let live_bits = 32 - extra_bits;
        if let Some(last) = frontier_in.last_mut() {
            *last = (1u32 << live_bits) - 1;
        }
    }
    let frontier_out_seed = vec![0; CALLGRAPH_WORDS];
    let inputs = vec![
        encode_u32_words(&nodes),
        encode_u32_words(&edge_offsets),
        encode_u32_words(&edge_targets),
        encode_u32_words(&edge_kind_mask),
        encode_u32_words(&node_tags),
        encode_u32_words(&frontier_in),
        encode_u32_words(&frontier_out_seed),
    ];
    GraphInputs {
        inputs,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        frontier_out_seed,
    }
}

fn graph_input_bytes() -> u64 {
    ((CALLGRAPH_NODES as usize * 2
        + CALLGRAPH_NODES as usize
        + 1
        + CALLGRAPH_EDGES as usize * 2
        + CALLGRAPH_WORDS * 2)
        * 4) as u64
}

fn release_benchmark_csr_forward_baseline(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = node_count.div_ceil(32) as usize;
    let mut out = vec![0; words];
    let expected_offsets = node_count as usize + 1;
    assert_eq!(
        edge_offsets.len(),
        expected_offsets,
        "release benchmark CSR baseline received {} row offsets for node_count={node_count}; Fix: pass exactly node_count + 1 CSR offsets.",
        edge_offsets.len()
    );
    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    assert!(
        edge_targets.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "release benchmark CSR baseline received edge_count={edge_count} but targets_len={} kind_mask_len={}. Fix: pass complete CSR edge buffers.",
        edge_targets.len(),
        edge_kind_mask.len()
    );
    for (index, pair) in edge_offsets.windows(2).enumerate() {
        assert!(
            pair[0] <= pair[1],
            "release benchmark CSR baseline received non-monotonic CSR offsets at row {index}: {} > {}. Fix: rebuild CSR row pointers before collecting release evidence.",
            pair[0],
            pair[1]
        );
    }
    for src in 0..node_count {
        let src_word = (src / 32) as usize;
        let src_bit = 1u32 << (src % 32);
        if src_word >= frontier_in.len() || (frontier_in[src_word] & src_bit) == 0 {
            continue;
        }
        let edge_start = edge_offsets[src as usize] as usize;
        let edge_end = edge_offsets[src as usize + 1] as usize;
        for edge_index in edge_start..edge_end {
            if (edge_kind_mask[edge_index] & allow_mask) == 0 {
                continue;
            }
            let dst = edge_targets[edge_index];
            if dst < node_count {
                out[(dst / 32) as usize] |= 1u32 << (dst % 32);
            }
        }
    }
    out
}

fn callgraph_witness_digest(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> u32 {
    let mut digest = 0x811C_9DC5u32;
    for src in 0..node_count {
        let src_word = (src / 32) as usize;
        let src_bit = 1u32 << (src % 32);
        if src_word >= frontier_in.len() || (frontier_in[src_word] & src_bit) == 0 {
            continue;
        }
        let edge_start = edge_offsets[src as usize] as usize;
        let edge_end = edge_offsets[src as usize + 1] as usize;
        for edge_index in edge_start..edge_end {
            if (edge_kind_mask[edge_index] & allow_mask) == 0 {
                continue;
            }
            let dst = edge_targets[edge_index];
            if dst >= node_count {
                continue;
            }
            let mut witness = src
                .wrapping_mul(0x045D_9F3B)
                .wrapping_add(dst.rotate_left(7))
                .wrapping_add(edge_index as u32);
            for round in 0..12 {
                witness = witness
                    .rotate_left(5)
                    .wrapping_mul(0x85EB_CA6B)
                    .wrapping_add(0xC2B2_AE35 ^ round);
            }
            digest ^= witness;
            digest = digest.rotate_left(3).wrapping_mul(0x0100_0193);
        }
    }
    digest
}
