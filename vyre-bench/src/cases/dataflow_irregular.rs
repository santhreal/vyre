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
use vyre_primitives::graph::csr_forward_traverse::csr_forward_traverse;
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::predicate::edge_kind;

#[cfg(test)]
mod tests;

mod metrics;
use metrics::{ifds_skewed_baseline_metric_points, ifds_skewed_metric_points};

const NODE_COUNT: u32 = 1_048_576;
const FRONTIER_WORDS: usize = NODE_COUNT.div_ceil(32) as usize;
const HIGH_DEGREE_THRESHOLD: u32 = 24;
const IFDS_REACH_MASK: u32 = edge_kind::ASSIGNMENT
    | edge_kind::CALL_ARG
    | edge_kind::RETURN
    | edge_kind::PHI
    | edge_kind::ALIAS
    | edge_kind::MEM_STORE
    | edge_kind::MEM_LOAD
    | edge_kind::MUT_REF;
const SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

struct DataflowIfdsSkewedPrepared {
    program: Program,
    inputs: Vec<Vec<u8>>,
    input_bytes_total: u64,
    baseline_output: Vec<u8>,
    baseline_wall_ns: u64,
    stats: IfdsSkewedStats,
    resident: Option<ResidentInputSet>,
}

#[derive(Clone, Copy, Debug, Default)]
struct IfdsSkewedStats {
    nodes: u32,
    edges: u32,
    frontier_words: u32,
    active_sources: u64,
    allowed_edges_from_active: u64,
    filtered_edges_from_active: u64,
    output_words_set: u64,
    max_degree: u32,
    high_degree_sources: u64,
}

struct IfdsSkewedFixture {
    nodes: Vec<u32>,
    edge_offsets: Vec<u32>,
    edge_targets: Vec<u32>,
    edge_kind_mask: Vec<u32>,
    node_tags: Vec<u32>,
    frontier_in: Vec<u32>,
    frontier_out_seed: Vec<u32>,
    stats: IfdsSkewedStats,
}

struct IfdsSkewedOracle {
    output: Vec<u32>,
    allowed_edges_from_active: u64,
    filtered_edges_from_active: u64,
    output_words_set: u64,
}

/// Skewed exploded-supergraph IFDS step with edge-kind filtering.
struct DataflowIfdsSkewedStep;

impl BenchCase for DataflowIfdsSkewedStep {
    fn id(&self) -> BenchId {
        BenchId("dataflow.ifds.skewed.step.1m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Dataflow IFDS Skewed Step 1M".to_string(),
            description: "One IFDS propagation step over a million-node skewed exploded-supergraph CSR with packed frontier bits and filtered edge kinds".to_string(),
            tags: vec![
                "dataflow".to_string(),
                "ifds".to_string(),
                "graph".to_string(),
                "csr".to_string(),
                "bitset".to_string(),
                "skewed-degree".to_string(),
                "irregular".to_string(),
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
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<DataflowIfdsSkewedPrepared>()
            .map(|prepared| {
                (
                    prepared.input_bytes_total,
                    prepared.baseline_output.len() as u64,
                )
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(prepare_ifds_skewed_step(Some(ctx))?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<DataflowIfdsSkewedPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<DataflowIfdsSkewedPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared IFDS skewed payload had the wrong type".to_string(),
                )
            })?;

        let mut dispatch_config = ctx.dispatch_config.clone();
        let workgroup = dispatch_config
            .workgroup_override
            .unwrap_or_else(|| prepared.program.workgroup_size());
        if workgroup.contains(&0) {
            return Err(BenchError::ExecutionFailed(format!(
                "IFDS skewed benchmark received invalid workgroup {:?}. Fix: use positive dispatch dimensions.",
                workgroup
            )));
        }
        dispatch_config.grid_override.get_or_insert([
            prepared.stats.nodes.div_ceil(workgroup[0]),
            1,
            1,
        ]);

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
                custom: ifds_skewed_metric_points(
                    prepared.stats,
                    prepared.baseline_wall_ns,
                    timed.wall_ns,
                    resident_used,
                    workgroup[0],
                ),
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(prepared.baseline_wall_ns),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(prepared.baseline_output.len() as u64),
                custom: ifds_skewed_baseline_metric_points(prepared.stats),
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

fn prepare_ifds_skewed_step(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedPrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let shape = ProgramGraphShape::new(fixture.stats.nodes, fixture.stats.edges);
    let program = csr_forward_traverse(shape, "frontier_in", "frontier_out", IFDS_REACH_MASK);

    let baseline_start = std::time::Instant::now();
    let oracle = ifds_skewed_cpu_oracle(&fixture);
    let baseline_wall_ns = baseline_start
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let mut stats = fixture.stats;
    stats.allowed_edges_from_active = oracle.allowed_edges_from_active;
    stats.filtered_edges_from_active = oracle.filtered_edges_from_active;
    stats.output_words_set = oracle.output_words_set;

    let inputs = ifds_skewed_inputs(&fixture);
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, "dataflow IFDS skewed"))
        .transpose()?
        .flatten();

    Ok(DataflowIfdsSkewedPrepared {
        program,
        inputs,
        input_bytes_total,
        baseline_output: vyre_primitives::wire::pack_u32_slice(&oracle.output),
        baseline_wall_ns,
        stats,
        resident,
    })
}

fn build_ifds_skewed_fixture(node_count: u32) -> Result<IfdsSkewedFixture, BenchError> {
    if !node_count.is_power_of_two() || node_count < 32 {
        return Err(BenchError::EnvironmentInvalid(format!(
            "IFDS skewed fixture requires a power-of-two node count >= 32, received {node_count}. Fix: choose a power-of-two exploded-supergraph size."
        )));
    }

    let words = node_count.div_ceil(32);
    let mut nodes = Vec::with_capacity(node_count as usize);
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::with_capacity((node_count as usize).saturating_mul(2));
    let mut edge_kind_mask = Vec::with_capacity((node_count as usize).saturating_mul(2));
    let mut node_tags = Vec::with_capacity(node_count as usize);
    let mut frontier_in = vec![0_u32; words as usize];

    let mut stats = IfdsSkewedStats {
        nodes: node_count,
        frontier_words: words,
        ..Default::default()
    };

    edge_offsets.push(0);
    for src in 0..node_count {
        let degree = skewed_degree(src);
        stats.max_degree = stats.max_degree.max(degree);
        if degree >= HIGH_DEGREE_THRESHOLD {
            stats.high_degree_sources += 1;
        }
        if source_is_active(src) {
            stats.active_sources += 1;
            frontier_in[(src / 32) as usize] |= 1_u32 << (src % 32);
        }
        nodes.push(ifds_node_kind(src));
        node_tags.push(ifds_node_tag(src));
        for edge in 0..degree {
            edge_targets.push(skewed_target(node_count, src, edge));
            edge_kind_mask.push(ifds_edge_kind(src, edge));
        }
        edge_offsets.push(u32::try_from(edge_targets.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "IFDS skewed fixture exceeded u32 edge offsets. Fix: split the benchmark graph."
                    .to_string(),
            )
        })?);
    }

    stats.edges = u32::try_from(edge_targets.len()).map_err(|_| {
        BenchError::EnvironmentInvalid(
            "IFDS skewed fixture exceeded u32 edge count. Fix: split the benchmark graph."
                .to_string(),
        )
    })?;

    Ok(IfdsSkewedFixture {
        nodes,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        node_tags,
        frontier_in,
        frontier_out_seed: vec![0_u32; words as usize],
        stats,
    })
}

fn ifds_skewed_inputs(fixture: &IfdsSkewedFixture) -> Vec<Vec<u8>> {
    vec![
        vyre_primitives::wire::pack_u32_slice(&fixture.nodes),
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_offsets),
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_targets),
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_kind_mask),
        vyre_primitives::wire::pack_u32_slice(&fixture.node_tags),
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_in),
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_out_seed),
    ]
}

fn ifds_skewed_cpu_oracle(fixture: &IfdsSkewedFixture) -> IfdsSkewedOracle {
    let mut output = fixture.frontier_out_seed.clone();
    let mut allowed_edges_from_active = 0_u64;
    let mut filtered_edges_from_active = 0_u64;

    for src in 0..fixture.stats.nodes {
        let src_word = (src / 32) as usize;
        let src_bit = 1_u32 << (src % 32);
        if (fixture.frontier_in[src_word] & src_bit) == 0 {
            continue;
        }
        let start = fixture.edge_offsets[src as usize] as usize;
        let end = fixture.edge_offsets[src as usize + 1] as usize;
        for edge in start..end {
            if (fixture.edge_kind_mask[edge] & IFDS_REACH_MASK) == 0 {
                filtered_edges_from_active += 1;
                continue;
            }
            allowed_edges_from_active += 1;
            let dst = fixture.edge_targets[edge];
            if dst < fixture.stats.nodes {
                output[(dst / 32) as usize] |= 1_u32 << (dst % 32);
            }
        }
    }

    IfdsSkewedOracle {
        output_words_set: output.iter().filter(|word| **word != 0).count() as u64,
        output,
        allowed_edges_from_active,
        filtered_edges_from_active,
    }
}

fn skewed_degree(src: u32) -> u32 {
    if src % 4096 == 0 {
        96
    } else if src % 257 == 0 {
        24
    } else if src % 31 == 0 {
        8
    } else if src % 7 == 0 {
        3
    } else {
        1
    }
}

fn skewed_target(node_count: u32, src: u32, edge: u32) -> u32 {
    let mask = node_count - 1;
    match edge & 7 {
        0 => src.wrapping_add((edge + 1).wrapping_mul(17)) & mask,
        1 => src.wrapping_sub((edge + 3).wrapping_mul(11)) & mask,
        _ => {
            let salt = edge.wrapping_mul(0x9E37_79B9).rotate_left((edge & 15) + 1);
            mix32(src ^ salt ^ src.rotate_left(edge & 15)) & mask
        }
    }
}

fn ifds_edge_kind(src: u32, edge: u32) -> u32 {
    const ALLOWED: [u32; 8] = [
        edge_kind::ASSIGNMENT,
        edge_kind::CALL_ARG,
        edge_kind::RETURN,
        edge_kind::PHI,
        edge_kind::ALIAS,
        edge_kind::MEM_STORE,
        edge_kind::MEM_LOAD,
        edge_kind::MUT_REF,
    ];
    match mix32(src ^ edge.wrapping_mul(0xA5A5_9651)) & 15 {
        0 => edge_kind::DOMINANCE,
        1 => edge_kind::CONTROL,
        value => ALLOWED[(value as usize - 2) % ALLOWED.len()],
    }
}

fn ifds_node_kind(src: u32) -> u32 {
    mix32(src ^ 0x1F_D5_0001) & 0x1F
}

fn ifds_node_tag(src: u32) -> u32 {
    let base = 1_u32 << (mix32(src ^ 0x51CE_7A6D) & 7);
    if src % 257 == 0 {
        base | 0x100
    } else {
        base
    }
}

fn source_is_active(src: u32) -> bool {
    src % 97 == 0 || src % 4096 == 0 || (mix32(src ^ 0xD1B5_4A32) & 0x3FF) == 0
}

fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}

inventory::submit! {
    &DataflowIfdsSkewedStep as &'static dyn BenchCase
}
