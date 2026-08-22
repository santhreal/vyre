use crate::api::case::BenchError;
use crate::api::suite::SuiteKind;
use crate::cases::mix32;
use crate::cases::queue_closure_oracle::{
    queue_closure_oracle, QueueClosureGraph, QueueClosureOracle,
};
use crate::cases::skewed_graph::{
    active_high_degree_sources, build_skewed_csr_arrays, skewed_degree as shared_skewed_degree,
    sparse_queue_capacity, SkewedCsrShape,
};

pub(super) const CSR_NODE_COUNT: u32 = 1_048_576;
pub(super) const CSR_ALLOW_MASK: u32 = 0b0111;
pub(super) const HIGH_DEGREE_THRESHOLD: u32 = 24;
pub(super) const UGLY_HUB_DEGREE: u32 = 2_048;
pub(super) const SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SkewedCsrStats {
    pub(super) node_count: u32,
    pub(super) edge_count: u32,
    pub(super) frontier_words: u32,
    pub(super) active_sources: u64,
    pub(super) allowed_edges_from_active: u64,
    pub(super) output_words_set: u64,
    pub(super) max_degree: u32,
    pub(super) high_degree_sources: u64,
}

impl crate::cases::queue_materialize::FrontierWords for SkewedCsrStats {
    fn frontier_words(&self) -> u32 {
        self.frontier_words
    }
}

pub(super) struct SkewedCsrFixture {
    pub(super) nodes: Vec<u32>,
    pub(super) edge_offsets: Vec<u32>,
    pub(super) edge_targets: Vec<u32>,
    pub(super) edge_kind_mask: Vec<u32>,
    pub(super) node_tags: Vec<u32>,
    pub(super) frontier_in: Vec<u32>,
    pub(super) frontier_out_seed: Vec<u32>,
    pub(super) stats: SkewedCsrStats,
}

pub(super) struct SkewedCsrOracle {
    pub(super) output: Vec<u32>,
    pub(super) allowed_edges_from_active: u64,
    pub(super) output_words_set: u64,
}

pub(super) fn build_skewed_csr_fixture(node_count: u32) -> Result<SkewedCsrFixture, BenchError> {
    let arrays = build_skewed_csr_arrays(&SkewedCsrShape {
        node_count,
        hub_degree: UGLY_HUB_DEGREE,
        high_degree_threshold: HIGH_DEGREE_THRESHOLD,
        fixture: "skewed CSR fixture",
        power_of_two_fix:
            "Fix: choose a power-of-two graph size so target generation stays branch-free.",
        node_kind: skewed_node_kind,
        node_tag: skewed_node_tag,
        edge_kind: skewed_edge_kind,
        source_is_active,
    })?;

    Ok(SkewedCsrFixture {
        nodes: arrays.nodes,
        edge_offsets: arrays.edge_offsets,
        edge_targets: arrays.edge_targets,
        edge_kind_mask: arrays.edge_kind_mask,
        node_tags: arrays.node_tags,
        frontier_in: arrays.frontier_in,
        frontier_out_seed: vec![0_u32; arrays.frontier_words as usize],
        stats: SkewedCsrStats {
            node_count,
            edge_count: arrays.edge_count,
            frontier_words: arrays.frontier_words,
            active_sources: arrays.active_sources,
            max_degree: arrays.max_degree,
            high_degree_sources: arrays.high_degree_sources,
            ..Default::default()
        },
    })
}

fn skewed_node_kind(src: u32) -> u32 {
    mix32(src) & 0x1F
}

pub(super) fn skewed_csr_inputs(fixture: &SkewedCsrFixture) -> Vec<Vec<u8>> {
    crate::cases::byte_pack::u32_input_bytes([
        &fixture.nodes,
        &fixture.edge_offsets,
        &fixture.edge_targets,
        &fixture.edge_kind_mask,
        &fixture.node_tags,
        &fixture.frontier_in,
        &fixture.frontier_out_seed,
    ])
}

pub(super) fn skewed_csr_queue_capacity(active_sources: u64) -> Result<u32, BenchError> {
    sparse_queue_capacity(
        active_sources,
        "skewed CSR queue benchmark requires at least one active source. Fix: seed the frontier before queue sizing.",
        "skewed CSR",
    )
}

crate::cases::queue_stage::define_queue_input_builder!(
    pub(super) skewed_csr_queue_inputs,
    SkewedCsrFixture,
    "skewed CSR queue inputs"
);

pub(super) fn skewed_csr_active_high_degree_sources(
    fixture: &SkewedCsrFixture,
    min_degree: u32,
) -> Result<u32, BenchError> {
    active_high_degree_sources(
        &fixture.frontier_in,
        &fixture.edge_offsets,
        fixture.stats.node_count,
        min_degree,
        "skewed CSR split queue",
    )
}

pub(super) fn materialize_skewed_csr_active_queue(
    fixture: &SkewedCsrFixture,
    queue_capacity: usize,
    context: &str,
) -> Result<Vec<u32>, BenchError> {
    crate::cases::skewed_graph::materialize_active_frontier_queue(
        &fixture.frontier_in,
        fixture.stats.node_count,
        fixture.stats.active_sources,
        queue_capacity,
        context,
    )
}

pub(super) fn skewed_csr_queue_closure_inputs(
    fixture: &SkewedCsrFixture,
    queue_capacity: u32,
) -> Result<Vec<Vec<u8>>, BenchError> {
    crate::cases::queue_stage::build_queue_closure_inputs(
        &fixture.frontier_in,
        &fixture.edge_offsets,
        &fixture.edge_targets,
        &fixture.edge_kind_mask,
        fixture.stats.active_sources,
        queue_capacity,
        "skewed CSR",
        |capacity| {
            materialize_skewed_csr_active_queue(fixture, capacity, "skewed CSR queue closure seed")
        },
    )
}

pub(super) fn skewed_csr_cpu_oracle(fixture: &SkewedCsrFixture) -> SkewedCsrOracle {
    let node_count = fixture.stats.node_count;
    let mut output = fixture.frontier_out_seed.clone();
    let mut allowed_edges_from_active = 0_u64;

    for src in 0..node_count {
        let src_word = (src / 32) as usize;
        let src_bit = 1_u32 << (src % 32);
        if (fixture.frontier_in[src_word] & src_bit) == 0 {
            continue;
        }
        let edge_start = fixture.edge_offsets[src as usize] as usize;
        let edge_end = fixture.edge_offsets[src as usize + 1] as usize;
        for edge in edge_start..edge_end {
            if (fixture.edge_kind_mask[edge] & CSR_ALLOW_MASK) == 0 {
                continue;
            }
            allowed_edges_from_active += 1;
            let dst = fixture.edge_targets[edge];
            if dst < node_count {
                output[(dst / 32) as usize] |= 1_u32 << (dst % 32);
            }
        }
    }

    SkewedCsrOracle {
        output_words_set: output.iter().filter(|word| **word != 0).count() as u64,
        output,
        allowed_edges_from_active,
    }
}

pub(super) fn skewed_csr_queue_closure_oracle(
    fixture: &SkewedCsrFixture,
    max_iters: u32,
    queue_capacity: u32,
) -> Result<QueueClosureOracle, BenchError> {
    let seed_queue = materialize_skewed_csr_active_queue(
        fixture,
        queue_capacity as usize,
        "skewed CSR queue closure oracle",
    )?;
    queue_closure_oracle(
        QueueClosureGraph {
            node_count: fixture.stats.node_count,
            edge_offsets: &fixture.edge_offsets,
            edge_targets: &fixture.edge_targets,
            edge_kind_mask: &fixture.edge_kind_mask,
            frontier_in: &fixture.frontier_in,
            seed_queue,
            allow_mask: CSR_ALLOW_MASK,
        },
        max_iters,
        queue_capacity,
        "skewed CSR",
        "raise the closure wave bound",
    )
}

fn skewed_degree(src: u32) -> u32 {
    shared_skewed_degree(src, UGLY_HUB_DEGREE)
}

fn skewed_edge_kind(src: u32, edge: u32) -> u32 {
    1_u32 << (mix32(src ^ edge.wrapping_mul(0xA5A5_9651)) & 3)
}

fn skewed_node_tag(src: u32) -> u32 {
    let base = 1_u32 << (mix32(src ^ 0xC001_D00D) & 7);
    if src % 4096 == 0 {
        base | 0x80
    } else {
        base
    }
}

fn source_is_active(src: u32) -> bool {
    src % 97 == 0 || src % 4096 == 0 || (mix32(src ^ 0xD1B5_4A32) & 0x3FF) == 0
}
