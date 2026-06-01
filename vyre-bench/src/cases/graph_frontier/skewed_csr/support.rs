use crate::api::case::BenchError;
use crate::api::suite::SuiteKind;

pub(super) const CSR_NODE_COUNT: u32 = 1_048_576;
pub(super) const CSR_ALLOW_MASK: u32 = 0b0111;
pub(super) const HIGH_DEGREE_THRESHOLD: u32 = 24;
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
    if !node_count.is_power_of_two() || node_count < 32 {
        return Err(BenchError::EnvironmentInvalid(format!(
            "skewed CSR fixture requires a power-of-two node count >= 32, received {node_count}. Fix: choose a power-of-two graph size so target generation stays branch-free."
        )));
    }

    let frontier_words = node_count.div_ceil(32);
    let mut nodes = Vec::with_capacity(node_count as usize);
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::with_capacity((node_count as usize).saturating_mul(2));
    let mut edge_kind_mask = Vec::with_capacity((node_count as usize).saturating_mul(2));
    let mut node_tags = Vec::with_capacity(node_count as usize);
    let mut frontier_in = vec![0_u32; frontier_words as usize];

    let mut stats = SkewedCsrStats {
        node_count,
        frontier_words,
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
        nodes.push(mix32(src) & 0x1F);
        node_tags.push(skewed_node_tag(src));
        for edge in 0..degree {
            edge_targets.push(skewed_target(node_count, src, edge));
            edge_kind_mask.push(skewed_edge_kind(src, edge));
        }
        let offset = u32::try_from(edge_targets.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "skewed CSR fixture exceeded u32 edge offsets. Fix: split the benchmark graph."
                    .to_string(),
            )
        })?;
        edge_offsets.push(offset);
    }

    stats.edge_count = u32::try_from(edge_targets.len()).map_err(|_| {
        BenchError::EnvironmentInvalid(
            "skewed CSR fixture exceeded u32 edge count. Fix: split the benchmark graph."
                .to_string(),
        )
    })?;

    Ok(SkewedCsrFixture {
        nodes,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        node_tags,
        frontier_in,
        frontier_out_seed: vec![0_u32; frontier_words as usize],
        stats,
    })
}

pub(super) fn skewed_csr_inputs(fixture: &SkewedCsrFixture) -> Vec<Vec<u8>> {
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

pub(super) fn skewed_csr_queue_capacity(active_sources: u64) -> Result<u32, BenchError> {
    if active_sources == 0 {
        return Err(BenchError::EnvironmentInvalid(
            "skewed CSR queue benchmark requires at least one active source. Fix: seed the frontier before queue sizing."
                .to_string(),
        ));
    }
    u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "skewed CSR active source count {active_sources} exceeds u32 indexing. Fix: split the frontier."
        ))
    })
}

pub(super) fn skewed_csr_queue_inputs(
    fixture: &SkewedCsrFixture,
    queue_capacity: u32,
) -> Result<Vec<Vec<u8>>, BenchError> {
    if u64::from(queue_capacity) < fixture.stats.active_sources {
        return Err(BenchError::EnvironmentInvalid(format!(
            "skewed CSR queue inputs require queue_capacity >= active_sources, got queue_capacity={queue_capacity} active_sources={}. Fix: size the queue from the packed frontier popcount.",
            fixture.stats.active_sources
        )));
    }
    Ok(vec![
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_in),
        vec![0_u8; queue_capacity as usize * std::mem::size_of::<u32>()],
        vyre_primitives::wire::pack_u32_slice(&[0]),
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_offsets),
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_targets),
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_kind_mask),
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_out_seed),
    ])
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

fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}
