//! Shared deterministic topology and sparse-queue sizing for irregular graph benchmarks.

use crate::api::case::BenchError;
use crate::cases::mix32;

/// Return the shared heavy-tailed degree used by IFDS and CSR fixtures.
pub(crate) fn skewed_degree(source: u32, ugly_hub_degree: u32) -> u32 {
    if source % 4096 == 0 {
        ugly_hub_degree
    } else if source % 257 == 0 {
        24
    } else if source % 31 == 0 {
        8
    } else if source % 7 == 0 {
        3
    } else {
        1
    }
}

/// Select a deterministic target in a power-of-two node space.
pub(crate) fn skewed_target(node_count: u32, source: u32, edge: u32) -> u32 {
    let mask = node_count - 1;
    match edge & 7 {
        0 => source.wrapping_add((edge + 1).wrapping_mul(17)) & mask,
        1 => source.wrapping_sub((edge + 3).wrapping_mul(11)) & mask,
        _ => {
            let salt = edge.wrapping_mul(0x9E37_79B9).rotate_left((edge & 15) + 1);
            mix32(source ^ salt ^ source.rotate_left(edge & 15)) & mask
        }
    }
}

/// How a case labels the nodes and edges of its skewed graph.
///
/// The topology is the same heavy-tailed CSR shape for every case that uses it:
/// `skewed_degree` decides the out-degree and `skewed_target` decides where each
/// edge lands. What a case supplies is the meaning it puts on a node and an
/// edge, and the names its errors carry.
pub(crate) struct SkewedCsrShape<'a> {
    pub(crate) node_count: u32,
    pub(crate) hub_degree: u32,
    pub(crate) high_degree_threshold: u32,
    /// Names the fixture in every error this walk reports.
    pub(crate) fixture: &'a str,
    /// The corrective action for a node count that is not a power of two.
    pub(crate) power_of_two_fix: &'a str,
    pub(crate) node_kind: fn(u32) -> u32,
    pub(crate) node_tag: fn(u32) -> u32,
    pub(crate) edge_kind: fn(u32, u32) -> u32,
    pub(crate) source_is_active: fn(u32) -> bool,
}

/// The CSR arrays and counts one skewed-graph walk produced.
///
/// Each case moves these into its own fixture and stats types, which name the
/// same values in the vocabulary of that workload.
pub(crate) struct SkewedCsrArrays {
    pub(crate) nodes: Vec<u32>,
    pub(crate) edge_offsets: Vec<u32>,
    pub(crate) edge_targets: Vec<u32>,
    pub(crate) edge_kind_mask: Vec<u32>,
    pub(crate) node_tags: Vec<u32>,
    pub(crate) frontier_in: Vec<u32>,
    pub(crate) frontier_words: u32,
    pub(crate) edge_count: u32,
    pub(crate) max_degree: u32,
    pub(crate) active_sources: u64,
    pub(crate) high_degree_sources: u64,
}

/// Build the skewed CSR graph a case's shape describes.
///
/// One walk decides the degree of every source, its target list, its edge kinds
/// and whether it seeds the frontier, so a change to the topology or to the
/// offset accounting is made once.
pub(crate) fn build_skewed_csr_arrays(
    shape: &SkewedCsrShape<'_>,
) -> Result<SkewedCsrArrays, BenchError> {
    let node_count = shape.node_count;
    if !node_count.is_power_of_two() || node_count < 32 {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{} requires a power-of-two node count >= 32, received {node_count}. {}",
            shape.fixture, shape.power_of_two_fix
        )));
    }

    let frontier_words = node_count.div_ceil(32);
    let mut nodes = Vec::with_capacity(node_count as usize);
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::with_capacity((node_count as usize).saturating_mul(2));
    let mut edge_kind_mask = Vec::with_capacity((node_count as usize).saturating_mul(2));
    let mut node_tags = Vec::with_capacity(node_count as usize);
    let mut frontier_in = vec![0_u32; frontier_words as usize];
    let mut max_degree = 0_u32;
    let mut active_sources = 0_u64;
    let mut high_degree_sources = 0_u64;

    edge_offsets.push(0);
    for src in 0..node_count {
        let degree = skewed_degree(src, shape.hub_degree);
        max_degree = max_degree.max(degree);
        high_degree_sources += u64::from(degree >= shape.high_degree_threshold);
        if (shape.source_is_active)(src) {
            active_sources += 1;
            frontier_in[(src / 32) as usize] |= 1_u32 << (src % 32);
        }
        nodes.push((shape.node_kind)(src));
        node_tags.push((shape.node_tag)(src));
        for edge in 0..degree {
            edge_targets.push(skewed_target(node_count, src, edge));
            edge_kind_mask.push((shape.edge_kind)(src, edge));
        }
        edge_offsets.push(u32::try_from(edge_targets.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(format!(
                "{} exceeded u32 edge offsets. Fix: split the benchmark graph.",
                shape.fixture
            ))
        })?);
    }

    let edge_count = u32::try_from(edge_targets.len()).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{} exceeded u32 edge count. Fix: split the benchmark graph.",
            shape.fixture
        ))
    })?;

    Ok(SkewedCsrArrays {
        nodes,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        node_tags,
        frontier_in,
        frontier_words,
        edge_count,
        max_degree,
        active_sources,
        high_degree_sources,
    })
}

/// Count the frontier sources whose out-degree reaches `min_degree`.
///
/// The count decides how a split queue is sized, so an overflow here is refused
/// rather than truncated. `context` names the queue in that error.
pub(crate) fn active_high_degree_sources(
    frontier_in: &[u32],
    edge_offsets: &[u32],
    node_count: u32,
    min_degree: u32,
    context: &str,
) -> Result<u32, BenchError> {
    let mut high_sources = 0_u32;
    for src in 0..node_count {
        let word = (src / 32) as usize;
        let bit = 1_u32 << (src % 32);
        if frontier_in[word] & bit == 0 {
            continue;
        }
        let start = edge_offsets[src as usize];
        let end = edge_offsets[src as usize + 1];
        if end.saturating_sub(start) >= min_degree {
            high_sources = high_sources.checked_add(1).ok_or_else(|| {
                BenchError::EnvironmentInvalid(format!(
                    "{context} high-degree active source count exceeded u32. Fix: split the frontier queue."
                ))
            })?;
        }
    }
    Ok(high_sources)
}

/// Convert an observed active-source count into a queue capacity with contextual errors.
pub(crate) fn sparse_queue_capacity(
    active_sources: u64,
    empty_error: &str,
    overflow_context: &str,
) -> Result<u32, BenchError> {
    if active_sources == 0 {
        return Err(BenchError::EnvironmentInvalid(empty_error.to_string()));
    }
    u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{overflow_context} active source count {active_sources} exceeds u32 indexing. Fix: split the frontier."
        ))
    })
}

/// Materialize an active source queue from a packed frontier bitset.
pub(crate) fn materialize_active_frontier_queue(
    frontier_in: &[u32],
    node_count: u32,
    active_sources: u64,
    queue_capacity: usize,
    context: &str,
) -> Result<Vec<u32>, BenchError> {
    let mut active_queue = Vec::new();
    let expected = u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{context} active source count {active_sources} exceeds u32 indexing. Fix: split the frontier."
        ))
    })?;
    let seen = vyre_reference::composition_witness::frontier_to_queue_witness_into(
        frontier_in,
        node_count,
        queue_capacity,
        &mut active_queue,
    );
    if seen != expected || active_queue.len() < expected as usize {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{context} queue_capacity {queue_capacity} cannot hold {expected} active sources. Fix: increase queue capacity."
        )));
    }
    Ok(active_queue)
}
