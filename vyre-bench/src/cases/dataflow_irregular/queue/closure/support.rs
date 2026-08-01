use crate::api::case::BenchError;
use vyre_foundation::ir::Program;

use super::QUEUE_CLOSURE_WORKGROUP_SIZE;
use crate::cases::dataflow_irregular::fixture::{
    materialize_ifds_active_queue, IfdsSkewedFixture, IFDS_REACH_MASK,
};

pub(in crate::cases::dataflow_irregular) struct QueueClosureOracle {
    pub(in crate::cases::dataflow_irregular) output: Vec<u32>,
    pub(in crate::cases::dataflow_irregular) iterations: u32,
    pub(in crate::cases::dataflow_irregular) changed: u32,
    pub(in crate::cases::dataflow_irregular) total_queue_pops: u64,
    pub(in crate::cases::dataflow_irregular) max_wave_queue_len: u32,
    pub(in crate::cases::dataflow_irregular) wave_queue_lengths: Vec<u32>,
}

pub(in crate::cases::dataflow_irregular) fn ifds_queue_closure_inputs(
    fixture: &IfdsSkewedFixture,
    queue_capacity: u32,
) -> Result<Vec<Vec<u8>>, BenchError> {
    crate::cases::queue_stage::build_queue_closure_inputs(
        &fixture.frontier_in,
        &fixture.edge_offsets,
        &fixture.edge_targets,
        &fixture.edge_kind_mask,
        fixture.stats.active_sources,
        queue_capacity,
        "IFDS",
        |capacity| materialize_ifds_active_queue(fixture, capacity, "IFDS queue closure seed"),
    )
}

pub(in crate::cases::dataflow_irregular) fn ifds_queue_closure_reset_program(
    frontier_words: u32,
    seed_queue_len: u32,
    queue_capacity: u32,
) -> Program {
    crate::cases::queue_stage::build_queue_closure_reset_program(
        frontier_words,
        seed_queue_len,
        queue_capacity,
        QUEUE_CLOSURE_WORKGROUP_SIZE,
    )
}

pub(in crate::cases::dataflow_irregular) fn ifds_skewed_queue_closure_oracle(
    fixture: &IfdsSkewedFixture,
    max_iters: u32,
    queue_capacity: u32,
) -> Result<QueueClosureOracle, BenchError> {
    let capacity = queue_capacity as usize;
    let mut accumulator = fixture.frontier_in.clone();
    let mut current =
        materialize_ifds_active_queue(fixture, capacity, "IFDS queue closure oracle seed")?;
    let mut next = Vec::with_capacity(capacity.min(fixture.stats.nodes as usize));
    let mut iterations = 0_u32;
    let mut total_queue_pops = 0_u64;
    let mut max_wave_queue_len = current.len() as u32;
    let mut wave_queue_lengths = Vec::new();

    while !current.is_empty() && iterations < max_iters {
        wave_queue_lengths.push(current.len() as u32);
        max_wave_queue_len = max_wave_queue_len.max(current.len() as u32);
        total_queue_pops = total_queue_pops.saturating_add(current.len() as u64);
        next.clear();
        for &src in &current {
            if src >= fixture.stats.nodes {
                continue;
            }
            let start = fixture.edge_offsets[src as usize] as usize;
            let end = fixture.edge_offsets[src as usize + 1] as usize;
            for edge in start..end {
                if fixture.edge_kind_mask[edge] & IFDS_REACH_MASK == 0 {
                    continue;
                }
                let dst = fixture.edge_targets[edge];
                if dst >= fixture.stats.nodes {
                    continue;
                }
                let dst_word = dst as usize / 32;
                let dst_bit = 1_u32 << (dst % 32);
                if accumulator[dst_word] & dst_bit != 0 {
                    continue;
                }
                accumulator[dst_word] |= dst_bit;
                if next.len() >= capacity {
                    return Err(BenchError::EnvironmentInvalid(format!(
                        "IFDS queue closure next wave exceeded queue_capacity={queue_capacity}. Fix: increase queue capacity or shard closure waves."
                    )));
                }
                next.push(dst);
            }
        }
        iterations = iterations.saturating_add(1);
        std::mem::swap(&mut current, &mut next);
    }

    if !current.is_empty() {
        return Err(BenchError::EnvironmentInvalid(format!(
            "IFDS queue closure did not converge within {max_iters} queue waves. Fix: raise CLOSURE_MAX_ITERS or use a smaller fixture diameter."
        )));
    }

    Ok(QueueClosureOracle {
        changed: u32::from(accumulator != fixture.frontier_in),
        output: accumulator,
        iterations,
        total_queue_pops,
        max_wave_queue_len,
        wave_queue_lengths,
    })
}
