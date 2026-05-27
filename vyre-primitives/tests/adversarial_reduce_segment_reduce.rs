//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for reduce::segment_reduce

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]

use vyre_primitives::reduce::segment_reduce::*;

fn cpu_ref(input: &[u32], segment_offsets: &[u32]) -> Vec<u32> {
    let num_segments = segment_offsets
        .len()
        .checked_sub(1)
        .expect("segment_reduce_sum CPU oracle received empty segment_offsets. Fix: pass at least one CSR-style offset.");
    let mut out = Vec::with_capacity(num_segments);
    for segment in 0..num_segments {
        let start = segment_offsets[segment] as usize;
        let end = segment_offsets[segment + 1] as usize;
        assert!(
            start <= end && end <= input.len(),
            "segment_reduce_sum CPU oracle received malformed segment {segment}: start={start}, end={end}, input_len={}. Fix: rebuild monotonic in-bounds segment offsets before parity comparison.",
            input.len()
        );
        out.push(input[start..end].iter().copied().fold(0, u32::wrapping_add));
    }
    out
}

mod adversarial_reduce_segment_reduce_part1 {

    include!("__split/adversarial_reduce_segment_reduce_part1.rs");
}
mod adversarial_reduce_segment_reduce_part2 {
    include!("__split/adversarial_reduce_segment_reduce_part2.rs");
}
mod adversarial_reduce_segment_reduce_part3 {
    include!("__split/adversarial_reduce_segment_reduce_part3.rs");
}
