//! `sink_intersection`  -  count how many of a query set are also in
//! a sink-family bitset. Used by rules that want a fractional
//! confidence ("X% of nodes reachable from source landed in sinks").

use crate::bitset::and::bitset_and;
use crate::bitset::bitset_words;
use crate::reduce::count::reduce_count;
use vyre_foundation::ir::Program;

use crate::security::flow_composition::fuse_security_flow;

pub(crate) const OP_ID: &str = "vyre-libs::security::sink_intersection";

/// Build a sink-intersection-count Program: AND `query_set` with
/// `sink_set` into `intersect_buf`, then popcount-reduce that into
/// `out_scalar`.
///
/// `reduce_count` seeds `out_scalar` to zero before accumulating, so
/// the count is the intersection's population and not an addition to
/// whatever the caller left in the slot.
#[must_use]
pub fn sink_intersection(
    node_count: u32,
    query_set: &str,
    sink_set: &str,
    intersect_buf: &str,
    out_scalar: &str,
) -> Program {
    let words = bitset_words(node_count);
    fuse_security_flow(
        OP_ID,
        &[
            bitset_and(query_set, sink_set, intersect_buf, words),
            reduce_count(intersect_buf, out_scalar, words),
        ],
        out_scalar,
    )
}

/// CPU oracle: count of bits set in `query AND sink`.
#[must_use]
#[cfg(test)]
pub(crate) fn cpu_ref(query_set: &[u32], sink_set: &[u32]) -> u32 {
    let intersection = vyre_reference::composition_witness::bitset_and_witness(query_set, sink_set);
    vyre_reference::composition_witness::reduce_count_witness(&intersection)
}

/// Soundness marker for [`sink_intersection`].
pub struct SinkIntersection;
impl vyre_spec::soundness::SoundnessTagged for SinkIntersection {
    fn soundness(&self) -> vyre_spec::soundness::Soundness {
        vyre_spec::soundness::Soundness::Exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_overlap_counts_all_set_bits() {
        assert_eq!(cpu_ref(&[0b1111], &[0b1111]), 4);
    }

    #[test]
    fn no_overlap_returns_zero() {
        assert_eq!(cpu_ref(&[0b1010], &[0b0101]), 0);
    }

    #[test]
    fn partial_overlap_counts_intersection() {
        assert_eq!(cpu_ref(&[0b1110], &[0b0111]), 2);
    }

    #[test]
    fn distributes_across_words() {
        assert_eq!(cpu_ref(&[0xFF00, 0x00FF], &[0xFFFF, 0xFFFF]), 16);
    }
}
