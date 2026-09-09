//! Strongly-connected-component decomposition driven through the dataflow
//! fixpoint: per-pivot forward/backward bitset packing and the host driver.

pub(super) fn write_pivot_bitsets(
    fwd_closure: &[u32],
    bwd_closure: &[u32],
    pivot: u32,
    n_us: usize,
    forward: &mut [u32],
    backward: &mut [u32],
) {
    forward.fill(0);
    backward.fill(0);
    let pivot_us = pivot as usize;
    // Pivot reaches itself.
    let pivot_word = pivot_us / 32;
    let pivot_bit = 1u32 << (pivot_us % 32);
    forward[pivot_word] |= pivot_bit;
    backward[pivot_word] |= pivot_bit;
    for v in 0..n_us {
        if fwd_closure[pivot_us * n_us + v] != 0 {
            forward[v / 32] |= 1u32 << (v % 32);
        }
        if bwd_closure[pivot_us * n_us + v] != 0 {
            backward[v / 32] |= 1u32 << (v % 32);
        }
    }
}

#[cfg(test)]
mod tests {
    use vyre_reference::composition_witness::{
        dense_reachability_bitsets_witness, dense_scc_components_witness,
    };

    #[test]
    fn forward_and_backward_bitsets_follow_chain_direction() {
        let adjacency = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(
            dense_reachability_bitsets_witness(&adjacency, 0, 3),
            (vec![0b111], vec![0b001])
        );
        assert_eq!(
            dense_reachability_bitsets_witness(&adjacency, 2, 3),
            (vec![0b100], vec![0b111])
        );
    }

    #[test]
    fn directed_chain_nodes_are_singleton_components() {
        let adjacency = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(dense_scc_components_witness(&adjacency, 3), vec![0, 1, 2]);
    }

    #[test]
    fn cycle_collapses_to_first_pivot() {
        assert_eq!(dense_scc_components_witness(&[0, 1, 1, 0], 2), vec![0, 0]);
    }

    #[test]
    fn mixed_cycle_and_chain_have_expected_components() {
        let adjacency = vec![
            0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        ];
        assert_eq!(
            dense_scc_components_witness(&adjacency, 5),
            vec![0, 0, 0, 3, 4]
        );
    }

    #[test]
    fn disconnected_and_self_loop_nodes_remain_distinct() {
        assert_eq!(dense_scc_components_witness(&[0; 16], 4), vec![0, 1, 2, 3]);
        assert_eq!(
            dense_scc_components_witness(&[1, 0, 0, 0, 0, 0, 0, 0, 0], 3),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn repeated_calls_do_not_retain_previous_components() {
        let cycle = dense_scc_components_witness(&[0, 1, 1, 0], 2);
        let disconnected = dense_scc_components_witness(&[0, 0, 0, 0], 2);
        let cycle_again = dense_scc_components_witness(&[0, 1, 1, 0], 2);
        assert_eq!(cycle, vec![0, 0]);
        assert_eq!(disconnected, vec![0, 1]);
        assert_eq!(cycle_again, cycle);
    }
}
