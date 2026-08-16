//! Subgroup simulator lane contracts.

use proptest::prelude::*;
use rayon::prelude::*;
use vyre_reference::subgroup::SubgroupSimulator;

#[test]
fn ballot_sets_expected_bits() {
    let simulator = SubgroupSimulator::default();
    assert_eq!(simulator.ballot(&[true, false, true, true]), 0b1101);
}

#[test]
fn shuffle_zeroes_out_of_range_lanes() {
    let simulator = SubgroupSimulator::new(4);
    assert_eq!(
        simulator.shuffle(&[10, 20, 30, 40], &[0, 2, 5, 1]),
        vec![10, 30, 0, 20]
    );
}

proptest! {
    #[test]
    fn subgroup_add_matches_parallel_wrapping_sum(values in prop::collection::vec(any::<u32>(), 0..128)) {
        let simulator = SubgroupSimulator::new(values.len().max(1));
        let expected = values.par_iter().copied().reduce(|| 0u32, u32::wrapping_add);
        prop_assert_eq!(simulator.add(&values), expected);
    }
}
