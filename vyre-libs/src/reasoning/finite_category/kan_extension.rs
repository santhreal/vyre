//! Kan extension of a set-valued functor along a re-indexing functor.

#[cfg(test)]
use super::adjoint::FiniteFunctor;
#[cfg(test)]
use crate::telemetry::{bump, dataflow_fixpoint_calls};
#[cfg(test)]
pub(crate) use vyre_reference::composition_witness::KanDirection;

/// Cardinality of the Kan extension at one object of the codomain.
#[must_use]
#[cfg(test)]
pub(crate) fn kan_extension_at(
    direction: KanDirection,
    k: &FiniteFunctor,
    f_image: &[u32],
    c: u32,
) -> u32 {
    bump(&dataflow_fixpoint_calls);
    vyre_reference::composition_witness::kan_extension_at_witness(direction, k, f_image, c)
}

/// Cardinality of the Kan extension at every object of a codomain of size
/// `c_n`, indexed by object.
#[must_use]
#[cfg(test)]
pub(crate) fn kan_extension_table(
    direction: KanDirection,
    k: &FiniteFunctor,
    f_image: &[u32],
    c_n: u32,
) -> Vec<u32> {
    bump(&dataflow_fixpoint_calls);
    vyre_reference::composition_witness::kan_extension_table_witness(direction, k, f_image, c_n)
}

#[cfg(test)]
mod tests {
    use super::KanDirection::{Left, Right};
    use super::*;

    #[test]
    fn an_empty_preimage_returns_the_fold_identity() {
        let k = FiniteFunctor {
            object_map: vec![0, 0],
        };
        let f = vec![3u32, 5];
        assert_eq!(kan_extension_at(Left, &k, &f, 1), 0);
        assert_eq!(kan_extension_at(Right, &k, &f, 1), 1);
    }

    #[test]
    fn left_sums_and_right_multiplies_over_the_preimage() {
        let k = FiniteFunctor {
            object_map: vec![0, 0, 1],
        };
        let f = vec![3u32, 5, 7];
        assert_eq!(kan_extension_at(Left, &k, &f, 0), 8);
        assert_eq!(kan_extension_at(Left, &k, &f, 1), 7);
        assert_eq!(kan_extension_at(Right, &k, &f, 0), 15);
        assert_eq!(kan_extension_at(Right, &k, &f, 1), 7);
    }

    #[test]
    fn table_form_matches_pointwise_evaluation() {
        let k = FiniteFunctor {
            object_map: vec![0, 0, 1],
        };
        let f = vec![3u32, 5, 7];
        for direction in [Left, Right] {
            let table = kan_extension_table(direction, &k, &f, 3);
            assert_eq!(table.len(), 3);
            for c in 0..3u32 {
                assert_eq!(table[c as usize], kan_extension_at(direction, &k, &f, c));
            }
        }
    }

    #[test]
    fn identity_functor_is_a_pointwise_no_op() {
        let k = FiniteFunctor::identity(4);
        let f = vec![11u32, 22, 33, 44];
        for direction in [Left, Right] {
            for c in 0..4u32 {
                assert_eq!(kan_extension_at(direction, &k, &f, c), f[c as usize]);
            }
        }
    }

    #[test]
    fn saturating_fold_does_not_wrap() {
        let k = FiniteFunctor {
            object_map: vec![0, 0, 0],
        };
        let f = vec![u32::MAX, u32::MAX, u32::MAX];
        assert_eq!(kan_extension_at(Left, &k, &f, 0), u32::MAX);
        assert_eq!(kan_extension_at(Right, &k, &f, 0), u32::MAX);
    }
}
