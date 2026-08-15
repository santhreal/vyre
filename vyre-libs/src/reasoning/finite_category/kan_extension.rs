//! Kan extension of a set-valued functor along a re-indexing functor.
//!
//! For a functor `K: M → C` and `F: M → Set`, the **left Kan extension**
//! `Lan_K F: C → Set` is the universal natural extension along K. Over finite
//! set-valued functors the value at a target object `c ∈ C` reduces to a
//! colimit, which is a sum:
//!
//! ```text
//! (Lan_K F)(c) = ∑_{m : K(m) = c} F(m)
//! ```
//!
//! The **right Kan extension** is the dual limit, a product:
//!
//! ```text
//! (Ran_K F)(c) = ∏_{m : K(m) = c} F(m)
//! ```
//!
//! The two differ in the fold alone: identity 0 under saturating addition,
//! identity 1 under saturating multiplication. [`KanDirection`] carries that
//! difference so one walk over the preimage serves both, and the empty preimage
//! returns the fold's identity, which is the initial set on the left and the
//! terminal set on the right.
//!
//! Pass composition extends a partially defined functor along a re-indexing
//! functor pointwise this way, without materializing the full diagram.

use super::adjoint::FiniteFunctor;
use crate::telemetry::{bump, dataflow_fixpoint_calls};

/// Which universal construction a Kan extension takes over the preimage.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KanDirection {
    /// Colimit: a sum whose empty case is the initial set, cardinality 0.
    Left,
    /// Limit: a product whose empty case is the terminal set, cardinality 1.
    Right,
}

impl KanDirection {
    /// Cardinality the fold starts from, and the answer over an empty preimage.
    const fn identity(self) -> u32 {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }

    /// Saturating fold step. Saturation, not wrapping: a cardinality that
    /// overflows `u32` is unbounded for every decision a caller makes on it,
    /// and `u32::MAX` says that where a wrapped small number would lie.
    const fn fold(self, accumulated: u32, value: u32) -> u32 {
        match self {
            Self::Left => accumulated.saturating_add(value),
            Self::Right => accumulated.saturating_mul(value),
        }
    }
}

/// Cardinality of the Kan extension at one object of the codomain.
#[must_use]
pub fn kan_extension_at(
    direction: KanDirection,
    k: &FiniteFunctor,
    f_image: &[u32],
    c: u32,
) -> u32 {
    bump(&dataflow_fixpoint_calls);
    fold_preimage(direction, k, f_image, c)
}

/// Cardinality of the Kan extension at every object of a codomain of size
/// `c_n`, indexed by object.
#[must_use]
pub fn kan_extension_table(
    direction: KanDirection,
    k: &FiniteFunctor,
    f_image: &[u32],
    c_n: u32,
) -> Vec<u32> {
    bump(&dataflow_fixpoint_calls);
    (0..c_n)
        .map(|c| fold_preimage(direction, k, f_image, c))
        .collect()
}

/// The walk both entry points share. Private so a table is one charged call
/// rather than one per object.
fn fold_preimage(direction: KanDirection, k: &FiniteFunctor, f_image: &[u32], c: u32) -> u32 {
    debug_assert_eq!(k.object_map.len(), f_image.len());
    let mut accumulated = direction.identity();
    for (m, &image) in k.object_map.iter().enumerate() {
        if image == c {
            accumulated = direction.fold(accumulated, f_image[m]);
        }
    }
    accumulated
}

#[cfg(test)]
mod tests {
    use super::KanDirection::{Left, Right};
    use super::*;

    #[test]
    fn an_empty_preimage_returns_the_fold_identity() {
        // K maps both M-objects to C-object 0, so C-object 1 has no preimage.
        let k = FiniteFunctor {
            object_map: vec![0, 0],
        };
        let f = vec![3u32, 5];
        assert_eq!(kan_extension_at(Left, &k, &f, 1), 0);
        assert_eq!(kan_extension_at(Right, &k, &f, 1), 1);
    }

    #[test]
    fn left_sums_and_right_multiplies_the_preimage() {
        // K(0)=0, K(1)=0, K(2)=1. F = [3, 5, 7].
        let k = FiniteFunctor {
            object_map: vec![0, 0, 1],
        };
        let f = vec![3u32, 5, 7];
        assert_eq!(kan_extension_at(Left, &k, &f, 0), 8);
        assert_eq!(kan_extension_at(Left, &k, &f, 1), 7);
        assert_eq!(kan_extension_at(Right, &k, &f, 0), 15);
        assert_eq!(kan_extension_at(Right, &k, &f, 1), 7);
    }

    /// The table form must agree with the pointwise form at every object, in
    /// both directions. A table that drifted from the walk it indexes is the
    /// failure this pins.
    #[test]
    fn the_table_agrees_with_the_pointwise_form() {
        let k = FiniteFunctor {
            object_map: vec![0, 1, 0, 2, 1],
        };
        let f = vec![1u32, 2, 3, 4, 5];
        for direction in [Left, Right] {
            let table = kan_extension_table(direction, &k, &f, 3);
            assert_eq!(table.len(), 3);
            for c in 0..3u32 {
                assert_eq!(table[c as usize], kan_extension_at(direction, &k, &f, c));
            }
        }
    }

    /// Extending along the identity functor recovers F itself, in both
    /// directions, because every preimage is a single object.
    #[test]
    fn extending_along_the_identity_recovers_f() {
        let k = FiniteFunctor::identity(4);
        let f = vec![2u32, 3, 5, 7];
        for direction in [Left, Right] {
            for c in 0..4u32 {
                assert_eq!(kan_extension_at(direction, &k, &f, c), f[c as usize]);
            }
        }
    }

    /// Saturation, not wrapping. Three copies of `u32::MAX` in one preimage
    /// overflow both folds; a wrapping sum would report 4294967293 and a
    /// wrapping product 4294967295 by accident, so the sum is what proves it.
    #[test]
    fn an_overflowing_preimage_saturates() {
        let k = FiniteFunctor {
            object_map: vec![0, 0, 0],
        };
        let f = vec![u32::MAX, u32::MAX, u32::MAX];
        assert_eq!(kan_extension_at(Left, &k, &f, 0), u32::MAX);
        assert_eq!(kan_extension_at(Right, &k, &f, 0), u32::MAX);
    }
}
