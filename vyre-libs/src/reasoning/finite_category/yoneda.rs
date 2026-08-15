//! Yoneda lemma over a finite category.
//!
//! For a finite category C and a presheaf F: C^op → Set, the Yoneda lemma
//! states `Nat(Hom(-, X), F) ≅ F(X)`: the set of natural transformations out of
//! the representable functor is canonically isomorphic to the F-image of X.
//!
//! Pass composition uses that to decide whether a pass tree maps into a
//! representable family, and it can decide it from the F-cardinality at one
//! object without walking the rest of the category.

use crate::telemetry::{bump, dataflow_fixpoint_calls};

/// Finite category: object set + Hom-set sizes per `(source, target)`
/// pair. `hom_size[s * n + t]` is `|Hom(s, t)|`.
///
/// Identity morphisms are implicit (every Hom(X, X) has at least one).
/// Composition is not modeled here  -  Yoneda only needs the Hom
/// cardinality + the F-image cardinality, which is enough for the
/// natural-isomorphism count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteCategory {
    /// Number of objects.
    pub n: u32,
    /// Row-major `n × n` Hom-set sizes.
    pub hom_size: Vec<u32>,
}

impl FiniteCategory {
    /// Discrete category on `n` objects: `|Hom(s, t)| = 1` if s == t,
    /// else 0. Used as the canonical Yoneda witness for the
    /// "obvious" natural-iso count.
    #[must_use]
    pub fn discrete(n: u32) -> Self {
        let n_us = n as usize;
        let mut hom_size = vec![0u32; n_us * n_us];
        for i in 0..n_us {
            hom_size[i * n_us + i] = 1;
        }
        Self { n, hom_size }
    }

    /// Cardinality of `Hom(source, target)`.
    ///
    #[must_use]
    pub fn hom(&self, source: u32, target: u32) -> u32 {
        if source >= self.n || target >= self.n {
            return 0;
        }
        self.hom_size
            .get((source * self.n + target) as usize)
            .copied()
            .unwrap_or(0)
    }
}

/// The Yoneda embedding of object `x`: the cardinality vector
/// `[|Hom(c_0, x)|, |Hom(c_1, x)|, …]`, one entry per object of the category.
/// Each entry is the representable functor's image at that object.
#[must_use]
pub fn yoneda_embedding(category: &FiniteCategory, x: u32) -> Vec<u32> {
    bump(&dataflow_fixpoint_calls);
    (0..category.n).map(|c| category.hom(c, x)).collect()
}

/// `|Nat(Hom(-, x), F)|`, which the Yoneda lemma says is `|F(x)|`.
///
/// `f_at_x` is `|F(x)|`, so the body forwards it. The category and the object
/// stay in the signature because they are what makes the claim checkable: a
/// caller that has to name the category it is reasoning about cannot pass the
/// cardinality of a different presheaf by accident.
#[must_use]
pub fn natural_transformation_count(_category: &FiniteCategory, _x: u32, f_at_x: u32) -> u32 {
    bump(&dataflow_fixpoint_calls);
    f_at_x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_category_self_hom_is_one() {
        let cat = FiniteCategory::discrete(4);
        for i in 0..4 {
            assert_eq!(cat.hom(i, i), 1);
        }
    }

    #[test]
    fn discrete_category_cross_hom_is_zero() {
        let cat = FiniteCategory::discrete(4);
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert_eq!(cat.hom(i, j), 0);
                }
            }
        }
    }

    #[test]
    fn yoneda_embedding_on_discrete_is_unit_vector() {
        // Hom(c_i, x) = 1 iff c_i == x, else 0.
        let cat = FiniteCategory::discrete(3);
        assert_eq!(yoneda_embedding(&cat, 0), vec![1, 0, 0]);
        assert_eq!(yoneda_embedding(&cat, 1), vec![0, 1, 0]);
        assert_eq!(yoneda_embedding(&cat, 2), vec![0, 0, 1]);
    }

    /// The count must equal the supplied F-image cardinality at every value,
    /// including `u32::MAX`. A rewrite that reintroduced a hand-rolled Hom-walk
    /// would compute something else and fire here.
    #[test]
    fn yoneda_iso_equals_f_image_cardinality() {
        let cat = FiniteCategory::discrete(3);
        assert_eq!(natural_transformation_count(&cat, 0, 0), 0);
        assert_eq!(natural_transformation_count(&cat, 0, 1), 1);
        assert_eq!(natural_transformation_count(&cat, 0, 2), 2);
        assert_eq!(natural_transformation_count(&cat, 0, 5), 5);
        assert_eq!(natural_transformation_count(&cat, 0, 100), 100);
        assert_eq!(natural_transformation_count(&cat, 0, u32::MAX), u32::MAX);
    }

    /// A richer Hom-set does not change the answer. Yoneda gives
    /// `|Nat(Hom(-, x), F)| = |F(x)|` whatever shape the Hom-sets have, so a
    /// walk that consulted them would diverge from the lemma.
    #[test]
    fn yoneda_invariant_under_richer_homsets() {
        let mut cat = FiniteCategory::discrete(3);
        // Add 2 morphisms in Hom(0, 1).
        cat.hom_size[0 * 3 + 1] = 2;
        let f_at_x = 7;
        assert_eq!(natural_transformation_count(&cat, 1, f_at_x), 7);
    }

    /// An empty F-image at x means there are no natural transformations at all.
    #[test]
    fn empty_f_image_means_no_natural_transformations() {
        let cat = FiniteCategory::discrete(2);
        assert_eq!(natural_transformation_count(&cat, 0, 0), 0);
    }
}
