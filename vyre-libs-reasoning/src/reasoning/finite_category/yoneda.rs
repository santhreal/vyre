//! Yoneda lemma over a finite category.

#[cfg(test)]
pub(crate) use vyre_reference::composition_witness::FiniteCategory;

/// The Yoneda embedding of object `x`: the cardinality vector
/// `[|Hom(c_0, x)|, |Hom(c_1, x)|, …]`, one entry per object of the category.
#[must_use]
#[cfg(test)]
pub(crate) fn yoneda_embedding(category: &FiniteCategory, x: u32) -> Vec<u32> {
    vyre_reference::composition_witness::yoneda_embedding_witness(category, x)
}

/// `|Nat(Hom(-, x), F)|`, which the Yoneda lemma says is `|F(x)|`.
#[must_use]
#[cfg(test)]
pub(crate) fn natural_transformation_count(category: &FiniteCategory, x: u32, f_at_x: u32) -> u32 {
    vyre_reference::composition_witness::natural_transformation_count_witness(category, x, f_at_x)
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
        let cat = FiniteCategory::discrete(3);
        assert_eq!(yoneda_embedding(&cat, 0), vec![1, 0, 0]);
        assert_eq!(yoneda_embedding(&cat, 1), vec![0, 1, 0]);
        assert_eq!(yoneda_embedding(&cat, 2), vec![0, 0, 1]);
    }

    #[test]
    fn natural_transformation_count_tracks_image_cardinality() {
        let cat = FiniteCategory::discrete(3);
        assert_eq!(natural_transformation_count(&cat, 0, 0), 0);
        assert_eq!(natural_transformation_count(&cat, 0, 1), 1);
        assert_eq!(natural_transformation_count(&cat, 0, 2), 2);
        assert_eq!(natural_transformation_count(&cat, 0, 5), 5);
        assert_eq!(natural_transformation_count(&cat, 0, 100), 100);
        assert_eq!(natural_transformation_count(&cat, 0, u32::MAX), u32::MAX);
    }

    #[test]
    fn rich_hom_set_preserves_count() {
        let cat = FiniteCategory {
            n: 2,
            hom_size: vec![3, 5, 2, 4],
        };
        let f_at_x = 7;
        assert_eq!(natural_transformation_count(&cat, 1, f_at_x), 7);
    }

    #[test]
    fn empty_image_has_zero_transformations() {
        let cat = FiniteCategory::discrete(2);
        assert_eq!(natural_transformation_count(&cat, 0, 0), 0);
    }
}
