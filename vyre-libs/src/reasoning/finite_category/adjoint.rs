//! Adjoint functor pairs between finite categories.

#[cfg(test)]
use crate::telemetry::{bump, dataflow_fixpoint_calls};
#[cfg(test)]
pub(crate) use vyre_reference::composition_witness::{AdjointPair, FiniteCategory, FiniteFunctor};

/// Given functors `F: C → D` and `G: D → C`, check `F ⊣ G` on finite categories `C, D`.
#[must_use]
#[cfg(test)]
pub(crate) fn adjoint_pair(
    c_cat: &FiniteCategory,
    d_cat: &FiniteCategory,
    f: &FiniteFunctor,
    g: &FiniteFunctor,
) -> AdjointPair {
    bump(&dataflow_fixpoint_calls);
    vyre_reference::composition_witness::adjoint_pair_witness(c_cat, d_cat, f, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_self_adjoint_on_discrete() {
        let cat = FiniteCategory::discrete(3);
        let id = FiniteFunctor::identity(3);
        let result = adjoint_pair(&cat, &cat, &id, &id);
        assert!(result.is_adjoint);
        assert!(result.witness.is_none());
    }

    #[test]
    fn non_adjoint_pinpoints_failure() {
        let cat = FiniteCategory::discrete(2);
        let f = FiniteFunctor {
            object_map: vec![0, 0],
        };
        let g = FiniteFunctor::identity(2);
        let result = adjoint_pair(&cat, &cat, &f, &g);
        assert!(!result.is_adjoint);
        assert!(result.witness.is_some());
    }

    #[test]
    fn partial_adjunction_rejected() {
        let cat = FiniteCategory::discrete(2);
        let f = FiniteFunctor::identity(2);
        let g = FiniteFunctor {
            object_map: vec![1, 0],
        };
        let result = adjoint_pair(&cat, &cat, &f, &g);
        assert!(!result.is_adjoint);
    }

    #[test]
    fn identity_adjoint_pair_for_any_size() {
        for n in [1u32, 2, 4, 8] {
            let cat = FiniteCategory::discrete(n);
            let id = FiniteFunctor::identity(n);
            assert!(adjoint_pair(&cat, &cat, &id, &id).is_adjoint);
        }
    }
}
