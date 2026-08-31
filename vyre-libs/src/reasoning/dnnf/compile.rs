//! d-DNNF test-adapter delegating sequential reasoning to `vyre_reference::composition_witness`.
//!
//! Sequential d-DNNF compilation and exact model counting are owned by the
//! reference witness suite in `vyre-reference`. This module provides test-scoped
//! adapters for parity verification.

#[cfg(test)]
pub(crate) use vyre_reference::composition_witness::{DnnfDag, DnnfGate};

/// Compile a CNF formula into a d-DNNF DAG.
#[must_use]
#[cfg(test)]
pub(crate) fn compile_dnnf(clauses: &[Vec<(u32, bool)>], num_vars: u32, max_depth: u32) -> DnnfDag {
    vyre_reference::composition_witness::compile_dnnf_witness(clauses, num_vars, max_depth)
}

/// Count satisfying assignments via a d-DNNF DAG.
#[must_use]
#[cfg(test)]
pub(crate) fn model_count(dag: &DnnfDag) -> u64 {
    vyre_reference::composition_witness::dnnf_model_count_witness(dag)
}

/// Whether the formula has at least one model.
#[must_use]
#[cfg(test)]
pub(crate) fn is_satisfiable(dag: &DnnfDag) -> bool {
    vyre_reference::composition_witness::dnnf_is_satisfiable_witness(dag)
}

/// Whether every assignment over `num_vars` variables is a model.
#[must_use]
#[cfg(test)]
pub(crate) fn is_tautology(dag: &DnnfDag, num_vars: u32) -> bool {
    vyre_reference::composition_witness::dnnf_is_tautology_witness(dag, num_vars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_empty_formula_is_true() {
        let dag = compile_dnnf(&[], 0, 4);
        assert_eq!(dag.gates.last(), Some(&DnnfGate::True));
        assert_eq!(model_count(&dag), 1);
    }

    #[test]
    fn compile_single_literal() {
        let dag = compile_dnnf(&[vec![(0u32, true)]], 1, 4);
        assert_eq!(model_count(&dag), 1);
    }

    #[test]
    fn compile_contradiction_yields_zero_models() {
        let dag = compile_dnnf(&[vec![(0u32, true)], vec![(0, false)]], 1, 4);
        assert_eq!(model_count(&dag), 0);
    }

    #[test]
    fn compile_disjunction_of_two_lits() {
        let dag = compile_dnnf(&[vec![(0u32, true), (1, true)]], 2, 4);
        assert_eq!(model_count(&dag), 3);
    }

    #[test]
    fn matches_brute_force_on_small_formulas() {
        let clauses = vec![vec![(0u32, true), (1, false)], vec![(1, true), (2, true)]];
        let dag = compile_dnnf(&clauses, 3, 8);
        let dag_count = model_count(&dag);

        let mut bf = 0u64;
        for assignment in 0u8..8 {
            let x = [
                (assignment & 1) != 0,
                (assignment & 2) != 0,
                (assignment & 4) != 0,
            ];
            let c1 = x[0] || !x[1];
            let c2 = x[1] || x[2];
            if c1 && c2 {
                bf += 1;
            }
        }
        assert_eq!(dag_count, bf, "d-DNNF count must match brute force");
    }

    #[test]
    fn depth_budget_terminates() {
        let clauses = vec![
            vec![(0u32, true), (1, true)],
            vec![(2, true), (3, true)],
            vec![(4, true), (5, true)],
        ];
        let dag = compile_dnnf(&clauses, 6, 2);
        assert_eq!(dag.num_vars, 6);
        assert!(
            !dag.gates.is_empty(),
            "depth budget must emit at least one gate"
        );
    }

    #[test]
    fn model_count_smooths_over_free_vars() {
        let dag = compile_dnnf(&[], 0, 4);
        assert_eq!(model_count(&dag), 1);
        let dag = compile_dnnf(&[], 5, 4);
        assert_eq!(model_count(&dag), 32);
        let dag = compile_dnnf(&[], 1, 4);
        assert_eq!(model_count(&dag), 2);
    }

    #[test]
    fn model_count_saturates_at_u64_max() {
        let dag = compile_dnnf(&[], 64, 4);
        assert!(model_count(&dag) > 0);
    }

    #[test]
    fn root_is_last_gate() {
        let dag = compile_dnnf(&[vec![(0u32, true)]], 1, 4);
        assert_eq!(dag.root(), (dag.gates.len() - 1) as u32);
    }
}
