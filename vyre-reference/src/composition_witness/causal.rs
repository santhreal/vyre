//! Sequential mathematical witnesses for do-calculus, interventions, and causal change-impact prediction.

/// Sequential witness for do-calculus Rule 2 edge reversal writing into caller storage.
pub fn do_rule2_reverse_incoming_witness_into(
    adjacency: &[u32],
    treatment_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) {
    let n = n as usize;
    assert_eq!(
        adjacency.len(),
        n * n,
        "Fix: do-calculus rule2 requires a complete n*n adjacency matrix: adjacency.len() == n*n, got len={} for n={n}.",
        adjacency.len()
    );
    assert_eq!(
        treatment_mask.len(),
        n,
        "Fix: do-calculus rule2 requires treatment_mask.len() == n, got len={} for n={n}.",
        treatment_mask.len()
    );
    let cells = n * n;
    if out.capacity() < cells {
        out.reserve(cells.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(cells, 0);
    for source in 0..n {
        for destination in 0..n {
            let original = if treatment_mask[destination] == 0 {
                adjacency[source * n + destination]
            } else {
                0
            };
            let reversed = if treatment_mask[source] != 0 {
                adjacency[destination * n + source]
            } else {
                0
            };
            out[source * n + destination] = original | reversed;
        }
    }
}

/// Sequential witness for do-calculus Rule 2 edge reversal.
#[must_use]
pub fn do_rule2_reverse_incoming_witness(
    adjacency: &[u32],
    treatment_mask: &[u32],
    n: u32,
) -> Vec<u32> {
    let n_usize = n as usize;
    let mut out = Vec::with_capacity(n_usize * n_usize);
    do_rule2_reverse_incoming_witness_into(adjacency, treatment_mask, n, &mut out);
    out
}

/// Sequential witness for intervention deletion of every incoming edge writing into caller storage.
pub fn do_intervention_delete_incoming_witness_into(
    adjacency: &[u32],
    intervention_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) {
    let n = n as usize;
    assert_eq!(
        adjacency.len(),
        n * n,
        "Fix: do-calculus intervention requires a complete n*n adjacency matrix: adjacency.len() == n*n, got len={} for n={n}.",
        adjacency.len()
    );
    assert_eq!(
        intervention_mask.len(),
        n,
        "Fix: do-calculus intervention requires intervention_mask.len() == n, got len={} for n={n}.",
        intervention_mask.len()
    );
    let cells = n * n;
    if out.capacity() < cells {
        out.reserve(cells.saturating_sub(out.len()));
    }
    out.clear();
    out.extend_from_slice(adjacency);
    for source in 0..n {
        for destination in 0..n {
            if intervention_mask[destination] != 0 {
                out[source * n + destination] = 0;
            }
        }
    }
}

/// Sequential witness for intervention deletion of every incoming edge.
#[must_use]
pub fn do_intervention_delete_incoming_witness(
    adjacency: &[u32],
    intervention_mask: &[u32],
    n: u32,
) -> Vec<u32> {
    let n_usize = n as usize;
    let mut out = Vec::with_capacity(n_usize * n_usize);
    do_intervention_delete_incoming_witness_into(adjacency, intervention_mask, n, &mut out);
    out
}

/// Sequential witness for dense Rule 3 subgraph extraction writing into caller storage.
pub fn do_rule3_subgraph_witness_into(
    adjacency: &[u32],
    keep_mask: &[u32],
    n: u32,
    reduced: &mut Vec<u32>,
    kept: &mut Vec<u32>,
) {
    let n = n as usize;
    assert_eq!(
        adjacency.len(),
        n * n,
        "Fix: do-calculus rule3 requires a complete n*n adjacency matrix: adjacency.len() == n*n, got len={} for n={n}.",
        adjacency.len()
    );
    assert_eq!(
        keep_mask.len(),
        n,
        "Fix: do-calculus rule3 requires keep_mask.len() == n, got len={} for n={n}.",
        keep_mask.len()
    );
    kept.clear();
    kept.extend(
        keep_mask
            .iter()
            .enumerate()
            .filter_map(|(index, &keep)| (keep != 0).then_some(index as u32)),
    );
    let count = kept.len();
    let cells = count * count;
    if reduced.capacity() < cells {
        reduced.reserve(cells.saturating_sub(reduced.len()));
    }
    reduced.clear();
    for &source in kept.iter() {
        for &destination in kept.iter() {
            reduced.push(adjacency[source as usize * n + destination as usize]);
        }
    }
}

/// Sequential witness for dense Rule 3 subgraph extraction.
#[must_use]
pub fn do_rule3_subgraph_witness(
    adjacency: &[u32],
    keep_mask: &[u32],
    n: u32,
) -> (Vec<u32>, Vec<u32>) {
    let mut reduced = Vec::new();
    let mut kept = Vec::new();
    do_rule3_subgraph_witness_into(adjacency, keep_mask, n, &mut reduced, &mut kept);
    (reduced, kept)
}

/// Sequential reachability closure witness writing into caller storage.
pub fn reachability_closure_witness_into(adjacency: &[u32], n: usize, closure: &mut Vec<u32>) {
    assert_eq!(
        adjacency.len(),
        n * n,
        "Fix: do-calculus reachability closure requires a complete n*n adjacency matrix: adjacency.len() == n*n, got len={} for n={n}.",
        adjacency.len()
    );
    let cells = n * n;
    if closure.capacity() < cells {
        closure.reserve(cells.saturating_sub(closure.len()));
    }
    closure.clear();
    closure.extend(adjacency.iter().map(|&edge| u32::from(edge != 0)));
    for pivot in 0..n {
        for source in 0..n {
            if closure[source * n + pivot] == 0 {
                continue;
            }
            for destination in 0..n {
                if closure[pivot * n + destination] != 0 {
                    closure[source * n + destination] = 1;
                }
            }
        }
    }
}

fn reachability_closure_witness(adjacency: &[u32], n: usize) -> Vec<u32> {
    let mut closure = Vec::with_capacity(n * n);
    reachability_closure_witness_into(adjacency, n, &mut closure);
    closure
}

/// Sequential impact from surgery witness writing into caller storage.
pub fn impact_from_surgery_witness_into(
    surgery: &[u32],
    intervention_mask: &[u32],
    n: usize,
    closure_scratch: &mut Vec<u32>,
    impact: &mut Vec<u32>,
) {
    reachability_closure_witness_into(surgery, n, closure_scratch);
    if impact.capacity() < n {
        impact.reserve(n.saturating_sub(impact.len()));
    }
    impact.clear();
    impact.resize(n, 0);
    for source in 0..n {
        if intervention_mask[source] == 0 {
            continue;
        }
        impact[source] = 1;
        for destination in 0..n {
            if closure_scratch[source * n + destination] != 0 {
                impact[destination] = 1;
            }
        }
    }
}

fn impact_from_surgery_witness(surgery: &[u32], intervention_mask: &[u32], n: usize) -> Vec<u32> {
    let mut closure_scratch = Vec::with_capacity(n * n);
    let mut impact = Vec::with_capacity(n);
    impact_from_surgery_witness_into(
        surgery,
        intervention_mask,
        n,
        &mut closure_scratch,
        &mut impact,
    );
    impact
}

/// Sequential intervention-form change-impact witness into caller-owned storage.
pub fn predict_impact_witness_into(
    adjacency: &[u32],
    intervention_mask: &[u32],
    n: u32,
    surgery_scratch: &mut Vec<u32>,
    closure_scratch: &mut Vec<u32>,
    impact: &mut Vec<u32>,
) {
    do_intervention_delete_incoming_witness_into(adjacency, intervention_mask, n, surgery_scratch);
    impact_from_surgery_witness_into(
        surgery_scratch,
        intervention_mask,
        n as usize,
        closure_scratch,
        impact,
    );
}

/// Sequential intervention-form change-impact witness.
#[must_use]
pub fn predict_impact_witness(adjacency: &[u32], intervention_mask: &[u32], n: u32) -> Vec<u32> {
    let mut surgery_scratch = Vec::new();
    let mut closure_scratch = Vec::new();
    let mut impact = Vec::new();
    predict_impact_witness_into(
        adjacency,
        intervention_mask,
        n,
        &mut surgery_scratch,
        &mut closure_scratch,
        &mut impact,
    );
    impact
}

/// Sequential observation-form change-impact witness into caller-owned storage.
pub fn predict_impact_observation_form_witness_into(
    adjacency: &[u32],
    observation_mask: &[u32],
    n: u32,
    surgery_scratch: &mut Vec<u32>,
    closure_scratch: &mut Vec<u32>,
    impact: &mut Vec<u32>,
) {
    do_rule2_reverse_incoming_witness_into(adjacency, observation_mask, n, surgery_scratch);
    impact_from_surgery_witness_into(
        surgery_scratch,
        observation_mask,
        n as usize,
        closure_scratch,
        impact,
    );
}

/// Sequential observation-form change-impact witness.
#[must_use]
pub fn predict_impact_observation_form_witness(
    adjacency: &[u32],
    observation_mask: &[u32],
    n: u32,
) -> Vec<u32> {
    let mut surgery_scratch = Vec::new();
    let mut closure_scratch = Vec::new();
    let mut impact = Vec::new();
    predict_impact_observation_form_witness_into(
        adjacency,
        observation_mask,
        n,
        &mut surgery_scratch,
        &mut closure_scratch,
        &mut impact,
    );
    impact
}
