//! Sequential mathematical witnesses for do-calculus, interventions, and causal change-impact prediction.

/// Sequential witness for do-calculus Rule 2 edge reversal.
#[must_use]
pub fn do_rule2_reverse_incoming_witness(
    adjacency: &[u32],
    treatment_mask: &[u32],
    n: u32,
) -> Vec<u32> {
    let n = n as usize;
    assert_eq!(adjacency.len(), n * n);
    assert_eq!(treatment_mask.len(), n);
    let mut output = vec![0; n * n];
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
            output[source * n + destination] = original | reversed;
        }
    }
    output
}

/// Sequential witness for intervention deletion of every incoming edge.
#[must_use]
pub fn do_intervention_delete_incoming_witness(
    adjacency: &[u32],
    intervention_mask: &[u32],
    n: u32,
) -> Vec<u32> {
    let n = n as usize;
    assert_eq!(adjacency.len(), n * n);
    assert_eq!(intervention_mask.len(), n);
    let mut output = adjacency.to_vec();
    for source in 0..n {
        for destination in 0..n {
            if intervention_mask[destination] != 0 {
                output[source * n + destination] = 0;
            }
        }
    }
    output
}

/// Sequential witness for dense Rule 3 subgraph extraction.
#[must_use]
pub fn do_rule3_subgraph_witness(
    adjacency: &[u32],
    keep_mask: &[u32],
    n: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = n as usize;
    assert_eq!(adjacency.len(), n * n);
    assert_eq!(keep_mask.len(), n);
    let kept = keep_mask
        .iter()
        .enumerate()
        .filter_map(|(index, &keep)| (keep != 0).then_some(index as u32))
        .collect::<Vec<_>>();
    let mut reduced = Vec::with_capacity(kept.len() * kept.len());
    for &source in &kept {
        for &destination in &kept {
            reduced.push(adjacency[source as usize * n + destination as usize]);
        }
    }
    (reduced, kept)
}

fn reachability_closure_witness(adjacency: &[u32], n: usize) -> Vec<u32> {
    assert_eq!(adjacency.len(), n * n);
    let mut closure = adjacency.iter().map(|&edge| u32::from(edge != 0)).collect::<Vec<_>>();
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
    closure
}

fn impact_from_surgery_witness(
    surgery: &[u32],
    intervention_mask: &[u32],
    n: usize,
) -> Vec<u32> {
    let closure = reachability_closure_witness(surgery, n);
    let mut impact = vec![0; n];
    for source in 0..n {
        if intervention_mask[source] == 0 {
            continue;
        }
        impact[source] = 1;
        for destination in 0..n {
            if closure[source * n + destination] != 0 {
                impact[destination] = 1;
            }
        }
    }
    impact
}

/// Sequential intervention-form change-impact witness.
#[must_use]
pub fn predict_impact_witness(
    adjacency: &[u32],
    intervention_mask: &[u32],
    n: u32,
) -> Vec<u32> {
    let surgery = do_intervention_delete_incoming_witness(adjacency, intervention_mask, n);
    impact_from_surgery_witness(&surgery, intervention_mask, n as usize)
}

/// Sequential observation-form change-impact witness.
#[must_use]
pub fn predict_impact_observation_form_witness(
    adjacency: &[u32],
    observation_mask: &[u32],
    n: u32,
) -> Vec<u32> {
    let surgery = do_rule2_reverse_incoming_witness(adjacency, observation_mask, n);
    impact_from_surgery_witness(&surgery, observation_mask, n as usize)
}
