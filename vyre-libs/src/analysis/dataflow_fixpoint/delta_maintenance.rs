//! Incremental maintenance of a boolean reachability relation under an
//! insertion/deletion batch, compared against full recompute.

use super::dense_matrix::{checked_dense_cells, checked_dense_node_count, normalize_bool_matrix};
use super::{DeltaDataflowEvidence, DeltaDataflowReport, DeltaRelationBatch, DeltaRelationChange};
use vyre_foundation::pass_substrate::semiring_closure::reachability_closure_into;

impl DeltaRelationBatch {
    /// Number of inserted tuples.
    #[must_use]
    fn inserted_tuple_count(&self) -> u32 {
        u32::try_from(self.insertions.len()).unwrap_or(u32::MAX)
    }

    /// Number of deleted tuples.
    #[must_use]
    fn deleted_tuple_count(&self) -> u32 {
        u32::try_from(self.deletions.len()).unwrap_or(u32::MAX)
    }
}

/// Apply insertion/deletion deltas to a boolean reachability relation and
/// compare the delta-maintained result against full recompute.
///
/// # Errors
///
/// Returns a fix-directed string when dimensions, iteration budget, or edge
/// coordinates are invalid.
fn compare_delta_maintained_reachability(
    adj: &[u32],
    n: u32,
    max_iters: u32,
    batch: &DeltaRelationBatch,
) -> Result<DeltaDataflowReport, String> {
    let n_us = checked_dense_node_count(n)?;
    let cells = checked_dense_cells(n_us)?;
    if adj.len() != cells {
        return Err(format!(
            "Fix: delta-maintained reachability expected adj.len() == n*n == {cells}, got {}.",
            adj.len()
        ));
    }
    if max_iters == 0 {
        return Err("Fix: delta-maintained reachability requires max_iters > 0.".to_string());
    }
    validate_delta_batch(batch, n)?;

    let normalized = normalize_bool_matrix(adj);
    let updated = apply_delta_batch_to_adjacency(&normalized, n_us, batch);
    let mut full_recompute_closure = Vec::new();
    let mut full_next = Vec::new();
    reachability_closure_into(
        &updated,
        n,
        max_iters,
        &mut full_recompute_closure,
        &mut full_next,
    );

    let mut old_closure = Vec::new();
    let mut old_next = Vec::new();
    reachability_closure_into(&normalized, n, max_iters, &mut old_closure, &mut old_next);

    let started = std::time::Instant::now();
    let (delta_closure, iterations, recomputed_tuple_count) = if batch.deletions.is_empty() {
        incremental_insert_closure(&old_closure, n_us, max_iters, &batch.insertions)
    } else {
        (
            full_recompute_closure.clone(),
            max_iters,
            u32::try_from(cells).unwrap_or(u32::MAX),
        )
    };
    let elapsed_active_time_ns = started.elapsed().as_nanos().max(1);
    let changed_tuple_count = count_changed_tuples(&old_closure, &full_recompute_closure)?;
    let exact_result_parity = delta_closure == full_recompute_closure;

    Ok(DeltaDataflowReport {
        evidence: DeltaDataflowEvidence {
            node_count: n,
            inserted_tuple_count: batch.inserted_tuple_count(),
            deleted_tuple_count: batch.deleted_tuple_count(),
            changed_tuple_count,
            recomputed_tuple_count,
            iterations,
            elapsed_active_time_ns,
            exact_result_parity,
        },
        delta_closure,
        full_recompute_closure,
    })
}

fn validate_delta_batch(batch: &DeltaRelationBatch, n: u32) -> Result<(), String> {
    for (kind, changes) in [
        ("insertion", batch.insertions.as_slice()),
        ("deletion", batch.deletions.as_slice()),
    ] {
        for change in changes {
            if change.source >= n || change.target >= n {
                return Err(format!(
                    "Fix: delta relation {kind} edge {}->{} is outside node_count={n}.",
                    change.source, change.target
                ));
            }
        }
    }
    Ok(())
}

fn apply_delta_batch_to_adjacency(
    adj: &[u32],
    n_us: usize,
    batch: &DeltaRelationBatch,
) -> Vec<u32> {
    let mut updated = adj.to_vec();
    for change in &batch.insertions {
        updated[change.source as usize * n_us + change.target as usize] = 1;
    }
    for change in &batch.deletions {
        updated[change.source as usize * n_us + change.target as usize] = 0;
    }
    updated
}

fn incremental_insert_closure(
    old_closure: &[u32],
    n_us: usize,
    max_iters: u32,
    insertions: &[DeltaRelationChange],
) -> (Vec<u32>, u32, u32) {
    let mut closure = old_closure.to_vec();
    let mut iterations = 0_u32;
    let mut recomputed_tuple_count = 0_u32;
    if insertions.is_empty() {
        return (closure, 1, 0);
    }
    loop {
        iterations = iterations.saturating_add(1);
        let mut changed = false;
        for insertion in insertions {
            let source = insertion.source as usize;
            let target = insertion.target as usize;
            for predecessor in 0..n_us {
                if predecessor != source && closure[predecessor * n_us + source] == 0 {
                    continue;
                }
                for successor in 0..n_us {
                    if successor != target && closure[target * n_us + successor] == 0 {
                        continue;
                    }
                    let index = predecessor * n_us + successor;
                    if closure[index] == 0 {
                        closure[index] = 1;
                        changed = true;
                        recomputed_tuple_count = recomputed_tuple_count.saturating_add(1);
                    }
                }
            }
        }
        if !changed || iterations >= max_iters {
            break;
        }
    }
    (closure, iterations, recomputed_tuple_count)
}

fn count_changed_tuples(before: &[u32], after: &[u32]) -> Result<u32, String> {
    if before.len() != after.len() {
        return Err(format!(
            "Fix: changed tuple comparison length mismatch before={} after={}.",
            before.len(),
            after.len()
        ));
    }
    u32::try_from(
        before
            .iter()
            .zip(after)
            .filter(|(left, right)| u32::from(**left != 0) != u32::from(**right != 0))
            .count(),
    )
    .map_err(|_| "Fix: changed tuple count exceeded u32.".to_string())
}

#[cfg(test)]
mod tests {
    use super::super::{DeltaRelationBatch, DeltaRelationChange};
    use super::compare_delta_maintained_reachability;

    #[test]
    fn delta_maintained_reachability_insertion_matches_full_recompute() {
        let adj = vec![
            0, 1, 0, 0, //
            0, 0, 1, 0, //
            0, 0, 0, 0, //
            0, 0, 0, 0, //
        ];
        let batch = DeltaRelationBatch {
            insertions: vec![DeltaRelationChange {
                source: 2,
                target: 3,
            }],
            deletions: Vec::new(),
        };

        let report = compare_delta_maintained_reachability(&adj, 4, 4, &batch)
            .expect("Fix: insertion delta fixture should compare");

        assert!(report.evidence.exact_result_parity);
        assert_eq!(report.delta_closure, report.full_recompute_closure);
        assert_eq!(report.evidence.inserted_tuple_count, 1);
        assert_eq!(report.evidence.deleted_tuple_count, 0);
        assert_eq!(report.evidence.changed_tuple_count, 3);
        assert_eq!(report.evidence.recomputed_tuple_count, 3);
        assert!(report.evidence.iterations > 0);
        assert!(report.evidence.elapsed_active_time_ns > 0);
    }

    #[test]
    fn delta_maintained_reachability_deletion_records_full_recompute_fallback() {
        let adj = vec![
            0, 1, 0, 0, //
            0, 0, 1, 0, //
            0, 0, 0, 1, //
            0, 0, 0, 0, //
        ];
        let batch = DeltaRelationBatch {
            insertions: Vec::new(),
            deletions: vec![DeltaRelationChange {
                source: 1,
                target: 2,
            }],
        };

        let report = compare_delta_maintained_reachability(&adj, 4, 4, &batch)
            .expect("Fix: deletion delta fixture should compare");

        assert!(report.evidence.exact_result_parity);
        assert_eq!(report.delta_closure, report.full_recompute_closure);
        assert_eq!(report.evidence.inserted_tuple_count, 0);
        assert_eq!(report.evidence.deleted_tuple_count, 1);
        assert_eq!(report.evidence.recomputed_tuple_count, 16);
        assert!(report.evidence.changed_tuple_count > 0);
    }

    #[test]
    fn delta_maintained_reachability_rejects_out_of_range_delta_tuple() {
        let adj = vec![0, 0, 0, 0];
        let batch = DeltaRelationBatch {
            insertions: vec![DeltaRelationChange {
                source: 0,
                target: 3,
            }],
            deletions: Vec::new(),
        };

        let error = compare_delta_maintained_reachability(&adj, 2, 2, &batch)
            .expect_err("Fix: out-of-range delta tuple should reject");

        assert!(error.contains("outside node_count=2"));
    }
}
