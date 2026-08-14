//! Worker sizing and the scoped-thread fan-out that prepares entries and proves
//! backends in parallel.

use crate::operation_selection::{prepare_entry, PreparedEntry, UnifiedEntry};
use crate::proof_timing::{emit_backend_proof_timing, emit_pair_proof_start, emit_pair_proof_timing};
use crate::reference_parity::compare_backend_against_reference;
use vyre_conform_spec::ConformanceResult;

pub(crate) struct PreparedEntryBatch {
    pub(crate) entries: Vec<PreparedEntry>,
    pub(crate) pairs: Vec<ConformanceResult>,
    pub(crate) any_failed: bool,
}

pub(crate) fn proof_worker_count(item_count: usize) -> usize {
    if item_count == 0 {
        return 0;
    }

    let requested = std::env::var("VYRE_CONFORM_PROOF_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|workers| *workers > 0);
    let detected = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(8);

    requested.unwrap_or(detected).min(item_count)
}

pub(crate) fn prepare_entries_in_parallel(
    entries: Vec<UnifiedEntry>,
    backends: &[&'static vyre_driver::BackendRegistration],
) -> PreparedEntryBatch {
    if entries.is_empty() {
        return PreparedEntryBatch {
            entries: Vec::new(),
            pairs: Vec::new(),
            any_failed: false,
        };
    }

    let worker_count = proof_worker_count(entries.len());
    let mut buckets = (0..worker_count).map(|_| Vec::new()).collect::<Vec<_>>();
    for (index, entry) in entries.into_iter().enumerate() {
        buckets[index % worker_count].push((index, entry));
    }

    let mut outcomes = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(buckets.len());
        for bucket in buckets {
            let ids = bucket
                .iter()
                .map(|(index, entry)| (*index, entry.id))
                .collect::<Vec<_>>();
            handles.push((
                ids,
                scope.spawn(move || {
                    bucket
                        .into_iter()
                        .map(|(index, entry)| {
                            let op_id = entry.id;
                            (index, op_id, prepare_entry(entry))
                        })
                        .collect::<Vec<_>>()
                }),
            ));
        }

        for (ids, handle) in handles {
            match handle.join() {
                Ok(mut worker_outcomes) => outcomes.append(&mut worker_outcomes),
                Err(payload) => {
                    let message = format!(
                        "proof preparation worker panicked: {}. Fix: witness preparation must return explicit fixture failures instead of unwinding.",
                        panic_message(payload)
                    );
                    outcomes.extend(
                        ids.into_iter()
                            .map(|(index, op_id)| (index, op_id, Err(message.clone()))),
                    );
                }
            }
        }
    });
    outcomes.sort_by_key(|(index, _, _)| *index);

    let mut prepared_entries = Vec::with_capacity(outcomes.len());
    let mut pairs = Vec::new();
    let mut any_failed = false;
    for (_, op_id, outcome) in outcomes {
        match outcome {
            Ok(prepared) => prepared_entries.push(prepared),
            Err(error) => {
                for backend in backends {
                    pairs.push(ConformanceResult {
                        op_id: op_id.into(),
                        backend_id: backend.id.to_string(),
                        passed: false,
                        message: error.clone(),
                        replay_capsule: None,
                    });
                }
                any_failed = true;
            }
        }
    }

    PreparedEntryBatch {
        entries: prepared_entries,
        pairs,
        any_failed,
    }
}

pub(crate) fn prove_backends_in_parallel(
    backends: &[&'static vyre_driver::BackendRegistration],
    prepared_entries: &[PreparedEntry],
) -> Vec<Vec<ConformanceResult>> {
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(backends.len());
        for &backend in backends {
            handles.push((
                backend,
                scope.spawn(move || prove_one_backend(backend, prepared_entries)),
            ));
        }

        let mut results = Vec::with_capacity(handles.len());
        for (backend, handle) in handles {
            match handle.join() {
                Ok(pairs) => results.push(pairs),
                Err(payload) => {
                    let message = format!(
                        "backend `{}` proof worker panicked: {}. Fix: proof workers must return pair failures instead of unwinding.",
                        backend.id,
                        panic_message(payload)
                    );
                    results.push(
                        prepared_entries
                            .iter()
                            .map(|entry| ConformanceResult {
                                op_id: entry.id.into(),
                                backend_id: backend.id.to_string(),
                                passed: false,
                                message: message.clone(),
                                replay_capsule: None,
                            })
                            .collect(),
                    );
                }
            }
        }
        results
    })
}

fn prove_one_backend(
    backend: &'static vyre_driver::BackendRegistration,
    prepared_entries: &[PreparedEntry],
) -> Vec<ConformanceResult> {
    let started = std::time::Instant::now();
    if prepared_entries.is_empty() {
        return Vec::new();
    }

    let worker_count = proof_worker_count(prepared_entries.len());
    let mut buckets = (0..worker_count).map(|_| Vec::new()).collect::<Vec<_>>();
    for (index, entry) in prepared_entries.iter().enumerate() {
        buckets[index % worker_count].push((index, entry));
    }

    let mut indexed_pairs = Vec::with_capacity(prepared_entries.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(buckets.len());
        for bucket in buckets {
            let ids = bucket
                .iter()
                .map(|(index, entry)| (*index, entry.id))
                .collect::<Vec<_>>();
            handles.push((
                ids,
                scope.spawn(move || {
                    bucket
                        .into_iter()
                        .map(|(index, entry)| {
                            emit_pair_proof_start(backend.id, entry.id);
                            let pair_started = std::time::Instant::now();
                            let pair = compare_backend_against_reference(backend, entry);
                            emit_pair_proof_timing(
                                backend.id,
                                entry.id,
                                pair.passed,
                                pair_started.elapsed(),
                            );
                            (index, pair)
                        })
                        .collect::<Vec<_>>()
                }),
            ));
        }

        for (ids, handle) in handles {
            match handle.join() {
                Ok(mut worker_pairs) => indexed_pairs.append(&mut worker_pairs),
                Err(payload) => {
                    let message = format!(
                        "backend `{}` proof shard worker panicked: {}. Fix: proof workers must return pair failures instead of unwinding.",
                        backend.id,
                        panic_message(payload)
                    );
                    indexed_pairs.extend(ids.into_iter().map(|(index, op_id)| {
                        (
                            index,
                            ConformanceResult {
                                op_id: op_id.into(),
                                backend_id: backend.id.to_string(),
                                passed: false,
                                message: message.clone(),
                                replay_capsule: None,
                            },
                        )
                    }));
                }
            }
        }
    });

    indexed_pairs.sort_by_key(|(index, _)| *index);
    let pairs = indexed_pairs
        .into_iter()
        .map(|(_, pair)| pair)
        .collect::<Vec<_>>();
    emit_backend_proof_timing(backend.id, pairs.len(), worker_count, started.elapsed());
    pairs
}

pub(crate) fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
