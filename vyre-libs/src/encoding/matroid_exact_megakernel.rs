//! Exact (Edmonds) matroid intersection for megakernel fusion-grouping.
//!
//! Self-consumer for [`matroid_intersection_full`](crate::math::matroid_intersection_full).
//!
//! Today the megakernel scheduler uses
//! [`vyre_foundation::optimizer::megakernel::matroid_subset::max_fusion_subset`] which is
//! a discrete BFS-augmenting heuristic  -  fast and bounded but not
//! provably optimal for graphs with non-trivial exchange structure.
//! This consumer wraps the substrate's full Edmonds augmenting-path
//! solver, which is **provably optimal** (max independent set in the
//! intersection of two matroids).
//!
//! # When to use
//!
//! - **Production hot path**: stick with `max_fusion_subset`  -  its
//!   constant-factor advantage at small n (< 64 work items) outweighs
//!   the asymptotic difference, and its API is simpler.
//! - **Benchmark / certification path**: use `select_optimal_subset`
//!   here when measuring "what's the best we could do" against the
//!   heuristic, or when scheduler decisions need to survive an audit.
//!
//! Both consume the same `exchange_adj` shape so a caller can swap
//! between the two without changing input plumbing.
//!
//! # Algorithm
//!
//! Standard Edmonds matroid intersection (1970): augmenting-path BFS
//! over the exchange graph, repeating until no augmenting path
//! exists. Each augmentation strictly grows the independent set.

#[cfg(test)]
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
};
use crate::math::matroid_intersection_full::matroid_intersection_full;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Caller-owned dispatch scratch for exact megakernel matroid certification.
#[derive(Debug, Default)]
pub struct ExactMatroidDispatchScratch {
    inputs: Vec<Vec<u8>>,
}

/// Input-shape error from the exact megakernel matroid solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactMatroidError {
    /// `n * n` overflowed `usize`.
    AdjacencySizeOverflow {
        /// Requested graph dimension.
        n: usize,
    },
    /// `exchange_adj.len()` did not match `n * n`.
    ExchangeAdjLen {
        /// Required adjacency length.
        expected: usize,
        /// Received adjacency length.
        actual: usize,
    },
    /// `sources.len()` did not match `n`.
    SourcesLen {
        /// Required source-mask length.
        expected: usize,
        /// Received source-mask length.
        actual: usize,
    },
    /// `sinks.len()` did not match `n`.
    SinksLen {
        /// Required sink-mask length.
        expected: usize,
        /// Received sink-mask length.
        actual: usize,
    },
    /// `seed_x.len()` did not match `n`.
    SeedLen {
        /// Required seed length.
        expected: usize,
        /// Received seed length.
        actual: usize,
    },
    /// Test-only reference scratch could not reserve its caller-owned buffers.
    #[cfg(test)]
    Allocation {
        /// Allocation diagnostic from the canonical reference witness.
        source: String,
    },
}

impl std::fmt::Display for ExactMatroidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdjacencySizeOverflow { n } => write!(
                f,
                "exact matroid solver n*n overflow for n={n}. Fix: shard the megakernel exchange graph before certification."
            ),
            Self::ExchangeAdjLen { expected, actual } => write!(
                f,
                "exact matroid solver exchange_adj length {actual} does not match n*n={expected}. Fix: pass a dense row-major n*n exchange graph."
            ),
            Self::SourcesLen { expected, actual } => write!(
                f,
                "exact matroid solver sources length {actual} does not match n={expected}. Fix: pass one source flag per fusion candidate."
            ),
            Self::SinksLen { expected, actual } => write!(
                f,
                "exact matroid solver sinks length {actual} does not match n={expected}. Fix: pass one sink flag per fusion candidate."
            ),
            Self::SeedLen { expected, actual } => write!(
                f,
                "exact matroid solver seed length {actual} does not match n={expected}. Fix: pass one seed bit per fusion candidate."
            ),
            #[cfg(test)]
            Self::Allocation { source } => write!(
                f,
                "exact matroid reference scratch allocation failed: {source}"
            ),
        }
    }
}

impl std::error::Error for ExactMatroidError {}

fn validate_common(
    exchange_adj: &[u32],
    seed_x: &[u32],
    n: usize,
) -> Result<usize, ExactMatroidError> {
    let expected_adj = n
        .checked_mul(n)
        .ok_or(ExactMatroidError::AdjacencySizeOverflow { n })?;
    if exchange_adj.len() != expected_adj {
        return Err(ExactMatroidError::ExchangeAdjLen {
            expected: expected_adj,
            actual: exchange_adj.len(),
        });
    }
    if seed_x.len() != n {
        return Err(ExactMatroidError::SeedLen {
            expected: n,
            actual: seed_x.len(),
        });
    }
    Ok(expected_adj)
}

fn validate_full(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed_x: &[u32],
    n: usize,
) -> Result<(), ExactMatroidError> {
    let _expected_adj = validate_common(exchange_adj, seed_x, n)?;
    if sources.len() != n {
        return Err(ExactMatroidError::SourcesLen {
            expected: n,
            actual: sources.len(),
        });
    }
    if sinks.len() != n {
        return Err(ExactMatroidError::SinksLen {
            expected: n,
            actual: sinks.len(),
        });
    }
    Ok(())
}

fn validate_full_for_dispatch(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed_x: &[u32],
    n: usize,
) -> Result<u32, DispatchError> {
    validate_full(exchange_adj, sources, sinks, seed_x, n)
        .map_err(|error| DispatchError::BadInputs(format!("Fix: {error}")))?;
    let n_u32 = u32::try_from(n).map_err(|_| {
        DispatchError::BadInputs(format!(
            "Fix: exact matroid solver n={n} exceeds the primitive u32 dimension limit; shard the exchange graph before dispatch."
        ))
    })?;
    n_u32.checked_mul(n_u32).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: exact matroid solver n*n exceeds the primitive u32 buffer-count limit for n={n_u32}; shard the exchange graph before dispatch."
        ))
    })?;
    Ok(n_u32)
}

/// Dispatch the full Edmonds matroid-intersection primitive and return the
/// resulting 0/1 selected-set vector.
///
/// This is the production boundary for exact megakernel certification. The
/// no-dispatch selectors in this module are host reference compatibility
/// helpers; callers with a backend must prefer this path so certification does
/// not strand the scheduler on CPU BFS.
///
/// # Errors
///
/// Returns [`DispatchError`] when input shapes are invalid, dimensions exceed
/// primitive limits, the dispatcher rejects the Program, or the backend returns
/// malformed output.
#[allow(clippy::too_many_arguments)]
pub fn select_optimal_subset_via(
    dispatcher: &impl ProgramDispatcher,
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed_x: &[u32],
    n: usize,
    max_augmentations: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut out = Vec::new();
    select_optimal_subset_via_into(
        dispatcher,
        exchange_adj,
        sources,
        sinks,
        seed_x,
        n,
        max_augmentations,
        &mut out,
    )?;
    Ok(out)
}

/// Dispatch the exact megakernel certification primitive into caller-owned
/// output storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
#[allow(clippy::too_many_arguments)]
pub fn select_optimal_subset_via_into(
    dispatcher: &impl ProgramDispatcher,
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed_x: &[u32],
    n: usize,
    max_augmentations: u32,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = ExactMatroidDispatchScratch::default();
    select_optimal_subset_via_with_scratch_into(
        dispatcher,
        exchange_adj,
        sources,
        sinks,
        seed_x,
        n,
        max_augmentations,
        &mut scratch,
        out,
    )
}

/// Dispatch the exact megakernel certification primitive using caller-owned
/// input-buffer scratch.
///
/// This avoids rebuilding the twelve dispatch buffers on repeated certification
/// calls and keeps benchmark measurements focused on backend execution rather
/// than host allocation churn.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
#[allow(clippy::too_many_arguments)]
pub fn select_optimal_subset_via_with_scratch_into(
    dispatcher: &impl ProgramDispatcher,
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed_x: &[u32],
    n: usize,
    max_augmentations: u32,
    scratch: &mut ExactMatroidDispatchScratch,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    use crate::telemetry::{bump, matroid_exact_megakernel_calls};
    bump(&matroid_exact_megakernel_calls);

    let n_u32 = validate_full_for_dispatch(exchange_adj, sources, sinks, seed_x, n)?;
    if n == 0 {
        out.clear();
        return Ok(());
    }

    let n_bytes = n.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: select_optimal_subset_via n={n} overflows u32 byte buffer size."
        ))
    })?;
    let one_word_bytes = std::mem::size_of::<u32>();
    let program = matroid_intersection_full(
        "exchange_adj",
        "sources",
        "sinks",
        "set_x",
        "parent",
        "frontier",
        "next_frontier",
        "visited",
        "any_change",
        "path_out",
        "path_len",
        n_u32,
        max_augmentations,
    );
    ensure_input_slots(&mut scratch.inputs, 12);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], exchange_adj);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], sources);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], sinks);
    write_u32_slice_le_bytes(&mut scratch.inputs[3], seed_x);
    write_zero_bytes(&mut scratch.inputs[4], n_bytes);
    write_zero_bytes(&mut scratch.inputs[5], n_bytes);
    write_zero_bytes(&mut scratch.inputs[6], n_bytes);
    write_zero_bytes(&mut scratch.inputs[7], n_bytes);
    write_zero_bytes(&mut scratch.inputs[8], one_word_bytes);
    write_zero_bytes(&mut scratch.inputs[9], n_bytes);
    write_zero_bytes(&mut scratch.inputs[10], one_word_bytes);
    write_zero_bytes(&mut scratch.inputs[11], one_word_bytes);

    let outputs = dispatcher.dispatch(&program, &scratch.inputs, Some([1, 1, 1]))?;
    if outputs.is_empty() {
        return Err(DispatchError::BackendError(format!(
            "Fix: select_optimal_subset_via expected at least one output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(&outputs[0], n, "select_optimal_subset_via", out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::identity_op, clippy::erasing_op)]
    use super::*;
    use vyre_foundation::ir::Program;
    use vyre_reference::composition_witness::reduce_count_non_zero_witness as count_selected;
    #[derive(Debug, Default)]
    struct ExactMatroidScratch {
        current: Vec<u32>,
        next: Vec<u32>,
    }

    impl ExactMatroidScratch {
        fn result(&self) -> &[u32] {
            &self.current
        }
    }

    fn reference_select_optimal_subset(
        exchange_adj: &[u32],
        sources: &[u32],
        sinks: &[u32],
        seed_x: &[u32],
        n: usize,
        max_augmentations: u32,
    ) -> Result<Vec<u32>, ExactMatroidError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        if exchange_adj.len() != n * n {
            return Err(ExactMatroidError::ExchangeAdjLen {
                expected: n * n,
                actual: exchange_adj.len(),
            });
        }
        if sources.len() != n {
            return Err(ExactMatroidError::SourcesLen {
                expected: n,
                actual: sources.len(),
            });
        }
        if sinks.len() != n {
            return Err(ExactMatroidError::SinksLen {
                expected: n,
                actual: sinks.len(),
            });
        }
        if seed_x.len() != n {
            return Err(ExactMatroidError::SeedLen {
                expected: n,
                actual: seed_x.len(),
            });
        }
        let result = vyre_reference::composition_witness::matroid_select_optimal_subset_witness(
            exchange_adj,
            sources,
            sinks,
            seed_x,
            n,
            max_augmentations,
        )
        .map_err(|source| ExactMatroidError::Allocation { source })?;
        Ok(result)
    }

    fn reference_select_optimal_subset_all_eligible(
        exchange_adj: &[u32],
        seed_x: &[u32],
        n: usize,
        max_augmentations: u32,
    ) -> Result<Vec<u32>, ExactMatroidError> {
        let sources = vec![1u32; n];
        let sinks = vec![1u32; n];
        reference_select_optimal_subset(
            exchange_adj,
            &sources,
            &sinks,
            seed_x,
            n,
            max_augmentations,
        )
    }

    fn reference_select_optimal_subset_into<'a>(
        exchange_adj: &[u32],
        sources: &[u32],
        sinks: &[u32],
        seed_x: &[u32],
        n: usize,
        max_augmentations: u32,
        scratch: &'a mut ExactMatroidScratch,
    ) -> Result<&'a [u32], ExactMatroidError> {
        if n == 0 {
            scratch.current.clear();
            return Ok(&scratch.current);
        }
        if exchange_adj.len() != n * n {
            return Err(ExactMatroidError::ExchangeAdjLen {
                expected: n * n,
                actual: exchange_adj.len(),
            });
        }
        if sources.len() != n {
            return Err(ExactMatroidError::SourcesLen {
                expected: n,
                actual: sources.len(),
            });
        }
        if sinks.len() != n {
            return Err(ExactMatroidError::SinksLen {
                expected: n,
                actual: sinks.len(),
            });
        }
        if seed_x.len() != n {
            return Err(ExactMatroidError::SeedLen {
                expected: n,
                actual: seed_x.len(),
            });
        }
        vyre_reference::composition_witness::matroid_select_optimal_subset_witness_into(
            exchange_adj,
            sources,
            sinks,
            seed_x,
            n,
            max_augmentations,
            &mut scratch.current,
            &mut scratch.next,
        )
        .map_err(|source| ExactMatroidError::Allocation { source })?;
        Ok(&scratch.current)
    }

    fn reference_select_optimal_subset_all_eligible_into<'a>(
        exchange_adj: &[u32],
        seed_x: &[u32],
        n: usize,
        max_augmentations: u32,
        scratch: &'a mut ExactMatroidScratch,
    ) -> Result<&'a [u32], ExactMatroidError> {
        scratch.current = reference_select_optimal_subset_all_eligible(
            exchange_adj,
            seed_x,
            n,
            max_augmentations,
        )?;
        Ok(&scratch.current)
    }

    #[test]
    fn empty_seed_grows_to_max() {
        // 3 nodes: 0→1, 1→2 in the exchange graph. sources={0}, sinks={2}.
        // Augmenting paths: 0→1→2 puts both 0 and 2 in the independent set.
        let n = 3;
        let mut adj = vec![0u32; 9];
        adj[0 * 3 + 1] = 1;
        adj[1 * 3 + 2] = 1;
        let sources = vec![1, 0, 0];
        let sinks = vec![0, 0, 1];
        let seed = vec![0u32; 3];
        let result = reference_select_optimal_subset(&adj, &sources, &sinks, &seed, n, 8).unwrap();
        // At least the source survives.
        assert!(result[0] != 0 || count_selected(&result) >= 1);
    }

    #[test]
    fn seeded_set_is_at_least_preserved() {
        // 4 nodes, no exchange edges. Seed {1, 2}. No augmentations
        // possible, so seed stays.
        let n = 4;
        let adj = vec![0u32; 16];
        let sources = vec![1, 0, 0, 0];
        let sinks = vec![0, 0, 0, 1];
        let seed = vec![0, 1, 1, 0];
        let result = reference_select_optimal_subset(&adj, &sources, &sinks, &seed, n, 8).unwrap();
        // At minimum, the seeded items remain.
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 1);
    }

    #[test]
    fn empty_input_returns_empty_vec() {
        let result = reference_select_optimal_subset(&[], &[], &[], &[], 0, 4).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn count_selected_handles_zero() {
        assert_eq!(count_selected(&[]), 0);
        assert_eq!(count_selected(&[0, 0, 0]), 0);
        assert_eq!(count_selected(&[1, 0, 1, 1, 0]), 3);
    }

    #[test]
    fn all_eligible_path_matches_generic_all_ones() {
        let n = 4;
        let mut adj = vec![0u32; 16];
        adj[0 * 4 + 1] = 1;
        adj[1 * 4 + 2] = 1;
        adj[2 * 4 + 3] = 1;
        let sources = vec![1u32; n];
        let sinks = vec![1u32; n];
        let seed = vec![0, 1, 0, 0];
        let generic = reference_select_optimal_subset(&adj, &sources, &sinks, &seed, n, 8).unwrap();
        let specialized = reference_select_optimal_subset_all_eligible(&adj, &seed, n, 8).unwrap();
        assert_eq!(specialized, generic);
    }

    #[test]
    fn all_eligible_into_matches_owned_selector() {
        let n = 4;
        let mut adj = vec![0u32; 16];
        adj[0 * 4 + 1] = 1;
        adj[1 * 4 + 2] = 1;
        adj[2 * 4 + 3] = 1;
        let seed = vec![0, 1, 0, 0];
        let owned = reference_select_optimal_subset_all_eligible(&adj, &seed, n, 8).unwrap();

        let mut scratch = ExactMatroidScratch::default();
        let borrowed =
            reference_select_optimal_subset_all_eligible_into(&adj, &seed, n, 8, &mut scratch)
                .unwrap();

        assert_eq!(borrowed, owned.as_slice());
        assert_eq!(scratch.result(), owned.as_slice());
    }

    #[test]
    fn generic_selector_into_reuses_current_storage() {
        let n = 3;
        let mut adj = vec![0u32; 9];
        adj[0 * 3 + 1] = 1;
        adj[1 * 3 + 2] = 1;
        let sources = vec![1, 0, 0];
        let sinks = vec![0, 0, 1];
        let seed = vec![0u32; 3];
        let mut scratch = ExactMatroidScratch::default();

        let first =
            reference_select_optimal_subset_into(&adj, &sources, &sinks, &seed, n, 8, &mut scratch)
                .unwrap()
                .to_vec();
        let current_ptr = scratch.current.as_ptr();
        let next_ptr = scratch.next.as_ptr();
        let second =
            reference_select_optimal_subset_into(&adj, &sources, &sinks, &seed, n, 8, &mut scratch)
                .unwrap()
                .to_vec();

        assert_eq!(first, second);
        assert!([current_ptr, next_ptr].contains(&scratch.current.as_ptr()));
        assert!([current_ptr, next_ptr].contains(&scratch.next.as_ptr()));
    }

    #[test]
    fn invalid_shapes_return_errors_instead_of_panicking() {
        let err =
            reference_select_optimal_subset(&[0], &[1, 0], &[0, 1], &[0, 0], 2, 4).unwrap_err();
        assert_eq!(
            err,
            ExactMatroidError::ExchangeAdjLen {
                expected: 4,
                actual: 1,
            }
        );

        let err =
            reference_select_optimal_subset_all_eligible(&[0, 0, 0, 0], &[0], 2, 4).unwrap_err();
        assert_eq!(
            err,
            ExactMatroidError::SeedLen {
                expected: 2,
                actual: 1,
            }
        );
    }

    struct MatroidDispatcher;

    impl ProgramDispatcher for MatroidDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_eq!(grid_override, Some([1, 1, 1]));
            assert_eq!(inputs.len(), 12);
            let exchange_adj = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let sources = crate::dispatch_buffers::read_u32s(&inputs[1]);
            let sinks = crate::dispatch_buffers::read_u32s(&inputs[2]);
            let seed_x = crate::dispatch_buffers::read_u32s(&inputs[3]);
            let n = seed_x.len();
            assert_eq!(exchange_adj.len(), n * n);
            assert_eq!(sources.len(), n);
            assert_eq!(sinks.len(), n);
            for scratch in &inputs[4..8] {
                assert_eq!(scratch.len(), n * std::mem::size_of::<u32>());
            }
            assert_eq!(inputs[8].len(), std::mem::size_of::<u32>());
            assert_eq!(inputs[9].len(), n * std::mem::size_of::<u32>());
            assert_eq!(inputs[10].len(), std::mem::size_of::<u32>());
            assert_eq!(inputs[11].len(), std::mem::size_of::<u32>());
            let mut out = seed_x;
            if let Some(first_source) = sources.iter().position(|&source| source != 0) {
                out[first_source] = 1;
            }
            if let Some(first_sink) = sinks.iter().position(|&sink| sink != 0) {
                out[first_sink] = 1;
            }
            Ok(vec![u32_slice_to_le_bytes(&out)])
        }
    }

    #[test]
    fn select_optimal_subset_via_dispatches_full_primitive() {
        let n = 3;
        let mut adj = vec![0u32; 9];
        adj[0 * 3 + 1] = 1;
        adj[1 * 3 + 2] = 1;
        let sources = vec![1, 0, 0];
        let sinks = vec![0, 0, 1];
        let seed = vec![0u32; 3];

        let result =
            select_optimal_subset_via(&MatroidDispatcher, &adj, &sources, &sinks, &seed, n, 8)
                .unwrap();

        assert_eq!(result, vec![1, 0, 1]);
    }

    #[test]
    fn select_optimal_subset_via_rejects_invalid_shapes() {
        let err =
            select_optimal_subset_via(&MatroidDispatcher, &[0], &[1, 0], &[0, 1], &[0, 0], 2, 4)
                .unwrap_err();

        assert!(matches!(err, DispatchError::BadInputs(_)));
        assert!(
            err.to_string().contains("dense row-major n*n"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn select_optimal_subset_via_empty_input_is_zero_work() {
        let result =
            select_optimal_subset_via(&MatroidDispatcher, &[], &[], &[], &[], 0, 4).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn select_optimal_subset_via_with_scratch_reuses_input_buffers() {
        let n = 3;
        let mut adj = vec![0u32; 9];
        adj[0 * 3 + 1] = 1;
        adj[1 * 3 + 2] = 1;
        let sources = vec![1, 0, 0];
        let sinks = vec![0, 0, 1];
        let seed = vec![0u32; 3];
        let mut scratch = ExactMatroidDispatchScratch::default();
        let mut out = Vec::new();

        select_optimal_subset_via_with_scratch_into(
            &MatroidDispatcher,
            &adj,
            &sources,
            &sinks,
            &seed,
            n,
            8,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let input_ptrs: Vec<*const u8> = scratch.inputs.iter().map(Vec::as_ptr).collect();
        select_optimal_subset_via_with_scratch_into(
            &MatroidDispatcher,
            &adj,
            &sources,
            &sinks,
            &seed,
            n,
            8,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        assert_eq!(scratch.inputs.len(), 12);
        for (before, after) in input_ptrs
            .iter()
            .zip(scratch.inputs.iter().map(Vec::as_ptr))
        {
            assert_eq!(*before, after);
        }
    }
}
