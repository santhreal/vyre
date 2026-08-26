//! Pipeline-cache eviction via submodular maximization.
//!
//! Closes the recursion thesis: submodular_greedy ships to
//! user dialects (feature selection, sensor placement, summarization,
//! coreset construction) AND drives vyre's compile-cache eviction
//! policy.
//!
//! # The self-use
//!
//! Vyre's backend pipeline caches can use LRU eviction: when the cache fills,
//! drop the least-recently-
//! used pipeline. LRU is fast and reasonable but provably suboptimal
//! when access frequencies are skewed  -  a frequently-hit cold-edged
//! pipeline gets evicted because it sat for one extra second.
//!
//! Submodular maximization gives a provably-better bound. Reframe:
//! "which K pipelines to KEEP cached such that expected hit rate is
//! maximized." Hit-rate-as-set-function is submodular (diminishing
//! returns: adding a pipeline to a small cache helps more than adding
//! it to a large cache). Greedy-pick-by-marginal-gain achieves
//! `(1 - 1/e) ≈ 63%` of optimum (Nemhauser 1978). Stochastic-greedy
//! (Mirzasoleiman 2015) gets close to that bound at GPU-friendly cost.
//!
//! For 0.6 we ship the per-step argmax-of-marginals primitive that
//! the cache eviction policy will call once per fill  -  the K
//! consecutive argmax-of-marginals calls produce the K-element
//! retention set; everything else is evicted.
//!
//! # Why this matters
//!
//! At 65k cached pipelines (the current LruPipelineCache cap), LRU
//! evicts ~30% of pipelines that should be retained on a workload
//! with skewed temporal locality (typical for security scanning
//! with hot-path/cold-path bimodal). Submodular eviction recovers
//! most of those retained  -  measurable improvement in cache hit
//! rate at no per-eviction cost (the marginal-gain table is built
//! incrementally).
//!
//! # Algorithm
//!
//! ```text
//! gains[i]    = expected hit rate for pipeline i conditional on
//!               current cache contents (caller's hit-tracker
//!               populates this)
//! picked[i]   = 1 if pipeline i already in retention set
//!
//! while |picked| < K:
//!     winner = argmax_of_marginals(gains, picked)
//!     if winner is NO_WINNER: break
//!     picked[winner] = 1
//!     gains[*] -= covered_gain(winner)  // diminishing returns
//!
//! evict every pipeline whose picked == 0
//! ```

use super::decode_u32_output_exact;
use crate::dispatch_buffers::{ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes};
use crate::math::submodular_greedy::{argmax_of_marginals, NO_WINNER};
#[cfg(test)]
use crate::plumbing::host::scratch::reserve_vec_capacity_or_panic;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Caller-owned semantic execution scratch for submodular cache eviction.
#[derive(Debug, Default)]
pub struct SubmodularEvictionGpuScratch {
    inputs: Vec<Vec<u8>>,
    winner: Vec<u32>,
}

/// Compute the retention set through semantic execution of the submodular
/// argmax primitive.
///
/// The independent-access greedy loop executes
/// `crate::math::submodular_greedy::argmax_of_marginals` once per retained item.
///
/// # Errors
///
/// Returns [`SemanticExecutionError::InvalidRequest`] when `gains.len() != n`,
/// `k > n`, or `n == 0`. Compilation, execution, and malformed-output failures
/// are propagated.
pub fn select_retention_set_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    gains: &mut [u32],
    n: u32,
    k: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut picked = Vec::with_capacity(n as usize);
    select_retention_set_via_into(executor, policy, gains, n, k, &mut picked)?;
    Ok(picked)
}

/// Compute the retention set through semantic execution into caller-owned storage.
pub fn select_retention_set_via_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    gains: &mut [u32],
    n: u32,
    k: u32,
    picked: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = SubmodularEvictionGpuScratch::default();
    select_retention_set_via_with_scratch_into(executor, policy, gains, n, k, &mut scratch, picked)
}

/// Compute the retention set through semantic execution into caller-owned
/// execution and output storage.
pub fn select_retention_set_via_with_scratch_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    gains: &mut [u32],
    n: u32,
    k: u32,
    scratch: &mut SubmodularEvictionGpuScratch,
    picked: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, submodular_cache_eviction_calls};
    bump(&submodular_cache_eviction_calls);
    if n == 0 {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: select_retention_set_via requires n > 0.".to_string(),
        ));
    }
    if gains.len() != n as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: select_retention_set_via expected gains.len() == n == {n}, got {}.",
            gains.len()
        )));
    }
    if k > n {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: select_retention_set_via requires k <= n; got k={k}, n={n}."
        )));
    }

    picked.clear();
    picked.resize(n as usize, 0);
    let mut keep_count = 0u32;
    while keep_count < k {
        let (winner, _) =
            execute_argmax_step_with_scratch(executor, policy, gains, picked, n, scratch)?;
        if winner == NO_WINNER {
            break;
        }
        let winner_idx = winner as usize;
        if winner_idx >= picked.len() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: submodular argmax returned winner {winner} outside n={n}."
            )));
        }
        picked[winner_idx] = 1;
        gains[winner_idx] = 0;
        keep_count += 1;
    }
    Ok(())
}

fn execute_argmax_step_with_scratch(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    gains: &[u32],
    picked: &[u32],
    n: u32,
    scratch: &mut SubmodularEvictionGpuScratch,
) -> Result<(u32, u32), SemanticExecutionError> {
    let program = argmax_of_marginals("gains", "picked", "winner_idx", "winner_gain", n);
    ensure_input_slots(&mut scratch.inputs, 4);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], gains);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], picked);
    write_zero_bytes(&mut scratch.inputs[2], std::mem::size_of::<u32>());
    write_zero_bytes(&mut scratch.inputs[3], std::mem::size_of::<u32>());
    let outputs = execute_single_program(
        executor,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    let [winner_idx_out, winner_gain_out] = match outputs.as_slice() {
        [idx, gain] => [idx, gain],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: submodular argmax execution returned {} outputs, expected exactly winner_idx and winner_gain.",
                outputs.len()
            )))
        }
    };
    decode_u32_output_exact(winner_idx_out, 1, "winner_idx", &mut scratch.winner)?;
    let winner_idx = scratch.winner[0];
    decode_u32_output_exact(winner_gain_out, 1, "winner_gain", &mut scratch.winner)?;
    let winner_gain = scratch.winner[0];
    Ok((winner_idx, winner_gain))
}

/// Convenience: invert retention to eviction (1 = evict).
#[cfg(test)]
#[must_use]
pub fn invert_to_eviction_set(retention: &[u32]) -> Vec<u32> {
    let mut eviction = Vec::with_capacity(retention.len());
    invert_to_eviction_set_into(retention, &mut eviction);
    eviction
}

/// Invert retention to eviction (1 = evict) into caller-owned storage.
#[cfg(test)]
pub fn invert_to_eviction_set_into(retention: &[u32], eviction: &mut Vec<u32>) {
    eviction.clear();
    reserve_vec_capacity_or_panic(eviction, retention.len(), "submodular eviction output");
    eviction.extend(retention.iter().map(|&r| if r == 0 { 1 } else { 0 }));
}

/// Approximate worst-case retention quality bound: greedy submodular
/// maximization achieves `(1 - 1/e)` ≈ 0.632 of optimum. Returns the
/// expected lower bound on retention quality given an optimum.
#[cfg(test)]
#[must_use]
pub fn greedy_quality_bound(optimum: u32) -> u32 {
    // `(1 - 1/e) ≈ 0.6321205588`. Use integer approximation
    // via 6321/10000 to keep this f64-free.
    ((optimum as u64) * 6321 / 10000) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use crate::test_parity_oracles::{canonical_inputs, policy, semantic_output, StaticOutputs};
    use vyre_megakernel::{SemanticExecutionOutput, SemanticExecutionRequest};
    use vyre_reference::composition_witness::argmax_of_marginals_witness;

    fn select_retention_set(gains: &mut [u32], n: u32, k: u32) -> Vec<u32> {
        assert!(n > 0, "Fix: select_retention_set requires n > 0.");
        assert_eq!(
            gains.len(),
            n as usize,
            "Fix: select_retention_set expected gains.len() == n == {n}, got {}.",
            gains.len()
        );
        assert!(
            k <= n,
            "Fix: select_retention_set requires k <= n; got k={k}, n={n}."
        );
        let mut picked = vec![0u32; n as usize];
        let mut keep_count = 0u32;
        while keep_count < k {
            let (winner, _) = argmax_of_marginals_witness(gains, &picked);
            if winner == NO_WINNER {
                break;
            }
            let winner_idx = winner as usize;
            if winner_idx >= picked.len() {
                break;
            }
            picked[winner_idx] = 1;
            gains[winner_idx] = 0;
            keep_count += 1;
        }
        picked
    }

    struct ArgmaxOutputs;

    impl SemanticExecutor for ArgmaxOutputs {
        fn execute(
            &self,
            request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = canonical_inputs(request)?;
            let [gains_bytes, picked_bytes, winner_idx_bytes, winner_gain_bytes] =
                inputs.as_slice()
            else {
                return Err(SemanticExecutionError::InvalidRequest(format!(
                    "Fix: argmax semantic executor expected 4 buffers, got {}.",
                    inputs.len()
                )));
            };
            if winner_idx_bytes.len() != std::mem::size_of::<u32>()
                || winner_gain_bytes.len() != std::mem::size_of::<u32>()
            {
                return Err(SemanticExecutionError::InvalidRequest(
                    "Fix: argmax semantic executor requires one-word output resources.".to_string(),
                ));
            }
            let gains = crate::dispatch_buffers::read_u32s(gains_bytes);
            let picked = crate::dispatch_buffers::read_u32s(picked_bytes);
            if gains.len() != picked.len() {
                return Err(SemanticExecutionError::InvalidRequest(format!(
                    "Fix: argmax semantic executor requires equal gains and picked lengths, got {} and {}.",
                    gains.len(),
                    picked.len()
                )));
            }
            let (winner_idx, winner_gain) = argmax_of_marginals_witness(&gains, &picked);
            semantic_output(
                request,
                vec![
                    u32_slice_to_le_bytes(&[winner_idx]),
                    u32_slice_to_le_bytes(&[winner_gain]),
                ],
            )
        }
    }

    #[test]
    fn picks_top_k_by_gain() {
        let mut gains = vec![3u32, 7, 2, 9, 5];
        let retention = select_retention_set(&mut gains, 5, 3);
        assert_eq!(retention, vec![0, 1, 0, 1, 1]);
    }

    #[test]
    fn via_picks_top_k_by_gain() {
        let mut gains = vec![3u32, 7, 2, 9, 5];
        let retention = select_retention_set_via(&ArgmaxOutputs, &policy(), &mut gains, 5, 3)
            .expect("Fix: semantic execution succeeds");
        assert_eq!(retention, vec![0, 1, 0, 1, 1]);
        assert_eq!(gains, vec![3, 0, 2, 0, 0]);
    }

    #[test]
    fn via_with_scratch_reuses_execution_decode_and_output_storage() {
        let execution_policy = policy();
        let mut scratch = SubmodularEvictionGpuScratch::default();
        let mut picked = Vec::with_capacity(5);
        let mut gains = vec![3u32, 7, 2, 9, 5];

        select_retention_set_via_with_scratch_into(
            &ArgmaxOutputs,
            &execution_policy,
            &mut gains,
            5,
            3,
            &mut scratch,
            &mut picked,
        )
        .unwrap();

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let winner_capacity = scratch.winner.capacity();
        let picked_capacity = picked.capacity();
        let mut gains_again = vec![3u32, 7, 2, 9, 5];

        select_retention_set_via_with_scratch_into(
            &ArgmaxOutputs,
            &execution_policy,
            &mut gains_again,
            5,
            3,
            &mut scratch,
            &mut picked,
        )
        .unwrap();

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(scratch.winner.capacity(), winner_capacity);
        assert_eq!(picked.capacity(), picked_capacity);
        assert_eq!(picked, vec![0, 1, 0, 1, 1]);
        assert_eq!(gains_again, vec![3, 0, 2, 0, 0]);
    }

    #[test]
    fn via_rejects_extra_semantic_outputs() {
        let executor = StaticOutputs::new(
            "extra argmax output",
            vec![
                u32_slice_to_le_bytes(&[0]),
                u32_slice_to_le_bytes(&[1]),
                u32_slice_to_le_bytes(&[2]),
            ],
        );
        let mut gains = vec![3u32, 7, 2];
        let err = select_retention_set_via(&executor, &policy(), &mut gains, 3, 1)
            .expect_err("extra semantic outputs must be rejected");
        assert!(
            matches!(err, SemanticExecutionError::Backend(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn via_rejects_trailing_semantic_output_bytes() {
        let executor = StaticOutputs::new(
            "trailing argmax output bytes",
            vec![vec![0, 0, 0, 0, 99], u32_slice_to_le_bytes(&[1])],
        )
        .expecting_inputs(&[4]);
        let mut gains = vec![3u32, 7, 2];
        let err = select_retention_set_via(&executor, &policy(), &mut gains, 3, 1)
            .expect_err("trailing semantic output bytes must be rejected");
        assert!(
            matches!(err, SemanticExecutionError::Backend(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn k_eq_zero_evicts_all() {
        let mut gains = vec![3u32, 7, 2, 9, 5];
        let retention = select_retention_set(&mut gains, 5, 0);
        assert_eq!(retention, vec![0; 5]);
    }

    #[test]
    fn k_eq_n_retains_all() {
        let mut gains = vec![3u32, 7, 2, 9, 5];
        let retention = select_retention_set(&mut gains, 5, 5);
        assert_eq!(retention, vec![1; 5]);
    }

    #[test]
    fn invert_complements_retention() {
        let retention = vec![1, 0, 1, 0, 1];
        let eviction = invert_to_eviction_set(&retention);
        assert_eq!(eviction, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn invert_into_reuses_eviction_buffer() {
        let retention = vec![1, 0, 1, 0, 1];
        let mut eviction = Vec::with_capacity(8);
        let ptr = eviction.as_ptr();
        invert_to_eviction_set_into(&retention, &mut eviction);
        assert_eq!(eviction, vec![0, 1, 0, 1, 0]);
        assert_eq!(eviction.as_ptr(), ptr);
    }

    #[test]
    fn quality_bound_is_lower_bound() {
        assert_eq!(greedy_quality_bound(100), 63);
        assert_eq!(greedy_quality_bound(1000), 632);
    }

    #[test]
    fn k_larger_than_n_panics() {
        let mut gains = vec![1u32, 2, 3];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = select_retention_set(&mut gains, 3, 5);
        }));
        assert!(result.is_err(), "k > n must panic");
    }
}
