//! Tier 3 - Parity: drives the ACTUAL Datalog-fixpoint IR (`math::scallop_join`, a monotone Lineage
//! join iterated to convergence via a ping-pong state/next + `changed` flag) through `reference_eval`
//! and asserts BIT-EXACT equality against the shipped `scallop_join::cpu_ref`. The op had NO
//! `reference_eval` test.
//!
//! Like sinkhorn this is an EXACT (not tolerance) comparison: both the IR and `cpu_ref` iterate the
//! identical `state <- lineage_gemm(state, join_rules)` (OR-of-bitset combine+accumulate, monotone) to
//! the same fixpoint, so the whole data-dependent convergence loop is deterministic and must agree
//! bit-for-bit. This is the STATIC-INDEX fixpoint class `reference_eval` executes faithfully (the join
//! reads `state[i*n+k]`/`join_rules[k*n+j]` by loop index, never a data-derived scatter target). The
//! final state lands in `state` (binding 0). A wrong join index, a swapped state/next ping-pong, a
//! non-monotone combine, or an off-by-one convergence check breaks the exact match.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use vyre_reference::value::Value;

use vyre_primitives::math::scallop_join::{cpu_ref, scallop_join};

fn pack(data: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(data))
}

fn words(v: &Value) -> Vec<u32> {
    v.to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Run the IR and return the final `state` (binding 0, first RW buffer).
fn run_ir(state: &[u32], join_rules: &[u32], n: u32, max_iterations: u32) -> Vec<u32> {
    let cells = (n * n) as usize;
    let program = scallop_join("state", "next", "join_rules", "changed", n, max_iterations);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(state),              // state (0, RW) <- final
            pack(&vec![0u32; cells]), // next (1, RW)
            pack(&[0u32]),            // changed (2, RW)
            pack(join_rules),         // join_rules (3, RO)
        ],
    )
    .expect("scallop_join reference evaluation must succeed");
    words(&outputs[0])
}

#[test]
fn scallop_join_ir_matches_lineage_fixpoint_oracle() {
    let n = 2u32;
    let max_iterations = 8u32;
    // Lineage bitset seed (2x2 row-major): each cell is a provenance bitmask.
    let state = vec![0b0001u32, 0b0000, 0b0000, 0b0010];
    // Join rules: transitions that OR-accumulate new provenance into the fixpoint.
    let join_rules = vec![0b0001u32, 0b0100, 0b0010, 0b0001];

    let got = run_ir(&state, &join_rules, n, max_iterations);
    let (want, iters) = cpu_ref(&state, &join_rules, n, max_iterations);

    // Non-vacuous: the monotone fixpoint must actually derive new facts (grow past the seed).
    assert!(
        iters >= 1,
        "fixpoint must run at least one iteration, got {iters}"
    );
    assert_ne!(
        want, state,
        "fixpoint must derive new provenance beyond the seed"
    );
    assert_eq!(
        got, want,
        "state diverged: IR={got:?} oracle={want:?} (iters={iters})"
    );
}

#[test]
fn scallop_join_ir_matches_oracle_dense_and_already_converged() {
    let n = 3u32;
    // A denser 3x3 lineage that takes several iterations to close.
    let state = vec![
        0b001, 0b000, 0b000, //
        0b000, 0b010, 0b000, //
        0b000, 0b000, 0b100,
    ];
    let join_rules = vec![
        0b001, 0b010, 0b000, //
        0b000, 0b010, 0b100, //
        0b100, 0b000, 0b100,
    ];
    let got = run_ir(&state, &join_rules, n, 16);
    let (want, iters) = cpu_ref(&state, &join_rules, n, 16);
    assert_eq!(
        got, want,
        "3x3 state diverged (iters={iters}): IR={got:?} oracle={want:?}"
    );

    // Already-at-fixpoint input: all-zero join derives nothing, state is returned unchanged in ONE
    // convergence check (exercises the immediate-termination path of the loop).
    let stable = vec![0b111u32; 9];
    let no_rules = vec![0u32; 9];
    let got_stable = run_ir(&stable, &no_rules, n, 16);
    let (want_stable, _) = cpu_ref(&stable, &no_rules, n, 16);
    assert_eq!(got_stable, want_stable, "already-converged path diverged");
}
