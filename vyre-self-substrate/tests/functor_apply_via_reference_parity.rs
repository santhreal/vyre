//! End-to-end parity for `logic::functorial_pass_composition::apply_pass_functor_via`.
//!
//! `functor_apply` (the column-mapping functor the IR-view pass-composition layer dispatches) had
//! NO IR-execution coverage anywhere: no `vyre-primitives/tests/*` parity file runs its Program,
//! and its only self-substrate consumer test uses a `FunctorDispatcher` MOCK that ignores the
//! `_program` argument and hand-computes the scatter, so `apply_pass_functor_via` validated buffer
//! packing/grid plumbing but NEVER executed the kernel (the mock-dispatcher-coherence gap).
//!
//! This runs the real `functor_apply_sized` Program through the shared `ReferenceEvalDispatcher`
//! and asserts it reproduces the host `apply_pass_functor` contract, a TARGET-CENTRIC GATHER
//! (each target lane scans all sources, taking the LAST source that maps to it) whose "highest
//! source index wins on collision, out-of-range mappings ignored" semantics must survive the full
//! dispatch round-trip. Collisions and out-of-range targets are generated deliberately so the
//! last-wins tie-break and OOB-drop are actually exercised (not a vacuous injective identity).
#![forbid(unsafe_code)]

use vyre_self_substrate::logic::functorial_pass_composition::{
    apply_pass_functor, apply_pass_functor_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn apply_pass_functor_via_matches_host_over_generated_mappings() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0xF00D_0F5Au32;
    let mut collision_cases = 0u32;
    let mut oob_cases = 0u32;
    for case in 0..400u32 {
        let n_cols = 1 + xorshift(&mut state) % 8; // 1..=8 source columns
        let target_n_cols = 1 + xorshift(&mut state) % 8; // 1..=8 target columns
        let view_in: Vec<u32> = (0..n_cols).map(|_| xorshift(&mut state) % 1000).collect();
        // Map each source column into [0, target_n_cols + 1): the extra slot forces occasional
        // out-of-range targets that the gather must drop. Small ranges force collisions.
        let column_mapping: Vec<u32> = (0..n_cols)
            .map(|_| xorshift(&mut state) % (target_n_cols + 1))
            .collect();

        // Track that the distribution actually exercises the interesting paths.
        let mut seen = std::collections::HashMap::new();
        for &m in &column_mapping {
            *seen.entry(m).or_insert(0u32) += 1;
            if m >= target_n_cols {
                oob_cases += 1;
            }
        }
        if seen.values().any(|&c| c > 1) {
            collision_cases += 1;
        }

        let via = apply_pass_functor_via(&dispatcher, &view_in, &column_mapping, target_n_cols)
            .expect("apply_pass_functor_via must dispatch through the reference backend");
        let host = apply_pass_functor(&view_in, &column_mapping, target_n_cols);
        assert_eq!(
            via, host,
            "case {case} (n_cols={n_cols}, target_n_cols={target_n_cols}): functor _via {via:?} != \
             host contract {host:?} (view_in={view_in:?}, column_mapping={column_mapping:?})"
        );
    }
    assert!(
        collision_cases > 80,
        "only {collision_cases}/400 cases had a colliding mapping, the last-wins tie-break is under-exercised"
    );
    assert!(
        oob_cases > 40,
        "only {oob_cases} out-of-range mappings generated, the OOB-drop path is under-exercised"
    );
}

#[test]
fn apply_pass_functor_via_resolves_collision_to_highest_source_index() {
    // Three sources all map to target column 1; the host contract is "highest source index wins",
    // so target 1 must hold view_in[2] = 9. Target 0 stays 0 (no source maps to it).
    let dispatcher = ReferenceEvalDispatcher;
    let view_in = vec![7u32, 8, 9];
    let column_mapping = vec![1u32, 1, 1];
    let target_n_cols = 3;
    let via = apply_pass_functor_via(&dispatcher, &view_in, &column_mapping, target_n_cols)
        .expect("apply_pass_functor_via must dispatch");
    let host = apply_pass_functor(&view_in, &column_mapping, target_n_cols);
    assert_eq!(
        host,
        vec![0, 9, 0],
        "sanity: host resolves the collision to the highest source"
    );
    assert_eq!(
        via, host,
        "the dispatched gather must resolve the collision to the same highest-source value"
    );
}
