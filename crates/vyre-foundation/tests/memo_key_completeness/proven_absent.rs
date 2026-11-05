use super::*;

// ---------------------------------------------------------------------------
// 4. Proven-absent results
// ---------------------------------------------------------------------------

/// PROVEN ABSENT. A poisoned fact cache does NOT change what `optimize`
/// produces, for a fixture whose fact table is provably wrong.
///
/// Why this exists: it is the difference between "this cache can be poisoned"
/// and "this cache changes compiled output", and reporting the first as if it
/// were the second is how a correct probe produces a wrong severity. The
/// previous test proves `build_cached` hands out a table describing a different
/// tree. This test then asks the only question that decides severity: does that
/// wrong table survive into the optimizer's output?
///
/// Method: run `pre_lowering::optimize` on the same program twice, once on a
/// thread whose fact cache has been deliberately poisoned with a
/// fingerprint-colliding program, and once on a freshly spawned thread, which
/// necessarily has an empty thread-local cache. Compare the raw `Debug`
/// rendering of the resulting entry trees, not `==`, because `Program`'s
/// equality is itself blind to some differences.
///
/// The two arms agree. Two reasons, both worth knowing: `optimize` canonicalizes
/// as its very first step, so the tree the passes see already matches its own
/// canonical form, and every consumer reads canonicalization-invariant
/// projections as documented above.
///
/// What breaks if this regresses: a failure here means a poisoned cache now
/// reaches generated code, which promotes the finding above from LATENT to LIVE
/// and makes it a release blocker. This test failing is the alarm.
#[test]
fn optimize_output_is_stable_under_a_poisoned_fact_cache() {
    let warm = {
        let _poison = ProgramFacts::build_cached(&indexed_primer());
        vyre_foundation::optimizer::optimize(indexed_target())
            .expect("registered optimizer must converge")
    };
    let cold = std::thread::spawn(|| {
        vyre_foundation::optimizer::optimize(indexed_target())
            .expect("registered optimizer must converge")
    })
    .join()
    .expect("cold arm must not panic");

    assert_eq!(
        format!("{:?}", warm.entry()),
        format!("{:?}", cold.entry()),
        "Fix: optimize() output now depends on thread-local fact-cache state, which promotes the \
         build_cached key gap from LATENT to a LIVE wrong-output defect. Do not silence this by \
         clearing the cache in the test."
    );
    assert_eq!(
        warm.fingerprint(),
        cold.fingerprint(),
        "warm and cold arms must agree on the canonical form too"
    );
}

/// PROVEN SOUND. The per-`Program` value memos are keyed by NOTHING, so they
/// structurally cannot serve another program's artifact, and they are cleared
/// on mutation rather than going stale.
///
/// Why this exists: the enumeration that produced this file found that most
/// reuse in `Program` is memoization on the value itself (`fingerprint`,
/// `stats`, `output_buffer_indices`, `has_indirect_dispatch`) rather than
/// lookup in a shared table. That distinction is the whole reason those are
/// safe while `build_cached` is not, and it is worth an explicit test so a
/// future refactor cannot quietly convert a value memo into a keyed cache.
///
/// This also covers the mutable-state hazard: a memo over a value whose inputs
/// can change is stale by construction unless every mutator invalidates it.
/// `entry_mut` is the mutation path, so it must clear all four.
#[test]
fn program_value_memos_are_stable_and_invalidate_on_mutation() {
    let mut program = Program::wrapped(
        vec![out_buf()],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );

    // Stable across repeated reads.
    let fingerprint = program.fingerprint();
    assert_eq!(program.fingerprint(), fingerprint);
    assert_eq!(program.stats().node_count, 2);
    assert_eq!(program.stats().node_count, 2);
    assert_eq!(program.output_buffer_indices(), &[0]);
    assert_eq!(program.output_buffer_indices(), &[0]);
    assert!(!program.has_indirect_dispatch());

    // A clone carries the memos and must agree exactly.
    let cloned = program.clone();
    assert_eq!(cloned.fingerprint(), fingerprint);
    assert_eq!(cloned.stats().node_count, 2);

    // Mutating through the sanctioned path must invalidate, not go stale.
    program
        .entry_mut()
        .push(Node::store("out", Expr::u32(1), Expr::u32(2)));
    assert_ne!(
        program.fingerprint(),
        fingerprint,
        "Fix: entry_mut must invalidate the fingerprint memo. A memo over mutable state that is \
         not invalidated is stale by construction."
    );
    assert_eq!(
        program.stats().node_count,
        3,
        "Fix: entry_mut must invalidate the stats memo."
    );
    assert!(
        !program.is_structurally_validated(),
        "Fix: entry_mut must clear structural validation state."
    );
}
