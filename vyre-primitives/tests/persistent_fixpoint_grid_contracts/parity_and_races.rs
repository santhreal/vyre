use super::*;

/// One differential case: the two builders must agree on the whole state
/// trajectory, and the grid form's decoded pass count must match the CPU
/// oracle exactly.
fn assert_drop_in_parity(
    label: &str,
    ir_body: &dyn Fn() -> Vec<Node>,
    cpu_step: &dyn Fn(&[u32], &mut [u32]),
    seed: &[u32],
    max_iterations: u32,
    expected_state: &[u32],
    expected_passes: u32,
) {
    let (oracle_state, oracle_passes) = cpu_ref(seed, max_iterations, cpu_step);
    assert_eq!(
        oracle_state, expected_state,
        "{label}: oracle final state must be the expected fixpoint"
    );
    assert_eq!(
        oracle_passes, expected_passes,
        "{label}: oracle must enter exactly the expected number of iterations"
    );

    // Every budget, not just the full one: a builder that drops, doubles,
    // or reorders a wave diverges from the other at some prefix even when
    // both agree at the cap.
    for budget in 1..=max_iterations {
        let workgroup = run_workgroup(ir_body(), seed, budget);
        let grid = run_grid(ir_body(), seed, budget);
        assert_eq!(
            grid.state, workgroup.state,
            "{label}: budget {budget} state diverged between the two builders"
        );
    }

    let grid = run_grid(ir_body(), seed, max_iterations);
    let workgroup = run_workgroup(ir_body(), seed, max_iterations);
    assert_eq!(
        grid.state, expected_state,
        "{label}: grid builder must reach the expected fixpoint"
    );
    assert_eq!(
        workgroup.state, expected_state,
        "{label}: single-workgroup builder must reach the same fixpoint"
    );
    assert_eq!(
        passes_from_flags(&grid.changed, max_iterations),
        expected_passes,
        "{label}: grid flag array must decode to the oracle's iteration count"
    );

    // Convergence depth, derived identically for both: the smallest
    // budget whose state already equals the fully converged state. This
    // is the pass count both builders expose observably, and it must
    // agree.
    let depth = |run: &dyn Fn(u32) -> Vec<u32>| -> u32 {
        (1..=max_iterations)
            .find(|budget| run(*budget) == expected_state)
            .expect("state must converge within the budget")
    };
    let grid_depth = depth(&|budget| run_grid(ir_body(), seed, budget).state);
    let workgroup_depth = depth(&|budget| run_workgroup(ir_body(), seed, budget).state);
    assert_eq!(
        grid_depth, workgroup_depth,
        "{label}: the two builders must converge at the same depth"
    );
    assert!(
        grid_depth == expected_passes || grid_depth + 1 == expected_passes,
        "{label}: convergence depth {grid_depth} must be the last changing wave of \
         {expected_passes} entered iterations"
    );
}

/// The two builders are drop-in equivalent on a shape that fits one
/// workgroup.
///
/// This is the test that proves `persistent_fixpoint_grid` is the same
/// algorithm and not merely a program that also terminates: same final
/// state element by element at every budget, same convergence depth, and
/// a decoded pass count equal to the CPU oracle's. Four transfer bodies:
/// one that is already at its fixpoint, one that converges after a single
/// change, one that needs several waves within a word, and one whose
/// carry walks across words so the result depends on the inter-wave
/// fence.
#[test]
fn grid_builder_is_a_drop_in_for_the_single_workgroup_builder() {
    let words = 4u32;
    assert!(
        words <= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0],
        "this case must fit one workgroup, where persistent_fixpoint is sound"
    );

    // Already a fixpoint: one iteration entered, nothing changes.
    assert_drop_in_parity(
        "identity",
        &|| identity_body(words),
        &|current, next| next.copy_from_slice(current),
        &[0b1010, 0b0101, 0, u32::MAX],
        8,
        &[0b1010, 0b0101, 0, u32::MAX],
        1,
    );

    // One changing wave, then a confirming wave that reads zero.
    assert_drop_in_parity(
        "or_const",
        &|| or_const_body(words, 0b1010),
        &|current, next| {
            for (out, value) in next.iter_mut().zip(current) {
                *out = *value | 0b1010;
            }
        },
        &[0, 0, 0b0100, 0b1010],
        8,
        &[0b1010, 0b1010, 0b1110, 0b1010],
        2,
    );

    // Several changing waves inside one word: 1 -> 0x101 -> 0x10101 ->
    // 0x1010101, then flat, so four iterations are entered.
    assert_drop_in_parity(
        "shift_or",
        &|| shift_or_body(words),
        &|current, next| {
            for (out, value) in next.iter_mut().zip(current) {
                *out = *value | (*value << 8);
            }
        },
        &[1, 1, 1, 1],
        8,
        &[0x0101_0101; 4],
        4,
    );

    // A carry across words: lane t reads the word lane t - 1 wrote in the
    // previous wave, so a missing or misplaced fence changes the answer.
    assert_drop_in_parity(
        "carry",
        &|| carry_body(words),
        &|current, next| {
            next[0] = current[0];
            for index in 1..current.len() {
                next[index] = current[index] | current[index - 1];
            }
        },
        &[0b1, 0, 0, 0],
        8,
        &[0b1; 4],
        4,
    );
}

/// Above one workgroup the grid builder reports a TRUSTWORTHY verdict
/// where `persistent_fixpoint` reports a false one.
///
/// This is the measured defect, reproduced deterministically. With 257
/// words the dispatch is two workgroups. `persistent_fixpoint`'s clear is
/// guarded on global lane 0, so the second group never clears the shared
/// flag it keeps setting; it therefore never reads zero, burns the whole
/// budget, and leaves the flag set. The state still happens to be right
/// here, so the only symptom is a flag that says "not converged" after a
/// run that converged in two passes of a fifteen-pass budget: a silent
/// false verdict a caller cannot distinguish from a real cap-out.
///
/// The grid builder's flag array is read after a grid-wide fence and never
/// cleared, so it decodes to the true pass count and the zero word marks
/// real convergence.
///
/// The `persistent_fixpoint` assertion below pins CURRENT behavior of a
/// builder that is documented as unsound above one workgroup and is kept
/// as-is because its callers' pass counts are denominated in it. If that
/// builder is ever made grid-correct, this is the assertion to update.
#[test]
fn two_workgroup_shape_exposes_the_false_verdict_the_grid_builder_fixes() {
    let words = 257u32;
    assert!(
        words > PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0],
        "this case must exceed one workgroup or it proves nothing"
    );
    let seed = vec![0u32; words as usize];
    let expected_state = vec![0b1010u32; words as usize];
    let budget = 15u32;

    let grid = run_grid(or_const_body(words, 0b1010), &seed, budget);
    assert_eq!(
        grid.state, expected_state,
        "grid builder must converge every word across both workgroups"
    );
    assert_eq!(
        passes_from_flags(&grid.changed, budget),
        2,
        "grid builder must report the two passes it actually needed"
    );
    assert_eq!(
        grid.changed[1], 0,
        "the wave that read zero is the convergence proof, and it must be genuinely zero"
    );
    assert_eq!(
        grid.changed.iter().filter(|word| **word == 1).count(),
        1,
        "exactly one wave changed state, so exactly one flag word is set"
    );

    let workgroup = run_workgroup(or_const_body(words, 0b1010), &seed, budget);
    assert_eq!(
        workgroup.state, expected_state,
        "the benign face of the race leaves the state correct, which is what hides it"
    );
    assert_eq!(
        workgroup.changed,
        vec![1u32],
        "single-workgroup builder reports not-converged after converging in two passes: the \
         false verdict persistent_fixpoint_grid exists to fix"
    );
}

/// The grid builder's verdict is invariant under adversarial group
/// ordering; `persistent_fixpoint`'s FLIPS.
///
/// This is the race made deterministic. `reference_eval_lane_reversed`
/// steps the workgroups and the invocations inside them back to front. For
/// a race-free program that is a no-op. For `persistent_fixpoint` at two
/// workgroups it changes the answer the caller reads back: in forward
/// order the second group runs last, never clears the shared word it keeps
/// setting, and leaves the flag at 1; in reversed order the group holding
/// lane 0 runs last, its clear ERASES the other group's set (the lost set),
/// and the flag ends at 0. Same program, same input, same converged state,
/// two opposite verdicts decided purely by which group the scheduler
/// stepped last. That is the definition of a race, and it is why the flag
/// cannot be trusted above one workgroup.
///
/// The grid builder's flag words are never cleared and are read only after
/// a grid-wide fence, so both orders agree on every word of both buffers.
#[test]
fn grid_verdict_is_order_invariant_where_the_workgroup_verdict_flips() {
    let words = 257u32;
    assert!(
        words > PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0],
        "the race needs more than one workgroup"
    );
    let seed = vec![0u32; words as usize];
    let expected_state = vec![0b1010u32; words as usize];
    let budget = 15u32;

    let grid_forward =
        run_grid_ordered(or_const_body(words, 0b1010), &seed, budget, Order::Forward);
    let grid_reversed =
        run_grid_ordered(or_const_body(words, 0b1010), &seed, budget, Order::Reversed);
    assert_eq!(
        grid_forward.state, expected_state,
        "grid builder must converge in forward order"
    );
    assert_eq!(
        grid_reversed.state, grid_forward.state,
        "grid state must not depend on group step order"
    );
    assert_eq!(
        grid_reversed.changed, grid_forward.changed,
        "grid verdict must not depend on group step order"
    );
    assert_eq!(
        passes_from_flags(&grid_forward.changed, budget),
        2,
        "grid builder must report two passes in either order"
    );

    let workgroup_forward =
        run_workgroup_ordered(or_const_body(words, 0b1010), &seed, budget, Order::Forward);
    let workgroup_reversed =
        run_workgroup_ordered(or_const_body(words, 0b1010), &seed, budget, Order::Reversed);
    assert_eq!(
        workgroup_forward.state, expected_state,
        "the state stays right in forward order, which is what hides the race"
    );
    assert_eq!(
        workgroup_reversed.state, expected_state,
        "the state stays right in reversed order too"
    );
    assert_eq!(
        workgroup_forward.changed,
        vec![1u32],
        "forward order: the group without lane 0 never clears, so the flag stays set"
    );
    assert_eq!(
        workgroup_reversed.changed,
        vec![0u32],
        "reversed order: lane 0's clear erases the other group's set, so the flag reads clear"
    );
    assert_ne!(
        workgroup_forward.changed, workgroup_reversed.changed,
        "an order-dependent verdict from one shared cleared word IS the lost-set race"
    );
}

/// The grid builder is what flips the driver's cooperative-launch gate
/// from closed to open.
///
/// `vyre_driver::grid_sync::contains_grid_sync` is the FIRST condition in
/// the backend's cooperative-launch fit check, and it short-circuits to
/// `false` before any device property is consulted. A program built from
/// `persistent_fixpoint` contains only workgroup-scope barriers, so the
/// backend is asked whether it can grid-synchronize a program with no grid
/// synchronization in it and correctly answers no, at every candidate
/// block count down to one. That refusal is load-bearing: it withholds
/// multi-group dispatch from a program whose termination protocol is the
/// racing shared flag instead of silently computing a wrong answer.
///
/// This test pins the flip. The grid builder's barriers make the predicate
/// true, which is the precondition for the residency arithmetic to run at
/// all. It also pins the kernel-split lowering shape for a backend without
/// a native grid barrier: `2 * max_iterations` fences split the entry into
/// `2 * max_iterations + 1` ordered dispatch segments.
#[test]
fn grid_builder_opens_the_driver_cooperative_launch_gate_the_workgroup_builder_closes() {
    let max_iterations = 6u32;
    let grid = persistent_fixpoint_grid(
        or_const_body(4, 0b1010),
        "current",
        "next",
        "changed",
        4,
        max_iterations,
    );
    let workgroup = persistent_fixpoint(
        or_const_body(4, 0b1010),
        "current",
        "next",
        "changed",
        4,
        max_iterations,
    );

    assert!(
        vyre_driver::grid_sync::contains_grid_sync(&grid),
        "the grid builder must satisfy the driver's grid-sync predicate or the cooperative \
         launch path is never even asked about the device"
    );
    assert!(
        !vyre_driver::grid_sync::contains_grid_sync(&workgroup),
        "the single-workgroup builder must NOT satisfy it: that refusal is what keeps its \
         racing termination protocol off a multi-group dispatch"
    );
    assert_eq!(
        vyre_driver::grid_sync::split_on_grid_sync(&grid).len(),
        2 * max_iterations as usize + 1,
        "a backend without a native grid barrier must see one ordered dispatch segment per \
         inter-fence span"
    );
}
