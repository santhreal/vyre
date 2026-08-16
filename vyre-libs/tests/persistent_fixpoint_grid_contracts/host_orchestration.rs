use super::*;

/// One dispatch of a program built for `waves_per_dispatch` waves, seeded
/// from `state`, returning the new state and the flag array.
fn dispatch_once(program: &Program, state: &[u32], waves_per_dispatch: u32) -> Run {
    let outputs = eval(
        program,
        &[
            pack(state),
            pack(&vec![0u32; state.len()]),
            // Re-zeroed on EVERY dispatch. The primitive never clears the
            // flag words, so a reused dirty buffer would make wave 0 read a
            // stale set and decode the wrong pass count.
            pack(&vec![0u32; waves_per_dispatch.max(1) as usize]),
        ],
        Order::Forward,
    );
    Run {
        state: unpack(&outputs[0]),
        changed: unpack(&outputs[2]),
    }
}

/// Drive convergence from the HOST: build the program once for a fixed
/// wave count, then re-dispatch it, feeding `current` back in, until a
/// batch reports a zero flag word.
///
/// Returns the final state, the number of dispatches, and the total
/// iterations entered across all of them.
fn run_host_orchestrated(
    body: &dyn Fn() -> Vec<Node>,
    seed: &[u32],
    waves_per_dispatch: u32,
    max_dispatches: u32,
) -> (Vec<u32>, u32, u32) {
    let words = u32::try_from(seed.len()).expect("seed length must fit u32");
    // Built ONCE. This is the whole point of host orchestration: the
    // transfer body is emitted `waves_per_dispatch` times, not
    // `total_budget` times.
    let program = persistent_fixpoint_grid(
        body(),
        "current",
        "next",
        "changed",
        words,
        waves_per_dispatch,
    );
    let mut state = seed.to_vec();
    let mut dispatches = 0u32;
    let mut total_passes = 0u32;
    for _ in 0..max_dispatches {
        let run = dispatch_once(&program, &state, waves_per_dispatch);
        state = run.state;
        dispatches += 1;
        total_passes += passes_from_flags(&run.changed, waves_per_dispatch);
        if run.changed.iter().any(|word| *word == 0) {
            break;
        }
    }
    (state, dispatches, total_passes)
}

/// The wave count is a caller-chosen parameter, so host-orchestrated
/// repetition needs no new entry point and reaches the same answer as one
/// fully unrolled dispatch.
///
/// `max_iterations` IS the emitted wave count: the builder emits exactly
/// `max_iterations` waves and sizes `changed` to exactly that many words.
/// So a host loop passes its per-dispatch batch size and re-dispatches the
/// SAME program, which emits the caller's transfer body once per wave in
/// the batch instead of once per total budget.
///
/// This pins that the batched protocol is not a different algorithm: batch
/// sizes 1, 2, and 5 all reach the fully unrolled 15-wave answer, agree on
/// total iterations entered, and cost the dispatch count the batch size
/// implies. It also pins that the flag buffer must be re-zeroed per
/// dispatch, which the helper above does, because the primitive never
/// clears it.
#[test]
fn host_orchestrated_batches_reach_the_same_answer_as_one_unrolled_dispatch() {
    let words = 257u32;
    assert!(
        words > PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0],
        "the multi-group shape is the one host orchestration exists for"
    );
    let seed = vec![0u32; words as usize];
    let expected_state = vec![0b1010u32; words as usize];

    // Reference: one dispatch with the whole budget unrolled.
    let unrolled = run_grid(or_const_body(words, 0b1010), &seed, 15);
    assert_eq!(unrolled.state, expected_state);
    assert_eq!(passes_from_flags(&unrolled.changed, 15), 2);

    // A single-wave program is the classic host-driven step: five nodes,
    // one flag word, and the same ABI width persistent_fixpoint had.
    let one_wave = persistent_fixpoint_grid(
        or_const_body(words, 0b1010),
        "current",
        "next",
        "changed",
        words,
        1,
    );
    assert_eq!(
        wave_nodes(&one_wave).len(),
        5,
        "a one-wave build emits exactly one wave, not the caller's total budget"
    );
    assert_eq!(
        one_wave
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("flag buffer must be declared")
            .count(),
        1,
        "the flag buffer is sized per EMITTED wave, so a one-wave build needs one word"
    );

    // Every batch size reaches the same fixpoint and the same total passes.
    for (waves_per_dispatch, expected_dispatches) in [(1u32, 2u32), (2, 1), (5, 1)] {
        let (state, dispatches, total_passes) = run_host_orchestrated(
            &|| or_const_body(words, 0b1010),
            &seed,
            waves_per_dispatch,
            15,
        );
        assert_eq!(
            state, expected_state,
            "batch {waves_per_dispatch}: host orchestration must reach the same fixpoint"
        );
        assert_eq!(
            total_passes, 2,
            "batch {waves_per_dispatch}: total iterations entered must match the unrolled run"
        );
        assert_eq!(
            dispatches, expected_dispatches,
            "batch {waves_per_dispatch}: a batch that spans the convergence point must finish in \
             one dispatch, and a single-wave batch needs the confirming second"
        );
    }

    // A batch that spans convergence still carries the grid fences, so the
    // cooperative-launch requirement does NOT go away with a smaller batch.
    assert!(
        vyre_driver::grid_sync::contains_grid_sync(&one_wave),
        "even a one-wave build contains GridSync, so host batching does not escape the \
         cooperative-residency ceiling"
    );
}

/// A transfer body that writes only SOME words of `next` silently CORRUPTS
/// the words it skips, in both builders, and still reports convergence.
///
/// I first predicted this shape would never converge, and this test
/// falsified that. The reason is the ping-pong: the copy writes
/// `current[w] = next[w]` for EVERY `w < words`, not only for the words the
/// transfer touched. So iteration 1 overwrites the skipped words of
/// `current` with whatever `next` held, and from iteration 2 on the two
/// buffers agree everywhere and the compare reports no change. Convergence
/// is reached promptly and the state is WRONG.
///
/// That makes this a wrong-answer defect wearing a converged verdict, which
/// is worse than a stuck loop and is why it is pinned rather than described.
/// Both builders are asserted because both share the compare-and-copy step.
#[test]
fn a_transfer_body_that_skips_words_silently_corrupts_them() {
    // Writes only word 0. Words 1..4 of `next` are never stored, so they
    // hold their zero seed and the copy propagates that zero into `current`.
    let partial_body = vec![Node::if_then(
        Expr::eq(lane(), Expr::u32(0)),
        vec![Node::store("next", lane(), Expr::load("current", lane()))],
    )];
    let seed = [9u32, 9, 9, 9];
    let budget = 4u32;

    let grid = run_grid_ordered(partial_body.clone(), &seed, budget, Order::Forward);
    assert_eq!(
        grid.state,
        vec![9u32, 0, 0, 0],
        "the skipped words are overwritten with the untouched `next`, so three \
         of four words are silently lost while word 0 survives"
    );
    assert_eq!(
        grid.changed,
        vec![1u32, 0, 0, 0],
        "and it REPORTS CONVERGENCE at wave 2: one changing wave, then zeros. \
         A caller reading this verdict is told the fixpoint was reached"
    );

    let workgroup = run_workgroup_ordered(partial_body, &seed, budget, Order::Forward);
    assert_eq!(
        workgroup.state,
        vec![9u32, 0, 0, 0],
        "the single-word builder loses the same three words"
    );
    assert_eq!(
        workgroup.changed,
        vec![0u32],
        "and also finishes reporting converged, because its flag is cleared \
         each iteration and the final iteration set nothing"
    );

    // The control: the SAME shape with a total write converges immediately,
    // so the assertions above are attributable to the partial write and not
    // to the seed, the budget, or the harness.
    let total = run_grid_ordered(identity_body(4), &seed, budget, Order::Forward);
    assert_eq!(
        total.changed,
        vec![0u32; budget as usize],
        "a body that writes every word converges on the first wave"
    );
    assert_eq!(total.state, seed, "and leaves the state untouched");
}

/// NEITHER builder may write `changed` with a plain `Node::Store`. Every write
/// to that word must be an atomic.
///
/// For the grid builder this is what makes the multi-group lost-set race
/// structurally impossible rather than merely unlikely: it has one word per
/// iteration and never clears, and a word that is never cleared cannot lose a
/// set.
///
/// For `persistent_fixpoint` it removes a latent hazard that existed at ANY
/// group count. It clears `changed[0]` once per iteration, and that clear used
/// to be a plain non-atomic store to a location every other write reaches
/// through `atomic_or`. At one workgroup the mixing was ordered by the
/// barriers around it (clear, barrier, sets, barrier, barrier, read) and so
/// was not a live race, which an earlier version of this comment got wrong.
/// It was still a hazard, because its correctness rested on an ordering
/// assumption invisible at the call site: weaken or move that barrier and the
/// program breaks without anything correctness-shaped being edited. The clear
/// is now an atomic exchange, which costs one lane one operation per iteration
/// and removes the assumption. The multi-group race is NOT fixed by that and
/// is not meant to be; above one workgroup use the grid builder.
///
/// Reintroducing a plain write to `changed` in either builder would restore
/// the hazard while every value-level test kept passing, because the reference
/// interpreter does not model L1 versus L2 and cannot reproduce a hardware
/// race. So the property has to be asserted structurally, on the emitted IR,
/// or it is not covered at all. The final assertion points the same predicate
/// at `next`, which IS written by plain stores, so a matcher that silently
/// stopped matching could not make this test pass.
#[test]
fn neither_builder_writes_changed_with_a_plain_store() {
    for budget in [1u32, 2, 3, 8] {
        let program =
            persistent_fixpoint_grid(identity_body(4), "current", "next", "changed", 4, budget);
        let plain_stores = count_nodes(
            &program,
            |node| matches!(node, Node::Store { buffer, .. } if buffer.as_str() == "changed"),
        );
        assert_eq!(
            plain_stores, 0,
            "budget {budget}: `changed` must be written only by atomic_or, but \
             {plain_stores} plain Node::Store(s) target it; that is the \
             same-location plain-versus-atomic conflict this builder exists to \
             remove"
        );

        // Every write to `changed` is accounted for: one atomic_or per wave,
        // so the zero above means "all atomic", never "nothing written".
        assert_eq!(
            atomic_or_indices(&program, "changed"),
            (0..budget).collect::<Vec<u32>>(),
            "budget {budget}: each wave i must atomic_or exactly word i"
        );
    }

    // `persistent_fixpoint` must ALSO be free of a plain write to `changed`.
    // Its lane-0 clear is an atomic exchange, so the mixing hazard is gone
    // there too and this assertion is what keeps it gone.
    let legacy = persistent_fixpoint(identity_body(4), "current", "next", "changed", 4, 3);
    let legacy_plain = count_nodes(
        &legacy,
        |node| matches!(node, Node::Store { buffer, .. } if buffer.as_str() == "changed"),
    );
    assert_eq!(
        legacy_plain, 0,
        "persistent_fixpoint must clear `changed` atomically, never with a plain \
         store; a plain write to a location every other write reaches through \
         atomic_or is correct only while the surrounding barriers stand"
    );

    // The probe is not vacuous: the SAME predicate shape, pointed at a buffer
    // that genuinely is written by plain stores, finds them. Without this a
    // broken matcher would report zero everywhere and every assertion above
    // would pass for the wrong reason.
    let plain_next = count_nodes(
        &legacy,
        |node| matches!(node, Node::Store { buffer, .. } if buffer.as_str() == "next"),
    );
    assert!(
        plain_next > 0,
        "the plain-store probe must detect the transfer body's plain writes to \
         `next`, otherwise the zero counts above prove nothing"
    );
}

/// On the KERNEL-SPLIT lowering path the collective early exit cannot skip
/// waves, because each wave becomes its own kernel launch and the host loop
/// launches all of them unconditionally.
///
/// This bounds the early-exit guarantee, and the bound is easy to miss because
/// nothing fails: the state is correct, `changed` decodes correctly, and only
/// the saved work disappears. `MemoryOrdering::GridSync` lowers either to a
/// native cooperative grid barrier or to a kernel split. Under the split,
/// `vyre_driver::grid_sync` dispatches every segment in order, and a
/// `Node::Return` inside segment N returns from THAT launch only; it cannot
/// prevent the host from launching segments N+1 onward. So a run that
/// converges at wave 2 of a 16-wave budget still issues every segment, and a
/// device-side pass counter reads the full budget rather than 2.
///
/// A downstream caller measured exactly that, passes equal to budget with
/// byte-correct state and a correct `[1, 0, 0, ...]` flag buffer, and it is
/// the expected behavior of this path rather than a defect in the waves. The
/// early exit saves launches ONLY under a native cooperative launch. Asserted
/// on segment structure so it stays true independent of any backend.
#[test]
fn the_split_path_launches_every_wave_because_return_is_per_segment() {
    let budget = 4u32;
    let program =
        persistent_fixpoint_grid(identity_body(4), "current", "next", "changed", 4, budget);

    // Two GridSync barriers per wave, so the wave list splits into
    // 2 * budget + 1 segments. Every one of these is a launch the host issues.
    let segments = vyre_driver::grid_sync::split_on_grid_sync(&program);
    assert_eq!(
        segments.len(),
        2 * budget as usize + 1,
        "budget {budget}: two grid fences per wave must yield 2*budget+1 segments, \
         and the host issues one launch per segment regardless of convergence"
    );

    // The exits survive the split, which is what makes them per-launch returns
    // rather than a loop break: they are distributed across segments, so no
    // single segment's Return can suppress a later segment's launch.
    let total_returns: usize = segments
        .iter()
        .map(|segment| count_nodes(segment, |node| matches!(node, Node::Return)))
        .sum();
    assert_eq!(
        total_returns, budget as usize,
        "all {budget} exits must survive splitting, one per wave; losing them \
         here would break the exit on the cooperative path too"
    );

    // And no single segment carries them all, which is the precise reason the
    // exit cannot gate the host loop.
    let max_in_one_segment = segments
        .iter()
        .map(|segment| count_nodes(segment, |node| matches!(node, Node::Return)))
        .max()
        .expect("split must produce at least one segment");
    assert!(
        max_in_one_segment < budget as usize,
        "the exits must be spread across segments (max {max_in_one_segment} in \
         one of {} segments); if one segment held all of them the host loop \
         could short-circuit, and this bound would not apply",
        segments.len()
    );
}
