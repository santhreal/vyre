//! Persistent-fixpoint Program builder for runtime and driver scheduling loops.

use vyre_foundation::ir::{Node, Program};
use vyre_primitives::fixpoint::persistent_fixpoint::{
    fixpoint_route, routed_persistent_fixpoint, FixpointRoute, FixpointState,
};

/// Build a persistent-fixpoint Program around a caller-supplied transfer body.
///
/// The generated program runs `transfer_body`, ping-pongs `current` and `next`,
/// and stops when the convergence flag reads zero or `max_iterations` is
/// reached. Runtime and driver crates call this self-substrate wrapper instead
/// of depending on the primitive catalog directly.
///
/// # Convergence-flag form
///
/// `words` sizes the widest buffer this wrapper declares, so `words` is also the
/// launch span here and it selects the harness. The selection and the `changed`
/// width that goes with it belong to
/// [`vyre_primitives::fixpoint::persistent_fixpoint::routed_persistent_fixpoint`],
/// which owns both halves; see [`FixpointRoute`] for why they cannot be chosen
/// separately.
///
/// The caller supplies `changed`, so it must be
/// [`FixpointRoute::changed_words`] zero-filled words for its `words` and
/// `max_iterations`. Use [`persistent_fixpoint_program_route`] to read that
/// width rather than re-deriving the threshold.
///
/// This wrapper can only see the buffers the harness declares. A caller whose
/// `transfer_body` reads buffers wider than `words`, or that widens the launch
/// through `DispatchConfig`, owns that span itself.
#[must_use]
pub fn persistent_fixpoint_program(
    transfer_body: Vec<Node>,
    current: &str,
    next: &str,
    changed: &str,
    words: u32,
    max_iterations: u32,
) -> Program {
    routed_persistent_fixpoint(
        transfer_body,
        FixpointState {
            current,
            next,
            changed,
            words,
            max_iterations,
        },
        words,
    )
    .0
}

/// The harness and `changed` width [`persistent_fixpoint_program`] selects for
/// `words` and `max_iterations`.
///
/// A caller allocates `changed` before it has a program, so it needs the width
/// without building one. Reading it here rather than comparing `words` against
/// the workgroup constant is what keeps the flag width and the harness in
/// agreement.
#[must_use]
pub fn persistent_fixpoint_program_route(words: u32, max_iterations: u32) -> FixpointRoute {
    fixpoint_route(words, max_iterations)
}

#[cfg(test)]
mod tests {
    use super::persistent_fixpoint_program;
    use vyre_foundation::ir::{Expr, Node, Program};
    use vyre_primitives::fixpoint::persistent_fixpoint::{
        count_grid_sync, declared_words, required_workgroups,
    };
    use vyre_primitives::fixpoint::persistent_fixpoint::{
        persistent_fixpoint, PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
    };

    /// Workgroups a host must launch to cover `program`.
    ///
    /// The wrapped primitive emits the convergence flag's `atomic_or`, and for an
    /// atomic-carrying program `vyre-driver`'s `dispatch_element_count_for_program`
    /// spans the LARGEST declared buffer, so the launch width is `words` rounded up
    /// to whole workgroups.
    /// Declared word count of the convergence-flag buffer.
    #[test]
    fn builds_program_with_caller_buffers() {
        let program = persistent_fixpoint_program(Vec::new(), "current", "next", "changed", 4, 8);
        let names = program
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect::<Vec<_>>();

        assert!(names.contains(&"current"));
        assert!(names.contains(&"next"));
        assert!(names.contains(&"changed"));
    }

    /// Locks out the multi-workgroup convergence-flag race.
    ///
    /// The single-word primitive keeps ONE `changed[0]` word, clears it from global
    /// lane 0 with a plain store, and orders that clear against every other lane's
    /// `atomic_or` with a workgroup-scoped `SeqCst` barrier only. Once the launch
    /// spans more than one workgroup nothing orders the clear against the sets:
    /// workgroup 0's next clear can erase workgroup 1's set, so workgroup 1 reads 0
    /// and `Return`s with unconverged state, and the post-dispatch flag read reports
    /// a convergence verdict no group agreed to. A multi-workgroup build must
    /// therefore never be handed one shared cleared word.
    #[test]
    fn multi_workgroup_wrapper_never_shares_one_cleared_convergence_word() {
        let program = persistent_fixpoint_program(Vec::new(), "current", "next", "changed", 257, 8);

        assert_eq!(
            required_workgroups(&program),
            2,
            "Fix: 257 words over a 256-wide workgroup must need two workgroups."
        );
        assert_eq!(
            declared_words(&program, "changed"),
            8,
            "Fix: a multi-workgroup fixpoint dispatch must use the per-iteration convergence-word protocol, not one shared cleared word."
        );
    }

    /// Grid-wide fences in `nodes`, counted through every nesting construct.

    /// A transfer body in which lane 0 publishes the value of the LAST element and
    /// nothing else writes: `if t == 0 { next[last] = 9 }`.
    ///
    /// Partitioned by global invocation id in the strictest sense: exactly one lane
    /// produces `next[last]`, and under the harness's ping-pong exactly one lane
    /// (the one whose compare covers `last`) copies it into `current[last]`. Above
    /// one workgroup those are lanes in DIFFERENT groups, which is what makes the
    /// shared convergence flag observable rather than masked. The fixpoint is
    /// unambiguous: the store is idempotent, so `current[last] == 9`.
    fn publish_last_element_body(next: &str, last: u32) -> Vec<Node> {
        vec![Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            vec![Node::store(next, Expr::u32(last), Expr::u32(9))],
        )]
    }

    /// Run `program` on the reference interpreter and return the final `current`
    /// vector paired with the final `changed` words. `reversed` steps the workgroups
    /// back to front; both orders are schedules real hardware is free to pick,
    /// because nothing in the IR orders one workgroup against another.
    fn run_fixpoint(
        program: &Program,
        reversed: bool,
        words: u32,
        changed_word_count: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        use vyre_reference::value::Value;

        let to_value = |data: &[u32]| {
            Value::Bytes(std::sync::Arc::from(vyre_primitives::wire::pack_u32_slice(
                data,
            )))
        };
        let zeros = vec![0_u32; words as usize];
        let inputs = vec![
            to_value(&zeros),
            to_value(&zeros),
            to_value(&vec![0_u32; changed_word_count as usize]),
        ];
        let results = if reversed {
            vyre_reference::reference_eval_lane_reversed(program, &inputs)
        } else {
            vyre_reference::reference_eval(program, &inputs)
        }
        .expect("Fix: the reference interpreter must execute the fixpoint program.");
        let decode = |value: &vyre_reference::value::Value| -> Vec<u32> {
            value
                .to_bytes()
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };
        (decode(&results[0]), decode(&results[2]))
    }

    /// Pins the routing threshold to the declared workgroup width.
    ///
    /// The threshold is `> PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`, read from the
    /// same constant the emitted program declares as its workgroup size, so the two
    /// can never drift apart. At exactly that width the launch is one workgroup and
    /// the compact single-word protocol is sound, so it stays in use; one word past
    /// it the launch is two workgroups and must switch. An off-by-one here puts a
    /// multi-workgroup dispatch back on the racing flag.
    #[test]
    fn routing_threshold_is_the_declared_workgroup_width() {
        let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

        let at_width =
            persistent_fixpoint_program(Vec::new(), "current", "next", "changed", width, 8);
        assert_eq!(
            at_width.workgroup_size(),
            PERSISTENT_FIXPOINT_WORKGROUP_SIZE
        );
        assert_eq!(required_workgroups(&at_width), 1);
        assert_eq!(
            declared_words(&at_width, "changed"),
            1,
            "Fix: a single-workgroup launch must keep the compact one-word convergence flag."
        );

        let past_width =
            persistent_fixpoint_program(Vec::new(), "current", "next", "changed", width + 1, 8);
        assert_eq!(required_workgroups(&past_width), 2);
        assert_eq!(
            declared_words(&past_width, "changed"),
            8,
            "Fix: one word past the workgroup width already needs the per-iteration convergence words."
        );
    }

    /// The two routes must not silently converge to the same emission.
    ///
    /// The grid form's soundness IS its `MemoryOrdering::GridSync` fences: they
    /// order the per-iteration flag write against every group's read. The
    /// single-workgroup form must carry none of them, because emitting one there
    /// would impose a cooperative launch on a dispatch that does not need it.
    #[test]
    fn grid_route_fences_the_grid_and_single_workgroup_route_does_not() {
        let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

        let single =
            persistent_fixpoint_program(Vec::new(), "current", "next", "changed", width, 4);
        assert_eq!(
            count_grid_sync(single.entry()),
            0,
            "Fix: a single-workgroup fixpoint program must not force a cooperative grid launch."
        );

        let grid =
            persistent_fixpoint_program(Vec::new(), "current", "next", "changed", width + 1, 4);
        assert_eq!(
            count_grid_sync(grid.entry()),
            8,
            "Fix: the grid form must fence each of its 4 waves twice, once after the transfer step and once after the compare."
        );
    }

    /// The grid form indexes `changed[iteration]`, so a one-word buffer there would
    /// be an out-of-bounds atomic write on iteration 1. The caller supplies that
    /// buffer, so the declared count is the contract it has to satisfy.
    #[test]
    fn grid_route_sizes_changed_to_one_word_per_iteration() {
        let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];
        for max_iterations in [1_u32, 2, 8, 64] {
            let program = persistent_fixpoint_program(
                Vec::new(),
                "current",
                "next",
                "changed",
                width + 1,
                max_iterations,
            );
            assert_eq!(
                declared_words(&program, "changed"),
                max_iterations,
                "Fix: the grid route needs one convergence word per iteration; {max_iterations} iterations need {max_iterations} words."
            );
        }
    }

    /// OBSERVED divergence: the pre-routing single-word harness returns WRONG state
    /// above one workgroup, it does not merely look unsound.
    ///
    /// 257 words, so the launch is two workgroups. Lane 0, in group 0, is the only
    /// producer of `next[256]`; lane 256, in group 1, is the only lane whose compare
    /// covers element 256, so it is the only writer of `current[256]` and the only
    /// lane that can set the convergence flag for it. Nothing orders the groups.
    ///
    /// Step group 1 first and it compares `current[256]` against a `next[256]` group
    /// 0 has not yet written, sees no change, reads the still-zero shared flag and
    /// retires for good. Group 0 then writes `next[256] = 9`, finds no change among
    /// the elements IT covers, and also retires. `current[256]` is never published:
    /// the dispatch reports convergence and yields 0 where the fixpoint is 9.
    #[test]
    fn single_word_harness_returns_wrong_state_above_one_workgroup() {
        let words = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0] + 1;
        let last = words - 1;
        let max_iterations = 4_u32;

        let unsound = persistent_fixpoint(
            publish_last_element_body("next", last),
            "current",
            "next",
            "changed",
            words,
            max_iterations,
        );
        assert_eq!(
            declared_words(&unsound, "changed"),
            1,
            "Fix: this fixture must exercise the single shared convergence word."
        );

        let (forward, forward_flag) = run_fixpoint(&unsound, false, words, 1);
        let (reversed, reversed_flag) = run_fixpoint(&unsound, true, words, 1);

        assert_eq!(
            forward[last as usize], 9,
            "Fix: stepping group 0 first must reach the fixpoint, proving the divergence is cross-workgroup ordering."
        );
        assert_eq!(
            forward_flag[0], 1,
            "Fix: the correct schedule must leave the flag set, since group 1 sets it and nobody clears it afterwards."
        );
        assert_eq!(
            reversed[last as usize],
            0,
            "Fix: this test records the OBSERVED wrong value the racing shared flag produces; if the single-word harness stops diverging here, re-derive the defect before deleting this test."
        );
        assert_eq!(
            reversed_flag[0], 0,
            "Fix: the shared flag must be observed claiming convergence while the last element is unpublished, which is what makes the wrong answer silent."
        );
    }

    /// The routed program is correct under BOTH workgroup orders at the size where
    /// the single-word harness diverges, which is the fix working end to end.
    #[test]
    fn grid_routed_wrapper_is_order_independent_where_single_word_diverges() {
        let words = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0] + 1;
        let last = words - 1;
        let max_iterations = 4_u32;

        let routed = persistent_fixpoint_program(
            publish_last_element_body("next", last),
            "current",
            "next",
            "changed",
            words,
            max_iterations,
        );
        assert_eq!(
            declared_words(&routed, "changed"),
            max_iterations,
            "Fix: this size must route to the grid harness."
        );

        for reversed in [false, true] {
            let (current, _) = run_fixpoint(&routed, reversed, words, max_iterations);
            assert_eq!(
                current[last as usize], 9,
                "Fix: the grid-routed program must reach the fixpoint in both workgroup orders (reversed={reversed})."
            );
        }
    }
}
