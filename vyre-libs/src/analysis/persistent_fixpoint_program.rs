//! Persistent-fixpoint Program builder for runtime and driver scheduling loops.

use crate::fixpoint::persistent_fixpoint::{
    fixpoint_route, routed_persistent_fixpoint, FixpointRoute, FixpointState,
};
use vyre_foundation::ir::{Node, Program};

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
/// [`crate::fixpoint::persistent_fixpoint::routed_persistent_fixpoint`],
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
    use crate::fixpoint::persistent_fixpoint::{
        declared_words, persistent_fixpoint, PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
    };
    use vyre_foundation::ir::{Expr, Node, Program};

    /// The wrapper declares the three buffers the caller named and no others.
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

    /// This wrapper's routing obligations, asserted once by the owner.
    ///
    /// They are obligations of the ROUTING, not of this wrapper: the threshold is
    /// the dispatch span, the flag is one word at one workgroup and one word per
    /// iteration past it, and the grid form fences twice per wave. Asserting them
    /// here was a second copy that could be weakened for this op alone, which is
    /// how the copies that motivated the contract drifted.
    #[test]
    fn the_wrapper_obeys_the_persistent_fixpoint_routing_contract() {
        let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];
        crate::fixpoint::routing_contract::assert_routes_on_dispatch_span(
            &crate::fixpoint::routing_contract::RoutedFixpointOp {
                name: "persistent_fixpoint_program",
                changed: "changed",
                at_one_workgroup: &|max_iterations| {
                    persistent_fixpoint_program(
                        Vec::new(),
                        "current",
                        "next",
                        "changed",
                        width,
                        max_iterations,
                    )
                },
                past_one_workgroup: &|max_iterations| {
                    persistent_fixpoint_program(
                        Vec::new(),
                        "current",
                        "next",
                        "changed",
                        width + 1,
                        max_iterations,
                    )
                },
                grid_harness: &|max_iterations| {
                    crate::fixpoint::routing_contract::bare_grid_harness(
                        "current",
                        "next",
                        "changed",
                        width + 1,
                        max_iterations,
                    )
                },
            },
        );
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
