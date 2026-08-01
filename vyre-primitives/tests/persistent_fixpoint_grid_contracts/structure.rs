use super::*;

/// The exit is COLLECTIVE: every `Node::Return` sits after a `GridSync`
/// barrier at the same top level.
///
/// Locks out the stranded-group hazard. A `Return` reached before the
/// grid-wide fence is decided per group on a value groups can disagree
/// about, so one group leaves while the others go on to wait at a barrier
/// the departed group will never arrive at. That is a hang, not a wrong
/// answer. Placing the flag read after a `GridSync` makes every group
/// observe the same word and reach the same verdict, so the whole grid
/// leaves together or none of it does.
#[test]
fn every_return_is_preceded_by_a_grid_sync_at_the_same_top_level() {
    let max_iterations = 6u32;
    let program = persistent_fixpoint_grid(
        or_const_body(4, 0b1010),
        "current",
        "next",
        "changed",
        4,
        max_iterations,
    );
    let waves = wave_nodes(&program);

    let mut grid_syncs_seen = 0usize;
    let mut returns_seen = 0usize;
    for (index, node) in waves.iter().enumerate() {
        if matches!(
            node,
            Node::Barrier {
                ordering: MemoryOrdering::GridSync
            }
        ) {
            grid_syncs_seen += 1;
        }
        if any_descendant(node, &mut |candidate| matches!(candidate, Node::Return)) {
            assert!(
                grid_syncs_seen > 0,
                "top-level node {index} carries a Return with no preceding GridSync: the exit \
                 would be decided per group and a departing group would strand the rest"
            );
            returns_seen += 1;
            // The barrier that decides THIS wave is the second of the
            // wave, so the count of fences already passed must be even
            // and at least two.
            assert_eq!(
                grid_syncs_seen % 2,
                0,
                "the Return at top-level node {index} must follow the wave's SECOND GridSync \
                 (the one after the compare step), not sit between the two"
            );
        }
    }

    assert_eq!(
        returns_seen, max_iterations as usize,
        "each wave must carry exactly one collective exit"
    );
    assert_eq!(
        count_nodes(&program, |node| matches!(node, Node::Return)),
        max_iterations as usize,
        "no Return may hide anywhere else in the tree"
    );
}

/// The per-iteration `changed` word is only ever `atomic_or`'d, never
/// plain-stored, and each wave owns its own word.
///
/// Locks out the lost-set race. `persistent_fixpoint` clears one shared
/// flag word with a plain `Node::store` and every group sets it with
/// `atomic_or`; a plain store is not ordered against another group's
/// atomic, so the clear can erase a set that already happened and the
/// group whose set was erased reads 0 and exits with unconverged state. A
/// word that is never written except by `atomic_or`, and never reused by
/// a later wave, cannot lose a set at all.
#[test]
fn changed_words_are_only_atomic_ored_and_never_cleared() {
    let max_iterations = 5u32;
    let program = persistent_fixpoint_grid(
        or_const_body(4, 0b1010),
        "current",
        "next",
        "changed",
        4,
        max_iterations,
    );

    assert_eq!(
        count_nodes(&program, |node| matches!(
            node,
            Node::Store { buffer, .. } if buffer.as_str() == "changed"
        )),
        0,
        "a plain store to the flag buffer reintroduces the clear-versus-set race"
    );

    let indices = atomic_or_indices(&program, "changed");
    assert_eq!(
        indices,
        (0..max_iterations).collect::<Vec<u32>>(),
        "wave i must own flag word i exclusively; a shared or reused word is the lost-set race"
    );

    let flag_decl = program
        .buffers
        .iter()
        .find(|buffer| buffer.name() == "changed")
        .expect("flag buffer must be declared");
    assert_eq!(
        flag_decl.count(),
        max_iterations,
        "the flag buffer must be one word per iteration"
    );
    assert_eq!(
        flag_decl.binding(),
        2,
        "flag binding must stay at 2 so callers keep reading outputs positionally"
    );
}

/// The fences between waves are `GridSync`, never `SeqCst`.
///
/// Locks out the workgroup-scope barrier. `MemoryOrdering::SeqCst` orders
/// memory only within a workgroup, so with more than one group it orders
/// nothing between a group's plain clear and another group's `atomic_or`:
/// exactly the lost-set race that lets one group's clear erase another
/// group's flag and send that group home unconverged. `GridSync` is the
/// ordering the driver lowers to a native cooperative grid barrier or a
/// kernel split, which is the only fence that spans groups.
#[test]
fn wave_fences_are_grid_sync_and_never_workgroup_scope() {
    let max_iterations = 7u32;
    let program = persistent_fixpoint_grid(
        carry_body(4),
        "current",
        "next",
        "changed",
        4,
        max_iterations,
    );

    assert_eq!(
        count_nodes(&program, |node| matches!(
            node,
            Node::Barrier {
                ordering: MemoryOrdering::GridSync
            }
        )),
        2 * max_iterations as usize,
        "each wave needs two grid-wide fences: one after the transfer, one after the compare"
    );
    assert_eq!(
        count_nodes(
            &program,
            |node| matches!(node, Node::Barrier { ordering } if !matches!(
                ordering,
                MemoryOrdering::GridSync
            ))
        ),
        0,
        "a workgroup-scope fence between waves orders nothing across groups"
    );
    assert_eq!(
        count_nodes(&program, |node| matches!(node, Node::Loop { .. })),
        0,
        "the wave form must not fall back to an in-kernel convergence loop"
    );
}

/// Exactly `max_iterations` waves are emitted, no more and no fewer.
///
/// Counts the wave-identifying nodes so a fencepost that drops the final
/// wave (`0..max_iterations - 1`) or doubles it (`0..=max_iterations`)
/// fails instead of silently changing the convergence budget.
#[test]
fn exactly_max_iterations_waves_are_emitted() {
    for max_iterations in [1u32, 2, 3, 8, 15] {
        let program = persistent_fixpoint_grid(
            or_const_body(4, 0b1010),
            "current",
            "next",
            "changed",
            4,
            max_iterations,
        );
        let waves = wave_nodes(&program);
        assert_eq!(
            waves.len(),
            5 * max_iterations as usize,
            "each wave is exactly five top-level nodes: transfer, fence, compare, fence, exit"
        );
        assert_eq!(
            atomic_or_indices(&program, "changed"),
            (0..max_iterations).collect::<Vec<u32>>(),
            "flag words must cover 0..max_iterations with no gap and no repeat"
        );
        assert_eq!(
            count_nodes(&program, |node| matches!(node, Node::Return)),
            max_iterations as usize,
            "one collective exit per wave"
        );
        assert_eq!(
            count_nodes(&program, |node| matches!(
                node,
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync
                }
            )),
            2 * max_iterations as usize,
            "two grid fences per wave"
        );
    }
}

/// A `max_iterations == 0` build emits no wave and still declares a legal
/// flag buffer.
///
/// The declared count is one word per iteration, and a zero-count buffer
/// declaration is not a valid binding, so the count is floored at 1 while
/// the body stays empty because no wave runs.
///
/// The zero case is asserted BESIDE a counted control in this same body on
/// purpose. A bare "emitted nothing" assertion cannot tell a correct zero
/// budget from a `wave_nodes` helper or a builder that emits nothing for
/// EVERY budget, because an empty result is indistinguishable from that bug
/// in isolation. Only the contrast discriminates, so the exact counts 0 and
/// 3 are pinned together and the flag buffer is checked to floor at 1 for
/// the zero budget while tracking the real budget at 3.
#[test]
fn zero_budget_emits_no_waves_and_a_floored_flag_buffer() {
    let flag_count = |program: &Program| {
        program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("flag buffer must be declared")
            .count()
    };
    let build = |max_iterations: u32| {
        persistent_fixpoint_grid(
            or_const_body(4, 0b1010),
            "current",
            "next",
            "changed",
            4,
            max_iterations,
        )
    };

    let zero = build(0);
    let three = build(3);

    assert_eq!(
        wave_nodes(&zero).len(),
        0,
        "a zero budget must emit exactly no wave nodes"
    );
    assert_eq!(
        wave_nodes(&three).len(),
        5 * 3,
        "the same builder must still emit five nodes per wave, so the zero \
         case above is a real floor and not a builder that emits nothing"
    );
    assert_eq!(
        flag_count(&zero),
        1,
        "the flag buffer count must be floored at one word"
    );
    assert_eq!(
        flag_count(&three),
        3,
        "a nonzero budget must declare one flag word per iteration, so the \
         floor above is a floor and not a hardcoded one-word buffer"
    );
}

/// The emitted program passes the IR validator even when the caller's
/// transfer body binds names at its own top level.
///
/// The body is spliced once per wave. Splicing it flat would put every
/// copy's top-level `let` in one region as duplicate sibling bindings,
/// which the validator rejects as V032, so each copy gets its own
/// `Node::Block` scope. This is the test that fails if that wrapper is
/// removed.
#[test]
fn repeated_transfer_body_with_top_level_bindings_stays_valid_ir() {
    let body = vec![
        Node::let_bind("carried", Expr::load("current", lane())),
        Node::if_then(
            Expr::lt(lane(), Expr::u32(4)),
            vec![Node::store(
                "next",
                lane(),
                Expr::bitor(Expr::var("carried"), Expr::u32(0b1010)),
            )],
        ),
    ];
    let program = persistent_fixpoint_grid(body, "current", "next", "changed", 4, 4);
    let errors = vyre_foundation::ir::validate(&program);
    assert!(
        errors.is_empty(),
        "grid wave form must be valid IR, got: {errors:?}"
    );
}
