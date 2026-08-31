//! The launch domain a program admits, read out of the program's own guards.
//!
//! WHY: a launch span derived from declared buffers takes the widest one, which
//! is the output for a gather and the whole destination for a scatter. Reading
//! the guard instead makes the domain a compiler-owned fact. Three failure
//! modes matter, and each has a contract here: claiming a bound a guard does
//! not give (dropped lanes, wrong results), refusing a bound a guard does give
//! (the oversized dispatch the analysis exists to remove), and a launch of zero
//! workgroups, which records no work at all.
//!
//! The guard forms are enumerated against every comparison the recognizer
//! accepts and every neighbouring one it must reject, so a new accepted form
//! has to be decided here rather than inherited.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};
use vyre_foundation::{admitted_logical_span, guarded_logical_span, launch_covers_full_input_span};

/// Resource span of every program below: far wider than any guard admits, so a
/// narrowed answer is visibly different from the buffer-derived one.
const RESOURCE_SPAN: u32 = 4096;

fn index() -> Expr {
    Expr::logical_index(0)
}

/// A program whose only effect is one store, under `cond`.
fn guarded_store(cond: Expr) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)],
        [256, 1, 1],
        vec![Node::if_then(
            cond,
            vec![Node::store("out", index(), Expr::u32(1))],
        )],
    )
}

/// A program whose only effect is one store, under no guard at all.
fn unguarded_store() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)],
        [256, 1, 1],
        vec![Node::store("out", index(), Expr::u32(1))],
    )
}

/// WHY: every accepted guard form must yield the exact bound it states, since
/// an off-by-one on the tight side drops the last lane and on the loose side
/// keeps the dispatch the analysis exists to shrink. `<=`, `>=`, and `==` are
/// one past their literal; `<` and `>` are the literal.
#[test]
fn every_accepted_guard_form_states_its_exact_bound() {
    let cases: Vec<(&str, Expr, u32)> = vec![
        ("index < 8", Expr::lt(index(), Expr::u32(8)), 8),
        ("index <= 8", Expr::le(index(), Expr::u32(8)), 9),
        ("8 > index", Expr::gt(Expr::u32(8), index()), 8),
        ("8 >= index", Expr::ge(Expr::u32(8), index()), 9),
        ("index == 8", Expr::eq(index(), Expr::u32(8)), 9),
        ("8 == index", Expr::eq(Expr::u32(8), index()), 9),
        (
            "index < 8 && index < 4",
            Expr::and(
                Expr::lt(index(), Expr::u32(8)),
                Expr::lt(index(), Expr::u32(4)),
            ),
            4,
        ),
        (
            "index < 4 && index < 8",
            Expr::and(
                Expr::lt(index(), Expr::u32(4)),
                Expr::lt(index(), Expr::u32(8)),
            ),
            4,
        ),
        (
            "index < 8 && unrelated",
            Expr::and(
                Expr::lt(index(), Expr::u32(8)),
                Expr::lt(Expr::logical_index(1), Expr::u32(2)),
            ),
            8,
        ),
        (
            "index < 8 || index < 4",
            Expr::or(
                Expr::lt(index(), Expr::u32(8)),
                Expr::lt(index(), Expr::u32(4)),
            ),
            8,
        ),
        (
            "chunk * lanes + index < 8",
            Expr::lt(
                Expr::add(Expr::mul(Expr::var("chunk"), Expr::u32(64)), index()),
                Expr::u32(8),
            ),
            8,
        ),
        (
            "index + chunk * lanes <= 8",
            Expr::le(
                Expr::add(index(), Expr::mul(Expr::var("chunk"), Expr::u32(64))),
                Expr::u32(8),
            ),
            9,
        ),
        (
            "8 > chunk + index",
            Expr::gt(Expr::u32(8), Expr::add(Expr::var("chunk"), index())),
            8,
        ),
        (
            "8 >= chunk + index",
            Expr::ge(Expr::u32(8), Expr::add(Expr::var("chunk"), index())),
            9,
        ),
    ];

    for (form, cond, bound) in cases {
        let program = guarded_store(cond);
        assert_eq!(
            guarded_logical_span(&program),
            Some(bound),
            "Fix: `{form}` bounds axis-0 logical index at {bound}."
        );
        assert_eq!(
            admitted_logical_span(&program, RESOURCE_SPAN),
            bound,
            "Fix: `{form}` must cap the {RESOURCE_SPAN}-element resource span."
        );
    }
}

/// WHY: a form the recognizer does not prove must leave the launch at the
/// resource span. Claiming a bound from a guard that does not give one drops
/// lanes, which is a wrong result rather than a slow one.
#[test]
fn no_unproven_guard_form_narrows_the_launch() {
    let cases: Vec<(&str, Expr)> = vec![
        ("index != 8", Expr::ne(index(), Expr::u32(8))),
        ("index > 8", Expr::gt(index(), Expr::u32(8))),
        ("index >= 8", Expr::ge(index(), Expr::u32(8))),
        ("8 < index", Expr::lt(Expr::u32(8), index())),
        ("8 <= index", Expr::le(Expr::u32(8), index())),
        ("axis 1 < 8", Expr::lt(Expr::logical_index(1), Expr::u32(8))),
        ("index < n", Expr::lt(index(), Expr::Var("n".into()))),
        ("index <= u32::MAX", Expr::le(index(), Expr::u32(u32::MAX))),
        (
            "chunk * index < 8",
            Expr::lt(Expr::mul(Expr::var("chunk"), index()), Expr::u32(8)),
        ),
        (
            "chunk - index < 8",
            Expr::lt(Expr::sub(Expr::var("chunk"), index()), Expr::u32(8)),
        ),
        (
            "chunk + axis 1 < 8",
            Expr::lt(
                Expr::add(Expr::var("chunk"), Expr::logical_index(1)),
                Expr::u32(8),
            ),
        ),
        (
            "chunk + index < n",
            Expr::lt(
                Expr::add(Expr::var("chunk"), index()),
                Expr::Var("n".into()),
            ),
        ),
        (
            "chunk + index > 8",
            Expr::gt(Expr::add(Expr::var("chunk"), index()), Expr::u32(8)),
        ),
        (
            "chunk + index <= u32::MAX",
            Expr::le(Expr::add(Expr::var("chunk"), index()), Expr::u32(u32::MAX)),
        ),
    ];

    for (form, cond) in cases {
        let program = guarded_store(cond);
        assert_eq!(
            guarded_logical_span(&program),
            None,
            "Fix: `{form}` proves no constant bound on axis-0 logical index."
        );
        assert_eq!(
            admitted_logical_span(&program, RESOURCE_SPAN),
            RESOURCE_SPAN,
            "Fix: an unbounded effect leaves the resource span standing."
        );
    }
}

/// WHY: `index <= u32::MAX` is the boundary of the `+1` the inclusive forms
/// apply. Wrapping it to zero would claim a launch of no lanes for a guard that
/// admits every lane.
#[test]
fn an_inclusive_guard_at_the_addressable_limit_does_not_wrap() {
    for cond in [
        Expr::le(index(), Expr::u32(u32::MAX)),
        Expr::ge(Expr::u32(u32::MAX), index()),
        Expr::eq(index(), Expr::u32(u32::MAX)),
    ] {
        let program = guarded_store(cond);
        assert_eq!(guarded_logical_span(&program), None);
        assert_eq!(
            admitted_logical_span(&program, RESOURCE_SPAN),
            RESOURCE_SPAN
        );
    }
}

/// WHY: an effect-free program affects no index, so the analysis answers zero
/// and the launch floor is applied by whoever sizes the launch. A grid of zero
/// workgroups records no work at all, so the admitted span never returns zero.
#[test]
fn an_effect_free_program_admits_no_index_and_still_launches_one_group() {
    for entry in [
        Vec::new(),
        vec![Node::barrier_with_ordering(MemoryOrdering::SeqCst)],
        vec![Node::let_bind("i", index())],
    ] {
        let program = Program::wrapped(Vec::new(), [256, 1, 1], entry);
        assert_eq!(
            guarded_logical_span(&program),
            Some(0),
            "Fix: a program with no effect affects no logical index."
        );
        assert_eq!(
            admitted_logical_span(&program, RESOURCE_SPAN),
            1,
            "Fix: a launch of zero workgroups records no work."
        );
        assert_eq!(admitted_logical_span(&program, 0), 1);
    }
}

/// WHY: a zero resource span with a real effect must still launch, and a guard
/// tighter than the resource span must not widen it.
#[test]
fn the_admitted_span_is_the_tighter_of_the_two_and_never_zero() {
    let guarded = guarded_store(Expr::lt(index(), Expr::u32(64)));
    assert_eq!(admitted_logical_span(&guarded, 8), 8);
    assert_eq!(admitted_logical_span(&guarded, 64), 64);
    assert_eq!(admitted_logical_span(&guarded, 4096), 64);
    assert_eq!(admitted_logical_span(&guarded, 0), 1);
    assert_eq!(admitted_logical_span(&unguarded_store(), 0), 1);
}

/// WHY: a local proven equal to axis-0 logical index carries the guard, because
/// that is the form a built program takes. A local the program rebinds does
/// not, and a nested guard narrows to the tighter of the two.
#[test]
fn a_proven_index_local_carries_the_guard_and_a_rebound_one_does_not() {
    let buffers = || vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)];
    let store = || Node::store("out", Expr::Var("i".into()), Expr::u32(1));

    let proven = Program::wrapped(
        buffers(),
        [256, 1, 1],
        vec![
            Node::let_bind("i", index()),
            Node::if_then(Expr::lt(Expr::Var("i".into()), Expr::u32(8)), vec![store()]),
        ],
    );
    assert_eq!(guarded_logical_span(&proven), Some(8));

    let rebound = Program::wrapped(
        buffers(),
        [256, 1, 1],
        vec![
            Node::let_bind("i", Expr::u32(0)),
            Node::if_then(Expr::lt(Expr::Var("i".into()), Expr::u32(8)), vec![store()]),
        ],
    );
    assert_eq!(
        guarded_logical_span(&rebound),
        None,
        "Fix: a guard on a local that is not the logical index bounds nothing."
    );

    let nested = Program::wrapped(
        buffers(),
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(index(), Expr::u32(64)),
            vec![Node::if_then(
                Expr::lt(index(), Expr::u32(8)),
                vec![Node::store("out", index(), Expr::u32(1))],
            )],
        )],
    );
    assert_eq!(guarded_logical_span(&nested), Some(8));
}

/// WHY: a chunked walk owns more cells than the launch has lanes, so it binds
/// `chunk * lanes + index` to a local and guards that local against the cell
/// count. The bound reaches the index through an addition, and an analysis that
/// recognizes only the bare index reports the walk unbounded, which dispatches
/// one lane per declared word instead of one per cell. A local carrying no
/// index still bounds nothing.
#[test]
fn a_chunked_cell_local_carries_its_guard_to_the_effect() {
    let buffers = || vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)];
    let cell = || Expr::add(Expr::mul(Expr::var("chunk"), Expr::u32(64)), index());
    let walk = |bind: Node| {
        Program::wrapped(
            buffers(),
            [64, 1, 1],
            vec![Node::loop_for(
                "chunk",
                Expr::u32(0),
                Expr::u32(4),
                vec![
                    bind,
                    Node::if_then(
                        Expr::lt(Expr::var("cell"), Expr::u32(200)),
                        vec![Node::store(
                            "out",
                            Expr::mul(Expr::var("cell"), Expr::u32(8)),
                            Expr::u32(1),
                        )],
                    ),
                ],
            )],
        )
    };

    let chunked = walk(Node::let_bind("cell", cell()));
    assert_eq!(
        guarded_logical_span(&chunked),
        Some(200),
        "Fix: a guard on `chunk * lanes + index` bounds axis-0 logical index."
    );
    assert_eq!(admitted_logical_span(&chunked, RESOURCE_SPAN), 200);

    let indexless = walk(Node::let_bind("cell", Expr::var("chunk")));
    assert_eq!(
        guarded_logical_span(&indexless),
        None,
        "Fix: a local carrying no logical index bounds nothing."
    );
}

/// WHY: the else arm runs where the guard is false, so the guard's bound does
/// not reach it. Carrying the bound into both arms would narrow a launch over
/// an effect that runs at every lane.
#[test]
fn a_guards_bound_does_not_reach_its_else_arm() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)],
        [256, 1, 1],
        vec![Node::if_then_else(
            Expr::lt(index(), Expr::u32(8)),
            vec![Node::store("out", index(), Expr::u32(1))],
            vec![Node::store("out", index(), Expr::u32(2))],
        )],
    );
    assert_eq!(guarded_logical_span(&program), None);
    assert_eq!(
        admitted_logical_span(&program, RESOURCE_SPAN),
        RESOURCE_SPAN
    );
}

/// WHY: the predicated-tail form is how a built program guards a scan: the
/// comparison is bound to a local, a value is selected to zero outside it, and
/// the effect runs where that value is nonzero. The bound reaches the effect
/// through two locals, so an analysis that only reads branch conditions reports
/// the program unbounded and keeps a dispatch sized from the widest buffer.
/// Masking a zeroed value keeps it zeroed; a select whose other arm is nonzero
/// proves nothing, and neither does a rebound predicate.
#[test]
fn a_predicated_tail_carries_its_bound_through_the_selected_value() {
    let buffers = || vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)];
    let store = || Node::store("out", index(), Expr::u32(1));
    let predicate = || Node::let_bind("in_bounds", Expr::lt(index(), Expr::u32(8)));
    let active = |value: Expr| Node::let_bind("active", value);
    let run_when_active = || {
        Node::if_then(
            Expr::ne(Expr::Var("active".into()), Expr::u32(0)),
            vec![store()],
        )
    };
    let program = |entry: Vec<Node>| Program::wrapped(buffers(), [256, 1, 1], entry);

    let selected = program(vec![
        predicate(),
        active(Expr::select(
            Expr::Var("in_bounds".into()),
            Expr::u32(1),
            Expr::u32(0),
        )),
        run_when_active(),
    ]);
    assert_eq!(
        guarded_logical_span(&selected),
        Some(8),
        "Fix: a value selected to zero outside the guard carries the guard's bound."
    );
    assert_eq!(admitted_logical_span(&selected, RESOURCE_SPAN), 8);

    let masked = program(vec![
        predicate(),
        Node::let_bind(
            "selected",
            Expr::select(Expr::Var("in_bounds".into()), Expr::u32(3), Expr::u32(0)),
        ),
        active(Expr::bitand(Expr::Var("selected".into()), Expr::u32(0xF))),
        run_when_active(),
    ]);
    assert_eq!(
        guarded_logical_span(&masked),
        Some(8),
        "Fix: masking a zeroed value leaves it zero outside the same bound."
    );

    let nonzero_tail = program(vec![
        predicate(),
        active(Expr::select(
            Expr::Var("in_bounds".into()),
            Expr::u32(1),
            Expr::u32(1),
        )),
        run_when_active(),
    ]);
    assert_eq!(
        guarded_logical_span(&nonzero_tail),
        None,
        "Fix: a select whose other arm is nonzero runs the effect at every lane."
    );

    let rebound_predicate = program(vec![
        predicate(),
        Node::assign("in_bounds", Expr::u32(1)),
        active(Expr::select(
            Expr::Var("in_bounds".into()),
            Expr::u32(1),
            Expr::u32(0),
        )),
        run_when_active(),
    ]);
    assert_eq!(
        guarded_logical_span(&rebound_predicate),
        None,
        "Fix: a rebound predicate carries no bound."
    );

    let dynamic_predicate = program(vec![
        Node::let_bind("in_bounds", Expr::lt(index(), Expr::Var("n".into()))),
        active(Expr::select(
            Expr::Var("in_bounds".into()),
            Expr::u32(1),
            Expr::u32(0),
        )),
        run_when_active(),
    ]);
    assert_eq!(
        guarded_logical_span(&dynamic_predicate),
        None,
        "Fix: a predicate against a runtime extent states no constant bound."
    );
}

/// WHY: a loop body runs more than once and the analysis reads it once, so an
/// index local the body reassigns no longer holds the index on the second
/// iteration. Trusting the guard written against it would claim a bound the
/// program does not have and drop every lane above it.
#[test]
fn a_loop_body_that_rebinds_the_index_local_loses_its_bound() {
    let buffers = || vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)];
    let guarded_store_on_local = || {
        Node::if_then(
            Expr::lt(Expr::Var("i".into()), Expr::u32(4)),
            vec![Node::store("out", Expr::Var("i".into()), Expr::u32(1))],
        )
    };

    let stable = Program::wrapped(
        buffers(),
        [256, 1, 1],
        vec![
            Node::let_bind("i", index()),
            Node::loop_(
                "step",
                Expr::u32(0),
                Expr::u32(2),
                vec![guarded_store_on_local()],
            ),
        ],
    );
    assert_eq!(
        guarded_logical_span(&stable),
        Some(4),
        "Fix: a loop that leaves the index local alone keeps the guard."
    );

    let reassigned = Program::wrapped(
        buffers(),
        [256, 1, 1],
        vec![
            Node::let_bind("i", index()),
            Node::loop_(
                "step",
                Expr::u32(0),
                Expr::u32(2),
                vec![
                    guarded_store_on_local(),
                    Node::assign(
                        "i",
                        Expr::add(Expr::Var("i".into()), Expr::u32(RESOURCE_SPAN)),
                    ),
                ],
            ),
        ],
    );
    assert_eq!(
        guarded_logical_span(&reassigned),
        None,
        "Fix: a reassigned index local carries no bound into the next iteration."
    );

    let shadowed = Program::wrapped(
        buffers(),
        [256, 1, 1],
        vec![
            Node::let_bind("i", index()),
            Node::loop_(
                "step",
                Expr::u32(0),
                Expr::u32(2),
                vec![
                    guarded_store_on_local(),
                    Node::let_bind("i", Expr::u32(RESOURCE_SPAN)),
                ],
            ),
        ],
    );
    assert_eq!(
        guarded_logical_span(&shadowed),
        None,
        "Fix: a rebound index local carries no bound into the next iteration."
    );
}

/// WHY: an atomic, a subgroup collective, and a workgroup-scoped buffer each
/// produce a result that depends on how many invocations ran, so a narrower
/// launch changes the value instead of skipping idle lanes. The guard is still
/// there and must not be believed. A shared-memory reduction is the case a
/// guard reads as narrow and the result is not: every lane of the group
/// contributes a partial, and narrowing to the one-element output leaves the
/// rest of the input unreduced.
#[test]
fn a_full_span_effect_keeps_the_resource_span_despite_its_guard() {
    let atomic = Program::wrapped(
        vec![
            BufferDecl::read_write("state", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(RESOURCE_SPAN),
        ],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(index(), Expr::u32(8)),
            vec![Node::store(
                "out",
                index(),
                Expr::atomic_add_ordered(
                    "state",
                    Expr::u32(0),
                    Expr::u32(1),
                    MemoryOrdering::Relaxed,
                ),
            )],
        )],
    );
    assert!(launch_covers_full_input_span(&atomic));
    assert_eq!(guarded_logical_span(&atomic), Some(8));
    assert_eq!(admitted_logical_span(&atomic, RESOURCE_SPAN), RESOURCE_SPAN);

    let subgroup = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN)],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(index(), Expr::u32(8)),
            vec![Node::store(
                "out",
                index(),
                Expr::subgroup_add(Expr::u32(1)),
            )],
        )],
    );
    assert!(launch_covers_full_input_span(&subgroup));
    assert_eq!(guarded_logical_span(&subgroup), Some(8));
    assert_eq!(
        admitted_logical_span(&subgroup, RESOURCE_SPAN),
        RESOURCE_SPAN
    );

    let shared = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(RESOURCE_SPAN),
            BufferDecl::workgroup("scratch", 1, DataType::U32).with_count(256),
        ],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(index(), Expr::u32(8)),
            vec![Node::store("out", index(), Expr::u32(1))],
        )],
    );
    assert!(launch_covers_full_input_span(&shared));
    assert_eq!(guarded_logical_span(&shared), Some(8));
    assert_eq!(admitted_logical_span(&shared, RESOURCE_SPAN), RESOURCE_SPAN);

    assert!(!launch_covers_full_input_span(&unguarded_store()));
}
