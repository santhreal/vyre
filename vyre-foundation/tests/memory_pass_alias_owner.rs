//! Every memory pass must ask `optimizer::passes::memory::alias` whether it may
//! rewrite across a gap, rather than carry its own copy of that proof.
//!
//! # The class this closes
//!
//! `dead_store_elim` and `store_to_load_forward` each carried a full
//! node-by-node "does this interfere with buffer `b`" match. Both were
//! exhaustive, both were tested, and they disagreed: a foreign
//! compare-exchange -- a lock acquisition, past which another invocation's
//! writes to `b` become visible -- blocked the dead-store proof and did not
//! block the forwarding proof. Forwarding therefore replaced a load with a
//! stale literal across a lock acquire. Nothing could see the disagreement
//! from inside either pass.
//!
//! So the assertion here is not about either pass by name. The pass set is read
//! from the registry at run time and every memory pass is held to the owner's
//! answer, which means a THIRD memory pass that reintroduces a private copy of
//! the analysis turns this suite red on the day it is registered rather than on
//! the day someone diffs two files.
//!
//! # Why the probes look like this
//!
//! Each probe is a rewrite the pass set demonstrably performs -- the control
//! case proves that -- with one node inserted into the gap that the alias owner
//! reports as interfering. The only thing standing between green and red is
//! whether the pass consulted the owner.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::{registered_pass_registrations, PassPhase, ProgramPass};
use vyre_foundation::visit::{for_each_expr, for_each_node};

/// A rewrite the memory passes perform, and the same rewrite with one
/// interfering node dropped into the gap.
struct AliasProbe {
    /// What the gap node is, for the failure message.
    label: &'static str,
    /// The rewrite with a harmless node in the gap. At least one memory pass
    /// must fire, or the blocked case proves nothing.
    control: fn() -> Program,
    /// The same rewrite with an interfering node in the gap. No memory pass
    /// may fire.
    blocked: fn() -> Program,
}

const PROBES: &[AliasProbe] = &[
    AliasProbe {
        label: "compare-exchange on another buffer between two stores to one lane",
        control: || dead_store_program(harmless_gap()),
        blocked: || dead_store_program(lock_acquire_gap()),
    },
    AliasProbe {
        label: "compare-exchange on another buffer between a store and its load",
        control: || forwarding_program(harmless_gap()),
        blocked: || forwarding_program(lock_acquire_gap()),
    },
    AliasProbe {
        label: "host trap between two stores to one lane",
        control: || dead_store_program(harmless_gap()),
        blocked: || dead_store_program(Node::trap(Expr::u32(0), "probe")),
    },
    AliasProbe {
        label: "host trap between a store and its load",
        control: || forwarding_program(harmless_gap()),
        blocked: || forwarding_program(Node::trap(Expr::u32(0), "probe")),
    },
];

/// The buffer every probe accesses on both sides of the gap.
const PROBE_BUFFER: &str = "a";

/// `a` and `sink` are declared outputs so `dead_buffer_elim` cannot delete the
/// buffer the probe measures; `lock` is ordinary storage, since only the
/// blocked variants name it.
fn buffers() -> Vec<BufferDecl> {
    vec![
        BufferDecl::output(PROBE_BUFFER, 0, DataType::U32).with_count(4),
        BufferDecl::output("sink", 1, DataType::U32).with_count(4),
        BufferDecl::storage("lock", 2, BufferAccess::ReadWrite, DataType::U32).with_count(4),
    ]
}

/// A gap node that touches no buffer at all: the passes must look straight
/// past it.
fn harmless_gap() -> Node {
    Node::let_bind("gap", Expr::u32(1))
}

/// Taking a lock held in `lock`. It never names the probe buffer, and past a
/// successful one another invocation's writes to it are visible.
fn lock_acquire_gap() -> Node {
    Node::let_bind(
        "gap",
        Expr::atomic_compare_exchange("lock", Expr::u32(0), Expr::u32(0), Expr::u32(1)),
    )
}

/// The probe buffer's lane 0 written twice with `gap` in between: the first
/// store is dead unless something in the gap can observe it.
fn dead_store_program(gap: Node) -> Program {
    Program::wrapped(
        buffers(),
        [1, 1, 1],
        vec![
            Node::store(PROBE_BUFFER, Expr::u32(0), Expr::u32(1)),
            gap,
            Node::store(PROBE_BUFFER, Expr::u32(0), Expr::u32(2)),
        ],
    )
}

/// The probe buffer's lane 0 written then read with `gap` in between: the load
/// forwards to the stored literal unless something in the gap can write it.
fn forwarding_program(gap: Node) -> Program {
    Program::wrapped(
        buffers(),
        [1, 1, 1],
        vec![
            Node::store(PROBE_BUFFER, Expr::u32(0), Expr::u32(7)),
            gap,
            Node::let_bind("x", Expr::load(PROBE_BUFFER, Expr::u32(0))),
            Node::store("sink", Expr::u32(0), Expr::var("x")),
        ],
    )
}

/// Every registered pass in [`PassPhase::Memory`], as `(name, instance)`.
///
/// Read from the inventory registry, not from a list here, so registering a new
/// memory pass enrolls it in this gate with no edit to this file.
fn memory_passes() -> Vec<(&'static str, Box<dyn ProgramPass>)> {
    let registrations =
        registered_pass_registrations().expect("the registered pass graph must schedule");
    let passes: Vec<_> = registrations
        .iter()
        .filter(|registration| registration.metadata.phase == PassPhase::Memory)
        .map(|registration| (registration.metadata.name, (registration.factory)()))
        .collect();
    assert!(
        passes.len() >= 2,
        "the alias owner exists because more than one memory pass needs it; \
         found {} in the registry, so this gate is not proving anything",
        passes.len()
    );
    passes
}

/// How many times `program` reads and writes `PROBE_BUFFER`.
///
/// The pass-independent observable: a memory pass that elides a store or
/// forwards a load moves one of these two numbers, and a pass that only
/// annotates layout or drops an unrelated buffer moves neither. Asserting on
/// the counts rather than on `PassResult::changed` holds every memory pass to
/// the alias answer without this file knowing what any of them do.
fn probe_buffer_accesses(program: &Program) -> BufferAccessCount {
    let mut count = BufferAccessCount::default();
    for_each_node(program.entry(), |node| {
        if matches!(node, Node::Store { buffer, .. } if buffer.as_ref() == PROBE_BUFFER) {
            count.stores += 1;
        }
    });
    for_each_expr(program.entry(), |expr| {
        if matches!(expr, Expr::Load { buffer, .. } if buffer.as_ref() == PROBE_BUFFER) {
            count.loads += 1;
        }
    });
    count
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct BufferAccessCount {
    stores: usize,
    loads: usize,
}

#[test]
fn no_memory_pass_elides_an_access_across_a_node_the_alias_owner_blocks() {
    for probe in PROBES {
        let control = (probe.control)();
        let blocked = (probe.blocked)();
        let control_before = probe_buffer_accesses(&control);
        let blocked_before = probe_buffer_accesses(&blocked);

        let mut elided_on_control = Vec::new();
        for (name, pass) in memory_passes() {
            let after = probe_buffer_accesses(&pass.transform(control.clone()).program);
            if after != control_before {
                elided_on_control.push(name);
            }
        }
        assert!(
            !elided_on_control.is_empty(),
            "probe `{}`: no memory pass touches an access to `{PROBE_BUFFER}` in \
             the control program, so the blocked case cannot tell a pass that \
             consulted the alias owner from one that had nothing to do",
            probe.label
        );

        for (name, pass) in memory_passes() {
            let after = probe_buffer_accesses(&pass.transform(blocked.clone()).program);
            assert_eq!(
                after, blocked_before,
                "probe `{}`: `{name}` changed the accesses to `{PROBE_BUFFER}` \
                 across a gap node that `optimizer::passes::memory::alias` \
                 reports as interfering. A memory pass must take its \
                 interference answer from that owner; a private copy is how the \
                 dead-store and forwarding proofs came to disagree about a lock \
                 acquire. Passes that fired on the control: {elided_on_control:?}",
                probe.label
            );
        }
    }
}

#[test]
fn a_load_across_a_lock_acquire_is_not_forwarded_to_the_stored_value() {
    // The concrete miscompile the shared owner fixes, asserted on the IR rather
    // than on which pass produced it: `x` must still be a Load of `a[0]` after
    // the whole memory phase, because the lock acquire admits a concurrent
    // write to `a` that the stored literal 7 does not reflect.
    let optimized = memory_passes().into_iter().fold(
        forwarding_program(lock_acquire_gap()),
        |program, (_, pass)| pass.transform(program).program,
    );
    assert_eq!(
        probe_buffer_accesses(&optimized).loads,
        1,
        "`Let(x, Load({PROBE_BUFFER}, 0))` must survive a compare-exchange in \
         the gap; forwarding the stored literal across a lock acquire publishes \
         a stale value. Got {optimized:?}"
    );
}
