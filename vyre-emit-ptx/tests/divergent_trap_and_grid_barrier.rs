//! A trap under non-uniform control flow must not leave a cooperative grid
//! barrier unarrived.
//!
//! WHY: closes the class "an early kernel exit that strands the whole-grid
//! barrier". `Node::Return` was already refused under a condition the emitter
//! could not prove grid-uniform, because the branch it lowers to would let some
//! invocations leave while others waited for them. A trap ends with the same
//! `bra $L_exit` and had no such check, so a guard trap in a grid-synced kernel
//! lowered silently and hung the launch: the whole-grid barrier is a monotonic
//! counter arrived at by one leader lane per CTA, and a leader that trapped never
//! bumped it, so every other CTA spun on a release target it could no longer
//! reach.
//!
//! The refusal is deliberately narrower than the one on `Return`. A CTA-scope
//! `bar.sync` waits only on non-exited threads, so a trap is safe next to one; and
//! a trap after the kernel's last grid barrier strands no arrival. Both stay
//! lowerable, and both are proved below, because a refusal that fired on them
//! would take every trapping op out of the certificate.
//!
//! What it does not catch: a hang from a lane that leaves through something other
//! than a trap or a `Return`, and a grid barrier whose arrival is lost for any
//! reason other than an early exit. It also says nothing about whether a lowered
//! trap records the right tag, which `trap_readback_launch_coverage` owns.

use vyre_emit_ptx::PtxEmitOptions;
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

#[path = "emit_probe/probe.rs"]
mod emit_probe;
use emit_probe::{lower_and_emit, region_program};

/// Trap tag every probe here declares, so the refusal can be matched on it.
const TAG: &str = "probe_oob";

/// A trap guarded on a per-invocation condition.
///
/// `Expr::gid_x()` is the one thing this emitter treats as varying by
/// construction, so the guard is non-uniform without depending on how far the
/// uniformity analysis happens to reach.
fn divergent_trap() -> Node {
    Node::if_then(
        Expr::gt(Expr::gid_x(), Expr::u32(7)),
        vec![Node::trap(Expr::gid_x(), TAG)],
    )
}

/// One store, one divergent trap and `barriers` whole-grid barriers, with
/// `trap_first` deciding whether the trap precedes them or follows them.
///
/// Everything else is identical between the two orders, so a difference in emit
/// outcome is attributable to whether an arrival remains ahead of the trap.
fn probe(barriers: usize, trap_first: bool) -> Program {
    let mut body = vec![Node::store("state", Expr::gid_x(), Expr::u32(1))];
    if trap_first {
        body.push(divergent_trap());
    }
    for index in 0..barriers {
        // A store between the barriers, so no pass can treat two adjacent
        // identical fences as one and quietly reduce the count under test.
        body.push(Node::barrier_with_ordering(MemoryOrdering::GridSync));
        body.push(Node::store(
            "state",
            Expr::gid_x(),
            Expr::u32(index as u32 + 2),
        ));
    }
    if !trap_first {
        body.push(divergent_trap());
    }
    region_program(
        "divergent-trap-grid-barrier-probe",
        vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(256)],
        body,
    )
}

/// Whole-grid barriers the lowered descriptor actually carries.
///
/// Read from the descriptor rather than from the probe's argument: a lowering
/// pass that merged two fences would otherwise make the count assertion below
/// describe a program that no longer exists.
fn lowered_grid_barriers(program: &Program) -> usize {
    vyre_lower::lower_physical(program)
        .expect("the probe must lower")
        .descriptor()
        .ops_iter()
        .filter(|op| {
            matches!(
                op.kind,
                vyre_lower::KernelOpKind::Barrier { ordering } if ordering.requires_grid_sync()
            )
        })
        .count()
}

/// Emit on the cooperative path, which is the only path that lowers a GridSync
/// barrier at all: with `cooperative_grid_sync` false the emitter refuses every
/// one of them and nothing below would be reached.
fn emit(program: &Program) -> Result<String, String> {
    let mut options = PtxEmitOptions::default();
    options.cooperative_grid_sync = true;
    lower_and_emit(program, options)
}

#[test]
fn a_divergent_trap_before_a_grid_barrier_is_refused_at_emit_time() {
    let error = emit(&probe(1, true))
        .expect_err("a divergent trap with a grid barrier ahead of it must not lower to PTX");

    assert!(
        error.contains("InvalidDescriptor"),
        "the strand must be refused as an invalid descriptor; got: {error}"
    );
    for fragment in [TAG, "not provably uniform", "leader", "arrives", "Fix:"] {
        assert!(
            error.contains(fragment),
            "the refusal must explain the unarrived-barrier mechanism, missing {fragment:?}; got: {error}"
        );
    }
}

#[test]
fn the_refusal_counts_the_barriers_still_ahead_of_the_trap() {
    let program = probe(3, true);
    let expected = lowered_grid_barriers(&program);
    assert!(
        expected > 1,
        "the probe must carry more than one barrier, or counting proves nothing over saying `one`"
    );
    let error = emit(&program).expect_err("barriers ahead of the trap must still be refused");

    assert!(
        error.contains(&format!("{expected} cooperative grid barrier(s) still ahead")),
        "the refusal must count the arrivals the trap would strand; wanted {expected}, got: {error}"
    );
}

#[test]
fn a_divergent_trap_after_the_last_grid_barrier_lowers() {
    let ptx = emit(&probe(1, false))
        .expect("a trap with no arrival left ahead of it strands nothing and must lower");

    assert!(
        ptx.contains("atom.global.add.u32"),
        "the probe must actually contain the lowered grid barrier, or it proves nothing: {ptx}"
    );
    assert!(
        ptx.contains("atom.global.cas.b32"),
        "the probe must actually contain the lowered trap claim, or it proves nothing: {ptx}"
    );
}

#[test]
fn a_divergent_trap_in_a_kernel_with_no_grid_barrier_lowers() {
    let ptx = emit(&probe(0, true))
        .expect("a trap in a kernel with no whole-grid arrival must stay lowerable");

    assert!(
        !ptx.contains("_vyre_grid_barrier"),
        "the no-barrier probe must declare no arrival counter, or it is the wrong control: {ptx}"
    );
    assert!(
        ptx.contains("atom.global.cas.b32"),
        "the trap must still be lowered in the no-barrier case: {ptx}"
    );
}
