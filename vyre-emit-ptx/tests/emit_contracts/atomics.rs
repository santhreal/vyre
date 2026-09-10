//! Test: atomics.
use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, global_rw, lit, op, shared_rw};

/// The shared-memory variant: `slot 0` is a 256-element workgroup bin array.
fn shared_atomic_kernel(atomic_op: AtomicOp) -> KernelDescriptor {
    atomic_kernel(
        "shared_atomic",
        shared_rw(0, DataType::U32, 256, "wg_bins"),
        256,
        atomic_op,
        MemoryOrdering::SeqCst,
        1,
    )
}

#[test]
fn atomic_add_emits_atom_global_add_u32() {
    let kernel = atomic_kernel(
        "atomic_add",
        global_rw(0, DataType::U32, "counter"),
        64,
        AtomicOp::Add,
        MemoryOrdering::SeqCst,
        1,
    );
    let s = emit(&kernel).unwrap();
    assert!(s.contains("atom.global.add.u32"));
}

#[test]
fn atomic_exchange_emits_atom_global_exch_b32() {
    let kernel = atomic_kernel(
        "atomic_exchange",
        global_rw(0, DataType::U32, "slot"),
        64,
        AtomicOp::Exchange,
        MemoryOrdering::SeqCst,
        1,
    );
    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("atom.global.exch.b32"),
        "PTX exch must use bit-size suffix, not .u32:\n{s}"
    );
    assert!(
        !s.contains("atom.global.exch.u32"),
        "ptxas rejects atom.global.exch.u32:\n{s}"
    );
}

#[test]
fn atomic_bitwise_emits_atom_global_b32_suffix() {
    for (atomic_op, mnemonic) in [
        (AtomicOp::And, "and"),
        (AtomicOp::Or, "or"),
        (AtomicOp::Xor, "xor"),
    ] {
        let kernel = atomic_kernel(
            "atomic_bitwise",
            global_rw(0, DataType::U32, "slot"),
            64,
            atomic_op,
            MemoryOrdering::Relaxed,
            1,
        );
        let s = emit(&kernel).unwrap();
        assert!(
            s.contains(&format!("atom.global.{mnemonic}.b32")),
            "PTX atom.{mnemonic} must use .b32, not .u32/.s32:\n{s}"
        );
    }
}

#[test]
fn atomic_bitwise_bool_operand_materializes_u32_before_atom() {
    let kernel = descriptor("atomic_bool_to_b32")
        .slot(global_rw(0, DataType::U32, "slot"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(1, 2),
                    op(KernelOpKind::BinOpKind(BinOp::Eq), [1, 2], 3),
                    op(
                        KernelOpKind::Atomic {
                            op: AtomicOp::Or,
                            ordering: MemoryOrdering::Relaxed,
                        },
                        [0, 0, 3],
                        4,
                    ),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    let atom_line = s
        .lines()
        .find(|line| line.contains("atom.global.or.b32"))
        .expect("atomic OR must emit .b32");
    assert!(
        s.contains("selp.u32"),
        "bool atomic operand must be materialized as 0/1 before atom.global.or.b32:\n{s}"
    );
    assert!(
        !atom_line.contains("], %p"),
        "ptxas rejects predicate operands for atom.global.or.b32; got:\n{atom_line}\n{s}"
    );
}

#[test]
fn atomic_min_max_emit_correct_mnemonic() {
    for (atomic_op, mnemonic) in [(AtomicOp::Min, "min"), (AtomicOp::Max, "max")] {
        let kernel = atomic_kernel(
            "atomic_minmax",
            global_rw(0, DataType::U32, "b"),
            64,
            atomic_op,
            MemoryOrdering::Relaxed,
            7,
        );
        let s = emit(&kernel).unwrap();
        assert!(s.contains(&format!("atom.global.{mnemonic}.u32")));
    }
}

/// An atomic on a workgroup-shared binding must lower to `atom.shared.*`
/// against the shared symbol, never to `atom.global.*`.
///
/// Defect this locks out: `emit_atomic` resolving its address through
/// `slot_to_ptr` (which is populated for global bindings only) and hardcoding
/// the `.global` state space. Shared bindings are absent from `slot_to_ptr`, so
/// the old code failed with the misleading `global pointer not preloaded`
/// binding error even though the IR is well formed and `BufferDecl::workgroup`
/// plus `Expr::atomic_add` compose to exactly this shape. Had the pointer been
/// present, `atom.global` on a shared address is an illegal-address fault or a
/// silently wrong read of unrelated global memory.
///
/// This is the enabler for workgroup-privatized histograms: privatizing 256 bins
/// into shared memory requires an atomic increment on shared memory.
#[test]
fn atomic_add_on_shared_binding_emits_atom_shared_not_atom_global() {
    let s = emit(&shared_atomic_kernel(AtomicOp::Add))
        .expect("Fix: an atomic on a workgroup-shared binding must emit, not error");

    assert!(
        s.contains("atom.shared.add.u32"),
        "Fix: shared-binding atomic add must lower to atom.shared.add.u32; emitted PTX:\n{s}"
    );
    assert!(
        !s.contains("atom.global"),
        "Fix: no atom.global may be emitted for a shared-only binding; a global-space \
         atomic on a shared address faults or corrupts unrelated memory. Emitted PTX:\n{s}"
    );
    assert!(
        s.contains(".shared .align 4 .b8 shared_buf_0[1024];"),
        "Fix: the 256-element u32 shared bin array must be declared as 1024 bytes; \
         emitted PTX:\n{s}"
    );
}

/// The shared atomic's address must be a 32-bit shared-window offset derived
/// from the shared symbol, not a 64-bit global address.
///
/// Defect this locks out: reusing the global `mul.wide.u32` + `add.u64` address
/// arithmetic for a shared operand. PTX shared addresses live in a distinct
/// 32-bit window, so a `.u64` register there is the wrong operand width and
/// addresses the wrong location.
#[test]
fn shared_atomic_address_is_a_shared_window_offset_from_the_shared_symbol() {
    let s = emit(&shared_atomic_kernel(AtomicOp::Add))
        .expect("Fix: an atomic on a workgroup-shared binding must emit, not error");

    let atom_line = s
        .lines()
        .find(|line| line.contains("atom.shared.add.u32"))
        .expect("Fix: shared atomic add must be emitted");
    let addr_reg = atom_line
        .split_once('[')
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inner, _)| inner.trim().to_string())
        .expect("Fix: shared atomic must address through [reg]");

    assert!(
        s.contains(&format!("mov.u32    {addr_reg}, shared_buf_0;"))
            || s.contains(&format!("add.u32    {addr_reg},")),
        "Fix: the shared atomic address register {addr_reg} must be produced by 32-bit \
         shared-symbol arithmetic; emitted PTX:\n{s}"
    );
    assert!(
        s.contains("mov.u32") && s.contains("shared_buf_0"),
        "Fix: the shared atomic must take its base from the shared symbol \
         shared_buf_0; emitted PTX:\n{s}"
    );
}

/// Every RMW mnemonic must honor the shared state space, not just add. A
/// per-mnemonic `.global` literal is exactly how one op gets fixed and the rest
/// keep faulting.
#[test]
fn every_shared_atomic_rmw_mnemonic_uses_the_shared_state_space() {
    for (op, expected) in [
        (AtomicOp::Add, "atom.shared.add.u32"),
        (AtomicOp::Or, "atom.shared.or.b32"),
        (AtomicOp::And, "atom.shared.and.b32"),
        (AtomicOp::Xor, "atom.shared.xor.b32"),
        (AtomicOp::Min, "atom.shared.min.u32"),
        (AtomicOp::Max, "atom.shared.max.u32"),
        (AtomicOp::Exchange, "atom.shared.exch.b32"),
    ] {
        let s = emit(&shared_atomic_kernel(op))
            .unwrap_or_else(|error| panic!("Fix: shared atomic {op:?} must emit: {error:?}"));
        assert!(
            s.contains(expected),
            "Fix: shared atomic {op:?} must emit `{expected}`; emitted PTX:\n{s}"
        );
        assert!(
            !s.contains("atom.global"),
            "Fix: shared atomic {op:?} must not emit an atom.global; emitted PTX:\n{s}"
        );
    }
}

/// What `emit_atomic` must produce when the value operand is a literal zero.
///
/// `AtomicOp` is `#[non_exhaustive]`, so a match outside its crate cannot be
/// checked for exhaustiveness. The fallback arm fails instead of guessing, so
/// a variant added to the data contract and listed in [`EMITTABLE_RMWS`] turns
/// this red until someone records whether zero is an identity for it. A
/// variant added and never listed there is not covered.
fn zero_value_lowering(atomic_op: AtomicOp) -> ZeroValueLowering {
    match atomic_op {
        // `x + 0`, `x | 0` and `x ^ 0` are `x` at every integer width and both
        // signednesses, so the rewrite needs no element type to be sound.
        AtomicOp::Add | AtomicOp::Or | AtomicOp::Xor => ZeroValueLowering::CoherentLoad,
        // `x & 0` is 0, `min(x, 0)` is 0 unsigned, `exch` stores 0, and
        // `max(x, 0)` is `x` only unsigned. None is an identity the emitter can
        // take without reading the element type.
        AtomicOp::And
        | AtomicOp::Min
        | AtomicOp::Max
        | AtomicOp::LruUpdate
        | AtomicOp::Exchange => ZeroValueLowering::KeepsItsAtomic,
        // Compare-exchange takes four operands and lowers through
        // `emit_atomic_cas`; the rest have no single-value RMW mnemonic.
        AtomicOp::CompareExchange
        | AtomicOp::CompareExchangeWeak
        | AtomicOp::FetchNand
        | AtomicOp::Opaque(_) => ZeroValueLowering::NotThisPath,
        _ => panic!(
            "Fix: record whether a zero value operand is an identity for {atomic_op:?} in zero_value_lowering."
        ),
    }
}

enum ZeroValueLowering {
    CoherentLoad,
    KeepsItsAtomic,
    NotThisPath,
}

/// Every single-value RMW mnemonic the emitter can produce.
const EMITTABLE_RMWS: [AtomicOp; 8] = [
    AtomicOp::Add,
    AtomicOp::Or,
    AtomicOp::And,
    AtomicOp::Xor,
    AtomicOp::Min,
    AtomicOp::Max,
    AtomicOp::LruUpdate,
    AtomicOp::Exchange,
];

fn global_atomic_kernel(atomic_op: AtomicOp, value: u32) -> KernelDescriptor {
    atomic_kernel(
        "global_atomic",
        global_rw(0, DataType::U32, "control"),
        64,
        atomic_op,
        MemoryOrdering::SeqCst,
        value,
    )
}

/// An identity read-modify-write on a global binding lowers to a coherent
/// load, and every other operation keeps its atomic.
///
/// An identity RMW writes nothing: it is a read spelled as an atomic, which is
/// how the resident work queue reads its control and status words. Left as
/// `atom`, each one occupies an L2 atomic slot that a plain load does not, and
/// invocations sharing an address serialize on it. `ld.global.cv` returns the
/// same value and bypasses L1 for the same freshness.
///
/// Defect this locks out: extending the rewrite to an operation where zero is
/// not an identity, which silently stops the store the program asked for.
/// `And` and `Exchange` are the dangerous pair, and `Max` is the plausible one
/// because zero is an identity for it at unsigned widths only.
///
/// Does not catch: whether `.cv` is the right cache qualifier for a given
/// memory ordering. The emitter encodes no ordering qualifier on either form,
/// so the two are equally strong in what it emits.
#[test]
fn an_identity_read_modify_write_on_a_global_binding_lowers_to_a_coherent_load() {
    for atomic_op in EMITTABLE_RMWS {
        let s = emit(&global_atomic_kernel(atomic_op, 0))
            .unwrap_or_else(|error| panic!("Fix: global atomic {atomic_op:?} must emit: {error:?}"));
        match zero_value_lowering(atomic_op) {
            ZeroValueLowering::CoherentLoad => {
                assert!(
                    s.contains("ld.global.cv.u32"),
                    "Fix: an identity {atomic_op:?} must lower to `ld.global.cv.u32`; emitted PTX:\n{s}"
                );
                assert!(
                    !s.contains("atom.global"),
                    "Fix: an identity {atomic_op:?} must not also emit an atom.global; emitted PTX:\n{s}"
                );
            }
            ZeroValueLowering::KeepsItsAtomic => {
                assert!(
                    s.contains("atom.global"),
                    "Fix: zero is not an identity for {atomic_op:?}, so it must keep its atom.global; emitted PTX:\n{s}"
                );
                assert!(
                    !s.contains("ld.global.cv"),
                    "Fix: {atomic_op:?} against zero still writes memory and must not become a load; emitted PTX:\n{s}"
                );
            }
            ZeroValueLowering::NotThisPath => {}
        }
    }
}

/// A non-zero value operand never lowers to a load, for any operation.
///
/// Defect this locks out: testing the operation without testing the operand,
/// which turns every `atom.global.add` in the program into a read.
#[test]
fn a_non_zero_value_operand_keeps_its_atomic_for_every_operation() {
    for atomic_op in EMITTABLE_RMWS {
        let s = emit(&global_atomic_kernel(atomic_op, 1))
            .unwrap_or_else(|error| panic!("Fix: global atomic {atomic_op:?} must emit: {error:?}"));
        assert!(
            s.contains("atom.global"),
            "Fix: {atomic_op:?} against 1 must keep its atom.global; emitted PTX:\n{s}"
        );
        assert!(
            !s.contains("ld.global.cv"),
            "Fix: {atomic_op:?} against 1 must not lower to a load; emitted PTX:\n{s}"
        );
    }
}

/// A workgroup-shared identity RMW keeps its atomic.
///
/// Defect this locks out: applying the rewrite to the shared state space, where
/// `ld.shared.cv` is not a legal PTX form and ptxas rejects the module.
#[test]
fn an_identity_read_modify_write_on_a_shared_binding_keeps_its_atomic() {
    for atomic_op in [AtomicOp::Add, AtomicOp::Or, AtomicOp::Xor] {
        let kernel = atomic_kernel(
            "shared_identity",
            shared_rw(0, DataType::U32, 256, "wg_bins"),
            256,
            atomic_op,
            MemoryOrdering::SeqCst,
            0,
        );
        let s = emit(&kernel)
            .unwrap_or_else(|error| panic!("Fix: shared atomic {atomic_op:?} must emit: {error:?}"));
        assert!(
            s.contains("atom.shared"),
            "Fix: an identity {atomic_op:?} on a shared binding must keep atom.shared; emitted PTX:\n{s}"
        );
        assert!(
            !s.contains(".cv."),
            "Fix: ld.shared.cv is not a legal PTX form; emitted PTX:\n{s}"
        );
    }
}
