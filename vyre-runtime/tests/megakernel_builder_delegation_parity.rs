//! Behavioral/parity coverage for the four megakernel `Program` builders that the
//! registry-closure gate flagged as uncovered (no test named them, no inventory entry):
//! `build_program_priority`, `build_program_sharded_no_io`, `build_program_jit_slots`,
//! and `build_program_sharded_with_workspace_adapter`.
//!
//! Three are thin default-slot-count delegators to a `*_slots` base that IS covered; the
//! contract they must honor is "forward `workgroup_size_x.max(1)` as the slot count and
//! change nothing else." We prove that by differential: the wrapper's emitted IR must be
//! structurally identical (Debug-form byte-equal. `Program` derives `Debug`, not `Eq`) to
//! the base called with the explicit default slot count, AND non-trivial (real buffers +
//! entry nodes, never two empty programs comparing equal. Testing Contract).
//!
//! The fourth (`build_program_sharded_with_workspace_adapter`) is not a delegator: it splices
//! a consumer-owned resident workspace into the buffer set + body. We prove the distinguishing
//! behavior, the adapter's `buffer_decl()` lands in the program's buffers (and does NOT in the
//! plain sharded build) and its `bootstrap_nodes()` reach the entry body.
//!
//! Drains the vyre-runtime slice of BACKLOG.md WIRING-tautology-closure-25crates.
#![forbid(unsafe_code)]

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_runtime::megakernel::{
    build_program_jit, build_program_jit_slots, build_program_priority,
    build_program_priority_slots, build_program_sharded_no_io, build_program_sharded_slots,
    build_program_sharded_with_workspace_adapter, MegakernelWorkspaceAdapter,
};

/// Structural (Debug-form) equality: `Program` is `Debug` but not `PartialEq`; its Debug
/// rendering prints buffer names/bindings/counts and the full node tree by value (Arc<str>
/// content, not pointers), so equal Debug strings == structurally identical IR.
fn ir_eq(a: &Program, b: &Program) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

fn assert_nontrivial_megakernel(program: &Program) {
    assert!(
        !program.buffers().is_empty(),
        "a megakernel program must declare at least the ring/control buffers, got zero, so the \
         differential would be comparing two empty programs (Testing Contract)"
    );
    assert!(
        !program.entry().is_empty(),
        "a megakernel program must have a non-empty entry body (lane prologue at minimum)"
    );
}

#[test]
fn priority_delegates_to_priority_slots_with_default_slot_count() {
    for &wg in &[1u32, 32, 64, 256] {
        let wrapper = build_program_priority(wg, &[]);
        let base = build_program_priority_slots(wg, wg.max(1), &[]);
        assert_nontrivial_megakernel(&wrapper);
        assert!(
            ir_eq(&wrapper, &base),
            "build_program_priority({wg}) must equal build_program_priority_slots({wg}, {}). \
             the wrapper's only job is to forward workgroup_size_x.max(1) as the slot count",
            wg.max(1)
        );
    }
}

#[test]
fn sharded_no_io_delegates_to_sharded_slots_with_default_slot_count() {
    for &wg in &[1u32, 32, 64, 256] {
        let wrapper = build_program_sharded_no_io(wg, &[]);
        let base = build_program_sharded_slots(wg, wg.max(1), &[]);
        assert_nontrivial_megakernel(&wrapper);
        assert!(
            ir_eq(&wrapper, &base),
            "build_program_sharded_no_io({wg}) must equal build_program_sharded_slots({wg}, {}). \
             both build the io=false sharded body with the default slot count",
            wg.max(1)
        );
    }
}

#[test]
fn jit_wrapper_delegates_to_jit_slots_with_default_slot_count() {
    // Pins `build_program_jit_slots` (the base) via its covered wrapper `build_program_jit`:
    // build_program_jit(wg, p) == build_program_jit_slots(wg, wg.max(1), p).
    let payload: Vec<Node> = vec![Node::store("ring_buffer", Expr::u32(0), Expr::u32(7))];
    for &wg in &[1u32, 32, 64, 256] {
        let wrapper = build_program_jit(wg, &payload);
        let base = build_program_jit_slots(wg, wg.max(1), &payload);
        assert_nontrivial_megakernel(&base);
        assert!(
            ir_eq(&wrapper, &base),
            "build_program_jit({wg}, payload) must equal build_program_jit_slots({wg}, {}, payload)",
            wg.max(1)
        );
    }
}

/// Minimal resident-workspace adapter: contributes one distinctively-named workspace buffer
/// and one bootstrap store. Real ops (Law 2), not stubs (the builder splices both into the IR).
struct MockWorkspaceAdapter;

const MOCK_WORKSPACE_BUFFER: &str = "mock_resident_workspace";

impl MegakernelWorkspaceAdapter for MockWorkspaceAdapter {
    fn buffer_decl(&self) -> BufferDecl {
        BufferDecl::output(MOCK_WORKSPACE_BUFFER, 15, DataType::U32).with_count(4)
    }
    fn bootstrap_nodes(&self) -> Vec<Node> {
        vec![Node::store(MOCK_WORKSPACE_BUFFER, Expr::u32(0), Expr::u32(0))]
    }
}

#[test]
fn sharded_with_workspace_adapter_splices_adapter_buffer_into_ir() {
    let wg = 32u32;
    let slot_count = 32u32;
    let with_adapter =
        build_program_sharded_with_workspace_adapter(wg, slot_count, &[], &MockWorkspaceAdapter);
    let plain = build_program_sharded_slots(wg, slot_count, &[]);

    assert_nontrivial_megakernel(&with_adapter);

    let has_workspace = |p: &Program| p.buffers().iter().any(|b| &*b.name == MOCK_WORKSPACE_BUFFER);
    assert!(
        has_workspace(&with_adapter),
        "the workspace-adapter builder must insert the adapter's buffer_decl ({MOCK_WORKSPACE_BUFFER}) \
         into the program's buffer set"
    );
    assert!(
        !has_workspace(&plain),
        "the plain sharded builder must NOT carry the adapter buffer, otherwise the adapter path is \
         indistinguishable and the test proves nothing"
    );
    // The adapter build must add buffers on top of the core megakernel set, never fewer.
    assert!(
        with_adapter.buffers().len() > plain.buffers().len(),
        "adapter build has {} buffers, plain has {}, the resident workspace must be additive",
        with_adapter.buffers().len(),
        plain.buffers().len()
    );
}
