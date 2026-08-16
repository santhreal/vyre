//! WHY: closes the class "this backend advertises a capability its emitter does
//! not implement", for the one capability where the consequence is silent wrong
//! data. `Node::trap` is how an op refuses an input outside its declared domain.
//! The PTX emitter lowers `KernelOpKind::Trap` to a source comment and a branch
//! to the kernel exit: the trapping lane leaves and nothing is recorded, so the
//! host cannot tell a refused dispatch from a completed one. While that is true,
//! a CUDA device profile reporting `supports_trap_propagation: true` makes
//! `vyre_foundation::program_caps::check_backend_capabilities` admit a
//! trap-declaring program, and every guard in it is a guard that does not exist.
//! Nine source files in this workspace declare traps.
//!
//! The assertion is a two-sided implication between what the emitter emits and
//! what the profile advertises, so it goes red in both directions: red today,
//! because the profile says true and the PTX addresses no sidecar; red again if
//! someone implements the sidecar and forgets to flip the profile.
//!
//! The variant space is one path, not a roster: the emitter has a single `Trap`
//! arm and the lowering inserts a single reserved sidecar binding, so one
//! trap-declaring descriptor exercises the whole class. What it does not catch:
//! whether a recorded trap is read back correctly, whether the recorded lane and
//! address are the trapping ones, and any other capability this profile
//! advertises.

use vyre_driver_cuda::CudaBackend;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_lower::TRAP_SIDECAR_NAME;

const OP_ID: &str = "test::trap_propagation_capability";

/// A program that refuses a zero input lane, which is the shape every domain
/// guard in the workspace has.
fn trap_declaring_program() -> Program {
    let lane = Expr::InvocationId { axis: 0 };
    let value = Expr::load("input", lane.clone());
    let body = vec![Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(4)),
        vec![
            Node::if_then(
                Expr::eq(value.clone(), Expr::u32(0)),
                vec![Node::trap(value.clone(), "test-zero-input-lane")],
            ),
            Node::store("out", lane, value),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

#[test]
fn cuda_advertises_trap_propagation_only_when_its_ptx_records_a_trap() {
    let program = trap_declaring_program();
    let lowered = vyre_lower::lower_verified(&program)
        .expect("Fix: a trap-declaring program must lower before its emission can be judged.");
    assert!(
        lowered
            .descriptor
            .bindings
            .slots
            .iter()
            .any(|slot| slot.name == TRAP_SIDECAR_NAME),
        "Fix: the lowering must insert the reserved `{TRAP_SIDECAR_NAME}` binding for a trap-declaring program, otherwise no backend has anywhere to record a trap and this contract judges nothing."
    );
    let ptx = vyre_emit_ptx::emit(&lowered.descriptor)
        .expect("Fix: a trap-declaring descriptor must emit PTX.");
    // The reserved sidecar is the only place a device can record a trap, and both
    // an entry-parameter binding and a module-scope global name it. Addressing it
    // is what "propagates a trap"; a comment and a branch to the exit do not.
    let records_trap = ptx.contains("trap_sidecar");

    let backend = CudaBackend::acquire()
        .expect("Fix: CudaBackend::acquire must succeed on a GPU-required machine.");
    let advertised = backend.device_profile().supports_trap_propagation;

    assert_eq!(
        records_trap, advertised,
        "Fix: the CUDA device profile reports supports_trap_propagation={advertised} while its PTX for a trap-declaring descriptor {} the reserved trap sidecar. Advertising true without a record admits a program whose domain guards then do nothing and returns wrong data with no error; emitting a record without advertising true refuses a program the backend can in fact run. Change both together.",
        if records_trap { "addresses" } else { "never mentions" }
    );
}
