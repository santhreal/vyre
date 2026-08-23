//! WHY: closes the class "this backend advertises a capability its emitter does
//! not implement", for the one capability where the consequence is silent wrong
//! data. `Node::trap` is how an op refuses an input outside its declared domain.
//! A CUDA device profile reporting `supports_trap_propagation: true` makes
//! `vyre_foundation::program_caps::check_backend_capabilities` admit a
//! trap-declaring program; if the emitter then drops the trap, every guard in that
//! program is a guard that does not exist and the launch returns wrong data with
//! no error. Nine source files in this workspace declare traps.
//!
//! The assertion is a two-sided implication between what the emitter emits and
//! what the profile advertises, so it goes red in both directions: red if someone
//! advertises the capability while the emitter stops writing a record, and red if
//! the emitter writes one while the profile still says false.
//!
//! "Emits a record" is checked as the module-scope symbol AND the atomic claim,
//! not as a mention of the name. A declaration with no store is a reserved four
//! words no lane ever writes, which reads back as zero on every launch and
//! therefore reports no trap: exactly the fail-open shape a name-only check would
//! call success.
//!
//! The variant space is one path, not a roster: the emitter has a single `Trap`
//! arm and the lowering inserts a single reserved sidecar binding, so one
//! trap-declaring descriptor exercises the whole class. What it does not catch:
//! whether the host reads the record back on every launch path (see
//! `trap_readback_launch_coverage`), whether the recorded lane and address are the
//! trapping ones, and any other capability this profile advertises.

#![cfg(feature = "device-tests")]

use vyre_driver::VyreBackend;
use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration};
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
    let lowered = vyre_lower::lower_physical(&program)
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
    let ptx = vyre_emit_ptx::emit(lowered.descriptor())
        .expect("Fix: a trap-declaring descriptor must emit PTX.");
    // The reserved sidecar is the only place a device can record a trap. Both the
    // module-scope declaration and the compare-and-swap that claims word 0 must be
    // present: a declaration alone reserves memory nothing writes, which reads back
    // as zero and reports no trap on a launch that trapped.
    let declares_sidecar = ptx.contains(vyre_emit_ptx::TRAP_SIDECAR_SYMBOL);
    let claims_record = ptx.contains("atom.global.cas.b32");
    let records_trap = declares_sidecar && claims_record;

    let backend = CudaBackend::acquire()
        .expect("Fix: CudaBackend::acquire must succeed on a GPU-required machine.");
    // Read through the registration, because that is the profile the runtime
    // admits a program against: `CudaBackend` itself is the device handle and does
    // not answer capability questions.
    let advertised = CudaBackendRegistration::new(backend)
        .device_profile()
        .supports_trap_propagation;

    assert_eq!(
        records_trap, advertised,
        "Fix: the CUDA device profile reports supports_trap_propagation={advertised} while its PTX for a trap-declaring descriptor declares the sidecar={declares_sidecar} and claims the record={claims_record}. Advertising true without a written record admits a program whose domain guards then do nothing and returns wrong data with no error; writing a record without advertising true refuses a program the backend can in fact run. Change both together."
    );
}
