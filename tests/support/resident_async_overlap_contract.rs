//! The resident asynchronous overlap contract, owned by the workspace.
//!
//! Two backends carried this test verbatim apart from acquisition, message text
//! and one assertion. The property is not backend-specific: two independent
//! mutable resident slots submitted back to back must both stay live until each
//! is retired, neither may recycle a command buffer, pipeline, sizes buffer or
//! resident resource before retirement, and timed retirement must separate
//! enqueue time from wait time.
//!
//! Device-measured time is the one position that genuinely differs, so it is a
//! parameter rather than a shared assumption: a backend that exposes a device
//! timer must report it, and one that does not is not silently excused from the
//! rest of the contract.
//!
//! Shared the same way as `tests/support/preferred_dispatch_backend_contract.rs`:
//! each consumer includes this file with `#[path]`.

#![allow(dead_code)]

use vyre_driver::{BackendError, DispatchConfig, Resource, VyreBackend};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Constant the fixture program adds to each input lane.
const ADDEND: u32 = 5;

/// One single-element u32 resident lane.
fn lane(name: &str, slot: u32, access: BufferAccess) -> BufferDecl {
    BufferDecl::storage(name, slot, access, DataType::U32).with_count(1)
}

/// The fixture: one read-write output lane, one read-only input lane, out = in + [`ADDEND`].
pub(crate) fn add_program() -> Program {
    Program::wrapped(
        vec![
            lane("out", 0, BufferAccess::ReadWrite),
            lane("input", 1, BufferAccess::ReadOnly),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::load("input", Expr::u32(0)), Expr::u32(ADDEND)),
        )],
    )
}

/// Whether a backend's timed retirement carries device-measured elapsed time.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceTiming {
    /// The backend exposes a device timer and must attribute time to it.
    Reported,
    /// The backend has no device timer on this path, so only host phases are asserted.
    Unreported,
}

/// WHY: resident callers need independent mutable slots to overlap queue work.
/// Both submissions must stay live before either result is retired, and timed
/// retirement must preserve device-owned attribution where the backend has a
/// device timer.
pub(crate) fn assert_resident_async_slots_retire_independently(
    backend: &dyn VyreBackend,
    label: &str,
    timing: DeviceTiming,
) {
    let program = add_program();
    let slots = [
        allocate(backend, label, "first output"),
        allocate(backend, label, "first input"),
        allocate(backend, label, "second output"),
        allocate(backend, label, "second input"),
    ];
    let [out_a, input_a, out_b, input_b] = &slots;

    let run = (|| {
        backend.upload_resident(out_a, &[0; 4])?;
        backend.upload_resident(input_a, &37_u32.to_le_bytes())?;
        backend.upload_resident(out_b, &[0; 4])?;
        backend.upload_resident(input_b, &91_u32.to_le_bytes())?;
        let pending_a = backend.dispatch_resident_async(
            &program,
            &[out_a.clone(), input_a.clone()],
            &DispatchConfig::default(),
        )?;
        let pending_b = backend.dispatch_resident_async(
            &program,
            &[out_b.clone(), input_b.clone()],
            &DispatchConfig::default(),
        )?;
        Ok::<_, BackendError>((
            pending_a.await_timed_result()?,
            pending_b.await_timed_result()?,
        ))
    })();
    let released = release(backend, &slots);

    let (timed_a, timed_b) = run.unwrap_or_else(|error| {
        panic!("Fix: both {label} resident async slots must retire with exact outputs: {error}")
    });
    released
        .unwrap_or_else(|error| panic!("Fix: {label} resident slot cleanup must succeed: {error}"));

    assert_eq!(
        timed_a.outputs,
        vec![(37 + ADDEND).to_le_bytes()],
        "Fix: {label} must retire the first resident slot with its own result"
    );
    assert_eq!(
        timed_b.outputs,
        vec![(91 + ADDEND).to_le_bytes()],
        "Fix: {label} must retire the second resident slot with its own result"
    );
    for timed in [timed_a, timed_b] {
        assert!(
            timed.enqueue_ns.is_some() && timed.wait_ns.is_some(),
            "Fix: native {label} asynchronous retirement must separate enqueue and wait time"
        );
        if timing == DeviceTiming::Reported {
            assert!(
                timed.device_ns.unwrap_or_default() > 0,
                "Fix: native {label} asynchronous retirement must preserve device timestamp \
                 attribution so release benchmarks do not fall back to readback wall time"
            );
        }
    }
}

fn allocate(backend: &dyn VyreBackend, label: &str, role: &str) -> Resource {
    backend.allocate_resident(4).unwrap_or_else(|error| {
        panic!("Fix: {label} must allocate the {role} resident slot: {error}")
    })
}

fn release(backend: &dyn VyreBackend, slots: &[Resource]) -> Result<(), BackendError> {
    let mut first_error = None;
    for slot in slots {
        if let Err(error) = backend.free_resident(slot.clone()) {
            first_error = first_error.or(Some(error));
        }
    }
    first_error.map_or(Ok(()), Err)
}
