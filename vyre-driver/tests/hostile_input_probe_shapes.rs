//! Shapes of the probe programs every backend's hostile-input obligations rest on.
//!
//! WHY: the obligations in `vyre_driver::hostile_input_closure` are asserted
//! against a fixed count of supplied buffers. `assert_hostile_bytes_stay_actionable`
//! dispatches `single_output_program` with exactly one byte vector,
//! `assert_trailing_inputs_rejected` supplies three to a two-buffer program to make
//! the third the over-supply, and `assert_zero_workgroup_rejected` relies on the X
//! dimension being the unlaunchable one. Let a probe's declaration drift and every
//! one of those assertions still passes while testing something else: an
//! over-supplied hostile case fails for the wrong reason, the trailing-input case
//! becomes the correct call, and the zero-workgroup case launches. Backends inherit
//! the drift silently because they only see the helper, never the probe.
//!
//! What this does not catch: whether a backend honours the obligations. That is
//! the concrete driver crates' contract targets, which call the helpers.

#![cfg(feature = "test-fixtures")]
#![forbid(unsafe_code)]

use vyre_driver::hostile_input_closure::{
    read_one_write_one_program, single_output_program, zero_workgroup_program,
};
use vyre_foundation::ir::BufferAccess;

#[test]
fn single_output_program_declares_one_caller_supplied_slot() {
    let program = single_output_program(7);
    let buffers = program.buffers();
    assert_eq!(
        buffers.len(),
        1,
        "Fix: the hostile-bytes probe is dispatched with exactly one supplied buffer, so it must \
         declare exactly one. Declared: {:?}",
        buffers.iter().map(|b| b.name()).collect::<Vec<_>>()
    );
    assert_eq!(
        buffers[0].access(),
        BufferAccess::ReadWrite,
        "Fix: the hostile-bytes probe slot must stay ReadWrite so the hostile slice is the value \
         under test."
    );
    assert!(
        !buffers[0].is_backend_allocated_output(),
        "Fix: the hostile-bytes probe slot must consume the caller's bytes. A backend-allocated \
         output receives no host bytes, so the one supplied hostile slice would become an \
         over-supply and every hostile case would fail for the wrong reason."
    );
    assert_eq!(
        program.workgroup_size(),
        [1, 1, 1],
        "Fix: the hostile-bytes probe must stay launchable; a zero dimension would be refused \
         before the hostile slice is ever read."
    );
}

#[test]
fn read_one_write_one_program_declares_exactly_two_caller_supplied_slots() {
    let program = read_one_write_one_program();
    let buffers = program.buffers();
    assert_eq!(
        buffers.len(),
        2,
        "Fix: the trailing-input probe defines two as the correct call and three as the \
         over-supply, so it must declare exactly two buffers. Declared: {:?}",
        buffers.iter().map(|b| b.name()).collect::<Vec<_>>()
    );
    assert_eq!(
        buffers[0].access(),
        BufferAccess::ReadOnly,
        "Fix: the trailing-input probe's first slot is the read input."
    );
    assert_eq!(
        buffers[1].access(),
        BufferAccess::ReadWrite,
        "Fix: the trailing-input probe's second slot is the written one."
    );
    for buffer in buffers {
        assert!(
            !buffer.is_backend_allocated_output(),
            "Fix: buffer `{}` must consume a caller-supplied slot, otherwise two buffers is \
             already an over-supply and the third one proves nothing.",
            buffer.name()
        );
    }
    assert_eq!(
        program.workgroup_size(),
        [1, 1, 1],
        "Fix: the trailing-input probe must stay launchable so the rejection is attributable to \
         the extra buffer."
    );
}

#[test]
fn zero_workgroup_program_zeroes_only_the_x_dimension() {
    let workgroup = zero_workgroup_program().workgroup_size();
    assert_eq!(
        workgroup[0], 0,
        "Fix: the zero-workgroup probe must keep a zero X dimension. Without it the dispatch \
         succeeds and every backend's zero-workgroup assertion passes vacuously."
    );
    assert_eq!(
        [workgroup[1], workgroup[2]],
        [1, 1],
        "Fix: the zero-workgroup probe isolates one zeroed dimension, so Y and Z stay 1. Got: \
         {workgroup:?}"
    );
}
