//! Launch geometry resolved for a runtime-sized buffer and frozen into its artifact.

use vyre::ir::{BufferDecl, DataType};
use vyre_conform::production::ProductionSession;

use super::harness::{assert_all_three_return, expected_bytes, inputs_for, xor_program};

// ---------------------------------------------------------------------------
// The launch grid. Same compile-time zero, second symptom.
// ---------------------------------------------------------------------------

/// A countless `read_write` launches enough threads to write every element.
///
/// Locks out the second defect the readback fix uncovered. Sizing the buffer
/// correctly is only half the request: the launch grid was inferred from the
/// same compile-time count, and a countless declaration reported zero words,
/// which rounds up to exactly one workgroup. The compiler also shrinks the
/// workgroup to 32 lanes when it believes the output is that small, so a 64
/// element dispatch ran 32 threads and returned a correctly sized buffer whose
/// second half was zeros under an `Ok`. That is the same silent wrong answer as
/// the empty readback, just harder to notice, so it is asserted the same way:
/// exact bytes, at lengths that straddle the 32 lane and 64 lane boundaries.
///
/// A single length cannot catch this. At 32 elements and below the truncated
/// launch happens to cover the whole buffer, so the bug is invisible; the
/// lengths above 32 are the ones that fail if it returns.
#[test]
fn countless_read_write_launches_enough_threads_for_every_element() {
    for len in [31u32, 32, 33, 48, 64, 65, 127, 128, 129, 256] {
        let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), len);
        let inputs = inputs_for(&program, len as usize * 4, len);
        assert_all_three_return(
            &program,
            &inputs,
            &expected_bytes(len),
            &format!("countless read_write, {len} elements"),
        );
    }
}

/// Artifact submission retains the compiler-selected grid for the resolved element count.
///
/// The materialized instance has no submission-time geometry input. A launch
/// wide enough to cover this buffer therefore comes from the admitted payload.
#[test]
fn artifact_submission_retains_the_resolved_element_count_geometry() {
    const WIDE: u32 = 4096;
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), WIDE);
    let inputs = inputs_for(&program, WIDE as usize * 4, WIDE);
    let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let registration = vyre_driver::backend_registration(vyre_driver_wgpu::WGPU_BACKEND_ID)
        .expect("WGPU artifact target must be registered");
    let production = ProductionSession::from_registration(&program, registration)
        .expect("WGPU adapter required");
    let execution = production.submit(&borrowed).expect("artifact submission");

    assert_eq!(
        execution.outputs[0].len(),
        WIDE as usize * 4,
        "the readback length and admitted launch geometry use the resolved element count"
    );
    assert_eq!(
        execution.outputs[0],
        expected_bytes(WIDE),
        "caller input bindings must not resize the compiler-selected launch"
    );
}
