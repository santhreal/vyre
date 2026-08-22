//! The launch grid resolved from a runtime-sized buffer, and the explicit
//! override that must survive it.

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

/// A caller who pins `grid_override` keeps exactly the launch they asked for.
///
/// Locks out the grid re-inference over-applying. Re-deriving the grid from the
/// resolved element count must happen only when the backend inferred that grid
/// in the first place. If it starts overwriting an explicit `grid_override`, a
/// caller who deliberately launched a partial or oversized grid silently gets a
/// different dispatch than the one they wrote.
#[test]
fn an_explicit_grid_override_is_not_replaced_by_the_resolved_element_count() {
    // 4096 elements against a single workgroup. No workgroup size this backend
    // picks can cover that in one launch, so the assertions below do not depend
    // on which lane count the compiler chose.
    const WIDE: u32 = 4096;
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), WIDE);
    let inputs = inputs_for(&program, WIDE as usize * 4, WIDE);
    let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let registration = vyre_driver::backend_registration(vyre_driver_wgpu::WGPU_BACKEND_ID)
        .expect("WGPU artifact target must be registered");
    let production =
        ProductionSession::compile_with_representative_inputs(&program, &borrowed, registration)
            .expect("WGPU adapter required");
    let pinned = production
        .submit_with_invocation_grid(&borrowed, [1, 1, 1])
        .expect("a pinned grid must still submit");

    assert_eq!(
        pinned[0].len(),
        WIDE as usize * 4,
        "the readback length is set by the resolved element count, not by the pinned grid"
    );
    // Element 0 is inside the one workgroup that was launched: 0 ^ 0x0F == 15.
    assert_eq!(
        pinned[0][..4],
        [15, 0, 0, 0],
        "the workgroup that was launched must still write its own elements"
    );
    // The last element is far outside it and must keep the zero it was seeded
    // with. If the override were silently widened to cover all 4096 elements,
    // this would hold 4095 ^ 0x0F.
    assert_eq!(
        pinned[0][WIDE as usize * 4 - 4..],
        [0, 0, 0, 0],
        "a one-workgroup override must NOT have been widened into a full launch"
    );
    assert_ne!(
        pinned[0],
        expected_bytes(WIDE),
        "a pinned partial grid must not produce the full-launch result"
    );
}
