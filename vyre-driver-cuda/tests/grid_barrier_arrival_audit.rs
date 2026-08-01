//! The grid-barrier arrival audit, which turns a SILENT wrong-answer barrier
//! into a loud dispatch error.
//!
//! One concern: a cooperative grid-sync launch whose module-scope
//! `_vyre_grid_barrier` counter was not zeroed first releases every barrier
//! immediately, returns success, and reports no error anywhere. The only symptom
//! is wrong cross-block data. That shape shipped on the resident dispatch paths
//! and cost a day of chasing a flake whose rate was about 50 percent, so the
//! audit exists to make the next occurrence fail at the dispatch instead of at
//! someone's token ids.
//!
//! The audit reads the counter after the launch stream is synchronized and
//! refuses when it exceeds `barriers * grid_blocks`, the most ONE launch can
//! contribute. It is an UPPER bound rather than exact equality on purpose: a
//! grid-uniform early exit legitimately skips later barriers and leaves the
//! counter below the bound.
//!
//! These tests assert real counter values and real dispatch outcomes. A
//! `!is_empty()` style check would pass for an audit that never ran.

mod common;

use common::{
    cross_block_grid_sync_expected, cross_block_grid_sync_inputs, cross_block_grid_sync_program,
    CROSS_BLOCK_GRID_SYNC_WORKGROUP,
};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;

/// Lanes for the cross-block fixture: four blocks, so a barrier that no-ops is
/// observable as a wrong value rather than as a coincidence.
const LANES: u32 = 4 * CROSS_BLOCK_GRID_SYNC_WORKGROUP;

fn backend() -> CudaBackend {
    CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.")
}

fn bytes_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn cooperative() -> DispatchConfig {
    let mut config = DispatchConfig::default();
    config.cooperative = true;
    config
}

/// The emitter marker the audit counts must still be in the emitted PTX.
///
/// The static barrier count lives nowhere else by the time a launch site needs
/// it, so the audit counts a comment marker the PTX emitter writes per barrier.
/// If that marker is renamed, the audit's ceiling silently becomes unbounded (a
/// zero count is refused, so it fails loudly, but only if this coupling is
/// noticed). This test names the coupling so a rename breaks HERE, next to the
/// explanation, rather than inside a dispatch six months later.
#[test]
fn emitted_grid_sync_ptx_still_carries_the_marker_the_arrival_audit_counts() {
    let program = cross_block_grid_sync_program(LANES);
    let ptx = vyre_driver_cuda::codegen::program_to_ptx_for_sm(&program, &cooperative(), 90)
        .expect("Fix: the cross-block grid-sync fixture must emit PTX.");

    let markers = ptx.matches("grid.sync barrier #").count();
    assert_eq!(
        markers, 1,
        "Fix: the one-barrier fixture must emit exactly one `grid.sync barrier #` marker, \
         because the arrival audit derives its ceiling by counting them. Got {markers}.\nPTX:\n{ptx}"
    );
    assert!(
        ptx.contains(".global .align 4 .u32 _vyre_grid_barrier[1];"),
        "Fix: the module-scope arrival counter must be declared, or there is nothing to audit."
    );
}

/// Distinct barriers must each contribute their own marker, so the ceiling scales
/// with the barrier count instead of pinning to one.
///
/// A ceiling computed as if there were always ONE barrier would refuse every
/// legitimate multi-barrier launch (counter reaches `b * gridSize` > `1 *
/// gridSize`), which is a false positive that would force someone to delete the
/// audit rather than fix a bug.
#[test]
fn arrival_ceiling_marker_count_scales_with_the_number_of_barriers() {
    let one = vyre_driver_cuda::codegen::program_to_ptx_for_sm(
        &cross_block_grid_sync_program(LANES),
        &cooperative(),
        90,
    )
    .expect("single-barrier fixture must emit PTX");
    let markers_one = one.matches("grid.sync barrier #").count();

    // The emitter numbers barriers from zero, so a second barrier must appear as
    // `#1` with its own release target rather than reusing `#0`.
    assert_eq!(markers_one, 1, "the fixture carries exactly one barrier");
    assert!(
        one.contains("grid.sync barrier #0"),
        "Fix: the first barrier must be numbered 0; the audit's ceiling and the barrier's \
         release target both derive from that index.\nPTX:\n{one}"
    );
    assert!(
        !one.contains("grid.sync barrier #1"),
        "Fix: a one-barrier program must not emit a second barrier index."
    );
}

/// A correct cooperative launch must PASS the audit and return the right data.
///
/// This is the false-positive guard. An audit that refused a healthy launch
/// would be worse than no audit, because the fix would be to remove it.
#[test]
fn correct_cooperative_grid_sync_launch_passes_the_arrival_audit() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let program = cross_block_grid_sync_program(LANES);
    let inputs = cross_block_grid_sync_inputs(LANES);

    let outputs = backend
        .dispatch(&program, &inputs, &cooperative())
        .expect("Fix: a healthy cooperative grid-sync launch must pass the arrival audit.");

    let actual = bytes_u32(
        outputs
            .last()
            .expect("the fixture declares an output buffer"),
    );
    assert_eq!(
        actual,
        cross_block_grid_sync_expected(LANES),
        "Fix: the audited launch must still produce the grid-synchronized values."
    );
}

/// Repeated launches of ONE loaded module must each pass the audit.
///
/// This is the exact shape of the bug: launch 2 onward is where a missing reset
/// shows up, because launch 1 always starts from a zero counter. Eight launches
/// means a missing reset would push the counter to eight times the ceiling, and
/// the audit refuses at the FIRST launch that exceeds it, so this fails on launch
/// 2 rather than silently returning wrong values on all eight.
#[test]
fn eight_sequential_launches_of_one_module_each_stay_within_the_arrival_ceiling() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let program = cross_block_grid_sync_program(LANES);
    let expected = cross_block_grid_sync_expected(LANES);

    for launch in 1..=8_u32 {
        // Re-upload every launch: `scratch` is read_write and the fixture's
        // expected value depends on its seeded state.
        let inputs = cross_block_grid_sync_inputs(LANES);
        let outputs = backend
            .dispatch(&program, &inputs, &cooperative())
            .unwrap_or_else(|error| {
                panic!(
                "Fix: launch {launch} of the same module must pass the arrival audit; a failure \
                 here means the per-launch counter reset regressed: {error}"
            )
            });
        let actual = bytes_u32(
            outputs
                .last()
                .expect("the fixture declares an output buffer"),
        );
        assert_eq!(
            actual, expected,
            "Fix: launch {launch} produced wrong cross-block values, so its barrier released \
             before every block arrived."
        );
    }
}

/// The audit must cover the resident dispatch path, not only the borrowed-host
/// path.
///
/// The resident path is where the missing reset actually shipped, and it is the
/// path a real consumer takes once any binding is device resident. An audit wired
/// into only the borrowed path would have caught nothing.
#[test]
fn resident_cooperative_grid_sync_launches_are_audited_too() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let program = cross_block_grid_sync_program(LANES);
    let expected = cross_block_grid_sync_expected(LANES);

    for launch in 1..=4_u32 {
        let inputs = cross_block_grid_sync_inputs(LANES);
        let borrowed: Vec<&[u8]> = inputs.iter().map(std::vec::Vec::as_slice).collect();
        let outputs = backend
            .dispatch_borrowed(&program, &borrowed, &cooperative())
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: resident/borrowed launch {launch} must pass the arrival audit: {error}"
                )
            });
        let actual = bytes_u32(
            outputs
                .last()
                .expect("the fixture declares an output buffer"),
        );
        assert_eq!(
            actual, expected,
            "Fix: resident launch {launch} lost grid synchronization."
        );
    }
}

/// A non-grid-sync program must not pay the audit, and must not be refused by it.
///
/// The audit reads device memory and needs a synchronized stream. Charging that
/// to every ordinary launch would be a real cost on the hot path, and an inert
/// lease is what keeps it off. A regression that made the lease non-inert would
/// show up as this test failing or as the whole suite slowing down.
#[test]
fn ordinary_non_cooperative_dispatch_is_unaffected_by_the_arrival_audit() {
    let backend = backend();
    let program = cross_block_grid_sync_program(LANES);
    let inputs = cross_block_grid_sync_inputs(LANES);

    // Same program, but the caller does not request cooperative launch. The
    // backend forces cooperative for a grid-sync program anyway, which is the
    // contract that keeps a grid-sync kernel off the non-cooperative path; assert
    // that rather than assuming it.
    let outputs = backend
        .dispatch(&program, &inputs, &DispatchConfig::default())
        .expect(
            "Fix: a grid-sync program must still dispatch when the caller omits `cooperative`.",
        );
    assert_eq!(
        bytes_u32(
            outputs
                .last()
                .expect("the fixture declares an output buffer")
        ),
        cross_block_grid_sync_expected(LANES),
        "Fix: a grid-sync program dispatched without an explicit cooperative flag must be forced \
         cooperative and still grid-synchronize, not silently run workgroup-scoped."
    );
}
