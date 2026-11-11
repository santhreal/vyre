//! A device fact the backend could not report is not a usable value.
//!
//! `DeviceProfile` spells "unknown" as `0` for every measured field, and a
//! consumer that saturates that to `1` gets a number that looks legal and is
//! wrong by orders of magnitude. The WGPU adapter probe reports no compute-unit
//! count, so a grid-stride reduction that read `compute_units.max(1)` launched
//! one workgroup over a million elements: the right answer at 0.08x of a
//! multithreaded CPU baseline, which no correctness test can see.
//!
//! These tests pin the accessor that answers the question instead, over both
//! halves of its domain.

#![forbid(unsafe_code)]

use vyre_driver::DeviceProfile;

#[test]
fn an_unreported_compute_unit_count_is_not_one_workgroup() {
    let unprobed = DeviceProfile::conservative("unprobed");
    assert_eq!(
        unprobed.compute_units, 0,
        "Fix: the conservative profile must report the count as unknown, or this test proves nothing"
    );
    assert!(
        unprobed.grid_stride_workgroups() > 1,
        "Fix: an unreported compute-unit count must not become a one-workgroup launch"
    );
    assert_eq!(
        unprobed.grid_stride_workgroups(),
        u32::MAX,
        "Fix: unknown means no device cap, so the shape decides the grid"
    );
}

#[test]
fn a_reported_compute_unit_count_is_the_request() {
    for units in [1u32, 2, 24, 128, 170, u32::MAX] {
        let mut profile = DeviceProfile::conservative("probed");
        profile.compute_units = units;
        assert_eq!(
            profile.grid_stride_workgroups(),
            units,
            "Fix: a reported compute-unit count is the request, unmodified"
        );
    }
}
