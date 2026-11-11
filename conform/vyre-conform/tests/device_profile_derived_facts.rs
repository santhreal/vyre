//! A device profile must not report a budget and deny the capability it is.
//!
//! # Why this exists
//!
//! `DeviceProfile` carries both a figure and the flag derived from it:
//! `max_shared_memory_bytes` beside `has_shared_memory`, a subgroup width
//! beside `supports_subgroup_ops`. `DeviceProfile::from_backend` derives each
//! flag from its figure, but a driver may spell the profile as a struct literal
//! over `DeviceProfile::conservative` and set only the figures. The flags then
//! stay at the conservative default and silently disagree with them.
//!
//! The SPIR-V driver did exactly that. It reported a 48 KiB workgroup scratch
//! budget with `has_shared_memory` false, and the whole-program compiler refuses
//! a program declaring workgroup scratch against a device reporting no shared
//! memory: 39 operations were refused as `program declares workgroup-scoped
//! scratch but the device reports no shared memory` on an RTX 4090. The same
//! profile denied subgroup support on a device whose warps are 32 lanes wide,
//! and the validator refused 7 more before emission. None of the 46 refusals
//! named the profile; each read as a device that lacked the hardware.
//!
//! So this asserts the agreement over whatever backends the binary links, read
//! from the registry at run time. A driver added later is checked without this
//! file being edited, and a driver that answers the trait methods and lets
//! `from_backend` build its profile passes by construction.
//!
//! # What it does not catch
//!
//! Agreement, not truth. A driver that reports no shared memory on a device
//! that has 48 KiB is consistent and wrong, and this passes it. Catching that
//! needs the device, which is what the conformance proof does.

use vyre_conform::backend_selection::semantic_execution_backends;

#[test]
fn a_reported_scratch_budget_is_reported_as_shared_memory() {
    for (backend, profile) in live_profiles() {
        assert_eq!(
            profile.has_shared_memory,
            profile.max_shared_memory_bytes > 0,
            "backend `{backend}` reports {} bytes of workgroup scratch and has_shared_memory={}. The compiler refuses a program declaring workgroup scratch against a device reporting no shared memory, so the disagreement refuses programs this device runs, and the refusal names the device rather than the profile.",
            profile.max_shared_memory_bytes,
            profile.has_shared_memory
        );
    }
}

#[test]
fn a_reported_subgroup_width_is_reported_as_subgroup_support() {
    for (backend, profile) in live_profiles() {
        assert_eq!(
            profile.supports_subgroup_ops,
            profile.subgroup_size > 0,
            "backend `{backend}` reports subgroup_size={} and supports_subgroup_ops={}. A collective needs a width to be a collective, and a width without the operations is a promise the backend cannot keep.",
            profile.subgroup_size,
            profile.supports_subgroup_ops
        );
        assert_eq!(
            profile.has_subgroup_shuffle, profile.supports_subgroup_ops,
            "backend `{backend}` reports has_subgroup_shuffle={} and supports_subgroup_ops={}. The validator reads one and the specializer reads the other, so a program admitted by one is refused by the other.",
            profile.has_subgroup_shuffle, profile.supports_subgroup_ops
        );
    }
}

#[test]
fn a_workgroup_a_device_admits_fits_the_invocation_limit_it_reports() {
    for (backend, profile) in live_profiles() {
        let [x, y, z] = profile.max_workgroup_size;
        assert!(
            x > 0 && y > 0 && z > 0,
            "backend `{backend}` reports a zero workgroup dimension in {:?}; no dispatch shape fits it",
            profile.max_workgroup_size
        );
        assert!(
            profile.max_invocations_per_workgroup > 0,
            "backend `{backend}` admits no invocations per workgroup, so every dispatch it is offered is refused"
        );
        assert!(
            u64::from(x) <= u64::from(profile.max_invocations_per_workgroup)
                && u64::from(y) <= u64::from(profile.max_invocations_per_workgroup)
                && u64::from(z) <= u64::from(profile.max_invocations_per_workgroup),
            "backend `{backend}` reports a per-dimension maximum {:?} above its {} invocations per workgroup, so a shape it admits by dimension is refused by count",
            profile.max_workgroup_size,
            profile.max_invocations_per_workgroup
        );
    }
}

/// Profiles of every linked backend this host can acquire.
///
/// A backend the host cannot acquire reports nothing to check; the acquisition
/// refusal is covered where acquisition is the subject.
fn live_profiles() -> Vec<(&'static str, vyre_driver::DeviceProfile)> {
    let backends = semantic_execution_backends()
        .expect("the registry initializes in a binary that links concrete drivers");
    assert!(
        !backends.is_empty(),
        "no semantic-execution backend is linked; this target requires `device-tests`, which pulls in `gpu`"
    );
    backends
        .into_iter()
        .filter_map(|backend| {
            backend
                .acquire()
                .ok()
                .map(|handle| (backend.id, handle.device_profile()))
        })
        .collect()
}
