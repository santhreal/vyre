//! A backend never reports a capability its own target compiler refuses.
//!
//! The megakernel envelope validates every target payload against the profile
//! the backend's target dialect registers, while a composition sizes its
//! geometry from the device profile the backend reports. When those two
//! disagree the composition builds a payload that cannot be constructed, and
//! the failure surfaces as `MKC017_MALFORMED_TARGET_PAYLOAD` far from the fact
//! that caused it. That is exactly how the wgpu reduction crossover benchmark
//! failed: the adapter advertised 1024 invocations per workgroup and the WGSL
//! dialect admits the WebGPU spec baseline of 256.
//!
//! The backend set is read through `vyre-registry-link`, the one crate that
//! makes every driver crate a real link anchor, so a newly linked backend is
//! covered here the moment it registers, with no list to update.
//!
//! Gated on `device-tests`: acquiring a backend acquires its device.
#![cfg(feature = "device-tests")]

use vyre_registry_link::backend::live_backend_registry;

#[test]
fn every_registered_backend_reports_geometry_its_target_compiler_admits() {
    let mut checked = Vec::new();

    let registrations =
        live_backend_registry().expect("Fix: the backend registry must build to be audited");

    for registration in registrations {
        let Ok(compiler) = registration.target_compiler() else {
            // A backend with no native target compiler has no profile to
            // disagree with; raw dispatch never builds a target payload.
            continue;
        };
        let Ok(backend) = registration.acquire() else {
            // No device of this kind on this host. The runner that owns one
            // covers it; a host without it has nothing to contradict.
            continue;
        };

        let target = compiler.profile();
        let device = backend.device_profile();

        for (axis, (extent, limit)) in device
            .max_workgroup_size
            .iter()
            .zip(target.max_workgroup_size())
            .enumerate()
        {
            assert!(
                *extent <= limit,
                "Fix: backend `{}` reports workgroup axis {axis} as {extent}, which its own target profile limit {limit} rejects at payload construction",
                registration.id
            );
        }

        assert!(
            device.max_invocations_per_workgroup <= target.max_invocations_per_workgroup(),
            "Fix: backend `{}` promises {} invocations per workgroup, more than the {} its own target profile admits",
            registration.id,
            device.max_invocations_per_workgroup,
            target.max_invocations_per_workgroup()
        );

        checked.push(registration.id);
    }

    assert!(
        !checked.is_empty(),
        "Fix: no registered backend acquired a device, so this test proved nothing; run it on a host with the hardware the linked backends need"
    );
    println!("audited backend profiles: {}", checked.join(", "));
}
