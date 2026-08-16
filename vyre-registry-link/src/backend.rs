//! The backend registry, with every driver crate that submits into it linked.
//!
//! The floor is per source: a driver crate that is linked but reached no
//! registration had its object file dropped, and every reader would report a
//! shorter backend set than the build declares. A driver whose registration is
//! compiled out on this target reports that itself, so the floor demands its
//! absence rather than a registration it never built.

use std::sync::LazyLock;

use vyre_driver::BackendError;
use vyre_driver::{
    registered_backends, registered_backends_by_precedence_slice, BackendRegistration,
};

/// Every driver crate this owner knows how to link, independent of features.
///
/// The cargo features select which of these a build links, and
/// [`linked_backend_sources`] reports that subset. This is the list a new driver
/// crate joins; the rules read the tree to prove it did.
pub const DECLARED_SOURCES: &[&str] = &[
    "vyre-driver-cuda",
    "vyre-driver-metal",
    "vyre-driver-reference",
    "vyre-driver-spirv",
    "vyre-driver-wgpu",
];

/// One driver crate linked into this build, with what it registers here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSource {
    /// Workspace member that submits the registration.
    pub crate_name: &'static str,
    /// Backend id the crate owns, whether or not this target registers it.
    pub backend_id: &'static str,
    /// Backend id the crate submits on this build target, read from the crate
    /// itself. `None` means the registration is compiled out here, so the floor
    /// requires its absence instead of its presence.
    pub registered_here: Option<&'static str>,
}

/// Referencing each crate's `registered_backend_id` is what links it. A `const`
/// backend id would inline at the use site and link nothing.
static SOURCES: LazyLock<Vec<BackendSource>> = LazyLock::new(|| {
    vec![
        #[cfg(feature = "cuda")]
        BackendSource {
            crate_name: "vyre-driver-cuda",
            backend_id: vyre_driver_cuda::CUDA_BACKEND_ID,
            registered_here: vyre_driver_cuda::registered_backend_id(),
        },
        #[cfg(feature = "metal")]
        BackendSource {
            crate_name: "vyre-driver-metal",
            backend_id: vyre_driver_metal::METAL_BACKEND_ID,
            registered_here: vyre_driver_metal::registered_backend_id(),
        },
        #[cfg(feature = "reference")]
        BackendSource {
            crate_name: "vyre-driver-reference",
            backend_id: vyre_driver_reference::CPU_REF_BACKEND_ID,
            registered_here: vyre_driver_reference::registered_backend_id(),
        },
        #[cfg(feature = "spirv")]
        BackendSource {
            crate_name: "vyre-driver-spirv",
            backend_id: vyre_driver_spirv::SPIRV_BACKEND_ID,
            registered_here: vyre_driver_spirv::registered_backend_id(),
        },
        #[cfg(feature = "wgpu")]
        BackendSource {
            crate_name: "vyre-driver-wgpu",
            backend_id: vyre_driver_wgpu::WGPU_BACKEND_ID,
            registered_here: vyre_driver_wgpu::registered_backend_id(),
        },
    ]
});

/// Every driver crate linked into this build, with what it registers here.
#[must_use]
pub fn linked_backend_sources() -> &'static [BackendSource] {
    &SOURCES
}

/// The linked driver crates by name, for evidence that has to state which
/// drivers a run judged.
#[must_use]
pub fn linked_backend_source_names() -> Vec<&'static str> {
    linked_backend_sources()
        .iter()
        .map(|source| source.crate_name)
        .collect()
}

/// The frozen backend registry, with every linked driver crate referenced.
///
/// # Errors
/// Returns the registry startup error when two providers conflict.
///
/// # Panics
/// Panics when this build links no driver crate at all, or when a linked driver
/// crate reached no registration, which means its object file was dropped at link
/// time and every rule reading the registry would judge a partial backend set.
pub fn live_backend_registry() -> Result<&'static [BackendRegistration], BackendError> {
    let registrations = registered_backends()?;
    assert_linked_sources_reached_registry(registrations);
    Ok(registrations)
}

/// The frozen backend registry in precedence order, with every linked driver
/// crate referenced.
///
/// # Errors
/// Returns the registry startup error when two providers conflict.
///
/// # Panics
/// Panics under the same conditions as [`live_backend_registry`].
pub fn live_backend_registry_by_precedence(
) -> Result<&'static [&'static BackendRegistration], BackendError> {
    let by_precedence = registered_backends_by_precedence_slice()?;
    assert_linked_sources_reached_registry(registered_backends()?);
    Ok(by_precedence)
}

fn assert_linked_sources_reached_registry(registrations: &[BackendRegistration]) {
    assert!(
        !linked_backend_sources().is_empty(),
        "Fix: this binary links no backend driver crate, so the backend registry it reads is empty by construction. Enable the `vyre-registry-link` features naming the drivers this consumer depends on"
    );
    for source in linked_backend_sources() {
        let present = registrations
            .iter()
            .any(|registration| registration.id == source.backend_id);
        match source.registered_here {
            Some(id) => {
                assert_eq!(
                    id, source.backend_id,
                    "Fix: `{}` reports backend id `{id}` while this owner records `{}`; make the two agree",
                    source.crate_name, source.backend_id
                );
                assert!(
                    present,
                    "Fix: `{}` is linked but backend `{}` never reached the registry, so this run is judging a partial backend set. Read the registry through `vyre_registry_link::backend`, which calls that crate, instead of naming it with a discarding import",
                    source.crate_name, source.backend_id
                );
            }
            None => assert!(
                !present,
                "Fix: `{}` compiles out its registration on this target, yet backend `{}` is in the registry; a second crate registers that id",
                source.crate_name, source.backend_id
            ),
        }
    }
}
