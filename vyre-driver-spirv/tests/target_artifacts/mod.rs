//! SPIR-V's arguments to the shared driver contracts.
//!
//! Three of this crate's test targets assert against the registered SPIR-V
//! target compiler, and each had restated the same expectation literal. The
//! payload format identity, its version and the neutral entry point are one
//! decision about this backend, so they are stated here once and the shared
//! contract is reached through this module.

#![allow(dead_code)]

use vyre_foundation::ir::BufferAccess;

#[path = "../../../tests/support/target_compiler_contract.rs"]
pub(crate) mod target_compiler_contract;

use target_compiler_contract::{single_lane_artifact, TargetExpectation};

/// What this backend declares about the payload its registered compiler produces.
///
/// The id comes from [`vyre_driver_spirv::registered_backend_id`] and not from
/// the `const`, because calling it is what keeps this crate's object file, and
/// its registration, in a linked test binary. A `const` inlines at the use
/// site and links nothing, which left the registry lookup reporting an
/// unlinked backend on the Mach-O leg of the matrix while the ELF legs passed.
pub(crate) fn spirv() -> TargetExpectation<'static> {
    TargetExpectation {
        backend_id: vyre_driver_spirv::registered_backend_id()
            .expect("Fix: this build must compile the SPIR-V registration."),
        format_identity: "spv",
        format_version: 1,
        entry_point: "main",
        output_access: BufferAccess::ReadWrite,
    }
}

/// The artifact this backend's payload is authentic for.
pub(crate) fn artifact() -> vyre_megakernel::Artifact {
    single_lane_artifact(BufferAccess::ReadWrite, 0)
}

/// An artifact this backend's payload must never be authentic for.
pub(crate) fn foreign_artifact() -> vyre_megakernel::Artifact {
    single_lane_artifact(BufferAccess::ReadWrite, 1)
}
