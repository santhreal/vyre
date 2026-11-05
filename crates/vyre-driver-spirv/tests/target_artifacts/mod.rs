//! SPIR-V's arguments to the shared driver contracts.
//!
//! Three of this crate's test targets assert against the registered SPIR-V
//! target compiler, and each had restated the same expectation literal. The
//! payload format identity, its version and the neutral entry point are one
//! decision about this backend, so they are stated here once and every target
//! reads them from here. The fixture artifacts the negative cases need come
//! from [`TargetExpectation`], which is the single owner of them.

use vyre_foundation::ir::BufferAccess;
use vyre_test_support::target_compiler_contract::TargetExpectation;

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
