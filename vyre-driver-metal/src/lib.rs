//! Native Metal backend registration boundary.
//!
//! The pure target compiler is registered on every host. Native device
//! materialization is registered only on Apple targets; `acquire()` on other
//! targets returns an actionable unsupported error.

// Metal.framework bindings (`metal::*`) are unsafe FFI at every device call.
// The override is the visible exception to the workspace `unsafe_code = "deny"`
// floor, reviewed through `xtask/unsafe-budget.txt`, and each site owes a
// SAFETY comment the `lint-unsafe-justification` gate reads.
#![allow(unsafe_code)]

use vyre_driver::{BackendError, VyreBackend};

/// Stable backend id for native Metal execution.
pub const METAL_BACKEND_ID: &str = "metal";
/// Validated target identity owned by the Metal driver.
pub const METAL_TARGET_ID: vyre_foundation::operation::TargetId =
    vyre_foundation::operation::TargetId::expect_valid(METAL_BACKEND_ID);

/// Metal external resource import/export and timeline synchronization.
mod external_resource;
pub use external_resource::{
    MetalExternalImportPolicy, MetalExternalMemoryDescriptor, MetalExternalMemoryHandle,
    MetalExternalResourceImporter,
};
mod materializer;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod runtime;
mod target_compiler;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use runtime::MetalBackend;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use runtime::{
    metal_resident_scan_resource_table, MetalResidentScanResourceEntry,
    MetalResidentScanResourceError, MetalResidentScanResourceLifetime,
    MetalResidentScanResourceTableEvidence, METAL_RESIDENT_SCAN_RESOURCE_TABLE_SCHEMA_VERSION,
};

/// Acquire the native Metal backend.
///
/// # Errors
///
/// Returns [`BackendError`] when the current target cannot expose
/// Metal.framework or when no Metal device is available.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn acquire() -> Result<Box<dyn VyreBackend>, BackendError> {
    MetalBackend::acquire().map(|backend| Box::new(backend) as Box<dyn VyreBackend>)
}

/// Acquire the native Metal backend on non-Apple targets.
///
/// # Errors
///
/// Always returns [`BackendError::UnsupportedFeature`] because this build
/// target cannot link Metal.framework.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn acquire() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::UnsupportedFeature {
        name: "Apple Metal.framework native runtime".to_string(),
        backend: METAL_BACKEND_ID.to_string(),
    })
}

/// Return the backend identifier submitted into the registry on this target.
#[must_use]
pub fn registered_backend_id() -> Option<&'static str> {
    Some(METAL_BACKEND_ID)
}

vyre_driver::register_backend! {
    id: METAL_BACKEND_ID,
    target_id: METAL_TARGET_ID,
    payload_format: Some(target_compiler::METAL_TARGET_FORMAT),
    reference_oracle: false,
    factory: acquire,
    target_compiler: Some(target_compiler::target_compiler_factory),
    materializer: Some(materializer::materializer_factory),
    rank: 25,
}

// One module of `tests/internal` names a private item of `runtime`, so the
// library compiles that file itself. The rest of the tree reaches the backend
// through the public trait and is compiled by the `all_tests` harness, which is
// what keeps every test case in one target.
#[cfg(all(test, any(target_os = "macos", target_os = "ios")))]
#[path = "../tests/internal/resident_table_metrics.rs"]
mod resident_table_metrics;
