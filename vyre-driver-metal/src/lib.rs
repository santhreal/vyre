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

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod materializer;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod runtime;
#[cfg(any(target_os = "macos", target_os = "ios"))]
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

/// Backend id this crate submits into the backend registry on this target, or
/// `None` on a target where the native registration is compiled out.
///
/// WHY: the registration below lives in this crate's object file, and a linker
/// keeps that object only when a symbol inside it is referenced. Naming the
/// crate with `use vyre_driver_metal as _;` references nothing, and reading
/// [`METAL_BACKEND_ID`] is a `const` that inlines at the use site, so neither
/// keeps the registration. Calling this function does, which is why the backend
/// registry owner calls it instead of importing the crate for effect. The
/// `Option` reports the target truth, so a floor over the linked set does not
/// demand a Metal registration from a build that never compiled one.
#[must_use]
pub fn registered_backend_id() -> Option<&'static str> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some(METAL_BACKEND_ID)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        None
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
inventory::submit! {
    vyre_driver::BackendRegistration {
        id: METAL_BACKEND_ID,
        target_id: METAL_TARGET_ID,
        payload_format: Some(target_compiler::METAL_TARGET_FORMAT),
        reference_oracle: false,
        factory: acquire,
        supported_ops: vyre_driver::core_supported_ops,
        semantic_operations: vyre_driver::dialect_only_supported_ops,
        target_compiler: Some(target_compiler::target_compiler_factory),
        materializer: Some(materializer::materializer_factory),
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
inventory::submit! {
    vyre_driver::BackendCapability {
        id: METAL_BACKEND_ID,
        dispatches: true,
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
inventory::submit! {
    vyre_driver::BackendPrecedence {
        id: METAL_BACKEND_ID,
        rank: 25,
    }
}

#[cfg(test)]
mod tests;
