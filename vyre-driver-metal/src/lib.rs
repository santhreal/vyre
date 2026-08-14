#![allow(unsafe_code)]
//! Native Metal backend registration boundary.
//!
//! The pure target compiler is registered on every host. Native device
//! materialization is registered only on Apple targets; `acquire()` on other
//! targets returns an actionable unsupported error.

use vyre_driver::backend::{BackendError, VyreBackend};

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

#[cfg(any(target_os = "macos", target_os = "ios"))]
inventory::submit! {
    vyre_driver::backend::BackendRegistration {
        id: METAL_BACKEND_ID,
        target_id: METAL_TARGET_ID,
        payload_format: Some(target_compiler::METAL_TARGET_FORMAT),
        reference_oracle: false,
        factory: acquire,
        supported_ops: vyre_driver::backend::core_supported_ops,
        semantic_operations: vyre_driver::backend::dialect_only_supported_ops,
        target_compiler: Some(target_compiler::target_compiler_factory),
        materializer: Some(materializer::materializer_factory),
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
inventory::submit! {
    vyre_driver::backend::BackendCapability {
        id: METAL_BACKEND_ID,
        dispatches: true,
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
inventory::submit! {
    vyre_driver::backend::BackendPrecedence {
        id: METAL_BACKEND_ID,
        rank: 25,
    }
}

#[cfg(test)]
mod tests;
