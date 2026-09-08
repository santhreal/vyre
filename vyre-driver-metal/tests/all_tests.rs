//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/measurement_registry/mod.rs`.
#[path = "measurement_registry/mod.rs"]
pub mod measurement_registry;

/// Integration tests from `tests/apple_math_comparators.rs`.
#[path = "apple_math_comparators.rs"]
pub mod apple_math_comparators;

/// Native backend coverage from `tests/internal/mod.rs`.
///
/// The module self-gates on `device-tests`, so it contributes nothing to a
/// default build and its non-Apple arm still pins what such a build refuses.
#[path = "internal/mod.rs"]
pub mod internal;

/// Integration tests from `tests/metal_hazard_certificates.rs`.
#[path = "metal_hazard_certificates.rs"]
pub mod metal_hazard_certificates;

/// Integration tests from `tests/metal_icb_dispatch_replay.rs`.
#[path = "metal_icb_dispatch_replay.rs"]
pub mod metal_icb_dispatch_replay;

/// Integration tests from `tests/metal_simd_scan_plan_registry.rs`.
#[path = "metal_simd_scan_plan_registry.rs"]
pub mod metal_simd_scan_plan_registry;

/// Integration tests from `tests/resident_async.rs`.
#[path = "resident_async.rs"]
pub mod resident_async;

/// Integration tests from `tests/target_compiler.rs`.
#[cfg(feature = "device-tests")]
#[path = "target_compiler.rs"]
pub mod target_compiler;
