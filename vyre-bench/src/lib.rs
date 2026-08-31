#![allow(unsafe_code)]
//! Benchmark library types and backend registration support.
//!
//! Backend registrations reach this binary through `vyre-registry-link`, which
//! owns every driver link anchor and asserts that each linked driver reached the
//! registry. Reading the registry through that owner is what keeps the driver
//! object files in the binary, so nothing here names a driver crate for effect.

/// API definitions for external benchmark drivers.
#[allow(missing_docs)]
pub mod api;
/// Reference test cases and standard regression suites.
#[allow(missing_docs)]
pub mod cases;
/// Target device capability and telemetry probes.
#[allow(missing_docs)]
pub mod probes;
/// The benchmark registry and metadata catalog.
#[allow(missing_docs)]
pub mod registry;
/// The device class a release measurement may have been taken on.
pub mod release_floor;
/// Parity release matrix verification logic.
#[allow(missing_docs)]
pub mod release_matrix;
/// HTML/Markdown report formatting and artifact writing.
#[allow(missing_docs)]
pub mod report;
/// Context and thread coordination for the test runner.
#[allow(missing_docs)]
pub mod runner;
