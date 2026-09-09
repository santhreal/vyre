//! Version-pinned expert-written native kernel baselines catalog for BACKLOG row 47.
//!
//! BACKLOG row 47 requires:
//! "Version-pinned expert-written native kernels are compared under identical
//! semantics, dtype, shapes, raggedness, initial and final state, target, stream,
//! toolchain and flags, clock and power state, warmup, interleaving, repetitions,
//! cache state, and objective."
//! "The version-pinned native kernel baselines must be vendored as a pinned external
//! baseline with its version recorded, never copied into a Vyre crate as an emitted payload."

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::equality::NativeComparisonConditions;
use super::measurement::CaseMeasurementRecord;

/// A version-pinned external expert-written native baseline kernel descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VersionPinnedNativeBaseline {
    /// Unique baseline identifier.
    pub baseline_id: String,
    /// Human-readable baseline title.
    pub name: String,
    /// Originating library / vendor (e.g. "NVIDIA CUB", "NVIDIA CUTLASS", "FlashAttention").
    pub vendor_or_library: String,
    /// Pinned release version string (e.g. "2.1.0", "3.5.0", "2.5.8").
    pub pinned_version: String,
    /// Canonical upstream repository or publication URL.
    pub upstream_url: String,
    /// Pinned commit hash or release tag.
    pub commit_or_tag: String,
    /// Cryptographic SHA256 digest of the vendored source code.
    pub source_sha256: String,
    /// Compiler toolchain required (e.g. "nvcc 12.4").
    pub toolchain: String,
    /// Compilation flags passed to the native compiler.
    pub compilation_flags: Vec<String>,
    /// The 14 required comparison conditions under which this baseline is measured.
    pub equality_conditions: NativeComparisonConditions,
    /// The recorded empirical measurement of the native kernel.
    pub measurement: Option<CaseMeasurementRecord>,
}

impl VersionPinnedNativeBaseline {
    /// Whether this baseline has a recorded, complete measurement.
    #[must_use]
    pub fn has_measurement(&self) -> bool {
        self.measurement.is_some()
    }

    /// Validate the completeness and provenance of the native baseline metadata.
    pub fn validate_metadata(&self) -> Result<(), String> {
        if self.baseline_id.is_empty() {
            return Err("baseline_id must not be empty".to_string());
        }
        if self.pinned_version.is_empty() {
            return Err(format!(
                "baseline `{}` is missing pinned_version",
                self.baseline_id
            ));
        }
        if self.upstream_url.is_empty() {
            return Err(format!(
                "baseline `{}` is missing upstream_url",
                self.baseline_id
            ));
        }
        if self.source_sha256.is_empty() {
            return Err(format!(
                "baseline `{}` is missing source_sha256 digest",
                self.baseline_id
            ));
        }
        let unset = self.equality_conditions.unset_dimensions();
        if !unset.is_empty() {
            return Err(format!(
                "baseline `{}` has unset equality dimensions: {:?}",
                self.baseline_id, unset
            ));
        }
        Ok(())
    }
}

/// Catalog of all pinned external native kernel baselines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NativeBaselineCatalog {
    /// Registered baselines keyed by baseline ID.
    pub baselines: BTreeMap<String, VersionPinnedNativeBaseline>,
}

impl NativeBaselineCatalog {
    /// Create an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self {
            baselines: BTreeMap::new(),
        }
    }

    /// Register a pinned native baseline.
    pub fn register(&mut self, baseline: VersionPinnedNativeBaseline) -> Result<(), String> {
        baseline.validate_metadata()?;
        let id = baseline.baseline_id.clone();
        if self.baselines.insert(id.clone(), baseline).is_some() {
            return Err(format!("duplicate baseline ID: `{id}`"));
        }
        Ok(())
    }

    /// Retrieve a baseline by identifier.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&VersionPinnedNativeBaseline> {
        self.baselines.get(id)
    }

    /// Return an iterator over all registered baselines.
    pub fn iter(&self) -> impl Iterator<Item = &VersionPinnedNativeBaseline> {
        self.baselines.values()
    }
}
