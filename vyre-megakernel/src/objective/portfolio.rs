//! How many artifacts one request may retain, and what they must cover.
//!
//! Compiling every shape to its own optimum produces a variant set that is
//! individually fast and collectively unaffordable: every variant costs compile
//! time, artifact bytes, load time and cache residency. A portfolio policy
//! states the ceiling and what the retained set must still cover, so the
//! selector can refuse a variant that buys less than it costs.

use serde::{Deserialize, Serialize};

/// What a retained artifact set must cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CoveragePolicy {
    /// One artifact serves every stated workload class.
    Single,
    /// Every stated workload class is served by some retained artifact.
    EveryWorkloadClass,
}

impl CoveragePolicy {
    /// Every declared coverage policy.
    pub const ALL: &'static [Self] = &[Self::Single, Self::EveryWorkloadClass];

    /// Stable identifier used in diagnostics and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::EveryWorkloadClass => "every_workload_class",
        }
    }

    /// Smallest retained artifact count this policy can be satisfied by over
    /// `classes` stated workload classes.
    #[must_use]
    pub const fn minimum_variants(self, classes: usize) -> usize {
        match self {
            Self::Single => 1,
            Self::EveryWorkloadClass => classes,
        }
    }
}

/// Ceiling and coverage requirement for one retained artifact set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortfolioPolicy {
    coverage: CoveragePolicy,
    max_variants: u32,
    max_aggregate_bytes: Option<u64>,
}

impl PortfolioPolicy {
    /// One artifact serving every class, with no aggregate byte ceiling.
    #[must_use]
    pub const fn single() -> Self {
        Self {
            coverage: CoveragePolicy::Single,
            max_variants: 1,
            max_aggregate_bytes: None,
        }
    }

    /// Construct an explicit policy.
    #[must_use]
    pub const fn new(coverage: CoveragePolicy, max_variants: u32) -> Self {
        Self {
            coverage,
            max_variants,
            max_aggregate_bytes: None,
        }
    }

    /// Bound the serialized bytes of the whole retained set.
    #[must_use]
    pub const fn with_max_aggregate_bytes(mut self, bytes: u64) -> Self {
        self.max_aggregate_bytes = Some(bytes);
        self
    }

    /// Coverage requirement.
    #[must_use]
    pub const fn coverage(&self) -> CoveragePolicy {
        self.coverage
    }

    /// Artifact ceiling.
    #[must_use]
    pub const fn max_variants(&self) -> u32 {
        self.max_variants
    }

    /// Aggregate serialized byte ceiling, when one is stated.
    #[must_use]
    pub const fn max_aggregate_bytes(&self) -> Option<u64> {
        self.max_aggregate_bytes
    }

    /// Whether `variants` artifacts covering `classes` classes satisfy this
    /// policy.
    #[must_use]
    pub const fn admits(&self, variants: u32, classes: usize) -> bool {
        variants as usize >= self.coverage.minimum_variants(classes)
            && variants <= self.max_variants
    }
}

impl Default for PortfolioPolicy {
    fn default() -> Self {
        Self::single()
    }
}
