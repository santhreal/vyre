//! Target-neutral launch geometry and requirement definitions.
//!
//! Substrate-neutral operations declare the execution constraints their
//! algorithm requires through [`GeometryRequirements`]. Backends lower those
//! requirements against authenticated target hardware profiles into a
//! concrete [`LaunchGeometry`] using the [`GeometryStrategy`] trait.
//!
//! This module contains no device names, no instruction names, and no concrete
//! device limits.

/// Constraint on cooperative execution width within a single parallel region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CooperativeWidth {
    /// The algorithm is correct at any execution width supported by the target.
    Agnostic,
    /// The algorithm requires at least `min_width` cooperative invocations.
    AtLeast(u32),
    /// The algorithm is written around an exact cooperative width.
    Exactly(u32),
}

impl Default for CooperativeWidth {
    #[inline]
    fn default() -> Self {
        Self::Agnostic
    }
}

/// Policy governing how many data elements an invocation processes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementPolicy {
    /// Exactly one element per invocation.
    Scalar,
    /// Invocations process multiples of `factor` elements.
    Multiple(u32),
    /// The backend may choose any elements-per-invocation scaling.
    Any,
}

impl Default for ElementPolicy {
    #[inline]
    fn default() -> Self {
        Self::Any
    }
}

/// Uniformity guarantees required across invocations in a subgroup or workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Uniformity {
    /// No cross-invocation uniformity constraint.
    None,
    /// Values must remain uniform across every invocation in a subgroup.
    SubgroupUniform,
    /// Values must remain uniform across every invocation in a workgroup.
    WorkgroupUniform,
}

impl Default for Uniformity {
    #[inline]
    fn default() -> Self {
        Self::None
    }
}

/// Substrate-neutral execution geometry requirements declared by an operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GeometryRequirements {
    /// Minimum or exact cooperative execution width.
    pub cooperative_width: CooperativeWidth,
    /// Minimum workgroup-shared memory in bytes needed by the algorithm.
    pub min_shared_bytes: u32,
    /// Divisibility or element processing policy per invocation.
    pub per_invocation_elements: ElementPolicy,
    /// Uniformity requirements for cross-invocation divergence prevention.
    pub subgroup_uniformity: Uniformity,
}

impl GeometryRequirements {
    /// Unconstrained geometry requirements.
    #[must_use]
    pub const fn agnostic() -> Self {
        Self {
            cooperative_width: CooperativeWidth::Agnostic,
            min_shared_bytes: 0,
            per_invocation_elements: ElementPolicy::Any,
            subgroup_uniformity: Uniformity::None,
        }
    }

    /// Geometry requirements with a specific cooperative width constraint.
    #[must_use]
    pub const fn cooperative(width: CooperativeWidth) -> Self {
        Self {
            cooperative_width: width,
            min_shared_bytes: 0,
            per_invocation_elements: ElementPolicy::Any,
            subgroup_uniformity: Uniformity::None,
        }
    }

    /// Attach minimum shared memory bytes required by the algorithm.
    #[must_use]
    pub const fn with_min_shared_bytes(mut self, bytes: u32) -> Self {
        self.min_shared_bytes = bytes;
        self
    }

    /// Attach an elements-per-invocation processing policy.
    #[must_use]
    pub const fn with_element_policy(mut self, policy: ElementPolicy) -> Self {
        self.per_invocation_elements = policy;
        self
    }

    /// Attach cross-invocation uniformity constraints.
    #[must_use]
    pub const fn with_subgroup_uniformity(mut self, uniformity: Uniformity) -> Self {
        self.subgroup_uniformity = uniformity;
        self
    }
}

/// Concrete execution launch geometry selected by a backend lowering strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchGeometry {
    /// Workgroup dimensions: `[x, y, z]`.
    pub workgroup: [u32; 3],
    /// Grid dimensions (workgroup counts): `[x, y, z]`.
    pub grid: [u32; 3],
    /// Elements processed per invocation.
    pub elements_per_invocation: u32,
    /// Pipeline stages or asynchronous buffering depth.
    pub pipeline_stages: u32,
    /// Workgroup-shared memory allocated in bytes.
    pub shared_bytes: u32,
}

impl Default for LaunchGeometry {
    #[inline]
    fn default() -> Self {
        Self {
            workgroup: [1, 1, 1],
            grid: [1, 1, 1],
            elements_per_invocation: 1,
            pipeline_stages: 1,
            shared_bytes: 0,
        }
    }
}

impl LaunchGeometry {
    /// Total invocations in one workgroup.
    #[must_use]
    pub const fn workgroup_invocations(&self) -> u32 {
        self.workgroup[0]
            .saturating_mul(self.workgroup[1])
            .saturating_mul(self.workgroup[2])
    }

    /// Total workgroups in the dispatch grid.
    #[must_use]
    pub const fn grid_total(&self) -> u32 {
        self.grid[0]
            .saturating_mul(self.grid[1])
            .saturating_mul(self.grid[2])
    }

    /// Total invocations dispatched across the entire grid.
    #[must_use]
    pub const fn total_invocations(&self) -> u64 {
        (self.workgroup_invocations() as u64).saturating_mul(self.grid_total() as u64)
    }

    /// Total logical data elements covered by the dispatch.
    #[must_use]
    pub const fn total_elements_covered(&self) -> u64 {
        self.total_invocations()
            .saturating_mul(self.elements_per_invocation as u64)
    }

    /// Whether this launch geometry contains non-zero dimensions.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.workgroup[0] > 0
            && self.workgroup[1] > 0
            && self.workgroup[2] > 0
            && self.grid[0] > 0
            && self.grid[1] > 0
            && self.grid[2] > 0
            && self.elements_per_invocation > 0
            && self.pipeline_stages > 0
    }
}

/// Errors occurring during geometry requirement satisfaction and lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryLoweringError {
    /// Requirements could not be satisfied on the target profile.
    UnsatisfiableRequirements(String),
    /// Requested workgroup invocations exceed target limits.
    ExceedsWorkgroupLimits {
        /// Requested invocation count.
        requested: u32,
        /// Maximum invocations admitted by target.
        max: u32,
    },
    /// Requested shared memory bytes exceed target limits.
    ExceedsSharedMemoryLimits {
        /// Requested shared memory bytes.
        requested: u32,
        /// Maximum shared memory bytes admitted by target.
        max: u32,
    },
    /// Requested cooperative width is unsupported by target architecture.
    UnsupportedCooperativeWidth {
        /// Requested cooperative width.
        requested: u32,
        /// Admitted width by target.
        admitted: u32,
    },
}

impl std::fmt::Display for GeometryLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsatisfiableRequirements(msg) => write!(f, "unsatisfiable geometry requirements: {msg}"),
            Self::ExceedsWorkgroupLimits { requested, max } => {
                write!(f, "requested workgroup size {requested} exceeds target maximum {max}")
            }
            Self::ExceedsSharedMemoryLimits { requested, max } => {
                write!(f, "requested shared memory {requested} bytes exceeds target maximum {max}")
            }
            Self::UnsupportedCooperativeWidth { requested, admitted } => {
                write!(f, "requested cooperative width {requested} is not admitted by target (max {admitted})")
            }
        }
    }
}

impl std::error::Error for GeometryLoweringError {}

/// Target lowering strategy for selecting concrete execution geometries.
pub trait GeometryStrategy: Send + Sync {
    /// Return candidate launch geometries ranked in preference order (highest ranked first).
    fn rank_geometries(
        &self,
        requirements: &GeometryRequirements,
        problem_elements: u32,
    ) -> Vec<LaunchGeometry>;

    /// Lower a single best launch geometry, or an error if requirements cannot be met.
    fn lower_geometry(
        &self,
        requirements: &GeometryRequirements,
        problem_elements: u32,
    ) -> Result<LaunchGeometry, GeometryLoweringError> {
        self.rank_geometries(requirements, problem_elements)
            .into_iter()
            .next()
            .ok_or_else(|| {
                GeometryLoweringError::UnsatisfiableRequirements(
                    "no admitting geometry candidate found for requirements on this target".to_string(),
                )
            })
    }
}
