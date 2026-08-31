//! Target-neutral launch geometry and requirement definitions.
//!
//! Substrate-neutral operations declare the execution constraints their
//! algorithm requires through [`GeometryRequirements`]. A backend reports which
//! widths its authenticated profile admits, and `vyre-megakernel` builds and
//! orders the concrete [`LaunchGeometry`] candidates under the compile
//! objective.
//!
//! This module contains no device names, no instruction names, and no concrete
//! device limits.

mod logical_span;

pub use logical_span::{
    admitted_logical_span, guarded_logical_span, launch_covers_full_input_span,
};

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

/// Substrate-neutral schedule constraints declared by an operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GeometryRequirements {
    /// Minimum or exact workgroup execution width.
    pub cooperative_width: CooperativeWidth,
    /// Minimum or exact subgroup execution width.
    pub subgroup_width: CooperativeWidth,
    /// Minimum workgroup-shared memory in bytes needed by the algorithm.
    pub min_shared_bytes: u32,
    /// Divisibility or element processing policy per invocation.
    pub per_invocation_elements: ElementPolicy,
    /// Uniformity requirements for cross-invocation divergence prevention.
    pub subgroup_uniformity: Uniformity,
    /// Whether all workgroups must be resident for a grid-wide synchronization.
    pub requires_cooperative_launch: bool,
    /// Strongest memory ordering required by atomics or barriers.
    pub memory_ordering: Option<crate::memory_model::MemoryOrdering>,
}

impl GeometryRequirements {
    /// Unconstrained geometry requirements.
    #[must_use]
    pub const fn agnostic() -> Self {
        Self {
            cooperative_width: CooperativeWidth::Agnostic,
            subgroup_width: CooperativeWidth::Agnostic,
            min_shared_bytes: 0,
            per_invocation_elements: ElementPolicy::Any,
            subgroup_uniformity: Uniformity::None,
            requires_cooperative_launch: false,
            memory_ordering: None,
        }
    }

    /// Geometry requirements with a specific workgroup width constraint.
    #[must_use]
    pub const fn cooperative(width: CooperativeWidth) -> Self {
        Self {
            cooperative_width: width,
            ..Self::agnostic()
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

    /// Attach a subgroup width constraint.
    #[must_use]
    pub const fn with_subgroup_width(mut self, width: CooperativeWidth) -> Self {
        self.subgroup_width = width;
        self
    }

    /// Require a cooperative launch for grid-wide synchronization.
    #[must_use]
    pub const fn with_cooperative_launch(mut self) -> Self {
        self.requires_cooperative_launch = true;
        self
    }

    /// Attach a minimum memory ordering.
    #[must_use]
    pub const fn with_memory_ordering(
        mut self,
        ordering: crate::memory_model::MemoryOrdering,
    ) -> Self {
        self.memory_ordering = Some(ordering);
        self
    }

    /// Compose two neutral constraint records.
    ///
    /// # Errors
    ///
    /// Returns a stable conflict when no schedule can satisfy both records.
    pub fn compose(self, other: Self) -> Result<Self, GeometryConstraintConflict> {
        Ok(Self {
            cooperative_width: compose_width(
                "workgroup",
                self.cooperative_width,
                other.cooperative_width,
            )?,
            subgroup_width: compose_width("subgroup", self.subgroup_width, other.subgroup_width)?,
            min_shared_bytes: self.min_shared_bytes.max(other.min_shared_bytes),
            per_invocation_elements: compose_elements(
                self.per_invocation_elements,
                other.per_invocation_elements,
            )?,
            subgroup_uniformity: compose_uniformity(
                self.subgroup_uniformity,
                other.subgroup_uniformity,
            ),
            requires_cooperative_launch: self.requires_cooperative_launch
                || other.requires_cooperative_launch,
            memory_ordering: match (self.memory_ordering, other.memory_ordering) {
                (Some(left), Some(right)) => Some(left.join(right)),
                (Some(ordering), None) | (None, Some(ordering)) => Some(ordering),
                (None, None) => None,
            },
        })
    }

    /// Derive the minimum neutral constraints required by program semantics.
    ///
    /// # Errors
    ///
    /// Returns a stable overflow reason when workgroup scratch cannot be represented.
    pub fn from_program(program: &crate::ir::Program) -> Result<Self, GeometryConstraintConflict> {
        let capabilities = crate::program_caps::scan(program);
        let effects = crate::operation::OperationEffects::from_program(program);
        let scratch_bytes = program
            .buffers
            .iter()
            .filter(|buffer| buffer.access == crate::ir::BufferAccess::Workgroup)
            .try_fold(0u32, |total, buffer| {
                let element_bytes = buffer.element.size_bytes().unwrap_or(0) as u32;
                let bytes = buffer
                    .count
                    .checked_mul(element_bytes)
                    .ok_or(GeometryConstraintConflict::SharedScratchOverflow)?;
                total
                    .checked_add(bytes)
                    .ok_or(GeometryConstraintConflict::SharedScratchOverflow)
            })?;
        let workgroup_width = if program.workgroup_size_is_schedule_only() {
            CooperativeWidth::Agnostic
        } else {
            CooperativeWidth::Exactly(
                program.workgroup_size[0]
                    .saturating_mul(program.workgroup_size[1])
                    .saturating_mul(program.workgroup_size[2]),
            )
        };
        let memory_ordering = program_memory_ordering(program);
        Ok(Self {
            cooperative_width: workgroup_width,
            subgroup_width: CooperativeWidth::Agnostic,
            min_shared_bytes: scratch_bytes,
            per_invocation_elements: ElementPolicy::Any,
            subgroup_uniformity: if effects.synchronizes {
                Uniformity::WorkgroupUniform
            } else if capabilities.subgroup_ops {
                Uniformity::SubgroupUniform
            } else {
                Uniformity::None
            },
            requires_cooperative_launch: capabilities.grid_sync,
            memory_ordering,
        })
    }
}

fn program_memory_ordering(
    program: &crate::ir::Program,
) -> Option<crate::memory_model::MemoryOrdering> {
    fn include(
        aggregate: &mut Option<crate::memory_model::MemoryOrdering>,
        ordering: crate::memory_model::MemoryOrdering,
    ) {
        *aggregate = Some(match aggregate {
            Some(current) => current.join(ordering),
            None => ordering,
        });
    }

    let mut ordering = None;
    crate::visit::for_each_node(program.entry(), |node| {
        if let crate::ir::Node::Barrier { ordering: required }
        | crate::ir::Node::LogicalBarrier { ordering: required } = node
        {
            include(&mut ordering, *required);
        }
    });
    crate::visit::for_each_expr(program.entry(), |expr| {
        if let crate::ir::Expr::Atomic {
            ordering: required, ..
        } = expr
        {
            include(&mut ordering, *required);
        }
    });
    ordering
}

impl std::fmt::Display for GeometryConstraintConflict {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExactWidth { scope, left, right } => {
                write!(
                    formatter,
                    "{scope} exact widths conflict: {left} versus {right}"
                )
            }
            Self::ZeroWidth { scope } => {
                write!(formatter, "{scope} width constraint is zero")
            }
            Self::WidthBelowMinimum {
                scope,
                exact,
                minimum,
            } => write!(
                formatter,
                "{scope} exact width {exact} is below required minimum {minimum}"
            ),
            Self::ZeroElementMultiple => formatter.write_str("element multiple constraint is zero"),
            Self::ElementMultipleOverflow { left, right } => write!(
                formatter,
                "element multiples {left} and {right} exceed the representable schedule constraint"
            ),
            Self::SharedScratchOverflow => formatter
                .write_str("workgroup scratch exceeds the representable schedule constraint"),
        }
    }
}

impl std::error::Error for GeometryConstraintConflict {}

/// Stable reason that two neutral schedule contracts cannot compose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryConstraintConflict {
    /// Two exact widths disagree.
    ExactWidth {
        /// Constraint scope.
        scope: &'static str,
        /// Left exact width.
        left: u32,
        /// Right exact width.
        right: u32,
    },
    /// A width constraint used zero.
    ZeroWidth {
        /// Constraint scope.
        scope: &'static str,
    },
    /// An exact width is smaller than a required minimum.
    WidthBelowMinimum {
        /// Constraint scope.
        scope: &'static str,
        /// Exact width.
        exact: u32,
        /// Required minimum.
        minimum: u32,
    },
    /// An element multiple used zero.
    ZeroElementMultiple,
    /// Element-policy least common multiple exceeded `u32`.
    ElementMultipleOverflow {
        /// Left multiple.
        left: u32,
        /// Right multiple.
        right: u32,
    },
    /// Workgroup scratch byte accounting exceeded `u32`.
    SharedScratchOverflow,
}

fn compose_width(
    scope: &'static str,
    left: CooperativeWidth,
    right: CooperativeWidth,
) -> Result<CooperativeWidth, GeometryConstraintConflict> {
    use CooperativeWidth::{Agnostic, AtLeast, Exactly};
    if matches!(left, AtLeast(0) | Exactly(0)) || matches!(right, AtLeast(0) | Exactly(0)) {
        return Err(GeometryConstraintConflict::ZeroWidth { scope });
    }
    match (left, right) {
        (Agnostic, width) | (width, Agnostic) => Ok(width),
        (AtLeast(left), AtLeast(right)) => Ok(AtLeast(left.max(right))),
        (Exactly(left), Exactly(right)) if left == right => Ok(Exactly(left)),
        (Exactly(left), Exactly(right)) => {
            Err(GeometryConstraintConflict::ExactWidth { scope, left, right })
        }
        (Exactly(exact), AtLeast(minimum)) | (AtLeast(minimum), Exactly(exact))
            if exact >= minimum =>
        {
            Ok(Exactly(exact))
        }
        (Exactly(exact), AtLeast(minimum)) | (AtLeast(minimum), Exactly(exact)) => {
            Err(GeometryConstraintConflict::WidthBelowMinimum {
                scope,
                exact,
                minimum,
            })
        }
    }
}
fn compose_elements(
    left: ElementPolicy,
    right: ElementPolicy,
) -> Result<ElementPolicy, GeometryConstraintConflict> {
    use ElementPolicy::{Any, Multiple, Scalar};
    if matches!(left, Multiple(0)) || matches!(right, Multiple(0)) {
        return Err(GeometryConstraintConflict::ZeroElementMultiple);
    }
    let left_policy = left;
    let left = match left {
        Any => return Ok(right),
        Scalar => 1,
        Multiple(value) => value,
    };
    let right = match right {
        Any => return Ok(left_policy),
        Scalar => 1,
        Multiple(value) => value,
    };
    let gcd = gcd(left, right);
    let multiple = left
        .checked_div(gcd)
        .and_then(|value| value.checked_mul(right))
        .ok_or(GeometryConstraintConflict::ElementMultipleOverflow { left, right })?;
    Ok(if multiple == 1 {
        Scalar
    } else {
        Multiple(multiple)
    })
}

const fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

const fn compose_uniformity(left: Uniformity, right: Uniformity) -> Uniformity {
    match (left, right) {
        (Uniformity::WorkgroupUniform, _) | (_, Uniformity::WorkgroupUniform) => {
            Uniformity::WorkgroupUniform
        }
        (Uniformity::SubgroupUniform, _) | (_, Uniformity::SubgroupUniform) => {
            Uniformity::SubgroupUniform
        }
        (Uniformity::None, Uniformity::None) => Uniformity::None,
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
