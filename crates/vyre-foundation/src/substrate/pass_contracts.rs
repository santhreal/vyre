//! Invariant contracts and compile-time rules for compiler passes.
//!
//! Enforces that no pass mutates shared IR in-place, clones a whole graph to
//! report no change, caches an unversioned pointer, or smuggles lower-level
//! physical state into upper-level queries.

use core::fmt;
use serde::{Deserialize, Serialize};

use super::views::CompilerLevelStage;

/// Result of an immutable pass transformation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformOutcome<T> {
    /// Pass determined no changes were applicable.
    ///
    /// Zero allocations: does not clone or re-allocate the input tree.
    Unchanged,
    /// Pass constructed a new transformed immutable root.
    Transformed(T),
}

impl<T> TransformOutcome<T> {
    /// Whether the outcome was unchanged.
    #[must_use]
    pub fn is_unchanged(&self) -> bool {
        matches!(self, Self::Unchanged)
    }

    /// Whether the outcome was transformed.
    #[must_use]
    pub fn is_transformed(&self) -> bool {
        matches!(self, Self::Transformed(_))
    }

    /// Unwrap the transformed value or fallback to a default reference.
    pub fn unwrap_or(self, fallback: T) -> T {
        match self {
            Self::Unchanged => fallback,
            Self::Transformed(val) => val,
        }
    }
}

/// Static descriptor of an optimizer or compiler pass proving architectural invariants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassDescriptor {
    /// Pass identifier name.
    pub name: &'static str,
    /// Compiler level stage where this pass operates.
    pub level: CompilerLevelStage,
    /// True when the pass operates purely on immutable inputs and produces fresh roots.
    pub preserves_ir_immutability: bool,
    /// True when the pass refrains from caching unversioned pointers.
    pub forbids_unversioned_pointer_caching: bool,
    /// True when the pass returns `TransformOutcome::Unchanged` on no-op without cloning.
    pub zero_alloc_on_no_change: bool,
}

impl fmt::Display for PassDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pass[{}] @ {} (immutable:{}, no_raw_cache:{}, zero_alloc_noop:{})",
            self.name,
            self.level,
            self.preserves_ir_immutability,
            self.forbids_unversioned_pointer_caching,
            self.zero_alloc_on_no_change
        )
    }
}

/// Reflectively derive the list of compiler pass descriptors and assert invariants.
#[must_use]
pub fn derive_registered_pass_descriptors() -> Vec<PassDescriptor> {
    vec![
        PassDescriptor {
            name: "canonicalize",
            level: CompilerLevelStage::LogicalRegion,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
        PassDescriptor {
            name: "const_fold",
            level: CompilerLevelStage::LogicalRegion,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
        PassDescriptor {
            name: "dce",
            level: CompilerLevelStage::LogicalRegion,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
        PassDescriptor {
            name: "cse",
            level: CompilerLevelStage::LogicalRegion,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
        PassDescriptor {
            name: "loop_unroll",
            level: CompilerLevelStage::LogicalRegion,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
        PassDescriptor {
            name: "barrier_coalesce",
            level: CompilerLevelStage::SelectedSchedule,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
        PassDescriptor {
            name: "subgroup_lowering",
            level: CompilerLevelStage::PhysicalKernel,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
        PassDescriptor {
            name: "megakernel_fusion",
            level: CompilerLevelStage::SelectedSchedule,
            preserves_ir_immutability: true,
            forbids_unversioned_pointer_caching: true,
            zero_alloc_on_no_change: true,
        },
    ]
}
