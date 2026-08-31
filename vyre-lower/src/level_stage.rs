//! The physical-kernel level's verifier, canonical form, and analyses.
//!
//! This crate already owns both: `verify` rejects a descriptor no emitter may
//! see, and `canonicalize_for_emit` states the form every emitter reads. The
//! registration is what makes them the physical-kernel level's own, so the
//! level closure over `IrLevel` finds a stage here instead of finding nothing
//! and reporting the level unverifiable.

use std::any::Any;

use vyre_foundation::optimizer::level_contract::{
    LevelAnalysis, LevelStage, LevelStageRegistration, LevelVerdict,
};
use vyre_spec::IrLevel;

use crate::canonicalize::canonicalize_for_emit;
use crate::{verify, KernelDescriptor};

/// Physical-kernel stage: register and shared layouts, transactions, phases.
struct PhysicalKernelStage;

impl LevelStage for PhysicalKernelStage {
    fn level(&self) -> IrLevel {
        IrLevel::PhysicalKernel
    }

    fn subject(&self) -> &'static str {
        "KernelDescriptor"
    }

    fn verify(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(descriptor) = subject.downcast_ref::<KernelDescriptor>() else {
            return LevelVerdict::WrongSubject {
                expected: "KernelDescriptor",
            };
        };
        match verify::verify(descriptor) {
            Ok(()) => LevelVerdict::Verified,
            Err(errors) => {
                LevelVerdict::Rejected(errors.iter().map(|error| format!("{error:?}")).collect())
            }
        }
    }

    fn is_canonical(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(descriptor) = subject.downcast_ref::<KernelDescriptor>() else {
            return LevelVerdict::WrongSubject {
                expected: "KernelDescriptor",
            };
        };
        if canonicalize_for_emit(descriptor) == *descriptor {
            LevelVerdict::Verified
        } else {
            LevelVerdict::rejected("descriptor differs from its emitter-ready canonical form")
        }
    }

    fn analyses(&self) -> &'static [LevelAnalysis] {
        &[
            LevelAnalysis {
                name: "descriptor_resource_bounds",
                invalidated_by_rewrite: true,
            },
            LevelAnalysis {
                name: "descriptor_storage_layout",
                invalidated_by_rewrite: true,
            },
        ]
    }
}

inventory::submit! {
    LevelStageRegistration { stage: &PhysicalKernelStage }
}

/// The level this crate registers a stage for.
///
/// A real function rather than a constant, so the reference in a reader survives
/// inlining and names this crate rather than a value copied out of it. The
/// linkage owner reads it to state which level this crate must have registered.
#[must_use]
pub fn registered_level_stage() -> IrLevel {
    PhysicalKernelStage.level()
}
