//! The verifier, canonical form, and analyses one IR level owns.
//!
//! [`level_pipelines`](crate::optimizer::level_pipeline::level_pipelines) states which
//! passes act at each level. A pipeline alone leaves three questions open: what
//! rejects a subject the level's passes must never see, what form the level's
//! passes converge to, and which derived facts are the level's to hold. Every
//! answer already exists in this workspace and is called on the production
//! path. What no level stated is which of them is its own, so a level shipped
//! with none of the three and nothing turned red.
//!
//! A stage states all three for one level. The subject is erased because the
//! levels below the logical one are owned by crates that depend on this one:
//! they register through `inventory`, the path
//! [`RewriteContractRegistration`](crate::optimizer::rewrite_contract::RewriteContractRegistration)
//! already uses. A stage that cannot recognize its own subject reports
//! [`LevelVerdict::WrongSubject`](crate::optimizer::level_contract::LevelVerdict::WrongSubject)
//! rather than verifying it, so an erased
//! argument of the wrong type is a refusal and never a pass.

use std::any::Any;
use std::sync::LazyLock;

use vyre_spec::IrLevel;

use crate::ir::{Program, ProgramGraph, ProgramGraphIdentityContext};
use crate::schedule::SelectedSchedule;

/// Outcome of verifying one subject at one level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LevelVerdict {
    /// Every invariant this level owns holds.
    Verified,
    /// The subject is of this level and violates the stated invariants.
    Rejected(Vec<String>),
    /// The subject is not of this level's type.
    WrongSubject {
        /// Type this level verifies.
        expected: &'static str,
    },
}

impl LevelVerdict {
    /// Whether the subject was verified.
    #[must_use]
    pub const fn is_verified(&self) -> bool {
        matches!(self, Self::Verified)
    }

    /// One rejection carrying `reason`.
    #[must_use]
    pub fn rejected(reason: impl Into<String>) -> Self {
        Self::Rejected(vec![reason.into()])
    }
}

/// A derived fact set one level holds.
///
/// A name has exactly one owning level. Two levels deriving the same fact is
/// the duplicate-analysis defect a per-level manager exists to prevent, so the
/// closure test rejects a repeated name rather than merging the rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LevelAnalysis {
    /// Stable analysis name, unique across levels.
    pub name: &'static str,
    /// Whether a rewrite at this level invalidates the fact set.
    pub invalidated_by_rewrite: bool,
}

/// The verifier, canonical form, and analyses of one IR level.
pub trait LevelStage: Send + Sync {
    /// Level this stage owns.
    fn level(&self) -> IrLevel;

    /// Type name of the subject this stage verifies.
    fn subject(&self) -> &'static str;

    /// Verify the invariants this level owns.
    fn verify(&self, subject: &dyn Any) -> LevelVerdict;

    /// Whether `subject` is already in this level's canonical form.
    ///
    /// A canonical form is stated as a decision rather than a rewrite because
    /// the levels below the logical one canonicalize into their own owned
    /// types, and an erased rewrite would have to name a return type every
    /// level shares. The idempotence a canonical form must have is checked as
    /// this predicate holding of a canonicalized subject and the canonicalizer
    /// being the level's own.
    fn is_canonical(&self, subject: &dyn Any) -> LevelVerdict;

    /// Fact sets this level derives.
    fn analyses(&self) -> &'static [LevelAnalysis];
}

/// Link-time registration of one level stage.
///
/// A crate that owns a level's subject submits its stage:
///
/// ```ignore
/// inventory::submit! {
///     LevelStageRegistration { stage: &MyStage }
/// }
/// ```
pub struct LevelStageRegistration {
    /// The registered stage.
    pub stage: &'static (dyn LevelStage + Sync),
}

inventory::collect!(LevelStageRegistration);

/// Every registered stage, ordered from whole program to target payload.
#[must_use]
pub fn registered_level_stages() -> Vec<&'static (dyn LevelStage + Sync)> {
    frozen_stages().clone()
}

/// The levels this crate's own stages register.
///
/// A reader that links this crate asserts its rows reached the registry against
/// this list, so a stage compiled out of the build is a reported absence rather
/// than a level that quietly has no verifier.
#[must_use]
pub fn stages_registered_here() -> &'static [IrLevel] {
    &[IrLevel::WholeGraph, IrLevel::Logical, IrLevel::Schedule]
}

/// The stage registered for `level`, when exactly one is.
#[must_use]
pub fn stage_for_level(level: IrLevel) -> Option<&'static (dyn LevelStage + Sync)> {
    let mut found = None;
    for stage in frozen_stages() {
        if stage.level() == level {
            if found.is_some() {
                return None;
            }
            found = Some(*stage);
        }
    }
    found
}

/// Verify `subject` at `level` through the stage that owns it.
///
/// A level with no registered stage cannot verify anything, which the closure
/// test reports as a missing stage rather than as a verified subject.
#[must_use]
pub fn verify_at_level(level: IrLevel, subject: &dyn Any) -> Option<LevelVerdict> {
    stage_for_level(level).map(|stage| stage.verify(subject))
}

/// Every analysis name paired with the level that owns it.
#[must_use]
pub fn analysis_owners() -> Vec<(&'static str, IrLevel)> {
    let mut owners: Vec<(&'static str, IrLevel)> = frozen_stages()
        .iter()
        .flat_map(|stage| {
            stage
                .analyses()
                .iter()
                .map(move |analysis| (analysis.name, stage.level()))
        })
        .collect();
    owners.sort_unstable();
    owners
}

/// The registry read once, ordered by level.
fn frozen_stages() -> &'static Vec<&'static (dyn LevelStage + Sync)> {
    static FROZEN: LazyLock<Vec<&'static (dyn LevelStage + Sync)>> = LazyLock::new(build_stages);
    &FROZEN
}

/// Read the registry and order it by level.
fn build_stages() -> Vec<&'static (dyn LevelStage + Sync)> {
    let mut stages: Vec<&'static (dyn LevelStage + Sync)> =
        inventory::iter::<LevelStageRegistration>
            .into_iter()
            .map(|registration| registration.stage)
            .collect();
    stages.sort_by_key(|stage| (stage.level(), stage.subject()));
    stages
}

/// The whole-graph level's subject: a composition and the provenance its
/// identity is derived from.
///
/// The graph alone cannot be verified against anything a caller can get wrong:
/// its builders reject a malformed composition on insert, so a graph that
/// exists has already passed the structural check. What a caller supplies
/// separately, and can supply wrongly, is the provenance outside the
/// topology: the symbol bindings and constant identities the composition's
/// identity covers.
#[derive(Debug, Clone)]
pub struct GraphComposition {
    /// The connected composition.
    pub graph: ProgramGraph,
    /// Facts outside the topology that the composition's identity covers.
    pub provenance: ProgramGraphIdentityContext,
}

/// Whole-graph stage: connected composition, dataflow, and identity.
struct WholeGraphStage;

impl LevelStage for WholeGraphStage {
    fn level(&self) -> IrLevel {
        IrLevel::WholeGraph
    }

    fn subject(&self) -> &'static str {
        "GraphComposition"
    }

    fn verify(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(composition) = subject.downcast_ref::<GraphComposition>() else {
            return LevelVerdict::WrongSubject {
                expected: "GraphComposition",
            };
        };
        match composition.graph.identity(&composition.provenance) {
            Ok(_) => LevelVerdict::Verified,
            Err(error) => LevelVerdict::rejected(error.to_string()),
        }
    }

    fn is_canonical(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(composition) = subject.downcast_ref::<GraphComposition>() else {
            return LevelVerdict::WrongSubject {
                expected: "GraphComposition",
            };
        };
        let bytes = match composition.graph.to_wire() {
            Ok(bytes) => bytes,
            Err(error) => return LevelVerdict::rejected(error.to_string()),
        };
        match ProgramGraph::from_wire(&bytes).and_then(|decoded| decoded.to_wire()) {
            Ok(again) if again == bytes => LevelVerdict::Verified,
            Ok(_) => LevelVerdict::rejected("graph wire bytes are not stable across a round trip"),
            Err(error) => LevelVerdict::rejected(error.to_string()),
        }
    }

    fn analyses(&self) -> &'static [LevelAnalysis] {
        &[LevelAnalysis {
            name: "graph_allocation_plan",
            invalidated_by_rewrite: true,
        }]
    }
}

/// Logical stage: region IR, iteration domains, effects.
struct LogicalStage;

impl LevelStage for LogicalStage {
    fn level(&self) -> IrLevel {
        IrLevel::Logical
    }

    fn subject(&self) -> &'static str {
        "Program"
    }

    fn verify(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(program) = subject.downcast_ref::<Program>() else {
            return LevelVerdict::WrongSubject {
                expected: "Program",
            };
        };
        let errors = crate::validate::validate(program);
        if errors.is_empty() {
            LevelVerdict::Verified
        } else {
            LevelVerdict::Rejected(errors.iter().map(ToString::to_string).collect())
        }
    }

    fn is_canonical(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(program) = subject.downcast_ref::<Program>() else {
            return LevelVerdict::WrongSubject {
                expected: "Program",
            };
        };
        if program.entry() == program.canonicalized().entry() {
            LevelVerdict::Verified
        } else {
            LevelVerdict::rejected("program body differs from its canonical form")
        }
    }

    fn analyses(&self) -> &'static [LevelAnalysis] {
        &[
            LevelAnalysis {
                name: "program_shape_facts",
                invalidated_by_rewrite: true,
            },
            LevelAnalysis {
                name: "program_use_facts",
                invalidated_by_rewrite: true,
            },
            LevelAnalysis {
                name: "program_type_facts",
                invalidated_by_rewrite: true,
            },
        ]
    }
}

/// Schedule stage: the selected schedule, its transforms and resource bounds.
struct ScheduleStage;

impl LevelStage for ScheduleStage {
    fn level(&self) -> IrLevel {
        IrLevel::Schedule
    }

    fn subject(&self) -> &'static str {
        "SelectedSchedule"
    }

    fn verify(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(schedule) = subject.downcast_ref::<SelectedSchedule>() else {
            return LevelVerdict::WrongSubject {
                expected: "SelectedSchedule",
            };
        };
        match schedule.validate() {
            Ok(()) => LevelVerdict::Verified,
            Err(error) => LevelVerdict::rejected(error.to_string()),
        }
    }

    fn is_canonical(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(schedule) = subject.downcast_ref::<SelectedSchedule>() else {
            return LevelVerdict::WrongSubject {
                expected: "SelectedSchedule",
            };
        };
        let bytes = match schedule.canonical_wire() {
            Ok(bytes) => bytes,
            Err(error) => return LevelVerdict::rejected(error.to_string()),
        };
        match serde_json::from_slice::<SelectedSchedule>(&bytes) {
            Ok(decoded) if decoded == *schedule => LevelVerdict::Verified,
            Ok(_) => LevelVerdict::rejected("schedule differs from its canonical wire form"),
            Err(error) => LevelVerdict::rejected(error.to_string()),
        }
    }

    fn analyses(&self) -> &'static [LevelAnalysis] {
        &[
            LevelAnalysis {
                name: "schedule_resource_bounds",
                invalidated_by_rewrite: true,
            },
            LevelAnalysis {
                name: "schedule_phase_dependencies",
                invalidated_by_rewrite: true,
            },
        ]
    }
}

inventory::submit! {
    LevelStageRegistration { stage: &WholeGraphStage }
}

inventory::submit! {
    LevelStageRegistration { stage: &LogicalStage }
}

inventory::submit! {
    LevelStageRegistration { stage: &ScheduleStage }
}
