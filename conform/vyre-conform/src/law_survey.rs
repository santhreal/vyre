//! The transform disposition of a registered operation, derived by execution.
//!
//! An operation that declares no algebraic law used to say nothing at all, and
//! the answers absence can carry, no legal rewrite exists, the semantics were
//! never characterized, and a declared law the run contradicts, were one blank
//! field. Nothing distinguished them and no reader could act on them.
//!
//! This module derives the answer instead of accepting a label for it. Every
//! family in [`crate::law_proof::PROVABLE_LAWS`] is executed against the
//! operation's own program through the reference oracle, and the verdicts
//! partition the operation into exactly one [`Disposition`]. The partition is a
//! total function of what ran, so it cannot be written, weakened, or shared
//! across a category of operations.
//!
//! The two verdict directions do not carry the same weight, and the partition
//! respects that. A refutation is a counterexample: the witness ran and the
//! bytes differed, so the law is false for this operation and any claim that
//! nothing about it is characterized is false with it. A confirmation ran the
//! witness over the fixture cases the registration carries and found no
//! counterexample, which is a necessary condition of the law and not the law
//! itself. So a confirmation is never promoted into a declared law: it counts
//! only where the registration already declares that family, and there it says
//! the declaration survived execution.
//!
//! Prose is not evidence. A refutation names the fixture case and the first
//! differing byte; an uncharacterized verdict names the shape that admits no
//! witness. Both come from the run, never from the registration.

use vyre_foundation::operation::SemanticOperation;

use crate::law_proof::{prove_law, LawProof, LawVerdict, PROVABLE_LAWS};

/// What executing every provable law family established about one operation.
///
/// Exhaustive and mutually exclusive. A variant added here fails to compile in
/// [`Disposition::name`] and in every reader that matches without a catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Disposition {
    /// A family the registration declares survived its witness on every fixture
    /// case that admitted one, and no declared family was refuted. The
    /// operation has a recorded rewrite and execution agrees with it.
    ProvenLaws,
    /// A witness ran to a counterexample and no declared family survived one.
    /// The refutation is what characterizes the operation: the families in the
    /// executable vocabulary that reach this shape are false for it.
    NoLegalRewrite,
    /// No witness reached a verdict either way: the registration carries no
    /// program builder, no fixture cases, or a shape that admits no witness.
    /// The semantics are not characterized in algebraic terms.
    Uncharacterized,
    /// A proof attempt could not complete. Neither a law nor a decision, and no
    /// recorded answer excuses it.
    Unrunnable,
}

impl Disposition {
    /// Every disposition, so a reader is judged against the whole partition
    /// rather than the members someone remembered.
    pub const ALL: &'static [Self] = &[
        Self::ProvenLaws,
        Self::NoLegalRewrite,
        Self::Uncharacterized,
        Self::Unrunnable,
    ];

    /// Stable name recorded in the generated disposition ledger.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ProvenLaws => "proven-laws",
            Self::NoLegalRewrite => "no-legal-rewrite",
            Self::Uncharacterized => "uncharacterized",
            Self::Unrunnable => "unrunnable",
        }
    }

    /// The disposition `name` spells, or `None` when nothing carries that name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|d| d.name() == name)
    }

    /// The absence decision a registration has to record to agree with this
    /// disposition, or `None` when the disposition is not an absence.
    #[must_use]
    pub const fn required_absence(self) -> Option<vyre_foundation::operation::AbsenceDecision> {
        match self {
            Self::NoLegalRewrite => {
                Some(vyre_foundation::operation::AbsenceDecision::NoLegalRewrite)
            }
            Self::Uncharacterized => {
                Some(vyre_foundation::operation::AbsenceDecision::Uncharacterized)
            }
            Self::ProvenLaws | Self::Unrunnable => None,
        }
    }
}

/// One operation, every family executed against it, and the disposition that
/// follows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSurvey {
    /// Stable operation identifier.
    pub op_id: &'static str,
    /// One row per family in [`PROVABLE_LAWS`], in vocabulary order.
    pub proofs: Vec<LawProof>,
    /// Families whose witness ran over at least one fixture case and found no
    /// counterexample, whether or not the registration declares them.
    pub confirmed: Vec<&'static str>,
    /// Families the registration declares and whose witness survived.
    pub upheld: Vec<&'static str>,
    /// Families the oracle refuted, with the counterexample detail.
    pub refuted: Vec<(&'static str, String)>,
    /// Families the registration declares and the oracle refuted.
    pub broken: Vec<&'static str>,
    /// The derived disposition.
    pub disposition: Disposition,
}

impl OperationSurvey {
    /// Why the operation carries this disposition, in the terms the run
    /// reported. Never a restatement of the disposition name.
    #[must_use]
    pub fn evidence(&self) -> String {
        match self.disposition {
            Disposition::ProvenLaws => {
                let cases: usize = self
                    .proofs
                    .iter()
                    .filter(|proof| self.upheld.contains(&proof.law))
                    .filter_map(|proof| match proof.verdict {
                        LawVerdict::Holds { cases } => Some(cases),
                        LawVerdict::Refuted { .. }
                        | LawVerdict::Unproven { .. }
                        | LawVerdict::Unrunnable { .. } => None,
                    })
                    .sum();
                format!(
                    "the reference oracle upheld the declared {} over {cases} fixture case witness run(s)",
                    self.upheld.join(", ")
                )
            }
            Disposition::NoLegalRewrite => {
                let first = self
                    .refuted
                    .first()
                    .map_or_else(String::new, |(law, detail)| format!("`{law}`: {detail}"));
                format!(
                    "the reference oracle refuted {}; {first}",
                    self.refuted
                        .iter()
                        .map(|(law, _)| *law)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Disposition::Uncharacterized => {
                if self.confirmed.is_empty() {
                    self.proofs
                        .iter()
                        .find_map(|proof| match &proof.verdict {
                            LawVerdict::Unproven { reason, .. } => Some(reason.clone()),
                            LawVerdict::Holds { .. }
                            | LawVerdict::Refuted { .. }
                            | LawVerdict::Unrunnable { .. } => None,
                        })
                        .unwrap_or_else(|| "no family produced a verdict".to_string())
                } else {
                    format!(
                        "no counterexample to {} was found over the fixture cases, which is a necessary condition of the law and not the law",
                        self.confirmed.join(", ")
                    )
                }
            }
            Disposition::Unrunnable => self
                .proofs
                .iter()
                .find_map(|proof| match &proof.verdict {
                    LawVerdict::Unrunnable { reason } => Some(reason.clone()),
                    LawVerdict::Holds { .. }
                    | LawVerdict::Refuted { .. }
                    | LawVerdict::Unproven { .. } => None,
                })
                .unwrap_or_else(|| "a proof attempt could not complete".to_string()),
        }
    }
}

/// Execute every provable family against `entry` and derive its disposition.
///
/// A `Holds` verdict that exercised zero cases establishes nothing, so it
/// counts as neither a confirmation nor a refutation. Selecting a plan already
/// requires one admitting case, and treating the count as load-bearing here is
/// what keeps a shape change from turning an empty run into evidence.
#[must_use]
pub fn survey_operation(entry: &SemanticOperation) -> OperationSurvey {
    let proofs: Vec<LawProof> = PROVABLE_LAWS
        .iter()
        .map(|law| prove_law(entry, law))
        .collect();
    let mut confirmed = Vec::new();
    let mut refuted = Vec::new();
    let mut unrunnable = false;
    for proof in &proofs {
        match &proof.verdict {
            LawVerdict::Holds { cases } if *cases > 0 => confirmed.push(proof.law),
            LawVerdict::Refuted { detail, .. } => refuted.push((proof.law, detail.clone())),
            LawVerdict::Unrunnable { .. } => unrunnable = true,
            LawVerdict::Holds { .. } | LawVerdict::Unproven { .. } => {}
        }
    }
    let upheld: Vec<&'static str> = confirmed
        .iter()
        .copied()
        .filter(|law| entry.laws.contains(law))
        .collect();
    let broken: Vec<&'static str> = refuted
        .iter()
        .map(|(law, _)| *law)
        .filter(|law| entry.laws.contains(law))
        .collect();
    let disposition = if unrunnable {
        Disposition::Unrunnable
    } else if !upheld.is_empty() && broken.is_empty() {
        Disposition::ProvenLaws
    } else if !refuted.is_empty() {
        Disposition::NoLegalRewrite
    } else {
        Disposition::Uncharacterized
    };
    OperationSurvey {
        op_id: entry.id,
        proofs,
        confirmed,
        upheld,
        refuted,
        broken,
        disposition,
    }
}

/// Every way a registration's recorded decision can disagree with what ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispositionDefect {
    /// The registration declares no law and records no absence decision.
    Silent,
    /// The registration declares a family the oracle refuted.
    RefutedLaw {
        /// The declared family the counterexample falsifies.
        law: &'static str,
        /// The counterexample.
        detail: String,
    },
    /// The registration records an absence class the run contradicts.
    WrongAbsence {
        /// What the registration records.
        recorded: &'static str,
        /// What the run establishes.
        derived: &'static str,
        /// What ran.
        evidence: String,
    },
    /// The run upholds a declared law, so the operation is not absent, and the
    /// registration records an absence class beside it.
    AbsenceBesideLaw {
        /// What the registration records.
        recorded: &'static str,
        /// What ran.
        evidence: String,
    },
    /// The registration declares laws that reached no verdict and records no
    /// absence decision either, so nothing states what the operation admits.
    DeclaredWithoutVerdict {
        /// What the run establishes.
        derived: &'static str,
        /// What ran.
        evidence: String,
    },
    /// A proof attempt could not complete.
    Unrunnable {
        /// What the run reported.
        evidence: String,
    },
}

impl DispositionDefect {
    /// One line naming the operation, what it records, and what ran.
    #[must_use]
    pub fn describe(&self, op_id: &str) -> String {
        match self {
            Self::Silent => format!(
                "operation `{op_id}` declares no algebraic law and records no absence decision"
            ),
            Self::RefutedLaw { law, detail } => format!(
                "operation `{op_id}` declares `{law}` and the reference oracle refuted it: {detail}"
            ),
            Self::WrongAbsence {
                recorded,
                derived,
                evidence,
            } => format!(
                "operation `{op_id}` records `{recorded}` and {evidence}, which is `{derived}`"
            ),
            Self::AbsenceBesideLaw { recorded, evidence } => format!(
                "operation `{op_id}` records the absence class `{recorded}` beside a law the run upholds; {evidence}"
            ),
            Self::DeclaredWithoutVerdict { derived, evidence } => format!(
                "operation `{op_id}` declares laws that reached no verdict and records no absence decision; {evidence}, which is `{derived}`"
            ),
            Self::Unrunnable { evidence } => {
                format!("operation `{op_id}` could not be proved: {evidence}")
            }
        }
    }
}

/// Judge one registration's recorded decision against its survey.
///
/// The recorded decision is authored; the disposition is executed. Every defect
/// is a disagreement between the two, so a decision cannot be written into
/// agreement with itself. The match over the disposition is exhaustive with no
/// catch-all: a member added to the partition fails to compile here until the
/// decision it demands is stated.
#[must_use]
pub fn judge(entry: &SemanticOperation, survey: &OperationSurvey) -> Vec<DispositionDefect> {
    let mut defects = Vec::new();
    let recorded = entry.absence();
    if entry.laws.is_empty() && recorded.is_none() {
        defects.push(DispositionDefect::Silent);
        return defects;
    }
    for law in &survey.broken {
        let detail = survey
            .refuted
            .iter()
            .find(|(name, _)| name == law)
            .map_or_else(String::new, |(_, detail)| detail.clone());
        defects.push(DispositionDefect::RefutedLaw { law, detail });
    }
    match survey.disposition {
        Disposition::Unrunnable => defects.push(DispositionDefect::Unrunnable {
            evidence: survey.evidence(),
        }),
        Disposition::ProvenLaws => {
            if let Some(absence) = recorded {
                defects.push(DispositionDefect::AbsenceBesideLaw {
                    recorded: absence.name(),
                    evidence: survey.evidence(),
                });
            }
        }
        Disposition::NoLegalRewrite => judge_absence(
            vyre_foundation::operation::AbsenceDecision::NoLegalRewrite,
            recorded,
            survey,
            &mut defects,
        ),
        Disposition::Uncharacterized => judge_absence(
            vyre_foundation::operation::AbsenceDecision::Uncharacterized,
            recorded,
            survey,
            &mut defects,
        ),
    }
    defects
}

/// Judge a recorded decision against the absence `required` by the disposition.
///
/// Each absence arm names its own decision, so the requirement travels as a
/// value rather than as an `Option` the caller has to re-derive and then trust.
fn judge_absence(
    required: vyre_foundation::operation::AbsenceDecision,
    recorded: Option<vyre_foundation::operation::AbsenceDecision>,
    survey: &OperationSurvey,
    defects: &mut Vec<DispositionDefect>,
) {
    match recorded {
        Some(absence) if absence == required => {}
        Some(absence) => defects.push(DispositionDefect::WrongAbsence {
            recorded: absence.name(),
            derived: required.name(),
            evidence: survey.evidence(),
        }),
        None => defects.push(DispositionDefect::DeclaredWithoutVerdict {
            derived: required.name(),
            evidence: survey.evidence(),
        }),
    }
}
