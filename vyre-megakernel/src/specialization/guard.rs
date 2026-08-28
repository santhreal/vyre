//! What a variant is selected by, and the two proofs a guard set must pass.
//!
//! A guard is a conjunction of terms over declared axes. Two properties decide
//! whether a set of them is usable, and both are decided here rather than
//! asserted in prose:
//!
//! **Exclusivity.** Two guards that can admit the same facts and carry the same
//! precedence make selection depend on iteration order. Disjointness is proved
//! from interval and member parts alone; a divisibility term only narrows a
//! guard, so ignoring it can report an overlap that is not one but never miss one
//! that is. A reported overlap is answered by making the guards disjoint or by
//! giving them distinct precedence, which is a stated decision rather than an
//! accident.
//!
//! **Coverage.** Every guard bound cuts each axis domain into atoms, and by
//! construction an atom is wholly inside or wholly outside every guard's
//! interval and member parts. Coverage is then exact: every cell of the atom
//! product is admitted by some guard, or the uncovered cell is named. A
//! divisibility term admits an atom only when every value in it divides, so the
//! tail a tiled variant leaves behind shows up as an uncovered cell instead of a
//! wrong answer at run time.
//!
//! The atom product is bounded. A contract whose domain and guards cut more than
//! [`MAX_COVERAGE_CELLS`] cells is rejected with the count, because a coverage
//! answer nobody can compute is not a coverage answer.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::axis::{AxisValue, SpecializationAxis};
use crate::error::{failure, CompilerFailureKind};
use crate::identity::Digest;
use crate::CompileError;

/// Cells the coverage proof will enumerate before it refuses to answer.
pub const MAX_COVERAGE_CELLS: usize = 4096;

/// The values one axis may take across the whole declared workload domain.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "domain", rename_all = "snake_case")]
pub enum AxisDomain {
    /// Every value in a closed interval.
    Interval {
        /// Smallest admitted value.
        low: u64,
        /// Largest admitted value.
        high: u64,
    },
    /// A finite set of admitted scalars.
    Members {
        /// Admitted values.
        members: BTreeSet<u64>,
    },
    /// Zero and one.
    Boolean,
    /// A finite set of admitted content identities.
    Identities {
        /// Admitted identities.
        identities: BTreeSet<Digest>,
    },
}

impl AxisDomain {
    /// Reject a domain that admits nothing.
    pub(super) fn validate(&self, axis: &SpecializationAxis) -> Result<(), CompileError> {
        match self {
            Self::Interval { low, high } if low > high => Err(failure(
                CompilerFailureKind::InvalidSpecializationContract,
                axis.field(),
                format!("declared interval {low}..={high} is inverted"),
                "state the interval with its smallest value first",
            )),
            Self::Members { members } if members.is_empty() => Err(failure(
                CompilerFailureKind::InvalidSpecializationContract,
                axis.field(),
                "declared member set is empty",
                "state at least one admitted value or remove the axis",
            )),
            Self::Identities { identities } if identities.is_empty() => Err(failure(
                CompilerFailureKind::InvalidSpecializationContract,
                axis.field(),
                "declared identity set is empty",
                "state at least one admitted identity or remove the axis",
            )),
            Self::Interval { .. }
            | Self::Members { .. }
            | Self::Boolean
            | Self::Identities { .. } => Ok(()),
        }
    }

    /// Whether this domain carries content identities rather than scalars.
    #[must_use]
    pub const fn is_identity_domain(&self) -> bool {
        matches!(self, Self::Identities { .. })
    }

    /// Whether the domain admits one value.
    #[must_use]
    pub fn admits(&self, value: AxisValue) -> bool {
        match (self, value) {
            (Self::Interval { low, high }, AxisValue::Scalar(scalar)) => {
                (*low..=*high).contains(&scalar)
            }
            (Self::Members { members }, AxisValue::Scalar(scalar)) => members.contains(&scalar),
            (Self::Boolean, AxisValue::Scalar(scalar)) => scalar <= 1,
            (Self::Identities { identities }, AxisValue::Identity(digest)) => {
                identities.contains(&digest)
            }
            (
                Self::Interval { .. } | Self::Members { .. } | Self::Boolean,
                AxisValue::Identity(_),
            )
            | (Self::Identities { .. }, AxisValue::Scalar(_)) => false,
        }
    }
}

/// One atom of one axis: a run of scalars, or one identity.
///
/// Every guard's interval and member parts either admit the whole atom or none
/// of it, which is what makes the coverage proof exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Atom {
    /// A closed run of scalars.
    Scalars {
        /// Smallest value in the run.
        low: u64,
        /// Largest value in the run.
        high: u64,
    },
    /// One content identity.
    Identity(Digest),
}

impl Atom {
    /// A value inside the atom, for naming an uncovered cell.
    fn witness(self) -> AxisValue {
        match self {
            Self::Scalars { low, .. } => AxisValue::Scalar(low),
            Self::Identity(digest) => AxisValue::Identity(digest),
        }
    }

    /// Whether the atom holds exactly one value.
    const fn is_point(self) -> bool {
        match self {
            Self::Scalars { low, high } => low == high,
            Self::Identity(_) => true,
        }
    }
}

/// One condition on one axis.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "term", rename_all = "snake_case")]
pub enum GuardTerm {
    /// The axis value lies in a closed interval.
    InRange {
        /// Axis the term reads.
        axis: SpecializationAxis,
        /// Smallest admitted value.
        low: u64,
        /// Largest admitted value.
        high: u64,
    },
    /// The axis value divides by a stated divisor.
    DivisibleBy {
        /// Axis the term reads.
        axis: SpecializationAxis,
        /// Divisor the value must be a multiple of.
        divisor: u64,
    },
    /// The axis value is one of a finite set.
    OneOf {
        /// Axis the term reads.
        axis: SpecializationAxis,
        /// Admitted values.
        members: BTreeSet<u64>,
    },
    /// The axis identity equals one content digest.
    Identity {
        /// Axis the term reads.
        axis: SpecializationAxis,
        /// Admitted identity.
        identity: Digest,
    },
}

impl GuardTerm {
    /// The axis this term reads.
    #[must_use]
    pub const fn axis(&self) -> &SpecializationAxis {
        match self {
            Self::InRange { axis, .. }
            | Self::DivisibleBy { axis, .. }
            | Self::OneOf { axis, .. }
            | Self::Identity { axis, .. } => axis,
        }
    }
}

/// Everything one guard states about one axis, after conjunction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct AxisConstraint {
    interval: Option<(u64, u64)>,
    members: Option<BTreeSet<u64>>,
    divisor: Option<u64>,
    identities: Option<BTreeSet<Digest>>,
}

impl AxisConstraint {
    /// Whether every value of `atom` satisfies this constraint.
    fn admits_atom(&self, atom: Atom) -> bool {
        match atom {
            Atom::Scalars { low, high } => {
                if let Some((constraint_low, constraint_high)) = self.interval {
                    if low < constraint_low || high > constraint_high {
                        return false;
                    }
                }
                if let Some(members) = &self.members {
                    if !atom.is_point() || !members.contains(&low) {
                        return false;
                    }
                }
                if let Some(divisor) = self.divisor {
                    if divisor > 1 && (!atom.is_point() || low % divisor != 0) {
                        return false;
                    }
                }
                self.identities.is_none()
            }
            Atom::Identity(digest) => self
                .identities
                .as_ref()
                .is_some_and(|identities| identities.contains(&digest)),
        }
    }

    /// Whether one value satisfies this constraint.
    fn admits(&self, value: AxisValue) -> bool {
        match value {
            AxisValue::Scalar(scalar) => {
                self.identities.is_none()
                    && self
                        .interval
                        .is_none_or(|(low, high)| (low..=high).contains(&scalar))
                    && self
                        .members
                        .as_ref()
                        .is_none_or(|members| members.contains(&scalar))
                    && self.divisor.is_none_or(|divisor| scalar % divisor == 0)
            }
            AxisValue::Identity(digest) => self
                .identities
                .as_ref()
                .is_some_and(|identities| identities.contains(&digest)),
        }
    }

    /// The largest scalar this constraint and the domain both admit.
    ///
    /// A range-guarded variant is compiled at this extent, so the schedule it
    /// receives covers every value the guard admits. Returning `None` means the
    /// axis carries identities, or admits nothing inside the domain.
    pub(super) fn largest_admitted(&self, domain: &AxisDomain) -> Option<u64> {
        match domain {
            AxisDomain::Identities { .. } => None,
            AxisDomain::Members { members } => members
                .iter()
                .rev()
                .copied()
                .find(|member| self.admits(AxisValue::Scalar(*member))),
            AxisDomain::Boolean => (0..=1)
                .rev()
                .find(|value| self.admits(AxisValue::Scalar(*value))),
            AxisDomain::Interval { low, high } => {
                if self.identities.is_some() {
                    return None;
                }
                let (constraint_low, constraint_high) = self.interval.unwrap_or((0, u64::MAX));
                let start = (*low).max(constraint_low);
                let end = (*high).min(constraint_high);
                if start > end {
                    return None;
                }
                if let Some(members) = &self.members {
                    return members
                        .iter()
                        .rev()
                        .copied()
                        .find(|member| (start..=end).contains(member));
                }
                let divisor = self.divisor.unwrap_or(1).max(1);
                let candidate = end - (end % divisor);
                (candidate >= start).then_some(candidate)
            }
        }
    }

    /// Whether this constraint admits any value the declared domain holds.
    ///
    /// A guard stating a term outside its axis domain is dead: the variant it
    /// selects can never be reached, and the compile bytes are wasted. The
    /// divisor part is read here, because a term admitting only multiples the
    /// domain does not contain is exactly the case worth reporting.
    pub(super) fn intersects_domain(&self, domain: &AxisDomain) -> bool {
        match domain {
            AxisDomain::Identities { identities } => identities
                .iter()
                .any(|identity| self.admits(AxisValue::Identity(*identity))),
            AxisDomain::Members { members } => members
                .iter()
                .any(|member| self.admits(AxisValue::Scalar(*member))),
            AxisDomain::Boolean => (0..=1).any(|value| self.admits(AxisValue::Scalar(value))),
            AxisDomain::Interval { low, high } => {
                if self.identities.is_some() {
                    return false;
                }
                let (constraint_low, constraint_high) = self.interval.unwrap_or((0, u64::MAX));
                let start = (*low).max(constraint_low);
                let end = (*high).min(constraint_high);
                if start > end {
                    return false;
                }
                if let Some(members) = &self.members {
                    return members.iter().any(|member| {
                        (start..=end).contains(member) && self.admits(AxisValue::Scalar(*member))
                    });
                }
                let divisor = self.divisor.unwrap_or(1).max(1);
                start
                    .checked_next_multiple_of(divisor)
                    .is_some_and(|first| first <= end)
            }
        }
    }

    /// Whether two constraints on one axis provably admit no common value.
    ///
    /// Only the interval, member and identity parts are read. A divisibility
    /// term narrows a guard, so leaving it out can leave two disjoint guards
    /// looking as though they might meet; it cannot make two meeting guards look
    /// disjoint, which is the direction that would matter.
    fn provably_disjoint(&self, other: &Self) -> bool {
        if let (Some((left_low, left_high)), Some((right_low, right_high))) =
            (self.interval, other.interval)
        {
            if left_high < right_low || right_high < left_low {
                return true;
            }
        }
        if let (Some(left), Some(right)) = (&self.members, &other.members) {
            if left.is_disjoint(right) {
                return true;
            }
        }
        if let (Some(members), Some((low, high))) = (&self.members, other.interval) {
            if !members.iter().any(|member| (low..=high).contains(member)) {
                return true;
            }
        }
        if let (Some((low, high)), Some(members)) = (self.interval, &other.members) {
            if !members.iter().any(|member| (low..=high).contains(member)) {
                return true;
            }
        }
        if let (Some(left), Some(right)) = (&self.identities, &other.identities) {
            if left.is_disjoint(right) {
                return true;
            }
        }
        self.identities.is_some() != other.identities.is_some()
    }

    /// Bounds this constraint contributes to the atom cut of its axis.
    fn breakpoints(&self, scalars: &mut BTreeSet<u64>, identities: &mut BTreeSet<Digest>) {
        if let Some((low, high)) = self.interval {
            scalars.insert(low);
            scalars.insert(high);
        }
        if let Some(members) = &self.members {
            scalars.extend(members.iter().copied());
        }
        if let Some(divisor) = self.divisor {
            if divisor > 1 {
                scalars.insert(divisor);
            }
        }
        if let Some(guard_identities) = &self.identities {
            identities.extend(guard_identities.iter().copied());
        }
    }
}

/// What selects one compiled variant.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VariantGuard {
    terms: Vec<GuardTerm>,
    precedence: u16,
}

impl VariantGuard {
    /// State a guard as a conjunction of terms at one precedence.
    ///
    /// Terms are held in canonical order so one guard has one serialization and
    /// one identity contribution.
    #[must_use]
    pub fn new(mut terms: Vec<GuardTerm>, precedence: u16) -> Self {
        terms.sort();
        terms.dedup();
        Self { terms, precedence }
    }

    /// The conjoined terms in canonical order.
    #[must_use]
    pub fn terms(&self) -> &[GuardTerm] {
        &self.terms
    }

    /// Order this guard is evaluated in; lower is evaluated first.
    #[must_use]
    pub const fn precedence(&self) -> u16 {
        self.precedence
    }

    /// Axes this guard reads.
    pub(super) fn axes(&self) -> BTreeSet<&SpecializationAxis> {
        self.terms.iter().map(GuardTerm::axis).collect()
    }

    /// Conjoin the terms per axis, rejecting a guard that admits nothing.
    pub(super) fn normalize(
        &self,
    ) -> Result<BTreeMap<SpecializationAxis, AxisConstraint>, CompileError> {
        let mut constraints: BTreeMap<SpecializationAxis, AxisConstraint> = BTreeMap::new();
        for term in &self.terms {
            let constraint = constraints.entry(term.axis().clone()).or_default();
            match term {
                GuardTerm::InRange { low, high, .. } => {
                    let (existing_low, existing_high) =
                        constraint.interval.unwrap_or((0, u64::MAX));
                    constraint.interval = Some((existing_low.max(*low), existing_high.min(*high)));
                }
                GuardTerm::DivisibleBy { divisor, .. } => {
                    let divisor = *divisor;
                    constraint.divisor = Some(match constraint.divisor {
                        Some(existing) => least_common_multiple(existing, divisor),
                        None => divisor,
                    });
                }
                GuardTerm::OneOf { members, .. } => {
                    constraint.members = Some(match constraint.members.take() {
                        Some(existing) => existing.intersection(members).copied().collect(),
                        None => members.clone(),
                    });
                }
                GuardTerm::Identity { identity, .. } => {
                    let mut stated = BTreeSet::new();
                    stated.insert(*identity);
                    constraint.identities = Some(match constraint.identities.take() {
                        Some(existing) => existing.intersection(&stated).copied().collect(),
                        None => stated,
                    });
                }
            }
        }
        for (axis, constraint) in &constraints {
            if unsatisfiable(constraint) {
                return Err(failure(
                    CompilerFailureKind::InvalidVariantGuard,
                    axis.field(),
                    "conjoined terms admit no value on this axis",
                    "state terms that can hold at once, or split them across two variants",
                ));
            }
        }
        Ok(constraints)
    }

    /// Whether the guard admits one complete set of axis facts.
    ///
    /// A fact the guard reads and the caller did not state is a rejection, not a
    /// wildcard: a variant selected on facts nobody supplied is a variant
    /// selected on a guess.
    pub(super) fn admits(
        &self,
        constraints: &BTreeMap<SpecializationAxis, AxisConstraint>,
        facts: &BTreeMap<SpecializationAxis, AxisValue>,
    ) -> bool {
        constraints.iter().all(|(axis, constraint)| {
            facts
                .get(axis)
                .is_some_and(|value| constraint.admits(*value))
        })
    }

    /// Whether the guard admits one complete set of stated facts.
    ///
    /// This is the whole of variant selection: a consumer states the facts it
    /// has and reads which guard admits them. It cannot alter a schedule,
    /// because a guard answers yes or no and carries no schedule.
    ///
    /// # Errors
    ///
    /// Returns an error when the guard's own terms cannot hold at once.
    pub fn admits_facts(
        &self,
        facts: &BTreeMap<SpecializationAxis, AxisValue>,
    ) -> Result<bool, CompileError> {
        Ok(self.admits(&self.normalize()?, facts))
    }
}

/// Whether a conjunction on one axis admits nothing at all.
fn unsatisfiable(constraint: &AxisConstraint) -> bool {
    if constraint
        .identities
        .as_ref()
        .is_some_and(BTreeSet::is_empty)
    {
        return true;
    }
    if constraint.identities.is_some()
        && (constraint.interval.is_some()
            || constraint.members.is_some()
            || constraint.divisor.is_some())
    {
        return true;
    }
    if constraint.members.as_ref().is_some_and(BTreeSet::is_empty) {
        return true;
    }
    if constraint.divisor == Some(0) {
        return true;
    }
    if let Some((low, high)) = constraint.interval {
        if low > high {
            return true;
        }
        if let Some(members) = &constraint.members {
            if !members.iter().any(|member| (low..=high).contains(member)) {
                return true;
            }
        }
    }
    false
}

fn least_common_multiple(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        return 0;
    }
    let product = left.checked_mul(right);
    let divisor = greatest_common_divisor(left, right);
    product.map_or(u64::MAX, |product| product / divisor)
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// The atom cut of one axis under one domain and every guard bound on it.
pub(super) fn atoms(
    domain: &AxisDomain,
    cuts: &BTreeSet<u64>,
    identities: &BTreeSet<Digest>,
) -> Vec<Atom> {
    match domain {
        AxisDomain::Identities {
            identities: declared,
        } => {
            let mut cut: Vec<Atom> = declared.iter().copied().map(Atom::Identity).collect();
            cut.extend(
                identities
                    .iter()
                    .filter(|identity| !declared.contains(identity))
                    .copied()
                    .map(Atom::Identity),
            );
            cut
        }
        AxisDomain::Members { members } => members
            .iter()
            .copied()
            .map(|member| Atom::Scalars {
                low: member,
                high: member,
            })
            .collect(),
        AxisDomain::Boolean => vec![
            Atom::Scalars { low: 0, high: 0 },
            Atom::Scalars { low: 1, high: 1 },
        ],
        AxisDomain::Interval { low, high } => scalar_atoms(*low, *high, cuts),
    }
}

/// Cut a closed interval at every stated point, keeping each point its own atom.
fn scalar_atoms(low: u64, high: u64, cuts: &BTreeSet<u64>) -> Vec<Atom> {
    let points: Vec<u64> = cuts
        .iter()
        .copied()
        .filter(|point| (low..=high).contains(point))
        .collect();
    let mut cut = Vec::new();
    let mut cursor = low;
    for point in points {
        if point > cursor {
            cut.push(Atom::Scalars {
                low: cursor,
                high: point - 1,
            });
        }
        cut.push(Atom::Scalars {
            low: point,
            high: point,
        });
        if point == u64::MAX {
            return cut;
        }
        cursor = point + 1;
    }
    if cursor <= high {
        cut.push(Atom::Scalars { low: cursor, high });
    }
    cut
}

/// Whether a guard admits every value of one cell.
pub(super) fn admits_cell(
    constraints: &BTreeMap<SpecializationAxis, AxisConstraint>,
    cell: &BTreeMap<&SpecializationAxis, Atom>,
) -> bool {
    constraints.iter().all(|(axis, constraint)| {
        cell.get(axis)
            .is_some_and(|atom| constraint.admits_atom(*atom))
    })
}

/// A value from every axis of one cell, for a diagnostic that names the gap.
pub(super) fn cell_witness(cell: &BTreeMap<&SpecializationAxis, Atom>) -> String {
    cell.iter()
        .map(|(axis, atom)| match atom.witness() {
            AxisValue::Scalar(scalar) => format!("{axis}={scalar}"),
            AxisValue::Identity(digest) => {
                format!("{axis}=identity:{:02x}{:02x}", digest.0[0], digest.0[1])
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Collect every bound a guard set contributes to the atom cut of each axis.
pub(super) fn breakpoints(
    guards: &[BTreeMap<SpecializationAxis, AxisConstraint>],
) -> BTreeMap<SpecializationAxis, (BTreeSet<u64>, BTreeSet<Digest>)> {
    let mut cuts: BTreeMap<SpecializationAxis, (BTreeSet<u64>, BTreeSet<Digest>)> = BTreeMap::new();
    for guard in guards {
        for (axis, constraint) in guard {
            let entry = cuts.entry(axis.clone()).or_default();
            constraint.breakpoints(&mut entry.0, &mut entry.1);
        }
    }
    cuts
}

/// Two guards that can meet and cannot be separated by precedence.
pub(super) fn unresolved_overlap(
    left: &BTreeMap<SpecializationAxis, AxisConstraint>,
    right: &BTreeMap<SpecializationAxis, AxisConstraint>,
) -> bool {
    !left.iter().any(|(axis, constraint)| {
        right
            .get(axis)
            .is_some_and(|other| constraint.provably_disjoint(other))
    })
}
