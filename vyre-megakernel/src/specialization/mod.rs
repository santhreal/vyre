//! The versioned contract that states what a compile may specialize on.
//!
//! Specialization without a contract is parameter substitution: a caller pins a
//! dimension, the compiler emits one artifact, and nothing records which inputs
//! that artifact is correct for. Two failures follow from the same absence. A
//! consumer can reuse a payload compiled for a shape it no longer has, because
//! no guard states the shape. And a workload that cannot be pinned falls back to
//! one generic schedule for every input, because there is no way to retain
//! several and say which serves what.
//!
//! A contract states the axes and their domains. A guard states what one variant
//! is selected by. Two proofs decide whether a set of guards is usable, and both
//! are computed rather than asserted: no two guards can admit the same facts at
//! the same precedence, and the guards together with the remainder cover every
//! value the domain declares.
//!
//! An axis is a typed fact. Application information reaches the compiler as the
//! configuration digest and as graph identity, so no compiler, backend, artifact
//! or runtime signature carries a caller's naming for a workload.

/// The classes of fact a compiled variant may specialize on.
mod axis;
/// One authenticated container for a whole guarded artifact set.
mod envelope;
/// What a variant is selected by, and the two proofs a guard set must pass.
mod guard;
/// Joint compilation of one guarded variant set and its remainder.
mod portfolio;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub use self::axis::{AxisValue, SpecializationAxis, TargetCapabilityAxis, TargetResourceAxis};
pub use self::envelope::{PortfolioEnvelope, PORTFOLIO_ENVELOPE_SCHEMA_VERSION};
pub use self::guard::{AxisDomain, GuardTerm, VariantGuard, MAX_COVERAGE_CELLS};
pub use self::portfolio::{
    compile_specialized_portfolio, compile_specialized_portfolio_measured, PortfolioVariant,
    SpecializedPortfolio, SpecializedRemainder, MAX_PROPOSED_VARIANTS,
};
use crate::error::{failure, CompilerFailureKind};
use crate::{CompileError, ValidatedCompileRequest};

/// Current canonical specialization contract schema.
pub const SPECIALIZATION_SCHEMA_VERSION: u16 = 1;

/// What serves the part of the domain no guard admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemainderKind {
    /// One correct artifact compiled without any guard's assumptions.
    Generic,
    /// The uncovered part of the domain is declared unsupported and rejected.
    Unsupported,
}

impl RemainderKind {
    /// Every declared remainder kind.
    pub const ALL: &'static [Self] = &[Self::Generic, Self::Unsupported];

    /// Stable identifier used in diagnostics and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Unsupported => "unsupported",
        }
    }
}

/// The axes one compile may specialize on, and the values each admits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecializationContract {
    schema_version: u16,
    #[serde(with = "axis_domain_pairs")]
    axes: BTreeMap<SpecializationAxis, AxisDomain>,
}

impl SpecializationContract {
    /// State a contract over declared axes, rejecting a domain that admits
    /// nothing and an axis whose domain is the wrong kind for it.
    ///
    /// # Errors
    ///
    /// Returns an error when an axis declares an empty or inverted domain, or
    /// when a content-identity axis declares scalars, or a scalar axis declares
    /// identities.
    pub fn new(axes: BTreeMap<SpecializationAxis, AxisDomain>) -> Result<Self, CompileError> {
        if axes.is_empty() {
            return Err(failure(
                CompilerFailureKind::InvalidSpecializationContract,
                "specialization.axes",
                "a contract declares no axis",
                "declare at least one axis, or compile one artifact without a contract",
            ));
        }
        for (axis, domain) in &axes {
            domain.validate(axis)?;
            if axis.is_identity_axis() != domain.is_identity_domain() {
                return Err(failure(
                    CompilerFailureKind::InvalidSpecializationContract,
                    axis.field(),
                    "axis and domain disagree on whether values are content identities",
                    "declare identities for a constant-identity axis and scalars for every other",
                ));
            }
        }
        Ok(Self {
            schema_version: SPECIALIZATION_SCHEMA_VERSION,
            axes,
        })
    }

    /// Schema this contract was stated under.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// The declared axes and their domains.
    #[must_use]
    pub const fn axes(&self) -> &BTreeMap<SpecializationAxis, AxisDomain> {
        &self.axes
    }

    /// Reject a contract whose axes name facts this request cannot produce.
    ///
    /// A dimension the graph does not declare, or a constant the request states
    /// no identity for, cannot be read at selection time, so a guard over it
    /// would admit nothing and the variant would be dead bytes.
    ///
    /// # Errors
    ///
    /// Returns an error naming the axis and the fact the request is missing.
    pub fn validate_for(&self, request: &ValidatedCompileRequest) -> Result<(), CompileError> {
        let graph_values: BTreeSet<u32> = request
            .graph()
            .values()
            .iter()
            .map(|value| value.id.0)
            .collect();
        for axis in self.axes.keys() {
            match axis {
                SpecializationAxis::SymbolicDimension { dimension } => {
                    if !request.facts().symbolic_bindings.contains_key(dimension) {
                        return Err(missing_fact(
                            axis,
                            format!("the graph declares no symbolic dimension `{dimension}`"),
                            "declare the dimension in the graph, or drop the axis",
                        ));
                    }
                }
                SpecializationAxis::ConstantIdentity { value } => {
                    let stated = request
                        .facts()
                        .constant_identities
                        .keys()
                        .any(|id| id.0 == *value);
                    if !stated {
                        return Err(missing_fact(
                            axis,
                            format!("the request states no content identity for value {value}"),
                            "supply the constant identity with the request, or drop the axis",
                        ));
                    }
                }
                SpecializationAxis::ValueLayout { value }
                | SpecializationAxis::ValueDensity { value } => {
                    if !graph_values.contains(value) {
                        return Err(missing_fact(
                            axis,
                            format!("the graph holds no value {value}"),
                            "name a value the graph declares, or drop the axis",
                        ));
                    }
                }
                SpecializationAxis::RetainedState
                | SpecializationAxis::LaunchBatch
                | SpecializationAxis::Concurrency
                | SpecializationAxis::TargetCapability { .. }
                | SpecializationAxis::TargetResource { .. } => {}
            }
        }
        Ok(())
    }

    /// Reject a guard that reads an axis the contract does not declare, or
    /// states values the axis domain does not admit.
    ///
    /// # Errors
    ///
    /// Returns an error naming the axis and what the guard states outside it.
    pub fn validate_guard(&self, guard: &VariantGuard) -> Result<(), CompileError> {
        for axis in guard.axes() {
            if !self.axes.contains_key(axis) {
                return Err(failure(
                    CompilerFailureKind::InvalidVariantGuard,
                    axis.field(),
                    "guard reads an axis the contract does not declare",
                    "declare the axis in the contract, or drop the term",
                ));
            }
        }
        let normalized = guard.normalize()?;
        for (axis, constraint) in &normalized {
            let domain = &self.axes[axis];
            if !constraint.intersects_domain(domain) {
                return Err(failure(
                    CompilerFailureKind::InvalidVariantGuard,
                    axis.field(),
                    "guard admits no value the declared domain holds",
                    "state a term inside the declared domain, or widen the domain",
                ));
            }
        }
        Ok(())
    }

    /// Reject facts the contract declares no domain for.
    ///
    /// Selection answers with an artifact compiled for the declared domain, so a
    /// stated value outside it is served by no member: the remainder is compiled
    /// for the domain too, and answering with it would hand back a schedule that
    /// launches over fewer points than the workload holds. An unstated axis is
    /// the same refusal from the other side, because a guard that cannot read it
    /// falls through to the remainder for a fact nothing checked.
    ///
    /// # Errors
    ///
    /// Returns an error naming the axis the facts leave unstated, state outside
    /// the declared domain, or name without declaration.
    pub fn admits_facts(
        &self,
        facts: &BTreeMap<SpecializationAxis, AxisValue>,
    ) -> Result<(), CompileError> {
        for axis in facts.keys() {
            if !self.axes.contains_key(axis) {
                return Err(failure(
                    CompilerFailureKind::UnsupportedWorkload,
                    axis.field(),
                    "stated facts name an axis the contract does not declare",
                    "state facts over the declared axes only, or compile a set whose contract declares it",
                ));
            }
        }
        for (axis, domain) in &self.axes {
            let Some(value) = facts.get(axis) else {
                return Err(failure(
                    CompilerFailureKind::UnsupportedWorkload,
                    axis.field(),
                    "stated facts leave a declared axis unstated",
                    "state one value for every axis the contract declares",
                ));
            };
            if !domain.admits(*value) {
                return Err(failure(
                    CompilerFailureKind::UnsupportedWorkload,
                    axis.field(),
                    "stated fact is outside the domain the contract declares",
                    "state a value the declared domain holds, or compile a set over a domain that holds it",
                ));
            }
        }
        Ok(())
    }

    /// Prove that a guard set is unambiguous and complete.
    ///
    /// # Errors
    ///
    /// Returns an error when two guards can admit the same facts at one
    /// precedence, when the atom product exceeds [`MAX_COVERAGE_CELLS`], or when
    /// a cell is admitted by no guard and the remainder is
    /// [`RemainderKind::Unsupported`].
    pub fn prove(
        &self,
        guards: &[VariantGuard],
        remainder: RemainderKind,
    ) -> Result<CoverageProof, CompileError> {
        for guard in guards {
            self.validate_guard(guard)?;
        }
        let normalized = guards
            .iter()
            .map(VariantGuard::normalize)
            .collect::<Result<Vec<_>, _>>()?;
        for (left, right) in ordered_pairs(guards.len()) {
            if guards[left].precedence() == guards[right].precedence()
                && guard::unresolved_overlap(&normalized[left], &normalized[right])
            {
                return Err(failure(
                    CompilerFailureKind::GuardOverlap,
                    format!("specialization.variants[{left}].guard"),
                    format!(
                        "guards {left} and {right} can admit the same facts at precedence {}",
                        guards[left].precedence()
                    ),
                    "separate the guards on some axis, or give one a distinct precedence",
                ));
            }
        }
        let cuts = guard::breakpoints(&normalized);
        let empty = (BTreeSet::new(), BTreeSet::new());
        let mut axes = Vec::with_capacity(self.axes.len());
        let mut cells = 1_usize;
        for (axis, domain) in &self.axes {
            let (scalars, identities) = cuts.get(axis).unwrap_or(&empty);
            let atoms = guard::atoms(domain, scalars, identities);
            cells = cells.saturating_mul(atoms.len());
            if cells > MAX_COVERAGE_CELLS {
                return Err(failure(
                    CompilerFailureKind::GuardCoverageGap,
                    "specialization.axes",
                    format!(
                        "the declared domain and stated guards cut more than {MAX_COVERAGE_CELLS} cells"
                    ),
                    "narrow an axis domain or state fewer guard bounds so coverage stays decidable",
                ));
            }
            axes.push((axis, atoms));
        }
        let order = precedence_order(guards);
        let mut covered = 0_usize;
        let mut gaps = 0_usize;
        let mut served = vec![0_usize; guards.len()];
        let mut first_gap = None;
        for indices in odometer(&axes) {
            let cell: BTreeMap<&SpecializationAxis, guard::Atom> = axes
                .iter()
                .zip(&indices)
                .map(|((axis, atoms), index)| (*axis, atoms[*index]))
                .collect();
            match order
                .iter()
                .copied()
                .find(|index| guard::admits_cell(&normalized[*index], &cell))
            {
                Some(index) => {
                    covered += 1;
                    served[index] += 1;
                }
                None => {
                    gaps += 1;
                    if first_gap.is_none() {
                        first_gap = Some(guard::cell_witness(&cell));
                    }
                }
            }
        }
        if let (RemainderKind::Unsupported, Some(witness)) = (remainder, first_gap.as_ref()) {
            return Err(failure(
                CompilerFailureKind::GuardCoverageGap,
                "specialization.remainder",
                format!("no guard admits {witness} and the remainder is declared unsupported"),
                "state a guard covering it, or compile a generic remainder",
            ));
        }
        Ok(CoverageProof {
            cells: covered + gaps,
            covered,
            gaps,
            served,
            remainder,
            first_gap,
        })
    }
}

/// What a proof of one guard set established.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverageProof {
    cells: usize,
    covered: usize,
    gaps: usize,
    served: Vec<usize>,
    remainder: RemainderKind,
    first_gap: Option<String>,
}

impl CoverageProof {
    /// Cells the atom cut produced.
    #[must_use]
    pub const fn cells(&self) -> usize {
        self.cells
    }

    /// Cells some guard admits.
    #[must_use]
    pub const fn covered(&self) -> usize {
        self.covered
    }

    /// Cells only the remainder serves.
    #[must_use]
    pub const fn gaps(&self) -> usize {
        self.gaps
    }

    /// Whether the guards alone cover the declared domain.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.gaps == 0
    }

    /// What serves the cells no guard admits.
    #[must_use]
    pub const fn remainder(&self) -> RemainderKind {
        self.remainder
    }

    /// A value from the first cell no guard admits, in canonical order.
    #[must_use]
    pub fn first_gap(&self) -> Option<&str> {
        self.first_gap.as_deref()
    }

    /// Cells each guard serves, in the order the guards were stated.
    ///
    /// A cell is charged to the first guard in precedence order that admits it,
    /// so this is what each variant would actually run rather than what it could
    /// admit if nothing preceded it. A guard serving no cell buys nothing.
    #[must_use]
    pub fn served(&self) -> &[usize] {
        &self.served
    }
}

fn missing_fact(axis: &SpecializationAxis, message: String, fix: &'static str) -> CompileError {
    failure(
        CompilerFailureKind::InvalidSpecializationContract,
        axis.field(),
        message,
        fix,
    )
}

/// Every unordered pair of distinct indices below `len`.
fn ordered_pairs(len: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..len).flat_map(move |left| ((left + 1)..len).map(move |right| (left, right)))
}

/// Every index tuple of the atom product, in canonical axis order.
fn odometer(axes: &[(&SpecializationAxis, Vec<guard::Atom>)]) -> Vec<Vec<usize>> {
    let mut tuples = vec![Vec::with_capacity(axes.len())];
    for (_, atoms) in axes {
        let mut next = Vec::with_capacity(tuples.len() * atoms.len());
        for prefix in &tuples {
            for index in 0..atoms.len() {
                let mut tuple = prefix.clone();
                tuple.push(index);
                next.push(tuple);
            }
        }
        tuples = next;
    }
    tuples
}

/// Guard indices in the order selection evaluates them.
///
/// Precedence decides first and the canonical guard order breaks the tie, so a
/// set of guards has one evaluation order whatever order the caller stated them
/// in.
pub(super) fn precedence_order(guards: &[VariantGuard]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..guards.len()).collect();
    order.sort_by(|left, right| {
        (guards[*left].precedence(), &guards[*left])
            .cmp(&(guards[*right].precedence(), &guards[*right]))
    });
    order
}

/// Serialize declared axes as ordered pairs.
///
/// An axis is a structured fact, not a name, so it cannot be a JSON object key.
/// The pair sequence keeps the canonical axis order the map already holds, which
/// is what makes the encoded body byte-comparable.
mod axis_domain_pairs {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{AxisDomain, SpecializationAxis};

    pub(super) fn serialize<S: Serializer>(
        axes: &BTreeMap<SpecializationAxis, AxisDomain>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        axes.iter().collect::<Vec<_>>().serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<SpecializationAxis, AxisDomain>, D::Error> {
        let pairs = Vec::<(SpecializationAxis, AxisDomain)>::deserialize(deserializer)?;
        Ok(pairs.into_iter().collect())
    }
}
