//! What one graph region does to the values it holds, and what a whole graph
//! proves about its outputs.
//!
//! A region is derived from the graph, not declared by a caller: the formats it
//! reads and writes, whether it reduces or carries state, and whether it
//! combines through atomics are facts the graph already states. This turns those
//! facts into a [`NumericContract`], and composes the per-region contracts into
//! the budget the graph's outputs carry.

use super::contract::{
    AtomicOrderSensitivity, ContractRefusal, Determinism, ErrorMeasure, NumericContract,
    Reassociation,
};
use super::format::ScalarFormat;

/// How one region combines the values it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegionArithmetic {
    /// Each output point is computed from its own inputs.
    Pointwise,
    /// Points are combined along one or more axes.
    Reduction {
        /// Upper bound on the values combined into one output point.
        terms: u64,
    },
    /// Each step reads the state the step before it wrote.
    Recurrence {
        /// Upper bound on the steps one submission advances the state.
        steps: u64,
    },
}

/// The graph facts a region contract is derived from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegionNumericFacts {
    /// The widest format the region reads, or `None` when it reads no number.
    pub input: Option<ScalarFormat>,
    /// The format the region writes, or `None` when it writes no number.
    pub output: Option<ScalarFormat>,
    /// How the region combines what it reads.
    pub arithmetic: RegionArithmetic,
    /// Whether the region combines through atomic memory effects.
    pub atomics: bool,
    /// Whether the combine the region performs is associative and commutative.
    pub reorderable: bool,
}

impl RegionNumericFacts {
    /// Facts for a region that holds no number.
    #[must_use]
    pub const fn opaque() -> Self {
        Self {
            input: None,
            output: None,
            arithmetic: RegionArithmetic::Pointwise,
            atomics: false,
            reorderable: false,
        }
    }
}

/// The contract a region carrying `facts` states.
///
/// A region that holds no number moves bits, so it states the exact contract. A
/// region that rounds states one unit in the last place of what it writes, which
/// is the ceiling on a correctly rounded operation, and a narrowing conversion
/// costs a second one because the result is rounded again on the way to storage.
/// A reduction and a recurrence price their step counts, and an atomic combine
/// over a rounding format is run-to-run variable because the landing order is
/// the device's rather than the schedule's.
///
/// # Errors
///
/// Returns the refusal that pricing the region's steps produced, which is
/// [`ContractRefusal::UnboundedMagnitude`] where the measure cannot be read as a
/// fraction of the exact magnitude.
pub fn region_contract(facts: &RegionNumericFacts) -> Result<NumericContract, ContractRefusal> {
    let Some(output) = facts.output.or(facts.input) else {
        return Ok(NumericContract::EXACT);
    };
    let intermediate = facts.input.unwrap_or(output);
    let mut contract = NumericContract::of(output).computing_in(intermediate);
    if !output.is_exact() {
        let rounds = 1 + u32::from(intermediate != output);
        contract = contract.within_ulp(rounds);
        contract = contract.reassociating(if facts.reorderable {
            Reassociation::WithinBudget
        } else {
            Reassociation::Forbidden
        });
    }
    if facts.atomics && !output.is_exact() {
        contract = contract
            .under(Determinism::RunToRunVariable)
            .sensitive_to(AtomicOrderSensitivity::Sensitive);
    }
    match facts.arithmetic {
        RegionArithmetic::Pointwise => Ok(contract),
        RegionArithmetic::Reduction { terms } => contract.over_reduction(saturating_count(terms)),
        RegionArithmetic::Recurrence { steps } => contract.over_recurrence(saturating_count(steps)),
    }
}

/// The contract a value carries after every region in `regions` has run.
///
/// The regions are composed in the order given, which is the order the graph
/// states, so the budget is the one an output carries rather than the widest
/// single region.
///
/// # Errors
///
/// Returns the first refusal composition produced: a region reading a format the
/// region before it does not produce, or an absolute bound meeting a relative
/// one with no magnitude proof between them.
pub fn graph_budget<'a>(
    regions: impl IntoIterator<Item = &'a NumericContract>,
) -> Result<NumericContract, ContractRefusal> {
    let mut composed: Option<NumericContract> = None;
    for region in regions {
        composed = Some(match composed {
            None => *region,
            Some(previous) => previous.compose(region)?,
        });
    }
    Ok(composed.unwrap_or(NumericContract::EXACT))
}

/// Whether a graph whose regions compose to `proven` stays inside `declared`.
///
/// # Errors
///
/// Returns [`ContractRefusal::BudgetExceeded`] when the composed error is wider
/// than the declared one, and [`ContractRefusal::FormatMismatch`] when the two
/// are stated over different storage formats, because a ULP count over one
/// format is not a count over another.
pub fn budget_admits(
    declared: &NumericContract,
    proven: &NumericContract,
) -> Result<(), ContractRefusal> {
    if declared.storage != proven.storage {
        return Err(ContractRefusal::FormatMismatch {
            first: declared.storage,
            second: proven.storage,
        });
    }
    let measure = match proven.approximation {
        super::contract::Approximation::Refused => proven.measure,
        super::contract::Approximation::Native { measure } => {
            wider_measure(proven.measure, measure)
        }
    };
    declared.admits(&measure)
}

/// Whether `declared` admits combining `terms` values of `contract` in an order
/// the program did not state.
///
/// The new order is a tree over the region's points, so it rounds `log2(n)`
/// times where the stated order rounds `n - 1` times, and both are priced from
/// the region's own contract. Every stage that can introduce a new order asks
/// this: the search when it ranks a reordering production, and the artifact
/// when it selects a route whose workers arrive in an order the schedule does
/// not fix. A budget that cannot be read against the region's format admits
/// nothing, because an unreadable comparison is not a proof.
#[must_use]
pub fn reordering_admitted(
    declared: &NumericContract,
    contract: &NumericContract,
    terms: u32,
) -> bool {
    contract
        .reassociating(Reassociation::WithinBudget)
        .over_reduction(terms)
        .and_then(|reordered| declared.admits(&reordered.measure))
        .is_ok()
}

/// The wider of two measures, read as fractions where they are comparable.
fn wider_measure(left: ErrorMeasure, right: ErrorMeasure) -> ErrorMeasure {
    if right.magnitude() > left.magnitude() {
        right
    } else {
        left
    }
}

/// A bound on steps as a step count, saturating at the count ceiling.
fn saturating_count(bound: u64) -> u32 {
    u32::try_from(bound).unwrap_or(u32::MAX)
}
