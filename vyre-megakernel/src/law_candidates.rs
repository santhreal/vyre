//! Candidates the declared laws authorize.
//!
//! Two derivations ship in the optimizer and, until this module, had no
//! consumer. `derive_program_alternative` reads the laws a combine declares and
//! rewrites the expressions of one program; `derive_region_alternatives`
//! composes the region law families into equivalent programs and retains the
//! chain of law names each composition was reached through. Candidate search
//! built its set from the schedule grammar alone, so a declared law authorized
//! nothing any selection ranked.
//!
//! One alternative here is one graph node's program replaced by an equivalent
//! one. Grouping, launch width and topology stay the grammar's business: a law
//! changes what a node computes with, not how it is scheduled. An alternative
//! therefore differs from the baseline in one node's measurements, and carries
//! the law chain that produced them as its derivation evidence.
//!
//! # Bounds
//!
//! Both derivations run under the caller's budget, once per graph node, before
//! any candidate is expanded. Nothing here grows with the candidate count.
//!
//! # Numerical permission
//!
//! A law whose rewrite declares a numerical contract the request did not grant
//! is never applied, so a request asking for the exact result ranks only
//! bit-exact alternatives. Region-law grants come from the request, which is
//! what a caller states. The value-level law set composes the request with the
//! element type: an integer combine is exact under any error budget, and a
//! rounding combine reads the exact set only where the request grants
//! reassociation.

use vyre_foundation::algebraic_reordering::every_declared_type_is_exact;
use vyre_foundation::ir::Program;
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::numeric::{Approximation, ErrorMeasure, NumericContract, Reassociation};
use vyre_foundation::optimizer::law_saturation::{derive_program_alternative, LawSaturationBudget};
use vyre_foundation::optimizer::region_law::{
    derive_region_alternatives, RegionDerivationBudget, RegionDerivationStop,
};
use vyre_foundation::optimizer::rewrite_contract::NumericalContract;

use crate::facts::{measure_node, NodeMeasurement};

/// One law-derived alternative of one graph node.
pub(crate) struct LawAlternative {
    /// Index of the node whose program the laws rewrote.
    pub(crate) node: usize,
    /// Law and rewrite names in application order.
    ///
    /// A region alternative states the law names its chain composed; a
    /// value-level alternative states the rewrite names the declared laws
    /// authorized. Both are the evidence the candidate carries.
    pub(crate) chain: Vec<String>,
    /// Measurements of the rewritten program.
    pub(crate) measured: NodeMeasurement,
}

/// Bounds one law-derivation pass over a graph stays inside.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LawDerivationBudget {
    /// Bounds the value-level expression derivation.
    pub(crate) value: LawSaturationBudget,
    /// Bounds the region-level law composition.
    pub(crate) region: RegionDerivationBudget,
}

impl Default for LawDerivationBudget {
    /// The bounds each derivation states for itself.
    ///
    /// Neither owner's default is restated here: a second copy of a bound is
    /// how a compile silently stops honoring the one that was tuned.
    fn default() -> Self {
        Self {
            value: LawSaturationBudget::default(),
            region: RegionDerivationBudget::default(),
        }
    }
}

/// What one law-derivation pass produced.
pub(crate) struct LawDerivation {
    /// Alternatives in node order, then in derivation order within a node.
    pub(crate) alternatives: Vec<LawAlternative>,
    /// Whether a bound stopped a derivation before its laws were exhausted.
    pub(crate) budget_reached: bool,
}

/// Why a law derivation could not run.
///
/// A derivation failure is a compiler fault, not a property of the program: the
/// expression substrate ran out of class ids, or the registered pass set cannot
/// be ordered. Returning it is what keeps a silently empty alternative set from
/// looking like a graph no law matches.
#[derive(Debug)]
pub(crate) enum LawDerivationError {
    /// The expression mirror could not be built or saturated.
    Value(String),
    /// The registered pass set the region laws cite cannot be ordered.
    Region(String),
}

/// The numerical contracts `numeric` grants a law.
///
/// `BitExact` is absent because the derivation admits every bit-exact law
/// unconditionally; listing it here would state a permission that is not the
/// caller's to give. `IntegerWrapping` is granted unconditionally because it
/// states that integer results are identical, wrapping included.
fn granted_contracts(numeric: Option<NumericContract>) -> Vec<NumericalContract> {
    let mut grants = vec![NumericalContract::IntegerWrapping];
    let Some(numeric) = numeric else {
        return grants;
    };
    if matches!(numeric.reassociation, Reassociation::WithinBudget) {
        grants.push(NumericalContract::FloatReassociation);
    }
    if !matches!(numeric.measure, ErrorMeasure::Exact) {
        grants.push(NumericalContract::FloatContraction);
    }
    if matches!(numeric.approximation, Approximation::Native { .. }) {
        grants.push(NumericalContract::ReducedPrecision);
    }
    grants
}

/// Whether the value-level derivation reads the exact law set for `program`.
///
/// Two facts compose here. Exactness is a property of the element type, so an
/// integer combine reads the exact set whatever error budget the request
/// states. A rounding element reads it only where the request grants
/// reassociation within a budget, which is the caller's permission to reorder a
/// rounding combine. Reading the request alone would withdraw legal integer
/// rewrites from a caller who granted more than bit-exactness.
fn derives_exact_laws(program: &Program, numeric: Option<NumericContract>) -> bool {
    every_declared_type_is_exact(program)
        || numeric
            .is_some_and(|numeric| matches!(numeric.reassociation, Reassociation::WithinBudget))
}

/// Derive every law-authorized alternative for every node of `logical`.
///
/// # Errors
///
/// Returns the substrate error the expression derivation reports, or the
/// scheduling error the pass registry reports for the region laws.
pub(crate) fn derive_law_alternatives(
    logical: &LogicalProgramGraph<'_>,
    numeric: Option<NumericContract>,
    budget: LawDerivationBudget,
) -> Result<LawDerivation, LawDerivationError> {
    let grants = granted_contracts(numeric);
    let mut alternatives = Vec::new();
    let mut budget_reached = false;

    for (index, node) in logical.graph().nodes().iter().enumerate() {
        let region = logical.region(node.id);

        let exact = derives_exact_laws(&node.program, numeric);
        let value = derive_program_alternative(&node.program, exact, budget.value)
            .map_err(|error| LawDerivationError::Value(error.to_string()))?;
        if let Some(value) = value {
            if !value.chain.is_empty() {
                alternatives.push(LawAlternative {
                    node: index,
                    chain: value.chain.iter().map(|name| (*name).to_owned()).collect(),
                    measured: measure_node(&value.program, region),
                });
            }
        }

        let derived = derive_region_alternatives(&node.program, &grants, budget.region)
            .map_err(|error| LawDerivationError::Region(error.to_string()))?;
        if derived.stop != RegionDerivationStop::Saturated {
            budget_reached = true;
        }
        for alternative in &derived.alternatives {
            alternatives.push(LawAlternative {
                node: index,
                chain: alternative
                    .chain
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect(),
                measured: measure_node(&alternative.program, region),
            });
        }
    }

    Ok(LawDerivation {
        alternatives,
        budget_reached,
    })
}
