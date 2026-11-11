//! Alternatives derived from declared laws, not from an anticipated recipe.
//!
//! An optimizer built from per-operation rewrite sequences finds the
//! reformulations somebody wrote down. This module inverts that: it reads the
//! laws a combine declares in [`crate::algebraic_law_registry`], turns each law
//! that states an equality between two region graphs into a rewrite, and runs
//! the resulting set to a bounded fixed point over an e-graph mirror of the
//! expression. A multi-step alternative is the composition of two declared
//! laws, and no code names the shape it produces.
//!
//! Nothing here executes the program. The derivation reads IR structure and
//! literal values and adds equalities; the host never evaluates a buffer.
//!
//! # Numerical contract
//!
//! The law id a combine's laws are registered under carries the element's
//! exactness, so a rounding element type reads a different law set: the
//! registry declares commutativity for a rounding add and declines
//! associativity, because two orders of the same addends round differently.
//! Passing `exact = false` therefore derives no reassociation, which is the
//! numerical contract being enforced by the law vocabulary rather than by a
//! special case.
//!
//! # What this does not do
//!
//! The mirror decomposes binary operators and interns everything else as an
//! opaque leaf, so a law about a unary operator or about a companion operator
//! named by op id derives nothing.
//! [`law_derivation`](crate::optimizer::law_saturation::law_derivation) records
//! that refusal per law, and the closure suite holds it to the declared law
//! set.

mod apply;
mod expr_lang;
mod law_rule;

pub use apply::{saturate_laws, LawSaturationReport, LawSaturationStop};
pub use expr_lang::{ExprLang, ExprMirror};
pub use law_rule::{
    derived_rewrites, law_derivation, DerivedKinds, DerivedRewrite, DerivedRewriteKind,
    LawDerivation,
};

use crate::ir::{Expr, Program};
use crate::optimizer::eqsat::EGraphError;
use crate::optimizer::rewrite::rewrite_program;

/// Bounds one derivation run stays inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LawSaturationBudget {
    /// Passes over the rewrite set.
    pub iterations: usize,
    /// Class count the mirror may grow to.
    pub classes: usize,
    /// Depth the extracted term may reach.
    pub extraction_depth: usize,
}

impl Default for LawSaturationBudget {
    /// Bounds that terminate on an associative and commutative operator over a
    /// handful of operands, which is the size of expression a scalar rewrite
    /// sees. A caller with a larger region raises them deliberately.
    fn default() -> Self {
        Self {
            iterations: 6,
            classes: 4096,
            extraction_depth: 64,
        }
    }
}

/// One derivation run: the smallest equivalent term and how it was reached.
#[derive(Debug)]
pub struct LawAlternatives {
    /// Smallest equivalent expression the derived equalities admit, when
    /// extraction stayed inside its depth budget.
    pub best: Option<Expr>,
    /// The saturated mirror, for a caller that asks which terms are equivalent.
    pub mirror: ExprMirror,
    /// Telemetry for the run.
    pub report: LawSaturationReport,
}

/// Derive the alternatives the declared laws admit for `expr`.
///
/// `exact` states whether the element type the expression combines is exact.
///
/// # Errors
///
/// Returns the substrate's allocation and class-id errors.
pub fn derive_alternatives(
    expr: &Expr,
    exact: bool,
    budget: LawSaturationBudget,
) -> Result<LawAlternatives, EGraphError> {
    let rewrites = derived_rewrites(exact);
    let mut mirror = ExprMirror::of(expr)?;
    let report = saturate_laws(&mut mirror, &rewrites, budget.iterations, budget.classes)?;
    let best = mirror.extract(budget.extraction_depth)?;
    Ok(LawAlternatives {
        best,
        mirror,
        report,
    })
}

/// One value-level alternative of a whole program.
#[derive(Debug, Clone)]
pub struct ProgramLawAlternative {
    /// The equivalent program.
    pub program: Program,
    /// Rewrite names the laws authorized, in first-application order.
    ///
    /// This is the derivation evidence a candidate carries. A ranked
    /// alternative that cannot name the laws it came from is indistinguishable
    /// from a hand-written recipe.
    pub chain: Vec<&'static str>,
}

/// Derive the value-level alternative the declared laws admit for `program`.
///
/// Every expression the program states is offered to
/// [`derive_alternatives`], and an expression whose derived term is smaller
/// than the one written is replaced by it. `None` when no expression changed,
/// so a caller ranks an alternative only where one exists.
///
/// `exact` states whether the element type the program combines is exact, and
/// is the numerical permission: a rounding element derives no reassociation
/// because the law registry declines to declare it.
///
/// # Errors
///
/// Returns the substrate's allocation and class-id errors from the first
/// expression whose mirror could not be built.
pub fn derive_program_alternative(
    program: &Program,
    exact: bool,
    budget: LawSaturationBudget,
) -> Result<Option<ProgramLawAlternative>, EGraphError> {
    let mut chain: Vec<&'static str> = Vec::new();
    let mut failure: Option<EGraphError> = None;
    let (rewritten, changed) = rewrite_program(program.clone(), |expr| {
        if failure.is_some() {
            return None;
        }
        match derive_alternatives(expr, exact, budget) {
            Ok(alternatives) => {
                let best = alternatives.best.filter(|best| best != expr)?;
                for name in alternatives.report.applied_rewrites {
                    if !chain.contains(&name) {
                        chain.push(name);
                    }
                }
                Some(best)
            }
            Err(error) => {
                failure = Some(error);
                None
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    if !changed {
        return Ok(None);
    }
    Ok(Some(ProgramLawAlternative {
        program: rewritten,
        chain,
    }))
}
