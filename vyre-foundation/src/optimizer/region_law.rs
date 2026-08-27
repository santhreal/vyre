//! Region alternatives derived from the declarative law families.
//!
//! [`law_saturation`](crate::optimizer::law_saturation) derives value-level alternatives
//! for the algebraic family: it reads the laws a combine declares and runs them
//! over an expression mirror. Four families state equalities that no expression
//! mirror can express, because their subject is a region rather than a value:
//! recurrence structure, reduction structure, access layout, and numerical
//! reformulation.
//!
//! A law here states an equality between two region graphs and names the
//! registered rewrite that realizes it. Naming the rewrite is what keeps this a
//! derivation rather than a second implementation: the transform already ships,
//! already carries a rewrite contract, and is already proved by its own suite.
//! What no table stated is which declarative law authorizes it, so a family
//! could ship with no derivation at all and every reader would read the pass
//! list and agree.
//!
//! # What is derived
//!
//! [`derive_region_alternatives`](crate::optimizer::region_law::derive_region_alternatives)
//! composes law rows to a bounded fixed point.
//! Each step applies one row's rewrite to a program and keeps the result when
//! it differs, so a two-law alternative is the composition of two declared
//! laws and no code names the shape it produces. The chain of law names is
//! retained per alternative as the derivation evidence a candidate carries.
//!
//! Nothing here executes the program. Every step is an IR-to-IR rewrite that
//! already runs inside the host compiler.
//!
//! # Numerical permission
//!
//! A row in a family that admits a value difference is derived only when the
//! caller grants the numerical contract its rewrite declares. The grant is per
//! contract rather than per family, because the contract is the fact a rewrite
//! states about itself and the family is the vocabulary the law is cited from.

use rustc_hash::FxHashSet;
use vyre_spec::RegionLawFamily;

use crate::ir::Program;
use crate::optimizer::rewrite_contract::{contract_for_pass, NumericalContract};
use crate::optimizer::{registered_pass_registrations, OptimizerError};

/// One declarative law and the registered rewrite that realizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionLaw {
    /// Family the law is cited from.
    pub family: RegionLawFamily,
    /// Stable law name, used in derivation evidence and reports.
    pub name: &'static str,
    /// The equality this law states, quantified over every region it matches.
    pub statement: &'static str,
    /// Registered pass whose rewrite realizes the equality.
    pub realized_by: &'static str,
}

const fn law(
    family: RegionLawFamily,
    name: &'static str,
    statement: &'static str,
    realized_by: &'static str,
) -> RegionLaw {
    RegionLaw {
        family,
        name,
        statement,
        realized_by,
    }
}

/// Every declarative law a region derivation may cite.
///
/// One row per equality, not one row per pass: a rewrite that realizes two
/// equalities appears twice, and two rewrites that realize the same equality
/// under different preconditions appear as two rows of the same law name only
/// when their statements agree.
pub const REGION_LAWS: &[RegionLaw] = &[
    law(
        RegionLawFamily::Algebraic,
        "canonical_operand_order",
        "a region whose operator applications are ordered by the canonical form computes what the same region computes in any admitted order",
        "canonicalize",
    ),
    law(
        RegionLawFamily::Algebraic,
        "literal_evaluation",
        "an operator application over literal operands equals the literal its arguments determine",
        "const_fold",
    ),
    law(
        RegionLawFamily::Recurrence,
        "peel_boundary_iteration",
        "a counted recurrence over a range equals its boundary iteration followed by the recurrence over the remaining range",
        "loop_peel",
    ),
    law(
        RegionLawFamily::Recurrence,
        "unroll_counted_recurrence",
        "a counted recurrence over a constant range equals the sequence of its iterations at the indices that range names",
        "loop_unroll",
    ),
    law(
        RegionLawFamily::Recurrence,
        "stage_shift_recurrence",
        "a recurrence whose body reads a value produced by an earlier iteration equals the same recurrence with that production shifted one stage earlier",
        "loop_software_pipeline",
    ),
    law(
        RegionLawFamily::Reduction,
        "split_partial_reduction",
        "a region combining values over one domain equals the same combines split into separate regions over that domain when no combine reads another's result",
        "loop_fission",
    ),
    law(
        RegionLawFamily::Reduction,
        "join_partial_reductions",
        "two regions combining values over the same domain equal one region containing both combines when neither reads the other's result",
        "loop_fusion",
    ),
    law(
        RegionLawFamily::Layout,
        "tile_index_space",
        "a region over a range equals a region over blocks of that range containing a region over each block, which reorders no access within a block",
        "loop_strip_mine",
    ),
    law(
        RegionLawFamily::Layout,
        "hoist_invariant_read",
        "a read whose address does not depend on the iteration index equals the same read performed once before the region",
        "read_only_load_hoist",
    ),
    law(
        RegionLawFamily::Layout,
        "order_storage_declarations",
        "a region's storage declarations may be held in any order, because a declaration moves no value",
        "buffer_decl_sort",
    ),
    law(
        RegionLawFamily::Numerical,
        "substitute_cheaper_operator",
        "an operator application equals a cheaper application over the same operands under the numerical contract the rewrite declares",
        "strength_reduce",
    ),
    law(
        RegionLawFamily::Numerical,
        "shift_index_origin",
        "a counted recurrence equals the same recurrence over a range starting at zero with its index shifted, under wrapping index arithmetic",
        "loop_lower_bound_normalize",
    ),
    law(
        RegionLawFamily::Numerical,
        "fold_index_range",
        "an index expression bounded by a counted range equals the value that range determines, under wrapping index arithmetic",
        "loop_var_range_fold",
    ),
];

/// Every law cited from `family`.
#[must_use]
pub fn laws_for_family(family: RegionLawFamily) -> Vec<&'static RegionLaw> {
    REGION_LAWS
        .iter()
        .filter(|law| law.family == family)
        .collect()
}

/// The law named `name`, if one is declared.
#[must_use]
pub fn region_law(name: &str) -> Option<&'static RegionLaw> {
    REGION_LAWS.iter().find(|law| law.name == name)
}

/// The numerical contract a law's rewrite declares.
///
/// `None` when the law names a pass with no declared contract, which the
/// closure suite rejects rather than deriving from.
#[must_use]
pub fn law_numerical_contract(law: &RegionLaw) -> Option<NumericalContract> {
    contract_for_pass(law.realized_by).map(|contract| contract.numerical)
}

/// Bounds one region derivation run stays inside.
///
/// Both bounds are the caller's. A derivation that composes laws without a
/// depth bound is how a bounded compile stops being bounded, and the
/// alternative count is what a caller ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionDerivationBudget {
    /// Longest chain of laws one alternative may be derived through.
    pub max_depth: usize,
    /// Most alternatives one run may return.
    pub max_alternatives: usize,
}

impl Default for RegionDerivationBudget {
    /// Two-law chains and eight alternatives: enough for a composition no
    /// single law produces, small enough to run inside a compile.
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_alternatives: 8,
        }
    }
}

/// Why a derivation run stopped.
///
/// A stop reason is part of the budget contract: a caller that cannot tell
/// exhaustion from saturation cannot tell a complete answer from a truncated
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionDerivationStop {
    /// Every admitted law was applied to every derived region and produced
    /// nothing new.
    Saturated,
    /// The depth bound was reached with laws still to compose.
    DepthReached,
    /// The alternative bound was reached.
    AlternativeLimit,
}

/// One derived region and the laws it was derived through.
#[derive(Debug, Clone)]
pub struct DerivedRegion {
    /// The equivalent program.
    pub program: Program,
    /// Law names in application order. Length is the derivation depth.
    pub chain: Vec<&'static str>,
}

/// What one derivation run produced.
#[derive(Debug, Clone)]
pub struct RegionDerivation {
    /// Alternatives, excluding the input program.
    pub alternatives: Vec<DerivedRegion>,
    /// Why the run stopped.
    pub stop: RegionDerivationStop,
}

impl RegionDerivation {
    /// Alternatives derived through at least two laws.
    #[must_use]
    pub fn composed(&self) -> Vec<&DerivedRegion> {
        self.alternatives
            .iter()
            .filter(|derived| derived.chain.len() > 1)
            .collect()
    }

    /// Every law name that contributed to an alternative.
    #[must_use]
    pub fn cited_laws(&self) -> Vec<&'static str> {
        let mut cited: Vec<&'static str> = self
            .alternatives
            .iter()
            .flat_map(|derived| derived.chain.iter().copied())
            .collect();
        cited.sort_unstable();
        cited.dedup();
        cited
    }
}

/// Laws admitted for a run: every bit-exact law, plus the value-changing laws
/// whose declared contract the caller granted.
fn admitted_laws(grants: &[NumericalContract]) -> Vec<&'static RegionLaw> {
    REGION_LAWS
        .iter()
        .filter(|law| match law_numerical_contract(law) {
            Some(NumericalContract::BitExact) => true,
            Some(contract) => grants.contains(&contract),
            None => false,
        })
        .collect()
}

/// Apply one law's rewrite to `program`, returning the result when it differs.
fn apply_law(law: &RegionLaw, program: &Program) -> Result<Option<Program>, OptimizerError> {
    let registrations = registered_pass_registrations()?;
    let Some(registration) = registrations
        .iter()
        .find(|registration| registration.metadata.name == law.realized_by)
    else {
        return Ok(None);
    };
    let pass = (registration.factory)();
    let result = pass.transform(program.clone());
    Ok(result.changed.then_some(result.program))
}

/// Derive the region alternatives the declared laws admit for `program`.
///
/// `grants` names the numerical contracts the caller admits. A law whose
/// rewrite declares a contract outside that set is not applied, so a caller
/// asking for bit-exact alternatives receives only bit-exact ones.
///
/// # Errors
///
/// Returns the scheduling error the pass registry reports when the registered
/// pass set cannot be ordered.
pub fn derive_region_alternatives(
    program: &Program,
    grants: &[NumericalContract],
    budget: RegionDerivationBudget,
) -> Result<RegionDerivation, OptimizerError> {
    if budget.max_alternatives == 0 {
        return Ok(RegionDerivation {
            alternatives: Vec::new(),
            stop: RegionDerivationStop::AlternativeLimit,
        });
    }
    let laws = admitted_laws(grants);
    let mut seen: FxHashSet<[u8; 32]> = FxHashSet::default();
    seen.insert(program.fingerprint());

    let mut alternatives: Vec<DerivedRegion> = Vec::new();
    let mut frontier: Vec<DerivedRegion> = vec![DerivedRegion {
        program: program.clone(),
        chain: Vec::new(),
    }];
    let mut stop = RegionDerivationStop::Saturated;

    for depth in 0..budget.max_depth {
        if frontier.is_empty() {
            break;
        }
        let mut next: Vec<DerivedRegion> = Vec::new();
        for source in &frontier {
            for law in &laws {
                let Some(derived) = apply_law(law, &source.program)? else {
                    continue;
                };
                if !seen.insert(derived.fingerprint()) {
                    continue;
                }
                let mut chain = source.chain.clone();
                chain.push(law.name);
                let entry = DerivedRegion {
                    program: derived,
                    chain,
                };
                alternatives.push(entry.clone());
                if alternatives.len() >= budget.max_alternatives {
                    return Ok(RegionDerivation {
                        alternatives,
                        stop: RegionDerivationStop::AlternativeLimit,
                    });
                }
                next.push(entry);
            }
        }
        if !next.is_empty() && depth + 1 == budget.max_depth {
            stop = RegionDerivationStop::DepthReached;
        }
        frontier = next;
    }

    Ok(RegionDerivation { alternatives, stop })
}
