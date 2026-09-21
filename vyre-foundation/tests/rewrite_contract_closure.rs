//! The rewrite contract registry is closed over the registered pass set.
//!
//! WHY: a rewrite whose level, preconditions, effects, numerical contract,
//! proof witness, profitability, and expansion bound are not recorded is a
//! rewrite nothing can rank, refuse, or prove, and the failure is silent: the
//! pass runs, candidate search explores it, and no reader can say what
//! authorized it. These tests derive both sides of the comparison at run time
//! from the pass registry and from the solver obligation table, so adding a
//! pass, removing a pass, adding an obligation family, or adding a scheduler
//! phase turns the suite red until a decision is recorded.
//!
//! Not covered here: whether a `Structural` argument is *true*. That is the
//! reviewer's obligation, and the argument is recorded in the registry so a
//! reviewer has one place to read it. Solver-discharged families are proved by
//! the `verify-rewrite-proofs` gate, not by this test.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::optimizer::algebraic_rules::arithmetic_rewrite_proof_contracts;
use vyre_foundation::optimizer::rewrite_contract::{
    contract_for_pass, registered_rewrite_contracts, BoundedExpansion, RewriteEffect,
    RewriteWitness,
};
use vyre_foundation::optimizer::{registered_pass_registrations, PassPhase};
use vyre_spec::IrLevel;

fn registered_pass_names() -> BTreeSet<&'static str> {
    registered_pass_registrations()
        .expect("the pass registry must schedule before any contract can be checked")
        .iter()
        .map(|registration| registration.metadata.name)
        .collect()
}

fn declared_phases() -> BTreeMap<&'static str, PassPhase> {
    registered_pass_registrations()
        .expect("the pass registry must schedule before any contract can be checked")
        .iter()
        .map(|registration| (registration.metadata.name, registration.metadata.phase))
        .collect()
}

#[test]
fn every_registered_pass_declares_exactly_one_rewrite_contract() {
    let registered = registered_pass_names();
    let contracts = registered_rewrite_contracts();
    let declared: BTreeSet<&'static str> = contracts.iter().map(|entry| entry.pass).collect();

    assert_eq!(
        declared.len(),
        contracts.len(),
        "Fix: two rewrite contracts declare the same pass; one shadows the other. Declared \
         passes: {declared:?}"
    );

    let undeclared: Vec<&str> = registered.difference(&declared).copied().collect();
    assert!(
        undeclared.is_empty(),
        "Fix: record a rewrite contract in \
         vyre-foundation/src/optimizer/rewrite_contract/shipped.rs for each of {undeclared:?}, or \
         state its opacity there."
    );

    let orphaned: Vec<&str> = declared.difference(&registered).copied().collect();
    assert!(
        orphaned.is_empty(),
        "Fix: {orphaned:?} declare a rewrite contract but register no pass; delete the contract \
         row with the pass."
    );
}

#[test]
fn contract_for_pass_answers_for_every_registered_pass_and_for_nothing_else() {
    for name in registered_pass_names() {
        assert!(
            contract_for_pass(name).is_some(),
            "Fix: contract_for_pass must answer for registered pass {name}."
        );
    }
    assert!(
        contract_for_pass("no_such_pass_exists").is_none(),
        "Fix: contract_for_pass must answer None for an unregistered name."
    );
}

#[test]
fn every_solver_obligation_family_is_claimed_by_exactly_one_pass() {
    let solver_families: BTreeSet<&'static str> = arithmetic_rewrite_proof_contracts()
        .iter()
        .map(|entry| entry.family)
        .collect();

    let mut claimed: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
    for entry in registered_rewrite_contracts() {
        for family in entry.witness.obligation_families() {
            claimed.entry(family).or_default().push(entry.pass);
        }
    }

    for (family, passes) in &claimed {
        assert!(
            solver_families.contains(family),
            "Fix: {passes:?} claim obligation family {family}, which no row of \
             arithmetic_rewrite_proof_contracts declares; add the obligation or correct the claim."
        );
        assert_eq!(
            passes.len(),
            1,
            "Fix: obligation family {family} is claimed by {passes:?}; one pass fires a family so \
             the gate has one owner to hold to it."
        );
    }

    let unclaimed: Vec<&str> = solver_families
        .iter()
        .filter(|family| !claimed.contains_key(*family))
        .copied()
        .collect();
    assert!(
        unclaimed.is_empty(),
        "Fix: obligation families {unclaimed:?} are discharged by the solver gate but no pass \
         claims them; either a pass witness must name the family or the obligation is dead."
    );
}

#[test]
fn a_witness_naming_an_obligation_names_at_least_one_family() {
    for entry in registered_rewrite_contracts() {
        if let RewriteWitness::Obligation(families) = entry.witness {
            assert!(
                !families.is_empty(),
                "Fix: {} declares an obligation witness that names no family, which discharges \
                 nothing.",
                entry.pass
            );
        }
    }
}

#[test]
fn a_semantic_level_rewrite_states_no_synchronization_effect() {
    for entry in registered_rewrite_contracts() {
        if entry.level.admits_physical_policy() {
            continue;
        }
        assert!(
            !entry.effects.contains(&RewriteEffect::Synchronization),
            "Fix: {} is declared at level {} and moves synchronization; adding, removing, or \
             moving a barrier is schedule-level policy.",
            entry.pass,
            entry.level.name()
        );
    }
}

#[test]
fn a_declared_scheduler_phase_agrees_with_the_declared_level() {
    let phases = declared_phases();
    for entry in registered_rewrite_contracts() {
        let phase = phases
            .get(entry.pass)
            .copied()
            .expect("every contract pass is registered, which the closure test proves");
        let required_physical = match phase {
            // Normalization and value rewrites state no hardware fact.
            PassPhase::Canonicalization
            | PassPhase::ScalarAlgebra
            | PassPhase::FusionCse
            | PassPhase::Cleanup => Some(false),
            // Synchronization and capability specialization are physical policy.
            PassPhase::Sync | PassPhase::Specialization => Some(true),
            // A loop or memory transform is either, depending on whether it
            // states geometry; a pass with no declared phase is not pinned.
            PassPhase::Loop
            | PassPhase::Memory
            | PassPhase::Dataflow
            | PassPhase::Megakernel
            | PassPhase::Unclassified => None,
            other => panic!(
                "Fix: scheduler phase {other:?} has no recorded level rule; record whether a pass \
                 in that phase may state physical policy."
            ),
        };
        if let Some(required) = required_physical {
            assert_eq!(
                entry.level.admits_physical_policy(),
                required,
                "Fix: {} runs in phase {phase:?} but is declared at level {}.",
                entry.pass,
                entry.level.name()
            );
        }
    }
}

#[test]
fn an_expansion_bound_admits_its_own_limit_and_refuses_one_node_past_it() {
    assert!(BoundedExpansion::NonGrowing.admits(10, 10));
    assert!(!BoundedExpansion::NonGrowing.admits(10, 11));

    assert!(BoundedExpansion::NodeFactor(3).admits(10, 30));
    assert!(!BoundedExpansion::NodeFactor(3).admits(10, 31));

    assert!(BoundedExpansion::NodeBudget(2).admits(10, 12));
    assert!(!BoundedExpansion::NodeBudget(2).admits(10, 13));

    // An overflowing bound is not a licence to shrink the input away; it means
    // the bound cannot be represented, so nothing is refused by arithmetic.
    assert!(BoundedExpansion::NodeFactor(u32::MAX).admits(usize::MAX, usize::MAX));
    assert!(BoundedExpansion::NodeBudget(u32::MAX).admits(usize::MAX, usize::MAX));
}

#[test]
fn only_a_witness_recording_opacity_is_kept_out_of_candidate_search() {
    assert!(RewriteWitness::Obligation(&["const_fold"]).admits_candidate_search());
    assert!(RewriteWitness::Structural("stated argument").admits_candidate_search());
    assert!(!RewriteWitness::Opaque("no proof recorded").admits_candidate_search());

    assert_eq!(
        RewriteWitness::Obligation(&["const_fold"]).obligation_families(),
        &["const_fold"]
    );
    assert!(RewriteWitness::Structural("stated argument")
        .obligation_families()
        .is_empty());
    assert!(RewriteWitness::Opaque("no proof recorded")
        .obligation_families()
        .is_empty());
}

#[test]
fn a_structural_or_opaque_witness_states_its_argument() {
    for entry in registered_rewrite_contracts() {
        let argument = match entry.witness {
            RewriteWitness::Structural(argument) | RewriteWitness::Opaque(argument) => argument,
            RewriteWitness::Obligation(_) => continue,
            other => panic!(
                "Fix: witness kind {other:?} has no recorded evidence rule; record whether it \
                 states an argument, names an obligation, or is refused by candidate search."
            ),
        };
        assert!(
            argument.split_whitespace().count() >= 4,
            "Fix: {} states the argument {argument:?}, which is too short to be one; state what \
             makes the rewrite value-preserving, or name the obligation family instead.",
            entry.pass
        );
    }
}

#[test]
fn every_declared_level_is_one_the_spec_enumerates() {
    let known: BTreeSet<IrLevel> = IrLevel::all().iter().copied().collect();
    for entry in registered_rewrite_contracts() {
        assert!(
            known.contains(&entry.level),
            "Fix: {} declares level {:?}, which IrLevel::all() does not enumerate.",
            entry.pass,
            entry.level
        );
    }
}
