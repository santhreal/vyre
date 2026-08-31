//! Every declarative law family derives region alternatives, and only the
//! families that admit a value difference cite value-changing rewrites.
//!
//! WHY this suite exists: the algebraic family had a derivation and the four
//! whose subject is a region had none, so a family could ship as vocabulary
//! nobody derived from and the pass list read the same either way. A family with
//! no law row now turns this suite red, and so does a law row naming a rewrite
//! the registry does not carry.
//!
//! The value-difference rule is checked in both directions against the rewrite
//! contracts, which are the rewrites' own declarations rather than a second
//! opinion recorded here: a law outside the numerical family may not name a
//! rewrite that changes values, and a rewrite that changes values must be cited
//! by a numerical law before a derivation can reach it.
//!
//! What this does NOT catch: whether a law's prose statement is the equality its
//! named rewrite implements. That equality is proved by the rewrite's own suite
//! and by its contract's witness; this suite holds the citation, the family
//! closure, and the numerical permission.

use std::collections::BTreeSet;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::region_law::{
    derive_region_alternatives, law_numerical_contract, laws_for_family, region_law,
    RegionDerivationBudget, RegionDerivationStop, REGION_LAWS,
};
use vyre_foundation::optimizer::rewrite_contract::{
    contract_for_pass, registered_rewrite_contracts, NumericalContract,
};
use vyre_spec::RegionLawFamily;

/// The `RegionLawFamily` variant names `vyre-spec` declares, read from source.
///
/// Parsed rather than read from `all()`, because a variant added to the enum and
/// omitted from `all()` is the silent hole this closure exists to catch.
fn declared_family_variants() -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_crate_directory("vyre-spec")
        .join("src")
        .join("region_law.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {path:?} to derive the family set: {err}"));
    let body = vyre_test_support::braced_body(&source, "pub enum RegionLawFamily {")
        .unwrap_or_else(|| panic!("Fix: {path:?} no longer declares `pub enum RegionLawFamily`"));
    vyre_test_support::top_level_variant_names(body)
}

/// A counted loop over a non-zero constant range, writing one element per
/// iteration. Constant bounds, a non-zero origin, and a contiguous store are
/// what the recurrence, layout, and numerical rewrites look for.
fn counted_store_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [1, 1, 1],
        vec![Node::loop_(
            "i",
            Expr::u32(1),
            Expr::u32(5),
            vec![Node::store(
                "output",
                Expr::var("i"),
                Expr::load("input", Expr::var("i")),
            )],
        )],
    )
}

/// A counted loop from zero whose first iteration is guarded, and whose body
/// computes its store index. Peeling substitutes the boundary index into the
/// lifted copy, which leaves an index over literals for the algebraic family to
/// evaluate: the second law only applies because the first one fired.
fn guarded_loop_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [1, 1, 1],
        vec![Node::loop_(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![
                Node::If {
                    cond: Expr::eq(Expr::var("i"), Expr::u32(0)),
                    then: vec![Node::store("output", Expr::u32(0), Expr::u32(7))],
                    otherwise: vec![],
                },
                Node::store(
                    "output",
                    Expr::mul(Expr::var("i"), Expr::u32(2)),
                    Expr::load("input", Expr::var("i")),
                ),
            ],
        )],
    )
}

/// Adding a law family turns this red until it derives something.
#[test]
fn every_declared_family_cites_at_least_one_law() {
    let declared = declared_family_variants();
    assert!(
        !declared.is_empty(),
        "Fix: the family variants must be readable from source"
    );

    let listed: BTreeSet<String> = RegionLawFamily::all()
        .iter()
        .map(|family| format!("{family:?}"))
        .collect();
    assert_eq!(
        listed, declared,
        "Fix: `RegionLawFamily::all()` must name every declared variant"
    );

    for family in RegionLawFamily::all() {
        assert!(
            !laws_for_family(*family).is_empty(),
            "Fix: family `{}` derives nothing. Record a law stating an equality it authorizes, \
             or delete the family.",
            family.name()
        );
    }
}

/// A law names a rewrite the registry carries, and law names are unique.
#[test]
fn every_law_names_a_contracted_rewrite() {
    let mut names = BTreeSet::new();
    for law in REGION_LAWS {
        assert!(
            names.insert(law.name),
            "Fix: law name `{}` is declared twice",
            law.name
        );
        assert!(
            contract_for_pass(law.realized_by).is_some(),
            "Fix: law `{}` names rewrite `{}`, which declares no contract. A derivation cannot \
             read its numerical contract or its expansion bound.",
            law.name,
            law.realized_by
        );
        assert_eq!(
            region_law(law.name),
            Some(law),
            "Fix: `region_law` must resolve every declared law name"
        );
    }
}

/// Only the numerical family may cite a rewrite that changes values, and every
/// rewrite that changes values must be cited by one.
#[test]
fn value_changing_rewrites_are_exactly_the_numerical_citations() {
    for law in REGION_LAWS {
        let contract = law_numerical_contract(law)
            .expect("Fix: every law's rewrite must declare a numerical contract");
        if law.family.admits_value_difference() {
            continue;
        }
        assert_eq!(
            contract,
            NumericalContract::BitExact,
            "Fix: law `{}` is cited from family `{}`, which admits no value difference, but \
             rewrite `{}` declares `{contract:?}`. Cite it from the numerical family or state a \
             bit-exact contract.",
            law.name,
            law.family.name(),
            law.realized_by
        );
    }

    let numerical_citations: BTreeSet<&str> = REGION_LAWS
        .iter()
        .filter(|law| law.family.admits_value_difference())
        .map(|law| law.realized_by)
        .collect();
    for contract in registered_rewrite_contracts() {
        if contract.numerical == NumericalContract::BitExact {
            continue;
        }
        assert!(
            numerical_citations.contains(contract.pass),
            "Fix: rewrite `{}` declares `{:?}`, so a derivation may only reach it through a \
             numerical law. Record one, or state a bit-exact contract.",
            contract.pass,
            contract.numerical
        );
    }
}

/// The derivation composes laws instead of naming a shape.
#[test]
fn derivation_composes_two_laws_over_a_counted_recurrence() {
    let program = guarded_loop_program();
    let derivation = derive_region_alternatives(
        &program,
        &[],
        RegionDerivationBudget {
            max_depth: 2,
            max_alternatives: 64,
        },
    )
    .expect("Fix: the pass registry must schedule");

    assert!(
        !derivation.alternatives.is_empty(),
        "Fix: a counted recurrence over a constant range must derive at least one alternative"
    );

    let single_step: BTreeSet<[u8; 32]> = derivation
        .alternatives
        .iter()
        .filter(|derived| derived.chain.len() == 1)
        .map(|derived| derived.program.fingerprint())
        .collect();
    let composed = derivation.composed();
    assert!(
        !composed.is_empty(),
        "Fix: two laws must compose into an alternative no single law produces"
    );
    for derived in composed {
        assert!(
            !single_step.contains(&derived.program.fingerprint()),
            "Fix: alternative derived through {:?} equals a one-law alternative, so the chain is \
             not evidence of what produced it",
            derived.chain
        );
    }
}

/// A bit-exact run reaches no rewrite that changes values.
#[test]
fn a_bit_exact_run_derives_no_value_changing_law() {
    let program = counted_store_program();
    let derivation = derive_region_alternatives(
        &program,
        &[],
        RegionDerivationBudget {
            max_depth: 2,
            max_alternatives: 64,
        },
    )
    .expect("Fix: the pass registry must schedule");

    for name in derivation.cited_laws() {
        let law = region_law(name).expect("Fix: a cited law must be declared");
        assert_eq!(
            law_numerical_contract(law),
            Some(NumericalContract::BitExact),
            "Fix: law `{name}` was derived without its numerical contract being granted"
        );
    }
}

/// Granting the contract a rewrite declares is what admits its law.
#[test]
fn granting_a_contract_admits_the_law_that_declares_it() {
    let program = counted_store_program();
    let budget = RegionDerivationBudget {
        max_depth: 1,
        max_alternatives: 64,
    };

    let refused = derive_region_alternatives(&program, &[], budget)
        .expect("Fix: the pass registry must schedule");
    let granted =
        derive_region_alternatives(&program, &[NumericalContract::IntegerWrapping], budget)
            .expect("Fix: the pass registry must schedule");

    let wrapping_laws: BTreeSet<&str> = REGION_LAWS
        .iter()
        .filter(|law| law_numerical_contract(law) == Some(NumericalContract::IntegerWrapping))
        .map(|law| law.name)
        .collect();
    assert!(
        !wrapping_laws.is_empty(),
        "Fix: this case needs a law whose rewrite declares wrapping index arithmetic"
    );

    let refused_citations: BTreeSet<&str> = refused.cited_laws().into_iter().collect();
    assert!(
        refused_citations.is_disjoint(&wrapping_laws),
        "Fix: a run granting no contract derived a wrapping law"
    );

    let granted_citations: BTreeSet<&str> = granted.cited_laws().into_iter().collect();
    assert!(
        granted_citations
            .iter()
            .any(|name| wrapping_laws.contains(name)),
        "Fix: granting wrapping index arithmetic must admit a wrapping law over this fixture; \
         cited {granted_citations:?}"
    );
}

/// Each bound stops the run and says so.
#[test]
fn each_bound_reports_its_own_stop_reason() {
    let program = counted_store_program();

    let capped = derive_region_alternatives(
        &program,
        &[],
        RegionDerivationBudget {
            max_depth: 3,
            max_alternatives: 1,
        },
    )
    .expect("Fix: the pass registry must schedule");
    assert_eq!(capped.alternatives.len(), 1);
    assert_eq!(capped.stop, RegionDerivationStop::AlternativeLimit);

    let shallow = derive_region_alternatives(
        &program,
        &[],
        RegionDerivationBudget {
            max_depth: 1,
            max_alternatives: 64,
        },
    )
    .expect("Fix: the pass registry must schedule");
    assert_eq!(
        shallow.stop,
        RegionDerivationStop::DepthReached,
        "Fix: a run that derived alternatives at its last admitted depth has laws left to compose"
    );
    assert!(
        shallow
            .alternatives
            .iter()
            .all(|derived| derived.chain.len() == 1),
        "Fix: a depth-one run must derive one-law alternatives only"
    );

    let none = derive_region_alternatives(
        &program,
        &[],
        RegionDerivationBudget {
            max_depth: 3,
            max_alternatives: 0,
        },
    )
    .expect("Fix: the pass registry must schedule");
    assert!(none.alternatives.is_empty());
    assert_eq!(none.stop, RegionDerivationStop::AlternativeLimit);
}
