//! The construct-law table is closed over the AST registry.
//!
//! WHY: a derivation that reads laws instead of per-kernel recipes is only as
//! closed as its record of which constructs expose which laws. A construct
//! added to the AST registry with no row would be treated as having no law by
//! omission, which is indistinguishable from a recorded decision that it has
//! none. The variant space is read from the generated tables, which the AST
//! registry macro emits, so this comparison cannot go stale by hand.
//!
//! Not covered here: whether a recorded opacity reason is the right call. That
//! is a review decision, and the reason is recorded so a reviewer can make it.

use std::collections::BTreeSet;

use vyre_foundation::ir::{Expr, Node, EXPR_VARIANT_NAMES, NODE_VARIANT_NAMES};
use vyre_foundation::optimizer::construct_law::{
    expr_construct_laws, laws_of, laws_of_node, node_construct_laws, ConstructLaws,
    EXPR_CONSTRUCT_LAWS, NODE_CONSTRUCT_LAWS,
};
use vyre_spec::RegionLawFamily;

fn recorded(table: &'static [ConstructLaws]) -> BTreeSet<&'static str> {
    table.iter().map(|entry| entry.construct).collect()
}

fn declared(names: &'static [&'static str]) -> BTreeSet<&'static str> {
    names.iter().copied().collect()
}

#[test]
fn every_declared_expression_construct_records_laws_or_opacity() {
    let declared = declared(EXPR_VARIANT_NAMES);
    let recorded = recorded(EXPR_CONSTRUCT_LAWS);
    assert_eq!(
        recorded.len(),
        EXPR_CONSTRUCT_LAWS.len(),
        "Fix: two expression construct-law rows name the same construct."
    );
    let unrecorded: Vec<&str> = declared.difference(&recorded).copied().collect();
    assert!(
        unrecorded.is_empty(),
        "Fix: record the law families or the opacity of {unrecorded:?} in \
         vyre-foundation/src/optimizer/construct_law.rs."
    );
    let orphaned: Vec<&str> = recorded.difference(&declared).copied().collect();
    assert!(
        orphaned.is_empty(),
        "Fix: {orphaned:?} carry a construct-law row but are not declared expression variants."
    );
}

#[test]
fn every_declared_statement_construct_records_laws_or_opacity() {
    let declared = declared(NODE_VARIANT_NAMES);
    let recorded = recorded(NODE_CONSTRUCT_LAWS);
    assert_eq!(
        recorded.len(),
        NODE_CONSTRUCT_LAWS.len(),
        "Fix: two statement construct-law rows name the same construct."
    );
    let unrecorded: Vec<&str> = declared.difference(&recorded).copied().collect();
    assert!(
        unrecorded.is_empty(),
        "Fix: record the law families or the opacity of {unrecorded:?} in \
         vyre-foundation/src/optimizer/construct_law.rs."
    );
    let orphaned: Vec<&str> = recorded.difference(&declared).copied().collect();
    assert!(
        orphaned.is_empty(),
        "Fix: {orphaned:?} carry a construct-law row but are not declared statement variants."
    );
}

#[test]
fn a_construct_records_laws_or_opacity_and_never_both_or_neither() {
    for entry in EXPR_CONSTRUCT_LAWS.iter().chain(NODE_CONSTRUCT_LAWS) {
        match (entry.families.is_empty(), entry.opacity) {
            (true, Some(_)) | (false, None) => {}
            (true, None) => panic!(
                "Fix: {} records neither a law family nor a reason it exposes none; a silent \
                 absence is not a decision.",
                entry.construct
            ),
            (false, Some(reason)) => panic!(
                "Fix: {} records law families and the opacity reason {reason:?}; a construct that \
                 exposes a law is not opaque.",
                entry.construct
            ),
        }
    }
}

#[test]
fn a_cited_family_is_one_the_frozen_vocabulary_declares() {
    let vocabulary: BTreeSet<RegionLawFamily> = RegionLawFamily::all().iter().copied().collect();
    for entry in EXPR_CONSTRUCT_LAWS.iter().chain(NODE_CONSTRUCT_LAWS) {
        for family in entry.families {
            assert!(
                vocabulary.contains(family),
                "Fix: {} cites family {family:?}, which RegionLawFamily::all() does not declare.",
                entry.construct
            );
            assert!(
                entry.admits(*family),
                "Fix: {} cites family {family:?} but admits() rejects it.",
                entry.construct
            );
        }
    }
}

#[test]
fn a_family_the_construct_does_not_cite_is_refused() {
    let load = expr_construct_laws("Load").expect("Load is a declared expression construct");
    assert!(load.admits(RegionLawFamily::Layout));
    assert!(!load.admits(RegionLawFamily::Numerical));

    let atomic = expr_construct_laws("Atomic").expect("Atomic is a declared expression construct");
    for family in RegionLawFamily::all() {
        assert!(
            !atomic.admits(*family),
            "Fix: Atomic is recorded opaque, so no family may be cited over it."
        );
    }
}

#[test]
fn an_undeclared_construct_name_has_no_row() {
    assert!(expr_construct_laws("NoSuchExpr").is_none());
    assert!(node_construct_laws("NoSuchNode").is_none());
    assert!(node_construct_laws("BinOp").is_none());
    assert!(expr_construct_laws("Loop").is_none());
}

#[test]
fn a_value_resolves_to_the_row_of_the_variant_it_holds() {
    assert_eq!(laws_of(&Expr::u32(7)).construct, "LitU32");
    assert_eq!(
        laws_of(&Expr::add(Expr::u32(1), Expr::u32(2))).construct,
        "BinOp"
    );
    assert!(laws_of(&Expr::add(Expr::u32(1), Expr::u32(2))).admits(RegionLawFamily::Algebraic));

    let loop_node = Node::Loop {
        var: "i".into(),
        from: Expr::u32(0),
        to: Expr::u32(4),
        body: Vec::new(),
    };
    assert_eq!(laws_of_node(&loop_node).construct, "Loop");
    assert!(laws_of_node(&loop_node).admits(RegionLawFamily::Recurrence));

    let barrier = Node::Return;
    assert!(
        laws_of_node(&barrier).opacity.is_some(),
        "Fix: Return is recorded opaque, so its row must carry the reason."
    );
}

#[test]
fn a_reduction_construct_cites_the_reduction_family() {
    for construct in ["SubgroupReduce"] {
        let entry = expr_construct_laws(construct).expect("declared expression construct");
        assert!(
            entry.admits(RegionLawFamily::Reduction),
            "Fix: {construct} combines lane values, so a reduction law applies to it."
        );
    }
    for construct in ["AllReduce", "ReduceScatter", "TileReduce", "TileMatmul"] {
        let entry = node_construct_laws(construct).expect("declared statement construct");
        assert!(
            entry.admits(RegionLawFamily::Reduction),
            "Fix: {construct} combines values over a domain, so a reduction law applies to it."
        );
    }
}
