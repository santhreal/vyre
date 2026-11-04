//! `public-api-paths` refuses every `pub` line shape the crate does not publish.
//!
//! The gate drops a duplicate path when the crate's own source declares that
//! name more than once, because a terminal id table per grammar and a per-op
//! `OP_ID` are one name over several facts rather than one item at several
//! paths. That decision is only sound for a declaration the crate publishes, in
//! the namespace the name occupies. Every shape below was counted as a
//! declaration and each one hid a real second path: an associated item, a
//! `#[cfg(test)]` item, a `pub(crate)` item, and a `const` whose value is
//! another module's item of the same name.
//!
//! This closes the class rather than the four incidents. `NotPublished::ALL` is
//! the reason list the gate compiles against, so `SHAPES` is sized from it and a
//! fifth reason fails to build until a row records the shape and the differential
//! it is expected to produce. `expected` matches `Namespace` exhaustively with no
//! catch-all arm, so a third namespace fails to build the same way.
//!
//! Not covered here: whether a shape the gate still counts as a declaration
//! ought to be refused. A line scanner cannot see a name a macro expands to, so
//! a macro-generated declaration counts and the gate reports its suppression
//! total in the run note instead of claiming otherwise.

use xtask::gates::public_api_paths::{
    declarations, duplicates, Declared, Namespace, NotPublished, Published,
};

/// The name every constant fixture publishes at two paths.
const NAME: &str = "WORD_INDEX";

/// The name the namespace fixture publishes at two paths in both namespaces.
const SHARED: &str = "shape";

/// A snapshot publishing one constant through two sibling modules.
const CONST_AT_TWO_PATHS: &str = "\
pub mod vyre_x::alpha
pub mod vyre_x::beta
pub const vyre_x::alpha::WORD_INDEX: u32
pub const vyre_x::beta::WORD_INDEX: u32
";

/// A snapshot publishing a module and a function of one name at two paths each.
const NAME_AT_TWO_PATHS_IN_BOTH_NAMESPACES: &str = "\
pub mod vyre_x::alpha
pub mod vyre_x::beta
pub mod vyre_x::alpha::shape
pub mod vyre_x::beta::shape
pub fn vyre_x::alpha::shape() -> u32
pub fn vyre_x::beta::shape() -> u32
";

/// The source of the module that owns the constant.
const OWNER: &str = "pub const WORD_INDEX: u32 = 7;\n";

/// One `pub` line shape, written both as the crate refuses it and as the crate
/// publishes it, so each reason is proved by the difference between the two.
struct Shape {
    reason: NotPublished,
    refused: &'static str,
    published: &'static str,
}

/// One row per reason the gate names, in `position` order.
const SHAPES: [Shape; NotPublished::ALL.len()] = [
    Shape {
        reason: NotPublished::AssociatedItem,
        refused: "impl Holder {\n    pub const WORD_INDEX: u32 = 7;\n}\n",
        published: "pub const WORD_INDEX: u32 = 7;\n",
    },
    Shape {
        reason: NotPublished::TestOnly,
        refused: "#[cfg(test)]\npub const WORD_INDEX: u32 = 7;\n",
        published: "pub const WORD_INDEX: u32 = 7;\n",
    },
    Shape {
        reason: NotPublished::Restricted,
        refused: "pub(crate) const WORD_INDEX: u32 = 7;\n",
        published: "pub const WORD_INDEX: u32 = 7;\n",
    },
    Shape {
        reason: NotPublished::Republication,
        refused: "pub const WORD_INDEX: u32 = control::WORD_INDEX;\n",
        published: "pub const WORD_INDEX: u32 = 7;\n",
    },
];

/// What the gate reports for one snapshot against one crate's declarations,
/// which is `duplicates` narrowed by the same predicate `run` applies.
fn reported(snapshot: &str, declared: &Declared) -> Vec<Published> {
    let mut found = duplicates(snapshot);
    found.retain(|published, _| !declared.shares(published.namespace, published.head()));
    found.into_keys().collect()
}

fn published(namespace: Namespace, tail: &str) -> Published {
    Published {
        namespace,
        tail: tail.to_string(),
    }
}

#[test]
fn every_reason_has_one_row_at_its_own_position() {
    for (at, shape) in SHAPES.iter().enumerate() {
        assert_eq!(
            shape.reason.position(),
            at,
            "row {at} records {:?}, whose position is {}",
            shape.reason,
            shape.reason.position()
        );
    }
}

#[test]
fn a_refused_shape_leaves_the_second_path_reported() {
    for shape in &SHAPES {
        let sources = [OWNER.to_string(), shape.refused.to_string()];
        let declared = declarations(&sources);
        assert!(
            !declared.shares(Namespace::Item, NAME),
            "{:?} was read as a second declaration of {NAME}",
            shape.reason
        );
        assert_eq!(
            declared.refused(shape.reason),
            1,
            "{:?} refused {} lines",
            shape.reason,
            declared.refused(shape.reason)
        );
        for other in NotPublished::ALL {
            if other != shape.reason {
                assert_eq!(
                    declared.refused(other),
                    0,
                    "{:?} was attributed to {other:?}",
                    shape.reason
                );
            }
        }
        assert_eq!(declared.refusals(), 1, "{:?}", shape.reason);
        assert_eq!(
            reported(CONST_AT_TWO_PATHS, &declared),
            vec![published(Namespace::Item, NAME)],
            "{:?} suppressed the second path of {NAME}",
            shape.reason
        );
    }
}

#[test]
fn the_same_shape_published_suppresses_the_second_path() {
    for shape in &SHAPES {
        let sources = [OWNER.to_string(), shape.published.to_string()];
        let declared = declarations(&sources);
        assert!(
            declared.shares(Namespace::Item, NAME),
            "{:?} written as published planted no second declaration",
            shape.reason
        );
        assert_eq!(declared.refusals(), 0, "{:?}", shape.reason);
        assert!(
            reported(CONST_AT_TWO_PATHS, &declared).is_empty(),
            "{:?} written as published still reported {NAME}",
            shape.reason
        );
    }
}

/// Whether a name two sibling modules share in the module namespace, and one
/// module declares in the item namespace, stays reported for `namespace`.
///
/// An exhaustive match with no catch-all arm: a third namespace does not
/// compile until this records what the gate owes it.
const fn expected(namespace: Namespace) -> bool {
    match namespace {
        Namespace::Module => false,
        Namespace::Item => true,
    }
}

#[test]
fn a_module_name_does_not_cover_an_item_name() {
    let sources = [
        "pub mod shape;\n".to_string(),
        "pub mod shape;\n".to_string(),
        "pub fn shape() -> u32 { 0 }\n".to_string(),
    ];
    let declared = declarations(&sources);
    let axis = duplicates(NAME_AT_TWO_PATHS_IN_BOTH_NAMESPACES);
    for namespace in [Namespace::Module, Namespace::Item] {
        assert!(
            axis.contains_key(&published(namespace, SHARED)),
            "the fixture publishes {SHARED} at one path in {namespace:?}"
        );
    }
    let reported = reported(NAME_AT_TWO_PATHS_IN_BOTH_NAMESPACES, &declared);
    for namespace in [Namespace::Module, Namespace::Item] {
        assert_eq!(
            reported.contains(&published(namespace, SHARED)),
            expected(namespace),
            "{namespace:?} reporting of {SHARED} does not match what this test records"
        );
    }
}
