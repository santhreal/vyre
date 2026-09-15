//! Comparing one stretch of generated IR against the same stretch in a sibling.
//!
//! # Why this has one owner
//!
//! A family of entry points that all walk a CSR row is only one loop if they
//! emit one loop, and the way to prove that is to slice the same span out of
//! each program's IR and compare the slices. Two crates wrote that slicing:
//! `vyre-primitives` proving its own clone family agrees, and `vyre-libs`
//! proving its resident sites reach the primitive's loop rather than a private
//! copy of it. Both needed the same three steps and both wrote them out.
//!
//! Two copies of a comparison helper are worse than two copies of a test,
//! because the helper decides what the test can see. One copy widening its slice
//! or forgiving a marker it cannot find weakens an assertion in a crate whose
//! author never read the change, and the difference does not show up as a
//! failure anywhere.
//!
//! # What a slice is
//!
//! A region runs from the `Let` that introduces one marker to the `Let` that
//! introduces the next, so it always begins and ends on a whole binding. It
//! starts inside the region body on purpose: the enclosing `Node::Region`
//! carries the entry point's own op id, which is part of the public contract and
//! differs between siblings by construction.

use vyre_foundation::ir::Program;

/// Erase one builder's private variable prefix so two uses of the same loop
/// compare equal.
///
/// Every builder in these families uses exactly one prefix, so replacing it with
/// a fixed marker is total: nothing else in the dump can collide with it.
#[must_use]
pub fn canonicalize(program: &Program, prefix: &str) -> String {
    format!("{:?}", program.entry()).replace(&format!("{prefix}_"), "Q_")
}

/// Slice a canonicalized dump between two `Let`-introduced markers.
///
/// # Panics
///
/// Panics when a marker is absent, is not introduced by a `Let`, or the two
/// markers appear out of order. Each is a broken assertion rather than a failed
/// one: a slice that silently came back empty would compare equal to any other
/// empty slice and pass.
#[must_use]
pub fn region(dump: &str, from: &str, to: &str) -> String {
    let bind_start = |marker: &str| {
        let at = dump
            .find(marker)
            .unwrap_or_else(|| panic!("Fix: canonicalized dump must contain `{marker}`:\n{dump}"));
        dump[..at].rfind("Let {").unwrap_or_else(|| {
            panic!("Fix: `{marker}` must be introduced by a Let binding:\n{dump}")
        })
    };
    let start = bind_start(from);
    let end = bind_start(to);
    assert!(
        start < end,
        "Fix: region markers are out of order in:\n{dump}"
    );
    dump[start..end].to_string()
}

/// The edge-kind allow test, destination load, destination bound check, and
/// destination word/bit split of one CSR queue program.
///
/// Shared by every queue entry point in the family; the emit that follows it,
/// named by `emit_var`, is what legitimately differs.
#[must_use]
pub fn edge_guard(program: &Program, prefix: &str, emit_var: &str) -> String {
    let dump = canonicalize(program, prefix);
    let emit = format!("Ident(\"{emit_var}\")").replace(&format!("{prefix}_"), "Q_");
    region(&dump, "Ident(\"Q_kind\")", &emit)
}
