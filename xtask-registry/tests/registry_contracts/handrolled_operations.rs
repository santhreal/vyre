//! The containment relation behind the `handrolled-operations` gate.
//!
//! WHY: the gate is pinned at zero, so its value is entirely in what it
//! reports. A relation that reports nothing passes the pin on every tree,
//! including one where an operation was rebuilt by hand, which is the failure
//! this file exists to make impossible. Every case below drives
//! [`handrolls`] directly, so the positive cases go red the moment the
//! relation stops finding a copy and the negative cases go red the moment it
//! starts inventing one.
//!
//! The two IR cases are the pair that matters. Attribution is not a field this
//! gate reads; it is encoded in the fingerprint, because a region naming the
//! composition behind it collapses to a hash of the operation it selects while
//! a region naming none inlines its body. `an_attributed_selection_is_not_a_
//! handroll` and `an_unattributed_copy_of_the_same_body_is_a_handroll` differ
//! in exactly that one field and must disagree.

use vyre_foundation::composition::{wrap_child_region, wrap_region};
use vyre_foundation::ir::{Expr, Ident, Node, Program};
use xtask_registry::gates::handrolled_operations::{
    handrolls, FingerprintedOperation, Handroll,
};
use xtask_registry::gates::lego_audit::{fingerprint_program, MIN_COMPARABLE_FINGERPRINT_BYTES};

/// A body long enough to clear the comparison floor once fingerprinted.
fn child_body() -> Vec<Node> {
    vec![
        Node::let_bind("acc", Expr::u32(0)),
        Node::let_bind("step", Expr::add(Expr::var("acc"), Expr::u32(1))),
        Node::store("out", Expr::u32(0), Expr::var("step")),
    ]
}

fn program_of(nodes: Vec<Node>) -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], nodes)
}

/// The child operation as it is registered: its body under its own region,
/// naming no parent.
fn child_program() -> Program {
    program_of(vec![wrap_region("vyre-test::child", child_body(), None)])
}

fn subjects<'a>(entries: &'a [(&'a str, Vec<u8>)]) -> Vec<FingerprintedOperation<'a>> {
    entries
        .iter()
        .map(|(id, fingerprint)| FingerprintedOperation {
            id: *id,
            fingerprint: fingerprint.as_slice(),
        })
        .collect()
}

#[test]
fn a_host_holding_another_whole_program_is_reported_once_at_its_first_offset() {
    let needle: Vec<u8> = (0u8..16).collect();
    let mut host = vec![0xAA; 4];
    host.extend_from_slice(&needle);
    host.extend_from_slice(&[0xBB; 4]);
    host.extend_from_slice(&needle);

    let entries = [("host", host), ("needle", needle)];
    assert_eq!(
        handrolls(&subjects(&entries)),
        vec![Handroll {
            host: "host",
            rebuilt: "needle",
            offset: 4,
        }],
        "Fix: a host containing another registered program must be reported \
         exactly once, at the lowest offset the copy starts at"
    );
}

#[test]
fn an_attributed_selection_is_not_a_handroll() {
    let child = child_program();
    let parent = program_of(vec![wrap_region(
        "vyre-test::parent",
        vec![wrap_child_region(
            "vyre-test::child",
            Ident::from("vyre-test::parent"),
            child_body(),
        )],
        None,
    )]);

    let entries = [
        ("vyre-test::parent", fingerprint_program(&parent)),
        ("vyre-test::child", fingerprint_program(&child)),
    ];
    assert_eq!(
        handrolls(&subjects(&entries)),
        Vec::new(),
        "Fix: a composition that names the operation it selected is the \
         correct spelling and must not be reported as a rebuild"
    );
}

#[test]
fn an_unattributed_copy_of_the_same_body_is_a_handroll() {
    let child = child_program();
    let parent = program_of(vec![wrap_region(
        "vyre-test::parent",
        vec![wrap_region("vyre-test::child", child_body(), None)],
        None,
    )]);

    let entries = [
        ("vyre-test::parent", fingerprint_program(&parent)),
        ("vyre-test::child", fingerprint_program(&child)),
    ];
    let found = handrolls(&subjects(&entries));
    assert_eq!(
        found.len(),
        1,
        "Fix: a composition that rebuilds a registered operation's whole \
         program without naming it must be reported: {found:?}"
    );
    assert_eq!(found[0].host, "vyre-test::parent");
    assert_eq!(found[0].rebuilt, "vyre-test::child");
}

#[test]
fn two_operations_that_are_the_same_program_are_a_duplicate_not_a_handroll() {
    let fingerprint = fingerprint_program(&child_program());
    let entries = [
        ("vyre-test::left", fingerprint.clone()),
        ("vyre-test::right", fingerprint),
    ];
    assert_eq!(
        handrolls(&subjects(&entries)),
        Vec::new(),
        "Fix: equal fingerprints are one operation registered twice, which \
         whats-similar reports at score 1.0; reporting it here as well would \
         make the pin count the same defect under two names"
    );
}

#[test]
fn a_program_under_the_comparison_floor_is_never_the_thing_a_host_rebuilt() {
    let needle = vec![0x09; MIN_COMPARABLE_FINGERPRINT_BYTES - 1];
    let mut host = vec![0xAA; 8];
    host.extend_from_slice(&needle);
    host.extend_from_slice(&[0xBB; 8]);

    let entries = [("host", host), ("tiny", needle)];
    assert_eq!(
        handrolls(&subjects(&entries)),
        Vec::new(),
        "Fix: a fingerprint under the comparison floor describes one or two \
         nodes, and finding it inside a larger program is coincidence; the \
         floor is what keeps a pin of zero honest"
    );
}

#[test]
fn a_host_is_never_reported_against_itself() {
    let fingerprint = fingerprint_program(&child_program());
    let entries = [("vyre-test::child", fingerprint)];
    assert_eq!(
        handrolls(&subjects(&entries)),
        Vec::new(),
        "Fix: every program contains itself, so self-containment is not a \
         finding"
    );
}
