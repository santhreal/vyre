//! The class closed here: an async copy pipeline whose tags do not pair, sent
//! to a backend that accepts it and copies into a destination the previous copy
//! is still writing, or never writes at all.
//!
//! `AsyncLoad`/`AsyncStore` start a copy under a tag and `AsyncWait` ends it.
//! The hashmap reference interpreter already refuses a second start of a tag in
//! flight, a wait with nothing pending, and an invocation that ends with a
//! transfer still pending, but it refuses them at run time, on the one input
//! that was executed. Nothing said so at compile time, so the defect reached a
//! backend and came back as wrong numbers.
//!
//! The rules decide whether multi-stage pipelining is safe to generate. A
//! depth-D pipeline keeps D copies in flight and the tag is the only thing that
//! tells them apart, so depth D needs D tags and one wait per tag before its
//! slot comes round. The tests below pin both directions: the rotated shapes a
//! pipeline emits stay accepted, and the shapes that silently overwrite or
//! silently drop are rejected.
//!
//! Not caught: a loop whose trip count is not a literal cannot be shown to run
//! its back edge, so a tag restarted there is not a duplicate start. The tag is
//! still reported as left in flight, which is the finding that matters.

use vyre_foundation::ir::{Expr, Node, Program, NODE_VARIANT_NAMES};
use vyre_foundation::validate::validate;

/// Wrap `entry` in the smallest program that carries it.
fn program(entry: Vec<Node>) -> Program {
    Program::wrapped(vec![], [64, 1, 1], entry)
}

/// Start a copy under `tag`.
fn load(tag: &str) -> Node {
    Node::AsyncLoad {
        source: "ssd".into(),
        destination: "vram".into(),
        offset: Box::new(Expr::u32(0)),
        size: Box::new(Expr::u32(64)),
        tag: tag.into(),
    }
}

/// Start a store under `tag`.
fn store(tag: &str) -> Node {
    Node::AsyncStore {
        source: "vram".into(),
        destination: "ssd".into(),
        offset: Box::new(Expr::u32(0)),
        size: Box::new(Expr::u32(64)),
        tag: tag.into(),
    }
}

fn wait(tag: &str) -> Node {
    Node::AsyncWait { tag: tag.into() }
}

/// A loop with literal bounds, which is what proves a back edge runs.
fn counted_loop(trips: u32, body: Vec<Node>) -> Node {
    Node::Loop {
        var: "i".into(),
        from: Expr::u32(0),
        to: Expr::u32(trips),
        body,
    }
}

/// Every code reported for `entry`, so a case can assert on one rule without
/// depending on which other rules a program happens to trip.
fn codes(entry: Vec<Node>) -> Vec<String> {
    validate(&program(entry))
        .iter()
        .map(|error| error.code().as_str().to_string())
        .collect()
}

/// The codes this pass owns, derived from the rule catalog rather than listed,
/// so a fourth async rule joins the clean assertions without an edit here.
fn async_codes() -> Vec<&'static str> {
    vyre_foundation::validate::rules()
        .iter()
        .filter(|rule| rule.invariant.starts_with("async"))
        .map(|rule| rule.code)
        .collect()
}

fn assert_reports(entry: Vec<Node>, code: &str, why: &str) {
    let reported = codes(entry);
    assert!(
        reported.iter().any(|found| found.as_str() == code),
        "{why}: expected {code}, got {reported:?}"
    );
}

fn assert_absent(entry: Vec<Node>, code: &str, why: &str) {
    let reported = codes(entry);
    assert!(
        !reported.iter().any(|found| found.as_str() == code),
        "{why}: {code} must not fire, got {reported:?}"
    );
}

/// No async rule fires at all.
fn assert_clean(entry: Vec<Node>, why: &str) {
    let owned = async_codes();
    let reported = codes(entry);
    assert!(
        !reported
            .iter()
            .any(|found| owned.contains(&found.as_str())),
        "{why}: {reported:?}"
    );
}

#[test]
fn a_second_start_of_a_tag_in_flight_is_rejected() {
    assert_reports(
        vec![load("stage0"), load("stage0"), wait("stage0")],
        "V131",
        "two copies under one tag land in the same destination",
    );
}

#[test]
fn a_second_store_under_a_tag_in_flight_is_rejected() {
    assert_reports(
        vec![store("drain"), store("drain"), wait("drain")],
        "V131",
        "a store starts a transfer exactly as a load does",
    );
}

#[test]
fn a_start_after_the_wait_is_accepted() {
    assert_clean(
        vec![load("stage0"), wait("stage0"), load("stage0"), wait("stage0")],
        "the tag is free once its copy has been waited",
    );
}

#[test]
fn a_depth_two_pipeline_with_two_tags_is_accepted() {
    assert_clean(
        vec![
            load("stage0"),
            load("stage1"),
            counted_loop(
                8,
                vec![
                    wait("stage0"),
                    load("stage0"),
                    wait("stage1"),
                    load("stage1"),
                ],
            ),
            wait("stage0"),
            wait("stage1"),
        ],
        "depth two with one tag per stage is the shape a pipeline emits",
    );
}

#[test]
fn a_prologue_start_paired_with_a_wait_at_the_top_of_the_body_is_accepted() {
    assert_clean(
        vec![
            load("stage0"),
            counted_loop(8, vec![wait("stage0"), load("stage0")]),
            wait("stage0"),
        ],
        "the single-buffer pipelined shape waits its tag before restarting it",
    );
}

#[test]
fn a_loop_that_restarts_a_tag_without_waiting_is_rejected() {
    assert_reports(
        vec![counted_loop(8, vec![load("stage0")]), wait("stage0")],
        "V131",
        "the back edge starts the tag again while the first copy is in flight",
    );
}

#[test]
fn a_loop_whose_trip_count_is_not_literal_carries_no_duplicate_start() {
    assert_absent(
        vec![
            Node::Loop {
                var: "i".into(),
                from: Expr::u32(0),
                to: Expr::var("n"),
                body: vec![load("stage0")],
            },
            wait("stage0"),
        ],
        "V131",
        "a body that may run at most once has no back edge to reuse the tag",
    );
}

#[test]
fn a_wait_with_nothing_pending_is_rejected() {
    assert_reports(
        vec![wait("stage0")],
        "V132",
        "a wait for a copy nobody started blocks on nothing",
    );
}

#[test]
fn a_wait_after_the_tag_was_already_waited_is_rejected() {
    assert_reports(
        vec![load("stage0"), wait("stage0"), wait("stage0")],
        "V132",
        "the second wait has no copy left to wait for",
    );
}

#[test]
fn a_start_on_one_branch_and_a_wait_after_the_join_is_accepted() {
    assert_clean(
        vec![
            Node::If {
                cond: Expr::LitBool(true),
                then: vec![load("stage0")],
                otherwise: vec![],
            },
            wait("stage0"),
        ],
        "a wait is wrong only when no path could have started the tag",
    );
}

#[test]
fn a_start_on_every_branch_followed_by_a_start_is_rejected() {
    assert_reports(
        vec![
            Node::If {
                cond: Expr::LitBool(true),
                then: vec![load("stage0")],
                otherwise: vec![load("stage0")],
            },
            load("stage0"),
            wait("stage0"),
        ],
        "V131",
        "the copy is in flight however the program reached the third start",
    );
}

#[test]
fn a_start_on_a_branch_that_leaves_the_invocation_does_not_reach_the_join() {
    assert_reports(
        vec![
            Node::If {
                cond: Expr::LitBool(true),
                then: vec![load("stage0"), Node::Return],
                otherwise: vec![],
            },
            wait("stage0"),
        ],
        "V132",
        "the only path past the branch is the one that started nothing",
    );
}

#[test]
fn a_start_inside_a_block_reaches_the_wait_outside_it() {
    assert_clean(
        vec![Node::Block(vec![load("stage0")]), wait("stage0")],
        "a block is a scope for names, not a boundary for transfers",
    );
}

#[test]
fn a_copy_started_and_never_waited_is_rejected() {
    assert_reports(
        vec![load("stage0")],
        "V133",
        "a copy nobody waited for lands in a destination nothing synchronized",
    );
}

#[test]
fn a_copy_started_on_one_branch_and_never_waited_is_rejected() {
    assert_reports(
        vec![Node::If {
            cond: Expr::LitBool(true),
            then: vec![load("stage0")],
            otherwise: vec![],
        }],
        "V133",
        "a copy left pending on any path is a copy nothing ordered",
    );
}

#[test]
fn a_copy_pending_where_the_invocation_returns_is_rejected() {
    assert_reports(
        vec![load("stage0"), Node::Return, wait("stage0")],
        "V133",
        "the wait after the Return runs on no path",
    );
}

#[test]
fn a_copy_started_every_iteration_and_never_waited_is_rejected() {
    assert_reports(
        vec![counted_loop(8, vec![load("stage0")])],
        "V133",
        "a loop that only starts copies leaves the last one pending",
    );
}

#[test]
fn every_async_node_variant_is_accounted_for() {
    let async_variants: Vec<&str> = NODE_VARIANT_NAMES
        .iter()
        .copied()
        .filter(|name| name.starts_with("Async"))
        .collect();
    assert_eq!(
        async_variants,
        ["AsyncLoad", "AsyncStore", "AsyncWait"],
        "a new async node variant needs a decision in the tag analysis. \
         Fix: give it a start or an end in \
         vyre-foundation/src/validate/async_pipeline.rs and a case here."
    );
}

#[test]
fn every_async_rule_in_the_catalog_has_a_case_here() {
    let owned = async_codes();
    assert_eq!(
        owned,
        ["V128", "V131", "V132", "V133"],
        "an async rule joined the catalog without a case in this suite. \
         Fix: add the case, then record the code here."
    );
}
