//! Every data-derived loop bound in a registered `vyre-libs` program is clamped.
//!
//! A `Node::Loop` whose `to` resolves to a `Load` runs for as many iterations
//! as that buffer word states, so one out-of-contract `u32` asks for four
//! billion: hours in the reference interpreter, a watchdog reset on a device.
//! The out-of-bounds store check inside a body does not bound the loop, because
//! it discards the result of an iteration that already ran.
//!
//! The scan reads the operation registry at run time and builds each canonical
//! program, so a new registration whose loop bound reaches a `Load` without
//! passing through a clamp turns this test red.
//!
//! The classification itself is [`vyre_foundation::loop_bounds`], which reads
//! the per-variant decision from `visit::expr_magnitude`. That match has no
//! catch-all arm over `Expr`, so an expression variant added to the IR fails to
//! compile until someone states where its magnitude comes from, and an operator
//! added to `BinOp` or `UnOp` classifies as unattributable and turns this red.
//! A copy of that decision living here would go stale against the IR in
//! silence, which is the same failure as having no scan.
//!
//! What this does not catch: a bound clamped against the wrong buffer, and a
//! body that indexes a buffer the clamp does not name. Both are per-site
//! review, not a property the IR carries.

use vyre_foundation::ir::{Expr, Node};
use vyre_foundation::loop_bounds::data_derived_loop_bounds_in;
use vyre_foundation::operation::OperationRegistration;
use vyre_libs::state_machine::TableStateMachineComposer;

/// Every unclamped loop in `nodes`, rendered one finding per line.
fn findings(nodes: &[Node]) -> Vec<String> {
    data_derived_loop_bounds_in(nodes)
        .into_iter()
        .map(|finding| match finding.source {
            Some(buffer) => format!(
                "loop `{}` has a trip count read from buffer `{}` with no clamp against an extent",
                finding.var.as_str(),
                buffer.as_str()
            ),
            None => format!(
                "loop `{}` has a trip count this crate cannot attribute to a buffer or a host fact",
                finding.var.as_str()
            ),
        })
        .collect()
}

/// The scan reports the defect this row closes, and stops reporting it once the
/// clamp is applied. Without this pair the scan could pass by never failing.
#[test]
fn the_scan_separates_a_data_derived_bound_from_a_clamped_one() {
    let unclamped = vec![Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::load("counts", Expr::u32(0)),
        vec![Node::store("out", Expr::var("i"), Expr::u32(1))],
    )];
    assert_eq!(
        findings(&unclamped).len(),
        1,
        "a loop bounded by a raw load must be reported"
    );

    let clamped = vec![Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::min(Expr::load("counts", Expr::u32(0)), Expr::buf_len("out")),
        vec![Node::store("out", Expr::var("i"), Expr::u32(1))],
    )];
    assert!(
        findings(&clamped).is_empty(),
        "a load clamped against a buffer extent is a bound: {:?}",
        findings(&clamped)
    );

    // The same load reached through a binding, which is the shape every CSR row
    // walk in this crate uses.
    let through_binding = vec![
        Node::let_bind("end", Expr::load("offsets", Expr::u32(1))),
        Node::loop_for(
            "e",
            Expr::u32(0),
            Expr::var("end"),
            vec![Node::store("out", Expr::var("e"), Expr::u32(1))],
        ),
    ];
    assert_eq!(
        findings(&through_binding).len(),
        1,
        "a bound must be followed through its `let`"
    );

    // The buffer the bound reads is named, so a finding says which producer
    // contract to clamp against rather than only that one is missing.
    assert_eq!(
        data_derived_loop_bounds_in(&through_binding)[0]
            .source
            .as_ref()
            .map(|buffer| buffer.as_str().to_owned()),
        Some("offsets".to_owned())
    );
}

/// The three scan bodies the shared state-machine composer emits carry the
/// clamp. This composer is ungated, so this case runs under every feature
/// selection and is what links `vyre_libs` into this binary.
#[test]
fn state_machine_scan_bodies_bound_their_trip_counts() {
    let composer = TableStateMachineComposer::new("transitions");

    let walk = vec![composer.walk_input_slice(
        "step",
        "input",
        Expr::u32(0),
        Expr::load("lengths", Expr::u32(0)),
    )];
    assert!(
        findings(&walk).is_empty(),
        "walk_input_slice: {:?}",
        findings(&walk)
    );

    let linear = composer.linear_scan_body(
        "input",
        "accept",
        "matches",
        Expr::load("lengths", Expr::u32(0)),
    );
    assert!(
        findings(&linear).is_empty(),
        "linear_scan_body: {:?}",
        findings(&linear)
    );

    let tiled = composer.tiled_decode_scan_body(
        "accept",
        "matches",
        Expr::load("lengths", Expr::u32(0)),
        4,
        |index| Expr::load("input", index),
        |_index, _value| None,
    );
    assert!(
        findings(&tiled).is_empty(),
        "tiled_decode_scan_body: {:?}",
        findings(&tiled)
    );
}

/// The range ordering builder bounds its scan loop against the counts buffer and extent buffers.
#[test]
fn range_ordering_scan_body_bounds_its_trip_counts() {
    let (nodes, _) = vyre_libs::range_ordering::match_order(Expr::u32(0), Expr::u32(1), "test");
    assert!(
        findings(&nodes).is_empty(),
        "match_order: {:?}",
        findings(&nodes)
    );
}
/// Every registered program bounds its data-derived loops, and every
/// registration that exposes no program says why it is exempt.
///
/// A registration with no builder has no composed body to scan. That is only
/// true of an intrinsic, which is an emitter arm in each backend rather than
/// IR, and the registry already refuses an entry carrying neither a builder nor
/// a signature. Skipping an unbuilt registration in silence would let a
/// composition that lost its builder go unscanned and keep this test green, so
/// the exemption is asserted rather than assumed.
#[test]
fn every_registered_program_bounds_its_data_derived_loops() {
    let mut checked = 0_usize;
    let mut failures = Vec::new();

    for registration in inventory::iter::<OperationRegistration> {
        let Some(build) = registration.build else {
            assert!(
                registration.signature.is_some(),
                "`{}` exposes no program builder and no signature, so this scan cannot read its \
                 loop bounds and nothing else states them. Give it a builder, or a signature if it \
                 is an intrinsic.",
                registration.id
            );
            continue;
        };
        checked += 1;
        for finding in findings(build().entry()) {
            failures.push(format!("{}: {finding}", registration.id));
        }
    }

    assert!(
        checked > 0,
        "no registration exposed a program builder, so nothing was scanned"
    );
    assert!(
        failures.is_empty(),
        "{} unbounded loop bound(s) across {checked} registered program(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
