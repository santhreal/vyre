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

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::loop_bounds::data_derived_loop_bounds_in;
use vyre_foundation::operation::OperationRegistration;
use vyre_libs::state_machine::TableStateMachineComposer;
use vyre_reference::value::Value;
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

/// The CSR traversal composer bounds edge loops against targets and mask buffers.
#[cfg(feature = "graph")]
#[test]
fn csr_traversal_scan_body_bounds_its_trip_counts() {
    let composer = vyre_libs::csr::CsrTraversalComposer::forward("test", 10, 20, 0xFF);
    let row_loop = composer.emit_row_bounds_and_loop(
        Expr::u32(0),
        "edge",
        vec![Node::store("out", Expr::var("edge"), Expr::u32(1))],
    );
    assert!(
        findings(&row_loop).is_empty(),
        "emit_row_bounds_and_loop: {:?}",
        findings(&row_loop)
    );

    let expand = composer.emit_edge_expand(
        "front_out",
        Expr::u32(0),
        |idx| idx,
        Vec::new,
    );
    assert!(
        findings(&expand).is_empty(),
        "emit_edge_expand: {:?}",
        findings(&expand)
    );

    let backward = composer.emit_backward_scan_full(
        Expr::u32(0),
        "front_in",
        "front_out",
        Vec::new,
        Vec::new,
    );
    assert!(
        findings(&backward).is_empty(),
        "emit_backward_scan_full: {:?}",
        findings(&backward)
    );
}

#[test]
fn hostile_state_machine_linear_scan_returns_in_bounded_steps() {
    let composer = TableStateMachineComposer::new("transitions");
    let body = composer.linear_scan_body(
        "input",
        "accept",
        "matches",
        Expr::load("lengths", Expr::u32(0)),
    );

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("transitions", 1, BufferAccess::ReadOnly, DataType::U32).with_count(256),
            BufferDecl::storage("lengths", 2, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("accept", 3, BufferAccess::ReadOnly, DataType::U32).with_count(16),
            BufferDecl::storage("matches", 4, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ],
        [128, 1, 1],
        vec![wrap_anonymous_region("test_state_machine", body)],
    );

    let hostile_len: u32 = u32::MAX;
    let inputs = vec![
        Value::from(vec![1u32, 2, 3, 4].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
        Value::from(vec![0u32; 256].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
        Value::from(hostile_len.to_le_bytes().to_vec()),
        Value::from(vec![0u32; 16].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
        Value::from(vec![0u32; 4].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
    ];

    let (outputs, steps) = vyre_reference::reference_eval_step_count(&program, &inputs)
        .expect("hostile state machine run must succeed");

    assert!(
        steps < 5_000,
        "hostile trip count must terminate in bounded steps; took {steps} steps"
    );
    assert!(!outputs.is_empty());
}

#[cfg(feature = "decode")]
#[test]
fn hostile_ziftsieve_literal_copy_returns_in_bounded_steps() {
    use vyre_libs::decode::ziftsieve::{ziftsieve_literal_copy, ZiftsieveBuffers, ZiftsieveExtents};

    let buffers = ZiftsieveBuffers {
        input: "input",
        output: "output",
        seq_literal_start: "seq_start",
        seq_literal_len: "seq_len",
        seq_literal_offset: "seq_offset",
    };
    let extents = ZiftsieveExtents {
        input_len: 4,
        seq_count: 1,
        max_output: 4,
    };
    let program = ziftsieve_literal_copy(buffers, extents);

    let hostile_len: u32 = u32::MAX;
    let inputs = vec![
        Value::from(vec![10u32, 20, 30, 40].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
        Value::from(0u32.to_le_bytes().to_vec()),
        Value::from(hostile_len.to_le_bytes().to_vec()),
        Value::from(0u32.to_le_bytes().to_vec()),
        Value::from(vec![0u32; 4].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
    ];

    let (outputs, steps) = vyre_reference::reference_eval_step_count(&program, &inputs)
        .expect("hostile ziftsieve run must succeed");

    assert!(
        steps < 5_000,
        "hostile trip count must terminate in bounded steps; took {steps} steps"
    );
    assert!(!outputs.is_empty());
}

#[cfg(feature = "decode")]
#[test]
fn in_contract_ziftsieve_literal_copy_produces_identical_bytes() {
    use vyre_libs::decode::ziftsieve::{ziftsieve_literal_copy, ZiftsieveBuffers, ZiftsieveExtents};

    let buffers = ZiftsieveBuffers {
        input: "input",
        output: "output",
        seq_literal_start: "seq_start",
        seq_literal_len: "seq_len",
        seq_literal_offset: "seq_offset",
    };
    let extents = ZiftsieveExtents {
        input_len: 4,
        seq_count: 1,
        max_output: 4,
    };
    let program = ziftsieve_literal_copy(buffers, extents);

    let inputs = vec![
        Value::from(vec![10u32, 20, 30, 40].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
        Value::from(0u32.to_le_bytes().to_vec()),
        Value::from(3u32.to_le_bytes().to_vec()),
        Value::from(0u32.to_le_bytes().to_vec()),
        Value::from(vec![0u32; 4].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),
    ];

    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("in-contract ziftsieve run must succeed");
    let out_bytes = outputs[0].to_bytes();
    let words: Vec<u32> = out_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0..3], [10, 20, 30]);
    assert_eq!(words[3], 0);
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
