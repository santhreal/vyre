//! Wire-format rev 5: the `BufferRef` tag, and reading rev 4.
//!
//! Rev 5 exists for exactly one reason, expression tag 22. A variant that
//! encodes but decodes to something else, or that silently loses its buffer
//! name, would turn a serialized program into a kernel reading the wrong
//! binding, which no other test in this crate would notice.
//!
//! The version tests pin the other half of the change. Rev 5 only APPENDS a
//! tag, so rev-4 bytes still mean what they meant and must keep decoding:
//! narrowing the accepted range to a single version would invalidate every
//! stored program and conformance certificate for no benefit. Anything older
//! predates the metadata layout and must still be refused.

use vyre_foundation::serial::wire::framing::{
    MIN_SUPPORTED_WIRE_FORMAT_VERSION, WIRE_FORMAT_VERSION,
};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// A program whose call passes a buffer by reference alongside a scalar.
fn program_with_a_buffer_reference() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("table", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::call(
                "test::wire::lookup",
                vec![Expr::buffer_ref("table"), Expr::u32(3)],
            ),
        )],
    )
}

/// A program with no rev-5 construct, so a rev-4 encoder could have produced
/// these exact bytes.
fn rev_four_shaped_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::load("a", Expr::u32(2)), Expr::u32(7)),
        )],
    )
}

/// Overwrite the little-endian u16 schema version in the header.
///
/// Layout is magic(4) then version(2), so the version occupies bytes 4 and 5.
fn with_version(mut bytes: Vec<u8>, version: u16) -> Vec<u8> {
    let raw = version.to_le_bytes();
    bytes[4] = raw[0];
    bytes[5] = raw[1];
    bytes
}

/// Collect the buffer named by every `Expr::BufferRef` reachable from the
/// program's entry, so the assertion checks the decoded name and not merely
/// that some `BufferRef` came back.
fn buffer_reference_names(program: &Program) -> Vec<String> {
    let mut found = Vec::new();
    for node in program.entry() {
        collect_from_node(node, &mut found);
    }
    found
}

fn collect_from_node(node: &Node, found: &mut Vec<String>) {
    match node {
        Node::Let { value, .. } | Node::Assign { value, .. } => collect_from_expr(value, found),
        Node::Store { index, value, .. } => {
            collect_from_expr(index, found);
            collect_from_expr(value, found);
        }
        Node::Block(inner) => inner.iter().for_each(|n| collect_from_node(n, found)),
        Node::Region { body, .. } => body.iter().for_each(|n| collect_from_node(n, found)),
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            collect_from_expr(cond, found);
            then.iter().for_each(|n| collect_from_node(n, found));
            otherwise.iter().for_each(|n| collect_from_node(n, found));
        }
        Node::Loop { from, to, body, .. } => {
            collect_from_expr(from, found);
            collect_from_expr(to, found);
            body.iter().for_each(|n| collect_from_node(n, found));
        }
        _ => {}
    }
}

fn collect_from_expr(expr: &Expr, found: &mut Vec<String>) {
    match expr {
        Expr::BufferRef { buffer } => found.push(buffer.to_string()),
        Expr::Call { args, .. } => args.iter().for_each(|a| collect_from_expr(a, found)),
        Expr::Load { index, .. } => collect_from_expr(index, found),
        Expr::BinOp { left, right, .. } => {
            collect_from_expr(left, found);
            collect_from_expr(right, found);
        }
        _ => {}
    }
}

/// The name is the whole payload of the tag. Losing or truncating it retargets
/// the callee at a different binding.
#[test]
fn a_buffer_reference_survives_a_round_trip_with_its_name() {
    let program = program_with_a_buffer_reference();
    let bytes = program.to_wire().expect("encode");
    let decoded = Program::from_wire(&bytes).expect("decode");

    assert_eq!(
        buffer_reference_names(&decoded),
        vec!["table".to_string()],
        "the decoded program must carry the same buffer reference"
    );
    assert!(
        program.structural_eq(&decoded),
        "the whole program must round-trip, not just the buffer reference"
    );
}

/// A scalar argument beside the buffer reference must not be swallowed by it:
/// the two occupy adjacent positions in the same argument list.
#[test]
fn the_scalar_argument_beside_it_round_trips_too() {
    let decoded = Program::from_wire(
        &program_with_a_buffer_reference()
            .to_wire()
            .expect("encode"),
    )
    .expect("decode");
    let dump = format!("{:?}", decoded.entry());
    assert!(
        dump.contains("LitU32(3)"),
        "the scalar argument must survive alongside the buffer reference: {dump}"
    );
}

/// The encoder must stamp rev 5. A program containing tag 22 labelled rev 4
/// would be read by a rev-4 decoder as an unknown tag at best.
#[test]
fn the_encoder_stamps_the_current_schema_version() {
    let bytes = program_with_a_buffer_reference().to_wire().expect("encode");
    assert_eq!(
        u16::from_le_bytes([bytes[4], bytes[5]]),
        WIRE_FORMAT_VERSION,
        "encoded programs must carry the current schema version"
    );
    assert_eq!(WIRE_FORMAT_VERSION, 5, "rev 5 introduced the BufferRef tag");
}

/// Rev 5 only appends a tag, so rev-4 bytes still mean what they meant.
/// Rejecting them would invalidate stored programs for no gain.
#[test]
fn a_rev_four_program_still_decodes() {
    let original = rev_four_shaped_program();
    let bytes = with_version(original.to_wire().expect("encode"), 4);
    let decoded = Program::from_wire(&bytes)
        .unwrap_or_else(|error| panic!("rev 4 must remain readable: {error}"));
    assert!(
        original.structural_eq(&decoded),
        "a rev-4 program must decode to the same program"
    );
}

/// The floor is a real floor. Rev 3 predates the metadata layout, so accepting
/// it would misread composition-safety flags rather than fail.
#[test]
fn a_rev_three_program_is_rejected_with_the_accepted_range() {
    let bytes = with_version(rev_four_shaped_program().to_wire().expect("encode"), 3);
    let error = Program::from_wire(&bytes)
        .expect_err("rev 3 predates the metadata layout and must be refused")
        .to_string();
    assert!(
        error.contains('3'),
        "the error must name the version it refused, got: {error}"
    );
    assert_eq!(
        MIN_SUPPORTED_WIRE_FORMAT_VERSION, 4,
        "rev 4 is the oldest layout this decoder understands"
    );
}

/// A version past the current one cannot be guessed at: it may use tags this
/// decoder has never seen.
#[test]
fn a_future_version_is_still_rejected() {
    let bytes = with_version(
        rev_four_shaped_program().to_wire().expect("encode"),
        WIRE_FORMAT_VERSION + 1,
    );
    assert!(
        Program::from_wire(&bytes).is_err(),
        "a newer schema version must be refused, not decoded on a guess"
    );
}
