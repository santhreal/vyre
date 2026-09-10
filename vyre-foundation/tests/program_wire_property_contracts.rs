//! Executable property coverage for the untrusted Program wire parser.
//!
//! The libFuzzer target stays valuable for open-ended discovery, but normal
//! CI also needs deterministic generated/adversarial coverage for
//! `Program::from_wire`. These tests exercise valid generated programs,
//! arbitrary hostile bytes, truncations, digest-refreshed body mutations, and
//! explicit oversized LEB128 counts without requiring cargo-fuzz.

use proptest::prelude::*;
use vyre_foundation::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::serial::wire::framing::{MAGIC, WIRE_FORMAT_VERSION};
use vyre_foundation::serial::wire::MAX_NODES;

const HEADER_LEN: usize = 40;
const OPAQUE_ENDIAN_FIXED_FLAG: u16 = 1 << 2;

fn program_for_seed(seed: u32) -> Program {
    let count = (seed % 63).saturating_add(1);
    let buffers = vec![
        BufferDecl::read("input", 0, DataType::U32).with_count(count),
        BufferDecl::read_write("scratch", 1, DataType::U32).with_count(count),
        BufferDecl::output("out", 2, DataType::U32).with_count(count),
    ];
    Program::wrapped(
        buffers,
        [(seed % 32).saturating_add(1), 1, 1],
        entry_for_seed(seed, count),
    )
}

fn entry_for_seed(seed: u32, count: u32) -> Vec<Node> {
    let bounded_idx = Expr::rem(Expr::gid_x(), Expr::u32(count));
    match seed % 12 {
        0 => vec![Node::store("out", bounded_idx, Expr::u32(seed))],
        1 => vec![
            Node::let_bind("idx", bounded_idx.clone()),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::load("input", Expr::var("idx")),
            ),
        ],
        2 => vec![Node::If {
            cond: Expr::lt(Expr::gid_x(), Expr::buf_len("out")),
            then: vec![Node::store("out", bounded_idx.clone(), Expr::u32(1))],
            otherwise: vec![Node::store("scratch", bounded_idx, Expr::u32(0))],
        }],
        3 => vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32((seed % 8).saturating_add(1)),
            vec![Node::store(
                "scratch",
                Expr::var("i"),
                Expr::u32(seed.rotate_left(3)),
            )],
        )],
        4 => vec![Node::Block(vec![
            Node::let_bind("a", Expr::add(Expr::u32(seed), Expr::u32(7))),
            Node::store("out", bounded_idx, Expr::var("a")),
        ])],
        5 => vec![Node::barrier(), Node::Return],
        6 => vec![Node::let_bind(
            "mix",
            Expr::Select {
                cond: Box::new(Expr::eq(Expr::u32(seed & 1), Expr::u32(0))),
                true_val: Box::new(Expr::bitxor(Expr::u32(seed), Expr::u32(0xA5A5_5A5A))),
                false_val: Box::new(Expr::u32(seed.reverse_bits())),
            },
        )],
        7 => vec![Node::let_bind(
            "call",
            Expr::Call {
                op_id: "test.generated.identity".into(),
                args: vec![Expr::u32(seed), Expr::buf_len("input")],
            },
        )],
        8 => vec![Node::If {
            cond: Expr::ne(Expr::buf_len("input"), Expr::u32(0)),
            then: vec![Node::Block(vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::u32(seed),
            )])],
            otherwise: vec![],
        }],
        9 => vec![Node::let_bind(
            "cmp",
            Expr::BinOp {
                op: BinOp::Ge,
                left: Box::new(Expr::buf_len("input")),
                right: Box::new(Expr::u32(count)),
            },
        )],
        10 => vec![Node::Assign {
            name: "tmp".into(),
            value: Expr::add(Expr::u32(seed), Expr::load("input", Expr::u32(0))),
        }],
        _ => vec![Node::Region {
            generator: "generated.wire.property".into(),
            source_region: None,
            body: std::sync::Arc::new(vec![Node::store("out", Expr::u32(0), Expr::u32(seed))]),
        }],
    }
}

fn assert_decode_fail_closed(bytes: &[u8]) -> Result<(), TestCaseError> {
    match Program::from_wire(bytes) {
        Ok(program) => {
            let round = program.to_wire().map_err(|error| {
                TestCaseError::fail(format!(
                    "decoded program failed canonical re-encode: {error}"
                ))
            })?;
            let reparsed = Program::from_wire(&round).map_err(|error| {
                TestCaseError::fail(format!("canonical reparse failed: {error}"))
            })?;
            prop_assert!(
                program.structural_eq(&reparsed),
                "decoded program must be structurally stable after canonical reparse"
            );
        }
        Err(error) => {
            let msg = error.to_string();
            prop_assert!(
                msg.contains("Fix:"),
                "wire parser errors must be actionable, got: {msg}"
            );
        }
    }
    Ok(())
}

fn refresh_digest(bytes: &mut [u8]) {
    let digest = blake3::hash(&bytes[HEADER_LEN..]);
    bytes[8..HEADER_LEN].copy_from_slice(digest.as_bytes());
}

fn vir0_envelope(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&WIRE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&OPAQUE_ENDIAN_FIXED_FLAG.to_le_bytes());
    out.extend_from_slice(blake3::hash(body).as_bytes());
    out.extend_from_slice(body);
    out
}

fn put_leb_u64(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn generated_programs_round_trip_byte_stably(seed in any::<u32>()) {
        let program = program_for_seed(seed);
        let bytes = program.to_wire().map_err(|error| {
            TestCaseError::fail(format!("generated program must encode: {error}"))
        })?;
        let decoded = Program::from_wire(&bytes).map_err(|error| {
            TestCaseError::fail(format!("encoded generated program must decode: {error}"))
        })?;
        prop_assert!(
            program.structural_eq(&decoded),
            "generated program changed structurally after wire round-trip"
        );
        let reencoded = decoded.to_wire().map_err(|error| {
            TestCaseError::fail(format!("decoded generated program must re-encode: {error}"))
        })?;
        prop_assert_eq!(bytes, reencoded, "canonical wire bytes must be stable");
    }

    #[test]
    fn arbitrary_hostile_bytes_never_panic_and_fail_closed(bytes in proptest::collection::vec(any::<u8>(), 0..=4096)) {
        assert_decode_fail_closed(&bytes)?;
    }

    #[test]
    fn truncations_of_generated_programs_never_panic(seed in any::<u32>(), cut in 0usize..=4096) {
        let bytes = program_for_seed(seed).to_wire().map_err(|error| {
            TestCaseError::fail(format!("generated program must encode before truncation: {error}"))
        })?;
        let end = cut.min(bytes.len());
        assert_decode_fail_closed(&bytes[..end])?;
    }

    #[test]
    fn digest_refreshed_body_mutations_never_panic(seed in any::<u32>(), flip_offset in 0usize..=4096, bit in 0u8..=7) {
        let mut bytes = program_for_seed(seed).to_wire().map_err(|error| {
            TestCaseError::fail(format!("generated program must encode before mutation: {error}"))
        })?;
        prop_assume!(bytes.len() > HEADER_LEN);
        let index = HEADER_LEN + (flip_offset % (bytes.len() - HEADER_LEN));
        bytes[index] ^= 1u8 << bit;
        refresh_digest(&mut bytes);
        assert_decode_fail_closed(&bytes)?;
    }
}

#[test]
fn oversized_node_count_is_rejected_before_allocation() {
    let mut body = Vec::new();
    put_leb_u64(
        &mut body,
        u64::try_from(MAX_NODES)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    );
    let err = Program::from_wire(&vir0_envelope(&body))
        .expect_err("Fix: oversized node count must be rejected")
        .to_string();
    assert!(
        err.contains("node count") && err.contains("Fix:"),
        "oversized node count error must name the field and include Fix:, got: {err}"
    );
}

#[test]
fn unterminated_leb128_node_count_is_rejected() {
    let body = vec![0x80; 11];
    let err = Program::from_wire(&vir0_envelope(&body))
        .expect_err("Fix: unterminated LEB128 must be rejected")
        .to_string();
    assert!(
        err.contains("LEB128") && err.contains("Fix:"),
        "unterminated LEB128 error must name the encoding and include Fix:, got: {err}"
    );
}

/// A dense memory region refuses every element type whose own payload
/// describes further structure.
///
/// WHY: the region format writes one element tag plus a count, and the
/// element's payload is written by a per-variant arm beside it. Four arms
/// exist. Six variants need one. The two the list omitted, `Vec` and
/// `TensorShaped`, encoded to a bare tag whose element and count never
/// reached the blob, and the decoder refused the result: a Program that
/// encoded without error and could not be read back.
///
/// What this does not catch: a variant added to `DataType` after this test
/// was written. That is closed by `dense_element_tag`, whose match is
/// exhaustive with no catch-all, so a new variant stops the crate compiling.
#[test]
fn a_structured_element_is_refused_by_the_dense_region_encoder() {
    let structured = [
        (
            "Vec",
            DataType::Vec {
                element: Box::new(DataType::F32),
                count: 4,
            },
        ),
        (
            "TensorShaped",
            DataType::TensorShaped {
                element: Box::new(DataType::F32),
                shape: [2u32, 3].into_iter().collect(),
            },
        ),
        (
            "SparseCsr",
            DataType::SparseCsr {
                element: Box::new(DataType::F32),
            },
        ),
        (
            "SparseCoo",
            DataType::SparseCoo {
                element: Box::new(DataType::F32),
            },
        ),
        (
            "SparseBsr",
            DataType::SparseBsr {
                element: Box::new(DataType::F32),
                block_rows: 2,
                block_cols: 2,
            },
        ),
        (
            "DeviceMesh",
            DataType::DeviceMesh {
                axes: [2u32, 2].into_iter().collect(),
            },
        ),
    ];

    for (name, element) in structured {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("input", 0, element).with_count(4),
                BufferDecl::output("out", 1, DataType::U32).with_count(4),
            ],
            [1, 1, 1],
            vec![Node::Store {
                buffer: "out".into(),
                index: Expr::u32(0),
                value: Expr::u32(1),
            }],
        );
        let error = program.to_wire().expect_err(&format!(
            "{name} is not a dense region element and must be refused at encode"
        ));
        let rendered = error.to_string();
        assert!(
            rendered.contains("not valid buffer elements"),
            "{name}: the refusal must name the domain it enforces, got: {rendered}"
        );
    }
}

/// Every flat element type a dense region accepts survives a wire roundtrip.
///
/// WHY: refusing the structured types is only half the contract. Without
/// this, tightening the refusal to reject everything would pass the test
/// above.
#[test]
fn a_flat_element_survives_the_dense_region_roundtrip() {
    let flat = [
        DataType::U32,
        DataType::I32,
        DataType::U64,
        DataType::I64,
        DataType::U8,
        DataType::U16,
        DataType::I8,
        DataType::I16,
        DataType::I4,
        DataType::Bool,
        DataType::Bytes,
        DataType::F16,
        DataType::BF16,
        DataType::F32,
        DataType::F64,
        DataType::F8E4M3,
        DataType::F8E5M2,
        DataType::FP4,
        DataType::NF4,
        DataType::Vec2U32,
        DataType::Vec4U32,
        DataType::Tensor,
        DataType::Array { element_size: 12 },
    ];

    for element in flat {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("input", 0, element.clone()).with_count(4),
                BufferDecl::output("out", 1, DataType::U32).with_count(4),
            ],
            [1, 1, 1],
            vec![Node::Store {
                buffer: "out".into(),
                index: Expr::u32(0),
                value: Expr::u32(1),
            }],
        );
        let encoded = program
            .to_wire()
            .unwrap_or_else(|error| panic!("{element} is a dense element and must encode: {error}"));
        let decoded = Program::from_wire(&encoded)
            .unwrap_or_else(|error| panic!("{element} must decode back: {error}"));
        let read_back = decoded
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "input")
            .expect("input buffer survives the roundtrip")
            .element();
        assert_eq!(
            read_back, element,
            "{element} must decode to the element it was declared with"
        );
    }
}
