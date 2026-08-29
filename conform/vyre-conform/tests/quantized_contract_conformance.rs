//! What a program built from a quantized contract computes.
//!
//! WHY: the byte interpretation and the graph-level error bound of a quantized
//! value are proven here rather than asserted elsewhere. The contract
//! states where a field sits and what a code means; a builder turns that into
//! IR. Nothing in the contract's own suite executes that IR, so a lane law that
//! is self-consistent and wrong would pass it. Each case here evaluates the
//! emitted program through the parity oracle and compares it against the
//! independent decode in `vyre-spec`, which the contract shares no code with.
//!
//! What these cases do not prove: what a device computes. The oracle runs the
//! reference interpreter.

#![forbid(unsafe_code)]

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::numeric::{
    FieldTarget, GroupAxis, PackingOrder, QuantizedContract, ScalarFormat,
};
use vyre_reference::value::Value;
use vyre_spec::i4_to_i32;

/// Region identity of the fixture programs.
const OP_ID: &str = "vyre-conform::quantized_contract_conformance";

/// The INT4 layout the library packs.
fn packed_i4() -> QuantizedContract {
    QuantizedContract::symmetric(ScalarFormat::I4, ScalarFormat::F32, ScalarFormat::U32)
}

/// Pack `codes` into container words through the contract's own field law.
fn pack(contract: &QuantizedContract, codes: &[u32]) -> Vec<u32> {
    let count = u64::try_from(codes.len()).expect("Fix: the fixture packs fewer than 2^64 codes");
    let mut words = vec![
        0u32;
        usize::try_from(contract.container_words(count))
            .expect("Fix: the fixture packs into a host-sized buffer")
    ];
    for (index, code) in codes.iter().enumerate() {
        let field = contract.field(u64::try_from(index).expect("Fix: the index fits"));
        let mask = u32::try_from(field.mask).expect("Fix: a sub-word field masks into u32");
        let word = usize::try_from(field.word).expect("Fix: the word index fits");
        words[word] |= (code & mask) << field.shift_bits;
    }
    words
}

/// Little-endian bytes of `words`.
fn word_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A program that decodes every packed element into `target`.
fn decode_program(contract: &QuantizedContract, elements: u32, target: FieldTarget) -> Program {
    let index = Expr::gid_x();
    let body = vec![
        Node::let_bind("code", contract.load_field("packed", index.clone())),
        Node::store(
            "out",
            index.clone(),
            contract.decode_field(Expr::var("code"), target),
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("packed", 0, BufferAccess::ReadOnly, DataType::U32).with_count(
                u32::try_from(contract.container_words(u64::from(elements)))
                    .expect("Fix: the fixture declares fewer than 2^32 container words"),
            ),
            BufferDecl::output("out", 1, target.data_type()).with_count(elements),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(Expr::lt(index, Expr::u32(elements)), body)],
        )],
    )
}

/// A program that dequantizes every packed element by its group scale.
fn dequantize_program(contract: &QuantizedContract, elements: u32) -> Program {
    let index = Expr::gid_x();
    let group = u32::try_from(contract.group_elements()).expect("Fix: the group fits in u32");
    let body = vec![
        Node::let_bind("code", contract.load_field("packed", index.clone())),
        Node::let_bind(
            "value",
            contract.decode_field(Expr::var("code"), FieldTarget::Float32),
        ),
        Node::let_bind(
            "scale",
            Expr::load("scales", Expr::div(index.clone(), Expr::u32(group))),
        ),
        Node::store(
            "out",
            index.clone(),
            Expr::mul(Expr::var("value"), Expr::var("scale")),
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("packed", 0, BufferAccess::ReadOnly, DataType::U32).with_count(
                u32::try_from(contract.container_words(u64::from(elements)))
                    .expect("Fix: the fixture declares fewer than 2^32 container words"),
            ),
            BufferDecl::storage("scales", 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(elements.div_ceil(group)),
            BufferDecl::output("out", 2, DataType::F32).with_count(elements),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(Expr::lt(index, Expr::u32(elements)), body)],
        )],
    )
}

/// Run `program` over `inputs` and return the one output buffer's bytes.
fn run(program: &Program, inputs: Vec<Value>, output_bytes: usize) -> Vec<u8> {
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("Fix: the reference oracle must execute the quantized fixture program");
    outputs
        .iter()
        .map(vyre_reference::value::Value::to_bytes)
        .find(|bytes| bytes.len() == output_bytes)
        .expect("Fix: the fixture declares one output buffer of the stated size")
}

/// Decoded signed integers of `codes` under `contract`.
fn decoded_i32(contract: &QuantizedContract, codes: &[u32]) -> Vec<i32> {
    let elements = u32::try_from(codes.len()).expect("Fix: the fixture decodes fewer than 2^32");
    let program = decode_program(contract, elements, FieldTarget::SignedInt32);
    let bytes = run(
        &program,
        vec![Value::from(word_bytes(&pack(contract, codes)))],
        codes.len() * 4,
    );
    bytes
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// Decoded binary32 values of `codes` under `contract`.
fn decoded_f32(contract: &QuantizedContract, codes: &[u32]) -> Vec<f32> {
    let elements = u32::try_from(codes.len()).expect("Fix: the fixture decodes fewer than 2^32");
    let program = decode_program(contract, elements, FieldTarget::Float32);
    let bytes = run(
        &program,
        vec![Value::from(word_bytes(&pack(contract, codes)))],
        codes.len() * 4,
    );
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

#[test]
fn every_four_bit_code_decodes_to_the_value_the_spec_states() {
    let codes = (0..16u32).collect::<Vec<_>>();
    let expected = codes
        .iter()
        .map(|code| i4_to_i32(u8::try_from(*code).expect("a four-bit code fits in a byte")))
        .collect::<Vec<_>>();
    for order in [PackingOrder::LowFieldFirst, PackingOrder::HighFieldFirst] {
        let contract = packed_i4().packed(order);
        assert_eq!(
            decoded_i32(&contract, &codes),
            expected,
            "the emitted lane law must decode every code as the spec does under {order:?}"
        );
    }
}

#[test]
fn a_high_field_first_buffer_is_not_readable_as_a_low_field_first_one() {
    let codes = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let high = packed_i4().packed(PackingOrder::HighFieldFirst);
    let packed = pack(&high, &codes);
    let misread = {
        let low = packed_i4();
        let program = decode_program(&low, 8, FieldTarget::SignedInt32);
        let bytes = run(&program, vec![Value::from(word_bytes(&packed))], 32);
        bytes
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>()
    };
    let read = decoded_i32(&high, &codes);
    assert_ne!(
        misread, read,
        "two packing orders are two layouts, which is why one region may not read both"
    );
}

#[test]
fn a_cross_format_buffer_decodes_under_its_own_storage_family() {
    let signed_eight =
        QuantizedContract::symmetric(ScalarFormat::I8, ScalarFormat::F32, ScalarFormat::U32);
    assert_eq!(signed_eight.fields_per_container(), 4);
    let codes = [0u32, 1, 127, 128, 129, 255, 200, 64];
    let expected = codes
        .iter()
        .map(|code| i32::from(*code as u8 as i8))
        .collect::<Vec<_>>();
    assert_eq!(decoded_i32(&signed_eight, &codes), expected);

    let unsigned_eight =
        QuantizedContract::symmetric(ScalarFormat::U8, ScalarFormat::F32, ScalarFormat::U32);
    let unsigned = codes
        .iter()
        .map(|code| i32::try_from(*code).expect("a byte code fits"))
        .collect::<Vec<_>>();
    assert_eq!(
        decoded_i32(&unsigned_eight, &codes),
        unsigned,
        "an unsigned grid carries no sign to extend"
    );
}

#[test]
fn a_tail_block_decodes_without_reading_past_the_buffer() {
    let contract = packed_i4();
    let codes = (0..13u32).map(|index| index % 16).collect::<Vec<_>>();
    assert_eq!(
        contract.container_words(13),
        2,
        "thirteen nibbles occupy two words, the second partly filled"
    );
    let expected = codes
        .iter()
        .map(|code| i4_to_i32(u8::try_from(*code).expect("a four-bit code fits in a byte")))
        .collect::<Vec<_>>();
    assert_eq!(decoded_i32(&contract, &codes), expected);
}

#[test]
fn an_extreme_scale_dequantizes_within_the_stated_step() {
    let group = 8u32;
    let contract = packed_i4().grouped_by(vec![GroupAxis {
        axis: 0,
        extent: group,
    }]);
    let codes = (0..16u32).collect::<Vec<_>>();
    let scales = [1.0e-30f32, 1.0e30];
    let program = dequantize_program(&contract, 16);
    let bytes = run(
        &program,
        vec![
            Value::from(word_bytes(&pack(&contract, &codes))),
            Value::from(
                scales
                    .iter()
                    .flat_map(|scale| scale.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        ],
        64,
    );
    let values = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<_>>();
    for (index, value) in values.iter().enumerate() {
        let code = u8::try_from(index).expect("the fixture indexes 16 codes");
        let scale = f64::from(scales[index / usize::try_from(group).expect("the group fits")]);
        let oracle = f64::from(i4_to_i32(code)) * scale;
        assert!(
            value.is_finite(),
            "an extreme scale must not produce an infinity at index {index}"
        );
        let error = (f64::from(*value) - oracle).abs();
        let tolerance = oracle.abs() * f64::from(f32::EPSILON) + f64::MIN_POSITIVE;
        assert!(
            error <= tolerance,
            "index {index} dequantized to {value} against oracle {oracle}"
        );
    }
}

#[test]
fn a_nonzero_zero_point_centers_the_grid_it_was_calibrated_for() {
    let unsigned =
        QuantizedContract::symmetric(ScalarFormat::U8, ScalarFormat::F32, ScalarFormat::U32)
            .affine();
    unsigned
        .check()
        .expect("an affine per-tensor byte grid is readable");
    let codes = [0u32, 64, 128, 192, 255];
    let decoded = decoded_f32(&unsigned, &codes);
    let zero_point = 128.0f32;
    let scale = 0.01f32;
    for (index, code) in codes.iter().enumerate() {
        let centered = (decoded[index] - zero_point) * scale;
        let oracle = (f64::from(*code) - f64::from(zero_point)) * f64::from(scale);
        assert!(
            (f64::from(centered) - oracle).abs() <= oracle.abs() * f64::from(f32::EPSILON) + 1e-12,
            "code {code} centered to {centered} against oracle {oracle}"
        );
    }
    assert!(
        decoded[0] < zero_point && decoded[4] > zero_point,
        "an affine grid holds values on both sides of its zero point"
    );
}

#[test]
fn the_stated_accumulator_holds_a_sum_a_narrower_one_would_overflow() {
    let contract = packed_i4().accumulating_in(ScalarFormat::I32);
    let elements = 1024u32;
    let codes = vec![8u32; usize::try_from(elements).expect("Fix: the count fits")];
    let index = Expr::var("lane");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("packed", 0, BufferAccess::ReadOnly, DataType::U32).with_count(
                u32::try_from(contract.container_words(u64::from(elements)))
                    .expect("Fix: the fixture declares fewer than 2^32 container words"),
            ),
            BufferDecl::output("out", 1, DataType::I32).with_count(1),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::gid_x(), Expr::u32(0)),
                vec![
                    Node::let_bind("acc", Expr::i32(0)),
                    Node::loop_for(
                        "lane",
                        Expr::u32(0),
                        Expr::u32(elements),
                        vec![
                            Node::let_bind("code", contract.load_field("packed", index)),
                            Node::let_bind(
                                "value",
                                contract.decode_field(Expr::var("code"), FieldTarget::SignedInt32),
                            ),
                            Node::assign(
                                "acc",
                                Expr::add(
                                    Expr::var("acc"),
                                    Expr::mul(Expr::var("value"), Expr::var("value")),
                                ),
                            ),
                        ],
                    ),
                    Node::store("out", Expr::u32(0), Expr::var("acc")),
                ],
            )],
        )],
    );
    let bytes = run(
        &program,
        vec![Value::from(word_bytes(&pack(&contract, &codes)))],
        4,
    );
    let sum = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let oracle = i64::from(elements) * 64;
    assert_eq!(
        i64::from(sum),
        oracle,
        "the most negative four-bit code squares to 64, and the stated accumulator holds every one"
    );
    assert!(
        oracle > i64::from(i16::MAX),
        "the sum leaves a sixteen-bit accumulator, so the stated width is what keeps it exact"
    );
    #[allow(clippy::cast_possible_truncation)]
    let narrower = oracle as i16;
    assert_ne!(
        i64::from(narrower),
        oracle,
        "and a narrower accumulator wraps the same sum"
    );
}

#[test]
fn a_mixed_precision_read_of_one_buffer_names_one_value() {
    let contract = packed_i4();
    let codes = (0..16u32).collect::<Vec<_>>();
    let integers = decoded_i32(&contract, &codes);
    let floats = decoded_f32(&contract, &codes);
    for (index, value) in integers.iter().enumerate() {
        assert!(
            (floats[index] - *value as f32).abs() < f32::EPSILON,
            "reading the same code as an integer and as a float must name one value at {index}"
        );
    }
}

#[test]
fn a_dequantized_group_stays_inside_the_step_the_contract_states() {
    let group = 16u32;
    let contract = packed_i4().grouped_by(vec![GroupAxis {
        axis: 0,
        extent: group,
    }]);
    let reals = (0..16)
        .map(|index| -1.0 + 2.0 * f64::from(index) / 15.0)
        .collect::<Vec<_>>();
    let peak = reals
        .iter()
        .fold(0.0f64, |widest, value| widest.max(value.abs()));
    let scale = peak / 7.0;
    let codes = reals
        .iter()
        .map(|value| {
            let level = (value / scale).round().clamp(-8.0, 7.0);
            #[allow(clippy::cast_possible_truncation)]
            let code = level as i32;
            u32::from(u8::try_from(code & 0xF).expect("a four-bit code fits in a byte"))
        })
        .collect::<Vec<_>>();
    let decoded = decoded_f32(&contract, &codes);

    let step = contract.dequantization_measure().magnitude();
    for (index, value) in reals.iter().enumerate() {
        let dequantized = f64::from(decoded[index]) * scale;
        let error = (dequantized - value).abs();
        assert!(
            error <= step * peak + f64::EPSILON,
            "index {index} dequantized to {dequantized} against {value}, outside the stated step"
        );
    }
    let widest = reals
        .iter()
        .enumerate()
        .map(|(index, value)| (f64::from(decoded[index]) * scale - value).abs())
        .fold(0.0f64, f64::max);
    assert!(
        widest > 0.0,
        "a four-bit grid cannot hold sixteen distinct reals exactly, so the bound must be exercised"
    );
}
