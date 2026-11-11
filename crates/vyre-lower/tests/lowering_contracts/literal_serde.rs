//! WHY: closes the class "a descriptor literal survives one serde format and
//! not another". The non-finite f32 escape was asked for in every format, and
//! its reader answers through `deserialize_any`, which a compact format cannot
//! answer at all. Three shipped paths use one: `vyre-debug` decodes a dumped
//! `KernelDescriptor` with bincode, the wgpu emitter writes that dump, and the
//! metal emitter hashes a descriptor through the same encoding. So the escape
//! made every f32 literal, finite ones included, undecodable there, and changed
//! the bytes a descriptor hash is taken over.
//!
//! Every variant of the literal union is round-tripped, not just the one the
//! escape is about: `control_for` matches the union exhaustively, so a variant
//! added to `LiteralValue` fails this file to compile until it is given a
//! control shape and therefore a case.
//!
//! The compact shape is judged against a control enum that mirrors
//! `LiteralValue` with plain fields and no custom representation, so the
//! assertion is "what the derived impl would have written" rather than a byte
//! string that goes stale when a variant is added. The f32 roster is the
//! IEEE-754 classes the format has to carry, which the format fixes rather than
//! this crate.
//!
//! What it does not catch: a format that is neither self-describing nor able to
//! answer `deserialize_f32`, and any loss in a dialect's own module text, which
//! is that dialect's golden corpus to catch.

use serde::{Deserialize, Serialize};
use vyre_lower::LiteralValue;

/// `LiteralValue`'s shape with no custom representation on any field.
///
/// Variant order matches, because a compact format writes a variant index.
#[derive(Debug, Serialize, Deserialize)]
enum PlainLiteral {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

/// The plain shape carrying the same value.
///
/// Matched exhaustively on purpose: a variant added to `LiteralValue` stops this
/// file from compiling rather than going untested.
fn control_for(literal: &LiteralValue) -> PlainLiteral {
    match literal {
        LiteralValue::U32(value) => PlainLiteral::U32(*value),
        LiteralValue::I32(value) => PlainLiteral::I32(*value),
        LiteralValue::F32(value) => PlainLiteral::F32(*value),
        LiteralValue::Bool(value) => PlainLiteral::Bool(*value),
    }
}

/// One value per IEEE-754 class the encoding has to carry.
fn every_f32_class() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.5,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xffc0_1234),
    ]
}

/// Every literal the union can hold, with the boundary values of each variant.
fn every_literal() -> Vec<LiteralValue> {
    let mut literals = vec![
        LiteralValue::U32(0),
        LiteralValue::U32(1),
        LiteralValue::U32(u32::MAX),
        LiteralValue::I32(i32::MIN),
        LiteralValue::I32(-1),
        LiteralValue::I32(0),
        LiteralValue::I32(i32::MAX),
        LiteralValue::Bool(false),
        LiteralValue::Bool(true),
    ];
    literals.extend(every_f32_class().into_iter().map(LiteralValue::F32));
    literals
}

/// Equality that reads an f32 by its bits, because a NaN is not equal to itself
/// and a NaN payload is exactly what the escape exists to carry.
fn identical(left: &LiteralValue, right: &LiteralValue) -> bool {
    match (left, right) {
        (LiteralValue::F32(left), LiteralValue::F32(right)) => left.to_bits() == right.to_bits(),
        (left, right) => left == right,
    }
}

fn compact(value: &impl Serialize) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .expect("Fix: a descriptor literal must encode in the format the descriptor dump uses.")
}

#[test]
fn every_literal_variant_round_trips_through_a_compact_format() {
    for literal in every_literal() {
        let bytes = compact(&literal);
        let (decoded, _): (LiteralValue, usize) = bincode::serde::decode_from_slice(
            &bytes,
            bincode::config::standard(),
        )
        .unwrap_or_else(|error| {
            panic!("Fix: `{literal:?}` must decode back out of a compact descriptor dump: {error}")
        });
        assert!(
            identical(&decoded, &literal),
            "Fix: a compact format must carry every literal exactly; `{literal:?}` came back as `{decoded:?}`."
        );
    }
}

#[test]
fn a_compact_literal_is_the_bytes_the_derived_shape_writes() {
    for literal in every_literal() {
        assert_eq!(
            compact(&literal),
            compact(&control_for(&literal)),
            "Fix: a compact descriptor encoding must stay byte-identical to the plain shape; `{literal:?}` changed it, which moves every descriptor hash taken over it."
        );
    }
}

#[test]
fn every_literal_variant_round_trips_through_a_self_describing_format() {
    for literal in every_literal() {
        let encoded = serde_json::to_string(&literal)
            .unwrap_or_else(|error| panic!("Fix: `{literal:?}` must encode as JSON: {error}"));
        let decoded: LiteralValue = serde_json::from_str(&encoded)
            .unwrap_or_else(|error| panic!("Fix: `{encoded}` must decode back: {error}"));
        assert!(
            identical(&decoded, &literal),
            "Fix: JSON must carry every literal exactly; `{literal:?}` came back as `{decoded:?}`."
        );
    }
}

#[test]
fn a_self_describing_format_escapes_only_the_numbers_it_cannot_write() {
    let finite = serde_json::to_string(&LiteralValue::F32(1.5))
        .expect("Fix: a finite literal must encode as JSON.");
    assert!(
        finite.contains("1.5"),
        "Fix: a finite literal must stay a plain JSON number, got `{finite}`."
    );
    for value in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
        let encoded = serde_json::to_string(&LiteralValue::F32(value))
            .expect("Fix: a non-finite literal must encode as JSON.");
        assert!(
            encoded.contains(&format!("{:08x}", value.to_bits())),
            "Fix: JSON has no non-finite number, so `{value:?}` must carry its bit pattern; got `{encoded}`."
        );
    }
}
