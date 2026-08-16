//! WHY: closes the class "a descriptor literal survives one serde format and
//! not another". The non-finite f32 escape was asked for in every format, and
//! its reader answers through `deserialize_any`, which a compact format cannot
//! answer at all. Three shipped paths use one: `vyre-debug` decodes a dumped
//! `KernelDescriptor` with bincode, the wgpu emitter writes that dump, and the
//! metal emitter hashes a descriptor through the same encoding. So the escape
//! made every f32 literal, finite ones included, undecodable there, and changed
//! the bytes a descriptor hash is taken over.
//!
//! The compact shape is judged against a control enum that mirrors
//! `LiteralValue` with a plain `f32` field and no custom representation, so the
//! assertion is "what the derived impl would have written" rather than a byte
//! string that goes stale when a variant is added. The value roster is the
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
    #[allow(dead_code)]
    U32(u32),
    #[allow(dead_code)]
    I32(i32),
    F32(f32),
    #[allow(dead_code)]
    Bool(bool),
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

fn compact(value: &impl Serialize) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .expect("Fix: a descriptor literal must encode in the format the descriptor dump uses.")
}

#[test]
fn every_f32_literal_class_round_trips_through_a_compact_format() {
    for value in every_f32_class() {
        let bytes = compact(&LiteralValue::F32(value));
        let (decoded, _): (LiteralValue, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap_or_else(
                |error| {
                    panic!(
                        "Fix: `{value:?}` must decode back out of a compact descriptor dump: {error}"
                    )
                },
            );
        let LiteralValue::F32(returned) = decoded else {
            panic!("Fix: an f32 literal must decode as an f32 literal, got {decoded:?}");
        };
        assert_eq!(
            returned.to_bits(),
            value.to_bits(),
            "Fix: a compact format must carry every f32 bit pattern exactly; `{value:?}` came back as `{returned:?}`."
        );
    }
}

#[test]
fn a_compact_f32_literal_is_the_bytes_the_derived_shape_writes() {
    for value in every_f32_class() {
        assert_eq!(
            compact(&LiteralValue::F32(value)),
            compact(&PlainLiteral::F32(value)),
            "Fix: a compact descriptor encoding must stay byte-identical to the plain shape; `{value:?}` changed it, which moves every descriptor hash taken over it."
        );
    }
}

#[test]
fn a_self_describing_format_keeps_the_non_finite_escape() {
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
        let decoded: LiteralValue = serde_json::from_str(&encoded)
            .unwrap_or_else(|error| panic!("Fix: `{encoded}` must decode back: {error}"));
        let LiteralValue::F32(returned) = decoded else {
            panic!("Fix: an f32 literal must decode as an f32 literal, got {decoded:?}");
        };
        assert_eq!(returned.to_bits(), value.to_bits());
    }
}
