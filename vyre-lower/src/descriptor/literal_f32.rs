//! Serde representation of an f32 literal.
//!
//! A descriptor travels to a materializer inside a target-module bundle, and
//! that bundle is JSON, which has no non-finite number. `serde_json` writes
//! `f32::NEG_INFINITY` as `null` and then refuses to read `null` back as an
//! f32, so every op whose literal pool holds an infinity produced a bundle no
//! backend could decode: `vyre-libs::nn::top_k` and `nn::softmax_top_k` seed a
//! running maximum with negative infinity and failed target-module decode with
//! `invalid type: null, expected f32`.
//!
//! Only the values the plain encoding could not represent change shape. A
//! finite literal is still written as a number, byte for byte what the derived
//! impl wrote, so no other serde surface that carries a descriptor changes at
//! all. A non-finite literal is written as its IEEE-754 bit pattern in hex,
//! which is exact for every f32 including each NaN payload, and reads back
//! through `f32::from_bits`.
//!
//! The escape is asked for only where the format is self-describing and the
//! plain encoding is lossy. A compact format cannot answer `deserialize_any`.
//! Descriptor dumps and descriptor hashes both use compact binary encoding,
//! which carries all 32 bits in a number and wants no escape at all. It reads
//! and writes exactly what the derived implementation did, byte for byte: a
//! descriptor hash keeps its value and a dumped descriptor stays readable
//! across this change.

use std::fmt;

use serde::de::{Unexpected, Visitor};
use serde::{Deserializer, Serializer};

/// Radix prefix for the non-finite escape. Present so a reader can tell an
/// escaped bit pattern from a decimal literal someone wrote by hand.
const BITS_PREFIX: &str = "0x";

pub(super) fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
    if value.is_finite() || !serializer.is_human_readable() {
        return serializer.serialize_f32(*value);
    }
    serializer.serialize_str(&format!("{BITS_PREFIX}{:08x}", value.to_bits()))
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    if !deserializer.is_human_readable() {
        return deserializer.deserialize_f32(LiteralVisitor);
    }
    deserializer.deserialize_any(LiteralVisitor)
}

struct LiteralVisitor;

impl Visitor<'_> for LiteralVisitor {
    type Value = f32;

    fn expecting(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "a finite f32 number, or a non-finite one as `{BITS_PREFIX}` and eight hex digits"
        )
    }

    fn visit_f32<E: serde::de::Error>(self, value: f32) -> Result<f32, E> {
        Ok(value)
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<f32, E> {
        Ok(value as f32)
    }

    fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<f32, E> {
        Ok(value as f32)
    }

    fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<f32, E> {
        Ok(value as f32)
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<f32, E> {
        let digits = value
            .strip_prefix(BITS_PREFIX)
            .ok_or_else(|| E::invalid_value(Unexpected::Str(value), &self))?;
        let bits = u32::from_str_radix(digits, 16)
            .map_err(|_| E::invalid_value(Unexpected::Str(value), &self))?;
        let value = f32::from_bits(bits);
        if value.is_finite() {
            // A finite value has a number encoding, so accepting it here too
            // would give one literal two spellings and break the bundle's
            // canonical-bytes check on re-encode.
            return Err(E::invalid_value(
                Unexpected::Float(f64::from(value)),
                &"a non-finite f32 bit pattern; write a finite literal as a number",
            ));
        }
        Ok(value)
    }
}
