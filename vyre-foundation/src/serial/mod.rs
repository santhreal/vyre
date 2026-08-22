//! IR serialization formats.
//!
//! Vyre programs are frozen data structures that must survive transmission,
//! caching, and versioning. This module defines the two stable serialization
//! formats: a compact binary wire format for machines and a canonical text
//! format for humans.

/// Canonical text representation.
///
/// The text format is human-readable and version-agnostic. It is used for
/// debugging, logging, and diffing IR in tests.
pub mod text;

/// Binary wire format.
///
/// The wire format is a compact little-endian byte stream designed for
/// network transmission and on-disk caching. Every validated `Program` can
/// be round-tripped through this format without loss.
pub mod wire;

/// Output set serialization.
///
/// Persistent encoding of which buffers are writable outputs, used by
/// the wire format and persistent cache layers.
pub mod output_set;

/// Reusable on-wire envelope for magic, version, length-prefixed sections,
/// and word arrays. Higher-layer types such as `CompiledDfa` in
/// `vyre-primitives`, `GpuLiteralSet` in `vyre-libs`, and scan databases in
/// `vyre-scan` compose this primitive instead of reimplementing framing,
/// version, and truncation handling.
pub(crate) mod envelope;
pub use envelope::{EnvelopeError, WireReader, WireWriter};

/// Round-trip and corruption assertions every wire-format type is held to.
///
/// A consumer implements [`wire_round_trip::WireRoundTrip`] for its type and
/// calls [`wire_round_trip::assert_envelope_roundtrip`] once instead of
/// restating the same five encode/decode tests.
pub mod wire_round_trip;

pub(crate) fn put_leb_u32(out: &mut Vec<u8>, value: u32) {
    put_leb_u64(out, u64::from(value));
}

pub(crate) fn put_leb_u64(out: &mut Vec<u8>, mut value: u64) {
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
