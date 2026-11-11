//! Generic round-trip / robustness assertion helpers for any
//! wire-format consumer.
//!
//! Every type that ships its own `to_bytes` / `from_bytes` pair on top
//! of this envelope used to write the same five tests:
//!   1. `round_trip`
//!   2. `rejects_bad_magic`
//!   3. `rejects_version_mismatch`
//!   4. `rejects_truncated_header`
//!   5. `rejects_truncated_section`
//!
//! These helpers reduce that to one call per type. Consumers call
//! `assert_envelope_roundtrip(&value)` and the helper drives the full
//! suite. The bound `T: WireRoundTrip` is provided by consumers as a
//! thin trait that exposes the type's `to_bytes` / `from_bytes` plus
//! its declared magic + version.

use crate::serial::envelope::{EnvelopeError, WireWriter};

/// Adapter trait consumers implement to plug their wire format
/// into [`assert_envelope_roundtrip`]. The `to_bytes` and
/// `from_bytes` methods are forwarded to the type's own; the
/// `MAGIC` / `VERSION` consts let the helpers fabricate
/// deliberately-corrupted blobs.
pub trait WireRoundTrip: Sized {
    /// Wire-format magic the type stamps on every blob.
    const MAGIC: [u8; 4];
    /// Wire version the type stamps on every blob.
    const VERSION: u32;
    /// Encoder error type. Not exercised here  -  consumers pre-
    /// validate that `to_bytes` returns `Ok` for the sample.
    type EncodeError: std::fmt::Debug;
    /// Decoder error type. Used to confirm that mutated blobs
    /// surface as typed errors instead of panics.
    type DecodeError: std::fmt::Debug;

    /// Encode a sample value.
    ///
    /// # Errors
    /// Forwarded from the type's own encoder.
    fn to_bytes(&self) -> Result<Vec<u8>, Self::EncodeError>;

    /// Decode a previously-encoded blob.
    ///
    /// # Errors
    /// Forwarded from the type's own decoder.
    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::DecodeError>;

    /// Comparison hook so the helper can assert structural equality
    /// after a round trip without requiring `PartialEq` on the type
    /// itself (some engines hold non-comparable buffers / programs).
    fn structurally_eq(&self, other: &Self) -> bool;
}

/// Drive the standard wire-format assertion suite against `sample`.
///
/// Asserts:
///   - encode succeeds
///   - decode of the encoded bytes returns a value that
///     `structurally_eq`s the original
///   - mutating the magic byte produces a typed decode error
///   - mutating the version dword produces a typed decode error
///   - truncating the trailing byte produces a typed decode error
///   - feeding an 8-byte buffer (header only, zero sections) is a
///     decoder concern  -  helper does NOT assert success/failure
///     because section-counts vary by consumer.
///
/// Intentionally panics on assertion failure (this is a test
/// helper, not a runtime path).
///
/// # Panics
///
/// Panics when encoding/decoding fails for the supplied valid sample, when
/// the encoded header is malformed, or when corruption/truncation does not
/// surface as a typed decode error.
pub fn assert_envelope_roundtrip<T>(sample: &T)
where
    T: WireRoundTrip + std::fmt::Debug,
{
    let encoded = sample.to_bytes();
    assert!(
        encoded.is_ok(),
        "Fix: encode sample; restore this invariant before continuing: {encoded:?}"
    );
    let Ok(bytes) = encoded else {
        return;
    };
    assert!(
        bytes.len() >= 8,
        "wire blob must include at least the 8-byte header"
    );
    assert_eq!(
        &bytes[0..4],
        T::MAGIC.as_slice(),
        "magic mismatch in encoded blob"
    );
    let version_field = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let expected_version = T::VERSION;
    assert!(
        version_field == expected_version,
        "version mismatch in encoded blob: got {version_field}, expected {expected_version}"
    );

    let decoded = T::from_bytes(&bytes);
    assert!(
        decoded.is_ok(),
        "Fix: decode round trip; restore this invariant before continuing: {decoded:?}"
    );
    let Ok(back) = decoded else {
        return;
    };
    assert!(
        sample.structurally_eq(&back),
        "round-tripped value diverges from original"
    );

    // Mutate magic.
    let mut mutated = bytes.clone();
    mutated[0] ^= 0xFF;
    assert!(
        T::from_bytes(&mutated).is_err(),
        "mutated magic must surface as a typed error"
    );

    // Mutate version.
    let mut mutated = bytes.clone();
    let bumped = T::VERSION.wrapping_add(1);
    mutated[4..8].copy_from_slice(&bumped.to_le_bytes());
    assert!(
        T::from_bytes(&mutated).is_err(),
        "mutated version must surface as a typed error"
    );

    // Truncate one byte off the tail.
    if bytes.len() > 8 {
        let truncated = &bytes[..bytes.len() - 1];
        assert!(
            T::from_bytes(truncated).is_err(),
            "truncated trailing byte must surface as a typed error"
        );
    }
}

/// Helper for tests that want to fabricate blobs with arbitrary
/// magic + version. Returns a header-only buffer (no sections).
/// Useful for asserting that consumers reject empty-section blobs
/// when their schema requires N sections.
#[must_use]
pub fn header_only(magic: &[u8; 4], version: u32) -> Vec<u8> {
    WireWriter::new(magic, version).into_bytes()
}

/// Confirm that the `EnvelopeError` matches an expected variant
/// (without requiring the consumer's wrapper enum to expose
/// `PartialEq`).
///
/// # Panics
///
/// Panics when `err` does not match the expected envelope-error category.
pub fn assert_envelope_error_kind(err: &EnvelopeError, kind: ExpectedEnvelopeError) {
    let matches = matches!(
        (err, kind),
        (
            EnvelopeError::Truncated { .. },
            ExpectedEnvelopeError::Truncated
        ) | (
            EnvelopeError::BadMagic { .. },
            ExpectedEnvelopeError::BadMagic
        ) | (
            EnvelopeError::VersionMismatch { .. },
            ExpectedEnvelopeError::VersionMismatch
        ) | (
            EnvelopeError::SectionTooLarge { .. },
            ExpectedEnvelopeError::SectionTooLarge
        )
    );
    assert!(
        matches,
        "expected envelope error kind {kind:?}, got {err:?}"
    );
}

/// Variant tags for [`assert_envelope_error_kind`]. Mirrors
/// [`EnvelopeError`] but is decoupled from the consumer's wrapper
/// enum so they can match on it without re-exporting the
/// variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedEnvelopeError {
    /// `EnvelopeError::Truncated`
    Truncated,
    /// `EnvelopeError::BadMagic`
    BadMagic,
    /// `EnvelopeError::VersionMismatch`
    VersionMismatch,
    /// `EnvelopeError::SectionTooLarge`
    SectionTooLarge,
}
