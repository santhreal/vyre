//! Boundary tests for the reusable on-wire envelope (magic + version + sections).
//!
//! Every consumer of `WireWriter`/`WireReader` depends on correct framing.
//! These tests exercise truncation, bad magic, version mismatch, and
//! section-length overflow.

use vyre_foundation::serial::{section_len, EnvelopeError, WireReader, WireWriter};

const MAGIC: &[u8; 4] = b"TEST";
const VERSION: u32 = 1;

// ------------------------------------------------------------------
// Happy path
// ------------------------------------------------------------------

#[test]
fn envelope_round_trip_bytes() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_section(b"hello").unwrap();
    let bytes = writer.into_bytes();

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    assert_eq!(reader.read_section().unwrap(), b"hello");
}

#[test]
fn envelope_round_trip_words() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_words(&[0xDEADBEEF, 0xCAFEBABE]).unwrap();
    let bytes = writer.into_bytes();

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    assert_eq!(reader.read_words().unwrap(), vec![0xDEADBEEF, 0xCAFEBABE]);
}

#[test]
fn envelope_round_trip_u32() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_u32(42);
    let bytes = writer.into_bytes();

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    assert_eq!(reader.read_u32().unwrap(), 42);
}

#[test]
fn envelope_multiple_sections_in_order() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_section(b"first").unwrap();
    writer.write_section(b"second").unwrap();
    let bytes = writer.into_bytes();

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    assert_eq!(reader.read_section().unwrap(), b"first");
    assert_eq!(reader.read_section().unwrap(), b"second");
}

// ------------------------------------------------------------------
// Magic rejection
// ------------------------------------------------------------------

#[test]
fn envelope_rejects_bad_magic() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_section(b"x").unwrap();
    let bytes = writer.into_bytes();

    let err = WireReader::new(&bytes, b"WRNG", VERSION).unwrap_err();
    assert!(matches!(err, EnvelopeError::BadMagic { .. }));
}

#[test]
fn envelope_rejects_empty_input() {
    let err = WireReader::new(&[], MAGIC, VERSION).unwrap_err();
    assert!(matches!(err, EnvelopeError::Truncated { .. }));
}

#[test]
fn envelope_rejects_short_header() {
    let err = WireReader::new(&[0, 1, 2], MAGIC, VERSION).unwrap_err();
    assert!(matches!(err, EnvelopeError::Truncated { .. }));
}

// ------------------------------------------------------------------
// Version mismatch
// ------------------------------------------------------------------

#[test]
fn envelope_rejects_version_mismatch() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_section(b"x").unwrap();
    let bytes = writer.into_bytes();

    let err = WireReader::new(&bytes, MAGIC, 999).unwrap_err();
    assert!(matches!(err, EnvelopeError::VersionMismatch { .. }));
}

#[test]
fn envelope_version_zero_is_rejected_when_expecting_one() {
    let mut writer = WireWriter::new(MAGIC, 0);
    writer.write_section(b"x").unwrap();
    let bytes = writer.into_bytes();

    let err = WireReader::new(&bytes, MAGIC, VERSION).unwrap_err();
    assert!(matches!(err, EnvelopeError::VersionMismatch { .. }));
}

// ------------------------------------------------------------------
// Truncation detection
// ------------------------------------------------------------------

#[test]
fn envelope_rejects_truncated_section() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_section(b"hello world").unwrap();
    let mut bytes = writer.into_bytes();
    // Truncate inside the section body (after 8-byte header + 4-byte length).
    bytes.truncate(8 + 4 + 3);

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    let err = reader.read_section().unwrap_err();
    assert!(matches!(err, EnvelopeError::Truncated { .. }));
}

#[test]
fn envelope_rejects_truncated_word_array() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_words(&[1, 2, 3]).unwrap();
    let mut bytes = writer.into_bytes();
    // Truncate mid-word.
    bytes.truncate(bytes.len() - 1);

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    let err = reader.read_words().unwrap_err();
    assert!(matches!(err, EnvelopeError::Truncated { .. }));
}

#[test]
fn envelope_rejects_truncated_u32() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_u32(42);
    let mut bytes = writer.into_bytes();
    // Remove last byte of the u32.
    bytes.pop();

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    let err = reader.read_u32().unwrap_err();
    assert!(matches!(err, EnvelopeError::Truncated { .. }));
}

#[test]
fn envelope_rejects_header_only() {
    let bytes = WireWriter::new(MAGIC, VERSION).into_bytes();
    assert_eq!(bytes.len(), 8);

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    let err = reader.read_section().unwrap_err();
    assert!(matches!(err, EnvelopeError::Truncated { .. }));
}

// ------------------------------------------------------------------
// Section-too-large
// ------------------------------------------------------------------

/// A count past the `u32` prefix is refused, and the largest that fits is not.
///
/// WHY: both writers derive their prefix from `section_len`, and the two
/// tests this replaced reached it by allocating a slice longer than
/// `u32::MAX`. That cost four and sixteen gibibytes on a 64-bit host and did
/// not compile at all on a 32-bit one, where no slice is ever that long. The
/// bound is a function of the count, so it is proved by value on every host.
#[test]
fn envelope_section_length_is_bounded_by_the_u32_prefix() {
    assert_eq!(section_len(0).expect("an empty section fits the prefix"), 0);
    assert_eq!(
        section_len(u32::MAX as usize).expect("the largest section fits the prefix"),
        u32::MAX
    );

    let Some(past_bound) = (u32::MAX as usize).checked_add(1) else {
        // A 32-bit host cannot express a count past the prefix, so the
        // refusal is unreachable there rather than untested.
        return;
    };
    assert!(matches!(
        section_len(past_bound),
        Err(EnvelopeError::SectionTooLarge { len, max })
            if len == past_bound && max == u32::MAX as usize
    ));
}

// ------------------------------------------------------------------
// Edge cases
// ------------------------------------------------------------------

#[test]
fn envelope_empty_section_round_trips() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_section(b"").unwrap();
    let bytes = writer.into_bytes();

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    assert_eq!(reader.read_section().unwrap(), b"");
}

#[test]
fn envelope_empty_words_round_trip() {
    let mut writer = WireWriter::new(MAGIC, VERSION);
    writer.write_words(&[]).unwrap();
    let bytes = writer.into_bytes();

    let mut reader = WireReader::new(&bytes, MAGIC, VERSION).unwrap();
    assert_eq!(reader.read_words().unwrap(), Vec::<u32>::new());
}

#[test]
fn envelope_different_magics_are_independent() {
    let mut w1 = WireWriter::new(b"ONE!", 1);
    w1.write_section(b"a").unwrap();
    let bytes1 = w1.into_bytes();

    let mut w2 = WireWriter::new(b"TWO!", 1);
    w2.write_section(b"b").unwrap();
    let bytes2 = w2.into_bytes();

    assert_ne!(bytes1, bytes2);
    assert!(WireReader::new(&bytes1, b"TWO!", 1).is_err());
    assert!(WireReader::new(&bytes2, b"ONE!", 1).is_err());
}
