//! Adversarial tests for wire-format corruption detection.
//!
//! The wire decoder must reject tampered or corrupted payloads with
//! structured errors rather than panics or silent acceptance.

use vyre_test_support::wire_hostile_inputs::{decode_error_string, minimal_program_bytes};

#[test]
fn wire_decoder_rejects_corrupted_checksum() {
    let mut bytes = minimal_program_bytes();
    // Corrupt a single byte in the body (after the 40-byte header).
    // The header contains: magic(4) + version(2) + flags(2) + checksum(32) = 40 bytes.
    if bytes.len() > 45 {
        bytes[45] = bytes[45].wrapping_add(1);
    }

    let error = decode_error_string(&bytes, "corrupt checksum");
    assert!(
        error.contains("IntegrityMismatch"),
        "a tampered body must be reported as an integrity mismatch. The previous form of this \
         assertion also accepted any message containing `Fix:`, which every error in this \
         decoder carries, so it passed whatever the decoder said. Got: {error}"
    );
}

#[test]
fn wire_decoder_rejects_truncated_body() {
    let bytes = minimal_program_bytes();
    // Truncate the body but leave the header intact  -  the checksum
    // will still be valid for a shorter body, but the decoder should
    // hit EOF before finishing node parsing.
    let truncated = &bytes[..bytes.len().saturating_sub(4)];

    let error = decode_error_string(truncated, "truncated body");
    assert!(
        error.contains("TruncatedPayload") || error.contains("IntegrityMismatch"),
        "a body short of what the header covers must be reported as truncation or as an \
         integrity mismatch, not as some other fault, got: {error}"
    );
}

#[test]
fn wire_decoder_rejects_wrong_magic() {
    let mut bytes = minimal_program_bytes();
    // Corrupt the magic bytes at the start.
    if bytes.len() >= 4 {
        bytes[0] = b'X';
        bytes[1] = b'X';
        bytes[2] = b'X';
        bytes[3] = b'X';
    }

    let error = decode_error_string(&bytes, "wrong magic");
    assert!(
        error.contains("MagicMismatch"),
        "bytes that are long enough for a magic but do not carry VIR0 must be reported as a \
         magic mismatch, not as truncation: the caller holds a complete blob of the wrong \
         format and re-fetching it changes nothing. Got: {error}"
    );
}

#[test]
fn wire_decoder_rejects_empty_input() {
    let error = decode_error_string(&[], "empty input");
    assert!(
        error.contains("TruncatedPayload"),
        "an input too short to hold the magic must be reported as truncation, not as a magic \
         mismatch: there are no magic bytes to compare. Got: {error}"
    );
    assert!(
        !error.contains("MagicMismatch"),
        "truncation and magic mismatch are separate repairs and must not be reported \
         together. Got: {error}"
    );
}

/// WHY: the two reports are chosen by a length comparison against the magic,
/// so the byte on each side of that comparison is where a `<` written as `<=`
/// survives every other case in this file. One byte short is truncation; the
/// exact length with wrong content is a mismatch and never truncation.
#[test]
fn the_byte_either_side_of_the_magic_length_picks_a_different_report() {
    let magic = &minimal_program_bytes()[..4];

    let short = decode_error_string(&magic[..3], "one byte short of the magic");
    assert!(
        short.contains("TruncatedPayload") && !short.contains("MagicMismatch"),
        "one byte short of the magic is truncation, got: {short}"
    );

    let exact = decode_error_string(b"XXXX", "exactly the magic length, wrong content");
    assert!(
        exact.contains("MagicMismatch") && !exact.contains("TruncatedPayload"),
        "a full-length wrong magic is a mismatch, not truncation, got: {exact}"
    );
}

#[test]
fn wire_decoder_rejects_header_only() {
    let bytes = minimal_program_bytes();
    // Keep only the 40-byte header, drop all body bytes.
    let header_only = &bytes[..40.min(bytes.len())];

    let error = decode_error_string(header_only, "header-only input");
    assert!(
        error.contains("TruncatedPayload") || error.contains("IntegrityMismatch"),
        "a header whose body is absent must be reported as truncation or as an integrity \
         mismatch, got: {error}"
    );
}
