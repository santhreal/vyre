use super::*;
use vyre_reference::composition_witness::utf8_validate_witness as reference_utf8_validate;
#[test]
fn program_uses_block_sized_workgroup() {
    let program = utf8_validate("source", "classes", 513);
    assert_eq!(program.workgroup_size(), UTF8_VALIDATE_WORKGROUP_SIZE);
}

#[test]
fn dispatch_grid_packs_byte_lanes_into_blocks() {
    assert_eq!(utf8_validate_dispatch_grid(0), [1, 1, 1]);
    assert_eq!(utf8_validate_dispatch_grid(1), [1, 1, 1]);
    assert_eq!(utf8_validate_dispatch_grid(256), [1, 1, 1]);
    assert_eq!(utf8_validate_dispatch_grid(257), [2, 1, 1]);
    assert_eq!(utf8_validate_dispatch_grid(513), [3, 1, 1]);
}

#[test]
fn reference_ascii() {
    assert_eq!(reference_utf8_validate(b"Hello"), vec![UTF8_ASCII; 5]);
}

#[test]
fn reference_2_byte_sequence() {
    // U+00E9 (é) = 0xC3 0xA9  -  LEAD_2 + CONT
    assert_eq!(
        reference_utf8_validate(&[0xC3, 0xA9]),
        vec![UTF8_LEAD_2, UTF8_CONT]
    );
}

#[test]
fn reference_3_byte_sequence() {
    // U+20AC (€) = 0xE2 0x82 0xAC  -  LEAD_3 + CONT + CONT
    assert_eq!(
        reference_utf8_validate(&[0xE2, 0x82, 0xAC]),
        vec![UTF8_LEAD_3, UTF8_CONT, UTF8_CONT]
    );
}

#[test]
fn reference_4_byte_sequence() {
    // U+1F600 (😀) = 0xF0 0x9F 0x98 0x80  -  LEAD_4 + CONT × 3
    assert_eq!(
        reference_utf8_validate(&[0xF0, 0x9F, 0x98, 0x80]),
        vec![UTF8_LEAD_4, UTF8_CONT, UTF8_CONT, UTF8_CONT]
    );
}

#[test]
fn reference_overlong_lead_invalid() {
    // 0xC0/0xC1 are forbidden lead bytes (overlong 2-byte
    // encodings of ASCII).
    assert_eq!(
        reference_utf8_validate(&[0xC0, 0xC1]),
        vec![UTF8_INVALID, UTF8_INVALID]
    );
}

#[test]
fn reference_out_of_range_lead_invalid() {
    // 0xF8..0xFF would imply 5+ byte sequences  -  banned since RFC 3629.
    assert_eq!(
        reference_utf8_validate(&[0xF8, 0xFC, 0xFF]),
        vec![UTF8_INVALID, UTF8_INVALID, UTF8_INVALID]
    );
}

#[test]
fn reference_rejects_stray_continuation() {
    assert_eq!(reference_utf8_validate(&[0x80]), vec![UTF8_INVALID]);
    assert_eq!(
        reference_utf8_validate(&[b'a', 0xBF]),
        vec![UTF8_ASCII, UTF8_INVALID]
    );
}

#[test]
fn reference_rejects_truncated_sequences() {
    assert_eq!(reference_utf8_validate(&[0xC3]), vec![UTF8_INVALID]);
    assert_eq!(
        reference_utf8_validate(&[0xE2, 0x82]),
        vec![UTF8_INVALID, UTF8_INVALID]
    );
    assert_eq!(
        reference_utf8_validate(&[0xF0, 0x9F, 0x98]),
        vec![UTF8_INVALID, UTF8_INVALID, UTF8_INVALID]
    );
}

#[test]
fn reference_rejects_surrogate_and_overlong_sequences() {
    assert_eq!(
        reference_utf8_validate(&[0xED, 0xA0, 0x80]),
        vec![UTF8_INVALID, UTF8_INVALID, UTF8_INVALID]
    );
    assert_eq!(
        reference_utf8_validate(&[0xE0, 0x80, 0x80]),
        vec![UTF8_INVALID; 3]
    );
    assert_eq!(
        reference_utf8_validate(&[0xF0, 0x80, 0x80, 0x80]),
        vec![UTF8_INVALID; 4]
    );
}
