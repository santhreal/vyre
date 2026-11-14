//! Adversarial coverage for the byte-histogram text primitive.
//!
//! Focus: hostile boundaries, overflow, invalid offsets, property invariants.
#![cfg(feature = "text")]
#![allow(clippy::needless_range_loop)]

use vyre_foundation::ir::DataType;
use vyre_libs::text::byte_histogram_256_u8;
use vyre_reference::composition_witness::byte_histogram_witness as reference_byte_histogram;

#[test]
fn reference_byte_histogram_empty() {
    let got = reference_byte_histogram(b"");
    assert!(got.iter().all(|&c| c == 0));
}

#[test]
fn reference_byte_histogram_all_same_byte() {
    for byte in [0x00, 0x7F, 0x80, 0xFF] {
        let input = vec![byte; 1024];
        let got = reference_byte_histogram(&input);
        assert_eq!(got[byte as usize], 1024, "byte 0x{byte:02X} count mismatch");
        assert!(
            got.iter()
                .enumerate()
                .all(|(i, &c)| i == byte as usize || c == 0),
            "only byte 0x{byte:02X} should have non-zero count"
        );
    }
}

#[test]
fn reference_byte_histogram_hostile_lengths() {
    for len in [0, 1, 31, 32, 33, 255, 256, 1023, 1024] {
        let input = vec![0xABu8; len];
        let got = reference_byte_histogram(&input);
        assert_eq!(got[0xAB], len as u32, "count mismatch for length {len}");
    }
}

#[test]
fn reference_byte_histogram_every_byte_once() {
    let input: Vec<u8> = (0..=255).collect();
    let got = reference_byte_histogram(&input);
    assert!(got.iter().all(|&c| c == 1));
}

#[test]
fn packed_u8_byte_histogram_uses_one_source_byte_per_element() {
    let program = byte_histogram_256_u8("source", "histogram", 1024);
    let source = program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == "source")
        .expect("Fix: packed-u8 byte histogram source buffer must be declared");
    let histogram = program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == "histogram")
        .expect("Fix: byte histogram output buffer must be declared");

    assert_eq!(source.element(), DataType::U8);
    assert_eq!(source.count(), 1024);
    assert_eq!(
        source.count() as usize * DataType::U8.min_bytes(),
        1024,
        "Fix: packed-u8 byte histogram must consume one byte per source byte."
    );
    assert_eq!(
        source.count() as usize * DataType::U32.min_bytes(),
        4096,
        "Fix: compatibility byte histogram remains the four-byte-per-source-byte path."
    );
    assert_eq!(histogram.element(), DataType::U32);
    assert_eq!(histogram.count(), 256);
}
