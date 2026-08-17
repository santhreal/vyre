//! Independent-source contract matrix for `hash::crc32` and packed FNV-1a32.
//!
//! Nothing here re-implements the production algorithm. CRC-32 is pinned three
//! ways that do not share a line of code with `hash::crc32`: the published
//! CRC-32/ISO-HDLC check vectors, a bit-at-a-time polynomial division that
//! never builds a lookup table, and the concatenation law, which fixes the
//! chunk algebra against the value the whole-input walker produces for the
//! joined bytes. FNV-1a32 is pinned to its published vectors and to a modular
//! reference that multiplies in `u64` instead of wrapping in `u32`.
//!
//! Volume testing.volume - do NOT weaken to shape-only asserts.

#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

use std::num::NonZeroU32;

use vyre_libs::hash::crc32::{CRC32_INIT, CRC32_POLY};
use vyre_libs::hash::fnv1a::{FNV1A32_OFFSET, FNV1A32_PRIME};
use vyre_reference::composition_witness::{
    crc32_combine_chunks_witness as crc32_combine_chunks, crc32_combine_witness as crc32_combine,
    crc32_pair_reduce_chunks_witness as crc32_pair_reduce_chunks, crc32_witness as crc32,
    fnv1a32_witness as fnv1a32, Crc32ChunkWitness as Crc32Chunk,
};

fn crc32_chunk(bytes: &[u8]) -> Crc32Chunk {
    Crc32Chunk {
        len: bytes.len() as u64,
        crc: crc32(bytes),
    }
}

fn fnv1a32_packed_u32_low8(words: &[u32]) -> u32 {
    let bytes: Vec<u8> = words.iter().map(|&w| (w & 0xFF) as u8).collect();
    fnv1a32(&bytes)
}

/// Reflected IEEE 802.3 polynomial, from the CRC-32/ISO-HDLC specification.
const SPEC_CRC32_POLY: u32 = 0xEDB8_8320;

/// Specified CRC-32/ISO-HDLC register preset and final xor.
const SPEC_CRC32_INIT: u32 = 0xFFFF_FFFF;

/// FNV-1a 32-bit offset basis from the FNV specification.
const SPEC_FNV1A32_OFFSET: u32 = 0x811C_9DC5;

/// FNV-1a 32-bit prime from the FNV specification.
const SPEC_FNV1A32_PRIME: u32 = 0x0100_0193;

/// Published CRC-32/ISO-HDLC check values.
///
/// `"123456789"` is the catalogue check value for this CRC; the remaining
/// strings are the values every zlib-compatible CRC-32 must reproduce.
const CRC32_CHECK_VECTORS: &[(&[u8], u32)] = &[
    (b"", 0x0000_0000),
    (b"a", 0xE8B7_BE43),
    (b"abc", 0x3524_41C2),
    (b"123456789", 0xCBF4_3926),
    (b"message digest", 0x2015_9D7F),
    (b"abcdefghijklmnopqrstuvwxyz", 0x4C27_50BD),
    (
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        0x1FC2_E6D2,
    ),
    (
        b"12345678901234567890123456789012345678901234567890123456789012345678901234567890",
        0x7CA9_4A72,
    ),
    (b"The quick brown fox jumps over the lazy dog", 0x414F_A339),
];

/// Published FNV-1a 32-bit check values.
const FNV1A32_CHECK_VECTORS: &[(&[u8], u32)] = &[
    (b"", 0x811C_9DC5),
    (b"a", 0xE40C_292C),
    (b"b", 0xE70C_2DE5),
    (b"c", 0xE60C_2C52),
    (b"foobar", 0xBF9C_F968),
    (b"chongo was here!\n", 0xD499_30D5),
];

#[test]
fn crc32_public_constants_match_the_specification() {
    assert_eq!(
        CRC32_POLY, SPEC_CRC32_POLY,
        "Fix: CRC32_POLY must be the reflected IEEE 802.3 polynomial."
    );
    assert_eq!(
        CRC32_INIT, SPEC_CRC32_INIT,
        "Fix: CRC32_INIT must be the specified CRC-32/ISO-HDLC preset."
    );
    assert_eq!(
        FNV1A32_OFFSET, SPEC_FNV1A32_OFFSET,
        "Fix: FNV1A32_OFFSET must be the specified FNV-1a 32-bit offset basis."
    );
    assert_eq!(
        FNV1A32_PRIME, SPEC_FNV1A32_PRIME,
        "Fix: FNV1A32_PRIME must be the specified FNV-1a 32-bit prime."
    );
}

#[test]
fn crc32_matches_the_published_check_vectors() {
    for (case_idx, (bytes, published)) in CRC32_CHECK_VECTORS.iter().enumerate() {
        assert_eq!(
            crc32(bytes),
            *published,
            "Fix: crc32 published vector {case_idx} len={} must match the CRC-32/ISO-HDLC check value.",
            bytes.len()
        );
        assert_eq!(
            bitwise_crc32(bytes),
            *published,
            "Fix: the bit-at-a-time reference must reproduce published vector {case_idx}; it is the independent source of truth for the generated matrix."
        );
    }
}

#[test]
fn crc32_matches_bitwise_polynomial_division_matrix() {
    for (case_idx, bytes) in byte_cases().iter().enumerate() {
        assert_eq!(
            crc32(bytes),
            bitwise_crc32(bytes),
            "Fix: crc32 adversarial case {case_idx} len={} must match bit-at-a-time polynomial division.",
            bytes.len()
        );
    }
}

#[test]
fn crc32_chunk_algebra_matches_the_concatenation_law_matrix() {
    for (case_idx, bytes) in byte_cases().iter().enumerate() {
        let chunk_size = NonZeroU32::new(1 + (case_idx % 64) as u32).expect("chunk size");
        let expected = bitwise_crc32(bytes);

        let mut reduced: Vec<Crc32Chunk> = if bytes.is_empty() {
            vec![crc32_chunk(bytes)]
        } else {
            bytes
                .chunks(chunk_size.get() as usize)
                .map(crc32_chunk)
                .collect()
        };
        while reduced.len() > 1 {
            reduced = crc32_pair_reduce_chunks(&reduced)
                .expect("Fix: generated CRC chunk lengths must not overflow.");
        }
        let folded = reduced[0];
        assert_eq!(
            folded.crc, expected,
            "Fix: crc32 map-reduce adversarial case {case_idx} must equal the CRC of the joined bytes."
        );
        assert_eq!(
            folded.len,
            bytes.len() as u64,
            "Fix: crc32 map-reduce adversarial case {case_idx} must preserve byte length."
        );

        for split in 0..=bytes.len().min(8) {
            let left = crc32_chunk(&bytes[..split]);
            let right = crc32_chunk(&bytes[split..]);
            assert_eq!(
                crc32_combine(left.crc, right.crc, right.len),
                expected,
                "Fix: crc32_combine adversarial case {case_idx} split={split} must equal the CRC of the joined bytes."
            );
            assert_eq!(
                crc32_combine_chunks(left, right)
                    .expect("Fix: generated CRC chunk length must not overflow.")
                    .crc,
                expected,
                "Fix: crc32_combine_chunks adversarial case {case_idx} split={split} must equal the CRC of the joined bytes."
            );
        }

        assert_eq!(
            crc32_chunk(bytes),
            Crc32Chunk {
                len: bytes.len() as u64,
                crc: expected,
            },
            "Fix: crc32_chunk adversarial case {case_idx} must carry the joined-bytes CRC and length."
        );
    }
}

#[test]
fn fnv1a32_matches_the_published_check_vectors() {
    for (case_idx, (bytes, published)) in FNV1A32_CHECK_VECTORS.iter().enumerate() {
        assert_eq!(
            fnv1a32(bytes),
            *published,
            "Fix: fnv1a32 published vector {case_idx} must match the FNV-1a 32-bit check value."
        );
        assert_eq!(
            modular_fnv1a32(bytes),
            *published,
            "Fix: the modular reference must reproduce published vector {case_idx}; it is the independent source of truth for the generated matrix."
        );
    }
}

#[test]
fn fnv1a32_packed_low8_matches_modular_reference_matrix() {
    for (case_idx, words) in packed_u32_cases().iter().enumerate() {
        let bytes: Vec<u8> = words.iter().map(|word| (*word & 0xFF) as u8).collect();
        let expected = modular_fnv1a32(&bytes);
        assert_eq!(
            fnv1a32_packed_u32_low8(words),
            expected,
            "Fix: fnv1a32_packed_u32_low8 adversarial case {case_idx} len={} must match the modular reference over the low bytes.",
            words.len()
        );
        assert_eq!(
            fnv1a32(&bytes),
            expected,
            "Fix: fnv1a32 byte path adversarial case {case_idx} must match the modular reference."
        );
    }
}

/// CRC-32/ISO-HDLC by bit-at-a-time polynomial division.
///
/// This is the textbook shift-register form: no lookup table, no byte-indexed
/// slice, no GF(2) matrix. It shares no construction with the production
/// walker, which is the only reason it can prove that walker right, and it is
/// itself proved by the published check vectors above.
fn bitwise_crc32(bytes: &[u8]) -> u32 {
    let mut register = SPEC_CRC32_INIT;
    for &byte in bytes {
        register ^= u32::from(byte);
        for _ in 0..8 {
            let feedback = register & 1;
            register >>= 1;
            if feedback != 0 {
                register ^= SPEC_CRC32_POLY;
            }
        }
    }
    register ^ SPEC_CRC32_INIT
}

/// FNV-1a 32-bit with the product taken in `u64` and reduced modulo 2^32.
///
/// The production walker relies on `u32` wrapping multiplication. Doing the
/// arithmetic in a wider type and reducing explicitly is a different
/// construction of the same specified function, so agreement is evidence rather
/// than tautology.
fn modular_fnv1a32(bytes: &[u8]) -> u32 {
    const MODULUS: u64 = 1 << 32;
    let mut hash = u64::from(SPEC_FNV1A32_OFFSET);
    for &byte in bytes {
        hash = ((hash ^ u64::from(byte)) * u64::from(SPEC_FNV1A32_PRIME)) % MODULUS;
    }
    hash as u32
}

fn byte_cases() -> Vec<Vec<u8>> {
    let mut cases = Vec::new();
    let lengths = [
        0usize, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024,
    ];
    let fills = [0u8, 1, 0xFF, 0x7F, 0x80, 0x61, 0x00];

    for len in lengths {
        for fill in fills {
            cases.push(vec![fill; len]);
        }
        cases.push((0..len).map(|idx| idx as u8).collect());
        cases.push(
            (0..len)
                .map(|idx| idx.wrapping_mul(41).wrapping_add(3) as u8)
                .collect(),
        );
    }

    for seed in [0x01, 0x11, 0xBE, 0xEF, 0x80, 0xFE] {
        for len in lengths {
            cases.push(lcg_bytes(seed, len));
        }
    }

    for case in 0..16384usize {
        let len = case % 513;
        cases.push(lcg_bytes(case as u8 ^ 0xA5, len));
    }

    cases
}

fn packed_u32_cases() -> Vec<Vec<u32>> {
    let mut cases = Vec::new();
    let lengths = [
        0usize, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024,
    ];
    let fills = [0u32, 1, 0xFF, 0x61, 0xFFFF_FF61, 0xCAFE_00BE, u32::MAX];

    for len in lengths {
        for fill in fills {
            cases.push(vec![fill; len]);
        }
        cases.push((0..len).map(|idx| idx as u32).collect());
        cases.push((0..len).map(|idx| (idx as u32) << 24).collect());
    }

    for seed in [0x01, 0x11, 0xBE, 0xEF, 0x80, 0xFE] {
        for len in lengths {
            cases.push(lcg_words(seed, len));
        }
    }

    for case in 0..16384usize {
        let len = case % 513;
        cases.push(lcg_words(case as u32 ^ 0x5A5A_5A5A, len));
    }

    cases
}

fn lcg_bytes(seed: u8, len: usize) -> Vec<u8> {
    let mut state = u32::from(seed);
    (0..len)
        .map(|idx| {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .rotate_left((idx % 13) as u32);
            (state ^ (idx as u32).wrapping_mul(0x85EB_CA6B)) as u8
        })
        .collect()
}

fn lcg_words(seed: u32, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .rotate_left((idx % 31) as u32);
            state ^ (idx as u32).wrapping_mul(0x85EB_CA6B)
        })
        .collect()
}
