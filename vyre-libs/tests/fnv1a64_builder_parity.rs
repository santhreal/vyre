//! Parity against the CPU reference for every published FNV-1a64 program builder.
//!
//! WHY this exists: the builders differ only in loop bound (a static `n` or the
//! dynamic `buf_len`) and in source element type (one byte per `DataType::U32`
//! lane, or packed `DataType::U8`), and each pair shares one body. A test that
//! named two of them let the packed dynamic variant ship with no parity check at
//! all, because the family looks covered from the outside.
//!
//! The row set is compared against the `pub fn fnv1a64_program*` declarations
//! read out of the source at run time, so a fifth builder fails this test
//! instead of arriving unproven.
#![cfg(feature = "hash")]

use std::collections::BTreeSet;

use vyre_foundation::ir::Program;
use vyre_libs::hash::fnv1a::{
    fnv1a64, fnv1a64_program, fnv1a64_program_n, fnv1a64_program_n_u8, fnv1a64_program_u8,
};
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;
use vyre_test_support::monorepo::vyre_crate_directory;

/// The file that declares the builder family, read to derive the member set.
const FAMILY_SOURCE: &str = "src/hash/fnv1a.rs";

/// Source element layout a builder declares for its input buffer.
#[derive(PartialEq, Eq)]
enum Source {
    /// One `DataType::U32` element per source byte, low byte significant.
    U32Lanes,
    /// One `DataType::U8` element per source byte.
    PackedBytes,
}

impl Source {
    /// Encode message bytes in this layout.
    fn encode(&self, bytes: &[u8]) -> Vec<u8> {
        match self {
            Source::U32Lanes => pack_u32(&bytes.iter().map(|&b| u32::from(b)).collect::<Vec<_>>()),
            Source::PackedBytes => bytes.to_vec(),
        }
    }
}

/// One published builder plus the input layout it consumes.
struct Row {
    /// Builder name, matched against the source declarations.
    name: &'static str,
    /// Builds the program for a message of `n` source elements.
    build: fn(&str, &str, u32) -> Program,
    /// Layout this builder's input buffer declares.
    source: Source,
}

fn rows() -> [Row; 4] {
    [
        Row {
            name: "fnv1a64_program",
            build: |input, out, _n| fnv1a64_program(input, out),
            source: Source::U32Lanes,
        },
        Row {
            name: "fnv1a64_program_u8",
            build: |input, out, _n| fnv1a64_program_u8(input, out),
            source: Source::PackedBytes,
        },
        Row {
            name: "fnv1a64_program_n",
            build: fnv1a64_program_n,
            source: Source::U32Lanes,
        },
        Row {
            name: "fnv1a64_program_n_u8",
            build: fnv1a64_program_n_u8,
            source: Source::PackedBytes,
        },
    ]
}

/// Run one builder over `message` and recombine its two output words.
fn hash_of(row: &Row, message: &[u8]) -> u64 {
    let n = u32::try_from(message.len()).expect("message length fits u32");
    let program = (row.build)("input", "out", n);
    let inputs = [
        Value::from(row.source.encode(message)),
        Value::from(vec![0u8; 8]),
    ];
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .unwrap_or_else(|error| panic!("{} must evaluate: {error}", row.name));
    let bytes = outputs[0].to_bytes();
    assert_eq!(
        bytes.len(),
        8,
        "{} must write exactly two u32 words",
        row.name
    );
    let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let hi = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    (u64::from(hi) << 32) | u64::from(lo)
}

/// Deterministic pseudo-random message of `len` bytes.
fn scrambled(len: usize) -> Vec<u8> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        })
        .collect()
}

#[test]
fn every_published_builder_matches_the_cpu_reference() {
    // 512 bytes exercise carry propagation between the two halves of the
    // synthesized 64-bit prime multiply on every update.
    let long = scrambled(512);
    let messages: [&[u8]; 4] = [b"abc", b"foobar", b"a", long.as_slice()];
    for row in rows() {
        for message in messages {
            assert_eq!(
                hash_of(&row, message),
                fnv1a64(message),
                "{} must match the CPU reference over {} byte(s)",
                row.name,
                message.len()
            );
        }
    }
}

#[test]
fn u32_lane_builders_ignore_the_high_lane_bits() {
    // Low bytes spell "abc"; the high 24 bits of each lane must not participate.
    let words = [0xFFFF_FF61u32, 0xCAFE_0062, 0x8000_0063];
    let lanes = pack_u32(&words);
    for row in rows()
        .into_iter()
        .filter(|row| row.source == Source::U32Lanes)
    {
        let program = (row.build)("input", "out", 3);
        let inputs = [Value::from(lanes.clone()), Value::from(vec![0u8; 8])];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .unwrap_or_else(|error| panic!("{} must evaluate: {error}", row.name));
        let bytes = outputs[0].to_bytes();
        let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let hi = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(
            (u64::from(hi) << 32) | u64::from(lo),
            fnv1a64(b"abc"),
            "{} must mask each u32 lane to its low byte",
            row.name
        );
    }
}

#[test]
fn canonical_fnv1a64_vectors_are_pinned() {
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_F739_67E8);
    assert_eq!(fnv1a64(b"abc"), 0xE71F_A219_0541_574B);
}

#[test]
fn the_rows_are_every_builder_the_source_publishes() {
    let source =
        std::fs::read_to_string(vyre_crate_directory("vyre-primitives").join(FAMILY_SOURCE))
            .expect("the FNV-1a source file must be readable");
    let declared: BTreeSet<String> = source
        .lines()
        .filter_map(|line| line.trim().strip_prefix("pub fn fnv1a64_program"))
        .filter_map(|rest| rest.split('(').next())
        .map(|suffix| format!("fnv1a64_program{suffix}"))
        .collect();
    let covered: BTreeSet<String> = rows().iter().map(|row| row.name.to_string()).collect();

    assert!(
        !declared.is_empty(),
        "the declaration scan found no `pub fn fnv1a64_program*` in {FAMILY_SOURCE}, so it proves nothing"
    );
    assert_eq!(
        declared, covered,
        "every published FNV-1a64 builder needs a parity row. Fix: add the row, do not edit this assertion."
    );
}
