//! Generated CPU-reference matrix for public u32 hardware intrinsic builders.
//!
//! These intrinsics are Cat-C because backends lower them to dedicated hardware
//! instructions or barriers. The CPU reference path is still the conformance
//! oracle, so the public builders must stay byte-exact over edge-heavy and
//! generated lanes, including dispatch extents larger than one workgroup.

mod gate_fixtures;

use gate_fixtures::{generated_u32_with_edges, run_eval_single};
use vyre_foundation::ir::Program;
struct U32Case {
    name: &'static str,
    build: fn(&str, &str, u32) -> Program,
    expected: fn(u32) -> u32,
}

const CASES: &[U32Case] = &[
    U32Case {
        name: "bit_reverse_u32",
        build: vyre_primitives::hardware::bit_reverse_u32::bit_reverse_u32,
        expected: u32::reverse_bits,
    },
    U32Case {
        name: "popcount_u32",
        build: vyre_primitives::hardware::popcount_u32::popcount_u32,
        expected: u32::count_ones,
    },
    U32Case {
        name: "storage_barrier",
        build: vyre_primitives::hardware::storage_barrier::storage_barrier,
        expected: |value| value,
    },
    U32Case {
        name: "workgroup_barrier",
        build: vyre_primitives::hardware::workgroup_barrier::workgroup_barrier,
        expected: |value| value,
    },
];

const INPUT_EDGES: [u32; 12] = [
    0,
    1,
    2,
    3,
    31,
    32,
    63,
    64,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffe,
    u32::MAX,
];

fn generated_input(len: usize, seed: u32) -> Vec<u32> {
    generated_u32_with_edges(len, seed, &INPUT_EDGES)
}

fn run(program: &Program, input: &[u32]) -> Vec<u8> {
    let input_bytes = vyre_primitives::wire::pack_u32_slice(input);
    let output_bytes = vec![0u8; input.len().max(1) * 4];
    run_eval_single(program, vec![input_bytes, output_bytes])
}

#[test]
fn generated_u32_hardware_intrinsics_match_host_semantics() {
    let lengths = [
        1usize, 2, 3, 4, 31, 32, 63, 64, 65, 127, 128, 257, 1024, 4096,
    ];
    let mut checked_lanes = 0usize;

    for case in CASES {
        for &len in &lengths {
            let input = generated_input(len, case.name.len() as u32 ^ len as u32);
            let program = (case.build)("input", "out", len as u32);
            let got = run(&program, &input);
            let expected_words: Vec<u32> = input.iter().copied().map(case.expected).collect();
            let expected = vyre_primitives::wire::pack_u32_slice(&expected_words);
            assert_eq!(got, expected, "{} failed for len {len}", case.name);
            checked_lanes += len;
        }
    }

    assert_eq!(checked_lanes, CASES.len() * lengths.iter().sum::<usize>());
}
