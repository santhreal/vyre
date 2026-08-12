//! Real-IR parity for production primitives that previously had CPU-oracle-only coverage.
//!
//! Every case executes the emitted `Program` through `vyre_reference::reference_eval`.
//! Iterative state is fed through bounded host redispatch where the primitive contract
//! requires it. The expected values come from the primitive CPU oracle or a direct,
//! independently computed scalar contract.

#![cfg(all(
    feature = "decode",
    feature = "reduce",
    feature = "graph",
    feature = "bitset",
    feature = "math",
    feature = "cpu-parity"
))]
#![forbid(unsafe_code)]

use vyre_foundation::ir::Program;
use vyre_reference::value::Value;

use vyre_primitives::bitset::stochastic_compute;
use vyre_primitives::decode::hex;
use vyre_primitives::graph::{path_reconstruct, scc_decompose};
use vyre_primitives::math::{dp_accountant, spectral_shape};
use vyre_primitives::reduce::radix_sort;

fn pack(words: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(words))
}

fn unpack(value: &Value) -> Vec<u32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte u32 chunk")))
        .collect()
}

fn output_words(program: &Program, outputs: &[Value], name: &str) -> Vec<u32> {
    let index = vyre_reference::output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: `{name}` must remain a declared reference output"));
    unpack(&outputs[index])
}

fn evaluate(program: &Program, inputs: Vec<Value>) -> Vec<Value> {
    vyre_reference::reference_eval(program, &inputs)
        .unwrap_or_else(|error| panic!("Fix: production IR reference evaluation failed: {error}"))
}

/// Hex decoding must execute the generated table lookup and nibble composition.
///
/// This catches swapped high and low nibbles, ASCII-index drift, and invalid-nibble
/// behavior that CPU-oracle-to-CPU-oracle comparisons cannot observe in emitted IR.
#[test]
fn hex_decode_real_ir_matches_oracle_for_mixed_case_and_invalid_nibbles() {
    let bytes = b"4D6aZ1";
    let input: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
    let program = hex::hex_decode("input", "output", "table", input.len() as u32);
    let outputs = evaluate(
        &program,
        vec![
            pack(&input),
            pack(&vec![0; input.len() / 2]),
            pack(hex::hex_decode_table_ref()),
        ],
    );

    assert_eq!(
        output_words(&program, &outputs, "output"),
        hex::hex_decode_reference_packed(bytes)
    );
}

/// Stable masked-key sorting must preserve original keys while ordering by the selected bits.
///
/// The cases cover the stable all-equal mask, duplicate low-bit keys, partial-byte keys,
/// and the full-width boundary. A shape-only test cannot detect rank or tie-break errors.
#[test]
fn radix_sort_real_ir_matches_stable_oracle_across_bit_widths() {
    let cases: &[(&[u32], u32)] = &[
        (&[9, 1, 7, 3], 0),
        (&[0x21, 0x11, 0x22, 0x12, 0x01], 4),
        (&[0x1ff, 0x001, 0x101, 0x0ff], 8),
        (&[u32::MAX, 0, 7, 7, 1 << 31], 32),
    ];

    for &(input, bits) in cases {
        let program = radix_sort::radix_sort("input", "output", input.len() as u32, bits);
        let outputs = evaluate(&program, vec![pack(input), pack(&vec![0; input.len()])]);
        assert_eq!(
            output_words(&program, &outputs, "output"),
            radix_sort::cpu_ref(input, bits),
            "bits={bits} input={input:?}"
        );
    }
}

/// SCC stamping must preserve prior assignments across bounded host redispatch.
///
/// This locks out unconditional overwrites and proves that sequential pivot passes carry
/// the read-write component vector exactly as the production dispatcher does.
#[test]
fn scc_decompose_real_ir_matches_two_pivot_host_redispatch() {
    let node_count = 6;
    let passes = [
        ([0b00_1111u32], [0b01_1101u32], 3u32),
        ([0b11_0000u32], [0b11_0100u32], 5u32),
    ];
    let mut expected = vec![u32::MAX; node_count as usize];
    let mut actual = expected.clone();

    for (forward, backward, pivot) in passes {
        expected = scc_decompose::cpu_ref(node_count, &forward, &backward, &expected, pivot);
        let program =
            scc_decompose::scc_decompose(node_count, "forward", "backward", "components", pivot);
        let outputs = evaluate(
            &program,
            vec![pack(&forward), pack(&backward), pack(&actual)],
        );
        actual = output_words(&program, &outputs, "components");
        assert_eq!(actual, expected, "pivot={pivot}");
    }
}

/// Path reconstruction must report the true length and deterministic zero-padded tail.
///
/// Root termination and a bounded cycle exercise both loop exits in emitted IR. This
/// catches the historical bug where the loop-control sentinel replaced the true length.
#[test]
fn path_reconstruct_real_ir_matches_root_and_cycle_oracles() {
    let cases: &[(&[u32], u32, u32)] = &[(&[0, 0, 1, 2], 3, 8), (&[1, 0], 0, 6)];

    for &(parent, target, max_depth) in cases {
        let mut expected_path = Vec::new();
        let expected_len = path_reconstruct::cpu_ref(parent, target, max_depth, &mut expected_path);
        let program =
            path_reconstruct::path_reconstruct("parent", "target", "path", "length", max_depth);
        let outputs = evaluate(
            &program,
            vec![
                pack(parent),
                pack(&[target]),
                pack(&vec![0; max_depth as usize]),
                pack(&[0]),
            ],
        );

        assert_eq!(output_words(&program, &outputs, "path"), expected_path);
        assert_eq!(
            output_words(&program, &outputs, "length"),
            vec![expected_len]
        );
    }
}

/// Stochastic multiplication must execute wordwise AND across every emitted lane.
///
/// Alternating, all-zero, all-one, and high-bit words catch wrong operators, truncated
/// grids, and stale output words that a bit-count-only probability check would miss.
#[test]
fn stochastic_and_mul_real_ir_matches_exact_word_oracle() {
    let lhs = [0xaaaa_5555, u32::MAX, 0, 1 << 31, 0x1357_9bdf];
    let rhs = [0x0f0f_f0f0, 0, u32::MAX, 1 << 31, 0x2468_ace0];
    let program = stochastic_compute::stochastic_and_mul("lhs", "rhs", "output", lhs.len() as u32);
    let outputs = evaluate(
        &program,
        vec![pack(&lhs), pack(&rhs), pack(&vec![0; lhs.len()])],
    );

    assert_eq!(
        output_words(&program, &outputs, "output"),
        stochastic_compute::cpu_ref(&lhs, &rhs)
    );
}

/// Spectral edge clipping must execute the shared vector-scalar map with exact boundaries.
///
/// Values below, at, and above the edge, including `u32::MAX`, catch swapped operands,
/// strict-vs-inclusive boundary mistakes, and unsigned truncation in emitted IR.
#[test]
fn spectral_shape_real_ir_matches_integer_projection_of_f64_oracle() {
    let values = [0, 1, 4, 5, 10, u32::MAX];
    let edge = 4u32;
    let program = spectral_shape::mp_edge_clip("values", "edge", "output", values.len() as u32);
    let outputs = evaluate(
        &program,
        vec![pack(&values), pack(&[edge]), pack(&vec![0; values.len()])],
    );
    let expected: Vec<u32> = spectral_shape::mp_edge_clip_cpu(
        &values
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
        f64::from(edge),
    )
    .into_iter()
    .map(|value| value as u32)
    .collect();

    assert_eq!(output_words(&program, &outputs, "output"), expected);
}

/// Gaussian RDP emission must match the documented integer division for every lane.
///
/// Uneven quotients prove truncation direction while distinct denominators catch swapped
/// inputs and missing multiplication by two. Zero divisors are tested separately below.
#[test]
fn dp_accountant_real_ir_matches_documented_lane_formula() {
    let alpha = [8, 13, 21, 100, u32::MAX - 1];
    let sigma_squared = [2, 3, 5, 11, 1];
    let expected: Vec<u32> = alpha
        .iter()
        .zip(sigma_squared)
        .map(|(&a, s2)| a / (2 * s2))
        .collect();
    let program =
        dp_accountant::gaussian_rdp_step("alpha", "sigma_squared", "output", alpha.len() as u32);
    let outputs = evaluate(
        &program,
        vec![
            pack(&alpha),
            pack(&sigma_squared),
            pack(&vec![0; alpha.len()]),
        ],
    );

    assert_eq!(output_words(&program, &outputs, "output"), expected);
}

/// Invalid dimensions and unsafe privacy divisors must fail instead of fabricating output.
///
/// These are the hostile boundaries where silent empty buffers, divide-by-zero saturation,
/// or denominator overflow would turn a proving parity suite into a false pass.
#[test]
fn production_ir_invalid_boundaries_fail_loudly() {
    for program in [
        hex::hex_decode("input", "output", "table", 3),
        radix_sort::radix_sort("input", "output", 0, 8),
        stochastic_compute::stochastic_and_mul("lhs", "rhs", "output", 0),
        spectral_shape::mp_edge_clip("values", "edge", "output", 0),
        dp_accountant::gaussian_rdp_step("alpha", "sigma", "output", 0),
    ] {
        assert!(program.stats().trap(), "invalid builder must emit a trap");
    }

    let program = dp_accountant::gaussian_rdp_step("alpha", "sigma", "output", 1);
    for sigma_squared in [0, 1 << 31] {
        let error = vyre_reference::reference_eval(
            &program,
            &[pack(&[8]), pack(&[sigma_squared]), pack(&[0])],
        )
        .expect_err("unsafe sigma-squared must reject a zero or overflowing denominator");
        assert!(
            error.to_string().contains("sigma_squared"),
            "unexpected invalid-sigma diagnostic: {error}"
        );
    }
}
