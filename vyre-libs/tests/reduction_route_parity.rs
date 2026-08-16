//! Exact parity and contention-shape contracts for scalar sum reduction routes.

#![cfg(all(feature = "reduce", feature = "cpu-parity"))]
#![forbid(unsafe_code)]

use vyre_reference::value::Value;

use vyre_libs::reduce::{sum, workgroup_tree};

fn pack(words: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(words))
}

fn evaluate_output(program: &vyre_foundation::ir::Program, inputs: &[Value]) -> Vec<u32> {
    let outputs = vyre_reference::reference_eval(program, inputs).unwrap_or_else(|error| {
        panic!("Fix: reduction route reference evaluation failed: {error}")
    });
    let output_index = vyre_reference::output_index(program, "out")
        .expect("Fix: each scalar reduction route must expose `out`");
    outputs[output_index]
        .to_bytes()
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte u32 output")))
        .collect()
}

fn generated_values(count: u32) -> Vec<u32> {
    (0..count)
        .map(|index| index.wrapping_mul(17).wrapping_add(3) & 0xff)
        .collect()
}

/// Atomic and workgroup-tree routes must produce the same exact wrapping sum.
///
/// Sizes straddle lane, tile, and multi-chunk boundaries. This catches missed tails,
/// duplicate chunks, and route-specific identity errors before timing can select a winner.
#[test]
fn atomic_and_tree_routes_match_exactly_across_crossover_boundaries() {
    for count in [1, 31, 32, 33, 255, 256, 257, 1024] {
        let values = generated_values(count);
        let expected = sum::cpu_ref(&values);
        let atomic = sum::reduce_sum("values", "out", count);
        let tile = count.min(256).next_power_of_two();
        let tree = workgroup_tree::workgroup_sum_u32("values", "out", count, tile);

        assert_eq!(
            evaluate_output(&atomic, &[pack(&values), pack(&[0])]),
            vec![expected],
            "atomic count={count}"
        );
        assert_eq!(
            evaluate_output(&tree, &[pack(&values)]),
            vec![expected],
            "tree count={count} tile={tile}"
        );
    }
}

/// Both routes must retain modulo-2^32 semantics under adversarial overflow.
///
/// A wider or saturating accumulator in only one route would make performance routing
/// input-dependent and could let the faster route return a different public result.
#[test]
fn atomic_and_tree_routes_preserve_wrapping_overflow() {
    let values = [u32::MAX, 1, u32::MAX, 2, 0x8000_0000, 0x8000_0000];
    let expected = sum::cpu_ref(&values);
    let atomic = sum::reduce_sum("values", "out", values.len() as u32);
    let tree = workgroup_tree::workgroup_sum_u32("values", "out", values.len() as u32, 8);

    assert_eq!(
        evaluate_output(&atomic, &[pack(&values), pack(&[0])]),
        vec![expected]
    );
    assert_eq!(evaluate_output(&tree, &[pack(&values)]), vec![expected]);
}

/// The two candidates must remain physically distinct contention strategies.
///
/// The benchmark is meaningless if refactoring makes both builders emit an atomic RMW
/// or removes the tree scratch buffer while preserving names and outputs.
#[test]
fn route_ir_exposes_atomic_and_workgroup_memory_contention_difference() {
    let atomic = sum::reduce_sum("values", "out", 1 << 20);
    let tree = workgroup_tree::workgroup_sum_u32("values", "out", 1 << 20, 256);

    assert_eq!(atomic.stats().atomic_op_count, 1);
    assert_eq!(tree.stats().atomic_op_count, 0);
    assert_eq!(
        atomic
            .buffers()
            .iter()
            .filter(|buffer| buffer.kind() == vyre_foundation::ir::MemoryKind::Shared)
            .count(),
        0
    );
    assert_eq!(
        tree.buffers()
            .iter()
            .filter(|buffer| buffer.kind() == vyre_foundation::ir::MemoryKind::Shared)
            .map(|buffer| buffer.count())
            .collect::<Vec<_>>(),
        vec![256]
    );
}
