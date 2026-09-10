//! What a matrix fragment operand proves about the value behind it.
//!
//! The tile-matmul lowering decided on element type and extents alone and then
//! filled any operand slot it could not resolve with the scalar id bound to the
//! tile's name, so a `MatrixMma` operand list proved nothing about whether a
//! fragment had been formed. These cases pin the replacement: a fragment is
//! formed only from tiles whose declaration states a subgroup-distributed
//! fragment, the operand arity is derived from that declaration, and a
//! declaration that states no fragment lowers through the scalar expansion with
//! no matrix op at all.
//!
//! What they do not catch: whether the words a fragment carries are the words a
//! given target's matrix instruction expects in a given lane. That is the
//! emitter's own contract and needs a device oracle.

use std::collections::HashMap;

use vyre_foundation::ir::{DataType, Layout, Residency};
use vyre_lower::{lower, KernelOpKind};
use vyre_test_support::tile_programs::{tile_matmul_program, TileOperand};

/// `c[16,8] = a[16,16] x b[16,8]` with the stated residency on both inputs.
fn matmul_with_input_residency(residency: Residency) -> vyre_foundation::ir::Program {
    tile_matmul_program(
        [32, 1, 1],
        &TileOperand::new(
            DataType::F16,
            vec![16, 16],
            Layout::RowMajor,
            residency,
            256,
        ),
        &TileOperand::new(DataType::F16, vec![16, 8], Layout::ColumnMajor, residency, 128),
        &TileOperand::new(
            DataType::F32,
            vec![16, 8],
            Layout::RowMajor,
            Residency::Register,
            128,
        ),
    )
}

#[test]
fn only_a_subgroup_distributed_declaration_forms_a_matrix_fragment() {
    // Every residency a tile can declare. The match below has no catch-all, so
    // an added residency fails to compile until a decision is recorded for it.
    let residencies = [
        Residency::Register,
        Residency::Subgroup,
        Residency::Workgroup,
        Residency::Global,
    ];

    for residency in residencies {
        let forms_a_fragment = match residency {
            Residency::Subgroup => true,
            Residency::Register | Residency::Workgroup | Residency::Global => false,
        };

        let program = matmul_with_input_residency(residency);
        let descriptor = lower(&program)
            .unwrap_or_else(|error| panic!("Fix: {residency:?} inputs must lower: {error}"));
        let matrix_ops: Vec<_> = descriptor
            .ops_iter()
            .filter(|op| matches!(op.kind, KernelOpKind::MatrixMma(_)))
            .collect();

        if !forms_a_fragment {
            assert!(
                matrix_ops.is_empty(),
                "Fix: {residency:?} tiles state no subgroup-distributed fragment, so the product must lower without a matrix op"
            );
            // The product is still computed: the scalar expansion is the
            // untiled candidate and stays reachable.
            let multiplies = descriptor
                .ops_iter()
                .filter(|op| {
                    matches!(
                        op.kind,
                        KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Mul)
                    )
                })
                .count();
            assert!(
                multiplies > 0,
                "Fix: {residency:?} tiles must still lower to a computed product"
            );
            continue;
        }

        assert_eq!(
            matrix_ops.len(),
            1,
            "Fix: {residency:?} tiles must form exactly one matrix fragment product"
        );
        let matrix_op = matrix_ops[0];
        let KernelOpKind::MatrixMma(spec) = &matrix_op.kind else {
            unreachable!("filtered on MatrixMma")
        };

        // Arity comes from the declaration, not from a constant.
        let words = spec
            .operand_words()
            .expect("a formed fragment states its operand words");
        assert_eq!(
            matrix_op.operands.len(),
            words.iter().sum::<u32>() as usize,
            "Fix: the operand list must carry exactly the words the declaration derives"
        );

        // No operand word may be a repeat of another within the same operand: a
        // repeated id is what a substitute scalar id produced, and it makes a
        // fragment operand indistinguishable from an unformed one.
        let mut base = 0usize;
        for (slot, count) in words.iter().enumerate() {
            let operand = &matrix_op.operands[base..base + *count as usize];
            let distinct: std::collections::BTreeSet<u32> = operand.iter().copied().collect();
            assert_eq!(
                distinct.len(),
                operand.len(),
                "Fix: operand {slot} repeats a word id, so the fragment is not distinguishable from a substitute scalar"
            );
            base += *count as usize;
        }

        // Every word must be the result of an op in the descriptor, so no word
        // is an id the lowering invented or borrowed from a scalar binding that
        // never produced a value.
        let producers: HashMap<u32, &vyre_lower::KernelOp> = descriptor
            .ops_iter()
            .filter_map(|op| op.result.map(|result| (result, op)))
            .collect();
        for word in &matrix_op.operands {
            assert!(
                producers.contains_key(word)
                    || producers
                        .values()
                        .any(|op| op.result_ids().any(|id| id == *word)),
                "Fix: fragment operand word {word} has no producing op in the descriptor"
            );
        }
    }
}

#[test]
fn a_swizzled_declaration_states_no_fragment_orientation() {
    // A swizzled layout is a storage permutation no fragment orientation
    // expresses, so it is answered before an operand word exists rather than
    // reinterpreted as row-major.
    let program = tile_matmul_program(
        [32, 1, 1],
        &TileOperand::new(
            DataType::F16,
            vec![16, 16],
            Layout::Swizzled {
                permutation: vec![1, 0],
                period: 8,
            },
            Residency::Subgroup,
            256,
        ),
        &TileOperand::new(
            DataType::F16,
            vec![16, 8],
            Layout::ColumnMajor,
            Residency::Subgroup,
            128,
        ),
        &TileOperand::new(
            DataType::F32,
            vec![16, 8],
            Layout::RowMajor,
            Residency::Register,
            128,
        ),
    );

    let descriptor = lower(&program).expect("a swizzled tile must still lower");
    assert!(
        descriptor
            .ops_iter()
            .all(|op| !matches!(op.kind, KernelOpKind::MatrixMma(_))),
        "Fix: a swizzled tile states no fragment orientation and must not form a matrix op"
    );
}
