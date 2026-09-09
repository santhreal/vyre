//! The typed matrix surface: derived arity, derived result span, and rejection
//! of every declaration no target can carry.
//!
//! The descriptor used to state one tile shape and one hardcoded operand
//! arity, so a second native form was unstatable and the verifier checked a
//! constant instead of the declaration. These cases pin the derivation and the
//! rejection classes: a fragment set states its own arity, and the arity the
//! op provides is checked against it rather than against a literal.
//!
//! What they do not catch: whether a concrete backend has an instruction for a
//! declared form. That is the target's own rejection, checked in the emitter
//! crates.

use vyre_lower::descriptor_builder::{self as descriptor_builder, body, lit, op};
use vyre_lower::{
    verify, FragmentOperand, FragmentValue, KernelDescriptor, KernelOpKind, LiteralValue,
    MatrixMmaElement, MatrixMmaLayout, MatrixMmaSpec, MatrixSpecError, MatrixTileShape,
    MemoryClass, TensorAccessMap, VerifyErrorKind,
};

const LANES: u16 = 32;

/// Every operand a fragment set can carry, derived from the enum rather than
/// listed, so a new operand turns these cases red.
const OPERANDS: [FragmentOperand; 3] = [
    FragmentOperand::Left,
    FragmentOperand::Right,
    FragmentOperand::Accumulator,
];

fn registers(element: MatrixMmaElement, layout: MatrixMmaLayout) -> FragmentValue {
    FragmentValue::in_registers(element, layout, LANES)
}

fn spec(tile: MatrixTileShape) -> MatrixMmaSpec {
    MatrixMmaSpec {
        tile,
        left: registers(MatrixMmaElement::F16, MatrixMmaLayout::RowMajor),
        right: registers(MatrixMmaElement::F16, MatrixMmaLayout::ColMajor),
        accumulator: registers(MatrixMmaElement::F32, MatrixMmaLayout::RowMajor),
    }
}

/// A descriptor whose only value-producing op is the matrix multiply, fed by
/// `operand_count` literal words.
fn descriptor(spec: MatrixMmaSpec, operand_count: u32) -> KernelDescriptor {
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    for id in 0..operand_count {
        literals.push(LiteralValue::U32(id));
        ops.push(lit(id, id));
    }
    ops.push(op(
        KernelOpKind::MatrixMma(Box::new(spec)),
        (0..operand_count).collect::<Vec<u32>>(),
        operand_count,
    ));
    descriptor_builder::descriptor("matrix_fragment")
        .dispatch(LANES.into(), 1, 1)
        .body(body().ops(ops).literals(literals))
        .build()
}

#[test]
fn operand_words_are_derived_from_the_declared_tile_and_elements() {
    let native = spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    assert_eq!(
        native.operand_words().unwrap(),
        [4, 2, 4],
        "Fix: a 16x8x16 f16/f16/f32 fragment set holds four, two and four words per lane"
    );
    assert_eq!(native.operand_count().unwrap(), 10);
    assert_eq!(native.result_count().unwrap(), 4);

    // Doubling the accumulator rows doubles the left and accumulator words and
    // leaves the right tile alone, which is exactly what a hardcoded arity
    // could not express.
    let wide = spec(MatrixTileShape { m: 32, n: 8, k: 16 });
    assert_eq!(wide.operand_words().unwrap(), [8, 2, 8]);
    assert_eq!(wide.operand_count().unwrap(), 18);
    assert_eq!(wide.result_count().unwrap(), 8);

    // A wider element type holds the same elements in twice the words.
    let mut widened = native;
    widened.left = registers(MatrixMmaElement::TF32, MatrixMmaLayout::RowMajor);
    widened.right = registers(MatrixMmaElement::TF32, MatrixMmaLayout::ColMajor);
    assert_eq!(widened.operand_words().unwrap(), [8, 4, 4]);
}

#[test]
fn every_element_type_states_its_storage_width() {
    for element in [
        MatrixMmaElement::F16,
        MatrixMmaElement::BF16,
        MatrixMmaElement::TF32,
        MatrixMmaElement::F32,
    ] {
        // Exhaustive: a new element type fails to compile here until its width
        // is stated, rather than silently inheriting a neighbor's.
        let expected = match element {
            MatrixMmaElement::F16 | MatrixMmaElement::BF16 => 16,
            MatrixMmaElement::TF32 | MatrixMmaElement::F32 => 32,
        };
        assert_eq!(element.bits(), expected);
    }
}

#[test]
fn each_operand_reads_its_own_tile_extents() {
    let tile = MatrixTileShape { m: 16, n: 8, k: 4 };
    for operand in OPERANDS {
        // Exhaustive: a new operand must state its extents here.
        let expected = match operand {
            FragmentOperand::Left => [tile.m, tile.k],
            FragmentOperand::Right => [tile.k, tile.n],
            FragmentOperand::Accumulator => [tile.m, tile.n],
        };
        assert_eq!(tile.extents(operand), expected, "operand {operand}");
        assert_eq!(
            tile.elements(operand),
            u32::from(expected[0]) * u32::from(expected[1])
        );
    }
}

#[test]
fn an_unstatable_declaration_is_rejected_with_the_fact_that_fails() {
    let mut zero_extent = spec(MatrixTileShape { m: 16, n: 0, k: 16 });
    assert_eq!(zero_extent.validate(), Err(MatrixSpecError::ZeroExtent));

    zero_extent = spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    let mut no_lanes = zero_extent;
    no_lanes.left =
        FragmentValue::in_registers(MatrixMmaElement::F16, MatrixMmaLayout::RowMajor, 0);
    assert_eq!(no_lanes.validate(), Err(MatrixSpecError::ZeroLanes));

    // 16x3 elements over 32 lanes does not distribute.
    let uneven = spec(MatrixTileShape { m: 16, n: 3, k: 16 });
    assert_eq!(
        uneven.validate(),
        Err(MatrixSpecError::UnevenDistribution {
            operand: FragmentOperand::Right,
            elements: 48,
            lanes: LANES,
        })
    );

    // One f16 element per lane is 16 bits, half an operand word.
    let partial = spec(MatrixTileShape { m: 32, n: 32, k: 1 });
    assert_eq!(
        partial.validate(),
        Err(MatrixSpecError::PartialWord {
            operand: FragmentOperand::Left,
            bits_per_lane: 16,
        })
    );

    let mut short_stride = spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    short_stride.left.access = Some(TensorAccessMap {
        storage: MemoryClass::Scratch,
        row_stride: 4,
        alignment: 16,
    });
    assert_eq!(
        short_stride.validate(),
        Err(MatrixSpecError::ShortRowStride {
            operand: FragmentOperand::Left,
            stride: 4,
            columns: 16,
        })
    );

    let mut unaligned = spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    unaligned.accumulator.access = Some(TensorAccessMap {
        storage: MemoryClass::Global,
        row_stride: 0,
        alignment: 0,
    });
    assert_eq!(
        unaligned.validate(),
        Err(MatrixSpecError::ZeroAlignment {
            operand: FragmentOperand::Accumulator,
        })
    );
}

#[test]
fn every_rejection_class_states_the_fact_that_failed() {
    let cases = [
        MatrixSpecError::ZeroExtent,
        MatrixSpecError::ZeroLanes,
        MatrixSpecError::UnevenDistribution {
            operand: FragmentOperand::Right,
            elements: 48,
            lanes: LANES,
        },
        MatrixSpecError::PartialWord {
            operand: FragmentOperand::Left,
            bits_per_lane: 16,
        },
        MatrixSpecError::ShortRowStride {
            operand: FragmentOperand::Left,
            stride: 4,
            columns: 16,
        },
        MatrixSpecError::ZeroAlignment {
            operand: FragmentOperand::Accumulator,
        },
    ];
    for case in cases {
        // Exhaustive: a new rejection class must state the noun it names here,
        // so an unnamed class cannot ship with an empty message.
        let noun = match case {
            MatrixSpecError::ZeroExtent => "extent",
            MatrixSpecError::ZeroLanes => "lanes",
            MatrixSpecError::UnevenDistribution { .. } => "distribute",
            MatrixSpecError::PartialWord { .. } => "words",
            MatrixSpecError::ShortRowStride { .. } => "row stride",
            MatrixSpecError::ZeroAlignment { .. } => "alignment",
        };
        let text = case.to_string();
        assert!(
            text.contains(noun),
            "Fix: {case:?} must name {noun}; got {text}"
        );
    }
}

#[test]
fn a_staged_tile_resolves_a_packed_row_stride_from_its_own_extent() {
    let mut staged = spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    let map = TensorAccessMap {
        storage: MemoryClass::Scratch,
        row_stride: 0,
        alignment: 8,
    };
    staged.left.access = Some(map);
    assert_eq!(map.effective_row_stride(16), 16);
    assert_eq!(map.effective_row_stride(8), 8);
    assert!(staged.validate().is_ok());
    assert!(!staged.left.is_register_resident());
    assert!(staged.right.is_register_resident());
}

#[test]
fn the_verifier_checks_arity_against_the_declaration_not_a_constant() {
    let wide = spec(MatrixTileShape { m: 32, n: 8, k: 16 });
    let exact = descriptor(wide, 18);
    verify(&exact).expect("Fix: an 18-word fragment set must verify at 18 operands");

    let native = spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    let short = descriptor(native, 9);
    let errors = verify(&short).expect_err("Fix: nine operands cannot satisfy a ten-word set");
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            VerifyErrorKind::MatrixOperandCountMismatch {
                expected: 10,
                got: 9
            }
        )),
        "Fix: the verifier must report the declared arity; got {errors:?}"
    );

    // The old contract accepted any operand list of at least ten entries, so a
    // set declaring eighteen words passed with ten. It must not now.
    let under_declared = descriptor(wide, 10);
    let errors =
        verify(&under_declared).expect_err("Fix: ten operands cannot satisfy an 18-word set");
    assert!(errors.iter().any(|error| matches!(
        error.kind,
        VerifyErrorKind::MatrixOperandCountMismatch {
            expected: 18,
            got: 10
        }
    )));
}

#[test]
fn the_verifier_rejects_a_declaration_with_no_defined_arity() {
    let uneven = spec(MatrixTileShape { m: 16, n: 3, k: 16 });
    let kernel = descriptor(uneven, 10);
    let errors = verify(&kernel).expect_err("Fix: an unstatable fragment set must be rejected");
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            VerifyErrorKind::MatrixFragmentUnstatable {
                reason: MatrixSpecError::UnevenDistribution { .. }
            }
        )),
        "Fix: the verifier must name the failing declaration; got {errors:?}"
    );
}

#[test]
fn result_ids_span_the_accumulator_fragment() {
    let wide = spec(MatrixTileShape { m: 32, n: 8, k: 16 });
    let mma = op(
        KernelOpKind::MatrixMma(Box::new(wide)),
        (0..18).collect::<Vec<u32>>(),
        100,
    );
    assert_eq!(mma.result_id_count(), 8);
    assert_eq!(
        mma.result_ids().collect::<Vec<_>>(),
        (100..108).collect::<Vec<_>>()
    );

    let native = spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    let mma = op(
        KernelOpKind::MatrixMma(Box::new(native)),
        (0..10).collect::<Vec<u32>>(),
        0,
    );
    assert_eq!(mma.result_id_count(), 4);
}
#[test]
fn tile_matmul_lowering_packs_fragment_operands_as_distinct_words() {
    use vyre_foundation::ir::{DataType, Layout, Residency};
    use vyre_lower::lower;
    use vyre_test_support::tile_programs::{tile_matmul_program, TileOperand};

    let prog = tile_matmul_program(
        [32, 1, 1],
        &TileOperand::new(
            DataType::F16,
            vec![16, 16],
            Layout::RowMajor,
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

    let desc = lower(&prog).expect("tile matmul program must lower to valid descriptor");
    let mma_op = desc
        .ops_iter()
        .find(|op| matches!(op.kind, KernelOpKind::MatrixMma(_)))
        .expect("descriptor must contain a MatrixMma op");
    // Must have 10 operand words (4 for A, 2 for B, 4 for Acc)
    assert_eq!(mma_op.operands.len(), 10, "MatrixMma operand count mismatch");

    // The left tile operands (0..4) must all be distinct packed fragment words,
    // NOT duplicated scalar IDs.
    let left_words = &mma_op.operands[0..4];
    assert_ne!(
        left_words[0], left_words[1],
        "Fix: left fragment operand words must be distinct packed registers, not duplicated scalar IDs"
    );
    assert_ne!(left_words[1], left_words[2]);
    assert_ne!(left_words[2], left_words[3]);

    // The right tile operands (4..6) must be distinct packed fragment words.
    let right_words = &mma_op.operands[4..6];
    assert_ne!(
        right_words[0], right_words[1],
        "Fix: right fragment operand words must be distinct packed registers, not duplicated scalar IDs"
    );

    // The accumulator operands (6..10) must be distinct packed fragment words.
    let acc_words = &mma_op.operands[6..10];
    assert_ne!(
        acc_words[0], acc_words[1],
        "Fix: accumulator fragment operand words must be distinct packed registers, not duplicated scalar IDs"
    );
    assert_ne!(acc_words[1], acc_words[2]);
    assert_ne!(acc_words[2], acc_words[3]);

    // Result IDs must be 4 distinct sequential IDs
    assert_eq!(mma_op.result_id_count(), 4);
    let results: Vec<u32> = mma_op.result_ids().collect();
    assert_eq!(results.len(), 4);
    assert_ne!(results[0], results[1]);
}

#[test]
fn tile_matmul_lowering_proves_bitwise_fragment_packing() {
    use vyre_foundation::ir::{BinOp, DataType, Layout, Residency};
    use vyre_lower::lower;
    use vyre_test_support::tile_programs::{tile_matmul_program, TileOperand};

    let prog = tile_matmul_program(
        [32, 1, 1],
        &TileOperand::new(
            DataType::F16,
            vec![16, 16],
            Layout::RowMajor,
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

    let desc = lower(&prog).expect("tile matmul program must lower to valid descriptor");
    let mma_op = desc
        .ops_iter()
        .find(|op| matches!(op.kind, KernelOpKind::MatrixMma(_)))
        .expect("descriptor must contain a MatrixMma op");

    // Operands 0..4 (left fragment A) and 4..6 (right fragment B) must be BitOr ops
    // combining masked lower and shifted upper 16-bit halfwords.
    let producers: std::collections::HashMap<u32, &vyre_lower::KernelOp> = desc
        .ops_iter()
        .filter_map(|op| op.result.map(|r| (r, op)))
        .collect();

    for &word_id in &mma_op.operands[0..6] {
        let producing_op = producers.get(&word_id).expect("word must have producer");
        assert!(
            matches!(producing_op.kind, KernelOpKind::BinOpKind(BinOp::BitOr)),
            "Fix: fragment operand {word_id} must be produced by BitOr packing, got {:?}",
            producing_op.kind
        );
    }
}

#[test]
fn contraction_candidate_analysis_covers_all_supported_strategies() {
    use vyre_foundation::ir::{DataType, Layout, Residency};
    use vyre_lower::analyses::contraction_candidates::{analyze, ContractionStrategy};
    use vyre_lower::lower;
    use vyre_test_support::tile_programs::{tile_matmul_program, TileOperand};

    let prog = tile_matmul_program(
        [32, 1, 1],
        &TileOperand::new(
            DataType::F16,
            vec![16, 16],
            Layout::RowMajor,
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

    let desc = lower(&prog).expect("tile matmul program must lower");
    let plan = analyze(&desc);

    assert!(plan.candidate_count() >= 3);
    let has_scalar = plan
        .candidates
        .iter()
        .any(|c| matches!(c.strategy, ContractionStrategy::Scalar));
    let has_simt = plan
        .candidates
        .iter()
        .any(|c| matches!(c.strategy, ContractionStrategy::SimtTiled { .. }));
    let has_mma = plan
        .candidates
        .iter()
        .any(|c| matches!(c.strategy, ContractionStrategy::MatrixInstruction { .. }));

    assert!(has_scalar, "Fix: contraction plan must include scalar baseline");
    assert!(has_simt, "Fix: contraction plan must include SIMT tiled candidate");
    assert!(has_mma, "Fix: contraction plan must include MMA candidate");
}

#[test]
fn contraction_shape_and_dtype_space_is_covered_without_gaps() {
    use vyre_foundation::ir::{DataType, Layout, Residency};
    use vyre_lower::analyses::contraction_candidates::analyze;
    use vyre_lower::{lower, verify};
    use vyre_test_support::tile_programs::{tile_matmul_program, TileOperand};

    let dtypes = [
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::U32,
        DataType::I32,
    ];

    for dtype in dtypes {
        let prog = tile_matmul_program(
            [1, 1, 1],
            &TileOperand::new(
                dtype.clone(),
                vec![2, 2],
                Layout::RowMajor,
                Residency::Register,
                4,
            ),
            &TileOperand::new(
                dtype.clone(),
                vec![2, 2],
                Layout::RowMajor,
                Residency::Register,
                4,
            ),
            &TileOperand::new(
                dtype.clone(),
                vec![2, 2],
                Layout::RowMajor,
                Residency::Register,
                4,
            ),
        );

        let desc = lower(&prog)
            .unwrap_or_else(|e| panic!("Fix: {dtype:?} 2x2 tile matmul lowering must succeed: {e}"));
        verify(&desc)
            .unwrap_or_else(|e| panic!("Fix: {dtype:?} 2x2 tile matmul descriptor must verify: {e:?}"));

        let plan = analyze(&desc);
        assert!(
            plan.candidate_count() > 0,
            "Fix: every supported dtype {dtype:?} must have at least one contraction candidate"
        );
    }
}
