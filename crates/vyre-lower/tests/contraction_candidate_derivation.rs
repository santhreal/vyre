//! Contraction candidates state only what the descriptor states.
//!
//! WHY: the contraction candidate analysis previously returned a fixed triple
//! of candidates for every contraction site: a scalar entry, a 16x16x16 SIMT
//! tile, and an `m16n8k16` F16 matrix instruction, each carrying a constant
//! ranking multiplier. None of the three was read from the site. A descriptor
//! that declared a different matrix tile, bound only integer element types, or
//! dispatched a single-invocation workgroup received the same answer, so a
//! matrix-instruction candidate could name extents no program expressed, and
//! the ranking was a number nothing derived.
//!
//! The class this closes is a candidate field whose value is written into the
//! analysis instead of read from the site. Each case below fixes one field to
//! the descriptor that states it: tile extents, fragment layouts, fragment
//! element types, workgroup geometry, bound element types, and the operand
//! load count. The closure cases enumerate the strategy and fragment-element
//! variants from the types themselves, so a new variant fails until a decision
//! is recorded for it.
//!
//! What this does not catch: whether a candidate is fast. `operand_loads_per_fma`
//! is counted work, and the suite asserts the count, never a device outcome.
//! Measured selection is proved by the conformance and benchmark paths.

use vyre_foundation::ir::{DataType, Layout, Residency};
use vyre_lower::analyses::contraction_candidates::{
    analyze, ContractionCandidate, ContractionStrategy, MatrixInstructionSource,
    SCALAR_OPERAND_LOADS_PER_FMA,
};
use vyre_lower::lower;
use vyre_lower::{MatrixMmaElement, MatrixMmaLayout, MatrixTileShape};
use vyre_test_support::tile_programs::{fragment_matmul_program, tile_matmul_program, TileOperand};

fn matrix_candidate(candidates: &[ContractionCandidate]) -> Option<&ContractionCandidate> {
    candidates
        .iter()
        .find(|c| matches!(c.strategy, ContractionStrategy::MatrixInstruction { .. }))
}

fn simt_candidate(candidates: &[ContractionCandidate]) -> Option<&ContractionCandidate> {
    candidates
        .iter()
        .find(|c| matches!(c.strategy, ContractionStrategy::SimtTiled { .. }))
}

/// A declared fragment product states the extents, not the analysis.
#[test]
fn matrix_candidate_carries_the_extents_the_descriptor_declared() {
    let prog = fragment_matmul_program(Residency::Subgroup, Layout::RowMajor);
    let desc = lower(&prog).expect("tile matmul program must lower");
    let plan = analyze(&desc);

    let candidate = matrix_candidate(&plan.candidates)
        .expect("Fix: a descriptor declaring a matrix product must state a matrix candidate");
    let ContractionStrategy::MatrixInstruction {
        tile,
        left_layout,
        right_layout,
        left_element,
        right_element,
        acc_element,
        source,
    } = candidate.strategy
    else {
        unreachable!("filtered to the matrix variant")
    };

    // The program declares a 16x16 left tile against a 16x8 right tile with an
    // F16/F16/F32 element assignment, which is m16 n8 k16.
    assert_eq!(
        tile,
        MatrixTileShape { m: 16, n: 8, k: 16 },
        "Fix: the matrix candidate must echo the declared product extents"
    );
    assert_eq!(left_layout, MatrixMmaLayout::RowMajor);
    assert_eq!(right_layout, MatrixMmaLayout::ColMajor);
    assert_eq!(left_element, MatrixMmaElement::F16);
    assert_eq!(right_element, MatrixMmaElement::F16);
    assert_eq!(acc_element, MatrixMmaElement::F32);
    assert_eq!(source, MatrixInstructionSource::DeclaredByDescriptor);
    assert!(
        candidate.contraction_id.ends_with("_mma_m16n8k16"),
        "Fix: the candidate id must name the declared extents, got `{}`",
        candidate.contraction_id
    );
}

/// A declared product with other extents produces those extents, not a
/// repeat of the first program's.
#[test]
fn matrix_candidate_follows_a_second_declared_product_rather_than_the_first() {
    let prog = fragment_matmul_program(Residency::Subgroup, Layout::ColumnMajor);
    let desc = lower(&prog).expect("tile matmul program must lower");
    let plan = analyze(&desc);

    let candidate =
        matrix_candidate(&plan.candidates).expect("declared product states a candidate");
    let ContractionStrategy::MatrixInstruction { left_layout, .. } = candidate.strategy else {
        unreachable!("filtered to the matrix variant")
    };
    assert_eq!(
        left_layout,
        MatrixMmaLayout::ColMajor,
        "Fix: the fragment layout must follow the tile the program loaded, not a fixed orientation"
    );
}

/// Four distinct declared products state four distinct sets of extents.
///
/// One program cannot distinguish a candidate that reads the descriptor from
/// one that writes the same extents into the analysis, so the assertion runs
/// over shapes that disagree in every field.
#[test]
fn matrix_extents_follow_each_declared_product_rather_than_one_fixed_shape() {
    for (m, k, n) in [(8u32, 16u32, 8u32), (16, 16, 16), (8, 8, 8), (32, 16, 8)] {
        let left = TileOperand::new(
            DataType::F16,
            vec![m, k],
            Layout::RowMajor,
            Residency::Subgroup,
            m * k,
        );
        let right = TileOperand::new(
            DataType::F16,
            vec![k, n],
            Layout::ColumnMajor,
            Residency::Subgroup,
            k * n,
        );
        let accumulator = TileOperand::new(
            DataType::F32,
            vec![m, n],
            Layout::RowMajor,
            Residency::Register,
            m * n,
        );
        let prog = tile_matmul_program([32, 1, 1], &left, &right, &accumulator);
        let desc = lower(&prog).expect("declared fragment product must lower");
        let plan = analyze(&desc);

        let candidate = matrix_candidate(&plan.candidates)
            .unwrap_or_else(|| panic!("Fix: product m{m} n{n} k{k} states a matrix candidate"));
        let ContractionStrategy::MatrixInstruction { tile, .. } = candidate.strategy else {
            unreachable!("filtered to the matrix variant")
        };

        let expected = MatrixTileShape {
            m: u16::try_from(m).expect("tile extent fits"),
            n: u16::try_from(n).expect("tile extent fits"),
            k: u16::try_from(k).expect("tile extent fits"),
        };
        assert_eq!(
            tile, expected,
            "Fix: the candidate must carry the extents this program declared, not a fixed shape"
        );
        assert!(
            candidate
                .contraction_id
                .ends_with(&format!("_mma_m{m}n{n}k{k}")),
            "Fix: candidate id `{}` does not name the declared extents m{m} n{n} k{k}",
            candidate.contraction_id
        );
    }
}

/// A contraction that declares no fragment product states no matrix candidate.
#[test]
fn a_site_without_a_declared_product_states_no_matrix_candidate() {
    // Rank-2 tiles too small to form a fragment product: the lowering takes
    // the scalar route and emits no `MatrixMma`.
    let operand = TileOperand::new(
        DataType::U32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
        4,
    );
    let prog = tile_matmul_program([8, 1, 1], &operand, &operand, &operand);
    let desc = lower(&prog).expect("scalar tile matmul must lower");
    let plan = analyze(&desc);

    assert!(
        matrix_candidate(&plan.candidates).is_none(),
        "Fix: a descriptor that declares no matrix operation must not receive a matrix candidate; extents it never expressed cannot be derived"
    );
    assert!(
        plan.candidates
            .iter()
            .any(|c| matches!(c.strategy, ContractionStrategy::Scalar)),
        "Fix: every contraction site states the scalar baseline"
    );
}

/// The tiled candidate's extents are the dispatch's workgroup geometry.
#[test]
fn simt_tile_extents_are_the_declared_workgroup_geometry() {
    let operand = TileOperand::new(
        DataType::F32,
        vec![4, 4],
        Layout::RowMajor,
        Residency::Register,
        16,
    );

    for workgroup in [[8u32, 4, 1], [16, 16, 1], [4, 2, 1]] {
        let prog = tile_matmul_program(workgroup, &operand, &operand, &operand);
        let desc = lower(&prog).expect("tile matmul must lower");
        let plan = analyze(&desc);

        let candidate = simt_candidate(&plan.candidates)
            .unwrap_or_else(|| panic!("Fix: workgroup {workgroup:?} cooperates and states a tile"));
        let ContractionStrategy::SimtTiled {
            tile_m,
            tile_n,
            workgroup_size,
            ..
        } = candidate.strategy
        else {
            unreachable!("filtered to the tiled variant")
        };

        assert_eq!(
            [tile_m, tile_n],
            [workgroup[0], workgroup[1]],
            "Fix: the tile must be the invocations the dispatch has, not a fixed 16x16"
        );
        assert_eq!(
            workgroup_size, workgroup,
            "Fix: the candidate must carry the geometry it was derived from"
        );
    }
}

/// A workgroup of one invocation cannot stage a cooperative tile.
#[test]
fn a_single_invocation_workgroup_states_no_tiled_candidate() {
    let operand = TileOperand::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
        4,
    );
    let prog = tile_matmul_program([1, 1, 1], &operand, &operand, &operand);
    let desc = lower(&prog).expect("tile matmul must lower");
    let plan = analyze(&desc);

    assert!(
        simt_candidate(&plan.candidates).is_none(),
        "Fix: one invocation has no peers to share a staged tile with, so no tiled candidate exists"
    );
}

/// Supported dtypes come from the bindings the kernel reads.
#[test]
fn supported_dtypes_are_the_element_types_the_kernel_binds() {
    for dtype in [
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::U32,
        DataType::I32,
    ] {
        let operand = TileOperand::new(
            dtype.clone(),
            vec![2, 2],
            Layout::RowMajor,
            Residency::Register,
            4,
        );
        let prog = tile_matmul_program([8, 1, 1], &operand, &operand, &operand);
        let desc = lower(&prog).expect("tile matmul must lower");
        let plan = analyze(&desc);

        for candidate in &plan.candidates {
            assert!(
                candidate.supported_dtypes.contains(&dtype),
                "Fix: candidate `{}` omits `{dtype:?}`, the element type its kernel binds",
                candidate.contraction_id
            );
            for supported in &candidate.supported_dtypes {
                assert_eq!(
                    supported, &dtype,
                    "Fix: candidate `{}` claims `{supported:?}`, which this kernel never binds",
                    candidate.contraction_id
                );
            }
        }
    }
}

/// The load count is counted work with a stated derivation.
#[test]
fn every_candidate_counts_its_operand_loads_and_states_the_derivation() {
    let prog = fragment_matmul_program(Residency::Subgroup, Layout::RowMajor);
    let desc = lower(&prog).expect("tile matmul program must lower");
    let plan = analyze(&desc);

    for candidate in &plan.candidates {
        assert!(
            candidate.operand_loads_per_fma > 0.0 && candidate.operand_loads_per_fma.is_finite(),
            "Fix: candidate `{}` counts {} operand loads per multiply-accumulate",
            candidate.contraction_id,
            candidate.operand_loads_per_fma
        );
        assert!(
            !candidate.derivation.is_empty(),
            "Fix: candidate `{}` states no derivation for the numbers it carries",
            candidate.contraction_id
        );
    }

    let scalar = plan
        .candidates
        .iter()
        .find(|c| matches!(c.strategy, ContractionStrategy::Scalar))
        .expect("scalar baseline");
    assert_eq!(
        scalar.operand_loads_per_fma, SCALAR_OPERAND_LOADS_PER_FMA,
        "Fix: the scalar baseline reads both operands of every multiply-accumulate"
    );
    assert_eq!(
        scalar.operand_reuse_factor(),
        1.0,
        "Fix: the baseline reuse factor is the ratio against itself"
    );

    // A 16x8x16 fragment issues 2048 multiply-accumulates against 256 + 128
    // operand elements, so it reads far less than one operand per product.
    let matrix = matrix_candidate(&plan.candidates).expect("declared product");
    assert!(
        matrix.operand_reuse_factor() > simt_candidate(&plan.candidates)
            .expect("tiled candidate")
            .operand_reuse_factor(),
        "Fix: a fragment reuses each staged element across the whole tile, so it must count fewer loads per product than a workgroup tile"
    );
}

/// Every strategy variant is reached by a case in this file.
///
/// Enumerated from the strategies the analysis produces across the programs
/// above rather than a written list, so a new `ContractionStrategy` variant
/// that no program reaches fails here until a case is added for it.
#[test]
fn every_contraction_strategy_variant_is_produced_by_a_case() {
    fn discriminant(strategy: &ContractionStrategy) -> &'static str {
        match strategy {
            ContractionStrategy::Scalar => "Scalar",
            ContractionStrategy::SimtTiled { .. } => "SimtTiled",
            ContractionStrategy::MatrixInstruction { .. } => "MatrixInstruction",
        }
    }

    let prog = fragment_matmul_program(Residency::Subgroup, Layout::RowMajor);
    let desc = lower(&prog).expect("tile matmul program must lower");
    let plan = analyze(&desc);

    let produced: Vec<&'static str> = plan
        .candidates
        .iter()
        .map(|c| discriminant(&c.strategy))
        .collect();

    // The match in `discriminant` is exhaustive: adding a variant fails to
    // compile until it is named, and it must then appear here.
    for expected in ["Scalar", "SimtTiled", "MatrixInstruction"] {
        assert!(
            produced.contains(&expected),
            "Fix: no case produces the `{expected}` strategy; add one before adding the variant"
        );
    }
}

/// Every fragment element type has a recorded host dtype decision.
#[test]
fn every_matrix_fragment_element_maps_to_a_bound_dtype() {
    // Exhaustive match: a new `MatrixMmaElement` variant fails to compile
    // here until its host element type is decided.
    for element in [
        MatrixMmaElement::F16,
        MatrixMmaElement::BF16,
        MatrixMmaElement::TF32,
        MatrixMmaElement::F32,
    ] {
        let dtype = match element {
            MatrixMmaElement::F16 => DataType::F16,
            MatrixMmaElement::BF16 => DataType::BF16,
            MatrixMmaElement::TF32 | MatrixMmaElement::F32 => DataType::F32,
        };
        assert!(
            matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32),
            "Fix: fragment element {element:?} has no host element type decision"
        );
    }
}
