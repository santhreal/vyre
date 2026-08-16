//! Contract tests for neutral affine access-map representation.
//!
//! Verifies Section 185.1:
//! - Static, symbolic, and dynamic shapes, strides, slices, transpositions, and subviews.
//! - Bounds, alias, element-size, and alignment evidence preservation through lowering.
//! - Zero-copy view admission only when consumer ABI accepts the resulting layout.

use vyre_lower::analyses::{
    AffineAccessMap, AffineMapError, ConsumerAbiRequirement, DimExtent, SliceSpec, StrideExpr,
};

#[test]
fn standard_row_major_access_map_is_contiguous() {
    let map = AffineAccessMap::standard_row_major(&[8, 16, 32], 4, 16);
    assert_eq!(map.rank(), 3);
    assert!(map.is_contiguous());
    assert_eq!(map.element_size_bytes, 4);
    assert_eq!(map.alignment_bytes, 16);
    assert_eq!(map.offset_bytes, 0);

    // Linear offset computation
    assert_eq!(map.linear_offset_bytes(&[0, 0, 0]), Some(0));
    assert_eq!(map.linear_offset_bytes(&[0, 0, 1]), Some(4));
    assert_eq!(map.linear_offset_bytes(&[0, 1, 0]), Some(32 * 4));
    assert_eq!(map.linear_offset_bytes(&[1, 0, 0]), Some(16 * 32 * 4));
}

#[test]
fn transposition_updates_strides_and_preserves_evidence() {
    let map = AffineAccessMap::standard_row_major(&[4, 8], 4, 16);
    assert!(map.is_contiguous());

    let transposed = map.transpose(&[1, 0]).expect("Fix: 2D transpose must succeed");
    assert_eq!(transposed.shape, vec![DimExtent::Static(8), DimExtent::Static(4)]);
    assert_eq!(
        transposed.strides,
        vec![StrideExpr::Static(1), StrideExpr::Static(8)]
    );
    // Transposed 2D matrix is non-contiguous in row-major interpretation
    assert!(!transposed.is_contiguous());
    assert_eq!(transposed.alignment_bytes, 16);
    assert_eq!(transposed.element_size_bytes, 4);

    // Invalid permutations are rejected
    assert!(matches!(
        map.transpose(&[0]),
        Err(AffineMapError::RankMismatch { .. })
    ));
    assert!(matches!(
        map.transpose(&[0, 0]),
        Err(AffineMapError::InvalidPermutation { .. })
    ));
}

#[test]
fn slicing_creates_valid_strided_subviews() {
    let map = AffineAccessMap::standard_row_major(&[10, 20], 4, 16);
    let slice = SliceSpec {
        start: 2,
        stop: 8,
        step: 2,
    };
    assert_eq!(slice.element_count().unwrap(), 3); // indices 2, 4, 6

    let subview = map.slice(0, slice).expect("Fix: valid slice must succeed");
    assert_eq!(subview.shape[0], DimExtent::Static(3));
    assert_eq!(subview.shape[1], DimExtent::Static(20));
    // Offset shifted by start * stride * element_size = 2 * 20 * 4 = 160 bytes
    assert_eq!(subview.offset_bytes, 160);
    // Stride along dim 0 scaled by step = 20 * 2 = 40
    assert_eq!(subview.strides[0], StrideExpr::Static(40));

    // Zero step is rejected
    let zero_step_slice = SliceSpec { start: 0, stop: 5, step: 0 };
    assert!(matches!(
        map.slice(0, zero_step_slice),
        Err(AffineMapError::ZeroStep)
    ));
}

#[test]
fn zero_copy_admission_checks_consumer_abi_requirements() {
    let contiguous_map = AffineAccessMap::standard_row_major(&[32, 64], 4, 16);
    let transposed_map = contiguous_map.transpose(&[1, 0]).unwrap();

    let strict_abi = ConsumerAbiRequirement {
        required_alignment_bytes: 16,
        element_size_bytes: 4,
        require_contiguous: true,
        allow_strided: false,
        min_leading_extent: Some(16),
    };

    // Contiguous map matches strict ABI
    assert!(contiguous_map.is_zero_copy_compatible_with(&strict_abi));

    // Transposed non-contiguous map is rejected by strict contiguous ABI
    assert!(!transposed_map.is_zero_copy_compatible_with(&strict_abi));

    // Strided-tolerant ABI admits transposed map if alignment matches
    let strided_abi = ConsumerAbiRequirement {
        required_alignment_bytes: 16,
        element_size_bytes: 4,
        require_contiguous: false,
        allow_strided: true,
        min_leading_extent: None,
    };
    assert!(transposed_map.is_zero_copy_compatible_with(&strided_abi));

    // Element size mismatch is always rejected
    let f16_abi = ConsumerAbiRequirement {
        element_size_bytes: 2,
        ..strided_abi.clone()
    };
    assert!(!contiguous_map.is_zero_copy_compatible_with(&f16_abi));

    // Alignment violation is rejected
    let high_align_abi = ConsumerAbiRequirement {
        required_alignment_bytes: 64,
        ..strided_abi
    };
    let unaligned_map = AffineAccessMap {
        alignment_bytes: 4,
        ..contiguous_map
    };
    assert!(!unaligned_map.is_zero_copy_compatible_with(&high_align_abi));
}

#[test]
fn symbolic_and_dynamic_dimensions_preserve_bounds() {
    let symbolic_dim = DimExtent::Symbolic("batch_size".to_string());
    let dynamic_dim = DimExtent::Dynamic { min: 1, max: Some(1024) };

    assert!(!symbolic_dim.is_static());
    assert!(symbolic_dim.is_non_empty());
    assert!(!dynamic_dim.is_static());
    assert!(dynamic_dim.is_non_empty());

    let map = AffineAccessMap::new(
        vec![symbolic_dim, dynamic_dim],
        vec![StrideExpr::Symbolic("stride_b".to_string()), StrideExpr::Static(1)],
        0,
        4,
        16,
    );

    assert_eq!(map.rank(), 2);
    // Non-static strides mean is_contiguous returns false
    assert!(!map.is_contiguous());
}
