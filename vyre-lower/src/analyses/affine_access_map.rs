//! Neutral affine access-map representation for lowered tensor layouts.
//!
//! Logical tensor layout, lowered access maps, and physical device layout are
//! distinct layers. This module owns the neutral affine access-map model used
//! during descriptor analysis, preserving bounds, alias, element-size, and
//! alignment evidence through lowering without assuming compile-time Rust generic dimensions.

use serde::{Deserialize, Serialize};
use std::fmt;
use super::gcd_u32;

/// Dimension extent representation for static, symbolic, and dynamic shapes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimExtent {
    /// Dimension with a known static size.
    Static(u64),
    /// Dimension whose extent is named by a symbolic parameter (e.g. "batch", "seq_len").
    Symbolic(String),
    /// Dimension with dynamic runtime bounds.
    Dynamic {
        /// Lower bound on dimension extent.
        min: u64,
        /// Optional upper bound on dimension extent.
        max: Option<u64>,
    },
}

impl DimExtent {
    /// Returns the static extent if known at compile time.
    #[must_use]
    pub const fn static_extent(&self) -> Option<u64> {
        match self {
            Self::Static(n) => Some(*n),
            _ => None,
        }
    }

    /// True if the dimension extent is statically known.
    #[must_use]
    pub const fn is_static(&self) -> bool {
        matches!(self, Self::Static(_))
    }

    /// True if the extent is known to be non-zero (or min >= 1).
    #[must_use]
    pub const fn is_non_empty(&self) -> bool {
        match self {
            Self::Static(n) => *n > 0,
            Self::Symbolic(_) => true,
            Self::Dynamic { min, .. } => *min > 0,
        }
    }
}

/// Stride expression representation for neutral access maps.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrideExpr {
    /// Contiguous dimension (stride computed from inner extents).
    Contiguous,
    /// Explicit static stride in elements.
    Static(i64),
    /// Symbolic stride named by an expression.
    Symbolic(String),
    /// Dynamic stride with known element or byte bounds.
    Dynamic {
        /// Estimated or minimum element stride.
        stride_elements: Option<i64>,
    },
}

impl StrideExpr {
    /// Returns the static stride if known.
    #[must_use]
    pub const fn static_stride(&self) -> Option<i64> {
        match self {
            Self::Static(s) => Some(*s),
            _ => None,
        }
    }
}

/// Slice specification along a single dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SliceSpec {
    /// Inclusive start coordinate in elements.
    pub start: i64,
    /// Exclusive stop coordinate in elements.
    pub stop: i64,
    /// Step / stride between sampled coordinates (must be non-zero).
    pub step: i64,
}

impl SliceSpec {
    /// Build a forward slice with unit step.
    #[must_use]
    pub const fn forward(start: i64, stop: i64) -> Self {
        Self {
            start,
            stop,
            step: 1,
        }
    }

    /// Number of elements produced by this slice.
    #[must_use]
    pub fn element_count(&self) -> Result<u64, AffineMapError> {
        if self.step == 0 {
            return Err(AffineMapError::ZeroStep);
        }
        if self.step > 0 {
            if self.stop <= self.start {
                return Ok(0);
            }
            let diff = (self.stop - self.start) as u64;
            let step = self.step as u64;
            Ok((diff + step - 1) / step)
        } else {
            if self.start <= self.stop {
                return Ok(0);
            }
            let diff = (self.start - self.stop) as u64;
            let step = (-self.step) as u64;
            Ok((diff + step - 1) / step)
        }
    }
}

/// Consumer ABI requirement against which a subview or access map is admitted for zero-copy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConsumerAbiRequirement {
    /// Required memory alignment in bytes.
    pub required_alignment_bytes: u32,
    /// Expected element size in bytes.
    pub element_size_bytes: u32,
    /// Whether the consumer ABI requires strictly contiguous dense storage.
    pub require_contiguous: bool,
    /// Maximum allowed stride discontinuity or non-standard stride multiplier.
    pub allow_strided: bool,
    /// Required leading dimension minimum extent.
    pub min_leading_extent: Option<u64>,
}

impl Default for ConsumerAbiRequirement {
    fn default() -> Self {
        Self {
            required_alignment_bytes: 4,
            element_size_bytes: 4,
            require_contiguous: true,
            allow_strided: false,
            min_leading_extent: None,
        }
    }
}

/// Errors returned by affine access-map transformations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AffineMapError {
    /// Rank mismatch between map and coordinates or transformation specification.
    RankMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions provided.
        actual: usize,
    },
    /// Dimension index out of bounds.
    DimensionOutOfBounds {
        /// Requested dimension index.
        dim: usize,
        /// Rank of the access map.
        rank: usize,
    },
    /// Permutation is not a valid bijective permutation of dimensions.
    InvalidPermutation {
        /// Rank of the access map.
        rank: usize,
    },
    /// Slice step cannot be zero.
    ZeroStep,
    /// Invalid slice range or negative extent.
    InvalidSliceRange {
        /// Slice start coordinate.
        start: i64,
        /// Slice stop coordinate.
        stop: i64,
        /// Slice step stride.
        step: i64,
    },
    /// Subview origin exceeds dimension bounds.
    OriginOutOfBounds {
        /// Dimension index.
        dim: usize,
        /// Origin coordinate.
        origin: i64,
    },
    /// Alignment requirement unsatisfied.
    AlignmentViolation {
        /// Actual alignment in bytes.
        actual: u32,
        /// Required alignment in bytes.
        required: u32,
    },
}

impl fmt::Display for AffineMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { expected, actual } => {
                write!(
                    f,
                    "affine access-map rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::DimensionOutOfBounds { dim, rank } => {
                write!(
                    f,
                    "dimension {dim} out of bounds for access-map of rank {rank}"
                )
            }
            Self::InvalidPermutation { rank } => {
                write!(
                    f,
                    "invalid transposition permutation for access-map of rank {rank}"
                )
            }
            Self::ZeroStep => write!(f, "slice step cannot be zero"),
            Self::InvalidSliceRange { start, stop, step } => {
                write!(f, "invalid slice range [{start}..{stop} step {step}]")
            }
            Self::OriginOutOfBounds { dim, origin } => {
                write!(
                    f,
                    "subview origin {origin} out of bounds for dimension {dim}"
                )
            }
            Self::AlignmentViolation { actual, required } => {
                write!(
                    f,
                    "alignment violation: actual {actual} bytes, required {required} bytes"
                )
            }
        }
    }
}

impl std::error::Error for AffineMapError {}

/// Neutral affine access-map representation.
///
/// Preserves shape, strides, base offset, element size, alignment, and alias evidence
/// across lowering transformations without forcing concrete physical device layouts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AffineAccessMap {
    /// Logical shape dimensions.
    pub shape: Vec<DimExtent>,
    /// Strides along each dimension (in elements).
    pub strides: Vec<StrideExpr>,
    /// Base byte offset from buffer start.
    pub offset_bytes: i64,
    /// Size of each element in bytes.
    pub element_size_bytes: u32,
    /// Alignment guarantee in bytes.
    pub alignment_bytes: u32,
    /// Optional alias partition / class identifier.
    pub alias_class: Option<u32>,
    /// True when bounds checks have been formally established or proven safe.
    pub bounds_verified: bool,
}

impl AffineAccessMap {
    /// Create a standard contiguous row-major access map from static extents.
    #[must_use]
    pub fn standard_row_major(
        extents: &[u64],
        element_size_bytes: u32,
        alignment_bytes: u32,
    ) -> Self {
        let shape: Vec<DimExtent> = extents.iter().copied().map(DimExtent::Static).collect();
        let mut strides = Vec::with_capacity(extents.len());
        let mut current_stride = 1_i64;
        for &extent in extents.iter().rev() {
            strides.push(StrideExpr::Static(current_stride));
            current_stride = current_stride.saturating_mul(extent.max(1) as i64);
        }
        strides.reverse();

        Self {
            shape,
            strides,
            offset_bytes: 0,
            element_size_bytes,
            alignment_bytes,
            alias_class: None,
            bounds_verified: true,
        }
    }

    /// Create an access map with mixed static, symbolic, and dynamic extents.
    #[must_use]
    pub fn new(
        shape: Vec<DimExtent>,
        strides: Vec<StrideExpr>,
        offset_bytes: i64,
        element_size_bytes: u32,
        alignment_bytes: u32,
    ) -> Self {
        Self {
            shape,
            strides,
            offset_bytes,
            element_size_bytes,
            alignment_bytes,
            alias_class: None,
            bounds_verified: false,
        }
    }

    /// Number of logical dimensions (rank).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// True if all dimensions are statically known and row-major contiguous.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() != self.strides.len() {
            return false;
        }
        let mut expected_stride = 1_i64;
        for (dim, stride) in self.shape.iter().zip(&self.strides).rev() {
            let Some(extent) = dim.static_extent() else {
                return false;
            };
            let Some(actual_stride) = stride.static_stride() else {
                return false;
            };
            if actual_stride != expected_stride {
                return false;
            }
            expected_stride = expected_stride.saturating_mul(extent.max(1) as i64);
        }
        true
    }

    /// Check if a zero-copy view is admitted under the consumer ABI requirements.
    ///
    /// WHY: A zero-copy view is admitted only when the consumer ABI accepts the resulting layout
    /// (e.g. alignment, contiguity, element-size matching).
    #[must_use]
    pub fn is_zero_copy_compatible_with(&self, consumer_abi: &ConsumerAbiRequirement) -> bool {
        if self.element_size_bytes != consumer_abi.element_size_bytes {
            return false;
        }
        if self.alignment_bytes < consumer_abi.required_alignment_bytes {
            return false;
        }
        let offset_aligned =
            (self.offset_bytes % (consumer_abi.required_alignment_bytes as i64)) == 0;
        if !offset_aligned {
            return false;
        }
        if consumer_abi.require_contiguous && !self.is_contiguous() {
            return false;
        }
        if !consumer_abi.allow_strided && !self.is_contiguous() {
            return false;
        }
        if let Some(min_leading) = consumer_abi.min_leading_extent {
            if let Some(leading) = self.shape.first().and_then(DimExtent::static_extent) {
                if leading < min_leading {
                    return false;
                }
            }
        }
        true
    }

    /// Transpose dimensions according to a permutation vector.
    pub fn transpose(&self, perm: &[usize]) -> Result<Self, AffineMapError> {
        let rank = self.rank();
        if perm.len() != rank {
            return Err(AffineMapError::RankMismatch {
                expected: rank,
                actual: perm.len(),
            });
        }
        let mut seen = vec![false; rank];
        for &p in perm {
            if p >= rank || seen[p] {
                return Err(AffineMapError::InvalidPermutation { rank });
            }
            seen[p] = true;
        }

        let new_shape: Vec<DimExtent> = perm.iter().map(|&p| self.shape[p].clone()).collect();
        let new_strides: Vec<StrideExpr> = perm.iter().map(|&p| self.strides[p].clone()).collect();

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset_bytes: self.offset_bytes,
            element_size_bytes: self.element_size_bytes,
            alignment_bytes: self.alignment_bytes,
            alias_class: self.alias_class,
            bounds_verified: self.bounds_verified,
        })
    }

    /// Slice a single dimension, returning a new subview affine access map.
    pub fn slice(&self, dim: usize, slice: SliceSpec) -> Result<Self, AffineMapError> {
        let rank = self.rank();
        if dim >= rank {
            return Err(AffineMapError::DimensionOutOfBounds { dim, rank });
        }
        let count = slice.element_count()?;
        let current_stride = match &self.strides[dim] {
            StrideExpr::Static(s) => *s,
            _ => {
                return Err(AffineMapError::InvalidSliceRange {
                    start: slice.start,
                    stop: slice.stop,
                    step: slice.step,
                })
            }
        };

        let new_stride = current_stride.saturating_mul(slice.step);
        let offset_delta = current_stride
            .saturating_mul(slice.start)
            .saturating_mul(self.element_size_bytes as i64);

        let mut new_shape = self.shape.clone();
        new_shape[dim] = DimExtent::Static(count);

        let mut new_strides = self.strides.clone();
        new_strides[dim] = StrideExpr::Static(new_stride);

        let new_offset = self.offset_bytes.saturating_add(offset_delta);
        let new_alignment = gcd_u32(self.alignment_bytes, (new_offset.abs() as u32).max(1));

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset_bytes: new_offset,
            element_size_bytes: self.element_size_bytes,
            alignment_bytes: new_alignment.max(1),
            alias_class: self.alias_class,
            bounds_verified: self.bounds_verified,
        })
    }

    /// Compute linear byte offset for static coordinates.
    #[must_use]
    pub fn linear_offset_bytes(&self, coords: &[u64]) -> Option<i64> {
        if coords.len() != self.rank() {
            return None;
        }
        let mut total_elements = 0_i64;
        for (i, &coord) in coords.iter().enumerate() {
            let stride = self.strides.get(i)?.static_stride()?;
            total_elements = total_elements.checked_add(stride.checked_mul(coord as i64)?)?;
        }
        let element_bytes = total_elements.checked_mul(self.element_size_bytes as i64)?;
        self.offset_bytes.checked_add(element_bytes)
    }
}
