//! Closed tensor types and layout specifications.

use super::quantization::QuantizationMeaning;
use super::scalar::ScalarType;
use super::shape::{ShapeExprId, ShapeId};
use super::sparsity::Sparsity;

/// Memory layout and stride geometry of a tensor.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum TensorLayout {
    /// Contiguous row-major (C-style) layout.
    ContiguousRowMajor,
    /// Contiguous column-major (Fortran-style) layout.
    ContiguousColumnMajor,
    /// Explicit strided layout with symbolic stride expressions per dimension.
    Strided {
        /// Strides per dimension.
        strides: Vec<ShapeExprId>,
    },
    /// Block-tiled layout (e.g. for tensor cores / matrix multiplication).
    Tiled {
        /// Static tile extent per dimension.
        tile_shape: Vec<u32>,
    },
}

/// Closed tensor type specifying element type, interned shape, sparsity, quantization, and layout.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct TensorType {
    /// Element scalar type.
    pub element: ScalarType,
    /// Interned symbolic shape.
    pub shape: ShapeId,
    /// Sparsity format.
    pub sparsity: Sparsity,
    /// Optional quantization specification.
    pub quantization: Option<QuantizationMeaning>,
    /// Memory layout specification.
    pub layout: TensorLayout,
}

impl TensorType {
    /// Create a standard dense, unquantized, row-major tensor type.
    #[must_use]
    pub fn dense_row_major(element: ScalarType, shape: ShapeId) -> Self {
        Self {
            element,
            shape,
            sparsity: Sparsity::Dense,
            quantization: None,
            layout: TensorLayout::ContiguousRowMajor,
        }
    }

    /// Create a sparse tensor type.
    #[must_use]
    pub fn sparse(element: ScalarType, shape: ShapeId, sparsity: Sparsity) -> Self {
        Self {
            element,
            shape,
            sparsity,
            quantization: None,
            layout: TensorLayout::ContiguousRowMajor,
        }
    }

    /// Attach a quantization meaning to this tensor.
    #[must_use]
    pub fn with_quantization(mut self, quantization: QuantizationMeaning) -> Self {
        self.quantization = Some(quantization);
        self
    }

    /// Set an explicit memory layout.
    #[must_use]
    pub fn with_layout(mut self, layout: TensorLayout) -> Self {
        self.layout = layout;
        self
    }
}
