//! Closed quantization meaning and scale definitions.

use super::scalar::ScalarType;

/// Semantic meaning and encoding scheme for quantized tensors and buffers.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum QuantizationMeaning {
    /// Symmetric quantization around zero (`value = quantized * scale`).
    Symmetric {
        /// Symbolic identifier for the tensor/buffer scale factor.
        scale_symbol: String,
        /// Bit precision of the quantized elements (e.g. 4, 8).
        bits: u8,
    },
    /// Asymmetric affine quantization (`value = (quantized - zero_point) * scale`).
    Asymmetric {
        /// Symbolic identifier for the tensor scale factor.
        scale_symbol: String,
        /// Symbolic identifier for the zero-point offset.
        zero_point_symbol: String,
        /// Bit precision of the quantized elements.
        bits: u8,
    },
    /// Block-wise / group-wise scaling (e.g. per-128 block scale).
    BlockScaled {
        /// Block or group size along the quantized dimension.
        block_size: u32,
        /// Scalar type of the scale values (e.g. FP8, FP16).
        scale_type: ScalarType,
    },
    /// Microscaling format (e.g. MXFP8, MXFP4).
    Microscaling {
        /// Bit precision per element.
        bits: u8,
    },
}

impl QuantizationMeaning {
    /// Bit width of individual quantized elements.
    #[must_use]
    pub const fn bits(&self) -> u8 {
        match self {
            Self::Symmetric { bits, .. }
            | Self::Asymmetric { bits, .. }
            | Self::Microscaling { bits } => *bits,
            Self::BlockScaled { scale_type, .. } => scale_type.bit_width() as u8,
        }
    }

    /// Whether this is symmetric zero-centered quantization.
    #[must_use]
    pub const fn is_symmetric(&self) -> bool {
        matches!(self, Self::Symmetric { .. } | Self::Microscaling { .. })
    }
}
