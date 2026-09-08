//! Closed orthogonal vector types for the semantic type system.

use super::scalar::ScalarType;

/// Logical vector type representing SIMD / lane-parallel values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct VectorType {
    /// Scalar element type.
    pub element: ScalarType,
    /// Number of parallel lanes.
    pub lanes: u32,
}

impl VectorType {
    /// Create a new logical vector type.
    #[must_use]
    pub const fn new(element: ScalarType, lanes: u32) -> Self {
        Self { element, lanes }
    }

    /// Vec2 vector.
    #[must_use]
    pub const fn vec2(element: ScalarType) -> Self {
        Self::new(element, 2)
    }

    /// Vec4 vector.
    #[must_use]
    pub const fn vec4(element: ScalarType) -> Self {
        Self::new(element, 4)
    }

    /// Vec8 vector.
    #[must_use]
    pub const fn vec8(element: ScalarType) -> Self {
        Self::new(element, 8)
    }

    /// Vec16 vector.
    #[must_use]
    pub const fn vec16(element: ScalarType) -> Self {
        Self::new(element, 16)
    }

    /// Total bits occupied by the vector.
    #[must_use]
    pub const fn total_bits(&self) -> u32 {
        self.element.bit_width() * self.lanes
    }

    /// Total bytes occupied in storage.
    #[must_use]
    pub const fn total_bytes(&self) -> u32 {
        let bits = self.total_bits();
        (bits + 7) / 8
    }
}
