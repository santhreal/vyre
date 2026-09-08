//! Closed, orthogonal semantic type system (Row 102).
//!
//! Separates value representation (`ScalarType`, `VectorType`, `TensorType`, `RecordType`, `VariantType`),
//! shape semantics ([`ShapeInterner`], [`ShapeId`], [`ShapeExprId`]),
//! sparsity ([`Sparsity`]), quantization ([`QuantizationMeaning`]),
//! resource capabilities ([`ResourceCapability`]), mutability/ownership ([`OwnershipMutability`]),
//! lifetime/epoch ([`LifetimeEpoch`]), and numerical contracts ([`NumericalContract`])
//! into independent, orthogonal components.

pub mod capability;
pub mod composite;
pub mod contract;
pub mod lifetime;
pub mod ownership;
pub mod quantization;
pub mod scalar;
pub mod shape;
pub mod sparsity;
pub mod tensor;
pub mod vector;

pub use capability::ResourceCapability;
pub use composite::{RecordField, RecordType, VariantCase, VariantType};
pub use contract::{NumericalContract, RoundingMode, SaturationMode};
pub use lifetime::LifetimeEpoch;
pub use ownership::OwnershipMutability;
pub use quantization::QuantizationMeaning;
pub use scalar::ScalarType;
pub use shape::solver::{ShapeProofCertificate, ShapeProofKind, ShapeSolver};
pub use shape::{ShapeConstraint, ShapeExprId, ShapeId, ShapeInterner, SymbolicDim};
pub use sparsity::Sparsity;
pub use tensor::{TensorLayout, TensorType};
pub use vector::VectorType;

/// Closed orthogonal semantic type sum.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum SemanticType {
    /// Primitive scalar value.
    Scalar(ScalarType),
    /// Fixed-lane SIMD vector.
    Vector(VectorType),
    /// Multidimensional tensor with interned shape, sparsity, and layout.
    Tensor(TensorType),
    /// Structured product type with named/typed fields.
    Record(RecordType),
    /// Discriminated sum type with tagged cases.
    Variant(VariantType),
    /// Memory resource reference with explicit capability, ownership, and lifetime.
    Resource {
        /// Element type stored in the resource.
        element: Box<SemanticType>,
        /// Access capability.
        capability: ResourceCapability,
        /// Ownership and mutability discipline.
        ownership: OwnershipMutability,
        /// Bounding lifetime epoch.
        lifetime: LifetimeEpoch,
    },
}

impl SemanticType {
    /// Construct a scalar semantic type.
    #[must_use]
    pub const fn scalar(scalar: ScalarType) -> Self {
        Self::Scalar(scalar)
    }

    /// Construct a vector semantic type.
    #[must_use]
    pub const fn vector(element: ScalarType, lanes: u32) -> Self {
        Self::Vector(VectorType::new(element, lanes))
    }

    /// Construct a dense tensor semantic type.
    #[must_use]
    pub fn dense_tensor(element: ScalarType, shape: ShapeId) -> Self {
        Self::Tensor(TensorType::dense_row_major(element, shape))
    }

    /// Whether this type is a scalar.
    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    /// Whether this type is a tensor.
    #[must_use]
    pub const fn is_tensor(&self) -> bool {
        matches!(self, Self::Tensor(_))
    }

    /// Whether this type is a resource buffer.
    #[must_use]
    pub const fn is_resource(&self) -> bool {
        matches!(self, Self::Resource { .. })
    }
}
