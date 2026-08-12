//! Canonical Category-C intrinsic catalog.
//!
//! Each entry owns one stable identity and signature together with its neutral
//! program builder, semantic classification, and deterministic fixtures.
//! Driver registration is intentionally absent: the shared driver consumes
//! this catalog and adapts entries into its own registration contract.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use vyre_foundation::dialect_lookup::{AttrSchema, Signature, TypedParam};
use vyre_foundation::operation::{OperationRegistry, OperationTier};

const NO_ATTRS: &[AttrSchema] = &[];

/// Unary unsigned-32 buffer signature.
pub const U32_UNARY_SIGNATURE: Signature = Signature {
    inputs: &[TypedParam {
        name: "input",
        ty: "buffer<u32>",
    }],
    outputs: &[TypedParam {
        name: "output",
        ty: "buffer<u32>",
    }],
    attrs: NO_ATTRS,
    bytes_extraction: false,
};

/// Binary unsigned-32 buffer signature.
pub const U32_BINARY_SIGNATURE: Signature = Signature {
    inputs: &[
        TypedParam {
            name: "left",
            ty: "buffer<u32>",
        },
        TypedParam {
            name: "right",
            ty: "buffer<u32>",
        },
    ],
    outputs: &[TypedParam {
        name: "output",
        ty: "buffer<u32>",
    }],
    attrs: NO_ATTRS,
    bytes_extraction: false,
};

/// Unary floating-point buffer signature.
pub const F32_UNARY_SIGNATURE: Signature = Signature {
    inputs: &[TypedParam {
        name: "input",
        ty: "buffer<f32>",
    }],
    outputs: &[TypedParam {
        name: "output",
        ty: "buffer<f32>",
    }],
    attrs: NO_ATTRS,
    bytes_extraction: false,
};

/// Ternary floating-point buffer signature.
pub const F32_TERNARY_SIGNATURE: Signature = Signature {
    inputs: &[
        TypedParam {
            name: "a",
            ty: "buffer<f32>",
        },
        TypedParam {
            name: "b",
            ty: "buffer<f32>",
        },
        TypedParam {
            name: "c",
            ty: "buffer<f32>",
        },
    ],
    outputs: &[TypedParam {
        name: "output",
        ty: "buffer<f32>",
    }],
    attrs: NO_ATTRS,
    bytes_extraction: false,
};

/// Intrinsic execution semantics used by conformance geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareSemantic {
    /// Element-wise unsigned-32 unary map.
    UnaryU32Map,
    /// Identity operation with barrier semantics.
    BarrierIdentityU32,
    /// Fused floating-point multiply-add.
    FmaF32,
    /// Floating-point inverse square root.
    InverseSqrtF32,
    /// Subgroup unsigned-32 addition.
    SubgroupAddU32,
    /// Subgroup ballot over unsigned predicates.
    SubgroupBallotU32,
    /// Subgroup unsigned-32 shuffle.
    SubgroupShuffleU32,
}

/// Buffer and lane geometry for one intrinsic facet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpShape {
    /// Number of input buffers.
    pub input_buffers: u8,
    /// Number of output buffers.
    pub output_buffers: u8,
    /// Bytes in one lane.
    pub lane_bytes: u8,
    /// Intrinsic execution semantics.
    pub semantic: HardwareSemantic,
}

impl OpShape {
    /// Construct one intrinsic geometry record.
    #[must_use]
    pub const fn new(
        input_buffers: u8,
        output_buffers: u8,
        lane_bytes: u8,
        semantic: HardwareSemantic,
    ) -> Self {
        Self {
            input_buffers,
            output_buffers,
            lane_bytes,
            semantic,
        }
    }

    /// Return total input and output buffer arity.
    #[must_use]
    pub const fn total_buffers(self) -> u8 {
        self.input_buffers + self.output_buffers
    }
}

/// Canonical semantic intrinsic view.
pub use vyre_foundation::operation::SemanticOperation;

/// Intrinsic-specific conformance geometry keyed by canonical operation identity.
pub struct IntrinsicFacet {
    /// Canonical semantic operation identifier.
    pub operation_id: &'static str,
    /// Fixture and lane geometry.
    pub shape: OpShape,
}

inventory::collect!(IntrinsicFacet);

static FACETS: LazyLock<BTreeMap<&'static str, &'static IntrinsicFacet>> = LazyLock::new(|| {
    let mut facets = BTreeMap::new();
    for facet in inventory::iter::<IntrinsicFacet> {
        assert!(
            facets.insert(facet.operation_id, facet).is_none(),
            "duplicate intrinsic facet `{}`; keep one facet per canonical operation",
            facet.operation_id
        );
    }
    facets
});

/// Resolve intrinsic-specific conformance geometry.
#[must_use]
pub fn intrinsic_facet(operation_id: &str) -> Option<&'static IntrinsicFacet> {
    FACETS.get(operation_id).copied()
}

/// Iterate canonical hardware-facing semantic intrinsic registrations.
pub fn all_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Intrinsic)
}
