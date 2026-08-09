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

pub type Fixture = Vec<Vec<u8>>;
pub type Fixtures = Vec<Fixture>;
pub type InputsFn = fn() -> Fixtures;
pub type ExpectedFn = fn() -> Fixtures;

const NO_ATTRS: &[AttrSchema] = &[];

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareSemantic {
    UnaryU32Map,
    BarrierIdentityU32,
    FmaF32,
    InverseSqrtF32,
    SubgroupAddU32,
    SubgroupBallotU32,
    SubgroupShuffleU32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpShape {
    pub input_buffers: u8,
    pub output_buffers: u8,
    pub lane_bytes: u8,
    pub semantic: HardwareSemantic,
}

impl OpShape {
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

    #[must_use]
    pub const fn total_buffers(self) -> u8 {
        self.input_buffers + self.output_buffers
    }
}

/// Canonical semantic intrinsic registration.
pub use vyre_foundation::operation::OperationRegistration as OpEntry;

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
pub fn all_entries() -> impl Iterator<Item = &'static OpEntry> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Intrinsic)
}
