//! Canonical Category-C intrinsic catalog.
//!
//! Each entry owns one stable identity and signature together with its neutral
//! program builder, semantic classification, and deterministic fixtures.
//! Driver registration is intentionally absent: the shared driver consumes
//! this catalog and adapts entries into its own registration contract.

use std::collections::HashMap;
use std::sync::LazyLock;

use thiserror::Error;
use vyre_foundation::dialect_lookup::{AttrSchema, Signature, TypedParam};
use vyre_foundation::ir::Program;

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

#[non_exhaustive]
pub struct OpEntry {
    pub id: &'static str,
    pub signature: Signature,
    pub build: fn() -> Program,
    pub test_inputs: Option<InputsFn>,
    pub expected_output: Option<ExpectedFn>,
    pub category: Option<&'static str>,
    pub shape: Option<OpShape>,
}

impl OpEntry {
    #[must_use]
    pub const fn new(
        id: &'static str,
        signature: Signature,
        build: fn() -> Program,
        test_inputs: Option<InputsFn>,
        expected_output: Option<ExpectedFn>,
    ) -> Self {
        Self {
            id,
            signature,
            build,
            test_inputs,
            expected_output,
            category: None,
            shape: None,
        }
    }

    #[must_use]
    pub const fn with_category(mut self, category: &'static str) -> Self {
        self.category = Some(category);
        self
    }

    #[must_use]
    pub const fn with_shape(mut self, shape: OpShape) -> Self {
        self.shape = Some(shape);
        self
    }

    #[must_use]
    pub const fn category(&self) -> Option<&'static str> {
        self.category
    }

    #[must_use]
    pub const fn shape(&self) -> Option<OpShape> {
        self.shape
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum RegistryError {
    #[error(
        "duplicate intrinsic id `{id}` with the same signature; keep exactly one canonical owner"
    )]
    DuplicateId { id: &'static str },
    #[error(
        "intrinsic id `{id}` was registered with mismatched signatures; use the canonical signature from its owning entry"
    )]
    SignatureMismatch { id: &'static str },
}

pub fn validate_entries<'a>(
    entries: impl IntoIterator<Item = &'a OpEntry>,
) -> Result<(), RegistryError> {
    let mut signatures: HashMap<&'static str, &'a Signature> = HashMap::new();
    for entry in entries {
        if let Some(first) = signatures.insert(entry.id, &entry.signature) {
            return if first == &entry.signature {
                Err(RegistryError::DuplicateId { id: entry.id })
            } else {
                Err(RegistryError::SignatureMismatch { id: entry.id })
            };
        }
    }
    Ok(())
}

inventory::collect!(OpEntry);

static ENTRIES: LazyLock<Vec<&'static OpEntry>> = LazyLock::new(|| {
    let mut entries = inventory::iter::<OpEntry>.into_iter().collect::<Vec<_>>();
    validate_entries(entries.iter().copied())
        .unwrap_or_else(|error| panic!("invalid intrinsic catalog: {error}"));
    entries.sort_unstable_by_key(|entry| entry.id);
    entries
});

pub fn all_entries() -> impl ExactSizeIterator<Item = &'static OpEntry> {
    ENTRIES.iter().copied()
}
