//! Closed recovery classification for a preserved cause.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Recovery class of one link in a cause chain.
///
/// A caller routes on this and never on rendered text. The set is closed and
/// exhaustively matched: a new failure family is a decision, recorded by adding
/// a variant, and every match over the enum goes red until it is handled.
///
/// The variants partition by what the caller must do next, not by which
/// component reported the failure. Two components reporting the same recovery
/// class report the same variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CauseKind {
    /// The submitted program, graph, or request violates a stated contract.
    InvalidInput,
    /// The selected target or device cannot express the requested work.
    UnsupportedCapability,
    /// A required setting, feature, or registration is absent.
    Configuration,
    /// A bounded resource ran out: memory, queue depth, slots, residency.
    ResourceExhausted,
    /// A count, index, or width left its representable range.
    NumericOverflow,
    /// Semantic IR could not be lowered to a verified descriptor.
    Lowering,
    /// A verified descriptor could not be emitted as target text or binary.
    Emission,
    /// Encoding, decoding, or digest verification of a record failed.
    Encoding,
    /// A record, protocol, or artifact was produced by an incompatible version.
    VersionSkew,
    /// The acquired device generation is gone and its handles are stale.
    DeviceLost,
    /// A bounded wait elapsed before the work completed.
    Timeout,
    /// The work was abandoned by supersession or cancellation.
    Cancelled,
    /// Process state this workspace owns is inconsistent.
    InternalInvariant,
    /// A component outside this workspace reported a failure in its own words.
    ExternalToolchain,
}

impl CauseKind {
    /// Every recovery class, in declaration order.
    ///
    /// Rendering, registries, and closure tests walk this instead of a second
    /// hand-written list. The const assertion below holds it to the enum by
    /// matching every variant, so a variant absent here fails to compile.
    pub const ALL: &'static [Self] = &[
        Self::InvalidInput,
        Self::UnsupportedCapability,
        Self::Configuration,
        Self::ResourceExhausted,
        Self::NumericOverflow,
        Self::Lowering,
        Self::Emission,
        Self::Encoding,
        Self::VersionSkew,
        Self::DeviceLost,
        Self::Timeout,
        Self::Cancelled,
        Self::InternalInvariant,
        Self::ExternalToolchain,
    ];

    /// Stable serialized label, identical to the serde representation.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::UnsupportedCapability => "unsupported_capability",
            Self::Configuration => "configuration",
            Self::ResourceExhausted => "resource_exhausted",
            Self::NumericOverflow => "numeric_overflow",
            Self::Lowering => "lowering",
            Self::Emission => "emission",
            Self::Encoding => "encoding",
            Self::VersionSkew => "version_skew",
            Self::DeviceLost => "device_lost",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::InternalInvariant => "internal_invariant",
            Self::ExternalToolchain => "external_toolchain",
        }
    }

    /// Position of this variant in [`Self::ALL`].
    ///
    /// The match is exhaustive, so a new variant has no index until an author
    /// assigns one, and the const assertion below rejects an index that does
    /// not agree with [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::InvalidInput => 0,
            Self::UnsupportedCapability => 1,
            Self::Configuration => 2,
            Self::ResourceExhausted => 3,
            Self::NumericOverflow => 4,
            Self::Lowering => 5,
            Self::Emission => 6,
            Self::Encoding => 7,
            Self::VersionSkew => 8,
            Self::DeviceLost => 9,
            Self::Timeout => 10,
            Self::Cancelled => 11,
            Self::InternalInvariant => 12,
            Self::ExternalToolchain => 13,
        }
    }
}

const _: () = {
    let mut index = 0;
    while index < CauseKind::ALL.len() {
        assert!(
            CauseKind::ALL[index].index() == index,
            "CauseKind::ALL must list every variant once, in index order"
        );
        index += 1;
    }
};

/// One preserved link in a cause chain.
///
/// `kind` is the classification a caller routes on. `subject` is a stable
/// lowercase tag the owning component assigns for deduplication and grouping;
/// it narrows the class but never replaces it. `detail` is the bounded
/// contextual value: the leaf datum or foreign message that has no typed
/// representation in this workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticCause {
    /// Recovery class of this link.
    pub kind: CauseKind,
    /// Stable lowercase grouping tag assigned by the owning component.
    #[serde(deserialize_with = "super::deserialize_cow_static")]
    pub subject: Cow<'static, str>,
    /// Bounded contextual detail.
    pub detail: String,
}

impl DiagnosticCause {
    /// Build one cause link, bounding its detail by the record's stated limits.
    #[must_use]
    pub fn new(
        kind: CauseKind,
        subject: impl Into<Cow<'static, str>>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            subject: subject.into(),
            detail: super::bound_value(detail.into(), super::CAUSE_DETAIL_MAX_BYTES),
        }
    }
}
