//! Compatibility facade for shared dataflow soundness contracts.
//!
//! `vyre-libs::dataflow` remains as a stable import path for older consumers,
//! but platform crates must not re-export downstream analysis engines. Concrete
//! IFDS, SSA, reaching-definition, callgraph, slicing, range, and related
//! analyses live in their owning engine crates and consume these shared
//! contracts from `vyre-foundation`.

use serde::{Deserialize, Serialize};

pub use vyre_foundation::soundness::{
    validate_dynamic_pipeline, validate_dynamic_primitive, validate_pipeline, validate_primitive,
    DynamicPrimitiveSoundness, DynamicSoundnessViolation, PrecisionContract, PrimitiveSoundness,
    Soundness, SoundnessTagged, SoundnessViolation,
};

/// Shared fact-schema version for security, borrowck, and Weir/Vyre bridges.
pub const SHARED_FACT_SCHEMA_VERSION: u16 = 1;

/// Cross-engine fact families accepted by the shared dataflow schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum SharedFactKind {
    /// Attacker-controlled or analysis source fact.
    Source,
    /// Security sink fact.
    Sink,
    /// Taint/dataflow reachability fact.
    Taint,
    /// Sanitizer or kill-set fact.
    Sanitizer,
    /// Program graph edge or call/control edge fact.
    GraphEdge,
    /// Rust borrow loan fact.
    BorrowLoan,
    /// Rust origin/region fact.
    BorrowOrigin,
    /// Rust origin subset/outlives fact.
    BorrowSubset,
    /// Dominance or authorization-guard fact.
    Dominance,
    /// Numeric range or bounds fact.
    Range,
    /// Source-to-sink witness/path fact.
    Witness,
}

impl SharedFactKind {
    /// Stable wire tag used by columnar schemas and release evidence.
    #[must_use]
    pub const fn wire_tag(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Sink => "sink",
            Self::Taint => "taint",
            Self::Sanitizer => "sanitizer",
            Self::GraphEdge => "graph_edge",
            Self::BorrowLoan => "borrow_loan",
            Self::BorrowOrigin => "borrow_origin",
            Self::BorrowSubset => "borrow_subset",
            Self::Dominance => "dominance",
            Self::Range => "range",
            Self::Witness => "witness",
        }
    }
}

/// Minimal cross-engine fact header.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SharedFactHeader {
    /// Schema version, currently [`SHARED_FACT_SCHEMA_VERSION`].
    pub schema_version: u16,
    /// Producer id such as `c-c11`, `rustc-nll`, or `weir`.
    pub producer: String,
    /// Shared fact family.
    pub kind: SharedFactKind,
    /// Stable producer-local fact id.
    pub fact_id: u64,
    /// Primary subject id.
    pub subject: u64,
    /// Optional object id.
    pub object: Option<u64>,
    /// Optional auxiliary id, usually a point, edge kind, or relation id.
    pub aux: Option<u64>,
    /// Stable file id, or zero when not source-spanned.
    pub file_id: u32,
    /// Start byte offset, inclusive.
    pub start_byte: u32,
    /// End byte offset, exclusive.
    pub end_byte: u32,
    /// Soundness label for the fact.
    pub soundness: Soundness,
}

impl SharedFactHeader {
    /// Build one shared fact header.
    #[must_use]
    pub fn new(
        producer: impl Into<String>,
        kind: SharedFactKind,
        fact_id: u64,
        subject: u64,
        soundness: Soundness,
    ) -> Self {
        Self {
            schema_version: SHARED_FACT_SCHEMA_VERSION,
            producer: producer.into(),
            kind,
            fact_id,
            subject,
            object: None,
            aux: None,
            file_id: 0,
            start_byte: 0,
            end_byte: 0,
            soundness,
        }
    }

    /// Attach an object id.
    #[must_use]
    pub const fn with_object(mut self, object: u64) -> Self {
        self.object = Some(object);
        self
    }

    /// Attach an auxiliary relation id.
    #[must_use]
    pub const fn with_aux(mut self, aux: u64) -> Self {
        self.aux = Some(aux);
        self
    }

    /// Attach a byte span.
    #[must_use]
    pub const fn with_span(mut self, file_id: u32, start_byte: u32, end_byte: u32) -> Self {
        self.file_id = file_id;
        self.start_byte = start_byte;
        self.end_byte = end_byte;
        self
    }

    /// Render the stable compact header used by schema contract tests.
    #[must_use]
    pub fn wire_header(&self) -> String {
        format!(
            "schema=v{};producer={};kind={};fact_id={};subject={};object={};aux={};file={};start={};end={};soundness={:?}",
            self.schema_version,
            self.producer,
            self.kind.wire_tag(),
            self.fact_id,
            self.subject,
            self.object.unwrap_or(0),
            self.aux.unwrap_or(0),
            self.file_id,
            self.start_byte,
            self.end_byte,
            self.soundness
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_security_source_fact_header_is_exact() {
        let header = SharedFactHeader::new("c-c11", SharedFactKind::Source, 1, 42, Soundness::Exact)
            .with_span(7, 100, 120);

        assert_eq!(
            header.wire_header(),
            "schema=v1;producer=c-c11;kind=source;fact_id=1;subject=42;object=0;aux=0;file=7;start=100;end=120;soundness=Exact"
        );
    }

    #[test]
    fn rust_borrow_subset_fact_header_is_exact() {
        let header = SharedFactHeader::new(
            "rustc-nll",
            SharedFactKind::BorrowSubset,
            9,
            3,
            Soundness::Exact,
        )
        .with_object(5)
        .with_aux(11);

        assert_eq!(
            header.wire_header(),
            "schema=v1;producer=rustc-nll;kind=borrow_subset;fact_id=9;subject=3;object=5;aux=11;file=0;start=0;end=0;soundness=Exact"
        );
    }

    #[test]
    fn weir_witness_fact_header_is_exact() {
        let header = SharedFactHeader::new("weir", SharedFactKind::Witness, 13, 21, Soundness::Exact)
            .with_object(34)
            .with_aux(55);

        assert_eq!(
            header.wire_header(),
            "schema=v1;producer=weir;kind=witness;fact_id=13;subject=21;object=34;aux=55;file=0;start=0;end=0;soundness=Exact"
        );
    }
}
