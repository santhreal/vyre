//! Versioned cross-engine analysis fact records.
//!
//! Concrete analysis engines own their internal fact bodies. This module owns
//! the structural interchange record shared across engine boundaries.

use crate::soundness::Soundness;
use serde::{Deserialize, Serialize};

/// Current structural analysis-fact schema version.
pub const ANALYSIS_FACT_SCHEMA_VERSION: u16 = 1;

/// Failure to structurally encode, decode, or admit an analysis fact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "error", rename_all = "snake_case")]
pub enum AnalysisFactCodecError {
    /// The record carries a stale or unknown schema version.
    UnsupportedVersion {
        /// Version carried by the rejected record.
        found: u16,
        /// Current version required by this decoder.
        expected: u16,
    },
    /// The structural representation is malformed.
    Malformed {
        /// Decoder or encoder failure detail.
        message: String,
    },
}

impl core::fmt::Display for AnalysisFactCodecError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedVersion { found, expected } => write!(
                formatter,
                "unsupported analysis fact schema version {found}; expected {expected}"
            ),
            Self::Malformed { message } => {
                write!(formatter, "malformed analysis fact record: {message}")
            }
        }
    }
}

impl std::error::Error for AnalysisFactCodecError {}

/// Cross-engine fact families accepted by the structural schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFactKind {
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

/// Structural cross-engine analysis fact record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AnalysisFactRecord {
    /// Schema version, currently [`ANALYSIS_FACT_SCHEMA_VERSION`].
    pub schema_version: u16,
    /// Producer id such as `c-c11`, `rustc-nll`, or `external-dataflow`.
    pub producer: String,
    /// Shared fact family.
    pub kind: AnalysisFactKind,
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

impl AnalysisFactRecord {
    /// Build one structural analysis fact record.
    #[must_use]
    pub fn new(
        producer: impl Into<String>,
        kind: AnalysisFactKind,
        fact_id: u64,
        subject: u64,
        soundness: Soundness,
    ) -> Self {
        Self {
            schema_version: ANALYSIS_FACT_SCHEMA_VERSION,
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

    /// Reject stale or unknown schema versions before consuming this record.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisFactCodecError::UnsupportedVersion`] unless the record
    /// carries [`ANALYSIS_FACT_SCHEMA_VERSION`].
    pub const fn validate_schema_version(&self) -> Result<(), AnalysisFactCodecError> {
        if self.schema_version == ANALYSIS_FACT_SCHEMA_VERSION {
            Ok(())
        } else {
            Err(AnalysisFactCodecError::UnsupportedVersion {
                found: self.schema_version,
                expected: ANALYSIS_FACT_SCHEMA_VERSION,
            })
        }
    }

    /// Encode this record as structural JSON.
    ///
    /// # Errors
    ///
    /// Rejects non-current versions and serialization failures.
    pub fn encode_json(&self) -> Result<Vec<u8>, AnalysisFactCodecError> {
        self.validate_schema_version()?;
        serde_json::to_vec(self).map_err(|error| AnalysisFactCodecError::Malformed {
            message: error.to_string(),
        })
    }

    /// Decode and version-admit one structural JSON record.
    ///
    /// # Errors
    ///
    /// Rejects malformed JSON and every non-current schema version.
    pub fn decode_json(bytes: &[u8]) -> Result<Self, AnalysisFactCodecError> {
        let record: Self =
            serde_json::from_slice(bytes).map_err(|error| AnalysisFactCodecError::Malformed {
                message: error.to_string(),
            })?;
        record.validate_schema_version()?;
        Ok(record)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> AnalysisFactRecord {
        AnalysisFactRecord::new(
            "rustc-nll",
            AnalysisFactKind::BorrowSubset,
            9,
            3,
            Soundness::Exact,
        )
        .with_object(5)
        .with_aux(11)
        .with_span(7, 100, 120)
    }

    /// WHY: structural interchange must preserve every field without relying on
    /// hand-rendered tokens. This does not validate engine-specific fact bodies.
    #[test]
    fn structural_json_round_trip_preserves_the_record() {
        let record = sample_record();
        let encoded = record.encode_json().expect("current record must encode");
        let decoded = AnalysisFactRecord::decode_json(&encoded)
            .expect("current structural record must decode");
        assert_eq!(decoded, record);
    }

    #[test]
    fn optional_zero_ids_remain_distinct_from_absence() {
        let absent = AnalysisFactRecord::new(
            "rustc-nll",
            AnalysisFactKind::BorrowLoan,
            1,
            5,
            Soundness::Exact,
        );
        let zero = absent.clone().with_object(0).with_aux(0);

        let decoded_absent =
            AnalysisFactRecord::decode_json(&absent.encode_json().expect("absent ids must encode"))
                .expect("absent ids must decode");
        let decoded_zero =
            AnalysisFactRecord::decode_json(&zero.encode_json().expect("zero ids must encode"))
                .expect("zero ids must decode");

        assert_eq!(decoded_absent.object, None);
        assert_eq!(decoded_absent.aux, None);
        assert_eq!(decoded_zero.object, Some(0));
        assert_eq!(decoded_zero.aux, Some(0));
        assert_ne!(decoded_absent, decoded_zero);
    }

    /// WHY: stale and future records must fail closed before an engine can
    /// reinterpret their fields. This covers version admission, not body semantics.
    #[test]
    fn decoder_rejects_every_non_current_version_class() {
        for rejected in [
            ANALYSIS_FACT_SCHEMA_VERSION - 1,
            ANALYSIS_FACT_SCHEMA_VERSION + 1,
            u16::MAX,
        ] {
            let mut record = sample_record();
            record.schema_version = rejected;
            let encoded = serde_json::to_vec(&record).expect("test record must serialize");
            assert_eq!(
                AnalysisFactRecord::decode_json(&encoded),
                Err(AnalysisFactCodecError::UnsupportedVersion {
                    found: rejected,
                    expected: ANALYSIS_FACT_SCHEMA_VERSION,
                })
            );
            assert_eq!(
                record.encode_json(),
                Err(AnalysisFactCodecError::UnsupportedVersion {
                    found: rejected,
                    expected: ANALYSIS_FACT_SCHEMA_VERSION,
                })
            );
        }
    }

    #[test]
    fn malformed_structural_record_is_rejected() {
        let error = AnalysisFactRecord::decode_json(br#"{"schema_version":1"#)
            .expect_err("truncated JSON must be rejected");
        assert!(matches!(error, AnalysisFactCodecError::Malformed { .. }));
    }
}
