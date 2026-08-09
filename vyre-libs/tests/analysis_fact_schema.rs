//! Security fact adapter contract for structural analysis interchange.

#![cfg(feature = "security")]

use vyre_libs::security::{AnalysisFact, AnalysisSourceSpan, FactId, FactKind};
use vyre_spec::{
    analysis::{AnalysisFactKind, AnalysisFactRecord},
    soundness::Soundness,
};

#[test]
fn security_fact_maps_to_structural_analysis_record() {
    let fact = AnalysisFact::exact(
        FactId(1),
        FactKind::Source,
        AnalysisSourceSpan::byte_range(7, 100, 120),
        42,
    );

    let record = fact.analysis_record("c-c11");
    assert_eq!(record.schema_version, 1);
    assert_eq!(record.producer, "c-c11");
    assert_eq!(record.kind, AnalysisFactKind::Source);
    assert_eq!(record.fact_id, 1);
    assert_eq!(record.subject, 42);
    assert_eq!(record.object, None);
    assert_eq!(record.aux, None);
    assert_eq!(
        (record.file_id, record.start_byte, record.end_byte),
        (7, 100, 120)
    );
    assert_eq!(record.soundness, Soundness::Exact);

    let encoded = record.encode_json().expect("adapter record must encode");
    assert_eq!(
        AnalysisFactRecord::decode_json(&encoded).expect("adapter record must decode"),
        record
    );
}
