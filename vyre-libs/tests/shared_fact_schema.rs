//! Shared fact-schema contract tests for security and external headers.

use vyre_libs::dataflow::{SharedFactHeader, SharedFactKind, Soundness};

#[cfg(feature = "security")]
use vyre_libs::security::{AnalysisFact, AnalysisSourceSpan, FactId, FactKind};

#[test]
#[cfg(feature = "security")]
fn c_security_fact_maps_to_exact_shared_header() {
    let fact = AnalysisFact::exact(
        FactId(1),
        FactKind::Source,
        AnalysisSourceSpan::byte_range(7, 100, 120),
        42,
    );

    // object and aux are absent (None) on a source fact: wire token is "-".
    assert_eq!(
        fact.shared_header("c-c11").wire_header(),
        "schema=v1;producer=c-c11;kind=source;fact_id=1;subject=42;object=-;aux=-;file=7;start=100;end=120;soundness=Exact"
    );
}


#[test]
fn external_witness_header_is_exact_shared_schema() {
    let header = SharedFactHeader::new(
        "external-dataflow",
        SharedFactKind::Witness,
        13,
        21,
        Soundness::Exact,
    )
    .with_object(34)
    .with_aux(55);

    assert_eq!(
        header.wire_header(),
        "schema=v1;producer=external-dataflow;kind=witness;fact_id=13;subject=21;object=34;aux=55;file=0;start=0;end=0;soundness=Exact"
    );
}
