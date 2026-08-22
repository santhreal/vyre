//! Tests for the analysis fact table: identity, ordering, and merge.

use super::*;

fn span(offset: u32) -> AnalysisSourceSpan {
    AnalysisSourceSpan::byte_range(7, offset, offset + 4)
}

fn fact(id: u64, kind: FactKind, subject: u64) -> AnalysisFact {
    AnalysisFact::exact(FactId(id), kind, span(id as u32), subject)
}

fn table() -> AnalysisFactTable {
    let mut source = fact(1, FactKind::Source, 10);
    source
        .payload
        .insert("name".to_string(), "req.user".to_string());
    let mut edge = fact(2, FactKind::Dataflow, 10);
    edge.object = Some(20);
    edge.provenance.push(FactId(1));
    let mut sink = fact(3, FactKind::Sink, 20);
    sink.payload
        .insert("kind".to_string(), "sql.query".to_string());
    AnalysisFactTable::new(vec![sink, edge, source])
}

#[test]
fn fact_table_to_columnar_sorts_by_fact_id_and_preserves_provenance_offsets() {
    let columns = table()
        .to_columnar()
        .expect("Fix: canonical fact table should validate and pack");

    assert_eq!(columns.ids, vec![1, 2, 3]);
    assert_eq!(
        columns.kinds,
        vec![
            FactKind::Source.tag(),
            FactKind::Dataflow.tag(),
            FactKind::Sink.tag()
        ]
    );
    assert_eq!(columns.file_ids, vec![7, 7, 7]);
    assert_eq!(columns.subjects, vec![10, 10, 20]);
    assert_eq!(columns.objects, vec![0, 20, 0]);
    assert_eq!(columns.provenance_offsets, vec![0, 0, 1, 1]);
    assert_eq!(columns.provenance_ids, vec![1]);
}

#[test]
fn fact_table_rejects_duplicate_ids() {
    let error = AnalysisFactTable::new(vec![
        fact(1, FactKind::Source, 1),
        fact(1, FactKind::Sink, 2),
    ])
    .validate()
    .expect_err("Fix: duplicate fact ids must be rejected");

    assert_eq!(error, AnalysisFactError::DuplicateFactId { id: FactId(1) });
}

#[test]
fn fact_table_rejects_missing_provenance_parent() {
    let mut derived = fact(2, FactKind::Dataflow, 10);
    derived.provenance.push(FactId(99));

    let error = AnalysisFactTable::new(vec![fact(1, FactKind::Source, 10), derived])
        .validate()
        .expect_err("Fix: missing provenance parents must be rejected");

    assert_eq!(
        error,
        AnalysisFactError::MissingProvenanceParent {
            id: FactId(2),
            parent: FactId(99),
        }
    );
}

#[test]
fn fact_table_rejects_inferred_fact_without_reason() {
    let mut inferred = fact(4, FactKind::Auth, 40);
    inferred.confidence_bps = 7500;
    inferred.reason.clear();

    let error = AnalysisFactTable::new(vec![inferred])
        .validate()
        .expect_err("Fix: inferred facts need a reason");

    assert_eq!(
        error,
        AnalysisFactError::MissingInferenceReason { id: FactId(4) }
    );
}

#[test]
fn finding_proof_bundle_validates_fact_backing_and_proof_path() {
    let fact_table = table();
    let bundle = FindingProofBundle {
        finding_id: "finding.sql.source-to-sink.1".to_string(),
        query_id: "vyre-libs::security::flows_to_with_sanitizer".to_string(),
        backend_id: "cpu-ref".to_string(),
        evidence_digest: "evidence:abc123".to_string(),
        precision_contract: PrecisionContract::ZeroFalsePositive,
        soundness: Soundness::Exact,
        primitive_soundness: vec![DynamicPrimitiveSoundness::new(
            "vyre-libs::security::sanitizer_dominates",
            Soundness::Exact,
        )],
        fact_ids: vec![FactId(1), FactId(2), FactId(3)],
        proof_path: vec![
            FindingProofStep::new(FactId(1), span(1), "source"),
            FindingProofStep::new(FactId(2), span(2), "dataflow-edge"),
            FindingProofStep::new(FactId(3), span(3), "sink"),
        ],
        confidence_bps: 9800,
        reason: "source reaches sql sink without sanitizer dominance".to_string(),
    };

    bundle
        .validate_against(&fact_table)
        .expect("Fix: fact-backed proof bundle should validate");
}

#[test]
fn finding_proof_bundle_rejects_llm_only_finding_without_facts() {
    let fact_table = table();
    let bundle = FindingProofBundle {
        finding_id: "finding.llm-only".to_string(),
        query_id: "manual".to_string(),
        backend_id: "cpu-ref".to_string(),
        evidence_digest: "evidence:abc123".to_string(),
        precision_contract: PrecisionContract::ZeroFalsePositive,
        soundness: Soundness::Exact,
        primitive_soundness: vec![DynamicPrimitiveSoundness::new("manual", Soundness::Exact)],
        fact_ids: Vec::new(),
        proof_path: vec![FindingProofStep::new(FactId(1), span(1), "source")],
        confidence_bps: 5000,
        reason: "model guessed from code text".to_string(),
    };

    let error = bundle
        .validate_against(&fact_table)
        .expect_err("Fix: factless findings must be rejected");

    assert_eq!(
        error,
        AnalysisFactError::FindingHasNoFacts {
            finding_id: "finding.llm-only".to_string(),
        }
    );
}

#[test]
fn finding_proof_bundle_rejects_missing_fact_reference() {
    let fact_table = table();
    let bundle = FindingProofBundle {
        finding_id: "finding.missing-fact".to_string(),
        query_id: "vyre-libs::security::flows_to".to_string(),
        backend_id: "cpu-ref".to_string(),
        evidence_digest: "evidence:abc123".to_string(),
        precision_contract: PrecisionContract::ZeroFalsePositive,
        soundness: Soundness::Exact,
        primitive_soundness: vec![DynamicPrimitiveSoundness::new(
            "vyre-libs::security::sanitizer_dominates",
            Soundness::Exact,
        )],
        fact_ids: vec![FactId(1), FactId(42)],
        proof_path: vec![FindingProofStep::new(FactId(1), span(1), "source")],
        confidence_bps: 9000,
        reason: "source reaches sink".to_string(),
    };

    let error = bundle
        .validate_against(&fact_table)
        .expect_err("Fix: findings must not reference absent facts");

    assert_eq!(
        error,
        AnalysisFactError::FindingReferencesMissingFact {
            finding_id: "finding.missing-fact".to_string(),
            fact_id: FactId(42),
        }
    );
}

#[test]
fn finding_proof_bundle_rejects_zero_false_positive_unfiltered_mayover() {
    let fact_table = table();
    let bundle = FindingProofBundle {
        finding_id: "finding.unfiltered-mayover".to_string(),
        query_id: "vyre-libs::security::flows_to".to_string(),
        backend_id: "cpu-ref".to_string(),
        evidence_digest: "evidence:abc123".to_string(),
        precision_contract: PrecisionContract::ZeroFalsePositive,
        soundness: Soundness::MayOver,
        primitive_soundness: vec![DynamicPrimitiveSoundness::new(
            "vyre-libs::security::flows_to",
            Soundness::MayOver,
        )],
        fact_ids: vec![FactId(1), FactId(2), FactId(3)],
        proof_path: vec![
            FindingProofStep::new(FactId(1), span(1), "source"),
            FindingProofStep::new(FactId(2), span(2), "dataflow-edge"),
            FindingProofStep::new(FactId(3), span(3), "sink"),
        ],
        confidence_bps: 9000,
        reason: "unfiltered over-approximate flow should not ship as zero-FP".to_string(),
    };

    let error = bundle
        .validate_against(&fact_table)
        .expect_err("Fix: unfiltered MayOver must not validate as zero false positive");

    match error {
        AnalysisFactError::FindingSoundnessViolation {
            finding_id,
            violation,
        } => {
            assert_eq!(finding_id, "finding.unfiltered-mayover");
            assert_eq!(violation.op_id, "vyre-libs::security::flows_to");
            assert_eq!(violation.soundness, Soundness::MayOver);
            assert_eq!(violation.contract, PrecisionContract::ZeroFalsePositive);
        }
        other => panic!("unexpected soundness validation error: {other:?}"),
    }
}

#[test]
fn finding_proof_bundle_rejects_declared_soundness_mismatch() {
    let fact_table = table();
    let bundle = FindingProofBundle {
        finding_id: "finding.soundness-mismatch".to_string(),
        query_id: "vyre-libs::security::flows_to_with_sanitizer".to_string(),
        backend_id: "cpu-ref".to_string(),
        evidence_digest: "evidence:abc123".to_string(),
        precision_contract: PrecisionContract::ZeroFalsePositive,
        soundness: Soundness::Exact,
        primitive_soundness: vec![DynamicPrimitiveSoundness::new(
            "vyre-libs::security::flows_to_with_sanitizer",
            Soundness::MayOver,
        )
        .with_sanitizer_filter()],
        fact_ids: vec![FactId(1), FactId(2), FactId(3)],
        proof_path: vec![
            FindingProofStep::new(FactId(1), span(1), "source"),
            FindingProofStep::new(FactId(2), span(2), "dataflow-edge"),
            FindingProofStep::new(FactId(3), span(3), "sink"),
        ],
        confidence_bps: 9000,
        reason: "declared exact despite MayOver primitive evidence".to_string(),
    };

    let error = bundle
        .validate_against(&fact_table)
        .expect_err("Fix: declared soundness must match primitive evidence join");

    assert_eq!(
        error,
        AnalysisFactError::FindingSoundnessMismatch {
            finding_id: "finding.soundness-mismatch".to_string(),
            declared: Soundness::Exact,
            computed: Soundness::MayOver,
        }
    );
}
