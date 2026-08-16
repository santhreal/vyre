//! Which test-case classes one op and backend row can prove, and which of
//! those the row is required to prove.

use std::collections::BTreeMap;

use xtask::release::conformance_op_matrix::OpMatrixReleaseBackendSpec;

use super::evidence::{ConformanceCaseClassEvidence, ConformanceEntry, ReleaseBackendCaseRow};

pub(super) fn release_backend_case_rows(
    specs: &[OpMatrixReleaseBackendSpec],
    entries: &BTreeMap<&str, &ConformanceEntry>,
) -> Vec<ReleaseBackendCaseRow> {
    specs
        .iter()
        .map(|spec| release_backend_case_row(spec, entries.get(spec.op_id.as_str()).copied()))
        .collect()
}

fn release_backend_case_row(
    spec: &OpMatrixReleaseBackendSpec,
    entry: Option<&ConformanceEntry>,
) -> ReleaseBackendCaseRow {
    let positive = spec.status == "supported"
        && entry.is_some_and(|entry| entry.has_test_inputs && entry.has_expected_output);
    let byte_output = entry.is_some_and(|entry| entry.has_expected_output);
    let unsupported_diagnostic =
        spec.status != "supported" || spec.test_case_classes.contains("unsupported_diagnostic");
    let class_sources = [
        (
            "positive",
            positive,
            if positive {
                "registered fixture inputs and expected output"
            } else {
                "missing supported-row fixture input/output pair"
            },
        ),
        (
            "negative",
            spec.test_case_classes.contains("negative"),
            "OP_MATRIX referenced tests contain reject/error/invalid/unsupported evidence",
        ),
        (
            "boundary",
            spec.test_case_classes.contains("boundary"),
            "OP_MATRIX referenced tests contain boundary/zero/limit/overflow evidence",
        ),
        (
            "adversarial",
            spec.test_case_classes.contains("adversarial"),
            "OP_MATRIX referenced tests contain adversarial/hostile/malformed evidence",
        ),
        (
            "byte_output",
            byte_output,
            if byte_output {
                "registered expected byte output"
            } else {
                "missing registered expected byte output"
            },
        ),
        (
            "unsupported_diagnostic",
            unsupported_diagnostic,
            if spec.status == "supported" {
                "supported row; unsupported diagnostics are only covered when referenced tests exercise them"
            } else {
                "OP_MATRIX backend status is non-supported and therefore carries explicit unsupported/not-applicable diagnostics"
            },
        ),
    ];
    let case_classes = class_sources
        .into_iter()
        .map(|(class, covered, source)| ConformanceCaseClassEvidence {
            class,
            covered,
            source: source.to_string(),
        })
        .collect::<Vec<_>>();
    let required_case_classes = required_case_classes_for_status(&spec.status);
    let missing_required_case_classes = required_case_classes
        .iter()
        .copied()
        .filter(|class| {
            !case_classes
                .iter()
                .any(|evidence| evidence.class == *class && evidence.covered)
        })
        .collect::<Vec<_>>();
    let blockers = missing_required_case_classes
        .iter()
        .map(|class| {
            format!(
                "conformance op/backend row `{}:{}` status `{}` is missing required `{class}` case-class evidence",
                spec.op_id, spec.backend, spec.status
            )
        })
        .collect::<Vec<_>>();
    ReleaseBackendCaseRow {
        op_id: spec.op_id.clone(),
        backend: spec.backend.clone(),
        status: spec.status.clone(),
        test_paths: spec.test_paths.clone(),
        case_classes,
        required_case_classes,
        missing_required_case_classes,
        blockers,
    }
}

fn required_case_classes_for_status(status: &str) -> Vec<&'static str> {
    if status == "supported" {
        vec!["positive", "byte_output"]
    } else {
        vec!["unsupported_diagnostic"]
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::super::evidence::REPORTED_CONFORMANCE_CASE_CLASSES;
    use super::*;

    fn evidence<'a>(
        row: &'a ReleaseBackendCaseRow,
        class: &str,
    ) -> &'a ConformanceCaseClassEvidence {
        row.case_classes
            .iter()
            .find(|evidence| evidence.class == class)
            .expect("case class evidence must be reported")
    }

    fn assert_all_case_classes_reported(row: &ReleaseBackendCaseRow) {
        assert_eq!(
            row.case_classes.len(),
            REPORTED_CONFORMANCE_CASE_CLASSES.len()
        );
        for class in REPORTED_CONFORMANCE_CASE_CLASSES {
            assert!(
                row.case_classes
                    .iter()
                    .any(|evidence| evidence.class == *class),
                "missing reported case class {class}"
            );
        }
    }

    #[test]
    fn release_backend_case_rows_report_all_case_classes_for_external_flow_row() {
        let entry = ConformanceEntry {
            id: "external.flow.alias_ifds".to_string(),
            requires_fixture: true,
            has_test_inputs: true,
            has_expected_output: true,
            tolerance_ulp: 0,
        };
        let spec = OpMatrixReleaseBackendSpec {
            op_id: entry.id.clone(),
            backend: "cuda".to_string(),
            status: "supported".to_string(),
            test_paths: vec!["tests/external_flow_boundary_adversarial.rs".to_string()],
            test_case_classes: BTreeSet::from(["negative", "boundary", "adversarial"]),
        };
        let entries = BTreeMap::from([(entry.id.as_str(), &entry)]);

        let rows = release_backend_case_rows(&[spec], &entries);
        let row = &rows[0];

        assert_all_case_classes_reported(row);
        assert_eq!(row.required_case_classes, vec!["positive", "byte_output"]);
        assert!(row.missing_required_case_classes.is_empty());
        assert!(row.blockers.is_empty());
        assert!(evidence(row, "positive").covered);
        assert!(evidence(row, "negative").covered);
        assert!(evidence(row, "boundary").covered);
        assert!(evidence(row, "adversarial").covered);
        assert!(evidence(row, "byte_output").covered);
        assert!(!evidence(row, "unsupported_diagnostic").covered);
    }

    #[test]
    fn release_backend_case_rows_accept_non_supported_rows_with_unsupported_diagnostic() {
        let spec = OpMatrixReleaseBackendSpec {
            op_id: "vyre-libs::scan::prefix_sum_u32".to_string(),
            backend: "cuda".to_string(),
            status: "not_applicable".to_string(),
            test_paths: Vec::new(),
            test_case_classes: BTreeSet::new(),
        };
        let entries = BTreeMap::new();

        let rows = release_backend_case_rows(&[spec], &entries);
        let row = &rows[0];

        assert_all_case_classes_reported(row);
        assert_eq!(row.required_case_classes, vec!["unsupported_diagnostic"]);
        assert!(row.missing_required_case_classes.is_empty());
        assert!(row.blockers.is_empty());
        assert!(!evidence(row, "positive").covered);
        assert!(!evidence(row, "byte_output").covered);
        assert!(evidence(row, "unsupported_diagnostic").covered);
    }

    #[test]
    fn release_backend_case_rows_block_supported_rows_missing_byte_output() {
        let entry = ConformanceEntry {
            id: "vyre-libs::scan::prefix_sum_u32".to_string(),
            requires_fixture: true,
            has_test_inputs: true,
            has_expected_output: false,
            tolerance_ulp: 0,
        };
        let spec = OpMatrixReleaseBackendSpec {
            op_id: entry.id.clone(),
            backend: "wgpu".to_string(),
            status: "supported".to_string(),
            test_paths: Vec::new(),
            test_case_classes: BTreeSet::new(),
        };
        let entries = BTreeMap::from([(entry.id.as_str(), &entry)]);

        let rows = release_backend_case_rows(&[spec], &entries);
        let row = &rows[0];

        assert_all_case_classes_reported(row);
        assert_eq!(row.required_case_classes, vec!["positive", "byte_output"]);
        assert_eq!(
            row.missing_required_case_classes,
            vec!["positive", "byte_output"]
        );
        assert_eq!(row.blockers.len(), 2);
        assert!(!evidence(row, "positive").covered);
        assert!(!evidence(row, "byte_output").covered);
    }
}
