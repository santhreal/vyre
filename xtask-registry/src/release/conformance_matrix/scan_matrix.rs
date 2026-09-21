//! Scan compatibility rows: which engine supports which regex semantics, and
//! which diagnostic a refusal carries.

use std::collections::BTreeSet;
use std::path::Path;

use vyre_libs::pattern::{regex_construct_diagnostic_code, RegexConstruct};

use super::evidence::{
    ScanConformanceFinding, ScanConformanceMatrixToml, ScanConformanceRowEvidence,
};
use super::read_text_bounded;

const SCAN_CONFORMANCE_MATRIX: &str = "docs/optimization/SCAN_CONFORMANCE_MATRIX.toml";
/// Regex semantics a release must report on. This is a release decision about
/// what the scan matrix has to answer for, not a fact about linked code, and it
/// is closed both ways: a row naming a semantics outside this set is a finding,
/// and a member with no row is a finding, so neither adding nor deleting a row
/// escapes the requirement.
const REQUIRED_SCAN_CONFORMANCE_SEMANTICS: &[&str] = &[
    "leftmost_semantics",
    "overlapping_matches",
    "capture_groups",
    "byte_mode",
    "unicode_mode",
    "streaming_chunks",
    "unsupported_constructs",
];
/// Engines every scan row must report a status for. A release decision about
/// which engines are compared, closed both ways: a row missing one of these is a
/// finding, and a row naming an engine outside the set is a finding.
const REQUIRED_SCAN_CONFORMANCE_ENGINES: &[&str] = &[
    "cpu_ref",
    "cuda",
    "wgpu",
    "metal",
    "rust_regex",
    "hyperscan",
    "vectorscan",
];
/// The only statuses a scan row may claim. A status outside this vocabulary is a
/// finding, so the set cannot be widened by writing a new word in the matrix.
const ALLOWED_SCAN_ENGINE_SUPPORT: &[&str] =
    &["supported", "unsupported", "not_applicable", "experimental"];

pub(super) fn read_scan_conformance_matrix(
    vyre_root: &Path,
) -> (Vec<ScanConformanceRowEvidence>, Vec<ScanConformanceFinding>) {
    let path = vyre_root.join(SCAN_CONFORMANCE_MATRIX);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            return (
                Vec::new(),
                vec![ScanConformanceFinding {
                    semantics: "<matrix>".to_string(),
                    engine: None,
                    issue: format!(
                        "could not read `{SCAN_CONFORMANCE_MATRIX}`: {error}. Fix: keep scan compatibility rows in the canonical conformance matrix."
                    ),
                }],
            );
        }
    };
    let matrix = match toml::from_str::<ScanConformanceMatrixToml>(&text) {
        Ok(matrix) => matrix,
        Err(error) => {
            return (
                Vec::new(),
                vec![ScanConformanceFinding {
                    semantics: "<matrix>".to_string(),
                    engine: None,
                    issue: format!(
                        "could not parse `{SCAN_CONFORMANCE_MATRIX}`: {error}. Fix: declare no_refusal_code once and use [[row]] entries with semantics, engine_support, evidence_path, and unsupported_diagnostic_code."
                    ),
                }],
            );
        }
    };

    let mut findings = Vec::new();
    if matrix.schema_version != 2 {
        findings.push(ScanConformanceFinding {
            semantics: "<matrix>".to_string(),
            engine: None,
            issue: format!(
                "schema_version {} is unsupported; expected 2",
                matrix.schema_version
            ),
        });
    }
    let live_codes = live_scan_diagnostic_codes();
    if !matrix.no_refusal_code.starts_with("VYRE_SCAN_") {
        findings.push(ScanConformanceFinding {
            semantics: "<matrix>".to_string(),
            engine: None,
            issue: format!(
                "no_refusal_code `{}` is outside the VYRE_SCAN_ namespace. Fix: name the sentinel the way every other scan diagnostic is named.",
                matrix.no_refusal_code
            ),
        });
    }
    if live_codes.contains(matrix.no_refusal_code.as_str()) {
        findings.push(ScanConformanceFinding {
            semantics: "<matrix>".to_string(),
            engine: None,
            issue: format!(
                "no_refusal_code `{}` is also a code the regex compiler emits, so a row that means `no refusal path` cannot be told from a row that names a real refusal. Fix: give the sentinel a spelling the compiler does not use.",
                matrix.no_refusal_code
            ),
        });
    }

    let mut seen_semantics = BTreeSet::new();
    let mut rows = Vec::new();
    for row in matrix.row {
        let semantics = row.semantics.trim().to_string();
        if !REQUIRED_SCAN_CONFORMANCE_SEMANTICS.contains(&semantics.as_str()) {
            findings.push(ScanConformanceFinding {
                semantics: row.semantics.clone(),
                engine: None,
                issue: "unknown scan conformance semantics. Fix: use a required scan semantics id."
                    .to_string(),
            });
        } else if !seen_semantics.insert(semantics.clone()) {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: "duplicate scan conformance semantics row. Fix: keep one row per semantics."
                    .to_string(),
            });
        }

        let evidence_path = row.evidence_path.trim();
        if evidence_path.is_empty() {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: "missing evidence_path. Fix: point at the source that judges this row."
                    .to_string(),
            });
        } else {
            match read_text_bounded(&vyre_root.join(evidence_path)) {
                Err(error) => findings.push(ScanConformanceFinding {
                    semantics: semantics.clone(),
                    engine: None,
                    issue: format!(
                        "evidence_path `{evidence_path}` cannot be read: {error}. Fix: point at committed source."
                    ),
                }),
                Ok(evidence_text) => {
                    if !evidence_text.contains(SCAN_CONFORMANCE_MATRIX) {
                        findings.push(ScanConformanceFinding {
                            semantics: semantics.clone(),
                            engine: None,
                            issue: format!(
                                "evidence_path `{evidence_path}` never names `{SCAN_CONFORMANCE_MATRIX}`, so nothing in it reads this row. Fix: cite source that judges the matrix, not source that merely exists."
                            ),
                        });
                    }
                }
            }
        }

        let code = row.unsupported_diagnostic_code.trim();
        if code.is_empty() {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: format!(
                    "unsupported_diagnostic_code is missing. Fix: name the code the regex compiler emits for this semantics, or `{}` when it has no refusal path.",
                    matrix.no_refusal_code
                ),
            });
        } else if code != matrix.no_refusal_code && !live_codes.contains(code) {
            let known = live_codes.iter().copied().collect::<Vec<_>>().join(", ");
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: format!(
                    "unsupported_diagnostic_code `{code}` is a code the regex compiler never emits. Fix: name one of {known}, or `{}` when the semantics has no refusal path.",
                    matrix.no_refusal_code
                ),
            });
        }

        for engine in REQUIRED_SCAN_CONFORMANCE_ENGINES {
            match row.engine_support.get(*engine).map(String::as_str) {
                Some(status) if ALLOWED_SCAN_ENGINE_SUPPORT.contains(&status) => {}
                Some(status) => findings.push(ScanConformanceFinding {
                    semantics: semantics.clone(),
                    engine: Some((*engine).to_string()),
                    issue: format!(
                        "engine support status `{status}` is invalid. Fix: use supported, unsupported, not_applicable, or experimental."
                    ),
                }),
                None => findings.push(ScanConformanceFinding {
                    semantics: semantics.clone(),
                    engine: Some((*engine).to_string()),
                    issue: "missing engine support status. Fix: every scan row must report every required engine."
                        .to_string(),
                }),
            }
        }
        for engine in row.engine_support.keys() {
            if !REQUIRED_SCAN_CONFORMANCE_ENGINES.contains(&engine.as_str()) {
                findings.push(ScanConformanceFinding {
                    semantics: semantics.clone(),
                    engine: Some(engine.clone()),
                    issue: "unknown engine in scan conformance matrix. Fix: dedup through the required engine set."
                        .to_string(),
                });
            }
        }
        rows.push(row);
    }

    for required in REQUIRED_SCAN_CONFORMANCE_SEMANTICS {
        if !seen_semantics.contains(*required) {
            findings.push(ScanConformanceFinding {
                semantics: (*required).to_string(),
                engine: None,
                issue: "missing required scan conformance semantics row".to_string(),
            });
        }
    }

    (rows, findings)
}

/// Every diagnostic code the regex compiler can emit, taken from the live
/// construct list rather than restated here. `RegexConstruct::ALL` is closed by
/// a test in its own crate, so a construct added tomorrow reaches this gate.
fn live_scan_diagnostic_codes() -> BTreeSet<&'static str> {
    RegexConstruct::ALL
        .iter()
        .copied()
        .map(regex_construct_diagnostic_code)
        .collect()
}
