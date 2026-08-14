//! The release op matrix read from `docs/optimization/OP_MATRIX.toml`.
//!
//! Reads a checked-in TOML table, so it links no vyre crate: the matrix is
//! the declared release surface, not the linked one.

use std::collections::BTreeSet;
use std::io;
use std::path::Path;

const MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES: u64 = 8_388_608;

/// What `docs/optimization/OP_MATRIX.toml` requires of a release.
pub struct OpMatrixCatalog {
    /// Operations the matrix requires a release to cover.
    pub required_ops: BTreeSet<String>,
    /// Raw `op:backend:status` rows as written.
    pub release_backend_rows: Vec<String>,
    /// The same rows parsed.
    pub release_backend_specs: Vec<OpMatrixReleaseBackendSpec>,
    /// Rows a required operation needs and the matrix omits.
    pub missing_release_backend_rows: Vec<String>,
    /// Rows whose declared status blocks a release.
    pub blocked_release_rows: Vec<String>,
    /// Problems hit reading the matrix, reported instead of raised.
    pub errors: Vec<String>,
}

/// One operation-and-backend row the op matrix declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpMatrixReleaseBackendSpec {
    /// Operation the row is about.
    pub op_id: String,
    /// Backend the row claims a status for.
    pub backend: String,
    /// Status claimed for that pair.
    pub status: String,
    /// Tests the row cites as proof.
    pub test_paths: Vec<String>,
    /// Case classes those tests were found to cover.
    pub test_case_classes: BTreeSet<&'static str>,
}

/// Read the op matrix. Read failures become `errors` so every problem in the
/// file is reported in one run.
pub fn read_conformance_required_op_matrix(vyre_root: &Path) -> OpMatrixCatalog {
    let matrix_path = vyre_root.join("docs/optimization/OP_MATRIX.toml");
    let text = match read_text_bounded(&matrix_path) {
        Ok(text) => text,
        Err(error) => {
            return OpMatrixCatalog {
                required_ops: BTreeSet::new(),
                release_backend_rows: Vec::new(),
                release_backend_specs: Vec::new(),
                missing_release_backend_rows: Vec::new(),
                blocked_release_rows: Vec::new(),
                errors: vec![format!(
                    "could not read OP_MATRIX at {}: {error}",
                    matrix_path.display()
                )],
            };
        }
    };
    let value = match toml::from_str::<toml::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            return OpMatrixCatalog {
                required_ops: BTreeSet::new(),
                release_backend_rows: Vec::new(),
                release_backend_specs: Vec::new(),
                missing_release_backend_rows: Vec::new(),
                blocked_release_rows: Vec::new(),
                errors: vec![format!(
                    "could not parse OP_MATRIX at {}: {error}",
                    matrix_path.display()
                )],
            };
        }
    };
    let rows = match value.get("op").and_then(toml::Value::as_array) {
        Some(rows) => rows,
        None => {
            return OpMatrixCatalog {
                required_ops: BTreeSet::new(),
                release_backend_rows: Vec::new(),
                release_backend_specs: Vec::new(),
                missing_release_backend_rows: Vec::new(),
                blocked_release_rows: Vec::new(),
                errors: vec![format!(
                    "OP_MATRIX at {} has no [[op]] array",
                    matrix_path.display()
                )],
            };
        }
    };
    if rows.is_empty() {
        return OpMatrixCatalog {
            required_ops: BTreeSet::new(),
            release_backend_rows: Vec::new(),
            release_backend_specs: Vec::new(),
            missing_release_backend_rows: Vec::new(),
            blocked_release_rows: Vec::new(),
            errors: vec![format!(
                "OP_MATRIX at {} has zero op rows",
                matrix_path.display()
            )],
        };
    }
    let mut required_ops = BTreeSet::new();
    let mut release_backend_rows = Vec::new();
    let mut release_backend_specs = Vec::new();
    let mut missing_release_backend_rows = Vec::new();
    let mut blocked_release_rows = Vec::new();
    for row in rows {
        let tier = row.get("tier").and_then(toml::Value::as_str).unwrap_or("");
        if tier == "foundation_ir" {
            continue;
        }
        let family = row
            .get("family")
            .and_then(toml::Value::as_str)
            .unwrap_or("<unknown>");
        for backend in ["reference", "cuda", "wgpu"] {
            if row.get(backend).and_then(toml::Value::as_str) == Some("blocked_release") {
                blocked_release_rows.push(format!("{family}:{backend}"));
            }
        }
        let Some(row_ops) = row.get("ops").and_then(toml::Value::as_array) else {
            continue;
        };
        let test_paths = row
            .get("tests")
            .and_then(toml::Value::as_array)
            .map(|tests| {
                tests
                    .iter()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let test_case_classes = classify_conformance_case_classes(vyre_root, &test_paths);
        for op in row_ops {
            if let Some(op) = op.as_str() {
                required_ops.insert(op.to_string());
                for backend in ["reference", "cuda", "wgpu"] {
                    match row.get(backend).and_then(toml::Value::as_str) {
                        Some("blocked_release") => {}
                        Some(status) if !status.trim().is_empty() => {
                            release_backend_rows.push(format!("{op}:{backend}:{status}"));
                            release_backend_specs.push(OpMatrixReleaseBackendSpec {
                                op_id: op.to_string(),
                                backend: backend.to_string(),
                                status: status.to_string(),
                                test_paths: test_paths.clone(),
                                test_case_classes: test_case_classes.clone(),
                            });
                        }
                        _ => missing_release_backend_rows.push(format!("{op}:{backend}")),
                    }
                }
            }
        }
    }
    OpMatrixCatalog {
        required_ops,
        release_backend_rows,
        release_backend_specs,
        missing_release_backend_rows,
        blocked_release_rows,
        errors: Vec::new(),
    }
}

/// Which case classes the named test files cover, from their names and text.
pub fn classify_conformance_case_classes(
    vyre_root: &Path,
    test_paths: &[String],
) -> BTreeSet<&'static str> {
    let mut classes = BTreeSet::new();
    for test_path in test_paths {
        let path = vyre_root.join(test_path);
        let text = read_text_bounded(&path).unwrap_or_default();
        let lowered = format!("{test_path}\n{text}").to_ascii_lowercase();
        classes.extend(crate::text_markers::classify_text(
            &lowered,
            &[
                ("negative", crate::text_markers::NEGATIVE_MARKERS),
                ("boundary", crate::text_markers::BOUNDARY_MARKERS),
                (
                    "adversarial",
                    &["adversarial", "hostile", "malformed", "fuzz"],
                ),
                ("unsupported_diagnostic", &["unsupported", "not_applicable"]),
            ],
        ));
    }
    classes
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(
        path,
        MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES,
        "conformance evidence",
    )
}
