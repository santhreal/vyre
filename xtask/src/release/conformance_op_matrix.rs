//! The release op matrix read from `docs/optimization/OP_MATRIX.toml`.
//!
//! Reads a checked-in TOML table, so it links no vyre crate: the matrix is
//! the declared release surface, not the linked one.

use std::collections::BTreeSet;
use std::path::Path;

use crate::release::conformance_evidence_semantics::read_conformance_text;

/// The backends release evidence must cover, as the op matrix names them.
///
/// Three readers spelled this set out: the two loops below and, three times,
/// the literal `3` a row count was multiplied by. A fourth backend would have
/// had to be added in five places, and the checks that were not updated would
/// have kept passing against a smaller release surface.
///
/// A conformance certificate records its executors under the ids the
/// conformance runner writes, which are not these column names. So a reader of
/// a certificate counts against [`RELEASE_BACKEND_COLUMNS::len`] rather than
/// matching these strings: the count is the release claim, and the spelling
/// belongs to the runner that produced the record.
pub const RELEASE_BACKEND_COLUMNS: [&str; 3] = ["reference", "cuda", "wgpu"];

/// Backend columns the matrix declares and release evidence does not judge.
///
/// `spirv` is declared `experimental` on every row and `foundation_ir` is the
/// IR's own status rather than a device, so neither belongs in the release
/// row count. They are named so that a column which is neither a release
/// backend nor one of these is reported rather than ignored.
pub const UNJUDGED_BACKEND_COLUMNS: [&str; 2] = ["spirv", "foundation_ir"];

/// What `docs/optimization/OP_MATRIX.toml` requires of a release.
#[derive(Default)]
pub struct OpMatrixCatalog {
    /// Operations the matrix requires a release to cover.
    pub required_ops: BTreeSet<String>,
    /// Operation ids more than one row declares. `required_ops` is a set, so a
    /// second row naming an operation is absorbed there and surfaces only as a
    /// release backend row count that no longer matches the required op count.
    pub duplicate_required_op_rows: BTreeSet<String>,
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
    /// Cited paths that could not be read, each with the read error.
    pub unreadable_test_paths: Vec<String>,
    /// Case classes those tests were found to cover.
    pub test_case_classes: BTreeSet<&'static str>,
}

/// Read the op matrix. Read failures become `errors` so every problem in the
/// file is reported in one run.
pub fn read_conformance_required_op_matrix(vyre_root: &Path) -> OpMatrixCatalog {
    let matrix_path = vyre_root.join("docs/optimization/OP_MATRIX.toml");
    let text = match read_conformance_text(&matrix_path) {
        Ok(text) => text,
        Err(error) => {
            return OpMatrixCatalog {
                errors: vec![format!(
                    "could not read OP_MATRIX at {}: {error}",
                    matrix_path.display()
                )],
                ..OpMatrixCatalog::default()
            };
        }
    };
    let value = match toml::from_str::<toml::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            return OpMatrixCatalog {
                errors: vec![format!(
                    "could not parse OP_MATRIX at {}: {error}",
                    matrix_path.display()
                )],
                ..OpMatrixCatalog::default()
            };
        }
    };
    let rows = match value.get("op").and_then(toml::Value::as_array) {
        Some(rows) => rows,
        None => {
            return OpMatrixCatalog {
                errors: vec![format!(
                    "OP_MATRIX at {} has no [[op]] array",
                    matrix_path.display()
                )],
                ..OpMatrixCatalog::default()
            };
        }
    };
    if rows.is_empty() {
        return OpMatrixCatalog {
            errors: vec![format!(
                "OP_MATRIX at {} has zero op rows",
                matrix_path.display()
            )],
            ..OpMatrixCatalog::default()
        };
    }
    let statuses: BTreeSet<&str> = value
        .get("backend_status_values")
        .and_then(toml::Value::as_array)
        .map(|values| values.iter().filter_map(toml::Value::as_str).collect())
        .unwrap_or_default();
    let mut unaccounted_backend_columns = BTreeSet::new();
    for row in rows {
        let Some(fields) = row.as_table() else {
            continue;
        };
        for (column, declared) in fields {
            let Some(declared) = declared.as_str() else {
                continue;
            };
            if statuses.contains(declared)
                && !RELEASE_BACKEND_COLUMNS.contains(&column.as_str())
                && !UNJUDGED_BACKEND_COLUMNS.contains(&column.as_str())
            {
                unaccounted_backend_columns.insert(column.clone());
            }
        }
    }
    let errors = unaccounted_backend_columns
        .into_iter()
        .map(|column| {
            format!(
                "OP_MATRIX row column `{column}` declares a backend status and no release check \
                 reads it: add it to RELEASE_BACKEND_COLUMNS or record why it is unjudged in \
                 UNJUDGED_BACKEND_COLUMNS"
            )
        })
        .collect::<Vec<_>>();
    let mut required_ops = BTreeSet::new();
    let mut duplicate_required_op_rows = BTreeSet::new();
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
        for backend in RELEASE_BACKEND_COLUMNS {
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
        let test_evidence = inspect_conformance_test_evidence(vyre_root, &test_paths);
        for op in row_ops {
            if let Some(op) = op.as_str() {
                if !required_ops.insert(op.to_string()) {
                    duplicate_required_op_rows.insert(op.to_string());
                }
                for backend in RELEASE_BACKEND_COLUMNS {
                    match row.get(backend).and_then(toml::Value::as_str) {
                        Some("blocked_release") => {}
                        Some(status) if !status.trim().is_empty() => {
                            release_backend_rows.push(format!("{op}:{backend}:{status}"));
                            release_backend_specs.push(OpMatrixReleaseBackendSpec {
                                op_id: op.to_string(),
                                backend: backend.to_string(),
                                status: status.to_string(),
                                test_paths: test_paths.clone(),
                                unreadable_test_paths: test_evidence.unreadable_paths.clone(),
                                test_case_classes: test_evidence.case_classes.clone(),
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
        duplicate_required_op_rows,
        release_backend_rows,
        release_backend_specs,
        missing_release_backend_rows,
        blocked_release_rows,
        errors,
    }
}

/// What one release conformance artifact reports about the op matrix.
///
/// Two artifacts derive these numbers: the registered-op matrix, whose observed
/// set is the live registry, and a per-backend conformance run, whose observed
/// set is the op ids the run emitted. The observed set differs, the matrix they
/// are judged against does not, so the judging happens here once.
#[derive(Default)]
pub struct OpMatrixCoverage {
    /// Operations the matrix requires a release to cover.
    pub catalog_required_op_count: usize,
    /// Required operations the observed set covers.
    pub catalog_covered_op_count: usize,
    /// Required operations the observed set does not cover.
    pub missing_catalog_ops: Vec<String>,
    /// Release backend rows the matrix declares.
    pub release_backend_row_count: usize,
    /// Those rows claiming `supported`.
    pub supported_release_backend_row_count: usize,
    /// Rows whose declared status blocks a release.
    pub op_matrix_blocked_release_count: usize,
}

/// Judge `catalog` against the operations a caller observed, appending the
/// blockers every conformance artifact raises.
///
/// `covers` answers whether the caller observed a required operation.
/// `missing_ops_blocker` renders the caller's own wording for the ones it did
/// not, because a registry matrix reports missing registrations while a backend
/// run reports missing coverage; every other blocker reads the same either way.
/// The blockers are appended in a fixed order, which is the order both artifacts
/// already recorded them in.
pub fn evaluate_op_matrix_coverage(
    catalog: &OpMatrixCatalog,
    covers: impl Fn(&str) -> bool,
    missing_ops_blocker: impl FnOnce(usize) -> String,
    blockers: &mut Vec<String>,
) -> OpMatrixCoverage {
    let missing_catalog_ops = catalog
        .required_ops
        .iter()
        .filter(|op| !covers(op.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let catalog_required_op_count = catalog.required_ops.len();
    let catalog_covered_op_count =
        catalog_required_op_count.saturating_sub(missing_catalog_ops.len());
    if catalog.required_ops.is_empty() {
        blockers.push("OP_MATRIX contributed zero conformance-required op ids".to_string());
    }
    if !missing_catalog_ops.is_empty() {
        blockers.push(missing_ops_blocker(missing_catalog_ops.len()));
    }
    if !catalog.blocked_release_rows.is_empty() {
        blockers.push(format!(
            "OP_MATRIX contains {} release backend row(s) marked blocked_release",
            catalog.blocked_release_rows.len()
        ));
    }
    if !catalog.missing_release_backend_rows.is_empty() {
        blockers.push(format!(
            "OP_MATRIX is missing {} release backend row(s)",
            catalog.missing_release_backend_rows.len()
        ));
    }
    let supported_release_backend_row_count =
        count_supported_release_backend_rows(&catalog.release_backend_rows);
    let expected_supported_rows =
        catalog_required_op_count.saturating_mul(RELEASE_BACKEND_COLUMNS.len());
    if supported_release_backend_row_count != expected_supported_rows {
        blockers.push(format!(
            "OP_MATRIX declares {supported_release_backend_row_count} supported release backend row(s), expected {expected_supported_rows}"
        ));
    }
    let expected_release_backend_rows =
        catalog_required_op_count.saturating_mul(RELEASE_BACKEND_COLUMNS.len());
    if catalog.release_backend_rows.len() < expected_release_backend_rows {
        blockers.push(format!(
            "OP_MATRIX declares {} release backend row(s), expected {expected_release_backend_rows} for {} coverage",
            catalog.release_backend_rows.len(),
            RELEASE_BACKEND_COLUMNS.join("/")
        ));
    }
    OpMatrixCoverage {
        catalog_required_op_count,
        catalog_covered_op_count,
        missing_catalog_ops,
        release_backend_row_count: catalog.release_backend_rows.len(),
        supported_release_backend_row_count,
        op_matrix_blocked_release_count: catalog.blocked_release_rows.len(),
    }
}

/// Rows claiming `supported` for an operation.
fn count_supported_release_backend_rows(rows: &[String]) -> usize {
    rows.iter()
        .filter(|row| {
            parse_release_backend_row(row)
                .is_some_and(|(_op, _backend, status)| status == "supported")
        })
        .count()
}

fn parse_release_backend_row(row: &str) -> Option<(&str, &str, &str)> {
    let (prefix, status) = row.rsplit_once(':')?;
    let (op, backend) = prefix.rsplit_once(':')?;
    Some((op, backend, status))
}

/// What the tests one op matrix row cites were found to prove.
pub struct ConformanceTestEvidence {
    /// Case classes the readable cited tests cover.
    pub case_classes: BTreeSet<&'static str>,
    /// Cited paths that could not be read, each with the read error. A citation
    /// nothing can read proves nothing; recording it as an uncovered class
    /// instead would hide the broken citation whenever the row is not required
    /// to cover that class.
    pub unreadable_paths: Vec<String>,
}

/// Which case classes the named test files cover, from their names and text.
pub fn inspect_conformance_test_evidence(
    vyre_root: &Path,
    test_paths: &[String],
) -> ConformanceTestEvidence {
    let mut case_classes = BTreeSet::new();
    let mut unreadable_paths = Vec::new();
    for test_path in test_paths {
        let path = vyre_root.join(test_path);
        let text = match read_conformance_text(&path) {
            Ok(text) => text,
            Err(error) => {
                unreadable_paths.push(format!("{test_path} ({error})"));
                continue;
            }
        };
        let lowered = format!("{test_path}\n{text}").to_ascii_lowercase();
        case_classes.extend(crate::text_markers::classify_text(
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
    ConformanceTestEvidence {
        case_classes,
        unreadable_paths,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a catalog with one required op and the three backend cells given.
    fn catalog(reference: &str, cuda: &str, wgpu: &str) -> OpMatrixCatalog {
        let op = "vyre-libs::security::taint_pollution";
        OpMatrixCatalog {
            required_ops: [op.to_string()].into_iter().collect(),
            duplicate_required_op_rows: BTreeSet::new(),
            release_backend_rows: vec![
                format!("{op}:reference:{reference}"),
                format!("{op}:cuda:{cuda}"),
                format!("{op}:wgpu:{wgpu}"),
            ],
            release_backend_specs: Vec::new(),
            missing_release_backend_rows: Vec::new(),
            blocked_release_rows: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn blockers_for(catalog: &OpMatrixCatalog) -> Vec<String> {
        let mut blockers = Vec::new();
        evaluate_op_matrix_coverage(
            catalog,
            |_| true,
            |count| format!("{count} missing"),
            &mut blockers,
        );
        blockers
    }

    /// WHY: every required operation owes a `supported` cell for reference,
    /// cuda and wgpu, and this function had no test at all, so the threshold
    /// it turns on was unpinned.
    #[test]
    fn three_supported_cells_answer_a_required_operation() {
        assert_eq!(
            blockers_for(&catalog("supported", "supported", "supported")),
            Vec::<String>::new()
        );
    }

    /// WHY: `experimental` is the status that means nobody looked, and
    /// `not_applicable` was briefly written into this matrix by a generator
    /// that mistook what a program needs for what a backend refuses. Neither
    /// is a release answer, and a rule that accepted either would have let
    /// that mistake ship as coverage.
    #[test]
    fn an_unproven_or_refused_cell_does_not_answer_its_row() {
        for status in ["experimental", "not_applicable"] {
            let blockers = blockers_for(&catalog("supported", "supported", status));
            assert_eq!(blockers.len(), 1, "{status}: {blockers:?}");
            assert!(
                blockers[0].contains("2 supported release backend row(s), expected 3"),
                "{status}: {blockers:?}"
            );
        }
    }

    /// WHY: the rule is arithmetic over three cells per required operation, so
    /// the way it breaks is an off-by-one on the operation count rather than a
    /// wrong verdict on one cell. Two operations with one unproven cell between
    /// them pin both the threshold and the reported numbers.
    #[test]
    fn the_supported_count_is_three_cells_for_every_required_operation() {
        let mut two = catalog("supported", "supported", "supported");
        let second = "vyre-libs::bitset::and";
        two.required_ops.insert(second.to_string());
        two.release_backend_rows.extend([
            format!("{second}:reference:supported"),
            format!("{second}:cuda:supported"),
            format!("{second}:wgpu:experimental"),
        ]);
        let blockers = blockers_for(&two);
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(
            blockers[0].contains("5 supported release backend row(s), expected 6"),
            "{blockers:?}"
        );
    }

    /// WHY: `blocked_release` is stripped from the rows before counting, so
    /// without its own blocker it would read as a missing cell rather than as
    /// a release blocker, and the two carry different corrective actions.
    #[test]
    fn a_blocked_release_row_blocks_on_its_own_terms() {
        let mut blocked = catalog("supported", "supported", "supported");
        blocked.blocked_release_rows = vec!["family:wgpu".to_string()];
        let blockers = blockers_for(&blocked);
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].contains("blocked_release"), "{blockers:?}");
    }

    /// Write one op matrix carrying `columns` on its single row and read it.
    fn read_matrix(columns: &str) -> OpMatrixCatalog {
        let root = tempfile::tempdir().expect("Fix: create a temporary directory.");
        let dir = root.path().join("docs/optimization");
        std::fs::create_dir_all(&dir).expect("Fix: create the matrix directory.");
        std::fs::write(
            dir.join("OP_MATRIX.toml"),
            format!(
                "schema = 2\n\
                 backend_status_values = [\"supported\", \"experimental\", \"not_applicable\", \
                 \"blocked_release\"]\n\n\
                 [[op]]\n\
                 family = \"vyre-libs::security::taint_pollution\"\n\
                 tier = \"libs\"\n\
                 ops = [\"vyre-libs::security::taint_pollution\"]\n\
                 reference = \"supported\"\n\
                 cuda = \"supported\"\n\
                 wgpu = \"supported\"\n\
                 spirv = \"experimental\"\n\
                 foundation_ir = \"supported\"\n\
                 {columns}"
            ),
        )
        .expect("Fix: write the matrix.");
        read_conformance_required_op_matrix(root.path())
    }

    /// WHY: the matrix is generated, and a generator that grows a backend
    /// column changes nothing in any release check: the row count is derived
    /// from the columns the reader names, so an unread column is invisible.
    /// Adding one must go red until somebody decides whether release evidence
    /// covers it.
    #[test]
    fn a_backend_column_no_release_check_reads_is_an_error() {
        let catalog = read_matrix("rocm = \"supported\"\n");
        assert_eq!(catalog.errors.len(), 1, "{:?}", catalog.errors);
        assert!(
            catalog.errors[0].contains("`rocm`") && catalog.errors[0].contains("no release check"),
            "Fix: the error must name the column, got {:?}",
            catalog.errors
        );
    }

    /// WHY: the two accounted sets are the whole decision. A matrix carrying
    /// only them has to read silently, or the error above is noise the reader
    /// always emits and nobody acts on it.
    #[test]
    fn the_accounted_backend_columns_read_without_an_error() {
        let catalog = read_matrix("");
        assert!(catalog.errors.is_empty(), "{:?}", catalog.errors);
        assert_eq!(catalog.release_backend_rows.len(), RELEASE_BACKEND_COLUMNS.len());
    }

    /// WHY: a status value the matrix does not declare is not a backend cell,
    /// and treating any string column as one would report `family` and `tier`
    /// as unread backends.
    #[test]
    fn a_column_whose_value_is_not_a_declared_status_is_not_a_backend_column() {
        let catalog = read_matrix("notes = \"see VX-1\"\n");
        assert!(catalog.errors.is_empty(), "{:?}", catalog.errors);
    }
}
