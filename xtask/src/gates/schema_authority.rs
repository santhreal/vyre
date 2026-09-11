//! The `schema-authority` gate: schema version authority and typed decode enforcement.
//!
//! Enforces that:
//! 1. All persisted, transmitted, cached, signed, and conformance records derive schema identity
//!    and version constants from `vyre_spec::schema_registry::SchemaRegistry`.
//! 2. No ad-hoc schema/wire version literals are declared outside the central schema registry.
//! 3. Generic `serde_json::Value` / `toml::Value` field indexing is eliminated from production
//!    decode paths, with types decoded before validation.

use std::path::Path;

use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Gate that enforces schema authority and typed decode contracts across the workspace.
pub struct SchemaAuthorityGate;

impl GateBehavior for SchemaAuthorityGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let tree = Tree::open(&ctx.root)?;
        let mut scanned_count = 0usize;

        for path in tree.paths() {
            let path_str = path.to_string_lossy().replace('\\', "/");

            if scan::is_test_tree(path)
                || path_str.contains("/tests/")
                || path_str.contains("/benches/")
                || path_str.contains("/test_fixtures/")
                || path_str.contains("/fixtures/")
                || path_str.ends_with("/tests.rs")
                || path_str.ends_with("vyre-spec/src/schema_registry.rs")
                || path_str.ends_with("vyre-spec/src/compatibility.rs")
                || path_str.ends_with("vyre-foundation/src/serial/schema_authority.rs")
                || path_str.starts_with("xtask")
            {
                continue;
            }

            if !path_str.ends_with(".rs") {
                continue;
            }

            let Ok(text) = std::fs::read_to_string(ctx.root.join(path)) else {
                continue;
            };

            scanned_count += 1;
            inspect_source_file(path, &text, &mut report);
        }

        report.cover_complete("production sources across the workspace", scanned_count);
        Ok(report)
    }
}

/// Inspect one production source file for schema authority and untyped indexing violations.
fn inspect_source_file(path: &Path, text: &str, report: &mut Report) {
    let mut in_test_module = false;
    for (line_idx, line) in text.lines().enumerate() {
        let line_num = line_idx + 1;
        let trimmed = line.trim();

        if trimmed.starts_with("#[cfg(test)]") || trimmed.starts_with("mod tests") {
            in_test_module = true;
        }
        if in_test_module {
            continue;
        }

        // Skip comments
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }
        // Check for hardcoded wire/schema version literals
        if is_hardcoded_version_literal(trimmed) {
            report.find(Finding::in_file(
                path.to_path_buf(),
                format!(
                    "line {line_num}: hardcoded schema or wire format version literal `{trimmed}` violates schema authority"
                ),
                "derive schema and wire format version from vyre_spec::schema_registry::SchemaId",
            ));
        }

        // Check for generic serde_json::Value / toml::Value string indexing in production decode
        if is_untyped_value_indexing(trimmed, line) {
            report.find(Finding::in_file(
                path.to_path_buf(),
                format!(
                    "line {line_num}: generic Value field indexing `{trimmed}` in production decode path"
                ),
                "decode into typed struct before validation instead of dynamically indexing generic Value",
            ));
        }
    }
}

/// Returns true if the line declares a raw wire/schema version integer or string literal.
fn is_hardcoded_version_literal(line: &str) -> bool {
    let patterns = [
        "wire_format_version: 1,",
        "wire_format_version: 2,",
        "wire_format_version: 3,",
        "schema_version: 1,",
        "schema_version: 2,",
        "receipt_version: 1,",
    ];
    for pat in patterns {
        if line.contains(pat) {
            return true;
        }
    }
    false
}

/// Returns true if the line uses untyped string-based indexing on generic Value structures.
fn is_untyped_value_indexing(trimmed: &str, full_line: &str) -> bool {
    if full_line.contains("// unknown-field-preserving") {
        return false;
    }
    if (trimmed.contains("serde_json::Value::as_") || trimmed.contains("toml::Value::as_"))
        && (trimmed.contains(".get(") || trimmed.contains(".and_then("))
    {
        return true;
    }
    false
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn a_duplicated_version_literal_is_a_finding() {
        let mut report = Report::clean();
        let snippet = "let cert = Certificate { wire_format_version: 1, op_id: String::new() };";
        inspect_source_file(Path::new("src/cert.rs"), snippet, &mut report);
        assert!(
            !report.findings.is_empty(),
            "Must catch wire_format_version literal"
        );
    }

    #[test]
    fn a_generic_value_indexing_is_a_finding() {
        let mut report = Report::clean();
        let snippet = "let val = doc.get(\"name\").and_then(serde_json::Value::as_str);";
        inspect_source_file(Path::new("src/parser.rs"), snippet, &mut report);
        assert!(
            !report.findings.is_empty(),
            "Must catch untyped Value indexing"
        );
    }

    #[test]
    fn an_unknown_field_preserving_annotation_is_permitted() {
        let mut report = Report::clean();
        let snippet = "let val = doc.get(\"name\").and_then(serde_json::Value::as_str); // unknown-field-preserving";
        inspect_source_file(Path::new("src/parser.rs"), snippet, &mut report);
        assert!(
            report.findings.is_empty(),
            "Must allow marked unknown-field-preserving"
        );
    }
}
