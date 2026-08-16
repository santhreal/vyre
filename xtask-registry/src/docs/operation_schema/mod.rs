//! Canonical generated operation schema built from live registrations.
//!
//! ## Layout
//!
//! - `schema` the wire types and the version they are pinned to
//! - `signature` one operation signature from a Program or a declaration
//! - `composition` the nested-operation chain a Program spells out
//! - `routing` category, feature route and the manifests that must declare it
//! - `assemble` the live registry join that produces the document
//! - `validate` the judgment every document has to pass

use std::io;
use std::path::Path;

use xtask::gate::{Gate, GateCtx, GateError, Report};

use self::assemble::build;
use self::schema::OperationSchema;
use self::validate::validate_schema;

pub mod assemble;
mod composition;
mod routing;
pub mod schema;
mod signature;
mod validate;

const DEFAULT_OUTPUT: &str = "docs/generated/OP_SCHEMA.json";
const MAX_SCHEMA_BYTES: u64 = 16_777_216;

/// Holds the canonical live operation contract schema to the registry.
pub struct OperationSchemaGate;

impl Gate for OperationSchemaGate {
    fn name(&self) -> &'static str {
        "operation-schema"
    }

    fn help(&self) -> &'static str {
        "Hold the canonical live operation contract schema to the registry; --write regenerates it, --validate PATH judges one document"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let schema = match build() {
            Ok(schema) => schema,
            Err(errors) => {
                return Ok(Report::from_messages(
                    errors,
                    "repair the registration the schema rejects, then run the gate again",
                ));
            }
        };
        if let Some(path) = ctx.flag("--validate") {
            let candidate = read_schema(Path::new(path)).map_err(|error| {
                GateError::new(error, "pass a readable schema document after --validate")
            })?;
            let mut report = match validate_schema(&candidate, Some(&schema)) {
                Ok(()) => Report::clean(),
                Err(errors) => Report::from_messages(
                    errors,
                    "repair the document, or regenerate it from the registry with --write",
                ),
            };
            report.note(format!(
                "{} live operation contract(s) in the validated document",
                candidate.operation_count
            ));
            return Ok(report);
        }
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates(DEFAULT_OUTPUT, &schema);
        let mut report = xtask::artifact_gate::settle_inspection(ctx, self.name(), inspection);
        report.note(format!(
            "{} live operation contract(s)",
            schema.operation_count
        ));
        Ok(report)
    }
}

fn read_schema(path: &Path) -> Result<OperationSchema, String> {
    let text =
        read_text_bounded(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))
}

pub(super) fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_SCHEMA_BYTES, "operation schema")
}

#[cfg(test)]
mod tests {
    use super::schema::SCHEMA_VERSION;

    /// The op-schema wire version is spelled in two languages: this crate
    /// generates the file, and `scripts/architecture_docs.py` re-checks it.
    /// They drifted once, generator on 3 against a script still demanding 2,
    /// and nothing went red. This fails the moment they disagree again.
    #[test]
    fn the_python_contract_pins_the_same_operation_schema_version() {
        let script = std::fs::read_to_string(
            xtask::checkout::checkout_root().join("scripts/architecture_docs.py"),
        )
        .expect("Fix: scripts/architecture_docs.py must be readable");

        let expected = format!("OPERATION_SCHEMA_VERSION = {SCHEMA_VERSION}");
        assert!(
            script.contains(&expected),
            "scripts/architecture_docs.py must declare `{expected}`; \
             bump it in the same change as SCHEMA_VERSION"
        );
        assert!(
            script.contains("!= OPERATION_SCHEMA_VERSION"),
            "scripts/architecture_docs.py must compare against \
             OPERATION_SCHEMA_VERSION, not a second literal"
        );
    }
}
