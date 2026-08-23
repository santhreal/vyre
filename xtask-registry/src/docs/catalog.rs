//! `cargo xtask catalog` renders subsystem views of the canonical operation schema.

use std::collections::BTreeMap;

use xtask::gate::{GateCtx, GateError, Report};
use xtask::toml_text::quote;

use crate::docs::operation_schema::assemble;
use crate::docs::operation_schema::schema::OperationRecord;

/// Holds the per-subsystem operation catalog to the live inventory.
pub struct Catalog;

impl xtask::gate::GateBehavior for Catalog {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let catalog = collect()?;
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates_text(CATALOG_PATH, render(&catalog));
        let mut report = xtask::artifact_gate::settle_inspection(ctx, ctx.gate_name()?, inspection);
        report.note(format!(
            "{} subsystem(s) in the live inventory",
            catalog.len()
        ));
        Ok(report)
    }
}

/// Repository-relative document this gate owns.
const CATALOG_PATH: &str = "docs/generated/catalog.toml";

/// Renders the live inventory as the TOML document the tree is held to.
fn render(catalog: &BTreeMap<String, Vec<OperationRecord>>) -> String {
    let mut text = String::from(
        "# Generated from the live operation registry by `cargo xtask catalog --write`.\n\
         # Edit the registrations, not this file.\n\
         schema_version = 1\n",
    );
    for (subsystem, rows) in catalog {
        text.push_str("\n[[subsystem]]\n");
        text.push_str(&format!("id = {}\n", quote(subsystem)));
        text.push_str("operations = [\n");
        for row in rows {
            text.push_str(&format!("  {},\n", quote(&row.id)));
        }
        text.push_str("]\n");
    }
    text
}

fn collect() -> Result<BTreeMap<String, Vec<OperationRecord>>, GateError> {
    let schema = assemble::build().map_err(schema_error)?;
    let mut by_subsystem: BTreeMap<String, Vec<OperationRecord>> = BTreeMap::new();
    for operation in schema.operations {
        by_subsystem
            .entry(subsystem_for(&operation.id))
            .or_default()
            .push(operation);
    }
    for rows in by_subsystem.values_mut() {
        rows.sort_by(|left, right| left.id.cmp(&right.id));
    }
    Ok(by_subsystem)
}

fn subsystem_for(operation_id: &str) -> String {
    operation_id
        .split("::")
        .nth(1)
        .or_else(|| operation_id.split('.').next())
        .unwrap_or("runtime")
        .to_string()
}

/// Turns operation schema build errors into one gate error, because a schema
/// that does not build leaves the gate nothing to compare the tree against.
fn schema_error(errors: Vec<String>) -> GateError {
    GateError::new(
        format!(
            "the canonical operation schema does not build: {}",
            errors.join("; ")
        ),
        "repair the registrations the schema rejects, then run the gate again",
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_renders_subsystems_and_extracts_subsystem_names() {
        assert_eq!(subsystem_for("vyre::runtime::alloc"), "runtime");
        assert_eq!(subsystem_for("vyre::tensor::matmul"), "tensor");
        assert_eq!(subsystem_for("vyre.tensor.matmul"), "vyre");
        assert_eq!(subsystem_for("simple"), "simple");

        let mut catalog = BTreeMap::new();
        let mut ops = Vec::new();
        let record = OperationRecord {
            id: "vyre::tensor::matmul".to_string(),
            tier: "T3".to_string(),
            category: "tensor".to_string(),
            signature: crate::docs::operation_schema::schema::OperationSignature {
                kind: "kernel".to_string(),
                buffers: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                attributes: Vec::new(),
                bytes_extraction: false,
            },
            features: Vec::new(),
            schedule_constraints: Default::default(),
            oracle: crate::docs::operation_schema::schema::OracleContract {
                reference_eval: true,
                flat_reference_facet: true,
                fixture_inputs: true,
                expected_output: true,
                tolerance_ulp: 0,
            },
            backend_support: BTreeMap::new(),
            target_facets: Vec::new(),
            laws: Vec::new(),
            composition_chain: Vec::new(),
        };
        ops.push(record);
        catalog.insert("tensor".to_string(), ops);
        let rendered = render(&catalog);
        assert!(rendered.contains("schema_version = 1"));
        assert!(rendered.contains("[[subsystem]]"));
        assert!(rendered.contains("id = \"tensor\""));
        assert!(rendered.contains("\"vyre::tensor::matmul\""));
    }
}
