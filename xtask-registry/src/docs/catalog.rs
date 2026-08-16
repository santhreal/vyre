//! `cargo xtask catalog` renders subsystem views of the canonical operation schema.

use std::collections::BTreeMap;

use xtask::gate::{Gate, GateCtx, GateError, Report};
use xtask::toml_text::quote;

use crate::docs::operation_schema::assemble;
use crate::docs::operation_schema::schema::OperationRecord;

/// Holds the per-subsystem operation catalog to the live inventory.
pub struct Catalog;

impl Gate for Catalog {
    fn name(&self) -> &'static str {
        "catalog"
    }

    fn help(&self) -> &'static str {
        "Hold docs/generated/catalog.toml to the live operation inventory; --write regenerates it"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let catalog = collect()?;
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates_text(CATALOG_PATH, render(&catalog));
        let mut report = xtask::artifact_gate::settle_inspection(ctx, self.name(), inspection);
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
