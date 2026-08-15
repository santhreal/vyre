//! `cargo xtask list-ops` renders the canonical live operation schema as Markdown.

use xtask::gate::{Gate, GateCtx, GateError, Report};
use xtask::toml_text::{array, quote};

use crate::docs::operation_schema::{self, OperationRecord};

/// Holds the schema-derived operation inventory to the live registry.
pub struct ListOps;

impl Gate for ListOps {
    fn name(&self) -> &'static str {
        "list-ops"
    }

    fn help(&self) -> &'static str {
        "Hold docs/generated/op-inventory.toml to the live operation registry; --write regenerates it"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let schema = operation_schema::build().map_err(|errors| {
            GateError::new(
                format!(
                    "the canonical operation schema does not build: {}",
                    errors.join("; ")
                ),
                "repair the registrations the schema rejects, then run the gate again",
            )
        })?;
        let body = render(&schema.operations);
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates_text(INVENTORY_PATH, body);
        let mut report = xtask::artifact_gate::settle_inspection(ctx, self.name(), inspection);
        report.note(format!(
            "{} operation(s) in the live inventory",
            schema.operations.len()
        ));
        Ok(report)
    }
}

/// Repository-relative document this gate owns.
const INVENTORY_PATH: &str = "docs/generated/op-inventory.toml";

fn render(operations: &[OperationRecord]) -> String {
    let mut text = String::from(
        "# Generated from the live operation registry by `cargo xtask list-ops --write`.\n\
         # Edit the registrations, not this file.\n\
         schema_version = 1\n",
    );
    for operation in operations {
        text.push_str("\n[[operation]]\n");
        text.push_str(&format!("id = {}\n", quote(&operation.id)));
        text.push_str(&format!("tier = {}\n", quote(&operation.tier)));
        text.push_str(&format!("category = {}\n", quote(&operation.category)));
        text.push_str(&format!("kind = {}\n", quote(&operation.signature.kind)));
        text.push_str(&format!(
            "buffers = {}\n",
            operation.signature.buffers.len()
        ));
        text.push_str(&format!(
            "bytes_extraction = {}\n",
            operation.signature.bytes_extraction
        ));
        text.push_str(&format!(
            "features = {}\n",
            array(operation.features.iter().map(String::as_str))
        ));
        text.push_str(&format!(
            "target_facets = {}\n",
            array(operation.target_facets.iter().map(String::as_str))
        ));
        text.push_str(&format!(
            "laws = {}\n",
            array(operation.laws.iter().map(String::as_str))
        ));
        text.push_str(&format!(
            "backends = {}\n",
            array(operation.backend_support.keys().map(String::as_str))
        ));
    }
    text
}

