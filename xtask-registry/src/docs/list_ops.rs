//! `cargo xtask list-ops` renders the canonical live operation schema as Markdown.

use xtask::gate::{GateCtx, GateError, Report};
use xtask::toml_text::{array, quote};

use crate::docs::operation_schema::assemble;
use crate::docs::operation_schema::schema::OperationRecord;

/// Holds the schema-derived operation inventory to the live registry.
pub struct ListOps;

impl xtask::gate::GateBehavior for ListOps {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let schema = assemble::build().map_err(|errors| {
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
        let mut report = xtask::artifact_gate::settle_inspection(ctx, ctx.gate_name()?, inspection);
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
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn list_ops_renders_canonical_toml_schema() {
        let mut backend_support = BTreeMap::new();
        backend_support.insert(
            "cuda".to_string(),
            crate::docs::operation_schema::schema::BackendSupport {
                status: "supported".to_string(),
                test_paths: vec!["conform/cuda.rs".to_string()],
            },
        );
        let record = OperationRecord {
            id: "vyre-primitives::math::add".to_string(),
            tier: "T2.5".to_string(),
            category: "math".to_string(),
            signature: crate::docs::operation_schema::schema::OperationSignature {
                kind: "kernel".to_string(),
                buffers: vec![crate::docs::operation_schema::schema::BufferSignature {
                    binding: 0,
                    name: "a".to_string(),
                    access: "read".to_string(),
                    memory: "global".to_string(),
                    element: "f32".to_string(),
                    count: 1,
                    pipeline_live_out: false,
                }],
                inputs: Vec::new(),
                outputs: Vec::new(),
                attributes: Vec::new(),
                bytes_extraction: true,
            },
            features: vec!["f32".to_string()],
            oracle: crate::docs::operation_schema::schema::OracleContract {
                reference_eval: true,
                flat_reference_facet: true,
                fixture_inputs: true,
                expected_output: true,
                tolerance_ulp: 0,
            },
            backend_support,
            target_facets: vec!["simd".to_string()],
            laws: vec!["commutative".to_string()],
            composition_chain: Vec::new(),
        };

        let rendered = render(&[record]);
        assert!(rendered.contains("schema_version = 1"));
        assert!(rendered.contains("[[operation]]"));
        assert!(rendered.contains("id = \"vyre-primitives::math::add\""));
        assert!(rendered.contains("tier = \"T2.5\""));
        assert!(rendered.contains("category = \"math\""));
        assert!(rendered.contains("bytes_extraction = true"));
        assert!(rendered.contains("features = [\"f32\"]"));
        assert!(rendered.contains("backends = [\"cuda\"]"));
    }
}
