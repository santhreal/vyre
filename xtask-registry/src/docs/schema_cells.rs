//! Markdown cell renderers shared by the schema-derived operation views.
//!
//! The catalog and the op inventory publish the same operation record in two
//! different table shapes. The column set differs, but a column that appears in
//! both renders identically, so each cell has one renderer here and the views
//! only choose which cells to emit and in which order.

use crate::docs::operation_schema::schema::{OperationRecord, TypedParameter};

/// The signature column: buffer bindings for program operations, an arrow form
/// for value operations.
pub(crate) fn signature_cell(row: &OperationRecord) -> String {
    if row.signature.kind == "program_buffers" {
        return row
            .signature
            .buffers
            .iter()
            .map(|buffer| {
                format!(
                    "{}:{}:{}:{}",
                    buffer.binding, buffer.name, buffer.access, buffer.element
                )
            })
            .collect::<Vec<_>>()
            .join("<br>");
    }
    let inputs = parameter_list(&row.signature.inputs);
    let outputs = parameter_list(&row.signature.outputs);
    format!("({inputs}) -> ({outputs})")
}

fn parameter_list(parameters: &[TypedParameter]) -> String {
    parameters
        .iter()
        .map(|parameter| format!("{}:{}", parameter.name, parameter.data_type))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The features column, one backtick-quoted feature per line.
pub(crate) fn features_cell(row: &OperationRecord) -> String {
    row.features
        .iter()
        .map(|feature| format!("`{feature}`"))
        .collect::<Vec<_>>()
        .join("<br>")
}

/// The declared backend statuses, one `backend:status` entry per backend. The
/// inventory view appends target facets to this list, so the entries are
/// returned unjoined.
pub(crate) fn backend_support_entries(row: &OperationRecord) -> Vec<String> {
    row.backend_support
        .iter()
        .map(|(backend, support)| format!("{backend}:{}", support.status))
        .collect()
}

/// The laws column, or the absence marker when no law covers the operation.
pub(crate) fn laws_cell(row: &OperationRecord) -> String {
    if row.laws.is_empty() {
        return "none declared".to_string();
    }
    row.laws.join("<br>")
}

/// The composition column: the indented chain, or the leaf marker.
pub(crate) fn composition_cell(row: &OperationRecord) -> String {
    if row.composition_chain.is_empty() {
        return "leaf".to_string();
    }
    row.composition_chain
        .iter()
        .map(|step| {
            format!(
                "{}{}{}",
                "&nbsp;".repeat(step.depth * 2),
                step.operation,
                if step.registered { "" } else { " (internal)" }
            )
        })
        .collect::<Vec<_>>()
        .join("<br>")
}
