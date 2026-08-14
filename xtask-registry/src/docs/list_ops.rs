//! `cargo xtask list-ops` renders the canonical live operation schema as Markdown.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use crate::docs::operation_schema::{self, OperationRecord};
use crate::docs::schema_cells;

pub(crate) fn run(args: &[String]) {
    let (out_path, check) = parse_args(args);
    let schema = operation_schema::build().unwrap_or_else(|errors| {
        for error in errors {
            eprintln!("Fix: {error}");
        }
        std::process::exit(1);
    });
    let body = build_markdown(&schema.operations);
    if check {
        let path = out_path.unwrap_or_else(default_inventory_path);
        match fs::read_to_string(&path) {
            Ok(current) if current == body => {
                println!("list-ops: schema-derived inventory agrees");
            }
            Ok(_) => {
                eprintln!(
                    "Fix: {} differs from the canonical operation schema",
                    path.display()
                );
                std::process::exit(1);
            }
            Err(error) => {
                eprintln!(
                    "Fix: read {} before list-ops check: {error}",
                    path.display()
                );
                std::process::exit(1);
            }
        }
    } else if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|error| {
                eprintln!("Fix: create {}: {error}", parent.display());
                std::process::exit(1);
            });
        }
        fs::write(&path, &body).unwrap_or_else(|error| {
            eprintln!("Fix: write {}: {error}", path.display());
            std::process::exit(1);
        });
        eprintln!(
            "Wrote {} from the canonical operation schema.",
            path.display()
        );
    } else {
        print!("{body}");
    }
}

fn parse_args(args: &[String]) -> (Option<PathBuf>, bool) {
    let mut path = None;
    let mut check = false;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--write" => {
                index += 1;
                path = Some(PathBuf::from(args.get(index).unwrap_or_else(|| {
                    eprintln!("Fix: --write needs a path");
                    std::process::exit(2);
                })));
            }
            "--check" => check = true,
            other => {
                eprintln!("Fix: unknown list-ops argument `{other}`");
                std::process::exit(2);
            }
        }
        index += 1;
    }
    (path, check)
}

fn default_inventory_path() -> PathBuf {
    xtask::checkout::checkout_root().join("docs/generated/OP_INVENTORY.md")
}

fn build_markdown(operations: &[OperationRecord]) -> String {
    use std::fmt::Write;

    let mut by_tier: BTreeMap<&str, Vec<&OperationRecord>> = BTreeMap::new();
    for operation in operations {
        by_tier
            .entry(operation.tier.as_str())
            .or_default()
            .push(operation);
    }

    let mut output = String::new();
    let _ = writeln!(
        output,
        "# Vyre operation inventory\n\n\
         This file is generated from `docs/generated/OP_SCHEMA.json` by \
         `cargo_full run --bin xtask -- list-ops --write docs/generated/OP_INVENTORY.md`.\n\
         The JSON schema is the authority. This page is a browsing view.\n"
    );
    let tier_count = by_tier.len();
    for (index, (tier, rows)) in by_tier.into_iter().enumerate() {
        let _ = writeln!(output, "## `{tier}` ({} operations)\n", rows.len());
        output.push_str("| operation | category | signature | features | oracle | backends | laws | composition |\n");
        output.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
        for row in rows {
            let oracle = format!(
                "reference={} flat-facet={} inputs={} expected={} tolerance={} ULP",
                row.oracle.reference_eval,
                row.oracle.flat_reference_facet,
                row.oracle.fixture_inputs,
                row.oracle.expected_output,
                row.oracle.tolerance_ulp
            );
            let mut backends = schema_cells::backend_support_entries(row);
            backends.extend(
                row.target_facets
                    .iter()
                    .map(|target| format!("target:{target}")),
            );
            let _ = writeln!(
                output,
                "| `{}` | `{}` | {} | {} | {} | {} | {} | {} |",
                row.id,
                row.category,
                schema_cells::signature_cell(row),
                schema_cells::features_cell(row),
                oracle,
                backends.join("<br>"),
                schema_cells::laws_cell(row),
                schema_cells::composition_cell(row)
            );
        }
        if index + 1 < tier_count {
            output.push('\n');
        }
    }
    output
}
