//! `cargo xtask catalog` renders subsystem views of the canonical operation schema.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::operation_schema::{self, OperationRecord};

const MAX_CATALOG_TEXT_BYTES: u64 = 4_194_304;

pub(crate) fn run(args: &[String]) {
    let mut out_dir = default_out_dir();
    let mut check = false;
    let mut arguments = args.iter().skip(2);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--out" => {
                out_dir = arguments.next().map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("Fix: `--out` needs a directory path");
                    std::process::exit(1);
                });
            }
            "--check" => check = true,
            other => {
                eprintln!("Fix: unknown flag `{other}` for catalog. See --help.");
                std::process::exit(1);
            }
        }
    }

    let catalog = collect();
    if check {
        check_against_disk(&catalog, &out_dir);
    } else {
        emit_to_disk(&catalog, &out_dir);
    }
}

fn default_out_dir() -> PathBuf {
    let mut starts = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    starts.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    for start in &starts {
        if let Some(root) = find_workspace_root(start) {
            return root.join("docs/catalog");
        }
    }
    eprintln!(
        "Fix: cannot find the Vyre repository root from {}. Run this command inside the checkout or pass `--out <dir>`.",
        starts
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(" or ")
    );
    std::process::exit(1);
}

fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    for candidate in start.ancestors() {
        let manifest = candidate.join("Cargo.toml");
        if candidate.join("docs").is_dir()
            && manifest.is_file()
            && read_text_bounded(&manifest).is_ok_and(|text| text.contains("[workspace]"))
        {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

fn collect() -> BTreeMap<String, Vec<OperationRecord>> {
    let schema = operation_schema::build().unwrap_or_else(|errors| {
        for error in errors {
            eprintln!("Fix: {error}");
        }
        std::process::exit(1);
    });
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
    by_subsystem
}

fn subsystem_for(operation_id: &str) -> String {
    operation_id
        .split("::")
        .nth(1)
        .or_else(|| operation_id.split('.').next())
        .unwrap_or("runtime")
        .to_string()
}

fn render(subsystem: &str, rows: &[OperationRecord]) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    let _ = writeln!(output, "# `{subsystem}` operations\n");
    output.push_str(
        "This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.\n\n",
    );
    let _ = writeln!(
        output,
        "{} operations are registered in this subsystem.\n",
        rows.len()
    );
    output.push_str("| operation | tier | category | signature | features | oracle | backend support | laws | composition |\n");
    output.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in rows {
        let signature = if row.signature.kind == "program_buffers" {
            row.signature
                .buffers
                .iter()
                .map(|buffer| {
                    format!(
                        "{}:{}:{}:{}",
                        buffer.binding, buffer.name, buffer.access, buffer.element
                    )
                })
                .collect::<Vec<_>>()
                .join("<br>")
        } else {
            let inputs = row
                .signature
                .inputs
                .iter()
                .map(|parameter| format!("{}:{}", parameter.name, parameter.data_type))
                .collect::<Vec<_>>()
                .join(", ");
            let outputs = row
                .signature
                .outputs
                .iter()
                .map(|parameter| format!("{}:{}", parameter.name, parameter.data_type))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({inputs}) -> ({outputs})")
        };
        let oracle = format!(
            "reference={} inputs={} expected={} tolerance={} ULP",
            row.oracle.reference_eval,
            row.oracle.fixture_inputs,
            row.oracle.expected_output,
            row.oracle.tolerance_ulp
        );
        let backends = row
            .backend_support
            .iter()
            .map(|(backend, support)| format!("{backend}:{}", support.status))
            .collect::<Vec<_>>()
            .join("<br>");
        let composition = if row.composition_chain.is_empty() {
            "leaf".to_string()
        } else {
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
        };
        let _ = writeln!(
            output,
            "| `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} |",
            row.id,
            row.tier,
            row.category,
            signature,
            row.features
                .iter()
                .map(|feature| format!("`{feature}`"))
                .collect::<Vec<_>>()
                .join("<br>"),
            oracle,
            backends,
            if row.laws.is_empty() {
                "none declared".to_string()
            } else {
                row.laws.join("<br>")
            },
            composition
        );
    }
    output
}

fn emit_to_disk(catalog: &BTreeMap<String, Vec<OperationRecord>>, out_dir: &Path) {
    fs::create_dir_all(out_dir).unwrap_or_else(|error| {
        eprintln!("Fix: create {}: {error}", out_dir.display());
        std::process::exit(1);
    });
    for (subsystem, rows) in catalog {
        let path = out_dir.join(format!("{subsystem}.md"));
        fs::write(&path, render(subsystem, rows)).unwrap_or_else(|error| {
            eprintln!("Fix: write {}: {error}", path.display());
            std::process::exit(1);
        });
    }
    fs::write(out_dir.join("README.md"), render_index(catalog)).unwrap_or_else(|error| {
        eprintln!("Fix: write catalog index: {error}");
        std::process::exit(1);
    });
    for orphan in stale_catalog_files(catalog, out_dir) {
        fs::remove_file(&orphan).unwrap_or_else(|error| {
            eprintln!("Fix: remove stale {}: {error}", orphan.display());
            std::process::exit(1);
        });
    }
    println!("wrote {} schema-derived subsystem catalogs", catalog.len());
}

fn check_against_disk(catalog: &BTreeMap<String, Vec<OperationRecord>>, out_dir: &Path) {
    let mut drift = Vec::new();
    for (subsystem, rows) in catalog {
        let path = out_dir.join(format!("{subsystem}.md"));
        match read_text_bounded(&path) {
            Ok(current) if current == render(subsystem, rows) => {}
            Ok(_) => drift.push(format!(
                "{} differs from the canonical schema",
                path.display()
            )),
            Err(error) => drift.push(format!("{} cannot be read: {error}", path.display())),
        }
    }
    let index_path = out_dir.join("README.md");
    match read_text_bounded(&index_path) {
        Ok(current) if current == render_index(catalog) => {}
        Ok(_) => drift.push(format!(
            "{} differs from the canonical schema",
            index_path.display()
        )),
        Err(error) => drift.push(format!("{} cannot be read: {error}", index_path.display())),
    }
    for orphan in stale_catalog_files(catalog, out_dir) {
        drift.push(format!("{} has no canonical subsystem", orphan.display()));
    }
    if !drift.is_empty() {
        for error in drift {
            eprintln!("Fix: {error}");
        }
        std::process::exit(1);
    }
    println!(
        "catalog: {} schema-derived subsystem catalogs agree",
        catalog.len()
    );
}

fn stale_catalog_files(
    catalog: &BTreeMap<String, Vec<OperationRecord>>,
    out_dir: &Path,
) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(out_dir) else {
        return Vec::new();
    };
    let mut stale = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "md"))
        .filter(
            |path| match path.file_stem().and_then(|stem| stem.to_str()) {
                Some("README") => false,
                Some(stem) => !catalog.contains_key(stem),
                None => false,
            },
        )
        .collect::<Vec<_>>();
    stale.sort();
    stale
}

fn render_index(catalog: &BTreeMap<String, Vec<OperationRecord>>) -> String {
    let mut output = String::from("# Vyre operation catalog\n\n");
    output.push_str(
        "These pages are generated browsing views of `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority for operation IDs, tiers, categories, signatures, features, oracles, backend support, laws, composition chains, and counts.\n\n",
    );
    output.push_str("| subsystem | operations |\n| --- | ---: |\n");
    for (subsystem, rows) in catalog {
        output.push_str(&format!(
            "| [`{subsystem}`]({subsystem}.md) | {} |\n",
            rows.len()
        ));
    }
    output
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_CATALOG_TEXT_BYTES + 1);
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_CATALOG_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} exceeds the catalog read cap", path.display()),
        ));
    }
    Ok(text)
}
