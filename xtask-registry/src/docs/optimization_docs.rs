//! Generate the source-owned optimizer pass reference.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use vyre_foundation::optimizer::pass_catalog::{
    optimization_catalog, OptimizationCatalogEntryKind,
};
use vyre_foundation::optimizer::{registered_pass_registrations, PassMetadata};

pub(crate) fn run(args: &[String]) {
    let (path, check) = parse_args(args);
    let body = build().unwrap_or_else(|error| {
        eprintln!("Fix: build optimization pass reference: {error}");
        std::process::exit(1);
    });
    let path = path.unwrap_or_else(default_path);
    if check {
        match fs::read_to_string(&path) {
            Ok(current) if current == body => {
                println!("optimization-docs: source registry agrees");
            }
            Ok(_) => {
                eprintln!(
                    "Fix: {} differs from the source optimizer registry; regenerate it",
                    path.display()
                );
                std::process::exit(1);
            }
            Err(error) => {
                eprintln!("Fix: read {}: {error}", path.display());
                std::process::exit(1);
            }
        }
    } else {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|error| {
                eprintln!("Fix: create {}: {error}", parent.display());
                std::process::exit(1);
            });
        }
        fs::write(&path, body).unwrap_or_else(|error| {
            eprintln!("Fix: write {}: {error}", path.display());
            std::process::exit(1);
        });
        println!("optimization-docs: wrote {}", path.display());
    }
}

fn parse_args(args: &[String]) -> (Option<PathBuf>, bool) {
    let mut path = None;
    let mut check = false;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                index += 1;
                path = Some(PathBuf::from(args.get(index).unwrap_or_else(|| {
                    eprintln!("Fix: --output needs a path");
                    std::process::exit(2);
                })));
            }
            "--check" => check = true,
            other => {
                eprintln!("Fix: unknown optimization-docs argument `{other}`");
                std::process::exit(2);
            }
        }
        index += 1;
    }
    (path, check)
}

fn default_path() -> PathBuf {
    xtask::checkout::checkout_root().join("docs/optimization/PASSES.md")
}

fn join(values: &[&str]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join("<br>")
    }
}

fn metadata_by_name() -> Result<BTreeMap<&'static str, PassMetadata>, String> {
    let registrations = registered_pass_registrations().map_err(|error| error.to_string())?;
    Ok(registrations
        .iter()
        .map(|registration| (registration.metadata.name, registration.metadata))
        .collect())
}

fn build() -> Result<String, String> {
    let metadata = metadata_by_name()?;
    let catalog = optimization_catalog().map_err(|error| error.to_string())?;
    let mut output = String::new();
    output.push_str(
        "# Optimizer pass reference\n\n\
         This page is generated from the live `vyre-foundation` optimizer registry by \
         `cargo_full run --bin xtask -- optimization-docs`. Edit pass registration \
         metadata, not this page.\n\n\
         The optimizer has one semantic layer before verified lowering. Concrete target \
         strategy is not registered in this catalog.\n\n",
    );
    output.push_str(
        "| id | kind | owner | phase | boundary | requires | invalidates | capabilities | ABI | invariant | termination | proof | benchmark |\n",
    );
    output.push_str(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for entry in catalog {
        let registered = metadata.get(entry.name);
        let kind = match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => "executable pass",
            OptimizationCatalogEntryKind::SupplementalRule => "supplemental rule",
        };
        let requires = registered.map_or_else(|| "none".to_string(), |row| join(row.requires));
        let invalidates =
            registered.map_or_else(|| "none".to_string(), |row| join(row.invalidates));
        let termination = match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => {
                "bounded by the scheduler restart and iteration budgets"
            }
            OptimizationCatalogEntryKind::SupplementalRule => {
                "bounded by its owning executable pass"
            }
        };
        let proof = match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => {
                "`optimizer::pass_invariants::audit_registered_passes`"
            }
            OptimizationCatalogEntryKind::SupplementalRule => {
                "owning pass differential and invariant fixtures"
            }
        };
        let _ = writeln!(
            output,
            "| `{}` | {} | `{}` | `{:?}` | `{:?}` | {} | {} | {} | `{}` | {} | {} | {} | `{}` |",
            entry.name,
            kind,
            entry.owner,
            entry.phase,
            entry.boundary_class,
            requires,
            invalidates,
            join(entry.requires_caps),
            entry.preserves_abi,
            entry.invariant,
            termination,
            proof,
            entry.benchmark,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a hand-maintained pass document previously omitted newly registered passes.
    /// This contract derives every executable row from the live inventory and every
    /// supplemental row from the source catalog. It does not prove pass semantics.
    #[test]
    fn generated_reference_covers_every_catalog_entry() {
        let catalog = optimization_catalog().expect("live optimizer catalog must build");
        let markdown = build().expect("optimizer reference must render");
        let row_count = markdown
            .lines()
            .filter(|line| line.starts_with("| `"))
            .count();
        assert_eq!(row_count, catalog.len());
        for entry in catalog {
            assert!(
                markdown.contains(&format!("| `{}` |", entry.name)),
                "missing optimizer catalog row {}",
                entry.name
            );
        }
    }

    /// WHY: dependency and invalidation metadata must remain visible when a pass moves.
    /// Supplemental rule rows intentionally inherit those fields from their owning pass.
    #[test]
    fn executable_rows_include_live_ordering_metadata() {
        let registrations = registered_pass_registrations().expect("pass order must derive");
        let markdown = build().expect("optimizer reference must render");
        for registration in registrations.iter() {
            let metadata = registration.metadata;
            assert!(markdown.contains(&format!("| `{}` | executable pass", metadata.name)));
            for requirement in metadata.requires {
                assert!(markdown.contains(requirement));
            }
            for invalidation in metadata.invalidates {
                assert!(markdown.contains(invalidation));
            }
        }
    }
}
