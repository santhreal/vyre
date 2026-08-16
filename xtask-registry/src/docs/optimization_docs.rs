//! Generate the source-owned optimizer pass reference.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use xtask::gate::{Gate, GateCtx, GateError, Report};
use xtask::toml_text::{array, quote};

use vyre_foundation::optimizer::pass_catalog::{
    optimization_catalog, OptimizationCatalogEntryKind,
};
use vyre_foundation::optimizer::{registered_pass_registrations, PassMetadata};

/// Holds the optimizer pass reference to the passes the source declares.
pub struct OptimizationDocs;

impl Gate for OptimizationDocs {
    fn name(&self) -> &'static str {
        "optimization-docs"
    }

    fn help(&self) -> &'static str {
        "Hold docs/generated/optimizer-passes.toml to the passes the source declares; --write regenerates it"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let body = build().map_err(|error| {
            GateError::new(
                format!("the optimizer pass reference does not build: {error}"),
                "repair the pass catalog the builder rejects, then run the gate again",
            )
        })?;
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates_text(PASSES_PATH, body);
        Ok(xtask::artifact_gate::settle_inspection(
            ctx,
            self.name(),
            inspection,
        ))
    }
}

/// Repository-relative document this gate owns.
const PASSES_PATH: &str = "docs/generated/optimizer-passes.toml";

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
        "# Generated from the live vyre-foundation optimizer registry by \
         `./cargo_full run --bin xtask -- optimization-docs --write`.\n\
         # Edit the pass registration metadata, not this file. The optimizer has one \
         semantic layer\n\
         # before verified lowering; concrete target strategy is not registered in this \
         catalog.\n\
         schema_version = 1\n",
    );
    for entry in catalog {
        let registered = metadata.get(entry.name);
        let kind = match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => "executable pass",
            OptimizationCatalogEntryKind::SupplementalRule => "supplemental rule",
        };
        let requires =
            registered.map_or_else(|| array([]), |row| array(row.requires.iter().copied()));
        let invalidates =
            registered.map_or_else(|| array([]), |row| array(row.invalidates.iter().copied()));
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
        let _ = write!(
            output,
            "\n[[pass]]\n\
             id = {}\n\
             kind = {}\n\
             owner = {}\n\
             phase = {}\n\
             boundary_class = {}\n\
             requires = {requires}\n\
             invalidates = {invalidates}\n\
             requires_capabilities = {}\n\
             preserves_abi = {}\n\
             invariant = {}\n\
             termination = {}\n\
             proof = {}\n\
             benchmark = {}\n",
            quote(entry.name),
            quote(kind),
            quote(entry.owner),
            quote(&format!("{:?}", entry.phase)),
            quote(&format!("{:?}", entry.boundary_class)),
            array(entry.requires_caps.iter().copied()),
            quote(&format!("{}", entry.preserves_abi)),
            quote(entry.invariant),
            quote(termination),
            quote(proof),
            quote(entry.benchmark),
        );
    }
    Ok(output)
}

/// `build` renders the document without touching the filesystem and is private
/// to this gate, so no integration test can render one to check.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::optimizer::pass_catalog::OptimizationCatalogEntry;

    /// Every catalog id the rendered document does not carry a row for.
    fn missing_ids<'a>(document: &str, catalog: &'a [OptimizationCatalogEntry]) -> Vec<&'a str> {
        catalog
            .iter()
            .map(|entry| entry.name)
            .filter(|name| !document.contains(&format!("id = \"{name}\"\n")))
            .collect()
    }

    /// WHY: a hand-maintained pass document previously omitted newly registered passes.
    /// This contract derives every executable row from the live inventory and every
    /// supplemental row from the source catalog. It does not prove pass semantics.
    #[test]
    fn generated_reference_covers_every_catalog_entry() {
        let catalog = optimization_catalog().expect("live optimizer catalog must build");
        let document = build().expect("optimizer reference must render");
        let row_count = document
            .lines()
            .filter(|line| line.trim() == "[[pass]]")
            .count();
        assert_eq!(row_count, catalog.len());
        assert_eq!(missing_ids(&document, &catalog), Vec::<&str>::new());
    }

    /// WHY: a coverage check that cannot go red is worse than none. This drops one
    /// rendered row and proves the check names exactly the id that left. The
    /// strength-reduction rules are the case that motivated it: they are IR rewrites
    /// with no op registration, so this document is the only artifact that names them.
    #[test]
    fn a_dropped_supplemental_rule_row_is_reported_missing() {
        let catalog = optimization_catalog().expect("live optimizer catalog must build");
        let document = build().expect("optimizer reference must render");
        let dropped = catalog
            .iter()
            .map(|entry| entry.name)
            .find(|name| name.starts_with("strength_reduce."))
            .expect("the catalog must name the strength reduction rules");
        let start = document
            .find(&format!("\n[[pass]]\nid = \"{dropped}\"\n"))
            .expect("the rendered document must carry that row");
        let tail = &document[start + 1..];
        let end = tail
            .find("\n[[pass]]\n")
            .map_or(document.len(), |offset| start + 1 + offset);
        let mutilated = format!("{}{}", &document[..start], &document[end..]);

        assert_eq!(missing_ids(&mutilated, &catalog), vec![dropped]);
    }

    /// WHY: dependency and invalidation metadata must remain visible when a pass moves.
    /// Supplemental rule rows intentionally inherit those fields from their owning pass.
    #[test]
    fn executable_rows_include_live_ordering_metadata() {
        let registrations = registered_pass_registrations().expect("pass order must derive");
        let document = build().expect("optimizer reference must render");
        for registration in registrations.iter() {
            let metadata = registration.metadata;
            assert!(document.contains(&format!(
                "id = \"{}\"\nkind = \"executable pass\"",
                metadata.name
            )));
            for requirement in metadata.requires {
                assert!(document.contains(requirement));
            }
            for invalidation in metadata.invalidates {
                assert!(document.contains(invalidation));
            }
        }
    }
}
