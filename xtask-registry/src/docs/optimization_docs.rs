//! Generate the source-owned optimizer pass reference.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use xtask::gate::{GateCtx, GateError, Report};
use xtask::toml_text::{array, quote};

use vyre_foundation::optimizer::pass_catalog::{
    optimization_catalog, OptimizationCatalogEntryKind,
};
use vyre_foundation::optimizer::rewrite_contract::contract_for_pass;
use vyre_foundation::optimizer::{registered_pass_registrations, PassMetadata};

/// Holds the optimizer pass reference to the passes the source declares.
pub struct OptimizationDocs;

impl xtask::gate::GateBehavior for OptimizationDocs {
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
            ctx.gate_name()?,
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
         # An executable pass row also carries its declared rewrite contract: the IR \
         level it\n\
         # owns, the evidence authorizing it, and the growth it may cause. A \
         supplemental rule\n\
         # row carries none, because the contract belongs to the pass that runs it.\n\
         schema_version = 2\n",
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
        // A supplemental rule fires inside its owning pass, so the contract keys
        // belong to that pass and are absent here rather than duplicated.
        let contract = match entry.kind {
            OptimizationCatalogEntryKind::ExecutablePass => {
                Some(contract_for_pass(entry.name).ok_or_else(|| {
                    format!(
                        "executable pass `{}` declares no rewrite contract. Fix: record one in \
                         vyre-foundation optimizer::rewrite_contract::shipped, or submit a \
                         RewriteContractRegistration from the crate that registers the pass.",
                        entry.name
                    )
                })?)
            }
            OptimizationCatalogEntryKind::SupplementalRule => None,
        };
        let contract_keys = contract.map_or_else(String::new, |contract| {
            format!(
                "level = {}\n\
                 witness = {}\n\
                 witness_argument = {}\n\
                 obligation_families = {}\n\
                 expansion = {}\n",
                quote(contract.level.name()),
                quote(contract.witness.kind()),
                quote(contract.witness.argument()),
                array(contract.witness.obligation_families().iter().copied()),
                quote(&contract.expansion.to_string()),
            )
        });
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
             benchmark = {}\n\
             {contract_keys}",
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

    /// One rendered row, id line through the row's last key.
    fn row_for<'a>(document: &'a str, id: &str) -> &'a str {
        let start = document
            .find(&format!("\n[[pass]]\nid = \"{id}\"\n"))
            .expect("the rendered document must carry that row");
        let tail = &document[start + 1..];
        tail.find("\n[[pass]]\n")
            .map_or(tail, |offset| &tail[..offset])
    }

    /// WHY: a contract nothing publishes is a comment. This proves every executable
    /// pass row carries its declared level, evidence, and expansion bound, and that a
    /// supplemental rule row carries none of them, because the contract belongs to the
    /// pass that runs the rule. It does not prove the declared facts are the right ones.
    #[test]
    fn executable_rows_publish_their_rewrite_contract() {
        const CONTRACT_KEYS: [&str; 5] = [
            "level = ",
            "witness = ",
            "witness_argument = ",
            "obligation_families = ",
            "expansion = ",
        ];
        let catalog = optimization_catalog().expect("live optimizer catalog must build");
        let document = build().expect("optimizer reference must render");
        for entry in &catalog {
            let row = row_for(&document, entry.name);
            match entry.kind {
                OptimizationCatalogEntryKind::ExecutablePass => {
                    let contract = contract_for_pass(entry.name)
                        .expect("every executable pass declares a contract");
                    for key in CONTRACT_KEYS {
                        assert!(
                            row.contains(key),
                            "row `{}` omits `{key}`: {row}",
                            entry.name
                        );
                    }
                    assert!(
                        row.contains(&format!("level = \"{}\"", contract.level.name())),
                        "row `{}` must state the declared level: {row}",
                        entry.name
                    );
                    assert!(
                        row.contains(&format!("expansion = \"{}\"", contract.expansion)),
                        "row `{}` must state the declared expansion bound: {row}",
                        entry.name
                    );
                }
                OptimizationCatalogEntryKind::SupplementalRule => {
                    for key in CONTRACT_KEYS {
                        assert!(
                            !row.contains(key),
                            "supplemental row `{}` must not restate `{key}`: {row}",
                            entry.name
                        );
                    }
                }
            }
        }
    }

    /// WHY: the projection is the artifact a reviewer reads to see which rewrites the
    /// compiler may choose on its own, so the witness kind must be the recorded one
    /// rather than a rendered constant.
    #[test]
    fn a_row_states_the_recorded_witness_kind() {
        let document = build().expect("optimizer reference must render");
        let registrations = registered_pass_registrations().expect("pass order must derive");
        for registration in registrations.iter() {
            let name = registration.metadata.name;
            let contract = contract_for_pass(name).expect("every registered pass has a contract");
            let row = row_for(&document, name);
            assert!(
                row.contains(&format!("witness = \"{}\"", contract.witness.kind())),
                "row `{name}` must state its recorded witness kind: {row}"
            );
            assert!(
                row.contains(&format!(
                    "witness_argument = \"{}\"",
                    contract.witness.argument()
                )),
                "row `{name}` must state the recorded argument: {row}"
            );
        }
    }
}
