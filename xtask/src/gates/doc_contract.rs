//! Every documented claim has a proving test, and says so in the document.
//!
//! Claim drift is how documentation lies. Each row in the claim manifest pins a
//! phrase in a document to a test path, so a document edit rides alongside the
//! test edit and removing a claim removes both.

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The manifest of documented claims.
const MANIFEST: &str = "contracts/doc_claims_manifest.toml";

/// Documented claims resolve to a phrase in a document and a test that runs.
pub struct DocClaims;

impl Gate for DocClaims {
    fn name(&self) -> &'static str {
        "doc-claims"
    }

    fn help(&self) -> &'static str {
        "claims whose document, phrase or proving test is missing"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let manifest = tree.read_toml(MANIFEST)?;
        let mut report = Report::clean();

        let claims = manifest
            .get("claim")
            .and_then(toml::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if claims.is_empty() {
            report.find(Finding::in_file(
                MANIFEST,
                "the manifest declares no claims",
                "declare each documented claim with its phrase and its proving test; an \
                 empty manifest turns this gate into coverage that measures nothing",
            ));
            return Ok(report);
        }

        for (index, claim) in claims.iter().enumerate() {
            let field = |name: &str| claim.get(name).and_then(toml::Value::as_str).unwrap_or("");
            let id = field("id");
            let label = if id.is_empty() {
                format!("claim {}", index + 1)
            } else {
                id.to_string()
            };
            let missing: Vec<&str> = ["id", "doc", "phrase", "test"]
                .into_iter()
                .filter(|name| field(name).is_empty())
                .collect();
            if !missing.is_empty() {
                report.find(Finding::in_file(
                    MANIFEST,
                    format!("{label} is missing {}", missing.join(", ")),
                    "a claim names its document, the phrase that states it, and the test \
                     that proves it; a row short of any of those pins nothing",
                ));
                continue;
            }

            let doc = field("doc");
            let phrase = field("phrase");
            let test = field("test");
            if !tree.absolute(doc).is_file() {
                report.find(Finding::in_file(
                    MANIFEST,
                    format!("{label} names a document that does not exist: {doc}"),
                    "restore the document, or delete the claim row and the test with it",
                ));
            } else if !tree.read(doc)?.contains(phrase) {
                report.find(Finding::in_file(
                    doc,
                    format!("{label} claims a phrase this document does not contain: {phrase}"),
                    "restore the phrase, or update the manifest row in the same change as \
                     the document edit that dropped it",
                ));
            }
            // A directory is acceptable: a claim may be proved by a suite rather
            // than one file.
            if !tree.absolute(test).exists() {
                report.find(Finding::in_file(
                    MANIFEST,
                    format!("{label} names a proving test that does not exist: {test}"),
                    "restore the test, or delete the claim from the manifest and the phrase \
                     from the document",
                ));
            }
        }

        report.note(format!("{} claim(s) checked", claims.len()));
        Ok(report)
    }
}
