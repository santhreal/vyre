//! Where a contract is written, and where it is not.
//!
//! Two rules live here, both about the boundary between source and document.
//! Claim drift is how documentation lies: each row in the claim manifest pins a
//! phrase in a document to a test path, so a document edit rides alongside the
//! test edit and removing a claim removes both. And a comment that says the rule
//! lives in a document is not a statement of the rule: it costs a reader a second
//! file and it outlives the file it names, which is what happened when the book
//! those comments pointed into was deleted and every pointer became a pointer to
//! nothing with no gate red to show for it.

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The manifest of documented claims.
const MANIFEST: &str = "contracts/doc_claims_manifest.toml";

/// Documented claims resolve to a phrase in a document and a test that runs.
pub struct DocClaims;

impl crate::gate::GateBehavior for DocClaims {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let manifest = tree.read_toml(MANIFEST)?;
        let mut report = Report::clean();

        let claims = manifest
            .get("claim")
            .and_then(toml::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        report.cover_complete("documented claims", claims.len());
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

/// The published documentation directory, and the extension its pages carry.
const DOCUMENT_DIRECTORY: &str = "docs/";
const DOCUMENT_SUFFIX: &str = ".md";

/// A source comment states its contract instead of naming a document.
pub struct ContractInSource;

impl crate::gate::GateBehavior for ContractInSource {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let sources = tree.all_rust();
        let mut report = Report::clean();
        report.cover_complete("source contract files", sources.len());
        let mut scanned = 0_usize;
        for path in &sources {
            let text = tree.read(path)?;
            scanned += 1;
            for (index, line) in text.lines().enumerate() {
                if !is_prose(line) {
                    continue;
                }
                for document in documents_named_in(line) {
                    report.find(Finding::at(
                        path.clone(),
                        u32::try_from(index + 1).unwrap_or(u32::MAX),
                        format!("this comment defers its contract to {document}"),
                        "state the rule in the source that has to follow it, then delete the \
                         pointer; a document can be deleted and the comment cannot tell",
                    ));
                }
            }
        }
        report.note(format!("{scanned} source file(s) scanned"));
        Ok(report)
    }
}

/// Whether a line is prose rather than code.
///
/// Comments only, deliberately: a path in code is an artifact the program reads
/// or writes, and a generator naming its own output owns that output rather than
/// deferring to it.
fn is_prose(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//") || trimmed.starts_with('*')
}

/// Every published document path one line names.
///
/// A match must start the path: a directory of the same name nested under
/// another one, such as the release evidence tree, is a different tree and
/// carries no contract.
fn documents_named_in(line: &str) -> Vec<&str> {
    let path_character = |character: char| {
        character.is_ascii_alphanumeric() || matches!(character, '.' | '/' | '-' | '_')
    };
    let bytes = line.as_bytes();
    let mut found = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = line[cursor..].find(DOCUMENT_DIRECTORY) {
        let start = cursor + offset;
        cursor = start + DOCUMENT_DIRECTORY.len();
        if start > 0 && path_character(char::from(bytes[start - 1])) {
            continue;
        }
        let end = line[start..]
            .find(|character| !path_character(character))
            .map_or(line.len(), |length| start + length);
        let candidate = &line[start..end];
        if candidate.ends_with(DOCUMENT_SUFFIX) {
            found.push(candidate);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the rule is about the published documentation tree, not about the
    /// word. A path nested under another directory names a different tree, and a
    /// path glued to a longer token is not a path at all. Reading either as a
    /// deferral would make the gate fire on the release evidence tree, and a gate
    /// with false positives gets switched off.
    #[test]
    fn only_a_published_document_path_is_a_deferral() {
        assert_eq!(
            documents_named_in("// the rule lives in docs/lego-block-rule.md"),
            vec!["docs/lego-block-rule.md"]
        );
        assert!(documents_named_in("// release/evidence/docs/notes.md").is_empty());
        assert!(documents_named_in("// see docs/generated/OP_SCHEMA.json").is_empty());
        assert_eq!(
            documents_named_in("//! docs/A.md and docs/B.md").len(),
            2,
            "two pointers on one line are two findings"
        );
    }

    /// WHY: a path in code is an artifact the program reads or writes, and the
    /// generators in this tree name their own output. Scanning code would report
    /// every one of them and the rule would be reverted rather than followed.
    #[test]
    fn a_path_in_code_is_not_a_deferral() {
        assert!(is_prose("    // docs/x.md"));
        assert!(is_prose("/// docs/x.md"));
        assert!(is_prose(" * docs/x.md"));
        assert!(!is_prose("    let path = \"docs/x.md\";"));
    }
}
