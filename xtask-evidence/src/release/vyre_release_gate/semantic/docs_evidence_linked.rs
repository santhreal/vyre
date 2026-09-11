//! `docs-evidence-linked`: the documentation authority its map claims is
//! published, and the README contract it cites proves what it claims.
//!
//! The map is the only artifact that states which documentation pages the
//! release is certified against, and until this check existed nothing read the
//! paths inside it. The generic markdown check asks whether the document carries
//! the words `Evidence sources:`, which an empty heading satisfies, so the map
//! went on requiring an optimization readme after that page left the
//! tree and the release gate reported a closed requirement.
//!
//! The README contract artifact was cited by the same requirement and read the
//! same way: existence and valid JSON. A contract reporting a missing release
//! token, an empty README or its own blockers satisfied that, so the tokens the
//! artifact exists to prove went unchecked.

use std::path::Path;

use super::super::checks::{check_readme_contract, first_json_evidence};
use super::super::gate_inputs::Requirement;
use super::super::paths::{read_text_bounded, resolve_manifest_path};

/// The evidence document that lists the documentation authority.
const MAP: &str = "evidence/docs/docs-evidence-map.md";

/// The contract artifact proving the published README carries the release
/// tokens, an example, and no blockers of its own.
const README_CONTRACT: &str = "vyre-readme-contracts.json";

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    check_authority_map(requirement, base_dir, failures);
    if let Some(contract) = first_json_evidence(requirement, base_dir, README_CONTRACT, failures) {
        check_readme_contract(&requirement.id, "vyre", &contract, failures);
    }
}

fn check_authority_map(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    if !requirement.evidence.iter().any(|evidence| evidence == MAP) {
        failures.push(format!(
            "requirement `{}` does not cite `{MAP}`, so the documentation authority it claims cannot be read",
            requirement.id
        ));
        return;
    }
    let path = resolve_manifest_path(base_dir, MAP);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read `{MAP}`: {error}",
                requirement.id
            ));
            return;
        }
    };
    let claimed = authority_paths(&text);
    if claimed.is_empty() {
        failures.push(format!(
            "requirement `{}` map `{MAP}` names no documentation authority, so it certifies nothing",
            requirement.id
        ));
        return;
    }
    let Some(root) = base_dir.parent() else {
        failures.push(format!(
            "requirement `{}` cannot resolve `{MAP}` claims: the manifest directory has no parent to read them against",
            requirement.id
        ));
        return;
    };
    for authority in claimed {
        if !root.join(&authority).exists() {
            failures.push(format!(
                "requirement `{}` map `{MAP}` requires documentation authority `{authority}`, which the checkout does not carry",
                requirement.id
            ));
        }
    }
}

/// Every repository path the map lists as documentation authority.
///
/// A list item whose whole body is one code span carrying a slash, which is how
/// the map states a path. The release-contract items below the list are
/// sentences, and reading one as a citation would report a rule as a missing
/// file.
fn authority_paths(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let item = line.trim().strip_prefix("- ")?;
            let span = item.trim().strip_prefix('`')?.strip_suffix('`')?;
            span.contains('/').then(|| span.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! `authority_paths` is crate-private and reads one document's list shape.
    //! An integration test reaches only the whole release gate, which cannot say
    //! whether a prose item was read as a path or a path was skipped.

    use std::fs;

    use super::super::super::gate_inputs::Requirement;
    use super::{authority_paths, check};

    /// A requirement citing the map and the README contract, both written under
    /// `base_dir` so the check reads real files.
    fn requirement() -> Requirement {
        Requirement {
            id: "docs-evidence-linked".to_string(),
            title: "docs".to_string(),
            status: "closed".to_string(),
            evidence: vec![
                "evidence/docs/docs-evidence-map.md".to_string(),
                "evidence/docs/vyre-readme-contracts.json".to_string(),
            ],
        }
    }

    /// Write a map citing one authority path that exists, so the map half of the
    /// check contributes no failure.
    fn write_satisfied_map(base_dir: &std::path::Path) {
        let docs = base_dir.join("evidence/docs");
        fs::create_dir_all(&docs).unwrap();
        fs::write(docs.join("docs-evidence-map.md"), "- `docs/DOCS.toml`\n").unwrap();
        let root = base_dir.parent().unwrap().join("docs");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("DOCS.toml"), "").unwrap();
    }

    /// WHY: the map mixes a path list with a rule list under the same bullet
    /// syntax. Reading a rule as a citation reports a sentence as a missing
    /// file; skipping a path leaves the claim unchecked, which is the defect
    /// this check closes.
    #[test]
    fn a_backticked_path_item_is_read_and_a_prose_item_is_not() {
        let text = "# Map\n\n- `docs/DOCS.toml`\n- `docs/optimization/BENCH_TARGETS.toml`\n\n- Every page must have one lifecycle classification.\n- `notapath`\n";
        assert_eq!(
            authority_paths(text),
            vec![
                "docs/DOCS.toml".to_string(),
                "docs/optimization/BENCH_TARGETS.toml".to_string()
            ]
        );
    }

    /// WHY: the contract artifact is cited by the requirement and was read for
    /// existence and valid JSON only. A contract that names a missing release
    /// token is the exact state it exists to reject, so the gate has to fail on
    /// it while the rest of the requirement is satisfied.
    #[test]
    fn a_readme_contract_missing_a_release_token_fails_the_requirement() {
        let dir = tempfile::tempdir().unwrap();
        let base_dir = dir.path().join("release");
        write_satisfied_map(&base_dir);
        fs::write(
            base_dir.join("evidence/docs/vyre-readme-contracts.json"),
            serde_json::json!({
                "exists": true,
                "source_bytes": 38_237,
                "missing_tokens": ["0.7.2"],
                "example_count": 6,
                "blockers": []
            })
            .to_string(),
        )
        .unwrap();
        let mut failures = Vec::new();

        check(&requirement(), &base_dir, &mut failures);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("README is missing required API/version tokens")),
            "Fix: the release gate must read the README contract it cites, not only parse it; failures={failures:?}"
        );
    }

    /// WHY: a check that fails on every input proves nothing. The satisfied
    /// contract has to pass, or the failure above is not evidence about tokens.
    #[test]
    fn a_satisfied_readme_contract_and_map_produce_no_failure() {
        let dir = tempfile::tempdir().unwrap();
        let base_dir = dir.path().join("release");
        write_satisfied_map(&base_dir);
        fs::write(
            base_dir.join("evidence/docs/vyre-readme-contracts.json"),
            serde_json::json!({
                "exists": true,
                "source_bytes": 38_237,
                "missing_tokens": [],
                "example_count": 6,
                "blockers": []
            })
            .to_string(),
        )
        .unwrap();
        let mut failures = Vec::new();

        check(&requirement(), &base_dir, &mut failures);

        assert_eq!(
            failures,
            Vec::<String>::new(),
            "Fix: a satisfied documentation requirement must produce no failure"
        );
    }

    /// WHY: an absent citation is how the artifact stops being read at all. The
    /// requirement has to fail rather than pass for want of the evidence.
    #[test]
    fn a_requirement_that_stops_citing_the_readme_contract_fails() {
        let dir = tempfile::tempdir().unwrap();
        let base_dir = dir.path().join("release");
        write_satisfied_map(&base_dir);
        let mut requirement = requirement();
        requirement.evidence.pop();
        let mut failures = Vec::new();

        check(&requirement, &base_dir, &mut failures);

        assert!(
            failures.iter().any(|failure| failure
                .contains("needs JSON evidence ending in `vyre-readme-contracts.json`")),
            "Fix: dropping the contract citation must fail the requirement; failures={failures:?}"
        );
    }
}
