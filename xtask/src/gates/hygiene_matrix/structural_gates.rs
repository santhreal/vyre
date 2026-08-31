use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::records::{
    HygieneFinding, StructuralGateArtifact, StructuralGateDeclaration, StructuralGateDocument,
    STRUCTURAL_GATE_SCHEMA_VERSION, STRUCTURAL_GATE_SOURCE,
};
use super::rules::read_text_bounded;
use super::threshold_policy::relative_to_vyre;

/// Read the structural-gate registry, or report why it could not be trusted.
///
/// Every failure path returns an empty declaration set plus a blocker, so an
/// unreadable or malformed registry exempts nothing rather than everything.
pub(crate) fn load_structural_gates(vyre_root: &Path) -> StructuralGateArtifact {
    let mut artifact = StructuralGateArtifact {
        schema_version: STRUCTURAL_GATE_SCHEMA_VERSION,
        source: STRUCTURAL_GATE_SOURCE,
        declarations: Vec::new(),
        blockers: Vec::new(),
    };
    let path = vyre_root.join(STRUCTURAL_GATE_SOURCE);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE} is unreadable: {error}. Fix: restore the structural-gate registry; without it every source-inspecting gate is a release blocker."
            ));
            return artifact;
        }
    };
    let document = match toml::from_str::<StructuralGateDocument>(&text) {
        Ok(document) => document,
        Err(error) => {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE} is not valid structural-gate TOML: {error}. Fix: repair the schema before release."
            ));
            return artifact;
        }
    };
    if document.schema != STRUCTURAL_GATE_SCHEMA_VERSION {
        artifact.blockers.push(format!(
            "{STRUCTURAL_GATE_SOURCE} declares schema {} but this gate reads schema {STRUCTURAL_GATE_SCHEMA_VERSION}. Fix: migrate the registry before release.",
            document.schema
        ));
        return artifact;
    }
    let mut seen = BTreeSet::new();
    for row in document.gate {
        if row.file.trim().is_empty() || row.tests.is_empty() {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE} has a row with an empty file or test set. Fix: name both."
            ));
            continue;
        }
        if row.reason.trim().is_empty() {
            artifact.blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE}: the tests in `{}` declare no reason. Fix: state why the property has no run-time witness.",
                row.file
            ));
            continue;
        }
        for test in row.tests {
            if test.trim().is_empty() {
                artifact.blockers.push(format!(
                    "{STRUCTURAL_GATE_SOURCE} has an empty test name in `{}`. Fix: name every test.",
                    row.file
                ));
                continue;
            }
            if !seen.insert((row.file.clone(), test.clone())) {
                artifact.blockers.push(format!(
                    "{STRUCTURAL_GATE_SOURCE}: `{test}` in `{}` is declared twice. Fix: keep one entry.",
                    row.file
                ));
                continue;
            }
            artifact.declarations.push(StructuralGateDeclaration {
                file: row.file.clone(),
                test,
                reason: row.reason.trim().to_string(),
            });
        }
    }
    artifact
}

/// Blockers for registry rows the tree no longer backs.
///
/// Derived from the same scan that produced the findings, so the registry
/// cannot outlive the gates it exempts.
pub(crate) fn stale_declaration_blockers(
    vyre_root: &Path,
    declarations: &[StructuralGateDeclaration],
    findings: &[HygieneFinding],
) -> Vec<String> {
    let mut inspecting = BTreeMap::<String, BTreeSet<&str>>::new();
    for finding in findings {
        if finding.pattern != "source_inspection_test" {
            continue;
        }
        let Some(test) = finding.test.as_deref() else {
            continue;
        };
        inspecting
            .entry(relative_to_vyre(vyre_root, Path::new(&finding.path)))
            .or_default()
            .insert(test);
    }
    let mut blockers = Vec::new();
    for declaration in declarations {
        match inspecting.get(&declaration.file) {
            None => blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE}: `{}` names `{}`, which contains no source-inspecting test. Fix: delete the row; a registry that outlives its gate exempts nothing and hides the next one.",
                declaration.test, declaration.file
            )),
            Some(tests) if !tests.contains(declaration.test.as_str()) => blockers.push(format!(
                "{STRUCTURAL_GATE_SOURCE}: `{}` is declared for `{}`, which no longer has a test by that name that inspects source. Fix: delete the row or correct the name.",
                declaration.test, declaration.file
            )),
            Some(_) => {}
        }
    }
    blockers
}

/// Whether a source-inspecting `finding` is covered by a reviewed declaration.
pub(crate) fn is_declared_structural_gate(
    vyre_root: &Path,
    finding: &HygieneFinding,
    structural_gates: &StructuralGateArtifact,
) -> bool {
    let Some(test) = finding.test.as_deref() else {
        return false;
    };
    let file = relative_to_vyre(vyre_root, Path::new(&finding.path));
    structural_gates
        .declarations
        .iter()
        .any(|declaration| declaration.file == file && declaration.test == test)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_registry(root: &Path, rows: &str) {
        let parent = root.join("docs/testing");
        std::fs::create_dir_all(&parent)
            .expect("Fix: structural-gate fixture directory must be creatable.");
        std::fs::write(
            root.join(STRUCTURAL_GATE_SOURCE),
            format!("schema = {STRUCTURAL_GATE_SCHEMA_VERSION}\n{rows}"),
        )
        .expect("Fix: structural-gate fixture must be writable.");
    }

    #[test]
    fn one_registry_row_expands_every_named_test() {
        let root =
            tempfile::tempdir().expect("Fix: structural-gate fixture root must be creatable.");
        write_registry(
            root.path(),
            r#"
[[gate]]
file = "crate/tests/contracts.rs"
tests = ["first_contract", "second_contract"]
reason = "The source member set has no run-time witness."
"#,
        );

        let artifact = load_structural_gates(root.path());

        assert!(artifact.blockers.is_empty(), "{:?}", artifact.blockers);
        assert_eq!(artifact.declarations.len(), 2);
        assert_eq!(artifact.declarations[0].test, "first_contract");
        assert_eq!(artifact.declarations[1].test, "second_contract");
        assert_eq!(
            artifact.declarations[0].reason,
            artifact.declarations[1].reason
        );
    }

    #[test]
    fn registry_row_without_tests_exempts_nothing() {
        let root =
            tempfile::tempdir().expect("Fix: structural-gate fixture root must be creatable.");
        write_registry(
            root.path(),
            r#"
[[gate]]
file = "crate/tests/contracts.rs"
tests = []
reason = "The source member set has no run-time witness."
"#,
        );

        let artifact = load_structural_gates(root.path());

        assert!(artifact.declarations.is_empty());
        assert_eq!(artifact.blockers.len(), 1);
        assert!(artifact.blockers[0].contains("empty file or test set"));
    }
}
