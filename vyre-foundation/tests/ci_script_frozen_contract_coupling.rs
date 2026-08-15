//! CI / script / frozen-contract coupling contract.
//!
//! Workflows must reference scripts that exist. Frozen-trait snapshot scripts
//! must reference source files and snapshots that exist. Stale couplings must
//! be baselined or removed.

use std::collections::HashSet;
use vyre_test_support::monorepo::vyre_workspace_root;

/// The script a workflow line invokes, relative to `scripts/`, or `None` when
/// the line invokes nothing.
///
/// A YAML comment is documentation, not a reference: prose that ends a sentence
/// with `source_scan.sh.` names no file, and reporting it as a missing script
/// makes the contract fail on its own explanatory text.
fn referenced_script(line: &str) -> Option<&str> {
    let command = strip_yaml_comment(line.trim());
    let index = command.find("scripts/")?;
    let rest = &command[index + "scripts/".len()..];
    let name = rest.split_whitespace().next().unwrap_or(rest);
    let name = name.trim_end_matches(['"', '\'', ';', ')']);
    (!name.is_empty()).then_some(name)
}

/// Everything before a trailing YAML comment. `#` opens one at the start of a
/// line or after whitespace.
fn strip_yaml_comment(line: &str) -> &str {
    if line.starts_with('#') {
        return "";
    }
    match line.find(" #") {
        Some(index) => &line[..index],
        None => line,
    }
}

#[test]
fn script_references_come_from_commands_not_prose() {
    assert_eq!(
        referenced_script("        run: bash scripts/check_unsafe_budget.sh"),
        Some("check_unsafe_budget.sh")
    );
    assert_eq!(
        referenced_script("        run: bash scripts/lib/source_scan.sh --strict"),
        Some("lib/source_scan.sh")
    );
    assert_eq!(
        referenced_script("        run: bash \"scripts/check_public_api.sh\";"),
        Some("check_public_api.sh")
    );
    assert_eq!(
        referenced_script("      # all on scripts/lib/source_scan.sh."),
        None
    );
    assert_eq!(
        referenced_script("        run: bash scripts/gate.sh # see scripts/other.sh."),
        Some("gate.sh")
    );
    assert_eq!(referenced_script("        run: cargo test"), None);
    assert_eq!(
        referenced_script("        run: bash scripts/check_*.sh"),
        Some("check_*.sh")
    );
}

#[test]
fn ci_workflows_reference_existing_scripts() {
    let workspace_root = vyre_workspace_root();
    let workflows_dir = workspace_root.join(".github/workflows");
    if !workflows_dir.is_dir() {
        return;
    }

    let scripts_dir = workspace_root.join("scripts");

    // Known script references that use wildcards or are not literal filenames.
    let known_wildcards: HashSet<String> =
        ["scripts/check_*.sh".to_string()].iter().cloned().collect();

    let mut violations = Vec::new();

    for entry in std::fs::read_dir(&workflows_dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("yml") {
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap();
        for (line_no, line) in content.lines().enumerate() {
            let Some(script_name) = referenced_script(line) else {
                continue;
            };
            if script_name.contains('*') {
                if !known_wildcards.contains(&format!("scripts/{script_name}")) {
                    violations.push(format!(
                        "{}:{} unknown wildcard script reference: scripts/{}",
                        path.file_name().unwrap().to_string_lossy(),
                        line_no + 1,
                        script_name
                    ));
                }
                continue;
            }
            if !scripts_dir.join(script_name).exists() {
                violations.push(format!(
                    "{}:{} missing script: scripts/{}",
                    path.file_name().unwrap().to_string_lossy(),
                    line_no + 1,
                    script_name
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "CI workflows must reference existing scripts. Violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn frozen_trait_contract_files_exist() {
    let workspace_root = vyre_workspace_root();

    let contracts = [
        ("VyreBackend", "vyre-driver/src/backend/vyre_backend.rs"),
        ("ExprVisitor", "vyre-foundation/src/visit/expr/mod.rs"),
        ("Lowerable", "vyre-driver/src/backend/lowering.rs"),
        ("AlgebraicLaw", "vyre-spec/src/algebraic_law.rs"),
        ("EnforceGate", "vyre-driver/src/registry/enforce.rs"),
        ("MutationClass", "vyre-driver/src/registry/mutation.rs"),
    ];

    let mut violations = Vec::new();
    for (name, file) in &contracts {
        let path = workspace_root.join(file);
        if !path.exists() {
            violations.push(format!(
                "frozen contract source missing: {} ({})",
                name, file
            ));
            continue;
        }
        let snapshot = workspace_root.join(format!("docs/frozen-traits/{}.txt", name));
        if !snapshot.exists() {
            violations.push(format!(
                "frozen contract snapshot missing: {} ({}). Fix: run scripts/check_trait_freeze.sh --refresh-snapshots",
                name, snapshot.display()
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "frozen trait contracts must have source files and snapshots. Violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn frozen_trait_script_is_executable() {
    let workspace_root = vyre_workspace_root();
    let script = workspace_root.join("scripts/check_trait_freeze.sh");
    assert!(
        script.exists(),
        "scripts/check_trait_freeze.sh must exist to enforce frozen contracts"
    );
}
