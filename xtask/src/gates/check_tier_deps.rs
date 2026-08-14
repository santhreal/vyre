//! `cargo xtask check-tier-deps` - reject upward tier dependencies in workspace manifests.
//!
//! Tier order (low → high): T1 foundation/spec/core → T2 intrinsics → T2.5 primitives
//! → self-substrate → T3 libs → reference/emit/conform → T4 drivers/runtime.

use std::fs;
use std::path::Path;
use std::process::{self, Command};

use toml::Value;

use crate::manifest_walk::MAX_MANIFEST_BYTES;

/// Run the tier-dependency gate.
pub(crate) fn run(args: &[String]) {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "USAGE:\n  cargo xtask check-tier-deps\n\n\
             Fails on upward tier dependencies, undeclared production edges, incomplete crate ownership, or generated crate-documentation drift."
        );
        return;
    }
    if args.len() > 2 {
        eprintln!("Fix: check-tier-deps takes no arguments.");
        process::exit(2);
    }

    let root = crate::checkout::checkout_root();
    let members = workspace_members(&root);
    let mut failures = Vec::new();

    for member in &members {
        let manifest = root.join(member).join("Cargo.toml");
        let tier = crate_tier(member);
        let text = read_bounded(&manifest);
        let table = parse_toml(&manifest, &text);
        scan_manifest(&member, tier, &table, &mut failures);
    }
    validate_cross_crate_promotion_contract(&root, &mut failures);
    validate_crate_ownership_registry(&root, &mut failures);

    if failures.is_empty() {
        println!(
            "check-tier-deps: {} workspace members; tier, ownership, and generated graph contracts agree",
            members.len()
        );
    } else {
        eprintln!("check-tier-deps: {} violation(s):", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        eprintln!(
            "Fix: remove the upward dependency or update the manifest and docs/CRATE_OWNERSHIP.toml together, then regenerate the ownership docs."
        );
        process::exit(1);
    }
}

fn validate_crate_ownership_registry(root: &Path, failures: &mut Vec<String>) {
    let script = root.join("scripts/crate_ownership.py");
    let output = match Command::new("python3")
        .arg(&script)
        .arg("--check")
        .current_dir(root)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            failures.push(format!(
                "could not launch `{}` with python3: {error}",
                script.display()
            ));
            return;
        }
    };
    if output.status.success() {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let message = stderr.trim();
    failures.push(if message.is_empty() {
        format!(
            "`{}` failed with status {} and no diagnostic",
            script.display(),
            output.status
        )
    } else {
        message.to_string()
    });
}

fn workspace_members(root: &Path) -> Vec<String> {
    let text = read_bounded(&root.join("Cargo.toml"));
    let table = parse_toml(&root.join("Cargo.toml"), &text);
    table
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Lower number = more fundamental (may be depended upon by higher tiers).
fn crate_tier(member_path: &str) -> u32 {
    let name = member_path.rsplit('/').next().unwrap_or(member_path);
    match name {
        "vyre-foundation" | "vyre-spec" | "vyre" | "vyre-macros" => 10,
        "vyre-primitives" => 25,
        "vyre-pass-engine" => 28,
        "vyre-libs" => 30,
        "vyre-reference" | "vyre-lower" | "vyre-emit-naga" | "vyre-emit-ptx"
        | "vyre-emit-spirv" => 35,
        "vyre-conform-spec" | "vyre-conform" => 35,
        "vyre-driver"
        | "vyre-driver-wgpu"
        | "vyre-driver-cuda"
        | "vyre-driver-spirv"
        | "vyre-driver-reference"
        | "vyre-runtime"
        | "vyre-aot"
        | "vyre-bench"
        | "vyre-debug"
        | "vyre-lints" => 40,
        "xtask" => 99,
        _ => 45,
    }
}

fn resolve_path_dep(member: &str, dep_path: &str) -> Option<String> {
    let base = crate::checkout::checkout_root().join(member).join(dep_path);
    let canonical = base.canonicalize().ok()?;
    let root = crate::checkout::checkout_root().canonicalize().ok()?;
    let rel = canonical.strip_prefix(&root).ok()?;
    if rel.as_os_str().is_empty() {
        return None;
    }
    let s = rel.to_string_lossy();
    let member = s
        .trim_start_matches("./")
        .trim_end_matches("/Cargo.toml")
        .trim_end_matches('\\');
    if member.ends_with("Cargo.toml") {
        member
            .strip_suffix("/Cargo.toml")
            .or_else(|| member.strip_suffix("\\Cargo.toml"))
            .map(str::to_string)
    } else {
        Some(member.to_string())
    }
}

fn dep_crate_name(dep_key: &str, value: &Value) -> Option<String> {
    if let Some(path) = value.get("path").and_then(Value::as_str) {
        return Some(path.to_string());
    }
    if let Some(pkg) = value.get("package").and_then(Value::as_str) {
        return Some(pkg.to_string());
    }
    Some(dep_key.to_string())
}

fn scan_manifest(member: &str, tier: u32, table: &Value, failures: &mut Vec<String>) {
    let deps_tables = [
        ("dependencies", table.get("dependencies")),
        ("dev-dependencies", table.get("dev-dependencies")),
        ("build-dependencies", table.get("build-dependencies")),
    ];
    for (dep_kind, deps) in deps_tables {
        let Some(deps) = deps else {
            continue;
        };
        let Some(deps) = deps.as_table() else {
            continue;
        };
        for (key, value) in deps {
            let Some(path) = value.get("path").and_then(Value::as_str) else {
                continue;
            };
            let resolved = resolve_path_dep(member, path);
            let fallback = dep_crate_name(key, value);
            let dep_name = resolved.or(fallback).unwrap_or_else(|| key.to_string());
            let dep_tier = crate_tier(&dep_name);
            if dep_tier > tier && tier < 99 && dep_kind != "dev-dependencies" {
                failures.push(format!(
                    "{member} (T{tier}) must not path-depend on {dep_name} (T{dep_tier}) via `{key}` = `{path}` in {dep_kind}"
                ));
            }
        }
    }
}

fn validate_cross_crate_promotion_contract(root: &Path, failures: &mut Vec<String>) {
    let crate_graph = read_contract_doc(root, "docs/CRATE_GRAPH.md", failures);
    let lego_rule = read_contract_doc(root, "docs/lego-block-rule.md", failures);
    failures.extend(cross_crate_promotion_contract_text_failures(
        crate_graph.as_deref().unwrap_or(""),
        lego_rule.as_deref().unwrap_or(""),
    ));
}

fn read_contract_doc(root: &Path, rel: &str, failures: &mut Vec<String>) -> Option<String> {
    let path = root.join(rel);
    match fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(error) => {
            failures.push(format!(
                "cross-crate promotion contract could not read `{rel}`: {error}"
            ));
            None
        }
    }
}

fn cross_crate_promotion_contract_text_failures(crate_graph: &str, lego_rule: &str) -> Vec<String> {
    let mut failures = Vec::new();
    // The generated crate graph proves the dependency surface exists and is
    // fresh (crate_ownership.py --check); the LEGO rule owns the promotion
    // contract text, so the marker requirement applies to the rule doc.
    if crate_graph.is_empty() {
        failures.push("docs/CRATE_GRAPH.md is empty or unreadable".to_string());
    }
    for marker in [
        "Cross-crate promotion patch contract",
        "import-path migration test",
        "check-tier-deps",
        "lego-audit",
    ] {
        if !lego_rule.contains(marker) {
            failures.push(format!(
                "docs/lego-block-rule.md is missing `{marker}` for cross-crate promotion ownership"
            ));
        }
    }
    failures
}

fn read_bounded(path: &Path) -> String {
    crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, "tier dependency manifest")
        .unwrap_or_else(|error| {
            panic!("Fix: cannot read {}: {error}", path.display());
        })
}

fn parse_toml(path: &Path, text: &str) -> Value {
    let table: toml::Table = toml::from_str(text).unwrap_or_else(|e| {
        panic!("Fix: parse {}: {e}", path.display());
    });
    Value::Table(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_DOC: &str = "Cross-crate promotion patch contract\nimport-path migration test\ncheck-tier-deps\nlego-audit\n";

    #[test]
    fn cross_crate_promotion_contract_accepts_complete_docs() {
        assert!(cross_crate_promotion_contract_text_failures("graph", VALID_DOC).is_empty());
    }

    #[test]
    fn cross_crate_promotion_contract_rejects_missing_markers() {
        let failures =
            cross_crate_promotion_contract_text_failures("graph", "check-tier-deps\nlego-audit\n");

        assert!(failures
            .iter()
            .any(|failure| failure.contains("import-path migration test")));
        assert!(failures
            .iter()
            .any(|failure| failure.contains("Cross-crate promotion patch contract")));
    }

    #[test]
    fn cross_crate_promotion_contract_rejects_missing_graph() {
        let failures = cross_crate_promotion_contract_text_failures("", VALID_DOC);

        assert!(failures
            .iter()
            .any(|failure| failure.contains("docs/CRATE_GRAPH.md")));
    }
}

#[cfg(test)]
mod dependency_kind_tests {
    use super::*;

    #[test]
    fn production_upward_path_dependency_fails() {
        let table = parse_toml(
            Path::new("fixture/Cargo.toml"),
            r#"
[dependencies]
vyre-driver = { path = "../vyre-driver" }
"#,
        );
        let mut failures = Vec::new();

        scan_manifest(
            "vyre-primitives",
            crate_tier("vyre-primitives"),
            &table,
            &mut failures,
        );

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("dependencies"));
    }

    #[test]
    fn dev_upward_path_dependency_is_allowed_for_contract_tests() {
        let table = parse_toml(
            Path::new("fixture/Cargo.toml"),
            r#"
[dev-dependencies]
vyre-driver = { path = "../vyre-driver" }
"#,
        );
        let mut failures = Vec::new();

        scan_manifest(
            "vyre-primitives",
            crate_tier("vyre-primitives"),
            &table,
            &mut failures,
        );

        assert!(failures.is_empty(), "{failures:?}");
    }
}
