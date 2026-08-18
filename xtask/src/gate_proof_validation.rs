//! Source-derived symbol validation for authoritative gate descriptor proofs.
//!
//! Validates that every gate descriptor proof mechanically resolves to a real,
//! unique `#[test]` function in its owning crate's source tree.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use crate::gate::GateDescriptor;

/// Structural record of a resolved symbol in a package.
#[derive(Clone, Debug)]
struct SymbolMatch {
    file: PathBuf,
    line: usize,
    is_test: bool,
}

/// Pre-indexed symbol map for a package source tree.
#[derive(Clone, Debug, Default)]
struct PackageSymbolIndex {
    symbols: HashMap<(Vec<String>, String), Vec<SymbolMatch>>,
    failures: Vec<String>,
}

impl PackageSymbolIndex {
    /// Build one index over `crate_src`.
    #[must_use]
    fn build(crate_src: &Path) -> Self {
        let mut index = Self::default();
        let mut files = Vec::new();
        collect_rs_files(crate_src, &mut files, &mut index.failures);
        for file in files {
            let relative = match file.strip_prefix(crate_src) {
                Ok(rel) => rel,
                Err(err) => {
                    index.failures.push(format!(
                        "failed to strip prefix `{}` from `{}`: {err}",
                        crate_src.display(),
                        file.display()
                    ));
                    continue;
                }
            };
            let mut file_mod_parts: Vec<String> = relative
                .with_extension("")
                .iter()
                .map(|s| s.to_string_lossy().to_string())
                .collect();
            if file_mod_parts.last().map(String::as_str) == Some("mod")
                || file_mod_parts.last().map(String::as_str) == Some("lib")
                || file_mod_parts.last().map(String::as_str) == Some("main")
            {
                file_mod_parts.pop();
            }

            let text = match fs::read_to_string(&file) {
                Ok(t) => t,
                Err(err) => {
                    index
                        .failures
                        .push(format!("failed to read file `{}`: {err}", file.display()));
                    continue;
                }
            };
            let syntax_file = match syn::parse_file(&text) {
                Ok(sf) => sf,
                Err(err) => {
                    index.failures.push(format!(
                        "failed to parse syntax for `{}`: {err}",
                        file.display()
                    ));
                    continue;
                }
            };

            let mut current_mod = file_mod_parts;
            index_items(
                &syntax_file.items,
                &mut current_mod,
                &file,
                &mut index.symbols,
            );
        }
        index
    }
}

fn index_items(
    items: &[syn::Item],
    current_mod: &mut Vec<String>,
    file_path: &Path,
    symbols: &mut HashMap<(Vec<String>, String), Vec<SymbolMatch>>,
) {
    for item in items {
        match item {
            syn::Item::Fn(item_fn) => {
                let fn_name = item_fn.sig.ident.to_string();
                let is_test = item_fn.attrs.iter().any(|attr| {
                    attr.path().is_ident("test")
                        || (attr.path().segments.len() == 2
                            && attr.path().segments[0].ident == "tokio"
                            && attr.path().segments[1].ident == "test")
                });
                symbols
                    .entry((current_mod.clone(), fn_name))
                    .or_default()
                    .push(SymbolMatch {
                        file: file_path.to_path_buf(),
                        line: 1,
                        is_test,
                    });
            }
            syn::Item::Mod(item_mod) => {
                current_mod.push(item_mod.ident.to_string());
                if let Some((_, inline_items)) = &item_mod.content {
                    index_items(inline_items, current_mod, file_path, symbols);
                }
                current_mod.pop();
            }
            _ => {}
        }
    }
}

fn collect_rs_files(dir: &Path, sink: &mut Vec<PathBuf>, failures: &mut Vec<String>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            failures.push(format!(
                "failed to read directory `{}`: {err}",
                dir.display()
            ));
            return;
        }
    };
    for entry_res in entries {
        let entry = match entry_res {
            Ok(e) => e,
            Err(err) => {
                failures.push(format!(
                    "failed to read directory entry in `{}`: {err}",
                    dir.display()
                ));
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(err) => {
                failures.push(format!(
                    "failed to read file type for `{}`: {err}",
                    path.display()
                ));
                continue;
            }
        };
        if file_type.is_symlink() {
            failures.push(format!(
                "proof source walk does not follow symlink `{}`; keep authoritative Rust proof sources inside the owning package tree",
                path.display()
            ));
        } else if file_type.is_dir() {
            collect_rs_files(&path, sink, failures);
        } else if file_type.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs") {
            sink.push(path);
        }
    }
}

/// Batch validate all gate descriptors in one pass with package source symbol indexing.
pub fn validate_all_descriptors(root: &Path, descriptors: &[GateDescriptor]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut seen_proofs = BTreeSet::new();

    // Check proof uniqueness across descriptors
    for desc in descriptors {
        if !seen_proofs.insert(desc.proof) {
            failures.push(format!(
                "duplicate proof identity in gate descriptors for gate `{}`: `{}`",
                desc.name, desc.proof
            ));
        }
    }

    // Build one index per package
    let mut indices: HashMap<&'static str, PackageSymbolIndex> = HashMap::new();
    for package in ["xtask", "xtask-registry", "xtask-evidence"] {
        let crate_dir = match package {
            "xtask" => root.join("xtask").join("src"),
            "xtask-registry" => root.join("xtask-registry").join("src"),
            "xtask-evidence" => root.join("xtask-evidence").join("src"),
            _ => continue,
        };
        if crate_dir.exists() {
            let index = PackageSymbolIndex::build(&crate_dir);
            failures.extend(index.failures.clone());
            indices.insert(package, index);
        } else {
            failures.push(format!(
                "crate source directory `{}` does not exist",
                crate_dir.display()
            ));
        }
    }

    for desc in descriptors {
        failures.extend(validate_proof_with_index(desc, &indices));
    }

    failures
}

fn validate_proof_with_index(
    descriptor: &GateDescriptor,
    indices: &HashMap<&'static str, PackageSymbolIndex>,
) -> Vec<String> {
    let mut failures = Vec::new();
    let proof = descriptor.proof.trim();
    if proof.is_empty() || proof == "definition-site mutation tests" {
        failures.push(format!(
            "gate `{}` declares no mutation-proof test",
            descriptor.name
        ));
        return failures;
    }
    if proof.ends_with("::enforces_invariants") {
        failures.push(format!(
            "gate `{}` declares generic invented proof placeholder `{proof}`",
            descriptor.name
        ));
        return failures;
    }

    let package_prefix = match descriptor.package {
        "xtask" => "crate::",
        "xtask-registry" => "xtask_registry::",
        "xtask-evidence" => "xtask_evidence::",
        other => {
            failures.push(format!(
                "gate `{}` declares unknown owner package `{other}`",
                descriptor.name
            ));
            return failures;
        }
    };

    // Check wrong package prefixes
    for (other_pkg, other_pfx) in [
        ("xtask", "crate::"),
        ("xtask-registry", "xtask_registry::"),
        ("xtask-evidence", "xtask_evidence::"),
    ] {
        if other_pkg != descriptor.package && proof.starts_with(other_pfx) {
            failures.push(format!(
                "gate `{}` owned by package `{}` declares proof `{proof}` belonging to package `{other_pkg}`",
                descriptor.name, descriptor.package
            ));
            return failures;
        }
    }

    if !proof.starts_with(package_prefix) {
        failures.push(format!(
            "gate `{}` owned by package `{}` declares proof `{proof}` missing required package prefix `{package_prefix}`",
            descriptor.name, descriptor.package
        ));
        return failures;
    }

    let relative_symbol = &proof[package_prefix.len()..];
    let segments: Vec<&str> = relative_symbol.split("::").collect();
    if segments.len() < 2 {
        failures.push(format!(
            "gate `{}` declares invalid short proof symbol `{proof}`",
            descriptor.name
        ));
        return failures;
    }

    let fn_name = segments[segments.len() - 1].to_string();
    let mod_segments: Vec<String> = segments[..segments.len() - 1]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let Some(index) = indices.get(descriptor.package) else {
        failures.push(format!(
            "package `{}` index not available for gate `{}`",
            descriptor.package, descriptor.name
        ));
        return failures;
    };

    let key = (mod_segments, fn_name.clone());
    let matches = index.symbols.get(&key);

    match matches {
        None => {
            failures.push(format!(
                "gate `{}` proof symbol `{proof}` does not exist in package `{}` source",
                descriptor.name, descriptor.package
            ));
        }
        Some(list) if list.len() > 1 => {
            failures.push(format!(
                "gate `{}` proof symbol `{proof}` is ambiguous: matches {} definitions in package `{}`",
                descriptor.name,
                list.len(),
                descriptor.package
            ));
        }
        Some(list) if !list[0].is_test => {
            failures.push(format!(
                "gate `{}` proof symbol `{proof}` resolves to function `{fn_name}` at {}:{} which is not annotated with #[test]",
                descriptor.name,
                list[0].file.display(),
                list[0].line
            ));
        }
        _ => {}
    }

    failures
}

/// Validate that a gate descriptor's proof symbol is package-qualified and resolves to a real #[test] function.
pub fn validate_proof_symbol(root: &Path, descriptor: &GateDescriptor) -> Vec<String> {
    let mut failures = Vec::new();
    let crate_dir = match descriptor.package {
        "xtask" => root.join("xtask").join("src"),
        "xtask-registry" => root.join("xtask-registry").join("src"),
        "xtask-evidence" => root.join("xtask-evidence").join("src"),
        other => {
            failures.push(format!(
                "gate `{}` declares unknown owner package `{other}`",
                descriptor.name
            ));
            return failures;
        }
    };
    if !crate_dir.exists() {
        failures.push(format!(
            "crate source directory `{}` does not exist",
            crate_dir.display()
        ));
        return failures;
    }
    let index = PackageSymbolIndex::build(&crate_dir);
    failures.extend(index.failures.clone());
    let mut indices = HashMap::new();
    indices.insert(descriptor.package, index);
    failures.extend(validate_proof_with_index(descriptor, &indices));
    failures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_metadata::GATE_METADATA;

    /// WHY: Section 182 requires proof identities to reject missing symbols fail-closed.
    #[test]
    fn proof_identity_rejects_missing_symbol() {
        let root = crate::checkout::checkout_root();
        let descriptor = GateDescriptor {
            name: "test-missing",
            help: "Help text",
            package: "xtask",
            areas: &["prepublish"],
            subject: "test subjects",
            artifacts: &[],
            prerequisites: &[],
            proof: "crate::gates::architecture_contract::tests::nonexistent_symbol_xyz",
        };
        let failures = validate_proof_symbol(&root, &descriptor);
        assert!(!failures.is_empty(), "must fail for missing symbol");
        assert!(
            failures[0].contains("does not exist in package `xtask` source"),
            "expected missing symbol error, got: {:?}",
            failures
        );
    }

    /// WHY: Section 182 requires proof identities to reject ambiguous symbols fail-closed.
    #[test]
    fn proof_identity_rejects_ambiguous_symbol() {
        let (_temporary, root) = crate::gates::fixture_checkout::checkout(&[
            (
                "xtask/src/gates/alpha.rs",
                "#[cfg(test)]\nmod tests {\n    #[test]\n    fn duplicate_proof() {}\n}\n",
            ),
            (
                "xtask/src/gates/alpha/mod.rs",
                "#[cfg(test)]\nmod tests {\n    #[test]\n    fn duplicate_proof() {}\n}\n",
            ),
        ]);
        let descriptor = GateDescriptor {
            name: "test-ambiguous",
            help: "Help text",
            package: "xtask",
            areas: &["prepublish"],
            subject: "test subjects",
            artifacts: &[],
            prerequisites: &[],
            proof: "crate::gates::alpha::tests::duplicate_proof",
        };
        let failures = validate_proof_symbol(&root, &descriptor);
        assert!(!failures.is_empty(), "must fail for ambiguous symbol");
        assert!(
            failures[0].contains("is ambiguous: matches 2 definitions in package `xtask`"),
            "expected ambiguous symbol error, got: {:?}",
            failures
        );
    }

    /// WHY: Section 182 requires proof identities to reject non-test functions fail-closed.
    #[test]
    fn proof_identity_rejects_non_test_symbol() {
        let root = crate::checkout::checkout_root();
        let descriptor = GateDescriptor {
            name: "test-non-test",
            help: "Help text",
            package: "xtask",
            areas: &["prepublish"],
            subject: "test subjects",
            artifacts: &[],
            prerequisites: &[],
            proof: "crate::gate_metadata::owned_by",
        };
        let failures = validate_proof_symbol(&root, &descriptor);
        assert!(!failures.is_empty(), "must fail for non-test symbol");
        assert!(
            failures[0].contains("not annotated with #[test]"),
            "expected non-test error, got: {:?}",
            failures
        );
    }

    /// WHY: Section 182 requires proof identities to reject symbols from other packages fail-closed.
    #[test]
    fn proof_identity_rejects_wrong_package_symbol() {
        let root = crate::checkout::checkout_root();
        let descriptor = GateDescriptor {
            name: "test-wrong-pkg",
            help: "Help text",
            package: "xtask",
            areas: &["prepublish"],
            subject: "test subjects",
            artifacts: &[],
            prerequisites: &[],
            proof: "xtask_registry::gates::abstraction_gate::tests::a_finding_under_every_nesting_variant_is_reported",
        };
        let failures = validate_proof_symbol(&root, &descriptor);
        assert!(!failures.is_empty(), "must fail for wrong package prefix");
        assert!(
            failures[0].contains("belonging to package `xtask-registry`"),
            "expected wrong-package error, got: {:?}",
            failures
        );
    }

    /// WHY: Section 182 requires proof identities to reject generic invented enforces_invariants placeholders.
    #[test]
    fn proof_identity_rejects_generic_enforces_invariants() {
        let root = crate::checkout::checkout_root();
        let descriptor = GateDescriptor {
            name: "test-generic",
            help: "Help text",
            package: "xtask",
            areas: &["prepublish"],
            subject: "test subjects",
            artifacts: &[],
            prerequisites: &[],
            proof: "crate::gates::architecture_contract::tests::enforces_invariants",
        };
        let failures = validate_proof_symbol(&root, &descriptor);
        assert!(!failures.is_empty(), "must fail for generic placeholder");
        assert!(
            failures[0].contains("generic invented proof placeholder"),
            "expected generic placeholder error, got: {:?}",
            failures
        );
    }

    /// WHY: Section 182 requires every descriptor proof to mechanically resolve to an existing #[test] function.
    #[test]
    fn all_descriptor_proofs_resolve_to_real_tests() {
        let root = crate::checkout::checkout_root();
        let failures = validate_all_descriptors(&root, GATE_METADATA);
        assert!(
            failures.is_empty(),
            "GATE_METADATA proofs failed batch validation: {:?}",
            failures
        );
    }
}
