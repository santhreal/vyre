//! A production crate must not declare capability compiled only for test builds.
//!
//! Test-gating a whole module hides it from every gate that walks production
//! items: the hot-path scan, the duplication scan, and the public API snapshot
//! all skip test-gated items. `xtask/src/gates/schedule_ownership.rs`
//! `is_test_gated` skips test-gated code explicitly, which means an execution
//! route or capability stays unjudged for as long as it is test-gated.
//!
//! None of the six current test-gated modules in `vyre-libs` hides a selection
//! route. `polyhedral_fusion.rs` operates on primitive slices and returns
//! `Vec<u32>` and `u32`, minting no decision variant and writing no geometry.
//! The would-be selectors behind test gates construct test fixture inputs.
//! The class this gate enforces is unshipped capability, not concealed selection.
//!
//! Two forms constitute this defect class:
//! 1. A public module declaration (`pub mod`) gated behind `cfg(test)` or
//!    `cfg(all(test, ...))` in a production crate.
//! 2. A production module file that compiles in production but whose public
//!    capability is empty because all public executable functions are
//!    individually test-gated.
//!
//! A feature gate is a product configuration and is not a finding: a supported
//! build can select the feature to compile the module. `cfg(test)` admits no
//! production build at all.
//!
//! Test-support and fixture modules in non-production crate layers (`tooling`,
//! `test-tooling`, `standalone-tooling`, `conformance`) and test trees
//! (`tests/`, `benches/`, `fuzz/`, `test/`) are exempt. The exemption is
//! derived from the declared layer in `docs/CRATE_OWNERSHIP.toml`.
//!
//! An inline `mod tests` or `mod test` module is exempt whether or not it
//! carries `pub`.
//!
//! What this gate does not catch: the gate sees a `cfg` attribute, not whether
//! anything would call the module if it compiled. A production helper function
//! that is only called by test code but carries no `cfg(test)` gate is not
//! reported; uncalled production code is judged by dead-code analysis and
//! reachability gates.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use syn::spanned::Spanned;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Crate layers from `docs/CRATE_OWNERSHIP.toml` exempt from production capability rules.
const EXEMPT_LAYERS: &[&str] = &[
    "tooling",
    "test-tooling",
    "standalone-tooling",
    "conformance",
];

/// A gate verifying that production crates do not declare test-only capability.
pub struct TestOnlyCapability;

impl crate::gate::GateBehavior for TestOnlyCapability {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        let registry = CrateLayerRegistry::read(&tree)?;
        let members = tree.member_manifests()?;

        let mut member_packages: Vec<(String, String, bool)> = Vec::new();
        for member in &members {
            let layer = registry.layer_of(&member.name);
            let is_exempt = EXEMPT_LAYERS.contains(&layer.as_str());
            member_packages.push((member.path.clone(), member.name.clone(), is_exempt));
        }

        let all_rust = tree.all_rust();
        let mut production_files = Vec::new();

        for path in &all_rust {
            if scan::is_test_tree(path) {
                continue;
            }
            let is_exempt_crate = member_packages.iter().any(|(member_path, _, is_exempt)| {
                scan::under(path, member_path) && *is_exempt
            });
            if !is_exempt_crate {
                production_files.push(path.clone());
            }
        }

        report.cover_complete("production rust source files", production_files.len());
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }

        let mut test_gated_submodules = BTreeSet::new();
        let mut parsed_files = BTreeMap::new();

        for path in &production_files {
            let text = tree.read(path)?;
            if let Ok(file) = syn::parse_file(&text) {
                collect_test_gated_submodules(path, &file, &mut test_gated_submodules);
                parsed_files.insert(path.clone(), file);
            }
        }

        let mut findings = Vec::new();
        for (path, file) in &parsed_files {
            inspect_file(path, file, &test_gated_submodules, &mut findings);
        }

        for finding in findings {
            report.find(finding);
        }

        Ok(report)
    }
}

/// Registry mapping crates to their declared layer in `docs/CRATE_OWNERSHIP.toml`.
struct CrateLayerRegistry {
    layers: BTreeMap<String, String>,
}

impl CrateLayerRegistry {
    /// Read layer mappings from `docs/CRATE_OWNERSHIP.toml`.
    fn read(tree: &Tree) -> Result<Self, GateError> {
        let table = tree.read_toml("docs/CRATE_OWNERSHIP.toml")?;
        let crates = table
            .get("crate")
            .and_then(toml::Value::as_array)
            .ok_or_else(|| {
                GateError::new(
                    "docs/CRATE_OWNERSHIP.toml declares no [[crate]] entries",
                    "Fix: record crate layer declarations in docs/CRATE_OWNERSHIP.toml",
                )
            })?;

        let mut layers = BTreeMap::new();
        for entry in crates {
            if let (Some(package), Some(layer)) = (
                entry.get("package").and_then(toml::Value::as_str),
                entry.get("layer").and_then(toml::Value::as_str),
            ) {
                layers.insert(package.to_string(), layer.to_string());
            }
        }

        Ok(Self { layers })
    }

    /// The layer declared for a package.
    fn layer_of(&self, package: &str) -> String {
        self.layers.get(package).cloned().unwrap_or_default()
    }
}

/// Collect paths of submodule files that are declared test-gated in their parent module.
fn collect_test_gated_submodules(
    parent_path: &Path,
    file: &syn::File,
    test_gated: &mut BTreeSet<PathBuf>,
) {
    let parent_dir = parent_path.parent().unwrap_or(Path::new(""));
    let stem = parent_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();

    for item in &file.items {
        if let syn::Item::Mod(item_mod) = item {
            if item_mod.content.is_none() && is_test_only_gated(&item_mod.attrs) {
                let mod_name = item_mod.ident.to_string();
                if stem == "mod" || stem == "lib" || stem == "main" {
                    test_gated.insert(parent_dir.join(format!("{mod_name}.rs")));
                    test_gated.insert(parent_dir.join(&mod_name).join("mod.rs"));
                } else {
                    test_gated.insert(parent_dir.join(stem).join(format!("{mod_name}.rs")));
                    test_gated.insert(parent_dir.join(stem).join(&mod_name).join("mod.rs"));
                }
            }
        }
    }
}

/// Inspect one parsed Rust source file for test-only capability.
fn inspect_file(
    path: &Path,
    file: &syn::File,
    test_gated_submodules: &BTreeSet<PathBuf>,
    findings: &mut Vec<Finding>,
) {
    let display = path.display().to_string();

    // Arm 1: Check for test-only public module declarations.
    for item in &file.items {
        if let syn::Item::Mod(item_mod) = item {
            if !matches!(item_mod.vis, syn::Visibility::Public(_)) {
                continue;
            }
            let name = item_mod.ident.to_string();
            // Inline test module exemption (e.g. `pub mod tests { ... }`).
            if (name == "tests" || name == "test") && item_mod.content.is_some() {
                continue;
            }
            if is_test_only_gated(&item_mod.attrs) {
                let line = item_mod.span().start().line as u32;
                findings.push(Finding::at(
                    path.to_path_buf(),
                    line,
                    format!("{display}:{line}: public module `{name}` is compiled only for test builds"),
                    "remove `#[cfg(test)]` so the capability is available in production builds, or move the test-only code under `tests/` or a fixture module",
                ));
            }
        }
    }

    // Arm 2: Check for production module files that are empty in production because
    // all public functions/capabilities inside them are individually test-gated.
    if !test_gated_submodules.contains(path) && !is_test_only_gated(&file.attrs) {
        inspect_file_arm2(path, file, findings);
    }
}

/// Inspect a file for Arm 2: public functions all test-gated.
fn inspect_file_arm2(path: &Path, file: &syn::File, findings: &mut Vec<Finding>) {
    let mut total_pub_fns = 0usize;
    let mut test_gated_pub_fns = 0usize;
    let mut other_pub_items = 0usize;

    for item in &file.items {
        match item {
            syn::Item::Fn(item_fn) if matches!(item_fn.vis, syn::Visibility::Public(_)) => {
                total_pub_fns += 1;
                if is_test_only_gated(&item_fn.attrs) {
                    test_gated_pub_fns += 1;
                }
            }
            syn::Item::Mod(_) => {
                // Modules are checked by Arm 1.
            }
            _ => {
                // Check if other items have public visibility and are ungated.
                if is_public_item(item) && !is_item_test_gated(item) {
                    if !is_auxiliary_type(item) {
                        other_pub_items += 1;
                    }
                }
            }
        }
    }

    if total_pub_fns > 0 && total_pub_fns == test_gated_pub_fns && other_pub_items == 0 {
        let display = path.display().to_string();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        findings.push(Finding::at(
            path.to_path_buf(),
            1,
            format!(
                "{display}:1: module `{stem}` compiles in production but all {total_pub_fns} public functions are compiled only for test builds"
            ),
            "remove `#[cfg(test)]` from public functions so module capability is available in production, or test-gate the module declaration in the parent module",
        ));
    }
}

/// Whether an item is public.
fn is_public_item(item: &syn::Item) -> bool {
    match item {
        syn::Item::Const(i) => matches!(i.vis, syn::Visibility::Public(_)),
        syn::Item::Enum(i) => matches!(i.vis, syn::Visibility::Public(_)),
        syn::Item::Static(i) => matches!(i.vis, syn::Visibility::Public(_)),
        syn::Item::Struct(i) => matches!(i.vis, syn::Visibility::Public(_)),
        syn::Item::Trait(i) => matches!(i.vis, syn::Visibility::Public(_)),
        syn::Item::Type(i) => matches!(i.vis, syn::Visibility::Public(_)),
        syn::Item::Use(i) => matches!(i.vis, syn::Visibility::Public(_)),
        _ => false,
    }
}

/// Whether an item has test-only attributes.
fn is_item_test_gated(item: &syn::Item) -> bool {
    match item {
        syn::Item::Const(i) => is_test_only_gated(&i.attrs),
        syn::Item::Enum(i) => is_test_only_gated(&i.attrs),
        syn::Item::Fn(i) => is_test_only_gated(&i.attrs),
        syn::Item::Static(i) => is_test_only_gated(&i.attrs),
        syn::Item::Struct(i) => is_test_only_gated(&i.attrs),
        syn::Item::Trait(i) => is_test_only_gated(&i.attrs),
        syn::Item::Type(i) => is_test_only_gated(&i.attrs),
        syn::Item::Use(i) => is_test_only_gated(&i.attrs),
        _ => false,
    }
}

/// Whether an item is an auxiliary error/telemetry type rather than a domain AST type.
fn is_auxiliary_type(item: &syn::Item) -> bool {
    match item {
        syn::Item::Enum(item_enum) => {
            let name = item_enum.ident.to_string();
            name.ends_with("Error")
        }
        syn::Item::Struct(item_struct) => {
            let name = item_struct.ident.to_string();
            name.ends_with("Telemetry")
        }
        syn::Item::Trait(item_trait) => {
            let name = item_trait.ident.to_string();
            name.ends_with("Sample")
        }
        _ => false,
    }
}

/// Whether attributes gate an item exclusively to test builds.
pub fn is_test_only_gated(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(is_test_only_attribute)
}

/// Whether one attribute gates an item exclusively to test builds.
fn is_test_only_attribute(attr: &syn::Attribute) -> bool {
    if attr.path().is_ident("test") {
        return true;
    }
    if !attr.path().is_ident("cfg") {
        return false;
    }
    match &attr.meta {
        syn::Meta::List(list) => {
            if let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) {
                // Multiple cfg arguments `cfg(a, b)` act as `all(a, b)`.
                !nested.iter().any(eval_cfg_for_production)
            } else if let Ok(meta) = list.parse_args::<syn::Meta>() {
                !eval_cfg_for_production(&meta)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Evaluate whether a `cfg` meta expression can be satisfied in any production build.
///
/// In production builds `test` is false, while target and feature configurations can be enabled.
fn eval_cfg_for_production(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::Path(path) => {
            // `test` is false in production builds.
            !path.is_ident("test")
        }
        syn::Meta::NameValue(_) => {
            // e.g. `feature = "..."`, `target_os = "..."` can be satisfied in production.
            true
        }
        syn::Meta::List(list) => {
            if list.path.is_ident("not") {
                if let Ok(inner) = list.parse_args::<syn::Meta>() {
                    !eval_cfg_for_production(&inner)
                } else {
                    true
                }
            } else if list.path.is_ident("all") {
                if let Ok(nested) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) {
                    nested.iter().all(eval_cfg_for_production)
                } else {
                    true
                }
            } else if list.path.is_ident("any") {
                if let Ok(nested) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) {
                    nested.iter().any(eval_cfg_for_production)
                } else {
                    true
                }
            } else {
                true
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn a_test_only_pub_mod_in_a_production_crate_is_reported() {
        let code = r#"
            #[cfg(test)]
            pub mod test_gated_submodule;
        "#;
        let file = syn::parse_file(code).expect("Fix: parse test code");
        let mut findings = Vec::new();
        let submodules = BTreeSet::new();
        inspect_file(Path::new("vyre-libs/src/encoding/mod.rs"), &file, &submodules, &mut findings);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("test_gated_submodule"));
        assert!(findings[0].message.contains("compiled only for test builds"));
    }

    #[test]
    fn feature_gated_pub_mod_is_not_reported() {
        let code = r#"
            #[cfg(any(
                feature = "nn-activation",
                feature = "nn-linear",
                feature = "nn-norm",
                feature = "nn-attention"
            ))]
            pub mod nn_attention_paging;

            #[cfg(any(test, feature = "test-fixtures"))]
            pub mod artifact_golden;
        "#;
        let file = syn::parse_file(code).expect("Fix: parse test code");
        let mut findings = Vec::new();
        let submodules = BTreeSet::new();
        inspect_file(Path::new("vyre-libs/src/encoding/mod.rs"), &file, &submodules, &mut findings);
        assert!(findings.is_empty(), "feature-gated modules must not produce findings");
    }

    #[test]
    fn inline_mod_tests_is_exempt() {
        let code = r#"
            #[cfg(test)]
            pub mod tests {
                #[test]
                fn test_something() {}
            }

            #[cfg(test)]
            mod test {
                #[test]
                fn test_inner() {}
            }
        "#;
        let file = syn::parse_file(code).expect("Fix: parse test code");
        let mut findings = Vec::new();
        let submodules = BTreeSet::new();
        inspect_file(Path::new("vyre-foundation/src/transform/parallelism.rs"), &file, &submodules, &mut findings);
        assert!(findings.is_empty(), "inline test modules must be exempt");
    }

    #[test]
    fn empty_production_module_with_test_gated_items_is_reported() {
        let code = r#"
            pub enum MegakernelScheduleError {
                CostLen { expected: usize, actual: usize },
            }

            pub struct MegakernelScaleTelemetry<'a> {
                pub frontier_density: &'a [f64],
            }

            #[cfg(test)]
            pub fn schedule_via_homotopy(costs: &[f64]) -> Vec<f64> {
                vec![]
            }

            #[cfg(test)]
            pub fn try_schedule_via_homotopy(costs: &[f64]) -> Result<Vec<f64>, MegakernelScheduleError> {
                Ok(vec![])
            }
        "#;
        let file = syn::parse_file(code).expect("Fix: parse test code");
        let mut findings = Vec::new();
        let submodules = BTreeSet::new();
        inspect_file(
            Path::new("vyre-libs/src/scheduling/megakernel_schedule.rs"),
            &file,
            &submodules,
            &mut findings,
        );
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("megakernel_schedule"));
        assert!(findings[0].message.contains("public functions are compiled only for test builds"));
    }

    #[test]
    fn production_module_with_ungated_items_is_clean() {
        let code = r#"
            pub enum FrontierDomain { Parser, Semantic }
            pub struct FrontierNode { pub id: u32 }
            pub fn run_pipeline(nodes: &[FrontierNode]) -> bool { true }
        "#;
        let file = syn::parse_file(code).expect("Fix: parse test code");
        let mut findings = Vec::new();
        let submodules = BTreeSet::new();
        inspect_file(Path::new("vyre-libs/src/scheduling/pipeline.rs"), &file, &submodules, &mut findings);
        assert!(findings.is_empty());
    }
}
