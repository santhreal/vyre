//! Proves that the vyre facade exposes no optimizer, search, lowering,
//! backend, or runtime internal symbols.
//!
//! Asserts that:
//! 1. All re-exports in `vyre/src/lib.rs` are derived from public seams (e.g. frontend IR,
//!    authenticated artifact admission, curated compiler entry points, soundness).
//! 2. No re-export originates from internal optimizer passes, search space planners,
//!    lowering/emission machinery, concrete driver internals, or runtime execution workers.
//! 3. The compiler module is an explicit, curated namespace rather than a whole-crate re-export.
//! 4. Adversarial re-export additions (optimizer, search, lowering, driver, runtime worker) are caught.

use std::path::PathBuf;
use syn::{Item, ItemMod, ItemUse, UsePath, UseRename, UseTree, Visibility};

/// Disallowed module path segments in public facade re-exports.
pub(crate) const FORBIDDEN_INTERNAL_PATH_SEGMENTS: &[&str] = &[
    "optimizer",
    "passes",
    "search",
    "candidate",
    "select",
    "cost",
    "extraction_cost",
    "lower",
    "lowering",
    "emit",
    "driver_cuda",
    "driver_metal",
    "driver_spirv",
    "driver_reference",
    "worker_process",
    "thread_pool",
    "resident_queue",
    "dispatch_table",
];

/// The facade source of this checkout.
///
/// The crate directory is resolved from the working directory through the
/// workspace member roster. A compiled-in manifest path names whichever
/// checkout last built this binary through the shared target directory, so the
/// re-exports read would be that tree's.
fn facade_source_path() -> PathBuf {
    vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("src/lib.rs")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExportedPath {
    pub segments: Vec<String>,
    pub is_whole_crate_alias: bool,
    pub is_glob: bool,
}

impl ExportedPath {
    pub(crate) fn full_path(&self) -> String {
        self.segments.join("::")
    }
}

fn collect_use_paths(tree: &UseTree, prefix: &mut Vec<String>, out: &mut Vec<ExportedPath>) {
    match tree {
        UseTree::Path(UsePath { ident, tree, .. }) => {
            prefix.push(ident.to_string());
            collect_use_paths(tree, prefix, out);
            prefix.pop();
        }
        UseTree::Name(syn::UseName { ident, .. }) => {
            prefix.push(ident.to_string());
            out.push(ExportedPath {
                segments: prefix.clone(),
                is_whole_crate_alias: false,
                is_glob: false,
            });
            prefix.pop();
        }
        UseTree::Rename(UseRename { ident, rename, .. }) => {
            // Check if this is a top-level whole crate alias like `pub use vyre_megakernel as compiler;`
            let is_top_crate = prefix.is_empty();
            prefix.push(format!("{ident} as {rename}"));
            out.push(ExportedPath {
                segments: prefix.clone(),
                is_whole_crate_alias: is_top_crate,
                is_glob: false,
            });
            prefix.pop();
        }
        UseTree::Glob(_) => {
            let mut segs = prefix.clone();
            segs.push("*".to_string());
            out.push(ExportedPath {
                segments: segs,
                is_whole_crate_alias: false,
                is_glob: true,
            });
        }
        UseTree::Group(syn::UseGroup { items, .. }) => {
            for item in items {
                collect_use_paths(item, prefix, out);
            }
        }
    }
}

pub(crate) fn extract_public_exports_from_ast(file: &syn::File) -> Vec<ExportedPath> {
    let mut exports = Vec::new();

    for item in &file.items {
        match item {
            Item::Use(ItemUse {
                vis: Visibility::Public(_),
                tree,
                ..
            }) => {
                let mut prefix = Vec::new();
                collect_use_paths(tree, &mut prefix, &mut exports);
            }
            Item::Mod(ItemMod {
                vis: Visibility::Public(_),
                ident,
                content: Some((_, mod_items)),
                ..
            }) => {
                // Inspect inner module items (e.g., `pub mod compiler { ... }`)
                for inner in mod_items {
                    if let Item::Use(ItemUse {
                        vis: Visibility::Public(_),
                        tree,
                        ..
                    }) = inner
                    {
                        let mut prefix = vec![ident.to_string()];
                        collect_use_paths(tree, &mut prefix, &mut exports);
                    }
                }
            }
            _ => {}
        }
    }

    exports
}

/// Validates that no exported path carries internal module segments or whole-crate aliases.
pub(crate) fn validate_facade_exports(exports: &[ExportedPath]) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    for exp in exports {
        let full = exp.full_path();

        // 1. Check for whole crate alias e.g. `pub use vyre_megakernel as compiler;`
        if exp.is_whole_crate_alias {
            errors.push(format!(
                "facade re-exports whole crate alias `{full}`; compiler and internal seams must be curated modules"
            ));
        }

        // 2. Check for wildcards
        if exp.is_glob {
            errors.push(format!(
                "facade re-exports wildcard `{full}`; exports must be explicit"
            ));
        }

        // 3. Check for forbidden internal module segments
        for seg in &exp.segments {
            for &forbidden in FORBIDDEN_INTERNAL_PATH_SEGMENTS {
                if seg == forbidden
                    || seg.starts_with(&format!("{forbidden}::"))
                    || seg.ends_with(&format!("::{forbidden}"))
                {
                    errors.push(format!(
                        "facade exposes internal `{forbidden}` symbol via re-export path `{full}`"
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[test]
fn facade_exposes_zero_internal_implementation_symbols() {
    let path = facade_source_path();
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read facade at {}: {e}", path.display()));
    let ast = syn::parse_file(&content).expect("parse vyre/src/lib.rs with syn");

    let exports = extract_public_exports_from_ast(&ast);
    assert!(!exports.is_empty(), "facade must have valid public exports");

    let result = validate_facade_exports(&exports);
    if let Err(errors) = result {
        panic!(
            "facade exports validation failed with {} error(s):\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
}

#[test]
fn facade_compiler_module_is_curated_not_whole_crate_alias() {
    let path = facade_source_path();
    let content = std::fs::read_to_string(&path).expect("read facade");
    let ast = syn::parse_file(&content).expect("parse facade AST");

    // Ensure `compiler` is defined as a curated `pub mod compiler`, not `pub use vyre_megakernel as compiler;`
    let has_curated_compiler_mod = ast.items.iter().any(|item| {
        if let Item::Mod(ItemMod {
            vis: Visibility::Public(_),
            ident,
            content: Some(_),
            ..
        }) = item
        {
            ident == "compiler"
        } else {
            false
        }
    });

    let has_whole_crate_compiler_alias = ast.items.iter().any(|item| {
        if let Item::Use(ItemUse {
            vis: Visibility::Public(_),
            tree,
            ..
        }) = item
        {
            if let UseTree::Rename(UseRename { ident, rename, .. }) = tree {
                ident == "vyre_megakernel" && rename == "compiler"
            } else {
                false
            }
        } else {
            false
        }
    });

    assert!(
        has_curated_compiler_mod,
        "facade must declare curated `pub mod compiler`"
    );
    assert!(
        !has_whole_crate_compiler_alias,
        "facade must NOT declare `pub use vyre_megakernel as compiler;`"
    );
}

#[test]
fn mutation_injecting_whole_crate_alias_is_caught() {
    let fake_source = "pub use vyre_megakernel as compiler;";
    let ast = syn::parse_file(fake_source).expect("parse fake source");
    let exports = extract_public_exports_from_ast(&ast);
    let result = validate_facade_exports(&exports);

    assert!(result.is_err(), "whole crate alias must fail validation");
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| e.contains("facade re-exports whole crate alias")));
}

#[test]
fn mutation_injecting_optimizer_reexport_is_caught() {
    let fake_source = "pub use vyre_foundation::optimizer::Pass;";
    let ast = syn::parse_file(fake_source).expect("parse fake source");
    let exports = extract_public_exports_from_ast(&ast);
    let result = validate_facade_exports(&exports);

    assert!(result.is_err(), "optimizer re-export must fail validation");
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| e.contains("facade exposes internal `optimizer` symbol")));
}

#[test]
fn mutation_injecting_search_reexport_is_caught() {
    let fake_source = "pub use vyre_megakernel::search::SearchSpace;";
    let ast = syn::parse_file(fake_source).expect("parse fake source");
    let exports = extract_public_exports_from_ast(&ast);
    let result = validate_facade_exports(&exports);

    assert!(result.is_err(), "search re-export must fail validation");
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| e.contains("facade exposes internal `search` symbol")));
}

#[test]
fn mutation_injecting_lowering_reexport_is_caught() {
    let fake_source = "pub use vyre_lower::lowering::PhysicalKernel;";
    let ast = syn::parse_file(fake_source).expect("parse fake source");
    let exports = extract_public_exports_from_ast(&ast);
    let result = validate_facade_exports(&exports);

    assert!(result.is_err(), "lowering re-export must fail validation");
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| e.contains("facade exposes internal `lower`") || e.contains("lowering")));
}

#[test]
fn mutation_injecting_runtime_worker_reexport_is_caught() {
    let fake_source = "pub use vyre_runtime::worker_process::WorkerPool;";
    let ast = syn::parse_file(fake_source).expect("parse fake source");
    let exports = extract_public_exports_from_ast(&ast);
    let result = validate_facade_exports(&exports);

    assert!(
        result.is_err(),
        "runtime worker re-export must fail validation"
    );
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| e.contains("facade exposes internal `worker_process` symbol")));
}
