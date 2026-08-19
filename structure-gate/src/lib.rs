//! Workspace structural gate: crate roster, one operation identity per
//! semantic operation, one home per concept, and one place per module.
//!
//! Run it with `./cargo_full run -p structure-gate`. It reads source text and depends
//! on no vyre crate, so it still judges the workspace while the workspace does
//! not compile.
//!
//! The workspace has exactly two operation-owning crates:
//!
//! - `vyre-libs` owns **every Category A operation**: any composition built
//!   from existing IR variants.
//! - `vyre-primitives` owns **every Category C operation**: an operation that
//!   needs a dedicated hardware contract.
//!
//! Any third crate that registers an operation splits an operation identity in
//! two, which is the defect this gate exists to make impossible. A tier
//! boundary is not a reason to re-register a kernel under a second id.
//!
//! Re-verifying a change to this gate means running `./cargo_full test -p
//! structure-gate`, which rebuilds first. The contract tests read the live tree
//! when they run but carry their rules from when they were built, and a target
//! directory shared by several checkouts used to hand this crate a binary built
//! from a different tree: the gate then answered today's tree with another
//! checkout's rules. `VYRE_CHECKOUT_ROOT` in `.cargo/config.toml` is now a
//! fingerprint input of this crate, so cargo rebuilds instead of reusing across
//! checkouts, and `checkout_provenance.rs` fails if that ever stops holding.
//!
//! What this scan does not see: an operation id handed to a `macro_rules!`
//! parameter and registered inside the macro body, when the macro is invoked
//! from another file. `vyre-primitives/src/bitset/mod.rs` passes
//! `op_id: "vyre-primitives::bitset::xor"` to a macro defined in
//! `bitset/binary_word.rs`, and `vyre-libs/src/logical/mod.rs` registers
//! `vyre-libs::logical::xor` through the same shape, so neither id enters the
//! model and that identity collision goes unreported. Resolving it needs a
//! crate-wide pass pairing macro definitions with their invocation sites, which
//! a per-file parser cannot do. An id written inline or through a file-local
//! `const` is read, wherever in the file the `const` sits.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process;

pub mod backend_vocabulary;
pub mod cfg_test;
pub mod crate_ownership;
pub mod geometry_constants;
pub mod module_layout;
pub mod registration_macro;
pub mod registration_text;
pub mod source_scan;
pub mod workspace_manifest;
pub mod workspace_rules;

// Source scan and route discovery.
pub use geometry_constants::geometry_constant_failures;
pub use source_scan::opaque_span;
pub use workspace_manifest::{
    crate_ident, discarding_imports, member_directory, submits_registrations, workspace_excludes,
    workspace_members, workspace_root, workspace_root_from, MAX_SOURCE_BYTES,
};
pub use workspace_rules::{
    category_home_failures, frontend_owner_failures, materializer_admission_failures,
    operation_identity_failures, registration_owner_failures, registry_link_failures,
    roster_failures, substrate_home_failures, DiscardingImport, Registration,
};

use crate::workspace_manifest::{
    crate_source_roots, read_source_bounded, relative, source_files, source_tree_files, SELF_CRATE,
};
use crate::workspace_rules::{
    crate_declares_frontend, CATEGORY_A_CRATE, CATEGORY_C_CRATE, FRONTEND_OWNERS,
};

use crate::backend_vocabulary::is_test_source;
use crate::module_layout::{
    directory_stutter_failures, generic_module_name_failures, numbered_sibling_failures,
    sibling_module_failures, source_test_directory_failures, CrateRoot, PUBLIC_API_SNAPSHOT_DIR,
    SOURCE_TREES,
};
use crate::registration_macro::{macro_definitions, submitting_macros};
use crate::registration_text::parse_registrations;

/// Everything the structural rules judge, read once from source text.
#[derive(Clone, Debug, Default)]
pub struct Workspace {
    /// Workspace member paths from the root manifest.
    pub members: Vec<String>,
    /// Every `OperationRegistration` found in member sources.
    pub registrations: Vec<Registration>,
    /// Source paths naming the substrate concept.
    pub substrate_paths: Vec<String>,
    /// `(crate, path)` for every source-language frontend stage found.
    pub frontend_paths: Vec<(String, String)>,
    /// `(path, source text)` for every concrete backend materializer.
    pub materializers: Vec<(String, String)>,
    /// Member crates that submit into an `inventory` registry.
    pub registry_submitters: Vec<String>,
    /// Every `use <crate> as _;` found in member sources.
    pub discarding_imports: Vec<DiscardingImport>,
    /// Crates the layout rules judge.
    pub crate_roots: Vec<CrateRoot>,
    /// Every `src/` module file of every crate root, checkout-relative.
    pub module_files: Vec<String>,
    /// Every source file of every crate root, checkout-relative.
    pub source_files: Vec<String>,
    /// Module paths the committed public-API snapshots publish.
    pub published_modules: Vec<String>,
    /// Every glob re-export of another workspace crate, as file, owning crate
    /// identifier, line and re-exported path.
    pub foreign_glob_reexports: Vec<(String, String, u32, String)>,
}

/// Read the workspace rooted at `root` into the structural model.
#[must_use]
pub fn scan(root: &Path) -> Workspace {
    let members = workspace_members(root);
    let registrations = scan_registrations(root, &members);
    let substrate_paths = scan_substrate_paths(root, &members);
    let frontend_paths = scan_frontend_paths(root, &members);
    let materializers = scan_materializers(root, &members);
    let registry_submitters = scan_registry_submitters(root, &members);
    let discarding_imports = scan_discarding_imports(root, &members);
    let crate_roots = scan_crate_roots(root);
    let module_files = scan_module_files(root, &crate_roots);
    let published_modules = scan_published_modules(root);
    let source_files = scan_source_files(root, &crate_roots);
    let foreign_glob_reexports =
        backend_vocabulary::scan_foreign_glob_reexports(root, &crate_roots);
    Workspace {
        members,
        registrations,
        substrate_paths,
        frontend_paths,
        materializers,
        registry_submitters,
        discarding_imports,
        crate_roots,
        module_files,
        source_files,
        published_modules,
        foreign_glob_reexports,
    }
}

/// Collect every structural violation in the workspace rooted at `root`.
///
/// `run` and the workspace contract tests share this one path so the gate and
/// its proving tests can never disagree about what the rule is.
///
/// The neutral-vocabulary rule takes `root` rather than the model: it streams the
/// production text of the crates in its roster and keeps only what matched, so
/// reading every one of those lines into [`Workspace`] first would hold most of
/// the workspace in memory to report a handful of lines.
#[must_use]
pub fn violations(root: &Path) -> Vec<String> {
    let workspace = scan(root);
    let mut failures = Vec::new();
    failures.extend(roster_failures(&workspace.members));
    failures.extend(registration_owner_failures(&workspace.registrations));
    failures.extend(operation_identity_failures(
        &workspace.registrations,
        &workspace.members,
    ));
    failures.extend(category_home_failures(&workspace.registrations));
    failures.extend(substrate_home_failures(&workspace.substrate_paths));
    failures.extend(frontend_owner_failures(&workspace.frontend_paths));
    failures.extend(materializer_admission_failures(&workspace.materializers));
    failures.extend(registry_link_failures(
        &workspace.registry_submitters,
        &workspace.discarding_imports,
    ));
    failures.extend(sibling_module_failures(&workspace.source_files));
    failures.extend(source_test_directory_failures(&workspace.module_files));
    failures.extend(generic_module_name_failures(
        &workspace.source_files,
        &workspace.crate_roots,
        &workspace.published_modules,
    ));
    failures.extend(numbered_sibling_failures(&workspace.source_files));
    failures.extend(directory_stutter_failures(&workspace.source_files));
    failures.extend(backend_vocabulary::foreign_glob_reexport_failures(
        &workspace.foreign_glob_reexports,
    ));
    failures.extend(backend_vocabulary::neutral_vocabulary_failures(root));
    failures.extend(geometry_constant_failures(root));
    failures
}

/// Run the crate-structure gate.
pub fn run(args: &[String]) {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "USAGE:\n  cargo run -p structure-gate\n\n\
             Fails when a crate outside vyre-foundation (Category A) or vyre-libs \
             (Category C) registers an operation, when one semantic operation is \
             registered under two identities, when a concept has more than one home, \
             when a src/ module file sits beside a directory of its own name, when a \
             module name states no contract, when a crate glob re-exports another \
             crate's items, when production source of a substrate-neutral crate names \
             a concrete backend, or when the workspace roster drifts."
        );
        return;
    }

    let root = workspace_root();
    let failures = violations(&root);

    if failures.is_empty() {
        println!(
            "crate-structure: roster, operation identity, concept homes, module layout, and neutral vocabulary agree"
        );
        return;
    }

    eprintln!("crate-structure: {} violation(s):", failures.len());
    for failure in &failures {
        eprintln!("  - {failure}");
    }
    eprintln!(
        "Fix: move the operation to its category owner ({CATEGORY_A_CRATE} for Category A, \
         {CATEGORY_C_CRATE} for Category C), delete the duplicate registration, move a \
         module file that sits beside its own directory to that directory's mod.rs, rename \
         a module that states no contract for what it holds, name the owner instead of \
         re-exporting it, state the neutral concept instead of one vendor's product, and \
         update docs/ARCHITECTURE.md plus docs/CRATE_OWNERSHIP.toml in the same change."
    );
    process::exit(1);
}

/// Every production operation registration the workspace declares.
///
/// A file the tree reaches only as test support is skipped: an integration test
/// registers fixture operations to drive the registry it is testing, and those
/// ids exist in no shipped binary. Judging them as production registrations
/// convicted a driver test of owning an operation and asked it to move its
/// fixture into a category crate, where it would then ship. The `#[cfg(test)]`
/// case is already stripped per file by [`parse_registrations`]; this is the
/// other half, where the gate sits in the parent directory instead of an
/// attribute.
fn scan_registrations(root: &Path, members: &[String]) -> Vec<Registration> {
    let mut registrations = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member).to_string();
        for path in source_files(root, member) {
            let file = relative(root, &path);
            if is_test_source(&file) {
                continue;
            }
            let Ok(text) = read_source_bounded(&path) else {
                continue;
            };
            for parsed in parse_registrations(&text) {
                registrations.push(Registration {
                    crate_name: crate_name.clone(),
                    file: file.clone(),
                    op_id: parsed.0,
                    tier: parsed.1,
                });
            }
        }
    }
    registrations
}

fn scan_substrate_paths(root: &Path, members: &[String]) -> Vec<String> {
    let mut paths = Vec::new();
    for member in members {
        for path in source_files(root, member) {
            let rel = relative(root, &path);
            let names_substrate = rel
                .split('/')
                .any(|segment| segment.contains("substrate") && !segment.ends_with("_test.rs"));
            if names_substrate {
                paths.push(rel);
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

/// Collect every frontend stage the workspace exposes.
///
/// A language-named stage directory is one signal; a crate whose name says it
/// is a frontend is the other, and it is emitted even when the crate keeps a
/// flat layout with no `lex/` or `preprocess/` directory at all.
fn scan_frontend_paths(root: &Path, members: &[String]) -> Vec<(String, String)> {
    let mut paths = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member).to_string();
        if FRONTEND_OWNERS
            .iter()
            .any(|(language, _)| crate_declares_frontend(&crate_name, language))
        {
            paths.push((crate_name.clone(), member.clone()));
        }
        for path in source_files(root, member) {
            let rel = relative(root, &path);
            if rel.contains("/lex/") || rel.contains("/preprocess/") {
                paths.push((crate_name.clone(), rel));
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

/// Every crate the layout rules judge, ordered by directory.
fn scan_crate_roots(root: &Path) -> Vec<CrateRoot> {
    let mut roots = crate_source_roots(root);
    roots.sort_by(|left, right| left.directory.cmp(&right.directory));
    roots.dedup();
    roots
}

/// Every `src/` module file of every crate root, checkout-relative.
fn scan_module_files(root: &Path, crate_roots: &[CrateRoot]) -> Vec<String> {
    let mut files = Vec::new();
    for crate_root in crate_roots {
        for path in source_tree_files(&root.join(&crate_root.directory).join("src")) {
            files.push(relative(root, &path));
        }
    }
    files.sort();
    files.dedup();
    files
}

/// Every source file of every crate root, checkout-relative.
///
/// Wider than [`scan_module_files`] by the trees in [`SOURCE_TREES`] other than
/// `src`: the name rules judge a fixture module the same way they judge a
/// library module, because a reader looks for one the same way.
fn scan_source_files(root: &Path, crate_roots: &[CrateRoot]) -> Vec<String> {
    let mut files = Vec::new();
    for crate_root in crate_roots {
        for tree in SOURCE_TREES {
            for path in source_tree_files(&root.join(&crate_root.directory).join(tree)) {
                files.push(relative(root, &path));
            }
        }
    }
    files.sort();
    files.dedup();
    files
}

/// Every module path the committed public-API snapshots publish.
///
/// An unreadable snapshot directory yields no exemptions rather than a blanket
/// one: losing the record of what is published makes the layout rules louder,
/// not quieter.
fn scan_published_modules(root: &Path) -> Vec<String> {
    let mut modules = Vec::new();
    let Ok(entries) = fs::read_dir(root.join(PUBLIC_API_SNAPSHOT_DIR)) else {
        return modules;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.extension().is_some_and(|ext| ext == "txt") {
            continue;
        }
        let Ok(text) = read_source_bounded(&path) else {
            continue;
        };
        for line in text.lines() {
            if let Some(module) = line.strip_prefix("pub mod ") {
                modules.push(module.trim().to_string());
            }
        }
    }
    modules.sort();
    modules.dedup();
    modules
}

/// Read every concrete backend materializer source.
///
/// Only `vyre-driver-*` members are scanned. `vyre-driver` itself is the owner
/// of the shared admission helpers, so finding their definitions there is the
/// rule being satisfied, not broken.
fn scan_materializers(root: &Path, members: &[String]) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member);
        if !crate_name.starts_with("vyre-driver-") {
            continue;
        }
        for path in source_files(root, member) {
            if path
                .file_name()
                .is_some_and(|name| name == "materializer.rs")
            {
                let Ok(text) = read_source_bounded(&path) else {
                    continue;
                };
                found.push((relative(root, &path), text));
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

/// Every member crate that submits into an `inventory` registry.
///
/// Read from the tree rather than listed here: a new submitting crate joins the
/// set the moment it submits, so the link rule judges it without an edit. The
/// macros that submit on a caller's behalf are read from the tree the same way,
/// in a first pass over the same files, because a crate that only invokes one
/// still needs its registrations linked.
fn scan_registry_submitters(root: &Path, members: &[String]) -> Vec<String> {
    let mut definitions: BTreeMap<String, String> = BTreeMap::new();
    for member in members {
        for path in source_files(root, member) {
            let Ok(text) = read_source_bounded(&path) else {
                continue;
            };
            definitions.extend(macro_definitions(&text));
        }
    }
    let submitting = submitting_macros(&definitions);

    let mut submitters = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member);
        for path in source_files(root, member) {
            let Ok(text) = read_source_bounded(&path) else {
                continue;
            };
            if submits_registrations(&text, &submitting) {
                submitters.push(crate_name.to_string());
                break;
            }
        }
    }
    submitters.sort();
    submitters.dedup();
    submitters
}

/// Every `use <crate> as _;` in member sources.
fn scan_discarding_imports(root: &Path, members: &[String]) -> Vec<DiscardingImport> {
    let mut imports = Vec::new();
    for member in members {
        for path in source_files(root, member) {
            let Ok(text) = read_source_bounded(&path) else {
                continue;
            };
            let file = relative(root, &path);
            for named in discarding_imports(&text) {
                imports.push(DiscardingImport {
                    file: file.clone(),
                    named,
                });
            }
        }
    }
    imports.sort_by(|left, right| (&left.file, &left.named).cmp(&(&right.file, &right.named)));
    imports.dedup();
    imports
}
