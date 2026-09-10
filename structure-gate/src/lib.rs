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
//! from another file. `vyre-libs/src/bitset/mod.rs` passes
//! `op_id: "vyre-libs::bitset::xor"` to `define_bitwise_binary_op!`, defined in
//! `vyre-libs/src/bitset/binary_word.rs`, so the id never enters the model and
//! a collision against it goes unreported. Resolving it needs a
//! crate-wide pass pairing macro definitions with their invocation sites, which
//! a per-file parser cannot do. An id written inline or through a file-local
//! `const` is read, wherever in the file the `const` sits.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{LazyLock, Mutex, PoisonError};

pub mod backend_vocabulary;
pub mod cfg_test;
pub mod crate_ownership;
pub mod duplicate_type_name;
pub mod geometry_constants;
pub mod module_layout;
pub mod registration_macro;
pub mod registration_text;
pub mod source_scan;
pub mod workspace_manifest;
pub mod workspace_rules;

// Source scan and route discovery.
pub use duplicate_type_name::duplicate_public_type_name_failures;
pub use geometry_constants::geometry_constant_failures;
pub use source_scan::opaque_span;
pub use workspace_manifest::{
    crate_ident, discarding_imports, member_directory, read_lanes, submits_registrations,
    workspace_excludes, workspace_members, workspace_root, workspace_root_from, MAX_SOURCE_BYTES,
};
pub use workspace_rules::{
    category_home_failures, frontend_owner_failures, is_category_a_crate,
    materializer_admission_failures, operation_identity_failures, registration_owner_failures,
    registry_link_failures, roster_failures, substrate_home_failures, DiscardingImport,
    Registration, CATEGORY_A_CRATE, CATEGORY_C_CRATE,
};

use crate::workspace_manifest::{
    crate_source_roots, member_sources, read_source_bounded, relative, tree_files, MemberSource,
    SELF_CRATE,
};
use crate::workspace_rules::{crate_declares_frontend, FRONTEND_OWNERS};

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
    let sources = member_sources(root, &members);
    let registrations = scan_registrations(&sources);
    let substrate_paths = scan_substrate_paths(&sources);
    let frontend_paths = scan_frontend_paths(&sources, &members);
    let materializers = scan_materializers(&sources);
    let registry_submitters = scan_registry_submitters(&sources);
    let discarding_imports = scan_discarding_imports(&sources);
    let crate_roots = scan_crate_roots(root);
    let published_modules = scan_published_modules(root, &crate_roots);
    let source_files = scan_source_files(root, &crate_roots);
    let module_files = scan_module_files(&crate_roots, &source_files);
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

/// Every source file of every crate root in the checkout, checkout-relative.
///
/// The same roster [`Workspace::source_files`] carries, without the rosters
/// derived from source text. A reader that wants the file list and reads the
/// text itself pays one walk here instead of every walk [`scan`] makes.
#[must_use]
pub fn source_file_roster(root: &Path) -> Vec<String> {
    scan_source_files(root, &scan_crate_roots(root))
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
    failures.extend(duplicate_public_type_name_failures(
        root,
        &workspace.crate_roots,
    ));
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
fn scan_registrations(sources: &[MemberSource]) -> Vec<Registration> {
    let mut registrations = Vec::new();
    for source in sources {
        if is_test_source(&source.file) {
            continue;
        }
        let Ok(text) = &source.text else {
            continue;
        };
        for parsed in parse_registrations(text) {
            registrations.push(Registration {
                crate_name: source.crate_name.clone(),
                file: source.file.clone(),
                op_id: parsed.0,
                tier: parsed.1,
            });
        }
    }
    registrations
}

fn scan_substrate_paths(sources: &[MemberSource]) -> Vec<String> {
    let mut paths: Vec<String> = sources
        .iter()
        .filter(|source| {
            source
                .file
                .split('/')
                .any(|segment| segment.contains("substrate") && !segment.ends_with("_test.rs"))
        })
        .map(|source| source.file.clone())
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

/// Collect every frontend stage the workspace exposes.
///
/// A language-named stage directory is one signal; a crate whose name says it
/// is a frontend is the other, and it is emitted even when the crate keeps a
/// flat layout with no `lex/` or `preprocess/` directory at all, so the members
/// are read as well as their files.
fn scan_frontend_paths(sources: &[MemberSource], members: &[String]) -> Vec<(String, String)> {
    let mut paths = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member);
        if FRONTEND_OWNERS
            .iter()
            .any(|(language, _)| crate_declares_frontend(crate_name, language))
        {
            paths.push((crate_name.to_string(), member.clone()));
        }
    }
    for source in sources {
        if source.file.contains("/lex/") || source.file.contains("/preprocess/") {
            paths.push((source.crate_name.clone(), source.file.clone()));
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

/// Every crate the layout rules judge, ordered by directory.
///
/// Resolved once per process per root. Discovery reads every manifest in the
/// checkout, and every contract in a consolidated test binary calls [`scan`],
/// so the reads happen once for the process rather than once per contract.
fn scan_crate_roots(root: &Path) -> Vec<CrateRoot> {
    static RESOLVED: LazyLock<Mutex<BTreeMap<PathBuf, Vec<CrateRoot>>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));

    let mut cache = RESOLVED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(found) = cache.get(root) {
        return found.clone();
    }
    let mut roots = crate_source_roots(root);
    roots.sort_by(|left, right| left.directory.cmp(&right.directory));
    roots.dedup();
    cache.insert(root.to_path_buf(), roots.clone());
    roots
}

/// Every `src/` module file of every crate root, checkout-relative.
///
/// Derived from [`scan_source_files`] rather than walked again. `src` is one of
/// [`SOURCE_TREES`], so walking it a second time reports the same paths and
/// pays another pass over every crate root to do it.
fn scan_module_files(crate_roots: &[CrateRoot], source_files: &[String]) -> Vec<String> {
    let mut files: Vec<String> = crate_roots
        .iter()
        .flat_map(|crate_root| {
            let prefix = if crate_root.directory.is_empty() {
                "src/".to_string()
            } else {
                format!("{}/src/", crate_root.directory)
            };
            source_files
                .iter()
                .filter(move |file| file.starts_with(&prefix))
                .cloned()
        })
        .collect();
    files.sort();
    files.dedup();
    files
}

/// Every source file of every crate root, checkout-relative.
///
/// Wider than [`scan_module_files`] by the trees in [`SOURCE_TREES`] other than
/// `src`: the name rules judge a fixture module the same way they judge a
/// library module, because a reader looks for one the same way.
///
/// Derived from the one tree walk the process makes rather than walked per
/// crate root. The roster is a prefix range of that walk's sorted file list, so
/// forty crate roots cost forty binary searches instead of a hundred and sixty
/// directory walks over a network mount.
fn scan_source_files(root: &Path, crate_roots: &[CrateRoot]) -> Vec<String> {
    let tree = tree_files(root);
    let mut files = Vec::new();
    for crate_root in crate_roots {
        let directory = root.join(&crate_root.directory);
        for source_tree in SOURCE_TREES {
            for path in tree.rust_sources_under(&directory.join(source_tree)) {
                files.push(relative(root, path));
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
///
/// A facade publishes a domain module with `pub use`, not `pub mod`, so reading
/// `pub mod` alone left every module `vyre` and `vyre-libs` republish out of the
/// set, and a published module whose name the layout rules ban lost the
/// exemption that keeps a consumer's import path stable.
///
/// A snapshot line for a re-export states no item kind: `pub use
/// vyre_libs::parsing` re-exports a module and `pub use vyre::validate`
/// re-exports a function of that name, and the two render identically. The kind
/// comes from the crate that declares the item: the re-exporting crate's own
/// source states which path each re-exported name binds, and the declaring
/// crate's snapshot states whether that path is a `pub mod`. A re-export joins
/// the set only when both agree, so a re-exported type or function never
/// exempts a module of the same name.
fn scan_published_modules(root: &Path, crate_roots: &[CrateRoot]) -> Vec<String> {
    let mut declared: BTreeSet<String> = BTreeSet::new();
    let mut reexports: Vec<String> = Vec::new();
    let Ok(entries) = fs::read_dir(root.join(PUBLIC_API_SNAPSHOT_DIR)) else {
        return Vec::new();
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
                declared.insert(module.trim().to_string());
            } else if let Some(reexport) = line.strip_prefix("pub use ") {
                let reexport = reexport.trim();
                if reexport.contains("::") && !reexport.contains('<') {
                    reexports.push(reexport.to_string());
                }
            }
        }
    }
    let mut modules: Vec<String> = declared.iter().cloned().collect();
    modules.extend(reexported_modules(
        root,
        crate_roots,
        &declared,
        &reexports,
    ));
    modules.sort();
    modules.dedup();
    modules
}

/// Every path in `reexports` that a crate in `crate_roots` publishes a module at.
///
/// A candidate whose final name is the final name of no declared module cannot
/// resolve to one, so it costs no source read. What survives that is confirmed
/// against the re-exporting crate's source, which is the only place the target
/// path of a re-exported name is written.
///
/// Bindings are read once per re-exporting crate: the two facade crates carry
/// every candidate in this checkout, and one pass over each answers all of them.
fn reexported_modules(
    root: &Path,
    crate_roots: &[CrateRoot],
    declared: &BTreeSet<String>,
    reexports: &[String],
) -> Vec<String> {
    let names: BTreeSet<&str> = declared
        .iter()
        .filter_map(|module| module.rsplit("::").next())
        .collect();
    let directories: BTreeMap<&str, &str> = crate_roots
        .iter()
        .map(|crate_root| (crate_root.ident.as_str(), crate_root.directory.as_str()))
        .collect();
    let mut bindings: BTreeMap<&str, Vec<(String, String)>> = BTreeMap::new();
    let mut found = Vec::new();
    for path in reexports {
        let Some(name) = path.rsplit("::").next() else {
            continue;
        };
        if !names.contains(name) {
            continue;
        }
        let Some((ident, _)) = path.split_once("::") else {
            continue;
        };
        let Some(directory) = directories.get(ident) else {
            continue;
        };
        let bound = bindings
            .entry(ident)
            .or_insert_with(|| crate_reexport_bindings(root, directory));
        if bound
            .iter()
            .any(|(bound, target)| bound == name && declared.contains(target))
        {
            found.push(path.clone());
        }
    }
    found
}

/// `(bound name, re-exported path)` for every `pub use` under one crate's `src`.
///
/// A statement is accumulated across lines because a brace group is routinely
/// wrapped, and only a line that starts with `pub use` opens one, which keeps a
/// doc comment quoting a re-export out of the result.
fn crate_reexport_bindings(root: &Path, directory: &str) -> Vec<(String, String)> {
    let tree = tree_files(root);
    let mut found = Vec::new();
    for path in tree.rust_sources_under(&root.join(directory).join("src")) {
        let Ok(text) = read_source_bounded(path) else {
            continue;
        };
        let mut pending = String::new();
        let mut open = false;
        for line in text.lines() {
            let trimmed = line.trim();
            if open {
                pending.push(' ');
                pending.push_str(trimmed);
            } else {
                let Some(rest) = trimmed.strip_prefix("pub use ") else {
                    continue;
                };
                pending.clear();
                pending.push_str(rest);
                open = true;
            }
            if let Some(end) = pending.find(';') {
                expand_use_tree("", &pending[..end], &mut found);
                open = false;
            }
        }
    }
    found
}

/// Expand one `use` tree into the `(bound name, path)` pair of every leaf.
///
/// A glob binds no name and `_` discards the one it would bind, so neither
/// publishes a path.
fn expand_use_tree(prefix: &str, spec: &str, out: &mut Vec<(String, String)>) {
    let spec = spec.trim();
    if spec.is_empty() {
        return;
    }
    if let Some(open) = spec.find('{') {
        let inner = &spec[open + 1..];
        let Some(close) = closing_brace(inner) else {
            return;
        };
        let head = join_use_path(prefix, &spec[..open]);
        for member in split_use_members(&inner[..close]) {
            expand_use_tree(&head, member, out);
        }
        return;
    }
    let (target, alias) = match spec.split_once(" as ") {
        Some((target, alias)) => (target.trim(), alias.trim()),
        None => (spec, ""),
    };
    if target.ends_with('*') {
        return;
    }
    let path = join_use_path(prefix, target);
    let name = if alias.is_empty() {
        path.rsplit("::").next().unwrap_or_default().to_string()
    } else {
        alias.to_string()
    };
    if name.is_empty() || name == "_" || path.is_empty() {
        return;
    }
    out.push((name, path));
}

/// Join one `use` tree segment onto the prefix that holds it.
///
/// `self` names the prefix itself, which is how `a::b::{self, c}` publishes `b`.
fn join_use_path(prefix: &str, tail: &str) -> String {
    let tail = tail.trim().trim_start_matches(':').trim_end_matches(':');
    let prefix = prefix.trim_end_matches(':');
    if tail.is_empty() || tail == "self" {
        return prefix.to_string();
    }
    if prefix.is_empty() {
        return tail.to_string();
    }
    format!("{prefix}::{tail}")
}

/// Byte offset of the `}` closing an already-consumed `{`.
fn closing_brace(inner: &str) -> Option<usize> {
    let mut depth = 0_usize;
    for (offset, byte) in inner.bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' if depth == 0 => return Some(offset),
            b'}' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Split one brace group at the commas that separate its own members.
fn split_use_members(body: &str) -> Vec<&str> {
    let mut members = Vec::new();
    let mut depth = 0_usize;
    let mut start = 0_usize;
    for (offset, byte) in body.bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                members.push(&body[start..offset]);
                start = offset + 1;
            }
            _ => {}
        }
    }
    members.push(&body[start..]);
    members
}

/// Read every concrete backend materializer source.
///
/// Only `vyre-driver-*` members are scanned. `vyre-driver` itself is the owner
/// of the shared admission helpers, so finding their definitions there is the
/// rule being satisfied, not broken.
fn scan_materializers(sources: &[MemberSource]) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for source in sources {
        if !source.crate_name.starts_with("vyre-driver-") {
            continue;
        }
        if source.file.rsplit('/').next() != Some("materializer.rs") {
            continue;
        }
        let Ok(text) = &source.text else {
            continue;
        };
        found.push((source.file.clone(), text.clone()));
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
/// in a first pass over the same corpus, because a crate that only invokes one
/// still needs its registrations linked.
fn scan_registry_submitters(sources: &[MemberSource]) -> Vec<String> {
    let mut definitions: BTreeMap<String, String> = BTreeMap::new();
    for source in sources {
        let Ok(text) = &source.text else {
            continue;
        };
        definitions.extend(macro_definitions(text));
    }
    let submitting = submitting_macros(&definitions);

    let mut submitters = Vec::new();
    for source in sources {
        let Ok(text) = &source.text else {
            continue;
        };
        if submits_registrations(text, &submitting) {
            submitters.push(source.crate_name.clone());
        }
    }
    submitters.sort();
    submitters.dedup();
    submitters
}

/// Every `use <crate> as _;` in member sources.
fn scan_discarding_imports(sources: &[MemberSource]) -> Vec<DiscardingImport> {
    let mut imports = Vec::new();
    for source in sources {
        let Ok(text) = &source.text else {
            continue;
        };
        for named in discarding_imports(text) {
            imports.push(DiscardingImport {
                file: source.file.clone(),
                named,
            });
        }
    }
    imports.sort_by(|left, right| (&left.file, &left.named).cmp(&(&right.file, &right.named)));
    imports.dedup();
    imports
}
