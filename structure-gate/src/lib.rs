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


use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use toml::Value;
use walkdir::WalkDir;

pub mod source_scan;

/// Category A owner: every composition, meaning anything that returns a
/// `Program` built from existing IR, including compiler-internal domains such
/// as solvers, encoding, analysis, scheduling, device and graph dispatch. Who
/// calls it does not move it; only rewriting it in host Rust does.
const CATEGORY_A_CRATE: &str = "vyre-libs";
/// Category C owner: strict hardware intrinsics, one emitter arm and one
/// reference-interpreter arm each. Absorbed the former standalone hardware
/// crate on 2026-08-13; the intrinsics live in `vyre-primitives/src/hardware`.
const CATEGORY_C_CRATE: &str = "vyre-primitives";

/// Directory that owns every module named `*substrate*`.
///
/// `vyre_foundation::pass_substrate` owns the CPU pass math outright: the pass
/// engine imports those functions and wraps them in dispatch rather than
/// reimplementing them. Renaming the pass-engine crate retired the second and
/// third homes for the name, so foundation is the only one left and the
/// exemption list it used to need is gone.
const SUBSTRATE_HOME: &str = "vyre-foundation/src/pass_substrate";

/// Closed workspace roster. A new member is a reviewable change here first.
const ALLOWED_MEMBERS: &[&str] = &[
    "conform/vyre-conform",
    "conform/vyre-conform-spec",
    "vyre",
    "vyre-aot",
    // Sole owner of the registry link anchors: it names every crate that submits
    // into an inventory registry so no consumer has to.
    "vyre-registry-link",
    "vyre-bench",
    "vyre-debug",
    "vyre-driver",
    "vyre-driver-cuda",
    "vyre-driver-metal",
    "vyre-driver-reference",
    "vyre-driver-spirv",
    "vyre-driver-wgpu",
    "vyre-emit-metal",
    "vyre-emit-naga",
    "vyre-emit-ptx",
    "vyre-emit-spirv",
    "vyre-foundation",
    "vyre-libs",
    "vyre-lints",
    "vyre-lower",
    "vyre-macros",
    "vyre-megakernel",
    "vyre-primitives",
    "vyre-reference",
    "vyre-runtime",
    "vyre-safetensors",
    // Narrowed to the optimizer pass engine and renamed with that narrowing.
    "vyre-pass-engine",
    "vyre-spec",
    "vyre-test-support",
    "structure-gate",
    "xtask",
    // The xtask subcommands that link vyre. Split out so a source edit no
    // longer rebuilds the compiler before a text-reading gate can run.
    "xtask-evidence",
    "xtask-registry",
];

/// Source languages and the single crate that owns each frontend.
///
/// A source frontend is a pile of Category A compositions: it parses with
/// vyre operations, so `vyre-libs` owns it like any other composition. A
/// separate CPU pipeline over the same language is the second frontend this
/// rule exists to reject. The tree-sitter C shell that used to be one left
/// the workspace as its own product rather than growing here.
///
/// The rust owner ships outside this workspace, so no member matches it and
/// every workspace crate that grows rust frontend stages is a second frontend.
/// Dropping the row instead would stop judging the language altogether.
const FRONTEND_OWNERS: &[(&str, &str)] = &[("c", "vyre-libs"), ("rust", "vyre-frontend-rust")];

/// One registered operation, as read from source text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Registration {
    /// Workspace member that submits the registration.
    pub crate_name: String,
    /// Workspace-relative source file.
    pub file: String,
    /// Resolved operation id, e.g. `vyre-foundation::hash::adler32`.
    pub op_id: String,
    /// `OperationTier` variant named in the registration, when present.
    pub tier: Option<String>,
}

impl Registration {
    /// Crate namespace the operation id claims, e.g. `vyre-foundation`.
    fn claimed_crate(&self) -> &str {
        self.op_id.split("::").next().unwrap_or(&self.op_id)
    }

    /// Trailing segment of the operation id, e.g. `adler32`.
    fn leaf(&self) -> &str {
        self.op_id.rsplit("::").next().unwrap_or(&self.op_id)
    }
}

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
    }
}

/// Collect every structural violation in the workspace rooted at `root`.
///
/// `run` and the workspace contract tests share this one path so the gate and
/// its proving tests can never disagree about what the rule is.
#[must_use]
pub fn violations(root: &Path) -> Vec<String> {
    let workspace = scan(root);
    let mut failures = Vec::new();
    failures.extend(roster_failures(&workspace.members));
    failures.extend(registration_owner_failures(&workspace.registrations));
    failures.extend(operation_identity_failures(&workspace.registrations));
    failures.extend(category_home_failures(&workspace.registrations));
    failures.extend(substrate_home_failures(&workspace.substrate_paths));
    failures.extend(frontend_owner_failures(&workspace.frontend_paths));
    failures.extend(materializer_admission_failures(&workspace.materializers));
    failures.extend(registry_link_failures(
        &workspace.registry_submitters,
        &workspace.discarding_imports,
    ));
    failures.extend(sibling_module_failures(&workspace.module_files));
    failures.extend(generic_module_name_failures(
        &workspace.source_files,
        &workspace.crate_roots,
        &workspace.published_modules,
    ));
    failures.extend(numbered_sibling_failures(&workspace.source_files));
    failures.extend(directory_stutter_failures(&workspace.source_files));
    failures
}

/// Workspace root, resolved from the directory the gate was invoked in.
///
/// Never compiled in. A target directory shared by several checkouts computes the
/// same unit hash for all of them, so cargo hands one checkout a binary another
/// one built; a path baked into that binary then names the wrong tree, and
/// `VYRE_CHECKOUT_ROOT` did not prevent it, because cargo does not export a
/// `relative = true` config variable to the process it runs. The tree the
/// operator invoked cargo in is the tree the gate must answer for.
///
/// # Panics
///
/// Panics when no ancestor of the working directory declares a `[workspace]`.
#[must_use]
pub fn workspace_root() -> PathBuf {
    let start = std::env::current_dir()
        .expect("Fix: the working directory must be readable to locate the vyre checkout");
    workspace_root_from(&start).unwrap_or_else(|| {
        panic!(
            "Fix: run this from inside the vyre checkout; no ancestor of `{}` has a Cargo.toml \
             declaring [workspace].",
            start.display()
        )
    })
}

/// The nearest ancestor of `start`, inclusive, whose manifest declares a workspace.
#[must_use]
pub fn workspace_root_from(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|directory| {
            read_source_bounded(&directory.join("Cargo.toml")).is_ok_and(|text| {
                text.lines()
                    .any(|line| line.trim_start().starts_with("[workspace]"))
            })
        })
        .map(Path::to_path_buf)
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
             module name states no contract, or when the workspace roster drifts."
        );
        return;
    }

    let root = workspace_root();
    let failures = violations(&root);

    if failures.is_empty() {
        println!(
            "crate-structure: roster, operation identity, concept homes, and module layout agree"
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
         a module that states no contract for what it holds, and update \
         docs/ARCHITECTURE.md plus docs/CRATE_OWNERSHIP.toml in the same change."
    );
    process::exit(1);
}

/// Reject workspace members outside the reviewed roster.
pub fn roster_failures(members: &[String]) -> Vec<String> {
    let mut failures = Vec::new();
    for member in members {
        if !ALLOWED_MEMBERS.contains(&member.as_str()) {
            failures.push(format!(
                "workspace member `{member}` is not on the reviewed roster; a product crate belongs outside this workspace, and a new platform crate is added to ALLOWED_MEMBERS in the same change"
            ));
        }
    }
    for allowed in ALLOWED_MEMBERS {
        if !members.iter().any(|member| member == allowed) {
            failures.push(format!(
                "roster lists `{allowed}` but the workspace does not contain it; delete the stale roster entry"
            ));
        }
    }
    failures
}

/// Reject operation registrations outside the two category owners.
pub fn registration_owner_failures(registrations: &[Registration]) -> Vec<String> {
    registrations
        .iter()
        .filter(|reg| reg.crate_name != CATEGORY_A_CRATE && reg.crate_name != CATEGORY_C_CRATE)
        .map(|reg| {
            format!(
                "{} registers `{}`; only {CATEGORY_A_CRATE} (Category A) and {CATEGORY_C_CRATE} (Category C) own operations",
                reg.file, reg.op_id
            )
        })
        .collect()
}

/// Reject one semantic operation carrying two identities, and ids that claim a
/// namespace their owning crate does not have.
pub fn operation_identity_failures(registrations: &[Registration]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut by_leaf: BTreeMap<&str, Vec<&Registration>> = BTreeMap::new();
    for reg in registrations {
        by_leaf.entry(reg.leaf()).or_default().push(reg);
    }
    for (leaf, regs) in by_leaf {
        let mut namespaces: Vec<&str> = regs.iter().map(|reg| reg.claimed_crate()).collect();
        namespaces.sort_unstable();
        namespaces.dedup();
        if namespaces.len() > 1 {
            let ids: Vec<&str> = regs.iter().map(|reg| reg.op_id.as_str()).collect();
            failures.push(format!(
                "operation `{leaf}` is registered under {} identities ({}); one kernel gets one id, and the higher layer calls it instead of re-registering it",
                ids.len(),
                ids.join(", ")
            ));
        }
    }
    for reg in registrations {
        if reg.claimed_crate() != reg.crate_name {
            failures.push(format!(
                "{} registers `{}` but lives in {}; the operation id namespace names its owning crate",
                reg.file, reg.op_id, reg.crate_name
            ));
        }
    }
    failures
}

/// Reject a Category A operation in the Category C crate and the reverse.
pub fn category_home_failures(registrations: &[Registration]) -> Vec<String> {
    let mut failures = Vec::new();
    for reg in registrations {
        let Some(tier) = reg.tier.as_deref() else {
            continue;
        };
        let hardware = matches!(tier, "Intrinsic" | "Hardware");
        if hardware && reg.crate_name == CATEGORY_A_CRATE {
            failures.push(format!(
                "{} registers Category C `{}` in {CATEGORY_A_CRATE}; hardware-contract operations live in {CATEGORY_C_CRATE}",
                reg.file, reg.op_id
            ));
        }
        if !hardware && reg.crate_name == CATEGORY_C_CRATE {
            failures.push(format!(
                "{} registers Category A `{}` in {CATEGORY_C_CRATE}; a composition over existing IR variants lives in {CATEGORY_A_CRATE}",
                reg.file, reg.op_id
            ));
        }
    }
    failures
}

/// Reject a second home for the substrate concept.
///
/// `vyre-foundation` owns the name. A type, trait, or module that restates it
/// anywhere else is a second definition of one concept, and the two drift
/// silently because nothing compares them.
pub fn substrate_home_failures(paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| !path.starts_with(SUBSTRATE_HOME))
        .map(|path| {
            format!(
                "`{path}` names the substrate concept outside {SUBSTRATE_HOME}; one concept gets one home"
            )
        })
        .collect()
}

/// Reject two crates owning a frontend for one source language.
///
/// A frontend announces itself two ways: a language-named stage directory
/// (`.../c/lex/`), or a crate name that says so (`vyre-frontend-c`). Keying
/// only on the directory missed a flat crate whose whole job was the second
/// frontend, which is the shape this rule exists to catch.
pub fn frontend_owner_failures(paths: &[(String, String)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (language, owner) in FRONTEND_OWNERS {
        for (found_crate, path) in paths {
            if found_crate == owner {
                continue;
            }
            if crate_declares_frontend(found_crate, language) {
                failures.push(format!(
                    "`{found_crate}` is a second {language} frontend crate; {owner} owns the {language} frontend"
                ));
            } else if path_names_language(path, language) {
                failures.push(format!(
                    "`{path}` puts a {language} frontend in {found_crate}; {owner} owns the {language} frontend"
                ));
            }
        }
    }
    failures.sort();
    failures.dedup();
    failures
}

/// True when a crate name declares itself the frontend for `language`.
fn crate_declares_frontend(crate_name: &str, language: &str) -> bool {
    let mut names_frontend = false;
    let mut names_language = false;
    for token in crate_name.split(['-', '_']) {
        names_frontend |= token.eq_ignore_ascii_case("frontend");
        names_language |= token.eq_ignore_ascii_case(language);
    }
    names_frontend && names_language
}

/// Trees a crate compiles source from, judged by the name rules.
///
/// `benches` and `examples` are in for the same reason `tests` is: a reader
/// looking for the fixture a benchmark uses reads its file name first.
const SOURCE_TREES: &[&str] = &["src", "tests", "benches", "examples"];

/// Names that state no contract.
///
/// A file called `helpers`, `types` or `utils` says nothing about what is
/// inside it, so finding the thing it holds means opening it, and deciding
/// where a new item goes means giving up and adding it there. Every name here
/// has that property; a name that states its contents does not.
///
/// The same word as a suffix is the same dumping ground with a qualifier
/// bolted on: `foo_ext` is whatever `foo` had no room for, and `spec_types` is
/// whatever the spec needed a home for. `is_banned_module_name` derives the
/// suffix family from this list so the two cannot drift apart.
const BANNED_MODULE_NAMES: &[&str] = &[
    "base", "common", "core", "ext", "extra", "glue", "helper", "helpers", "impl", "inner", "misc",
    "shared", "shim", "stuff", "support", "things", "types", "util", "utils", "wrapper",
];

/// Committed snapshot of the API each publishable crate reaches out with.
const PUBLIC_API_SNAPSHOT_DIR: &str = "docs/public-api";

/// One crate the module-layout rules judge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrateRoot {
    /// Checkout-relative crate directory.
    pub directory: String,
    /// Identifier a consumer writes to name this crate.
    ///
    /// Read from the manifest, `[lib] name` first and the package name second,
    /// never from the directory. A crate whose directory does not spell its
    /// package name would otherwise be looked up in the public-API snapshot
    /// under a name the snapshot never uses, and every published module of that
    /// crate would lose the exemption that keeps a published name stable.
    pub ident: String,
}

/// Reject a module file that sits beside a directory of the same name.
///
/// `foo.rs` next to `foo/` splits one module across two places in a listing:
/// the file declaring the module's own items sorts away from the directory
/// holding its children, so a reader looking for `foo` opens whichever the
/// editor shows first and finds half of it. `foo/mod.rs` is the same module
/// with the file and its children in one place.
///
/// The rule reads `src/` only. An integration test binary is named by its own
/// file, so `tests/foo.rs` beside `tests/foo/` is a binary next to its
/// fixtures rather than one module in two places.
#[must_use]
pub fn sibling_module_failures(module_files: &[String]) -> Vec<String> {
    let mut directories: BTreeSet<&str> = BTreeSet::new();
    for file in module_files {
        let mut rest = file.as_str();
        while let Some((parent, _)) = rest.rsplit_once('/') {
            if !directories.insert(parent) {
                break;
            }
            rest = parent;
        }
    }
    let mut failures: Vec<String> = module_files
        .iter()
        .filter_map(|file| {
            let stem = file.strip_suffix(".rs")?;
            directories.contains(stem).then(|| {
                format!(
                    "`{file}` sits beside its own directory `{stem}/`; one module is one place, so it belongs at `{stem}/mod.rs`"
                )
            })
        })
        .collect();
    failures.sort();
    failures.dedup();
    failures
}

/// Reject a file, module or binary whose name states no contract.
///
/// Judged over every source tree a crate compiles, not `src/` alone. The
/// prohibition was written for modules and went unenforced against
/// test-adjacent files, which is where the population moved: at the last count
/// 15 of 16 remaining banned names were `tests/common/mod.rs` or
/// `tests/support/mod.rs`.
///
/// A module is exempt only while the committed public-API snapshot publishes
/// it: renaming a published module renames a path a consumer already imports,
/// and this gate is not what decides to break one. The exemption is read from
/// the snapshot at run time, so it lapses by itself once the module stops
/// being published, and a crate with no snapshot cannot claim it at all. A file
/// outside `src/` has no public path and no exemption.
///
/// A Cargo binary root is judged by the binary's name, which is the word a
/// reader types to run it: an executable called `utils` states no more than a
/// module of that name. A binary has no module path, so no snapshot exempts one.
///
/// A name ending in two digit runs, `validation_findings_12_20`, names the
/// ticket that produced the file rather than the contract inside it, and the
/// ticket is closed by the time anyone reads the name.
///
/// What this does not catch: a specific name that is still wrong for its
/// contents, a published module that carries a banned name, and a directory
/// with no `mod.rs`, whose name no file states. The second one shows up as a
/// snapshot diff in the change that publishes it.
#[must_use]
pub fn generic_module_name_failures(
    module_files: &[String],
    crate_roots: &[CrateRoot],
    published_modules: &[String],
) -> Vec<String> {
    let published: BTreeSet<&str> = published_modules.iter().map(String::as_str).collect();
    let mut failures = Vec::new();
    for file in module_files {
        if let Some(binary) = binary_name_of(file) {
            if is_banned_module_name(binary) {
                failures.push(format!(
                    "`{file}` names the binary `{binary}`, which states no contract; name it for what the binary does"
                ));
            }
            continue;
        }
        let Some(name) = judged_name_of(file) else {
            continue;
        };
        if let Some(range) = ticket_range_of(name) {
            failures.push(format!(
                "`{file}` is named for ticket range `{range}`, not for a contract; name it for what it holds"
            ));
            continue;
        }
        if !is_banned_module_name(name) {
            continue;
        }
        let path = module_path_of(file, crate_roots);
        if path
            .as_deref()
            .is_some_and(|path| published.contains(path))
        {
            continue;
        }
        let published_note = path.map_or_else(String::new, |path| {
            format!(" ({path} is published at no public path, so renaming it breaks nothing)")
        });
        failures.push(format!(
            "`{file}` declares module `{name}`, which states no contract; name it for what it holds{published_note}"
        ));
    }
    failures.sort();
    failures.dedup();
    failures
}

/// Reject sibling files distinguished only by a number.
///
/// `nodes_00.rs` through `nodes_09.rs` in one directory convey nothing about
/// which of the ten classifies a given node, so finding the one that answers a
/// question means opening all ten, and a new case goes into whichever file the
/// author had open. A number inside a name that means something, `crc32`,
/// `float16`, `flash_attention_2`, has no numbered sibling, which is what
/// separates the two: the defect is the number carrying the distinction.
#[must_use]
pub fn numbered_sibling_failures(source_files: &[String]) -> Vec<String> {
    let mut families: BTreeMap<(&str, &str), Vec<&String>> = BTreeMap::new();
    for file in source_files {
        let Some(name) = judged_name_of(file) else {
            continue;
        };
        let Some(stem) = numbered_stem_of(name) else {
            continue;
        };
        let directory = file.rsplit_once('/').map_or("", |(head, _)| head);
        families.entry((directory, stem)).or_default().push(file);
    }
    let mut failures: Vec<String> = families
        .into_iter()
        .filter(|(_, family)| family.len() > 1)
        .flat_map(|((_, stem), family)| {
            let count = family.len();
            family.into_iter().map(move |file| {
                format!(
                    "`{file}` is one of {count} `{stem}_N` siblings, so the number carries the \
                     distinction and the name carries none; name each for what it holds"
                )
            })
        })
        .collect();
    failures.sort();
    failures.dedup();
    failures
}

/// Reject a file that repeats the name of the directory holding it.
///
/// `hardware/fma_f32/fma_f32.rs` states its contents once and its location
/// twice, and the reader who opens the directory has to decide whether the file
/// is the module or a part of it. The module is the directory, so the file is
/// `mod.rs`.
#[must_use]
pub fn directory_stutter_failures(source_files: &[String]) -> Vec<String> {
    let mut failures: Vec<String> = source_files
        .iter()
        .filter_map(|file| {
            let (parents, name) = file.rsplit_once('/')?;
            let stem = name.strip_suffix(".rs")?;
            let directory = parents.rsplit('/').next()?;
            (stem == directory).then(|| {
                format!(
                    "`{file}` repeats the directory `{directory}/` that holds it; the module is \
                     the directory, so this file is `{parents}/mod.rs`"
                )
            })
        })
        .collect();
    failures.sort();
    failures.dedup();
    failures
}

/// The stem a numbered sibling shares with its family, or `None`.
///
/// `nodes_09` is `nodes`; `float16` and `sha256` have no `_` before the digits,
/// so the digits are part of one word rather than a sibling index.
fn numbered_stem_of(name: &str) -> Option<&str> {
    let (stem, digits) = name.rsplit_once('_')?;
    (!digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && !stem.is_empty())
    .then_some(stem)
}

/// The `<digits>_<digits>` tail of a name written for a ticket, or `None`.
fn ticket_range_of(name: &str) -> Option<String> {
    let (head, second) = name.rsplit_once('_')?;
    let (_, first) = head.rsplit_once('_')?;
    (!first.is_empty()
        && !second.is_empty()
        && first.bytes().all(|byte| byte.is_ascii_digit())
        && second.bytes().all(|byte| byte.is_ascii_digit()))
    .then(|| format!("{first}_{second}"))
}

/// True when a name is a dumping ground by name alone.
///
/// A banned word standing alone, or the same word as a `_` suffix.
fn is_banned_module_name(name: &str) -> bool {
    BANNED_MODULE_NAMES.contains(&name)
        || BANNED_MODULE_NAMES.iter().any(|banned| {
            name.len() > banned.len() + 1
                && name.ends_with(banned)
                && name.as_bytes()[name.len() - banned.len() - 1] == b'_'
        })
}

/// The name a source file states, or `None` for a crate or binary root.
///
/// A `mod.rs` is named by its directory, which is the whole point of the
/// layout: reading the file name alone would judge every module in the
/// workspace as being called `mod`. A file under `tests/`, `benches/` or
/// `examples/` is judged the same way, because a reader looking for a fixture
/// reads that name for the same reason.
fn judged_name_of(file: &str) -> Option<&str> {
    if binary_name_of(file).is_some() {
        return None;
    }
    let inside = SOURCE_TREES
        .iter()
        .filter_map(|tree| file.split_once(&format!("/{tree}/")).map(|(_, rest)| rest))
        .min_by_key(|rest| rest.len())?;
    match inside.rsplit('/').next()? {
        "lib.rs" | "main.rs" => None,
        "mod.rs" => inside.rsplit('/').nth(1),
        other => other.strip_suffix(".rs"),
    }
}

/// The binary name a `src/` file declares, or `None` when it is not one.
///
/// Cargo takes both `src/bin/<name>.rs` and `src/bin/<name>/main.rs` as binary
/// roots. Anything deeper under `src/bin/<name>/` is an ordinary module of that
/// binary and is judged as one.
fn binary_name_of(file: &str) -> Option<&str> {
    let (_, inside) = file.split_once("/src/")?;
    let after = inside.strip_prefix("bin/")?;
    match after.split_once('/') {
        None => after.strip_suffix(".rs"),
        Some((name, "main.rs")) => Some(name),
        Some(_) => None,
    }
}

/// The module path a `src/` file declares, as a consumer writes it.
///
/// `vyre-libs/src/parsing/core/mod.rs` is `vyre_libs::parsing::core`. The crate
/// part comes from the [`CrateRoot`] whose directory holds the file, so it is
/// the name the manifest declares rather than the name the directory spells.
/// `None` when no scanned crate holds the file, which is the only honest answer:
/// a guessed crate name would be looked up in the public-API snapshot and miss.
fn module_path_of(file: &str, crate_roots: &[CrateRoot]) -> Option<String> {
    let mut path = crate_roots
        .iter()
        .filter(|crate_root| file.starts_with(&format!("{}/src/", crate_root.directory)))
        .max_by_key(|crate_root| crate_root.directory.len())
        .map(|crate_root| crate_root.ident.clone())?;
    let (_, inside) = file.split_once("/src/")?;
    let name = inside.rsplit('/').next()?;
    let parents = inside.rsplit_once('/').map_or("", |(head, _)| head);
    let tail = if name == "mod.rs" {
        Cow::Borrowed(parents)
    } else {
        let stem = name.strip_suffix(".rs")?;
        if parents.is_empty() {
            Cow::Borrowed(stem)
        } else {
            Cow::Owned(format!("{parents}/{stem}"))
        }
    };
    for segment in tail.split('/').filter(|segment| !segment.is_empty()) {
        path.push_str("::");
        path.push_str(segment);
    }
    Some(path)
}

/// Names `vyre-driver` owns for every backend, that a backend must not define.
///
/// `admit` and `admit_modules` are the admission decision itself; the other two
/// are the rejection vocabulary it answers with.
const SHARED_ADMISSION_HELPERS: &[&str] =
    &["invalid_module", "compile_error", "admit", "admit_modules"];

/// Call spellings that route a backend through the shared admission decision.
///
/// `admit_modules` is the descriptor-bound form: it calls `admit` and then
/// decodes each admitted module in the backend's own dialect, which is the only
/// part of materialization that differs per target. Accepting the bare `admit`
/// as well keeps a backend that needs the admitted list without the decode
/// callback inside the rule.
const SHARED_ADMISSION_CALLS: &[&str] = &["materialize::admit(", "admit_modules("];

/// Reject a concrete backend that decides target-payload admission by itself.
///
/// Admitting a payload is a property of the neutral artifact and the payload
/// envelope, so it is identical for every target. It was nonetheless written
/// once per backend, and the copies drifted until a payload two backends
/// rejected was accepted by the other two. `vyre_driver::materialize` is the
/// single decision; a backend that reimplements it has reopened that class.
pub fn materializer_admission_failures(materializers: &[(String, String)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (path, text) in materializers {
        for helper in SHARED_ADMISSION_HELPERS {
            if text.contains(&format!("fn {helper}(")) {
                failures.push(format!(
                    "`{path}` defines its own `{helper}`; call `vyre_driver::materialize::{helper}` instead"
                ));
            }
        }
        if !SHARED_ADMISSION_CALLS
            .iter()
            .any(|call| text.contains(call))
        {
            failures.push(format!(
                "`{path}` does not admit its target payload through `vyre_driver::materialize`"
            ));
        }
    }
    failures
}

/// Crate that owns every registry link anchor in this workspace.
const REGISTRY_LINK_OWNER: &str = "vyre-registry-link";

/// One `use <crate> as _;` read from member sources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscardingImport {
    /// Workspace-relative source file.
    pub file: String,
    /// Crate identifier as the import writes it, e.g. `vyre_libs`.
    pub named: String,
}

/// Reject a discarding import that names a crate submitting inventory registrations.
///
/// An `inventory` registration lives in the object file of the declaring crate,
/// and a linker keeps an archive member out of an rlib only when a symbol inside
/// it is referenced. `use vyre_libs as _;` names the crate and references no
/// symbol, so the registrations were dropped from every binary that did not
/// otherwise call into that crate: the production binary saw all 354 operation
/// registrations while three registry rules iterated an empty registry and
/// passed. A `const` backend id is no anchor either, because it inlines at the
/// use site. Reading the registry through [`REGISTRY_LINK_OWNER`] calls a real
/// function in each source crate, which is what keeps the object file in.
#[must_use]
pub fn registry_link_failures(submitters: &[String], imports: &[DiscardingImport]) -> Vec<String> {
    let mut failures = Vec::new();
    for import in imports {
        let Some(submitter) = submitters
            .iter()
            .find(|submitter| crate_ident(submitter) == import.named)
        else {
            continue;
        };
        failures.push(format!(
            "`{}` names `{submitter}` with `use {} as _;`, which references no symbol in it, so the linker drops that crate's inventory registrations and every registry read in this binary judges a partial set; read the registry through `{REGISTRY_LINK_OWNER}` instead",
            import.file, import.named
        ));
    }
    failures
}

/// Crate identifier for a crate name, e.g. `vyre_libs` for `vyre-libs`.
fn crate_ident(crate_name: &str) -> String {
    crate_name.replace('-', "_")
}

fn path_names_language(path: &str, language: &str) -> bool {
    path.split(['/', '.'])
        .any(|segment| segment.eq_ignore_ascii_case(language))
}

/// Largest source or manifest file this gate will read.
///
/// The gate walks whatever tree it is pointed at, so an unbounded
/// `read_to_string` lets one pathological file decide the process's memory.
/// Every read in this crate goes through here.
const MAX_SOURCE_BYTES: u64 = 16_777_216;

/// Read a source or manifest file, refusing anything over [`MAX_SOURCE_BYTES`].
fn read_source_bounded(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;

    let file = fs::File::open(path)?;
    if file.metadata()?.len() > MAX_SOURCE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} exceeds {MAX_SOURCE_BYTES} bytes", path.display()),
        ));
    }
    let mut text = String::new();
    file.take(MAX_SOURCE_BYTES + 1).read_to_string(&mut text)?;
    Ok(text)
}

/// The checkout-relative directory of a workspace member, by package name.
///
/// A member's directory is not always its package name: `vyre-conform` lives at
/// `conform/vyre-conform`. The roster is read from the root manifest at run
/// time, so a gate that needs its own crate directory gets it without a
/// compiled-in manifest path, which would name whichever checkout built the
/// binary.
///
/// # Panics
/// Panics when no member directory's manifest declares `package`.
#[must_use]
pub fn member_directory(root: &Path, package: &str) -> PathBuf {
    for member in workspace_members(root) {
        let manifest = root.join(&member).join("Cargo.toml");
        let Ok(text) = read_source_bounded(&manifest) else {
            continue;
        };
        let declared = toml::from_str::<toml::Table>(&text).ok().and_then(|table| {
            Value::Table(table)
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        if declared.as_deref() == Some(package) {
            return root.join(member);
        }
    }
    panic!(
        "Fix: no workspace member under {} declares package `{package}`; the roster in the root \
         Cargo.toml is what this resolves against.",
        root.display()
    )
}

/// The workspace member roster, as the root manifest declares it.
///
/// Every gate that resolves a name against the tree needs this list, so it has
/// one owner: a second copy drifts the moment a member is added under a path
/// one copy filters and the other does not.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn workspace_members(root: &Path) -> Vec<String> {
    workspace_paths(root, "members")
}

/// The paths the root manifest excludes from the workspace.
///
/// `exclude` is the other half of the roster: a directory that is neither a
/// member nor excluded is a directory cargo will pull in the day it grows a
/// manifest. Reading it beside [`workspace_members`] keeps both answers coming
/// from one parse of one file.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn workspace_excludes(root: &Path) -> Vec<String> {
    workspace_paths(root, "exclude")
}

/// One `[workspace]` array of paths, empty when the key is absent.
fn workspace_paths(root: &Path, key: &str) -> Vec<String> {
    let manifest = root.join("Cargo.toml");
    let text = read_source_bounded(&manifest)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest.display()));
    let table: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: parse {}: {error}", manifest.display()));
    Value::Table(table)
        .get("workspace")
        .and_then(|workspace| workspace.get(key))
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// This crate's own sources. Its tests carry example registrations that name
/// other crates on purpose, so scanning itself would report its own fixtures.
const SELF_CRATE: &str = "structure-gate";

fn source_files(root: &Path, member: &str) -> Vec<PathBuf> {
    if member == SELF_CRATE {
        return Vec::new();
    }
    source_tree_files(&root.join(member).join("src"))
}

/// Every `.rs` file under one source tree.
fn source_tree_files(directory: &Path) -> Vec<PathBuf> {
    WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .collect()
}

/// Every crate in the checkout that keeps its sources in a `src/` directory.
///
/// Read from the tree rather than the workspace roster, because the layout
/// rules judge tree shape and a crate outside the workspace grows the same
/// pairs and the same nameless modules: the external extension examples are
/// separate packages on purpose. A directory earns a place here by declaring
/// `[package]` and holding a `src/`, so a crate added anywhere in the checkout
/// is judged without an edit here. This crate is included; [`source_files`]
/// exempts it only because its registration fixtures name other crates, and a
/// rule over file names has no such fixtures.
fn crate_source_roots(root: &Path) -> Vec<CrateRoot> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| !matches!(entry.file_name().to_str(), Some(".git" | "target")))
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name() == "Cargo.toml")
        .filter_map(|entry| {
            let ident = manifest_crate_ident(entry.path())?;
            let directory = entry.path().parent()?.to_path_buf();
            directory.join("src").is_dir().then(|| CrateRoot {
                directory: relative(root, &directory),
                ident,
            })
        })
        .collect()
}

/// The identifier a manifest's library carries, or `None` for no package.
///
/// `[lib] name` wins where it is written, because that is the name a consumer
/// and `cargo public-api` both use; the package name is the default Cargo
/// applies when it is not.
fn manifest_crate_ident(manifest: &Path) -> Option<String> {
    let text = read_source_bounded(manifest).ok()?;
    let table: toml::Table = toml::from_str(&text).ok()?;
    let value = Value::Table(table);
    let package = value.get("package")?.get("name")?.as_str()?;
    let name = value
        .get("lib")
        .and_then(|lib| lib.get("name"))
        .and_then(Value::as_str)
        .unwrap_or(package);
    Some(crate_ident(name))
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn scan_registrations(root: &Path, members: &[String]) -> Vec<Registration> {
    let mut registrations = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member).to_string();
        for path in source_files(root, member) {
            let Ok(text) = read_source_bounded(&path) else {
                continue;
            };
            let file = relative(root, &path);
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

/// Tier implied by each `OperationRegistration` constructor.
///
/// `new` takes the tier as its second argument, so it is read there rather
/// than assumed. Guessing it wrong is worse than not knowing: mapping
/// `primitive` to `Library` once reported all 122 of one crate's intrinsics as
/// misplaced compositions and buried the real findings. `primitive` names the
/// owning crate, `vyre-primitives`, and builds `OperationTier::Intrinsic`.
const CONSTRUCTOR_TIERS: &[(&str, Option<&str>)] = &[
    ("::primitive(", Some("Intrinsic")),
    ("::library(", Some("Library")),
    ("::new(", None),
];

/// Remove every `#[cfg(test)]`-gated item before a production-code scan.
///
/// A test module registers fixture operations - `test::reference_echo`,
/// `test::call_u32` and friends - that exist in no production build. Counting
/// them as registry members reported four test doubles as misplaced production
/// operations, and pointed Phase 2 at code that was already correct.
///
/// The predicate is tokenized with string literals removed first, so
/// `#[cfg(feature = "test-utils")]` is not mistaken for a test gate.
pub fn strip_cfg_test_items(text: &str) -> Cow<'_, str> {
    let spans = cfg_test_spans(text);
    if spans.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut kept = String::with_capacity(text.len());
    let mut kept_from = 0usize;
    for span in spans {
        kept.push_str(&text[kept_from..span.0]);
        kept_from = span.1;
    }
    kept.push_str(&text[kept_from..]);
    Cow::Owned(kept)
}

/// Every `#[cfg(test)]`-gated item of one file, concatenated in source order.
///
/// The complement of [`strip_cfg_test_items`], for a caller that judges test
/// code rather than production code. Both read the same spans, so a scanner
/// improvement lands on both views at once: the coverage corpus that used
/// "everything after the first `#[cfg(test)]` marker" instead counted a crate's
/// production re-export list as test text, and 174 runtime symbols were
/// "covered" by a `pub use` block that names them.
#[must_use]
pub fn cfg_test_items(text: &str) -> String {
    let mut out = String::new();
    for (start, end) in cfg_test_spans(text) {
        out.push_str(&text[start..end]);
        out.push('\n');
    }
    out
}

/// Names of the modules one file declares behind a `#[cfg(test)]` gate.
///
/// A test module written in its own file - `mod tests;` beside `tests/mod.rs`,
/// or `mod core_tests;` beside `core_tests.rs` - carries no gating attribute of
/// its own, so [`cfg_test_items`] over that file returns nothing and a caller
/// judging test text would read the file as production code. The declaration is
/// the only place the gate is written, so it is read here rather than inferred
/// from a file name: `tests.rs` is a test module because a `#[cfg(test)] mod
/// tests;` says so, and a crate that ships a production module of that name is
/// not misread.
#[must_use]
pub fn cfg_test_module_declarations(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for (start, end) in cfg_test_spans(text) {
        let span = text[start..end].trim_end();
        let Some(body) = span.strip_suffix(';') else {
            continue;
        };
        let Some(declaration) = body.rsplit_once("mod ") else {
            continue;
        };
        let name = declaration.1.trim();
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            names.push(name.to_string());
        }
    }
    names
}

/// Byte spans of every test-gated item, attribute included, in source order.
///
/// Text not yet accounted for and where the next attribute is looked for are
/// separate cursors: a non-test `#[cfg(...)]` moves the search past its
/// predicate but keeps every byte, and sharing one cursor for both deleted the
/// whole file up to the last non-test attribute. That silently dropped the
/// `const` an id resolved through, so a real registration became no registration
/// and the rules below judged a registry they could not see.
fn cfg_test_spans(text: &str) -> Vec<(usize, usize)> {
    const ATTR: &str = "#[cfg(";
    let mut spans = Vec::new();
    let mut search = 0usize;
    while let Some(offset) = text[search..].find(ATTR) {
        let attr_start = search + offset;
        let predicate_start = attr_start + ATTR.len() - 1;
        let Some(predicate_end) = match_delimited(text, predicate_start, b'(', b')') else {
            break;
        };
        if !mentions_test(&text[predicate_start + 1..predicate_end]) {
            search = predicate_end + 1;
            continue;
        }
        let Some(attr_end) = text[predicate_end..].find(']').map(|at| predicate_end + at) else {
            break;
        };
        let Some(item_end) = end_of_item(text, attr_end + 1) else {
            break;
        };
        spans.push((attr_start, item_end));
        search = item_end;
    }
    spans
}

/// Byte index of the delimiter closing the one that opens at `open`.
///
/// Delimiters inside a string, char literal, raw string or comment are text:
/// counting them ends a `#[cfg(test)] mod tests { .. }` at a `}` written inside
/// a string and leaves the rest of the test module in the scanned text.
fn match_delimited(text: &str, open: usize, opener: u8, closer: u8) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open) != Some(&opener) {
        return None;
    }
    let mut depth = 0usize;
    let mut index = open;
    while index < bytes.len() {
        if let Some(span) = opaque_span(text, index) {
            index += span;
            continue;
        }
        if bytes[index] == opener {
            depth += 1;
        } else if bytes[index] == closer {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

/// True when the cfg predicate names `test` as a bare token.
fn mentions_test(predicate: &str) -> bool {
    let mut outside = String::with_capacity(predicate.len());
    let mut in_string = false;
    for character in predicate.chars() {
        match character {
            '"' => in_string = !in_string,
            _ if in_string => {}
            _ => outside.push(character),
        }
    }
    outside
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .any(|token| token == "test")
}

/// End of the item a gating attribute applies to, past `from`.
///
/// A braced item ends at its matching `}`; a declaration such as
/// `#[cfg(test)] mod tests;` ends at its `;`. Further attributes stacked on the
/// same item are skipped so the whole item is removed, not just the tail.
fn end_of_item(text: &str, from: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = from;
    while index < bytes.len() {
        match bytes[index] {
            b'{' => return match_delimited(text, index, b'{', b'}').map(|close| close + 1),
            b';' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}

/// Extract `(op_id, tier)` for every `OperationRegistration` in one file.
///
/// Two forms exist in the tree: a struct literal with named fields, and a
/// constructor call taking the id first. Both are scanned, because a gate that
/// understands only one form silently exempts every crate that uses the other -
/// which is how 140 registrations in one crate went unjudged.
///
/// Ids appear inline or through a file-local `const`, so the scan resolves both
/// without compiling the crate. That keeps the gate usable while the tree is
/// mid-migration and a crate does not build.
///
/// Test-gated items are removed first, so a fixture registration in a
/// `#[cfg(test)]` module is not counted as a production operation.
pub fn parse_registrations(text: &str) -> Vec<(String, Option<String>)> {
    let stripped = strip_cfg_test_items(text);
    let text = stripped.as_ref();
    let consts = string_consts(text);
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("OperationRegistration") {
        let body = &rest[start..];
        let after = &body["OperationRegistration".len()..];
        let constructor = CONSTRUCTOR_TIERS.iter().find(|(call, _)| {
            after
                .trim_start()
                .starts_with(call.trim_start_matches("::"))
                || after.starts_with(call)
        });
        if let Some((call, tier)) = constructor {
            if let Some(id) = first_argument(after, call).and_then(|raw| resolve_id(raw, &consts)) {
                let tier = tier
                    .map(|tier| tier.to_string())
                    .or_else(|| nth_argument(after, call, 1).map(tier_variant));
                found.push((id, tier));
            }
        } else {
            let block = &body[..struct_literal_end(body)];
            if let Some(id) = field_value(block, "id").and_then(|raw| resolve_id(raw, &consts)) {
                found.push((id, field_value(block, "tier").map(tier_variant)));
            }
        }
        rest = after;
    }
    found
}

/// First argument of a constructor call, as written.
fn first_argument<'a>(after: &'a str, call: &str) -> Option<&'a str> {
    nth_argument(after, call, 0)
}

/// Argument `index` of a constructor call, as written.
///
/// Splits on top-level commas only: `()`, `[]` and `{}` nest, and a comma
/// inside a string, char, raw string or comment is text rather than a
/// separator. Reading the id out of argument zero is how a registration enters
/// the gate's model, so a boundary read one argument early drops the
/// registration outright and the rules below then report a registry they never
/// saw.
///
/// `<` and `>` are ordinary characters. Counting them as delimiters was worse
/// than ignoring them: registration builders are closures, so `->`, `<` and
/// `>` appear as operators constantly, one unbalanced occurrence left the depth
/// permanently wrong, and a `->` dropped the depth far enough that the `)`
/// closing a nested `Some(` was read as the end of the whole call. The cost is
/// that a generic written with a top-level comma outside any delimiter pair -
/// a bare `Vec::<u8, Global>::new()` argument - would split; no registration in
/// the tree writes one.
fn nth_argument<'a>(after: &'a str, call: &str, index: usize) -> Option<&'a str> {
    let open = after.find(call)? + call.len();
    let rest = &after[open..];
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut argument = 0usize;
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(rest, offset) {
            offset += span;
            continue;
        }
        match bytes[offset] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' if depth > 0 => depth -= 1,
            b')' => {
                return (argument == index).then(|| rest[start..offset].trim());
            }
            b',' if depth == 0 => {
                if argument == index {
                    return Some(rest[start..offset].trim());
                }
                argument += 1;
                start = offset + 1;
            }
            _ => {}
        }
        offset += 1;
    }
    None
}

/// Byte length of the span starting at `at` whose interior is not code: a line
/// or block comment, a string, a char literal, or any prefixed or raw form of
/// those. `None` when ordinary code starts there.
///
/// The gate reads source text without compiling it, so nothing else
/// distinguishes a comma inside `", "` from an argument separator.
pub fn opaque_span(text: &str, at: usize) -> Option<usize> {
    let rest = &text[at..];
    if let Some(body) = rest.strip_prefix("//") {
        return Some(2 + body.find('\n').map_or(body.len(), |end| end + 1));
    }
    if rest.starts_with("/*") {
        return Some(block_comment_len(rest));
    }
    if rest.starts_with('"') {
        return Some(escaped_string_len(rest));
    }
    if rest.starts_with('\'') {
        return char_literal_len(rest);
    }
    prefixed_literal_len(text, at)
}

/// Byte length of the block comment starting at `rest`, which nests in Rust.
fn block_comment_len(rest: &str) -> usize {
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut offset = 0usize;
    while offset + 1 < bytes.len() {
        if bytes[offset] == b'/' && bytes[offset + 1] == b'*' {
            depth += 1;
            offset += 2;
        } else if bytes[offset] == b'*' && bytes[offset + 1] == b'/' {
            depth -= 1;
            offset += 2;
            if depth == 0 {
                return offset;
            }
        } else {
            offset += 1;
        }
    }
    rest.len()
}

/// Byte length of the backslash-escaped string starting at `rest`.
///
/// An unterminated literal consumes the remaining text: the alternative is to
/// resume scanning inside a string, where every delimiter is misread.
fn escaped_string_len(rest: &str) -> usize {
    let bytes = rest.as_bytes();
    let mut offset = 1usize;
    while offset < bytes.len() {
        match bytes[offset] {
            b'\\' => offset += 2,
            b'"' => return offset + 1,
            _ => offset += 1,
        }
    }
    rest.len()
}

/// Byte length of the char literal starting at `rest`, or `None` when the quote
/// opens a lifetime or a loop label instead.
fn char_literal_len(rest: &str) -> Option<usize> {
    let body = &rest[1..];
    if let Some(escape) = body.strip_prefix('\\') {
        let escaped = if escape.starts_with('u') {
            escape.find('}')? + 1
        } else {
            escape.chars().next()?.len_utf8()
        };
        return Some(2 + escaped + escape[escaped..].find('\'')? + 1);
    }
    let literal = body.chars().next()?.len_utf8();
    body[literal..].starts_with('\'').then_some(literal + 2)
}

/// Byte length of a literal carrying a `r`, `b` or `c` prefix, including every
/// raw form. `None` when the bytes are an ordinary identifier such as `bytes`
/// or `crc32`.
fn prefixed_literal_len(text: &str, at: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_') {
        return None;
    }
    let rest = &text[at..];
    let prefix = rest
        .bytes()
        .take(2)
        .take_while(|byte| matches!(*byte, b'r' | b'b' | b'c'))
        .count();
    if prefix == 0 {
        return None;
    }
    let body = &rest[prefix..];
    if rest[..prefix].contains('r') {
        let hashes = body.bytes().take_while(|byte| *byte == b'#').count();
        let quoted = &body[hashes..];
        if !quoted.starts_with('"') {
            return None;
        }
        return Some(prefix + hashes + raw_string_len(quoted, hashes));
    }
    if body.starts_with('"') {
        return Some(prefix + escaped_string_len(body));
    }
    if body.starts_with('\'') {
        return char_literal_len(body).map(|len| prefix + len);
    }
    None
}

/// Byte length of the raw string opening at `quoted`, closed by a quote
/// followed by `hashes` hash marks. Raw strings honour no escape.
fn raw_string_len(quoted: &str, hashes: usize) -> usize {
    let bytes = quoted.as_bytes();
    let mut offset = 1usize;
    while offset < bytes.len() {
        if bytes[offset] == b'"'
            && quoted[offset + 1..]
                .bytes()
                .take_while(|byte| *byte == b'#')
                .count()
                >= hashes
        {
            return offset + 1 + hashes;
        }
        offset += 1;
    }
    quoted.len()
}

/// Byte offset just past the struct literal that opens in `body`.
///
/// Registration fields hold closures, so the first `}` is almost never the end
/// of the literal. Counting depth is what keeps `id:` and `tier:` inside the
/// scanned window; stopping at the first brace silently drops most
/// registrations and makes every registration rule pass on an empty set. A
/// brace inside a string, char literal, raw string or comment is text and does
/// not count.
fn struct_literal_end(body: &str) -> usize {
    let bytes = body.as_bytes();
    let mut depth = 0usize;
    let mut opened = false;
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(body, offset) {
            offset += span;
            continue;
        }
        match bytes[offset] {
            b'{' => {
                depth += 1;
                opened = true;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if opened && depth == 0 {
                    return offset + 1;
                }
            }
            _ => {}
        }
        offset += 1;
    }
    body.len()
}

/// Map every `const NAME: &str = "value";` in a file to its literal.
///
/// Read over the whole text rather than line by line: a long id is wrapped
/// onto the line after the `=`, and a line-bound scan resolved none of those,
/// so every registration whose id came through such a const was dropped. The
/// declared type must name `str`, which keeps `const fn` bodies and const
/// generic parameters out of the map.
fn string_consts(text: &str) -> BTreeMap<String, String> {
    const KEYWORD: &str = "const ";
    let mut consts = BTreeMap::new();
    let mut cursor = 0usize;
    while let Some(offset) = text[cursor..].find(KEYWORD) {
        let start = cursor + offset + KEYWORD.len();
        cursor = start;
        let Some(end) = text[start..].find(';') else {
            break;
        };
        let Some((declared, value)) = text[start..start + end].split_once('=') else {
            continue;
        };
        let Some((name, declared_type)) = declared.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !declared_type.contains("str")
            || name.is_empty()
            || !name
                .chars()
                .all(|character| character.is_alphanumeric() || character == '_')
        {
            continue;
        }
        if let Some(literal) = string_literal(value) {
            consts.insert(name.to_string(), literal);
        }
    }
    consts
}

fn string_literal(text: &str) -> Option<String> {
    let start = text.find('"')?;
    let rest = &text[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Read one `field: value,` from a struct literal body.
fn field_value<'a>(block: &'a str, field: &str) -> Option<&'a str> {
    for line in block.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix(field) else {
            continue;
        };
        let Some(rest) = rest.strip_prefix(':') else {
            continue;
        };
        return Some(rest.trim().trim_end_matches(','));
    }
    None
}

fn resolve_id(raw: &str, consts: &BTreeMap<String, String>) -> Option<String> {
    if let Some(literal) = string_literal(raw) {
        return Some(literal);
    }
    consts.get(raw.trim()).cloned()
}

fn tier_variant(raw: &str) -> String {
    raw.rsplit("::").next().unwrap_or(raw).trim().to_string()
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
/// set the moment it submits, so the link rule judges it without an edit.
fn scan_registry_submitters(root: &Path, members: &[String]) -> Vec<String> {
    let mut submitters = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member);
        for path in source_files(root, member) {
            let Ok(text) = read_source_bounded(&path) else {
                continue;
            };
            if submits_registrations(&text) {
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

/// True when this source text submits an inventory registration.
///
/// Comments are skipped, so a doc comment explaining the linkage rule is not
/// mistaken for a registration.
#[must_use]
pub fn submits_registrations(text: &str) -> bool {
    code_offsets(text).any(|at| text[at..].starts_with("inventory::submit!"))
}

/// Crate identifiers named by a discarding import, as written.
///
/// Only a bare crate identifier counts. `use std::io::Read as _;` imports a
/// trait into scope, which is the legitimate use of the form and references a
/// symbol at every call site.
#[must_use]
pub fn discarding_imports(text: &str) -> Vec<String> {
    code_offsets(text)
        .filter_map(|at| discarded_crate(&text[at..]))
        .collect()
}

/// Byte offsets of code in `text`, with comments and literals skipped.
fn code_offsets(text: &str) -> impl Iterator<Item = usize> + '_ {
    let mut skip_to = 0usize;
    text.char_indices().filter_map(move |(at, _)| {
        if at < skip_to {
            return None;
        }
        if let Some(span) = opaque_span(text, at) {
            skip_to = at + span.max(1);
            return None;
        }
        Some(at)
    })
}

/// Crate identifier of a `use <crate> as _;` statement starting at `rest`.
fn discarded_crate(rest: &str) -> Option<String> {
    let statement = rest.strip_prefix("use ")?;
    let end = statement.find(';')?;
    let (path, alias) = statement.get(..end)?.split_once(" as ")?;
    if alias.trim() != "_" {
        return None;
    }
    let path = path.trim();
    if path.is_empty()
        || !path
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return None;
    }
    Some(path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(crate_name: &str, op_id: &str, tier: Option<&str>) -> Registration {
        Registration {
            crate_name: crate_name.to_string(),
            file: format!("{crate_name}/src/op.rs"),
            op_id: op_id.to_string(),
            tier: tier.map(str::to_string),
        }
    }

    #[test]
    fn a_third_registering_crate_is_rejected() {
        let failures = registration_owner_failures(&[registration(
            "vyre-pass-engine",
            "vyre-pass-engine::graph::toposort",
            Some("Library"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("vyre-pass-engine"));
    }

    #[test]
    fn the_two_category_owners_are_accepted() {
        let failures = registration_owner_failures(&[
            registration("vyre-libs", "vyre-libs::hash::adler32", None),
            registration(
                "vyre-primitives",
                "vyre-primitives::atomic::compare_exchange",
                None,
            ),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn one_kernel_registered_under_two_namespaces_is_rejected() {
        let failures = operation_identity_failures(&[
            registration("vyre-foundation", "vyre-foundation::hash::adler32", None),
            registration("vyre-libs", "vyre-libs::hash::adler32", None),
        ]);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("2 identities")),
            "{failures:?}"
        );
    }

    #[test]
    fn same_leaf_under_one_namespace_is_accepted() {
        let failures = operation_identity_failures(&[
            registration("vyre-foundation", "vyre-foundation::hash::adler32", None),
            registration("vyre-foundation", "vyre-foundation::graph::toposort", None),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn an_id_claiming_another_crates_namespace_is_rejected() {
        let failures = operation_identity_failures(&[registration(
            "vyre-libs",
            "vyre-foundation::hash::adler32",
            None,
        )]);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("names its owning crate")),
            "{failures:?}"
        );
    }

    #[test]
    fn a_hardware_operation_in_the_category_a_crate_is_rejected() {
        let failures = category_home_failures(&[registration(
            "vyre-libs",
            "vyre-libs::atomic::compare_exchange",
            Some("Intrinsic"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("Category C"));
    }

    #[test]
    fn a_composition_in_the_category_c_crate_is_rejected() {
        let failures = category_home_failures(&[registration(
            "vyre-primitives",
            "vyre-primitives::hash::adler32",
            Some("Library"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("Category A"));
    }

    #[test]
    fn each_category_in_its_own_home_is_accepted() {
        let failures = category_home_failures(&[
            registration("vyre-libs", "vyre-libs::hash::adler32", Some("Library")),
            registration(
                "vyre-primitives",
                "vyre-primitives::atomic::compare_exchange",
                Some("Intrinsic"),
            ),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_second_substrate_home_is_rejected() {
        // Illustrative names: the second homes this rule caught in the tree
        // (`vyre-libs/src/substrate_catalog.rs`, `vyre-driver/src/speculation_substrate.rs`)
        // have been renamed, so the fixture keeps the shape rather than a path.
        let failures = substrate_home_failures(&[
            "vyre-foundation/src/pass_substrate/semiring_closure.rs".to_string(),
            "vyre-driver/src/speculation_substrate.rs".to_string(),
            "vyre-libs/src/matmul_substrate.rs".to_string(),
        ]);

        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(failures.iter().any(|f| f.contains("speculation_substrate")));
        assert!(failures.iter().any(|f| f.contains("matmul_substrate")));
    }

    #[test]
    fn the_foundation_pass_substrate_home_is_accepted() {
        // ARCHITECTURE.md: foundation owns the CPU pass math and the pass engine
        // imports it, so this is the one home the name has. Every other home
        // stays a failure.
        let failures = substrate_home_failures(&[
            "vyre-foundation/src/pass_substrate/semiring_closure.rs".to_string(),
            "vyre-foundation/src/pass_substrate/mod.rs".to_string(),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_second_c_frontend_crate_is_rejected_by_its_name() {
        // vyre-libs owns the C frontend because it is built from Category A
        // compositions. The tree-sitter shell crate that was the second one
        // kept a flat layout with no lex/ directory, so only its name gave it
        // away. It has left the workspace; the rule keeps a replacement out.
        let failures = frontend_owner_failures(&[
            (
                "vyre-libs".to_string(),
                "vyre-libs/src/parsing/c/lex/keyword.rs".to_string(),
            ),
            ("vyre-frontend-c".to_string(), "vyre-frontend-c".to_string()),
        ]);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("`vyre-frontend-c` is a second c frontend crate"),
            "{failures:?}"
        );
    }

    #[test]
    fn a_language_stage_directory_outside_the_owner_is_rejected() {
        // The other signal: the crate name says nothing, but it holds a
        // language-named lexer stage that belongs to the owning crate.
        let failures = frontend_owner_failures(&[(
            "vyre-driver-wgpu".to_string(),
            "vyre-driver-wgpu/src/parsing/c/lex/keyword.rs".to_string(),
        )]);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("puts a c frontend in vyre-driver-wgpu"),
            "{failures:?}"
        );
    }

    #[test]
    fn the_declared_owner_of_a_language_is_accepted() {
        // Control: naming a frontend is only a failure for a non-owner, and
        // the rust owner is a crate whose own name declares the language.
        let failures = frontend_owner_failures(&[
            (
                "vyre-frontend-rust".to_string(),
                "vyre-frontend-rust".to_string(),
            ),
            (
                "vyre-libs".to_string(),
                "vyre-libs/src/parsing/c/lex/keyword.rs".to_string(),
            ),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_product_crate_on_the_roster_is_rejected() {
        let failures = roster_failures(&["vyre-foundation".to_string(), "vyre-scan".to_string()]);

        assert!(
            failures.iter().any(|f| f.contains("vyre-scan")),
            "{failures:?}"
        );
    }

    #[test]
    fn an_inline_registration_id_is_parsed() {
        let parsed = parse_registrations(
            r#"
inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        tier: vyre_foundation::operation::OperationTier::Library,
        id: "vyre-foundation::hash::adler32",
    }
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-foundation::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    /// Registration fields hold closures. A parser that stops at the first `}`
    /// truncates the literal before `id:` and reports no registration at all,
    /// which makes every registration rule pass on an empty set.
    /// Most registrations in the tree use the constructor form. A parser that
    /// reads only the struct-literal form exempts every crate that uses it.
    #[test]
    fn a_constructor_registration_is_parsed() {
        let parsed = parse_registrations(
            r#"
const ADLER32_OP_ID: &str = "vyre-foundation::hash::adler32";

fn registration() -> OperationRegistration {
    vyre_foundation::operation::OperationRegistration::primitive(
        ADLER32_OP_ID,
        || adler32_program("input", "out", 3),
        Some(|| { vec![vec![vec![1u8]]] }),
        Some(|| vec![vec![vec![2u8]]]),
    )
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-foundation::hash::adler32".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    /// `new` carries the tier in argument two. Assuming a tier for it once
    /// reported an entire crate's intrinsics as misplaced compositions.
    #[test]
    fn a_new_registration_reads_its_tier_argument() {
        let parsed = parse_registrations(
            r#"
    OperationRegistration::new(
        "vyre-primitives::hardware::fma_f32",
        OperationTier::Intrinsic,
        Some(fma_f32_program),
        Some(|| vec![vec![vec![1u8, 2u8]]]),
        None,
    )
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-primitives::hardware::fma_f32".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    #[test]
    fn a_constructor_registration_with_an_inline_id_is_parsed() {
        let parsed = parse_registrations(
            r#"
    OperationRegistration::library("vyre-libs::nn::attention", builder)
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::nn::attention".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_registration_whose_fields_contain_braces_is_still_parsed() {
        let parsed = parse_registrations(
            r#"
inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        build: Some(|| { let program = adler32("input", "out", 3); program }),
        test_inputs: Some(|| { vec![vec![vec![1u8]]] }),
        tier: vyre_foundation::operation::OperationTier::Library,
        id: "vyre-foundation::hash::adler32",
    }
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-foundation::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_const_backed_registration_id_is_resolved() {
        let parsed = parse_registrations(
            r#"
const OP_ID: &str = "vyre-libs::atomic::compare_exchange";

inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        tier: vyre_foundation::operation::OperationTier::Intrinsic,
        id: OP_ID,
    }
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::atomic::compare_exchange".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    #[test]
    fn a_registration_inside_a_test_module_is_not_a_production_operation() {
        let parsed = parse_registrations(
            r#"
            #[cfg(test)]
            mod tests {
                const ECHO_ID: &str = "test::reference_echo";
                fn fixture() {
                    OperationRegistration::library(ECHO_ID);
                }
            }
            "#,
        );

        assert_eq!(parsed, Vec::new());
    }

    #[test]
    fn a_production_registration_beside_a_test_module_is_still_counted() {
        let parsed = parse_registrations(
            r#"
            fn install() {
                OperationRegistration::library("vyre-libs::hash::crc32");
            }

            #[cfg(test)]
            mod tests {
                fn fixture() {
                    OperationRegistration::library("test::call_u32");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::crc32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_feature_named_test_something_does_not_exempt_a_registration() {
        let parsed = parse_registrations(
            r#"
            #[cfg(feature = "test-utils")]
            mod utils {
                fn install() {
                    OperationRegistration::library("vyre-libs::hash::fnv1a32");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::fnv1a32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_compound_test_predicate_still_strips_the_item() {
        let parsed = parse_registrations(
            r#"
            #[cfg(all(test, feature = "gpu"))]
            mod tests {
                fn fixture() {
                    OperationRegistration::library("test::reference_panic");
                }
            }
            "#,
        );

        assert_eq!(parsed, Vec::new());
    }

    #[test]
    fn a_test_gated_module_declaration_strips_only_the_declaration() {
        let parsed = parse_registrations(
            r#"
            #[cfg(test)]
            mod tests;

            fn install() {
                OperationRegistration::library("vyre-libs::hash::adler32");
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    /// Shape of `vyre-libs/src/hash/adler32.rs`: a braced `use` list, a const
    /// id, two test-gated `use` lines, a test-gated helper, then the real
    /// struct-literal registration, then the test module. The production id
    /// must survive all of it.
    #[test]
    fn a_production_registration_survives_a_file_full_of_test_gated_items() {
        let parsed = parse_registrations(
            r#"
            use vyre_libs::hash::adler32::{adler32_program, ADLER32_OP_ID};

            #[cfg(test)]
            use crate::buffer_names::fixed_name;
            #[cfg(test)]
            use vyre_libs::hash::adler32::adler32 as adler32_cpu_reference;

            const OP_ID: &str = "vyre-libs::hash::adler32";

            #[cfg(test)]
            fn cpu_ref(input: &[u8]) -> u32 {
                adler32_cpu_reference(input)
            }

            inventory::submit! {
                vyre_foundation::operation::OperationRegistration {
                    semantic_version: 1,
                    tier: vyre_foundation::operation::OperationTier::Library,
                    id: OP_ID,
                    build: Some(|| adler32("input", "out", 3)),
                    category: None,
                }
            }

            #[cfg(test)]
            mod tests {
                use super::*;
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    /// WHY this group: `nth_argument` decides where one constructor argument
    /// ends, and `parse_registrations` reads the operation id out of argument
    /// zero. Every shape the splitter misreads is an operation the gate never
    /// judges, and a registry the gate cannot see is a registry it reports
    /// clean. The splitter counted `<` and `>` as delimiters and read no
    /// literal or comment, so one `<`, one `->`, one quoted comma or one
    /// commented comma moved every later argument boundary.
    ///
    /// What this group does not pin: a generic written with a top-level comma
    /// outside any delimiter pair, such as a bare `Vec::<u8, Global>::new()`
    /// argument. `<` and `>` are ordinary characters on purpose, because Rust
    /// writes them as comparison, shift and return-arrow tokens far more often
    /// than as a balanced pair.
    #[test]
    fn a_top_level_comma_separates_arguments() {
        let call = "::primitive(OP_ID, builder, None)";

        assert_eq!(nth_argument(call, "::primitive(", 0), Some("OP_ID"));
        assert_eq!(nth_argument(call, "::primitive(", 1), Some("builder"));
        assert_eq!(nth_argument(call, "::primitive(", 2), Some("None"));
    }

    #[test]
    fn a_nested_call_in_the_first_argument_does_not_shift_the_count() {
        let call = r#"::new(op_id("bitset", "xor"), OperationTier::Intrinsic)"#;

        assert_eq!(
            nth_argument(call, "::new(", 0),
            Some(r#"op_id("bitset", "xor")"#)
        );
        assert_eq!(
            nth_argument(call, "::new(", 1),
            Some("OperationTier::Intrinsic")
        );
    }

    /// A single `<` used to raise the depth for the rest of the call, so every
    /// later comma read as nested and every argument after it disappeared.
    #[test]
    fn a_comparison_operator_is_not_an_opening_delimiter() {
        let call = "::library(OP_ID, |n| n < 4, None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some("|n| n < 4"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// `->` lowered the depth, so the `)` closing a nested `Some(` was read as
    /// the `)` closing the constructor: the argument came back missing its own
    /// closing paren and every later argument came back as nothing.
    #[test]
    fn a_return_arrow_is_not_a_closing_delimiter() {
        let call = "::new(OP_ID, OperationTier::Library, Some(|| -> Vec<u32> { vec![1] }), None)";

        assert_eq!(
            nth_argument(call, "::new(", 2),
            Some("Some(|| -> Vec<u32> { vec![1] })")
        );
        assert_eq!(nth_argument(call, "::new(", 3), Some("None"));
    }

    /// A generic argument carries its own comma. It is not a separator because
    /// it sits inside the parentheses of the argument it belongs to.
    #[test]
    fn a_generic_argument_comma_is_not_a_separator() {
        let call = "::new(OP_ID, OperationTier::Library, Some(pairs::<String, u32>), None)";

        assert_eq!(
            nth_argument(call, "::new(", 2),
            Some("Some(pairs::<String, u32>)")
        );
        assert_eq!(nth_argument(call, "::new(", 3), Some("None"));
    }

    #[test]
    fn a_comma_inside_a_string_literal_is_not_a_separator() {
        let call = r#"::library(OP_ID, ", ", None)"#;

        assert_eq!(nth_argument(call, "::library(", 1), Some(r#"", ""#));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_parenthesis_inside_a_string_literal_does_not_close_the_call() {
        let call = r#"::library(OP_ID, "f(", None)"#;

        assert_eq!(nth_argument(call, "::library(", 1), Some(r#""f(""#));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn an_escaped_quote_does_not_end_a_string_literal() {
        let call = r#"::library(OP_ID, "a\", b(", None)"#;

        assert_eq!(nth_argument(call, "::library(", 1), Some(r#""a\", b(""#));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_nested_closure_is_one_argument() {
        let call = "::library(OP_ID, |graph| move |node| visit(graph, node), None)";

        assert_eq!(
            nth_argument(call, "::library(", 1),
            Some("|graph| move |node| visit(graph, node)")
        );
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_raw_string_argument_is_read_whole() {
        let call = "::library(OP_ID, r#\"a, b)\"#, None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some("r#\"a, b)\"#"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// A comment sits inside the argument it interrupts, so the boundaries of
    /// the arguments after it must not move.
    #[test]
    fn a_comma_in_a_comment_is_not_a_separator() {
        let line_comment = "::library(\n    OP_ID, // one, two\n    builder,\n    None,\n)";
        let block_comment = "::library(OP_ID, /* one, two */ builder, None)";

        assert_eq!(nth_argument(line_comment, "::library(", 0), Some("OP_ID"));
        assert_eq!(nth_argument(line_comment, "::library(", 2), Some("None"));
        assert_eq!(nth_argument(block_comment, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_char_literal_comma_is_not_a_separator() {
        let call = "::library(OP_ID, ',', None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some("','"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// Adversarial case for the literal scanner itself: `'static` and a loop
    /// label open no char literal, so the scanner must not swallow the text up
    /// to the next quote.
    #[test]
    fn a_lifetime_is_not_a_char_literal() {
        let call = "::library(OP_ID, |text: &'static str| text.len(), None)";

        assert_eq!(
            nth_argument(call, "::library(", 1),
            Some("|text: &'static str| text.len()")
        );
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn an_escaped_quote_char_literal_is_read_whole() {
        let call = r"::library(OP_ID, '\'', None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some(r"'\''"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// WHY: stripping a `#[cfg(test)]` item used to delete every byte before
    /// the last non-test `#[cfg(...)]` attribute in the file, because one
    /// cursor served both "where to search next" and "what is still uncopied".
    /// The deleted span held the `const` the id resolved through, so a real
    /// registration became no registration. This is the shape of
    /// `vyre-primitives/src/hash/adler32.rs`: a const id, a feature-gated
    /// production registration, then a test module.
    #[test]
    fn a_non_test_cfg_attribute_keeps_the_text_before_it() {
        let parsed = parse_registrations(
            r#"
            pub const ADLER32_OP_ID: &str = "vyre-primitives::hash::adler32";

            #[cfg(feature = "inventory-registry")]
            inventory::submit! {
                OperationRegistration::primitive(ADLER32_OP_ID, builder)
            }

            #[cfg(test)]
            mod tests {
                fn fixture() {
                    OperationRegistration::library("test::reference_echo");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-primitives::hash::adler32".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    /// A long id is written on the line after the `=`. A line-bound const scan
    /// resolved none of those, and the registration was dropped in silence.
    #[test]
    fn a_const_id_wrapped_onto_the_next_line_is_resolved() {
        let parsed = parse_registrations(
            r#"
            pub const I4_MATVEC_F32_SCALED_OP_ID: &str =
                "vyre-primitives::math::quantized::i4x8_matvec_f32_scaled";

            inventory::submit! {
                OperationRegistration::primitive(I4_MATVEC_F32_SCALED_OP_ID, builder)
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-primitives::math::quantized::i4x8_matvec_f32_scaled".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    /// Adversarial case for the const scan: reading the whole text rather than
    /// one line at a time reaches every `const`, including one whose value only
    /// measures a string. The declared type is what says an id, so a `usize`
    /// const resolves nothing even when a well-formed id sits in its value.
    #[test]
    fn a_const_that_is_not_a_string_resolves_no_id() {
        let parsed = parse_registrations(
            r#"
            const OP_ID: usize = "vyre-libs::hash::adler32".len();

            inventory::submit! {
                OperationRegistration::primitive(OP_ID, builder)
            }
            "#,
        );

        assert_eq!(parsed, Vec::new());
    }

    /// A brace inside a string literal used to end the struct literal early,
    /// which truncated the scanned window before `id:` and dropped the
    /// registration. The literal here carries one unbalanced `}`, which is what
    /// a brace-counting scan cannot survive.
    #[test]
    fn a_brace_inside_a_string_does_not_end_the_struct_literal() {
        let parsed = parse_registrations(
            r#"
            inventory::submit! {
                vyre_foundation::operation::OperationRegistration {
                    build: Some(|| shader("fn main() }")),
                    tier: vyre_foundation::operation::OperationTier::Library,
                    id: "vyre-libs::text::format",
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::text::format".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    /// The item a `#[cfg(test)]` gates ends at its matching brace, and a brace
    /// written inside a string is not that brace. Ending the module early left
    /// its tail in the scanned text, and the fixture registration in that tail
    /// was counted as a production operation.
    #[test]
    fn a_brace_inside_a_test_module_string_does_not_end_the_module_early() {
        let parsed = parse_registrations(
            r#"
            fn install() {
                OperationRegistration::library("vyre-libs::hash::crc32");
            }

            #[cfg(test)]
            mod tests {
                fn shader() -> &'static str {
                    "fn main() }"
                }

                fn fixture() {
                    OperationRegistration::library("test::reference_echo");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::crc32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    fn discarding_import(file: &str, named: &str) -> DiscardingImport {
        DiscardingImport {
            file: file.to_string(),
            named: named.to_string(),
        }
    }

    #[test]
    fn a_discarding_import_of_a_submitting_crate_is_rejected() {
        let failures = registry_link_failures(
            &["vyre-libs".to_string(), "vyre-driver-cuda".to_string()],
            &[discarding_import(
                "conform/vyre-conform/tests/ulp_audit.rs",
                "vyre_libs",
            )],
        );

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("vyre-libs"));
        assert!(failures[0].contains("ulp_audit.rs"));
        assert!(failures[0].contains(REGISTRY_LINK_OWNER));
    }

    #[test]
    fn a_discarding_import_of_a_crate_that_registers_nothing_is_accepted() {
        let failures = registry_link_failures(
            &["vyre-libs".to_string()],
            &[discarding_import("vyre/src/lib.rs", "vyre_spec")],
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_crate_identifier_matches_its_hyphenated_crate_name() {
        let failures = registry_link_failures(
            &["vyre-driver-reference".to_string()],
            &[discarding_import(
                "conform/vyre-conform/src/main.rs",
                "vyre_driver_reference",
            )],
        );

        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn a_trait_import_is_not_a_discarding_crate_import() {
        let named = discarding_imports("fn read() {\n    use std::io::Read as _;\n}\n");

        assert!(named.is_empty(), "{named:?}");
    }

    #[test]
    fn a_crate_named_only_inside_a_comment_is_not_an_import() {
        let named = discarding_imports(
            "/// Naming the crate with `use vyre_libs as _;` references nothing.\npub fn anchor() {}\n",
        );

        assert!(named.is_empty(), "{named:?}");
    }

    #[test]
    fn a_discarding_crate_import_is_read_from_source() {
        let named = discarding_imports(
            "#[cfg(feature = \"gpu\")]\nuse vyre_driver_metal as _;\nuse vyre_libs as _;\n",
        );

        assert_eq!(named, vec!["vyre_driver_metal", "vyre_libs"]);
    }


    #[test]
    fn a_submission_inside_a_comment_does_not_make_a_crate_a_submitter() {
        assert!(!submits_registrations(
            "// This crate reads the registry; inventory::submit! lives in the driver.\n"
        ));
        assert!(submits_registrations(
            "inventory::submit! {\n    ExampleRegistration { id: \"example\" }\n}\n"
        ));
    }

    fn paths(files: &[&str]) -> Vec<String> {
        files.iter().map(|file| (*file).to_string()).collect()
    }

    #[test]
    fn a_module_file_beside_its_own_directory_is_rejected() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule.rs",
            "vyre-libs/src/rule/admission.rs",
        ]));

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("vyre-libs/src/rule/mod.rs"), "{failures:?}");
    }

    #[test]
    fn a_module_file_is_judged_against_a_directory_holding_no_direct_source() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule.rs",
            "vyre-libs/src/rule/admission/window.rs",
        ]));

        assert_eq!(failures.len(), 1, "{failures:?}");
    }

    #[test]
    fn a_module_inside_its_own_directory_is_accepted() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule/mod.rs",
            "vyre-libs/src/rule/admission.rs",
            "vyre-libs/src/lib.rs",
        ]));

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_same_named_file_in_another_directory_is_not_a_pair() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule.rs",
            "vyre-driver/src/rule/admission.rs",
        ]));

        assert!(failures.is_empty(), "{failures:?}");
    }

    fn crate_roots(pairs: &[(&str, &str)]) -> Vec<CrateRoot> {
        pairs
            .iter()
            .map(|(directory, ident)| CrateRoot {
                directory: (*directory).to_string(),
                ident: (*ident).to_string(),
            })
            .collect()
    }

    #[test]
    fn every_banned_module_name_is_rejected_as_a_file_and_as_a_directory() {
        for name in BANNED_MODULE_NAMES {
            let flat = format!("vyre-libs/src/scan/{name}.rs");
            let nested = format!("vyre-libs/src/scan/{name}/mod.rs");
            let failures = generic_module_name_failures(
                &[flat, nested],
                &crate_roots(&[("vyre-libs", "vyre_libs")]),
                &[],
            );

            assert_eq!(failures.len(), 2, "{name}: {failures:?}");
            assert!(
                failures.iter().all(|failure| failure
                    .contains(&format!("vyre_libs::scan::{name}"))),
                "{name}: {failures:?}"
            );
        }
    }

    #[test]
    fn a_qualifier_suffix_is_rejected_as_a_file_and_as_a_directory() {
        let failures = generic_module_name_failures(
            &paths(&[
                "vyre-libs/src/scan/window_ext.rs",
                "vyre-libs/src/scan/region_ext/mod.rs",
                "xtask/src/bin/dump_ext.rs",
            ]),
            &crate_roots(&[("vyre-libs", "vyre_libs"), ("xtask", "xtask")]),
            &[],
        );

        assert_eq!(failures.len(), 3, "{failures:?}");
    }

    #[test]
    fn a_published_module_keeps_its_name() {
        let files = paths(&["vyre-libs/src/parsing/core/mod.rs"]);
        let roots = crate_roots(&[("vyre-libs", "vyre_libs")]);

        assert!(generic_module_name_failures(
            &files,
            &roots,
            &["vyre_libs::parsing::core".to_string()]
        )
        .is_empty());
        assert_eq!(
            generic_module_name_failures(&files, &roots, &["vyre_libs::parsing".to_string()]).len(),
            1
        );
    }

    #[test]
    fn the_exemption_is_keyed_on_the_name_the_manifest_declares() {
        let files = paths(&["fuzz/src/harness/types/mod.rs"]);
        let published = ["vyre_fuzz::harness::types".to_string()];

        assert!(
            generic_module_name_failures(
                &files,
                &crate_roots(&[("fuzz", "vyre_fuzz")]),
                &published
            )
            .is_empty(),
            "a published module lost its exemption because the crate was named after its directory"
        );
        assert_eq!(
            generic_module_name_failures(&files, &crate_roots(&[("fuzz", "fuzz")]), &published)
                .len(),
            1
        );
    }

    #[test]
    fn a_module_in_no_scanned_crate_is_still_reported() {
        let failures =
            generic_module_name_failures(&paths(&["stray/src/types/mod.rs"]), &[], &[]);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            !failures[0].contains("published at no public path"),
            "the message claimed a public path it could not resolve: {failures:?}"
        );
    }

    #[test]
    fn a_crate_root_carries_no_module_name() {
        assert_eq!(judged_name_of("vyre-libs/src/lib.rs"), None);
        assert_eq!(judged_name_of("conform/vyre-conform/src/main.rs"), None);
        assert_eq!(judged_name_of("vyre-libs/src/scan/mod.rs"), Some("scan"));
        assert_eq!(judged_name_of("vyre-libs/src/scan/window.rs"), Some("window"));
    }

    #[test]
    fn every_source_tree_is_judged_and_a_file_in_none_is_not() {
        for tree in SOURCE_TREES {
            assert_eq!(
                judged_name_of(&format!("vyre-libs/{tree}/parity/mod.rs")),
                Some("parity"),
                "the {tree} tree went unjudged"
            );
            assert_eq!(
                judged_name_of(&format!("vyre-libs/{tree}/parity_support.rs")),
                Some("parity_support")
            );
        }
        assert_eq!(judged_name_of("release/changes/support.rs"), None);
        assert_eq!(
            judged_name_of("vyre-libs/tests/support/nested/deep/util.rs"),
            Some("util"),
            "a file nested under a judged tree is judged by its own name"
        );
    }

    #[test]
    fn a_number_is_a_defect_only_when_it_distinguishes_siblings() {
        let family: Vec<String> = (0..3)
            .map(|index| format!("vyre-libs/src/classify/nodes_0{index}.rs"))
            .collect();
        let failures = numbered_sibling_failures(&family);
        assert_eq!(failures.len(), 3, "{failures:?}");
        assert!(failures[0].contains("3 `nodes_N` siblings"), "{failures:?}");

        for lone in [
            "vyre-libs/src/nn/attention/flash_attention_2.rs",
            "vyre-primitives/src/hash/crc32.rs",
            "vyre-primitives/src/math/float16.rs",
            "vyre-libs/src/classify/nodes_00.rs",
        ] {
            assert!(
                numbered_sibling_failures(&paths(&[lone])).is_empty(),
                "{lone} has no numbered sibling, so its digits carry meaning"
            );
        }
        assert!(
            numbered_sibling_failures(&paths(&[
                "vyre-libs/src/classify/nodes_00.rs",
                "vyre-libs/src/emit/nodes_01.rs",
            ]))
            .is_empty(),
            "siblings are per directory; two directories are not one family"
        );
    }

    #[test]
    fn a_ticket_range_is_rejected_and_a_single_number_is_not() {
        let failures = generic_module_name_failures(
            &paths(&["vyre-foundation/tests/validation_findings_12_20.rs"]),
            &crate_roots(&[("vyre-foundation", "vyre_foundation")]),
            &[],
        );
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("ticket range `12_20`"), "{failures:?}");
        assert!(
            generic_module_name_failures(
                &paths(&["xtask-evidence/src/semantic/proof_workloads_12.rs"]),
                &crate_roots(&[("xtask-evidence", "xtask_evidence")]),
                &[],
            )
            .is_empty(),
            "one number can be a count or a size; a range names a ticket"
        );
    }

    #[test]
    fn a_file_repeating_its_directory_is_rejected() {
        let failures = directory_stutter_failures(&paths(&[
            "vyre-intrinsics/src/hardware/fma_f32/fma_f32.rs",
            "vyre-intrinsics/src/hardware/fma_f32/mod.rs",
            "vyre-intrinsics/src/hardware/fma_f32/lowering.rs",
        ]));
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("fma_f32/mod.rs`"),
            "the failure must name where the file belongs: {failures:?}"
        );
    }

    #[test]
    fn a_binary_root_is_judged_by_the_binary_name() {
        for name in BANNED_MODULE_NAMES {
            let flat = format!("xtask/src/bin/{name}.rs");
            let nested = format!("xtask/src/bin/{name}/main.rs");
            let failures = generic_module_name_failures(
                &[flat, nested],
                &crate_roots(&[("xtask", "xtask")]),
                &["xtask::bin".to_string()],
            );

            assert_eq!(failures.len(), 2, "{name}: {failures:?}");
            assert!(
                failures
                    .iter()
                    .all(|failure| failure.contains(&format!("names the binary `{name}`"))),
                "{name}: {failures:?}"
            );
        }
    }

    #[test]
    fn a_named_binary_and_its_own_modules_are_told_apart() {
        assert_eq!(binary_name_of("xtask/src/bin/scaffold_rule.rs"), Some("scaffold_rule"));
        assert_eq!(
            binary_name_of("xtask-registry/src/bin/vyre_new_op/main.rs"),
            Some("vyre_new_op")
        );
        assert_eq!(binary_name_of("xtask-registry/src/bin/vyre_new_op/run.rs"), None);
        assert_eq!(binary_name_of("vyre-libs/src/scan/window.rs"), None);
        assert_eq!(judged_name_of("xtask/src/bin/scaffold_rule.rs"), None);
        assert_eq!(
            judged_name_of("xtask-registry/src/bin/vyre_new_op/helpers.rs"),
            Some("helpers")
        );
    }

    #[test]
    fn a_descriptively_named_binary_is_accepted() {
        let failures = generic_module_name_failures(
            &paths(&[
                "xtask/src/bin/scaffold_rule.rs",
                "xtask-registry/src/bin/vyre_new_op/main.rs",
            ]),
            &crate_roots(&[("xtask", "xtask"), ("xtask-registry", "xtask_registry")]),
            &[],
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_module_path_is_read_through_the_src_boundary() {
        let roots = crate_roots(&[
            ("conform/vyre-conform", "vyre_conform"),
            ("vyre-libs", "vyre_libs"),
        ]);

        assert_eq!(
            module_path_of("conform/vyre-conform/src/report/common/mod.rs", &roots).as_deref(),
            Some("vyre_conform::report::common")
        );
        assert_eq!(
            module_path_of("vyre-libs/src/types.rs", &roots).as_deref(),
            Some("vyre_libs::types")
        );
        assert_eq!(module_path_of("vyre-libs/src/types.rs", &[]), None);
    }

    #[test]
    fn a_descriptive_module_name_is_accepted() {
        let failures = generic_module_name_failures(
            &paths(&[
                "vyre-libs/src/scan/regex_dfa.rs",
                "vyre-libs/src/graph/dispatch/mod.rs",
                "vyre-libs/src/lib.rs",
            ]),
            &crate_roots(&[("vyre-libs", "vyre_libs")]),
            &[],
        );

        assert!(failures.is_empty(), "{failures:?}");
    }
}
