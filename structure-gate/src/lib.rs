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
use std::path::{Path, PathBuf};
use std::process;

use toml::Value;
use walkdir::WalkDir;

pub mod backend_vocabulary;
pub mod cfg_test;
pub mod crate_ownership;
pub mod module_layout;
pub mod registration_text;
pub mod source_scan;
pub mod geometry_constants;
pub use geometry_constants::geometry_constant_failures;

use crate::module_layout::{
    directory_stutter_failures, generic_module_name_failures, numbered_sibling_failures,
    sibling_module_failures, CrateRoot, PUBLIC_API_SNAPSHOT_DIR, SOURCE_TREES,
};
use crate::registration_text::parse_registrations;
use crate::source_scan::code_offsets;

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
    failures.extend(sibling_module_failures(&workspace.module_files));
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

/// Reject one semantic operation carrying two identities, and ids that name a
/// crate the workspace does not have.
///
/// The namespace of an id is the crate that minted it, frozen from then on.
/// This rule used to require it to equal the crate the registration lives in,
/// which reported all 130 operations that moved to `vyre-libs` keeping their
/// `vyre-primitives::` ids. Where an operation lives is
/// [`registration_owner_failures`] and [`category_home_failures`], both of
/// which read the file the registration is written in. What is left here is
/// what the id itself can answer: two crates must not claim one kernel, and a
/// namespace must name a member the workspace carries.
pub fn operation_identity_failures(
    registrations: &[Registration],
    members: &[String],
) -> Vec<String> {
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
    let member_crates: std::collections::BTreeSet<&str> = members
        .iter()
        .map(|member| member.rsplit('/').next().unwrap_or(member))
        .collect();
    for reg in registrations {
        let claimed = reg.claimed_crate();
        if claimed.starts_with("vyre-") && !member_crates.contains(claimed) {
            failures.push(format!(
                "{} registers `{}`, whose namespace names `{claimed}`; no workspace member carries that name, so the id was minted by a crate that never existed or was renamed without a migration",
                reg.file, reg.op_id
            ));
        }
    }
    failures
}

/// Reject a Category A operation in the Category C crate and the reverse.
///
/// Both sides are read from the tree. The tier is the one the registration
/// declares in its own source, and the home is the crate whose `src` holds the
/// file that registration is written in. Neither is the operation id: the id
/// namespace is frozen at mint time, so 130 operations that moved crate still
/// spell the crate they left.
pub fn category_home_failures(registrations: &[Registration]) -> Vec<String> {
    let mut failures = Vec::new();
    for reg in registrations {
        let Some(tier) = reg.tier.as_deref() else {
            continue;
        };
        let hardware = matches!(tier, "Intrinsic" | "Hardware");
        if hardware && reg.crate_name == CATEGORY_A_CRATE && !reg.op_id.starts_with("vyre-primitives::") {
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
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed. Every gate in this
/// crate answers for the roster that manifest declares, so a gate that carried
/// on with an empty roster would report a clean tree it never read.
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
/// `[package]` and holding Rust source under `src/`, so a crate added anywhere
/// in the checkout is judged without an edit here. A `src/` emptied by a
/// deletion is not a crate root: the directory survives the pull that removed
/// every file in it. This crate is included; [`source_files`]
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
            crate::source_scan::carries_rust_source(&directory.join("src")).then(|| CrateRoot {
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

    /// Every workspace member the identity rule judges against.
    fn roster() -> Vec<String> {
        ALLOWED_MEMBERS
            .iter()
            .map(|name| (*name).to_string())
            .collect()
    }

    #[test]
    fn one_kernel_registered_under_two_namespaces_is_rejected() {
        let failures = operation_identity_failures(
            &[
                registration("vyre-foundation", "vyre-foundation::hash::adler32", None),
                registration("vyre-libs", "vyre-libs::hash::adler32", None),
            ],
            &roster(),
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("2 identities")),
            "{failures:?}"
        );
    }

    #[test]
    fn same_leaf_under_one_namespace_is_accepted() {
        let failures = operation_identity_failures(
            &[
                registration("vyre-foundation", "vyre-foundation::hash::adler32", None),
                registration("vyre-foundation", "vyre-foundation::graph::toposort", None),
            ],
            &roster(),
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    /// A frozen id keeps the namespace of the crate that minted it.
    ///
    /// Requiring the namespace to equal the crate the registration lives in
    /// reported all 130 operations that moved to `vyre-libs` and kept their
    /// `vyre-primitives::` ids. Where an operation lives is judged by the rules
    /// that read the file it is written in.
    #[test]
    fn an_operation_that_moved_crate_keeps_its_minting_namespace() {
        let failures = operation_identity_failures(
            &[registration(
                "vyre-libs",
                "vyre-primitives::graph::toposort",
                None,
            )],
            &roster(),
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn an_id_naming_no_workspace_member_is_rejected() {
        let failures = operation_identity_failures(
            &[registration(
                "vyre-libs",
                "vyre-departed::hash::adler32",
                None,
            )],
            &roster(),
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("no workspace member carries that name")),
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
}
