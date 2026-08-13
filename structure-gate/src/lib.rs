//! Workspace structural gate: crate roster, one operation identity per
//! semantic operation, and one home per concept.
//!
//! Run it with `cargo run -p structure-gate`. It reads source text and depends
//! on no vyre crate, so it still judges the workspace while the workspace does
//! not compile.
//!
//! The workspace has exactly two operation-owning crates:
//!
//! - `vyre-foundation` owns typed IR and **every Category A operation**: any
//!   `fn(..) -> Program` built from existing IR variants.
//! - `vyre-libs` owns **every Category C operation**: an operation that needs a
//!   dedicated hardware contract, plus compositions built over those.
//!
//! Any third crate that registers an operation splits an operation identity in
//! two, which is the defect this gate exists to make impossible. A tier
//! boundary is not a reason to re-register a kernel under a second id.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use toml::Value;
use walkdir::WalkDir;

/// The only workspace members allowed to register an operation.
const CATEGORY_A_CRATE: &str = "vyre-foundation";
/// Category C owner: hardware-contract operations and compositions over them.
const CATEGORY_C_CRATE: &str = "vyre-libs";

/// Directory that owns every module named `*substrate*`.
const SUBSTRATE_HOME: &str = "vyre-foundation/src/substrate";

/// Closed workspace roster. A new member is a reviewable change here first.
const ALLOWED_MEMBERS: &[&str] = &[
    "conform/vyre-conform",
    "conform/vyre-conform-spec",
    "vyre",
    "vyre-aot",
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
    "vyre-frontend-c",
    "vyre-frontend-rust",
    "vyre-grammar-gen",
    "vyre-libs",
    "vyre-lints",
    "vyre-lower",
    "vyre-macros",
    "vyre-megakernel",
    "vyre-reference",
    "vyre-runtime",
    "vyre-safetensors",
    "vyre-spec",
    "vyre-test-support",
    "structure-gate",
    "xtask",
];

/// Source languages and the single crate that owns each frontend.
const FRONTEND_OWNERS: &[(&str, &str)] = &[("c", "vyre-frontend-c"), ("rust", "vyre-frontend-rust")];

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
}

/// Read the workspace rooted at `root` into the structural model.
#[must_use]
pub fn scan(root: &Path) -> Workspace {
    let members = workspace_members(root);
    let registrations = scan_registrations(root, &members);
    let substrate_paths = scan_substrate_paths(root, &members);
    let frontend_paths = scan_frontend_paths(root, &members);
    Workspace {
        members,
        registrations,
        substrate_paths,
        frontend_paths,
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
    failures
}

/// Workspace root, resolved from the xtask manifest directory.
#[must_use]
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .expect("Fix: xtask must live under the vyre workspace root.")
}

/// Run the crate-structure gate.
pub fn run(args: &[String]) {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "USAGE:\n  cargo run -p structure-gate\n\n\
             Fails when a crate outside vyre-foundation (Category A) or vyre-libs \
             (Category C) registers an operation, when one semantic operation is \
             registered under two identities, when a concept has more than one home, \
             or when the workspace roster drifts."
        );
        return;
    }

    let root = workspace_root();
    let failures = violations(&root);

    if failures.is_empty() {
        println!("crate-structure: roster, operation identity, and concept homes agree");
        return;
    }

    eprintln!("crate-structure: {} violation(s):", failures.len());
    for failure in &failures {
        eprintln!("  - {failure}");
    }
    eprintln!(
        "Fix: move the operation to its category owner ({CATEGORY_A_CRATE} for Category A, \
         {CATEGORY_C_CRATE} for Category C), delete the duplicate registration, and update \
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
pub fn frontend_owner_failures(paths: &[(String, String)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (language, owner) in FRONTEND_OWNERS {
        for (found_crate, path) in paths {
            if path_names_language(path, language) && found_crate != owner {
                failures.push(format!(
                    "`{path}` puts a {language} frontend in {found_crate}; {owner} owns the {language} frontend"
                ));
            }
        }
    }
    failures
}

fn path_names_language(path: &str, language: &str) -> bool {
    path.split(['/', '.'])
        .any(|segment| segment.eq_ignore_ascii_case(language))
}


fn workspace_members(root: &Path) -> Vec<String> {
    let manifest = root.join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest.display()));
    let table: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: parse {}: {error}", manifest.display()));
    Value::Table(table)
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(Value::as_array)
        .map(|members| {
            members
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
    WalkDir::new(root.join(member).join("src"))
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .collect()
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
            let Ok(text) = fs::read_to_string(&path) else {
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
const CONSTRUCTOR_TIERS: &[(&str, &str)] = &[
    ("::primitive(", "Library"),
    ("::library(", "Library"),
    ("::new(", "Library"),
];

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
pub fn parse_registrations(text: &str) -> Vec<(String, Option<String>)> {
    let consts = string_consts(text);
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("OperationRegistration") {
        let body = &rest[start..];
        let after = &body["OperationRegistration".len()..];
        let constructor = CONSTRUCTOR_TIERS
            .iter()
            .find(|(call, _)| after.trim_start().starts_with(call.trim_start_matches("::")) || after.starts_with(call));
        if let Some((call, tier)) = constructor {
            if let Some(id) = first_argument(after, call).and_then(|raw| resolve_id(raw, &consts)) {
                found.push((id, Some((*tier).to_string())));
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
    let open = after.find(call)? + call.len();
    let rest = &after[open..];
    let end = rest.find(',').or_else(|| rest.find(')'))?;
    Some(rest[..end].trim())
}

/// Byte offset just past the struct literal that opens in `body`.
///
/// Registration fields hold closures, so the first `}` is almost never the end
/// of the literal. Counting depth is what keeps `id:` and `tier:` inside the
/// scanned window; stopping at the first brace silently drops most
/// registrations and makes every registration rule pass on an empty set.
fn struct_literal_end(body: &str) -> usize {
    let mut depth = 0usize;
    let mut opened = false;
    for (offset, byte) in body.char_indices() {
        match byte {
            '{' => {
                depth += 1;
                opened = true;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if opened && depth == 0 {
                    return offset + 1;
                }
            }
            _ => {}
        }
    }
    body.len()
}

/// Map every `const NAME: &str = "value";` in a file to its literal.
fn string_consts(text: &str) -> BTreeMap<String, String> {
    let mut consts = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("const ").or_else(|| {
            line.strip_prefix("pub const ")
                .or_else(|| line.strip_prefix("pub(crate) const "))
        }) else {
            continue;
        };
        let Some((name, value)) = rest.split_once('=') else {
            continue;
        };
        let name = name.split(':').next().unwrap_or(name).trim();
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

fn scan_frontend_paths(root: &Path, members: &[String]) -> Vec<(String, String)> {
    let mut paths = Vec::new();
    for member in members {
        let crate_name = member.rsplit('/').next().unwrap_or(member).to_string();
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
            "vyre-primitives",
            "vyre-primitives::hash::adler32",
            Some("Library"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("vyre-primitives"));
    }

    #[test]
    fn the_two_category_owners_are_accepted() {
        let failures = registration_owner_failures(&[
            registration("vyre-foundation", "vyre-foundation::hash::adler32", None),
            registration("vyre-libs", "vyre-libs::atomic::compare_exchange", None),
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
            "vyre-foundation",
            "vyre-foundation::atomic::compare_exchange",
            Some("Intrinsic"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("Category C"));
    }

    #[test]
    fn a_composition_in_the_category_c_crate_is_rejected() {
        let failures = category_home_failures(&[registration(
            "vyre-libs",
            "vyre-libs::hash::adler32",
            Some("Library"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("Category A"));
    }

    #[test]
    fn each_category_in_its_own_home_is_accepted() {
        let failures = category_home_failures(&[
            registration(
                "vyre-foundation",
                "vyre-foundation::hash::adler32",
                Some("Library"),
            ),
            registration(
                "vyre-libs",
                "vyre-libs::atomic::compare_exchange",
                Some("Intrinsic"),
            ),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_second_substrate_home_is_rejected() {
        let failures = substrate_home_failures(&[
            "vyre-foundation/src/substrate/graph/toposort.rs".to_string(),
            "vyre-foundation/src/pass_substrate/dataflow_fixpoint.rs".to_string(),
            "vyre-libs/src/substrate_catalog.rs".to_string(),
        ]);

        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(failures.iter().any(|f| f.contains("pass_substrate")));
        assert!(failures.iter().any(|f| f.contains("substrate_catalog")));
    }

    #[test]
    fn a_second_c_frontend_is_rejected() {
        let failures = frontend_owner_failures(&[
            ("vyre-frontend-c".to_string(), "vyre-frontend-c/src/lex/lexer.rs".to_string()),
            ("vyre-libs".to_string(), "vyre-libs/src/parsing/c/lex/keyword.rs".to_string()),
        ]);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("vyre-libs"));
    }

    #[test]
    fn a_product_crate_on_the_roster_is_rejected() {
        let failures = roster_failures(&[
            "vyre-foundation".to_string(),
            "vyre-scan".to_string(),
        ]);

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
                Some("Library".to_string())
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
}
