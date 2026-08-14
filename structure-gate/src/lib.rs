//! Workspace structural gate: crate roster, one operation identity per
//! semantic operation, and one home per concept.
//!
//! Run it with `cargo run -p structure-gate`. It reads source text and depends
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

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use toml::Value;
use walkdir::WalkDir;

/// Category A owner: every composition. `docs/ARCHITECTURE.md`, "Target
/// operation crate structure", decided 2026-08-12.
const CATEGORY_A_CRATE: &str = "vyre-libs";
/// Category C owner: strict hardware intrinsics, one emitter arm and one
/// reference-interpreter arm each. Absorbs `vyre-intrinsics` at migration.
const CATEGORY_C_CRATE: &str = "vyre-primitives";

/// Directory that owns every module named `*substrate*`.
///
/// `vyre_foundation::pass_substrate` is exempt because it owns the CPU pass
/// math outright: the GPU crate imports those functions and wraps them in
/// dispatch rather than reimplementing them. The exemption is about the name
/// only, and it retires when the three `*substrate*` concepts are renamed.
const SUBSTRATE_HOME: &str = "vyre-self-substrate/src/optimizer";
const SUBSTRATE_EXCEPTIONS: &[&str] = &["vyre-foundation/src/pass_substrate"];

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
    "vyre-frontend-rust",
    "vyre-grammar-gen",
    "vyre-libs",
    "vyre-lints",
    "vyre-lower",
    "vyre-macros",
    "vyre-megakernel",
    "vyre-primitives",
    "vyre-reference",
    "vyre-runtime",
    "vyre-safetensors",
    // Narrows to the GPU pass engine and is renamed at migration; the roster
    // moves with the rename.
    "vyre-self-substrate",
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
}

/// Read the workspace rooted at `root` into the structural model.
#[must_use]
pub fn scan(root: &Path) -> Workspace {
    let members = workspace_members(root);
    let registrations = scan_registrations(root, &members);
    let substrate_paths = scan_substrate_paths(root, &members);
    let frontend_paths = scan_frontend_paths(root, &members);
    let materializers = scan_materializers(root, &members);
    Workspace {
        members,
        registrations,
        substrate_paths,
        frontend_paths,
        materializers,
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
        // `Primitive` is the pre-rename spelling of `Intrinsic`; the registry
        // rename lands with the migration. Reading only the new name here
        // would report every current registration as a category violation and
        // bury the real ones.
        let hardware = matches!(tier, "Intrinsic" | "Hardware" | "Primitive");
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
/// The GPU pass engine owns the name. `SUBSTRATE_EXCEPTIONS` carries the
/// duplications `docs/ARCHITECTURE.md` sanctions by name; anything else is a
/// second home.
pub fn substrate_home_failures(paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| !path.starts_with(SUBSTRATE_HOME))
        .filter(|path| {
            !SUBSTRATE_EXCEPTIONS
                .iter()
                .any(|exception| path.starts_with(exception))
        })
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

/// Names of the admission helpers `vyre-driver` owns for every backend.
const SHARED_ADMISSION_HELPERS: &[&str] = &["invalid_module", "compile_error"];

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
        if !text.contains("materialize::admit(") {
            failures.push(format!(
                "`{path}` does not admit its target payload through `vyre_driver::materialize::admit`"
            ));
        }
    }
    failures
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

fn workspace_members(root: &Path) -> Vec<String> {
    let manifest = root.join("Cargo.toml");
    let text = read_source_bounded(&manifest)
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
/// misplaced compositions and buried the real findings.
const CONSTRUCTOR_TIERS: &[(&str, Option<&str>)] = &[
    ("::primitive(", Some("Primitive")),
    ("::library(", Some("Library")),
    ("::new(", None),
];

/// Remove every `#[cfg(test)]`-gated item before the registration scan.
///
/// A test module registers fixture operations - `test::reference_echo`,
/// `test::call_u32` and friends - that exist in no production build. Counting
/// them as registry members reported four test doubles as misplaced production
/// operations, and pointed Phase 2 at code that was already correct.
///
/// The predicate is tokenized with string literals removed first, so
/// `#[cfg(feature = "test-utils")]` is not mistaken for a test gate.
fn strip_cfg_test_items(text: &str) -> Cow<'_, str> {
    const ATTR: &str = "#[cfg(";
    if !text.contains(ATTR) {
        return Cow::Borrowed(text);
    }
    let mut out: Option<String> = None;
    let mut cursor = 0usize;
    while let Some(offset) = text[cursor..].find(ATTR) {
        let attr_start = cursor + offset;
        let predicate_start = attr_start + ATTR.len() - 1;
        let Some(predicate_end) = match_delimited(text, predicate_start, b'(', b')') else {
            break;
        };
        if !mentions_test(&text[predicate_start + 1..predicate_end]) {
            cursor = predicate_end + 1;
            continue;
        }
        let Some(attr_end) = text[predicate_end..].find(']').map(|at| predicate_end + at) else {
            break;
        };
        let Some(item_end) = end_of_item(text, attr_end + 1) else {
            break;
        };
        out.get_or_insert_with(String::new)
            .push_str(&text[cursor..attr_start]);
        cursor = item_end;
    }
    match out {
        Some(mut kept) => {
            kept.push_str(&text[cursor..]);
            Cow::Owned(kept)
        }
        None => Cow::Borrowed(text),
    }
}

/// Byte index of the delimiter closing the one that opens at `open`.
fn match_delimited(text: &str, open: usize, opener: u8, closer: u8) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open) != Some(&opener) {
        return None;
    }
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate().skip(open) {
        if *byte == opener {
            depth += 1;
        } else if *byte == closer {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
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
        let constructor = CONSTRUCTOR_TIERS
            .iter()
            .find(|(call, _)| after.trim_start().starts_with(call.trim_start_matches("::")) || after.starts_with(call));
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
/// Splits on top-level commas only, so a nested `Some(f(a, b))` argument does
/// not shift the count.
fn nth_argument<'a>(after: &'a str, call: &str, index: usize) -> Option<&'a str> {
    let open = after.find(call)? + call.len();
    let rest = &after[open..];
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut argument = 0usize;
    for (offset, byte) in rest.char_indices() {
        match byte {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' if depth > 0 => depth -= 1,
            ')' => {
                return (argument == index).then(|| rest[start..offset].trim());
            }
            ',' if depth == 0 => {
                if argument == index {
                    return Some(rest[start..offset].trim());
                }
                argument += 1;
                start = offset + 1;
            }
            _ => {}
        }
    }
    None
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
            if path.file_name().is_some_and(|name| name == "materializer.rs") {
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
            "vyre-self-substrate",
            "vyre-self-substrate::graph::toposort",
            Some("Library"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("vyre-self-substrate"));
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
        let failures = substrate_home_failures(&[
            "vyre-self-substrate/src/optimizer/dispatcher.rs".to_string(),
            "vyre-self-substrate/src/scheduling/homotopy_ilp.rs".to_string(),
            "vyre-libs/src/substrate_catalog.rs".to_string(),
        ]);

        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(failures.iter().any(|f| f.contains("scheduling")));
        assert!(failures.iter().any(|f| f.contains("substrate_catalog")));
    }

    #[test]
    fn the_sanctioned_foundation_pass_substrate_is_accepted() {
        // ARCHITECTURE.md: foundation owns the CPU pass math and the GPU crate
        // imports it, so this name is exempt. Every other second home stays a
        // failure.
        let failures = substrate_home_failures(&[
            "vyre-foundation/src/pass_substrate/dataflow_fixpoint.rs".to_string(),
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
            ("vyre-libs".to_string(), "vyre-libs/src/parsing/c/lex/keyword.rs".to_string()),
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
        assert!(failures[0].contains("puts a c frontend in vyre-driver-wgpu"), "{failures:?}");
    }

    #[test]
    fn the_declared_owner_of_a_language_is_accepted() {
        // Control: naming a frontend is only a failure for a non-owner, and
        // the rust owner is a crate whose own name declares the language.
        let failures = frontend_owner_failures(&[
            ("vyre-frontend-rust".to_string(), "vyre-frontend-rust".to_string()),
            ("vyre-libs".to_string(), "vyre-libs/src/parsing/c/lex/keyword.rs".to_string()),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
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
                Some("Primitive".to_string())
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
            use vyre_primitives::hash::adler32::{adler32_program, ADLER32_OP_ID};

            #[cfg(test)]
            use crate::buffer_names::fixed_name;
            #[cfg(test)]
            use vyre_primitives::hash::adler32::adler32 as adler32_cpu_reference;

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
}
