//! The neutral-vocabulary contract: whether a crate in a substrate-neutral layer
//! describes its own work in one vendor's words.
//!
//! A neutral crate that says `CUDA` where it means `backend`, or `WGSL` where it
//! means primary text, is where a rule meant for every target ends up written for
//! one. The code follows the prose: the next reader extends the vendor case and
//! leaves the others, and by then the drift is in the behaviour rather than the
//! comment. This rule reports the prose.
//!
//! Nothing the rule needs is compiled in. `backend-vocabulary.toml` carries the
//! layer decisions, the banned words, the neutral replacements and the two
//! exemptions; the roster is that layer table crossed with the layer each member
//! declares in `docs/CRATE_OWNERSHIP.toml`. So a new neutral crate is scanned
//! without an edit here, and a new dialect is a data row.
//!
//! Every row is held against the live tree on the same run that uses it, because
//! a wordlist nobody checks is a wordlist that quietly stops covering things: a
//! layer no member declares, a member layer with no decision, an exemption
//! outside a neutral layer, a term owned by a neutral layer, and an interface
//! allowance whose name appears nowhere under its own directory are findings.
//!
//! What this does not judge: test code, which names the backend it drives; an
//! explicit `pub use` of another crate's item, which is a reviewed surface
//! decision rather than an old spelling kept alive; and a neutral word used
//! wrongly, which no word list can see.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use toml::Value;

use crate::crate_ownership::{REGISTRY as OWNERSHIP_REGISTRY, Registry};
use crate::source_scan::mask_comments_and_strings;
use crate::{cfg_test_line_mask, read_source_bounded, relative, source_tree_files};

/// The contract data, inside the directory of the crate that owns the rule.
const DATA_FILE: &str = "backend-vocabulary.toml";

/// One concrete backend, vendor or dialect name a neutral crate may not write.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Term {
    /// The name, as an author would spell it.
    pub word: String,
    /// Layer that owns the concrete detail the word names.
    pub owner_layer: String,
    /// What a neutral crate says instead.
    pub neutral: String,
}

/// One external interface name a directory may write.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Interface {
    /// Checkout-relative directory prefix the allowance reaches.
    pub prefix: String,
    /// The name, spelled as the interface spells it.
    pub name: String,
    /// Why the name cannot be restated in neutral words.
    pub reason: String,
}

/// One substrate-neutral layer excused from the vocabulary rule, and why.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExemptLayer {
    /// Layer name, as the ownership registry spells it.
    pub layer: String,
    /// Why a crate in this layer may name a backend.
    pub reason: String,
}

/// The contract as `backend-vocabulary.toml` states it.
#[derive(Clone, Debug, Default)]
pub struct Contract {
    /// The words a neutral crate says instead of a vendor's.
    pub neutral_terms: Vec<String>,
    /// Whether each declared layer is substrate-neutral.
    pub layers: Vec<(String, bool)>,
    /// Neutral layers excused from the vocabulary rule.
    pub exempt_layers: Vec<ExemptLayer>,
    /// Banned names and their replacements.
    pub terms: Vec<Term>,
    /// External interface names allowed under one directory each.
    pub interfaces: Vec<Interface>,
}

/// The contract, plus what the ownership registry says about every member.
#[derive(Clone, Debug, Default)]
pub struct Neutrality {
    /// The data file, parsed.
    pub contract: Contract,
    /// Declared layer per member package.
    pub layers: BTreeMap<String, String>,
    /// Checkout-relative directory per member package.
    pub directories: BTreeMap<String, String>,
}

impl Neutrality {
    /// Read the contract and the ownership registry from the checkout at `root`.
    ///
    /// # Errors
    ///
    /// When either file is missing, unreadable, not TOML, or shaped so the rule
    /// cannot be built from it. Every one of those makes the rule silently cover
    /// nothing, so it is reported rather than defaulted.
    pub fn read(root: &Path) -> Result<Self, String> {
        let contract = read_contract(root)?;
        let (layers, directories) = read_registry(root)?;
        Ok(Self {
            contract,
            layers,
            directories,
        })
    }

    /// Whether a layer is substrate-neutral, or `None` when no row decides.
    #[must_use]
    pub fn layer_is_neutral(&self, layer: &str) -> Option<bool> {
        self.contract
            .layers
            .iter()
            .find(|(name, _)| name == layer)
            .map(|(_, neutral)| *neutral)
    }

    /// Whether a member's declared layer is substrate-neutral.
    ///
    /// A member the registry does not carry, or a layer with no decision, is not
    /// neutral: the layer-decision rule reports both, and treating an unjudged
    /// crate as neutral would report every backend name in it as well.
    #[must_use]
    pub fn package_is_neutral(&self, package: &str) -> bool {
        self.layers
            .get(package)
            .and_then(|layer| self.layer_is_neutral(layer))
            .unwrap_or(false)
    }

    /// Whether a layer is excused from the vocabulary rule.
    #[must_use]
    pub fn layer_is_exempt(&self, layer: &str) -> bool {
        self.contract
            .exempt_layers
            .iter()
            .any(|exempt| exempt.layer == layer)
    }

    /// Every member the vocabulary rule scans, as `(package, directory)`.
    ///
    /// Derived rather than listed: a member is in because its declared layer is
    /// neutral and not excused, so a crate that joins a neutral layer is scanned
    /// on the next run.
    #[must_use]
    pub fn roster(&self) -> Vec<(String, String)> {
        self.layers
            .iter()
            .filter(|(_, layer)| {
                self.layer_is_neutral(layer) == Some(true) && !self.layer_is_exempt(layer)
            })
            .filter_map(|(package, _)| {
                self.directories
                    .get(package)
                    .map(|directory| (package.clone(), directory.clone()))
            })
            .collect()
    }

    /// The names `line` states, in contract order, without repeats.
    ///
    /// Compared segment by segment rather than by substring. A name is a run of
    /// identifier segments, split on every non-alphanumeric byte and at camel-case
    /// boundaries, so `CudaDevice` names `CUDA`, `barracuda` does not, and a word
    /// spelled with a separator matches the run its own spelling splits into.
    /// Substring matching would need an allowance for every unrelated identifier
    /// that happens to carry a vendor's letters, and camel case is where a backend
    /// type name hides from a whole-word rule.
    #[must_use]
    pub fn words_in(&self, line: &str) -> Vec<&Term> {
        let segments = segments_of(line);
        self.contract
            .terms
            .iter()
            .filter(|term| {
                let wanted = segments_of(&term.word);
                !wanted.is_empty()
                    && segments
                        .windows(wanted.len())
                        .any(|run| run == wanted.as_slice())
            })
            .collect()
    }

    /// `line` with every interface name allowed for its directory blanked to
    /// spaces of the same width.
    ///
    /// Blanked rather than removed so a reported column still maps to the source,
    /// and so a blanked name cannot join its neighbours into a word that was never
    /// there.
    #[must_use]
    pub fn mask_interface_names(&self, line: &str, file: &str) -> String {
        let mut masked = line.to_string();
        for interface in self
            .contract
            .interfaces
            .iter()
            .filter(|interface| file.starts_with(&interface.prefix))
        {
            while let Some(at) = masked.find(&interface.name) {
                masked.replace_range(
                    at..at + interface.name.len(),
                    &" ".repeat(interface.name.len()),
                );
            }
        }
        masked
    }
}

/// Read and parse the contract data file.
fn read_contract(root: &Path) -> Result<Contract, String> {
    let path = crate::member_directory(root, crate::SELF_CRATE).join(DATA_FILE);
    let text = read_source_bounded(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let table: toml::Table = toml::from_str(&text)
        .map_err(|error| format!("{} is not readable as TOML: {error}", path.display()))?;
    let document = Value::Table(table);
    let label = DATA_FILE;

    let neutral_terms = document
        .get("vocabulary")
        .and_then(|vocabulary| vocabulary.get("neutral_terms"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} declares no [vocabulary] neutral_terms array"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{label} has a non-string entry in neutral_terms"))
        })
        .collect::<Result<Vec<String>, String>>()?;
    if neutral_terms.is_empty() {
        return Err(format!(
            "{label} lists no neutral term, so no banned word could state its replacement"
        ));
    }

    let mut layers = Vec::new();
    for row in rows(&document, "layer", label)? {
        let name = string_field(row, "name", "layer", label)?;
        let neutral = row
            .get("neutral")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("{label} [[layer]] `{name}` declares no boolean `neutral`"))?;
        layers.push((name, neutral));
    }
    if layers.is_empty() {
        return Err(format!(
            "{label} decides no layer, so the roster would be empty and the rule would pass forever"
        ));
    }

    let mut exempt_layers = Vec::new();
    for row in rows(&document, "exempt_layer", label)? {
        exempt_layers.push(ExemptLayer {
            layer: string_field(row, "layer", "exempt_layer", label)?,
            reason: string_field(row, "reason", "exempt_layer", label)?,
        });
    }

    let mut terms = Vec::new();
    for row in rows(&document, "term", label)? {
        terms.push(Term {
            word: string_field(row, "word", "term", label)?,
            owner_layer: string_field(row, "owner_layer", "term", label)?,
            neutral: string_field(row, "neutral", "term", label)?,
        });
    }
    if terms.is_empty() {
        return Err(format!(
            "{label} bans no word, so the vocabulary rule could not fail on any source"
        ));
    }

    let mut interfaces = Vec::new();
    for row in rows(&document, "interface", label)? {
        interfaces.push(Interface {
            prefix: string_field(row, "prefix", "interface", label)?,
            name: string_field(row, "name", "interface", label)?,
            reason: string_field(row, "reason", "interface", label)?,
        });
    }

    Ok(Contract {
        neutral_terms,
        layers,
        exempt_layers,
        terms,
        interfaces,
    })
}

/// The rows of one array-of-tables, or an empty slice when the key is absent.
fn rows<'a>(document: &'a Value, key: &str, label: &str) -> Result<&'a [Value], String> {
    match document.get(key) {
        None => Ok(&[]),
        Some(value) => value.as_array().map(Vec::as_slice).ok_or_else(|| {
            format!("{label} declares `{key}` as something other than a table array")
        }),
    }
}

/// One required string field of one row.
fn string_field(row: &Value, key: &str, kind: &str, label: &str) -> Result<String, String> {
    row.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{label} has a [[{kind}]] row with no string `{key}`"))
}

/// Declared layer and checkout-relative directory per member, from the registry.
fn read_registry(
    root: &Path,
) -> Result<(BTreeMap<String, String>, BTreeMap<String, String>), String> {
    let registry = Registry::read(root)?;
    let mut layers = BTreeMap::new();
    let mut directories = BTreeMap::new();
    for row in registry.rows() {
        layers.insert(row.package.clone(), row.layer.clone());
        directories.insert(row.package.clone(), row.path.clone());
    }
    Ok((layers, directories))
}

/// `text` as lowercase identifier segments.
///
/// A byte that cannot sit inside an identifier ends the current segment, and an
/// uppercase letter starts a new one when it follows a lowercase letter or digit
/// or precedes a lowercase letter, which splits `WGSLModule` into `wgsl` and
/// `module` rather than one run nothing matches.
#[must_use]
pub fn segments_of(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut segments = Vec::new();
    let mut current = String::new();
    for (index, letter) in text.char_indices() {
        if !letter.is_ascii_alphanumeric() {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            continue;
        }
        let previous = index.checked_sub(1).map(|at| bytes[at]);
        let next = bytes.get(index + 1).copied();
        let starts_segment = letter.is_ascii_uppercase()
            && (previous.is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
                || next.is_some_and(|byte| byte.is_ascii_lowercase()));
        if starts_segment && !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
        current.push(letter.to_ascii_lowercase());
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Whether the tree reaches `file` only as test support.
///
/// A `tests` directory or a `tests.rs` module is test code whatever declared it,
/// and the `#[cfg(test)]` attribute that gates it sits in the parent file rather
/// than in the file being read, so a reader of that file cannot see it.
#[must_use]
pub fn is_test_source(file: &str) -> bool {
    file.split('/')
        .any(|part| part == "tests" || part == "tests.rs")
}

/// Reject a data row that no longer describes the tree it judges.
///
/// The wordlist is the rule, so a row that covers nothing is the rule quietly
/// shrinking. Five ways that happens, all reported: a member declares a layer no
/// row decides, a row decides a layer no member declares, an exemption names a
/// layer that is not neutral, a term is owned by a layer the rows call neutral,
/// and an interface allowance names something that appears nowhere under its own
/// directory.
#[must_use]
pub fn contract_failures(root: &Path, neutrality: &Neutrality) -> Vec<String> {
    let mut failures = Vec::new();
    let contract = &neutrality.contract;
    let decided: BTreeSet<&str> = contract
        .layers
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    let declared: BTreeSet<&str> = neutrality.layers.values().map(String::as_str).collect();

    for layer in &declared {
        if !decided.contains(layer) {
            failures.push(format!(
                "{OWNERSHIP_REGISTRY} declares the layer `{layer}` and {DATA_FILE} decides nothing for it; record whether the layer is substrate-neutral, because a layer with no decision is skipped"
            ));
        }
    }
    for layer in &decided {
        if !declared.contains(layer) {
            failures.push(format!(
                "{DATA_FILE} decides the layer `{layer}` and no member declares it; delete the row, because a decision for a layer nobody declares records a rule that stopped covering anything"
            ));
        }
    }
    for exempt in &contract.exempt_layers {
        if neutrality.layer_is_neutral(&exempt.layer) != Some(true) {
            failures.push(format!(
                "{DATA_FILE} excuses the layer `{}` from the vocabulary rule and that layer is not substrate-neutral; delete the row, because the rule only reaches neutral layers and an exemption outside them excuses nothing",
                exempt.layer
            ));
        }
    }
    let mut seen = BTreeSet::new();
    for term in &contract.terms {
        if !seen.insert(term.word.as_str()) {
            failures.push(format!(
                "{DATA_FILE} bans `{}` twice; one row per word, so a reader sees one replacement",
                term.word
            ));
        }
        if segments_of(&term.word).is_empty() {
            failures.push(format!(
                "{DATA_FILE} bans `{}`, which holds no identifier segment and would match every line",
                term.word
            ));
        }
        if !contract.neutral_terms.contains(&term.neutral) {
            failures.push(format!(
                "{DATA_FILE} replaces `{}` with `{}`, which is not one of the neutral terms the contract names; a ban without a replacement tells an author to delete the sentence",
                term.word, term.neutral
            ));
        }
        match neutrality.layer_is_neutral(&term.owner_layer) {
            None => failures.push(format!(
                "{DATA_FILE} gives `{}` to the layer `{}`, which no row decides; name the layer that owns the concrete detail",
                term.word, term.owner_layer
            )),
            Some(true) => failures.push(format!(
                "{DATA_FILE} gives `{}` to the substrate-neutral layer `{}`, so the word is banned from its own owner; name the layer that owns the concrete detail",
                term.word, term.owner_layer
            )),
            Some(false) => {}
        }
    }
    for interface in &contract.interfaces {
        if !root.join(&interface.prefix).is_dir() {
            failures.push(format!(
                "{DATA_FILE} allows `{}` under `{}`, which is not a directory in this checkout; point the row at the directory that reads the interface, or delete it",
                interface.name, interface.prefix
            ));
            continue;
        }
        if !interface_name_is_used(root, interface) {
            failures.push(format!(
                "{DATA_FILE} allows `{}` under `{}` and no production source there names it; delete the row, because an allowance nothing uses excuses vocabulary somebody may add later without review",
                interface.name, interface.prefix
            ));
        }
    }
    failures
}

/// Whether any production source under an allowance's directory names it.
fn interface_name_is_used(root: &Path, interface: &Interface) -> bool {
    production_lines(root, &interface.prefix).any(|(_, _, line)| line.contains(&interface.name))
}

/// Reject a neutral crate that names a concrete backend in production source.
///
/// The lines come from the caller so the rule is judged the same way whatever
/// produced them: [`production_lines`] over the live tree for the gate, and a
/// synthesised line for the tests that prove each crate in the roster is reached.
#[must_use]
pub fn vocabulary_failures<I>(neutrality: &Neutrality, lines: I) -> Vec<String>
where
    I: IntoIterator<Item = (String, u32, String)>,
{
    let mut failures = Vec::new();
    for (file, number, text) in lines {
        let masked = neutrality.mask_interface_names(&text, &file);
        let found = neutrality.words_in(&masked);
        if found.is_empty() {
            continue;
        }
        let named = found
            .iter()
            .map(|term| format!("`{}`", term.word))
            .collect::<Vec<String>>()
            .join(", ");
        let replacements = found
            .iter()
            .map(|term| format!("`{}` -> {}", term.word, term.neutral))
            .collect::<Vec<String>>()
            .join(", ");
        failures.push(format!(
            "{file}:{number} is production source of a substrate-neutral crate and names {named}; state the neutral concept ({replacements}), or move the code into the crate that owns that backend when the concrete detail is load-bearing"
        ));
    }
    failures
}

/// Collect every neutral-vocabulary and contract-data violation under `root`.
///
/// Streams rather than reading the roster's production text into a model first:
/// the neutral crates hold most of the workspace's source, and the rule keeps
/// only the lines that matched.
#[must_use]
pub fn neutral_vocabulary_failures(root: &Path) -> Vec<String> {
    let neutrality = match Neutrality::read(root) {
        Ok(neutrality) => neutrality,
        Err(error) => {
            return vec![format!(
                "the neutral-vocabulary contract cannot be read: {error}; repair it, because the rule covers nothing while it is unreadable"
            )]
        }
    };
    let mut failures = contract_failures(root, &neutrality);
    for (_, directory) in neutrality.roster() {
        failures.extend(vocabulary_failures(
            &neutrality,
            production_lines(root, &format!("{directory}/src/")),
        ));
    }
    failures
}

/// Every production source line under one checkout-relative directory prefix.
///
/// Test code is excluded twice over: a file the tree reaches only as test support
/// is skipped by path, and a line inside a `#[cfg(test)]` item is skipped by the
/// span reader this crate already owns. A backend name in a test is the test
/// naming the backend it drives, which is what a backend test is for.
fn production_lines<'a>(
    root: &'a Path,
    prefix: &str,
) -> impl Iterator<Item = (String, u32, String)> + 'a {
    let directory = root.join(prefix.trim_end_matches('/'));
    source_tree_files(&directory)
        .into_iter()
        .filter_map(move |path| {
            let file = relative(root, &path);
            if is_test_source(&file) {
                return None;
            }
            let text = read_source_bounded(&path).ok()?;
            Some((file, text))
        })
        .flat_map(|(file, text)| {
            let test_only = cfg_test_line_mask(&text);
            text.lines()
                .enumerate()
                .filter(|(index, _)| !test_only.get(*index).copied().unwrap_or(false))
                .map(|(index, line)| {
                    (
                        file.clone(),
                        u32::try_from(index + 1).unwrap_or(u32::MAX),
                        line.to_string(),
                    )
                })
                .collect::<Vec<_>>()
        })
}

/// Reject a glob re-export of another workspace crate's items.
///
/// `pub use other_crate::module::*;` publishes every item that module holds at a
/// second path, so one constant answers to two names and neither is the owner.
/// That is what a compatibility spelling is: nobody wrote the list, so nobody can
/// say what it publishes, and the count changes when the owner grows an item.
/// Importing the names with `use` and re-exporting a written list are both fine;
/// the glob is the shape that keeps an old path alive wholesale.
///
/// What this does not catch: an explicit `pub use other_crate::Item`, which is a
/// reviewed surface decision, and a glob over the crate's own modules, which is
/// how a module assembles the surface it owns.
#[must_use]
pub fn foreign_glob_reexport_failures(reexports: &[(String, String, u32, String)]) -> Vec<String> {
    reexports
        .iter()
        .map(|(file, owner, number, path)| {
            format!(
                "{file}:{number} re-exports `{path}` with a glob, so every item `{path}` holds answers to a second path that `{owner}` does not own; import the names this file needs and let callers name the owner"
            )
        })
        .collect()
}

/// Every glob re-export of another workspace crate, read from member sources.
///
/// Comments and literals are masked first through the reader this crate owns, so
/// a `pub use` written inside a doc comment is prose rather than a re-export.
#[must_use]
pub fn scan_foreign_glob_reexports(
    root: &Path,
    crate_roots: &[crate::CrateRoot],
) -> Vec<(String, String, u32, String)> {
    let idents: BTreeSet<&str> = crate_roots.iter().map(|root| root.ident.as_str()).collect();
    let mut found = Vec::new();
    for crate_root in crate_roots {
        for path in source_tree_files(&root.join(&crate_root.directory).join("src")) {
            let file = relative(root, &path);
            let Ok(text) = read_source_bounded(&path) else {
                continue;
            };
            let masked = mask_comments_and_strings(&text);
            for (index, line) in masked.lines().enumerate() {
                let Some(path) = glob_reexport_path(line) else {
                    continue;
                };
                let Some(first) = path.split("::").next() else {
                    continue;
                };
                if first == crate_root.ident || !idents.contains(first) {
                    continue;
                }
                found.push((
                    file.clone(),
                    crate_root.ident.clone(),
                    u32::try_from(index + 1).unwrap_or(u32::MAX),
                    path.to_string(),
                ));
            }
        }
    }
    found
}

/// The path a `pub use <path>::*;` statement re-exports, without the glob.
fn glob_reexport_path(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("pub use ")?;
    let statement = rest.strip_suffix(';')?.trim_end();
    let path = statement.strip_suffix("::*")?;
    (!path.is_empty()).then_some(path)
}
