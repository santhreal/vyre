//! Whether a registered gate's rule can produce a finding at all.
//!
//! A gate reports through two channels. A note is information: how many files
//! were scanned, which baseline the trend was measured against. A finding is a
//! verdict: the runner counts findings, compares the count against the pinned
//! row, and fails. A rule that reports its violations as notes therefore cannot
//! fail, whatever its text says, and the sweep reports it as holding its
//! baseline forever. The cross-dialect reach-through rule sat in that state
//! while three dialects reached into a sibling's module tree.
//!
//! Registration, a baseline row and workflow wiring were already enforced. None
//! of the three can see this: an unfailable gate is registered, pinned at zero,
//! and named by a workflow.
//!
//! The check reads the source rather than running anything. For each gate it
//! finds the definition site that names it, then asks whether a `Finding`
//! construction is reachable from the body of that site through the functions
//! the body calls. Reachability stops at [`CALL_DEPTH`] levels, which is enough
//! for every shape in the tree: the rule inline in `run`, the rule in a helper
//! beside it, and the rule in a shared runner such as
//! [`crate::gates::scan::ratchet`].
//!
//! A definition site is either a `GateBehavior` implementation joined to its
//! descriptor name through a keyed registration tuple, or an invocation of a
//! macro that generates the behavior and carries a `name:` literal. Sixteen of
//! the registered gates use the macro shape, so a check that reads only direct
//! behavior implementations reports them as undefinable, and a check that fires
//! on a correct tree is worse than none. For a macro site the judged body is the
//! invocation plus the body of the macro it names, because the rule is split
//! across the two: the invocation supplies the inspection and the macro supplies
//! the call that settles it.
//!
//! A gate whose honest output is a note declares that in [`NOTE_ONLY_GATES`]
//! with the gate that carries the failing form, and a row there whose gate can
//! in fact fail is itself a failure: the reason has outlived the shape it
//! described.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::gate::GateError;
use crate::gates::scan::Tree;

/// How many call levels the reachability walk follows out of a `run` body.
///
/// Three covers the deepest real shape: `run` calls a check function, which
/// calls the helper that builds the finding.
const CALL_DEPTH: usize = 3;

/// How a finding reaches a report.
///
/// `Report::with_findings` and `Report::from_messages` are counted because a
/// gate may collect into a vector and hand the whole vector over, in which case
/// no `report.find` appears in its body.
const EMIT_MARKERS: &[&str] = &[
    "Finding::new(",
    "Finding::at(",
    "Finding::in_file(",
    "Report::with_findings(",
    "Report::from_messages(",
];

/// Gates whose only honest output is a note, and the gate that carries the
/// failing form of the same subject.
///
/// A row is a statement that the verdict belongs elsewhere, not that the
/// subject does not matter. Both halves are checked: the named gate must be
/// registered, and it must genuinely be unable to produce a finding, so a row
/// that outlives the shape it excused fails rather than lingering.
///
/// The table is empty. Every registered gate constructs a finding on some path,
/// and the one remaining note-only rule in the tree is a numbered check inside
/// a gate that fails on other checks, which is recorded where that check is
/// written rather than here: this table is keyed by gate name and a sub-check
/// has none.
const NOTE_ONLY_GATES: &[(&str, &str)] = &[];

/// Every registered gate whose rule cannot produce a finding, and every
/// note-only row that no longer describes one.
pub fn failures(root: &Path, gate_names: &[&str]) -> Result<Vec<String>, GateError> {
    let tree = Tree::open(root)?;
    let roots = gate_roots(&tree)?;
    let capability = capability_by_gate(&tree, &roots)?;
    Ok(verdict_failures(
        gate_names,
        &capability,
        &roots,
        NOTE_ONLY_GATES,
    ))
}

/// The disagreements between the registry and what the source can do.
///
/// Kept separate from reading the tree, and taking the note-only table as an
/// argument, so every branch is reachable from a test: three of the four
/// describe a state the current tree does not hold, and a branch only production
/// data can reach is a branch nothing proves.
fn verdict_failures(
    gate_names: &[&str],
    capability: &BTreeMap<String, bool>,
    roots: &[String],
    note_only: &[(&str, &str)],
) -> Vec<String> {
    let mut failures = Vec::new();
    for name in gate_names {
        let declared = note_only.iter().any(|(gate, _)| gate == name);
        match capability.get(*name) {
            None => failures.push(format!(
                "gate `{name}` is registered but no definition site under {} names it: no behavior definition is paired with its descriptor name, and no macro invocation carries `name: \"{name}\"`; the check cannot judge a gate it cannot find",
                roots.join(", ")
            )),
            Some(true) if declared => failures.push(format!(
                "gate `{name}` is listed as note-only but its rule does construct a finding; delete the row, because it excuses a gate that already fails"
            )),
            Some(false) if !declared => failures.push(format!(
                "gate `{name}` cannot produce a finding, so it reports and never fails; make the rule call report.find at the boundary it names, or record the gate in NOTE_ONLY_GATES with the gate that carries the failing form"
            )),
            _ => {}
        }
    }
    for (gate, _) in note_only {
        if !gate_names.contains(gate) {
            failures.push(format!(
                "NOTE_ONLY_GATES names `{gate}`, which is not a registered gate; delete the row"
            ));
        }
    }
    failures
}

/// The `src` directory of every workspace member that can declare a gate.
///
/// A gate implements `GateBehavior`, which this crate owns, so a member that can
/// declare one is this crate or a member that depends on it. The list is
/// read from the manifests rather than written here: a written list of two
/// roots reported the sixteen gates declared in the third as undefinable, and a
/// written list cannot notice a fourth crate.
fn gate_roots(tree: &Tree) -> Result<Vec<String>, GateError> {
    let owner = env!("CARGO_PKG_NAME");
    let workspace = tree.read_toml("Cargo.toml")?;
    let members = workspace
        .get("workspace")
        .and_then(|section| section.as_table())
        .and_then(|section| section.get("members"))
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            GateError::new(
                "the workspace manifest declares no [workspace] members array",
                "restore `members` in Cargo.toml, which is where the gate-declaring crates are derived from",
            )
        })?;
    let mut roots = BTreeSet::new();
    for member in members.iter().filter_map(|value| value.as_str()) {
        let manifest = tree.read_toml(format!("{member}/Cargo.toml"))?;
        let is_owner = manifest
            .get("package")
            .and_then(|section| section.as_table())
            .and_then(|section| section.get("name"))
            .and_then(|value| value.as_str())
            == Some(owner);
        let depends_on_owner = ["dependencies", "dev-dependencies", "build-dependencies"]
            .iter()
            .filter_map(|section| manifest.get(*section))
            .filter_map(|section| section.as_table())
            .any(|section| section.contains_key(owner));
        if is_owner || depends_on_owner {
            roots.insert(format!("{member}/src"));
        }
    }
    if roots.is_empty() {
        return Err(GateError::new(
            format!("no workspace member is `{owner}` or depends on it, so no source root declares a gate"),
            "run the gate from the workspace checkout that owns the gate trait",
        ));
    }
    Ok(roots.into_iter().collect())
}

/// One verdict per gate found in the source: whether a finding is reachable.
fn capability_by_gate(tree: &Tree, roots: &[String]) -> Result<BTreeMap<String, bool>, GateError> {
    let scope: Vec<&str> = roots.iter().map(String::as_str).collect();
    let mut sources = Vec::new();
    for path in tree.rust(&scope)? {
        let text = tree.read(&path)?;
        sources.push(without_test_modules(&without_comments(&text)));
    }
    let functions: Vec<(String, String)> =
        sources.iter().flat_map(|text| functions(text)).collect();
    let macros: Vec<(String, String)> =
        sources.iter().flat_map(|text| macro_bodies(text)).collect();
    let mut registrations = BTreeMap::new();
    for (type_name, gate_name) in sources.iter().flat_map(|text| behavior_registrations(text)) {
        if let Some(previous) = registrations.insert(type_name.clone(), gate_name.clone()) {
            if previous != gate_name {
                return Err(GateError::new(
                    format!(
                        "gate behavior `{type_name}` is registered as both `{previous}` and `{gate_name}`"
                    ),
                    "give every registered descriptor one distinct execution behavior",
                ));
            }
        }
    }
    let mut verdicts = BTreeMap::new();
    for text in &sources {
        for (gate, body) in gate_sites(text, &macros, &registrations) {
            let reachable = emits(&body, &functions, 0);
            verdicts.insert(gate, reachable);
        }
    }
    Ok(verdicts)
}

/// Whether a finding construction is reachable from `body`.
fn emits(body: &str, functions: &[(String, String)], depth: usize) -> bool {
    if EMIT_MARKERS.iter().any(|marker| body.contains(marker)) {
        return true;
    }
    if depth >= CALL_DEPTH {
        return false;
    }
    let called = called_names(body);
    functions
        .iter()
        .filter(|(name, _)| called.iter().any(|call| call == name))
        .any(|(_, callee)| emits(callee, functions, depth + 1))
}

/// Every function name called in `body`, method calls included.
///
/// A method call is kept rather than dropped: `scan::ratchet(..)` and
/// `self.check(..)` both name work whose findings belong to the gate, and a
/// name that matches no declared function costs one comparison.
///
/// Resolution is by name across every scanned crate, so a helper name two
/// crates both declare resolves to either. That errs toward reporting a gate as
/// able to fail, which is the direction that keeps the check off a correct tree.
fn called_names(body: &str) -> Vec<String> {
    let mut names = Vec::new();
    let bytes = body.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'(' {
            index += 1;
            continue;
        }
        let end = index;
        let mut start = end;
        while start > 0 && is_name_byte(bytes[start - 1]) {
            start -= 1;
        }
        index += 1;
        if start == end || bytes[start].is_ascii_digit() || declares_function(bytes, start) {
            continue;
        }
        names.push(body[start..end].to_string());
    }
    names
}

/// Whether the name ending just before `start` opens a `fn` declaration.
///
/// A macro body carries the `run` it generates, and a declaration read as a call
/// resolves to whatever else declares that name: every macro-declared gate then
/// reads as able to fail through an unrelated gate's `run`, which is sixteen
/// gates the check would stop judging.
fn declares_function(bytes: &[u8], start: usize) -> bool {
    let mut index = start;
    while index > 0 && bytes[index - 1].is_ascii_whitespace() {
        index -= 1;
    }
    index >= 2
        && &bytes[index - 2..index] == b"fn"
        && (index == 2 || !is_name_byte(bytes[index - 3]))
}

fn is_name_byte(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

/// Every `(gate name, judged body)` pair the text declares, either shape.
fn gate_sites(
    text: &str,
    macros: &[(String, String)],
    registrations: &BTreeMap<String, String>,
) -> Vec<(String, String)> {
    let mut sites = gate_run_bodies_registered(text, registrations);
    sites.extend(macro_gate_sites(text, macros));
    sites
}

/// Every `(gate name, run body)` pair an `impl GateBehavior for` block declares.
///
/// Production names come from the behavior registration table. The empty-map
/// fallback keeps parser unit fixtures independent while a real unregistered
/// behavior still leaves its authoritative descriptor without a matching site.
fn gate_run_bodies(text: &str) -> Vec<(String, String)> {
    gate_run_bodies_registered(text, &BTreeMap::new())
}

fn gate_run_bodies_registered(
    text: &str,
    registrations: &BTreeMap<String, String>,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for marker in [
        "impl crate::gate::GateBehavior for ",
        "impl xtask::gate::GateBehavior for ",
        "impl GateBehavior for ",
    ] {
        let mut from = 0;
        while let Some(at) = text[from..].find(marker) {
            let start = from + at;
            let type_start = start + marker.len();
            let Some(open) = text[start..].find('{').map(|offset| start + offset) else {
                break;
            };
            let type_name = text[type_start..open].trim();
            let block = balanced(text, open);
            from = open + block.len();
            if type_name.contains('$') {
                continue;
            }
            let name = registrations
                .get(type_name)
                .cloned()
                .unwrap_or_else(|| type_to_kebab(type_name));
            let Some(run) = block.find("fn run(") else {
                continue;
            };
            let Some(body_open) = block[run..].find('{').map(|offset| run + offset) else {
                continue;
            };
            out.push((name, balanced(block, body_open).to_string()));
        }
    }
    out
}

/// Map an implementation type to the descriptor name that registers it.
fn behavior_registrations(text: &str) -> Vec<(String, String)> {
    let mut registrations = Vec::new();
    let Some(gates_start) = text.find("GATES") else {
        return registrations;
    };
    let text = &text[gates_start..];
    let Some(slice_open) = text.find("&[") else {
        return registrations;
    };
    let Some(slice_close) = text[slice_open..].find("];") else {
        return registrations;
    };
    let text = &text[slice_open..slice_open + slice_close];
    let mut from = 0;
    while let Some(at) = text[from..].find('(') {
        let start = from + at + 1;
        let rest = &text[start..];
        let trimmed = rest.trim_start();
        if let Some(after_quote) = trimmed.strip_prefix('"') {
            if let Some(quote_end) = after_quote.find('"') {
                let name = &after_quote[..quote_end];
                let after_name = &after_quote[quote_end + 1..];
                if let Some(comma) = after_name.find(',') {
                    let after_comma = &after_name[comma + 1..];
                    if let Some(close) = after_comma.find(')') {
                        let behavior = after_comma[..close].trim();
                        let Some(behavior) = behavior.strip_prefix('&') else {
                            from = start;
                            continue;
                        };
                        let behavior = behavior.trim().trim_end_matches(',').trim();
                        if behavior.is_empty()
                            || behavior.contains('(')
                            || behavior.contains(')')
                            || behavior.contains(' ')
                            || behavior.contains('"')
                            || !behavior
                                .chars()
                                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
                        {
                            from = start;
                            continue;
                        }
                        if let Some(type_name) = behavior.rsplit("::").next() {
                            let type_name = type_name.trim();
                            if !type_name.is_empty() && !name.is_empty() {
                                registrations.push((type_name.to_string(), name.to_string()));
                            }
                        }
                        from = start + (rest.len() - after_comma.len()) + close + 1;
                        continue;
                    }
                }
            }
        }
        from = start;
    }
    registrations
}

/// Every `(gate name, judged body)` pair a macro invocation declares.
///
/// An invocation is a definition site when it carries a `name:` string literal,
/// which is how a macro that generates a `GateBehavior` implementation receives
/// the registered name. The judged body is the invocation followed by the body
/// it names, so the walk sees both the caller's rule expression and the generated
/// `run`.
fn macro_gate_sites(text: &str, macros: &[(String, String)]) -> Vec<(String, String)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'!' {
            index += 1;
            continue;
        }
        let bang = index;
        let mut start = bang;
        while start > 0 && is_name_byte(bytes[start - 1]) {
            start -= 1;
        }
        let mut open = bang + 1;
        while open < bytes.len() && bytes[open].is_ascii_whitespace() {
            open += 1;
        }
        if start == bang || !matches!(bytes.get(open), Some(b'{') | Some(b'(') | Some(b'[')) {
            index = bang + 1;
            continue;
        }
        let body = balanced(text, open);
        index = open + body.len();
        let Some(name) = field_literal(body, "name") else {
            continue;
        };
        let invoked = &text[start..bang];
        let expansion = macros
            .iter()
            .find(|(declared, _)| declared == invoked)
            .map_or("", |(_, generated)| generated.as_str());
        let mut judged = String::with_capacity(body.len() + expansion.len());
        judged.push_str(body);
        judged.push_str(expansion);
        out.push((name, judged));
    }
    out
}

/// Every `(macro name, macro body)` pair the text declares.
fn macro_bodies(text: &str) -> Vec<(String, String)> {
    const DECLARATION: &str = "macro_rules!";
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find(DECLARATION) {
        let after = from + at + DECLARATION.len();
        let rest = text[after..].trim_start();
        let name_start = text.len() - rest.len();
        let name_end = rest
            .find(|character: char| !character.is_alphanumeric() && character != '_')
            .unwrap_or(rest.len());
        let Some(open) = rest[name_end..]
            .find('{')
            .map(|offset| name_start + name_end + offset)
        else {
            break;
        };
        let body = balanced(text, open);
        from = open + body.len();
        if name_end > 0 {
            out.push((rest[..name_end].to_string(), body.to_string()));
        }
    }
    out
}

/// The string literal a `field:` entry carries, ignoring one inside a string.
fn field_literal(body: &str, field: &str) -> Option<String> {
    let bytes = body.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'"' {
            index = skip_string(bytes, index) + 1;
            continue;
        }
        let named = bytes[index..].starts_with(field.as_bytes())
            && bytes.get(index + field.len()) == Some(&b':')
            && (index == 0 || !is_name_byte(bytes[index - 1]));
        if named {
            let mut at = index + field.len() + 1;
            while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                at += 1;
            }
            if bytes.get(at) == Some(&b'"') {
                let close = skip_string(bytes, at);
                return Some(body[at + 1..close].to_string());
            }
        }
        index += 1;
    }
    None
}

/// Convert a PascalCase struct name to kebab-case gate name.
fn type_to_kebab(type_name: &str) -> String {
    let raw = type_name.split("::").last().unwrap_or(type_name).trim();
    let mut kebab = String::new();
    for (i, ch) in raw.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !kebab.ends_with('-') {
                kebab.push('-');
            }
            kebab.push(ch.to_ascii_lowercase());
        } else {
            kebab.push(ch);
        }
    }
    kebab
}

/// Every `(function name, body)` pair the text declares.
fn functions(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (start, _) in text.match_indices("fn ") {
        if start > 0 && is_name_byte(text.as_bytes()[start - 1]) {
            continue;
        }
        let after = &text[start + 3..];
        let name_end = after
            .find(|character: char| !character.is_alphanumeric() && character != '_')
            .unwrap_or(after.len());
        if name_end == 0 {
            continue;
        }
        let name = &after[..name_end];
        let Some(open) = after[name_end..]
            .find('{')
            .map(|offset| start + 3 + name_end + offset)
        else {
            continue;
        };
        // A `fn` used as a type, such as `&dyn Fn(&Path) -> bool`, has no body
        // before the next statement ends.
        if text[start..open].contains(';') {
            continue;
        }
        out.push((name.to_string(), balanced(text, open).to_string()));
    }
    out
}

/// The delimiter-balanced slice starting at the bracket at `open`.
///
/// A bracket inside a string or a comment would unbalance the walk, so both are
/// stepped over rather than counted. A macro invocation may be delimited by any
/// of the three bracket pairs, so the pair is read from the opening byte.
fn balanced(text: &str, open: usize) -> &str {
    let bytes = text.as_bytes();
    let (opener, closer) = match bytes.get(open) {
        Some(b'(') => (b'(', b')'),
        Some(b'[') => (b'[', b']'),
        _ => (b'{', b'}'),
    };
    let mut depth = 0usize;
    let mut index = open;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == opener {
            depth += 1;
        } else if byte == closer && depth > 0 {
            depth -= 1;
            if depth == 0 {
                return &text[open..=index];
            }
        } else if byte == b'"' {
            index = skip_string(bytes, index);
        } else if byte == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line(bytes, index);
            continue;
        }
        index += 1;
    }
    &text[open..]
}

/// The index of the closing quote of the string opening at `open`.
fn skip_string(bytes: &[u8], open: usize) -> usize {
    let mut index = open + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => return index,
            _ => index += 1,
        }
    }
    bytes.len().saturating_sub(1)
}

/// The index of the newline ending the line comment opening at `open`.
fn skip_line(bytes: &[u8], open: usize) -> usize {
    let mut index = open;
    while index < bytes.len() && bytes[index] != b'\n' {
        index += 1;
    }
    index
}

/// The index just past the `*/` closing the block comment opening at `open`.
fn skip_block(bytes: &[u8], open: usize) -> usize {
    let mut index = open + 2;
    let mut depth = 1usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            depth += 1;
            index += 2;
        } else if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') {
            depth -= 1;
            index += 2;
            if depth == 0 {
                return index;
            }
        } else {
            index += 1;
        }
    }
    bytes.len()
}

/// The index of the last byte of the raw string literal opening at `at`.
fn raw_string_end(bytes: &[u8], at: usize) -> Option<usize> {
    if at > 0 && is_name_byte(bytes[at - 1]) {
        return None;
    }
    let mut index = at;
    if bytes.get(index) == Some(&b'b') {
        index += 1;
    }
    if bytes.get(index) != Some(&b'r') {
        return None;
    }
    index += 1;
    let hash_start = index;
    while bytes.get(index) == Some(&b'#') {
        index += 1;
    }
    let hashes = index - hash_start;
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    index += 1;
    while index < bytes.len() {
        if bytes[index] == b'"'
            && bytes[index + 1..]
                .iter()
                .take(hashes)
                .filter(|byte| **byte == b'#')
                .count()
                == hashes
        {
            return Some(index + hashes);
        }
        index += 1;
    }
    Some(bytes.len().saturating_sub(1))
}

/// The index of the closing quote of the character literal opening at `at`.
///
/// A lifetime opens with the same byte and closes with nothing, so a quote that
/// does not close two or three bytes later is not a literal.
fn char_literal_end(bytes: &[u8], at: usize) -> Option<usize> {
    if bytes.get(at) != Some(&b'\'') {
        return None;
    }
    if bytes.get(at + 1) == Some(&b'\\') {
        let mut index = at + 2;
        while index < bytes.len() && bytes[index] != b'\'' {
            index += 1;
        }
        return (index < bytes.len()).then_some(index);
    }
    (bytes.get(at + 2) == Some(&b'\'')).then_some(at + 2)
}

/// The text with every comment removed.
///
/// A doc comment is prose about code, and prose carries examples: the macro
/// that generates an artifact gate documents itself with a complete invocation,
/// name literal included, and a rule's doc comment quotes the finding it
/// builds. Reading either as code invents a gate nothing declares and reports
/// an unfailable gate as able to fail. String and character literals are copied
/// through, because a gate name is read out of one.
fn without_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line(bytes, index);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block(bytes, index);
            out.push(b' ');
            continue;
        }
        if let Some(close) = raw_string_end(bytes, index) {
            out.extend_from_slice(&bytes[index..=close]);
            index = close + 1;
            continue;
        }
        if bytes[index] == b'"' {
            let close = skip_string(bytes, index);
            out.extend_from_slice(&bytes[index..=close]);
            index = close + 1;
            continue;
        }
        if let Some(close) = char_literal_end(bytes, index) {
            out.extend_from_slice(&bytes[index..=close]);
            index = close + 1;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The text with every `#[cfg(test)] mod` block removed.
///
/// A test constructs findings to assert on them, so counting a test would make
/// every gate look able to fail.
fn without_test_modules(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("#[cfg(test)]") {
        let (before, tail) = rest.split_at(at);
        out.push_str(before);
        let semicolon = tail.find(';');
        let block_open = tail.find('{');
        if let Some(semicolon) = semicolon {
            if block_open.map_or(true, |block_open| semicolon < block_open) {
                rest = &tail[semicolon + 1..];
                continue;
            }
        }
        let Some(open) = block_open else {
            return out;
        };
        let block = balanced(tail, open);
        rest = &tail[open + block.len()..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the whole point is telling a rule that fails from one that reports.
    /// Both shapes are read from source text here rather than from a gate that
    /// might change, so the walk is judged on the two forms it must separate:
    /// a `run` that constructs a finding inline, and one whose only output is a
    /// note.
    ///
    /// What this does not catch: a rule that constructs a finding on a path no
    /// input can reach. Reachability of the construction is what is measured,
    /// not reachability of the branch.
    #[test]
    fn a_run_that_only_notes_is_separated_from_one_that_finds() {
        let source = r#"
impl crate::gate::GateBehavior for Reports {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        report.note("counted 3 files".to_string());
        Ok(report)
    }
}
impl crate::gate::GateBehavior for Judges {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        report.find(Finding::new("wrong", "fix it"));
        Ok(report)
    }
}
"#;
        let functions = functions(source);
        let verdicts: Vec<(String, bool)> = gate_run_bodies(source)
            .into_iter()
            .map(|(gate, run)| (gate, emits(&run, &functions, 0)))
            .collect();

        assert_eq!(
            verdicts,
            vec![("reports".to_string(), false), ("judges".to_string(), true)]
        );
    }

    /// WHY: behavior type names are implementation details. Descriptor names such as
    /// `hot-path-reserve` cannot be reconstructed from `ReserveArgument`, so the
    /// capability census must join through the live registration table.
    #[test]
    fn behavior_registration_name_overrides_type_spelling() {
        let source = r#"
pub static GATES: &[(&str, &dyn GateBehavior)] = &[
    ("hot-path-reserve", &hot_path::ReserveArgument),
];
impl crate::gate::GateBehavior for ReserveArgument {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(Report::with_findings(vec![Finding::new("wrong", "fix it")]))
    }
}
"#;
        let registrations: BTreeMap<String, String> =
            behavior_registrations(source).into_iter().collect();
        let sites = gate_run_bodies_registered(source, &registrations);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].0, "hot-path-reserve");
    }

    /// WHY: local imported, local qualified, and dependency-qualified trait names
    /// are the live implementation spellings. Dropping one makes its owner
    /// package disappear from the capability census.
    #[test]
    fn an_imported_behavior_trait_still_declares_a_gate_site() {
        let source = r#"
pub static GATES: &[(&str, &dyn GateBehavior)] = &[
    ("imported-name", &ImportedBehavior),
    ("dependency-name", &DependencyBehavior),
];
impl GateBehavior for ImportedBehavior {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(Report::with_findings(vec![Finding::new("wrong", "fix it")]))
    }
}
impl xtask::gate::GateBehavior for DependencyBehavior {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(Report::with_findings(vec![Finding::new("wrong", "fix it")]))
    }
}
"#;
        let registrations: BTreeMap<String, String> =
            behavior_registrations(source).into_iter().collect();
        let sites = gate_run_bodies_registered(source, &registrations);
        let names: BTreeSet<String> = sites.into_iter().map(|(name, _)| name).collect();
        assert_eq!(
            names,
            BTreeSet::from(["dependency-name".to_string(), "imported-name".to_string()])
        );
    }

    /// WHY: rustfmt formats long registration tuples across multiple lines with
    /// a trailing comma after the behavior path. If the parser does not strip the
    /// trailing comma, multi-line entries fail validation and are dropped.
    #[test]
    fn multiline_behavior_registration_with_trailing_comma_is_parsed() {
        let source = r#"
pub static GATES: &[(&'static str, &'static dyn GateBehavior)] = &[
    (
        "bench-crossback",
        &bench::bench_crossback::BenchCrossbackGate,
    ),
    (
        "operation-schema",
        &docs::operation_schema::OperationSchemaGate,
    ),
];
impl xtask::gate::GateBehavior for BenchCrossbackGate {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(Report::with_findings(vec![Finding::new("wrong", "fix it")]))
    }
}
impl xtask::gate::GateBehavior for OperationSchemaGate {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(Report::with_findings(vec![Finding::new("wrong", "fix it")]))
    }
}
"#;
        let registrations: BTreeMap<String, String> =
            behavior_registrations(source).into_iter().collect();
        let sites = gate_run_bodies_registered(source, &registrations);
        let names: BTreeSet<String> = sites.into_iter().map(|(name, _)| name).collect();
        assert_eq!(
            names,
            BTreeSet::from(["bench-crossback".to_string(), "operation-schema".to_string()])
        );
    }


    /// WHY: a rule one call away from its finding is the common shape, and the
    /// walk has to follow it or every ratchet gate reads as unfailable. A helper
    /// the body does not call must not count, or the check degrades to asking
    /// whether the file mentions a finding anywhere.
    #[test]
    fn reachability_follows_a_called_helper_and_stops_at_an_uncalled_one() {
        let source = r#"
impl crate::gate::GateBehavior for Delegates {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        judge(&tree)
    }
}
impl crate::gate::GateBehavior for Strands {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(Report::clean())
    }
}
fn judge(tree: &Tree) -> Result<Report, GateError> {
    let mut report = Report::clean();
    report.find(Finding::at("a.rs", 1, "wrong", "fix it"));
    Ok(report)
}
"#;
        let functions = functions(source);
        let verdicts: Vec<(String, bool)> = gate_run_bodies(source)
            .into_iter()
            .map(|(gate, run)| (gate, emits(&run, &functions, 0)))
            .collect();

        assert_eq!(
            verdicts,
            vec![
                ("delegates".to_string(), true),
                ("strands".to_string(), false)
            ]
        );
    }

    /// WHY: tests construct findings in order to assert on them, so counting a
    /// test module makes every gate in the file read as able to fail. That is
    /// the failure mode that would make this whole check vacuous.
    #[test]
    fn a_finding_built_only_inside_a_test_module_does_not_count() {
        let source = r#"
impl crate::gate::GateBehavior for Reports {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(Report::clean())
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn builds_one() {
        let finding = Finding::new("wrong", "fix it");
        assert_eq!(finding.message, "wrong");
    }
}
"#;
        let production = without_test_modules(source);
        assert!(
            !production.contains("Finding::new("),
            "the test module is removed before the walk reads the file"
        );

        let functions = functions(&production);
        let verdicts: Vec<(String, bool)> = gate_run_bodies(&production)
            .into_iter()
            .map(|(gate, run)| (gate, emits(&run, &functions, 0)))
            .collect();
        assert_eq!(verdicts, vec![("reports".to_string(), false)]);
    }

    /// WHY: an out-of-line test module ends at a semicolon. Treating the next
    /// production block as its body hides the gate that follows it.
    #[test]
    fn an_out_of_line_test_module_does_not_hide_following_production() {
        let source =
            "#[cfg(test)]\nmod tests;\nfn production() { Finding::new(\"wrong\", \"fix it\"); }\n";
        let production = without_test_modules(source);
        assert!(!production.contains("mod tests"));
        assert!(production.contains("fn production()"));
        assert!(production.contains("Finding::new("));
    }

    /// WHY: a brace inside a string literal or a comment is the standard way a
    /// hand-written brace walk loses its place, and losing it here silently
    /// truncates a gate body and reports the gate as unfailable.
    #[test]
    fn the_brace_walk_steps_over_strings_and_comments() {
        let source = "fn f() {\n    let brace = \"{\";\n    // }\n    done();\n}\nfn g() {}\n";
        let open = source.find('{').expect("opening brace");
        let body = balanced(source, open);

        assert!(body.ends_with("}"), "the walk closes the body it opened");
        assert!(
            body.contains("done();"),
            "the walk did not stop at the brace in the string or the comment"
        );
        assert!(
            !body.contains("fn g"),
            "the walk stopped at the end of the first function"
        );
    }

    /// WHY: sixteen of the registered gates are declared by a macro that
    /// generates the `GateBehavior` implementation, so a check that reads only
    /// direct behavior blocks reports every one of them as undefinable and fires
    /// on a correct tree. The verdict has to come from the same walk either way,
    /// which means the macro body counts as part of the site: the invocation
    /// holds the rule and the macro holds the call that reports it.
    ///
    /// The second gate is the guard that matters: a macro body carries the `run`
    /// it generates, and reading that declaration as a call resolves it to some
    /// other gate's `run` and reports every macro-declared gate as able to fail.
    ///
    /// What this does not catch: a macro whose definition is outside the scanned
    /// roots. Only the invocation is judged then, which can read as unfailable.
    #[test]
    fn a_gate_declared_by_a_macro_is_judged_through_the_macro_body() {
        let source = r#"
macro_rules! settles {
    ($gate:ident, name: $name:literal, inspect: |$ctx:ident| $rule:expr $(,)?) => {
        pub struct $gate;
        impl crate::gate::GateBehavior for $gate {
            fn run(&self, $ctx: &GateCtx) -> Result<Report, GateError> {
                Ok(settle_inspection($ctx, $name, $rule))
            }
        }
    };
}
macro_rules! reports {
    ($gate:ident, name: $name:literal) => {
        pub struct $gate;
        impl crate::gate::GateBehavior for $gate {
            fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
                let mut report = Report::clean();
                report.note(count(&ctx.root));
                Ok(report)
            }
        }
    };
}
settles! { SettledGate, name: "settled", inspect: |ctx| inspect(&ctx.root), }
reports! { CountedGate, name: "counted" }
fn settle_inspection(ctx: &GateCtx, gate: &str, inspection: Inspection) -> Report {
    let mut report = Report::clean();
    report.find(Finding::in_file(gate, "diverged", "rewrite it"));
    report
}
fn count(root: &Path) -> String {
    String::new()
}
fn inspect(root: &Path) -> Inspection {
    Inspection::default()
}
"#;
        let production = without_test_modules(&without_comments(source));
        let functions = functions(&production);
        let macros = macro_bodies(&production);
        let verdicts: BTreeMap<String, bool> = gate_sites(&production, &macros, &BTreeMap::new())
            .into_iter()
            .map(|(gate, body)| (gate, emits(&body, &functions, 0)))
            .collect();

        assert_eq!(
            verdicts,
            BTreeMap::from([
                ("settled".to_string(), true),
                ("counted".to_string(), false),
            ]),
            "a macro-declared gate is judged, and judged by what its macro does"
        );
    }

    /// WHY: the macro that generates an artifact gate documents itself with a
    /// complete invocation, name literal and all. Read as code it declares a
    /// gate that does not exist, and the phantom carries whatever the
    /// surrounding prose quotes, so a doc comment showing a finding would report
    /// an unfailable gate as able to fail. Comments are prose in both
    /// directions.
    #[test]
    fn a_definition_site_inside_a_comment_declares_no_gate() {
        let source = r#"
/// Declare a gate whose whole body is one inspection.
///
/// ```ignore
/// documented! {
///     ExampleGate,
///     name: "example",
/// }
/// ```
/*
 * impl crate::gate::GateBehavior for Commented {
 *     fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
 *         Ok(Report::with_findings(vec![Finding::new("wrong", "fix it")]))
 *     }
 * }
 */
impl crate::gate::GateBehavior for Real {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        // Finding::new("wrong", "fix it")
        let quoted = "name: \"quoted\"";
        Ok(Report::clean())
    }
}
"#;
        let production = without_test_modules(&without_comments(source));
        let macros = macro_bodies(&production);
        let names: Vec<String> = gate_sites(&production, &macros, &BTreeMap::new())
            .into_iter()
            .map(|(gate, _)| gate)
            .collect();

        assert_eq!(
            names,
            vec!["real".to_string()],
            "only the declared gate is a site: not the documented example, not the commented-out impl, not the name in a string"
        );
        let verdicts: Vec<bool> = gate_sites(&production, &macros, &BTreeMap::new())
            .into_iter()
            .map(|(_, body)| emits(&body, &functions(&production), 0))
            .collect();
        assert_eq!(
            verdicts,
            vec![false],
            "the finding quoted in a comment does not make the rule able to fail"
        );
    }

    /// WHY: the roots were written down as two directories while three crates
    /// declared gates, and the check reported the sixteen gates in the third as
    /// having no definition site. A written list cannot notice a crate being
    /// added either, so the set is derived from the manifests: this crate, plus
    /// every workspace member that depends on it. The negative half is the point,
    /// because a set that includes every member is not a derivation.
    #[test]
    fn the_scanned_roots_are_derived_from_the_members_that_depend_on_this_crate() {
        let owner = env!("CARGO_PKG_NAME");
        let workspace =
            format!("[workspace]\nmembers = [\"{owner}\", \"nested/consumer\", \"stranger\"]\n");
        let owner_manifest = format!("[package]\nname = \"{owner}\"\n");
        let consumer_manifest =
            format!("[package]\nname = \"consumer\"\n\n[dependencies]\n{owner} = {{ path = \"../../{owner}\" }}\n");
        let stranger_manifest = "[package]\nname = \"stranger\"\n\n[dependencies]\nserde = \"1\"\n";
        let (_temporary, root) = crate::gates::fixture_checkout::checkout(&[
            ("Cargo.toml", workspace.as_str()),
            (&format!("{owner}/Cargo.toml"), owner_manifest.as_str()),
            (&format!("{owner}/src/lib.rs"), "fn owner() {}\n"),
            ("nested/consumer/Cargo.toml", consumer_manifest.as_str()),
            ("nested/consumer/src/lib.rs", "fn consumer() {}\n"),
            ("stranger/Cargo.toml", stranger_manifest),
            ("stranger/src/lib.rs", "fn stranger() {}\n"),
        ]);
        let tree = Tree::open(&root).expect("the fixture checkout opens");

        let roots = gate_roots(&tree).expect("the members are readable");

        assert_eq!(
            roots,
            vec!["nested/consumer/src".to_string(), format!("{owner}/src")],
            "the crate that owns the trait and the member that depends on it, and nothing else"
        );
    }

    /// WHY: three of the four verdicts describe a state the current tree does
    /// not hold, so nothing but a test reaches them, and an unreachable branch
    /// is an unproven one. The escape hatch is the reason: a row that excuses a
    /// gate which has since learned to fail, or names a gate nobody registers,
    /// is the same rot the check exists to find.
    #[test]
    fn every_disagreement_between_the_registry_and_the_source_is_reported() {
        let roots = vec!["xtask/src".to_string()];
        let capability =
            BTreeMap::from([("finds".to_string(), true), ("notes".to_string(), false)]);
        let excuses_finds: &[(&str, &str)] = &[("finds", "another gate")];
        let excuses_notes: &[(&str, &str)] = &[("notes", "another gate")];
        let excuses_nobody: &[(&str, &str)] = &[("retired", "another gate")];

        let clean = verdict_failures(&["finds"], &capability, &roots, &[]);
        assert!(
            clean.is_empty(),
            "a gate that can fail is not reported: {clean:?}"
        );

        let excused = verdict_failures(&["notes"], &capability, &roots, excuses_notes);
        assert!(
            excused.is_empty(),
            "a declared note-only gate that cannot fail is the state the row describes: {excused:?}"
        );

        let missing = verdict_failures(&["absent"], &capability, &roots, &[]);
        assert_eq!(missing.len(), 1);
        assert!(
            missing[0].contains("no definition site under xtask/src names it"),
            "a registered gate with no site names the roots that were read: {}",
            missing[0]
        );

        let unfailable = verdict_failures(&["notes"], &capability, &roots, &[]);
        assert_eq!(unfailable.len(), 1);
        assert!(
            unfailable[0].contains("cannot produce a finding"),
            "an undeclared note-only gate is reported: {}",
            unfailable[0]
        );

        let stale = verdict_failures(&["finds"], &capability, &roots, excuses_finds);
        assert_eq!(stale.len(), 1);
        assert!(
            stale[0].contains("does construct a finding"),
            "a row excusing a gate that can fail is reported: {}",
            stale[0]
        );

        let orphan = verdict_failures(&["finds"], &capability, &roots, excuses_nobody);
        assert_eq!(orphan.len(), 1);
        assert!(
            orphan[0].contains("is not a registered gate"),
            "a row naming an unregistered gate is reported: {}",
            orphan[0]
        );
    }
}
