//! The architecture document agrees with the live authorities it describes.
//!
//! The architecture document is prose about the workspace manifests, the release
//! train, the generated operation schema, the backend probe evidence, the crate
//! ownership registry and the docs manifest. Prose drifts from all six silently,
//! and this rule caught that drift from a Python script that one workflow
//! invoked and the gate sweep never saw. A rule outside the registry has no
//! baseline, no place in the sweep and no countable report.
//!
//! The optimization lane registry is judged here too, because a lane is an
//! assignment: a write glob that matches nothing and a `-p` package no manifest
//! declares both hand an owner a scope they cannot enter, and neither fails
//! until someone tries. `vyre-scan` survived three lanes here after the crate
//! stopped existing.

use std::collections::BTreeSet;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{glob_match, Tree};

/// The document this gate holds to the authorities.
const ARCHITECTURE: &str = "docs/ARCHITECTURE.md";

/// The docs manifest that has to classify it as current.
const MANIFEST: &str = "docs/DOCS.toml";

/// The optimization lane registry.
const LANES: &str = "docs/optimization/OWNERSHIP.toml";

/// Wire version of `docs/generated/OP_SCHEMA.json`.
///
/// `xtask-registry/src/docs/operation_schema.rs` owns the generator and pins the
/// same number. Reading the artifact's own field as the expected value would
/// make the check vacuous, so the expectation is written here and the
/// generator's own test compares the two constants directly. That drift shipped
/// once, with the generator on 3 and the checker still demanding 2, back when
/// the checker was a Python script whose number could only be compared as text.
pub const OPERATION_SCHEMA_VERSION: i64 = 4;

/// Tokens the document must still contain, each naming a live boundary.
const REQUIRED_TOKENS: &[&str] = &[
    "generated/OP_SCHEMA.json",
    "vyre-foundation::operation::OperationRegistry",
    "do not own shadow operation identities",
    "Cross-program composition",
    "vyre-megakernel",
    "Artifact",
    "vyre-runtime",
    "bytecode interpreter",
];

/// Case-insensitive literals the document must not contain.
const FORBIDDEN_LITERALS: &[&str] = &["## Four CI laws"];

/// Case-insensitive literals that describe a state the tree left behind.
const STALE_LITERALS: &[&str] = &[
    "planned compiler crate",
    "not a current workspace",
    "not present in the current",
    "until that crate exists",
    "until that boundary ships",
    "declared target rather than a shipped package",
];

/// What a disagreement with an authority costs, and how to close it.
const FIX: &str = "correct the document against the authority it describes, or correct the authority; the document is verified prose, not a second source of truth";

/// What a lane naming absent scope costs, and how to close it.
const LANE_FIX: &str = "delete the lane entry, or restore what it names; a write glob matching nothing and a -p package no manifest declares are both a scope nobody can enter";

/// The architecture document and the optimization lanes agree with the tree.
pub struct ArchitectureContract;

impl crate::gate::GateBehavior for ArchitectureContract {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut findings = Vec::new();
        let members = tree.member_manifests()?;
        judge_members(&tree, &members, &mut findings)?;
        let version = release_version(&tree, &mut findings)?;
        judge_operation_schema(&tree, &mut findings)?;
        judge_backend_evidence(&tree, &mut findings)?;
        judge_ownership(&tree, &mut findings)?;
        judge_document(&tree, version.as_deref(), &mut findings)?;
        judge_lanes(&tree, &members, &mut findings)?;
        let mut report = Report::with_findings(findings);
        report.cover_complete("workspace members", members.len());
        report.note(format!(
            "{} workspace member(s) read as the authority for the architecture prose",
            members.len()
        ));
        Ok(report)
    }
}

/// The workspace roster the document describes.
fn judge_members(
    tree: &Tree,
    members: &[crate::gates::scan::Member],
    findings: &mut Vec<Finding>,
) -> Result<(), GateError> {
    let manifest = tree.read_toml("Cargo.toml")?;
    let declared = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array);
    let Some(declared) = declared else {
        findings.push(Finding::in_file(
            "Cargo.toml",
            "workspace.members is missing or is not an array",
            FIX,
        ));
        return Ok(());
    };
    if declared.is_empty() {
        findings.push(Finding::in_file(
            "Cargo.toml",
            "workspace.members is empty",
            FIX,
        ));
    }
    for entry in declared {
        match entry.as_str() {
            Some(text) if text.contains('*') => findings.push(Finding::in_file(
                "Cargo.toml",
                format!("workspace member `{text}` is a glob rather than an explicit path"),
                FIX,
            )),
            Some(_) => {}
            None => findings.push(Finding::in_file(
                "Cargo.toml",
                "workspace.members holds a non-string entry",
                FIX,
            )),
        }
    }
    if !members
        .iter()
        .any(|member| member.path == "vyre-megakernel")
    {
        findings.push(Finding::in_file(
            "Cargo.toml",
            "workspace.members does not include vyre-megakernel, which the architecture names as the compiler crate",
            FIX,
        ));
    }
    Ok(())
}

/// The released version the document has to be verified against.
fn release_version(tree: &Tree, findings: &mut Vec<Finding>) -> Result<Option<String>, GateError> {
    let train = tree.read_toml("release/release-train.toml")?;
    let version = train
        .get("versions")
        .and_then(|versions| versions.get("vyre"))
        .and_then(toml::Value::as_str);
    match version {
        Some(value) => Ok(Some(value.to_string())),
        None => {
            findings.push(Finding::in_file(
                "release/release-train.toml",
                "versions.vyre is missing, so no document can be verified against a release",
                FIX,
            ));
            Ok(None)
        }
    }
}

/// The generated operation schema is internally coherent at the pinned version.
fn judge_operation_schema(tree: &Tree, findings: &mut Vec<Finding>) -> Result<(), GateError> {
    let path = "docs/generated/OP_SCHEMA.json";
    let value: serde_json::Value = serde_json::from_str(&tree.read(path)?).map_err(|error| {
        GateError::new(
            format!("cannot parse JSON `{path}`: {error}"),
            "regenerate the schema with `xtask operation-schema --write`",
        )
    })?;
    let operations = value
        .get("operations")
        .and_then(serde_json::Value::as_array);
    let tier_counts = value
        .get("tier_counts")
        .and_then(serde_json::Value::as_object);
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_i64);
    if schema_version != Some(OPERATION_SCHEMA_VERSION) {
        findings.push(Finding::in_file(
            path,
            format!(
                "schema_version is {schema_version:?}, and the architecture contract pins {OPERATION_SCHEMA_VERSION}"
            ),
            FIX,
        ));
    }
    let (Some(operations), Some(tier_counts)) = (operations, tier_counts) else {
        findings.push(Finding::in_file(
            path,
            "operations must be an array and tier_counts an object",
            FIX,
        ));
        return Ok(());
    };
    let declared_count = value
        .get("operation_count")
        .and_then(serde_json::Value::as_u64);
    let actual = u64::try_from(operations.len()).unwrap_or(u64::MAX);
    if declared_count != Some(actual) {
        findings.push(Finding::in_file(
            path,
            format!("operation_count is {declared_count:?} against {actual} operation entries"),
            FIX,
        ));
    }
    let tiered: u64 = tier_counts
        .values()
        .filter_map(serde_json::Value::as_u64)
        .sum();
    if tiered != actual {
        findings.push(Finding::in_file(
            path,
            format!("tier_counts sum to {tiered} against {actual} operation entries"),
            FIX,
        ));
    }
    Ok(())
}

/// The backend probe evidence has no blockers and a preferred backend that ran.
fn judge_backend_evidence(tree: &Tree, findings: &mut Vec<Finding>) -> Result<(), GateError> {
    let path = "release/evidence/backends/backend-matrix.json";
    let value: serde_json::Value = serde_json::from_str(&tree.read(path)?).map_err(|error| {
        GateError::new(
            format!("cannot parse JSON `{path}`: {error}"),
            "regenerate the backend evidence with `xtask backend-matrix --write`",
        )
    })?;
    match value.get("blockers").and_then(serde_json::Value::as_array) {
        Some(blockers) if blockers.is_empty() => {}
        Some(blockers) => findings.push(Finding::in_file(
            path,
            format!("backend evidence carries {} blocker(s)", blockers.len()),
            FIX,
        )),
        None => findings.push(Finding::in_file(
            path,
            "backend evidence has no blockers array",
            FIX,
        )),
    }
    let preferred = value
        .get("preferred_backend_id")
        .and_then(serde_json::Value::as_str);
    let rows = value.get("backends").and_then(serde_json::Value::as_array);
    let (Some(preferred), Some(rows)) = (preferred, rows) else {
        findings.push(Finding::in_file(
            path,
            "backend evidence has no preferred_backend_id or no probe rows",
            FIX,
        ));
        return Ok(());
    };
    let probed: BTreeSet<&str> = rows
        .iter()
        .filter_map(|row| row.get("id").and_then(serde_json::Value::as_str))
        .collect();
    if !probed.contains(preferred) {
        findings.push(Finding::in_file(
            path,
            format!("preferred backend `{preferred}` has no executable probe row"),
            FIX,
        ));
    }
    Ok(())
}

/// The ownership registry still carries the compiler crate as a shipped one.
fn judge_ownership(tree: &Tree, findings: &mut Vec<Finding>) -> Result<(), GateError> {
    let path = "docs/CRATE_OWNERSHIP.toml";
    let table = tree.read_toml(path)?;
    let Some(rows) = table.get("crate").and_then(toml::Value::as_array) else {
        findings.push(Finding::in_file(
            path,
            "the ownership registry has no [[crate]] rows",
            FIX,
        ));
        return Ok(());
    };
    let megakernel = rows.iter().find(|row| {
        row.get("package").and_then(toml::Value::as_str) == Some("vyre-megakernel")
            || row.get("path").and_then(toml::Value::as_str) == Some("vyre-megakernel")
    });
    match megakernel {
        None => findings.push(Finding::in_file(
            path,
            "the ownership registry has no vyre-megakernel crate row",
            FIX,
        )),
        Some(row) => {
            let responsibility = row
                .get("responsibility")
                .and_then(toml::Value::as_str)
                .unwrap_or_default();
            if !responsibility.contains("ProgramGraph") {
                findings.push(Finding::in_file(
                    path,
                    "the vyre-megakernel responsibility does not name ProgramGraph, which is what the crate compiles",
                    FIX,
                ));
            }
        }
    }
    let planned = table
        .get("planned")
        .and_then(toml::Value::as_table)
        .is_some_and(|planned| planned.contains_key("vyre-megakernel"));
    if planned {
        findings.push(Finding::in_file(
            path,
            "the ownership registry still carries planned.vyre-megakernel after the crate shipped",
            FIX,
        ));
    }
    Ok(())
}

/// The document itself: verified, current, and free of the state it left behind.
fn judge_document(
    tree: &Tree,
    version: Option<&str>,
    findings: &mut Vec<Finding>,
) -> Result<(), GateError> {
    let text = tree.read(ARCHITECTURE)?;
    if verification_date(&text).is_none() {
        findings.push(Finding::in_file(
            ARCHITECTURE,
            "no `Last verified: YYYY-MM-DD` line, so the prose claims no date",
            FIX,
        ));
    }
    if let Some(version) = version {
        if !text.contains(version) {
            findings.push(Finding::in_file(
                ARCHITECTURE,
                format!("the document is not verified against Vyre {version}"),
                FIX,
            ));
        }
    }
    let manifest = tree.read_toml(MANIFEST)?;
    let pages = manifest.get("page").and_then(toml::Value::as_array);
    let Some(pages) = pages else {
        findings.push(Finding::in_file(
            MANIFEST,
            "the docs manifest has no [[page]] rows",
            FIX,
        ));
        return Ok(());
    };
    let status = pages
        .iter()
        .find(|page| {
            page.get("path").and_then(toml::Value::as_str)
                == Some(ARCHITECTURE.trim_start_matches("docs/"))
        })
        .and_then(|page| page.get("status"))
        .and_then(toml::Value::as_str);
    if status != Some("current") {
        findings.push(Finding::in_file(
            MANIFEST,
            format!("`{ARCHITECTURE}` is classified {status:?} rather than current"),
            FIX,
        ));
    }
    for stale in stale_phrases(&text) {
        findings.push(Finding::in_file(
            ARCHITECTURE,
            format!("retains the stale architecture phrase `{stale}`"),
            FIX,
        ));
    }
    let normalized = normalize_whitespace(&text);
    for token in REQUIRED_TOKENS {
        if !normalized.contains(&normalize_whitespace(token)) {
            findings.push(Finding::in_file(
                ARCHITECTURE,
                format!("does not name the live boundary `{token}`"),
                FIX,
            ));
        }
    }
    Ok(())
}

/// Every stale phrase the document still carries.
///
/// Each of these described the workspace before the compiler crate shipped, the
/// operation set grew past nine, or the CI law list was retired. A pattern here
/// is a hand-written predicate rather than a regular expression because this
/// crate links no regex engine, and a hand-written one is what the reader of a
/// finding has to be able to check.
fn stale_phrases(text: &str) -> Vec<String> {
    let lowered = text.to_ascii_lowercase();
    let mut found = Vec::new();
    for literal in FORBIDDEN_LITERALS.iter().chain(STALE_LITERALS.iter()) {
        if lowered.contains(&literal.to_ascii_lowercase()) {
            found.push((*literal).to_string());
        }
    }
    if word_bounded(&lowered, "0.6") {
        found.push("version 0.6".to_string());
    }
    for spelling in ["9-op", "9 op", "nine-op", "nine op"] {
        if word_bounded(&lowered, spelling) {
            found.push(spelling.to_string());
        }
    }
    if let Some(matched) = wgpu_primary_path(&lowered) {
        found.push(matched);
    }
    if let Some(matched) = agent_identifier(&lowered) {
        found.push(matched);
    }
    found
}

/// Whether `needle` appears in `text` with no word character on either side.
fn word_bounded(text: &str, needle: &str) -> bool {
    occurrences(text, needle)
        .any(|at| !preceded_by_word(text, at) && !followed_by_word(text, at + needle.len()))
}

/// A claim that one backend is the primary production path.
fn wgpu_primary_path(text: &str) -> Option<String> {
    const CLAIM: &str = "primary production path";
    for at in occurrences(text, "wgpu") {
        let after = at + "wgpu".len();
        let line_end = text[after..]
            .find('\n')
            .map_or(text.len(), |offset| after + offset);
        let window_end = (after + 40 + CLAIM.len()).min(line_end);
        if text[after..window_end].contains(CLAIM) {
            return Some("WGPU as the primary production path".to_string());
        }
    }
    None
}

/// An agent session identifier, which names a workflow rather than the product.
fn agent_identifier(text: &str) -> Option<String> {
    const PREFIX: &str = "codex-";
    occurrences(text, PREFIX)
        .find(|at| {
            text[at + PREFIX.len()..]
                .chars()
                .next()
                .is_some_and(|glyph| glyph.is_ascii_alphanumeric())
        })
        .map(|_| "an agent session identifier".to_string())
}

/// Byte offsets of every occurrence of `needle`.
fn occurrences<'t>(text: &'t str, needle: &'t str) -> impl Iterator<Item = usize> + 't {
    let mut from = 0;
    std::iter::from_fn(move || {
        let at = text[from..].find(needle)? + from;
        from = at + 1;
        Some(at)
    })
}

fn preceded_by_word(text: &str, at: usize) -> bool {
    text[..at].chars().next_back().is_some_and(is_word)
}

fn followed_by_word(text: &str, at: usize) -> bool {
    text[at..].chars().next().is_some_and(is_word)
}

fn is_word(glyph: char) -> bool {
    glyph.is_ascii_alphanumeric() || glyph == '_'
}

/// The date on the document's `Last verified` line.
fn verification_date(text: &str) -> Option<&str> {
    text.lines().find_map(|line| {
        let date = line.strip_prefix("Last verified: ")?;
        let shaped = date.len() == 10
            && date.as_bytes()[4] == b'-'
            && date.as_bytes()[7] == b'-'
            && date
                .bytes()
                .enumerate()
                .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit());
        shaped.then_some(date)
    })
}

/// Collapse every run of whitespace so a reflowed paragraph still names its token.
fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every optimization lane names scope that exists.
fn judge_lanes(
    tree: &Tree,
    members: &[crate::gates::scan::Member],
    findings: &mut Vec<Finding>,
) -> Result<(), GateError> {
    let table = tree.read_toml(LANES)?;
    let lanes = table.get("lane").and_then(toml::Value::as_table);
    let Some(lanes) = lanes else {
        findings.push(Finding::in_file(
            LANES,
            "the lane registry has no [lane.*] tables",
            LANE_FIX,
        ));
        return Ok(());
    };
    if lanes.is_empty() {
        findings.push(Finding::in_file(
            LANES,
            "the lane registry declares no lane",
            LANE_FIX,
        ));
    }
    let packages: BTreeSet<&str> = members.iter().map(|member| member.name.as_str()).collect();
    for (lane, body) in lanes {
        let Some(body) = body.as_table() else {
            findings.push(Finding::in_file(
                LANES,
                format!("lane `{lane}` is not a table"),
                LANE_FIX,
            ));
            continue;
        };
        for key in ["purpose", "layer"] {
            let stated = body
                .get(key)
                .and_then(toml::Value::as_str)
                .is_some_and(|value| !value.trim().is_empty());
            if !stated {
                findings.push(Finding::in_file(
                    LANES,
                    format!("lane `{lane}` states no {key}"),
                    LANE_FIX,
                ));
            }
        }
        for key in ["write", "required_commands"] {
            let populated = body
                .get(key)
                .and_then(toml::Value::as_array)
                .is_some_and(|entries| !entries.is_empty());
            if !populated {
                findings.push(Finding::in_file(
                    LANES,
                    format!("lane `{lane}` has no {key} entries"),
                    LANE_FIX,
                ));
            }
        }
        judge_lane_scopes(tree, lane, body, findings);
        judge_lane_commands(&packages, lane, body, findings);
    }
    Ok(())
}

/// Every `write` and `avoid` entry of one lane names a repository-relative
/// scope that matches something in the tree.
fn judge_lane_scopes(tree: &Tree, lane: &str, body: &toml::Table, findings: &mut Vec<Finding>) {
    for key in ["write", "avoid"] {
        let entries = body
            .get(key)
            .and_then(toml::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for entry in entries {
            let Some(pattern) = entry.as_str().filter(|value| !value.trim().is_empty()) else {
                findings.push(Finding::in_file(
                    LANES,
                    format!("lane `{lane}` has an empty {key} entry"),
                    LANE_FIX,
                ));
                continue;
            };
            if pattern.starts_with('/') || pattern.split('/').any(|segment| segment == "..") {
                findings.push(Finding::in_file(
                    LANES,
                    format!(
                        "lane `{lane}` {key} entry `{pattern}` is not a repository-relative path"
                    ),
                    LANE_FIX,
                ));
                continue;
            }
            // A lane entry names a path scope, so `vyre-driver-*` covers the
            // files under every directory it matches as well as a file of
            // that name. Requiring the glob to match a whole path would
            // report every entry written as a directory, which is most of
            // them, and reporting a coherent registry is how a rule gets
            // switched off.
            let subtree = format!("{}/**", pattern.trim_end_matches('/'));
            let matched = tree.paths().iter().any(|path| {
                let path = path.to_string_lossy();
                glob_match(pattern, path.as_ref()) || glob_match(&subtree, path.as_ref())
            }) || tree.exists(pattern);
            if !matched {
                findings.push(Finding::in_file(
                    LANES,
                    format!("lane `{lane}` {key} entry `{pattern}` matches nothing in the tree"),
                    LANE_FIX,
                ));
            }
        }
    }
}

/// Every `-p` package one lane's required commands name is a workspace member.
fn judge_lane_commands(
    packages: &BTreeSet<&str>,
    lane: &str,
    body: &toml::Table,
    findings: &mut Vec<Finding>,
) {
    let commands = body
        .get("required_commands")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for command in commands {
        let Some(command) = command.as_str().filter(|value| !value.trim().is_empty()) else {
            findings.push(Finding::in_file(
                LANES,
                format!("lane `{lane}` has an empty required command"),
                LANE_FIX,
            ));
            continue;
        };
        findings.extend(undeclared_command_packages(packages, lane, command));
    }
}

/// Findings for each `-p` package one command names that no workspace manifest
/// declares, and for a `-p` with nothing after it.
fn undeclared_command_packages(
    packages: &BTreeSet<&str>,
    lane: &str,
    command: &str,
) -> Vec<Finding> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| **token == "-p")
        .filter_map(|(index, _)| match tokens.get(index + 1) {
            None => Some(Finding::in_file(
                LANES,
                format!("lane `{lane}` command `{command}` ends with a bare -p"),
                LANE_FIX,
            )),
            Some(package) if !packages.contains(package) => Some(Finding::in_file(
                LANES,
                format!(
                    "lane `{lane}` command `{command}` names package `{package}`, which no workspace manifest declares"
                ),
                LANE_FIX,
            )),
            Some(_) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the version and op-count phrases are the ones a reader most easily
    /// leaves behind, and a substring search for them reports every larger number
    /// that happens to contain the digits.
    #[test]
    fn a_stale_version_is_bounded_and_a_longer_number_is_not_one() {
        assert!(word_bounded("vyre 0.6 shipped", "0.6"));
        assert!(!word_bounded("vyre 10.62 shipped", "0.6"));
        assert!(word_bounded("the 9-op core", "9-op"));
        assert!(!word_bounded("the 19-oprah core", "9-op"));
    }

    #[test]
    fn the_primary_path_claim_is_found_only_within_its_window() {
        assert!(wgpu_primary_path("wgpu is the primary production path").is_some());
        let far = format!("wgpu{}primary production path", " ".repeat(60));
        assert!(wgpu_primary_path(&far).is_none());
        assert!(wgpu_primary_path("wgpu is one backend").is_none());
    }

    #[test]
    fn an_agent_identifier_needs_a_suffix() {
        assert!(agent_identifier("built by codex-9ab").is_some());
        assert!(agent_identifier("codex- alone is prose").is_none());
    }

    #[test]
    fn a_verification_date_must_be_shaped_like_a_date() {
        assert_eq!(
            verification_date("intro\nLast verified: 2026-08-15\nrest"),
            Some("2026-08-15")
        );
        assert_eq!(verification_date("Last verified: soon"), None);
        assert_eq!(verification_date("no line at all"), None);
    }

    #[test]
    fn a_required_token_survives_a_reflowed_paragraph() {
        assert!(normalize_whitespace("the\n  live\tregistry").contains("the live registry"));
    }

    /// WHY: a lane entry names a path scope, and most are written as a directory
    /// or a directory glob. Requiring the pattern to match a whole path reported
    /// `vyre-driver-*` as matching nothing while two driver crates were sitting
    /// in the tree, and a rule that reports a coherent registry gets switched
    /// off. The absent case must still fail, or the rule covers nothing.
    #[test]
    fn a_lane_entry_may_name_a_directory_or_a_directory_glob() {
        let listed = ["vyre-driver-cuda/src/lib.rs", "vyre-libs/src/pattern/dfa.rs"];
        for pattern in ["vyre-driver-*", "vyre-libs", "vyre-libs/src/pattern/**"] {
            let subtree = format!("{}/**", pattern.trim_end_matches('/'));
            assert!(
                listed
                    .iter()
                    .any(|path| glob_match(pattern, path) || glob_match(&subtree, path)),
                "`{pattern}` names a scope two listed files sit inside"
            );
        }
        for pattern in ["vyre-deleted", "vyre-driver-*/tests/**"] {
            let subtree = format!("{pattern}/**");
            assert!(
                !listed
                    .iter()
                    .any(|path| glob_match(pattern, path) || glob_match(&subtree, path)),
                "`{pattern}` names a scope nothing is inside"
            );
        }
    }
}
