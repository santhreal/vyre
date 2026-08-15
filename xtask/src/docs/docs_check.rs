//! The `docs-check` gate: what `docs/DOCS.toml` declares, what `docs/` holds,
//! and whether a reader can follow the links in either.
//!
//! The manifest is the authority. Every published page declares its lifecycle,
//! audience, owner, kind, reader-task section, authority source and generation
//! mode, and a generated page also names the generator that writes it. The two
//! navigation documents named below are rendered from those rows, so the
//! manifest and the navigation cannot disagree.
//!
//! The published set is read from the working tree rather than the git index.
//! Reading the index made a new page a failure only once it was staged, so
//! adding a crate went green locally and red in CI on somebody else's commit.
//! A page that is gitignored is not published and is skipped, because a reader
//! who clones the repository never receives it.
//!
//! Link resolution lives here rather than beside it, because the set of links
//! worth checking is exactly the set of pages the manifest calls active. Three
//! classes, in descending severity: the target escapes the repository root, so
//! it resolves for nobody; the target does not exist, so it resolves for
//! nobody; the target exists here and is gitignored, so it resolves for the
//! author and fails for every other reader. Anchor fragments are out of scope:
//! whether `#some-heading` still exists is a far weaker signal than whether the
//! file does, and folding it in buries the classes that matter under heading
//! churn.
//!
//! Historical pages are excluded from navigation and from link resolution. A
//! document under `docs/archive/` or `docs/legacy/` is a snapshot of what was
//! true on its date, and rewriting its links to resolve against today's tree
//! would falsify the record.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::output_arg::read_text_bounded;
use crate::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

/// The documentation tree this gate judges.
const DOCS: &str = "docs";
/// The manifest that declares every published page.
const MANIFEST: &str = "docs/DOCS.toml";
/// Generated navigation: the reading order.
const SUMMARY: &str = "docs/SUMMARY.md";
/// Generated navigation: the authority and lifecycle table.
const INDEX: &str = "docs/INDEX.md";
/// Bound on any one document this gate reads.
const MAX_DOCUMENT_BYTES: u64 = 8_388_608;
/// Manifest schema this gate reads.
const MANIFEST_VERSION: i64 = 2;

/// Lifecycle states a page may declare.
const STATUSES: [&str; 4] = ["archived", "current", "generated", "superseded"];
/// Readers a page may be written for.
const AUDIENCES: [&str; 4] = ["contributor", "extension", "release", "user"];
/// Whether a page is authored or written by a generator.
const GENERATIONS: [&str; 2] = ["generated", "manual"];
/// What a page is, independent of who reads it.
const KINDS: [&str; 11] = [
    "architecture",
    "evidence",
    "governance",
    "guide",
    "history",
    "lifecycle",
    "optimization",
    "ownership",
    "reference",
    "release",
    "testing",
];
/// Reader-task sections, in the order the summary renders them. A section the
/// manifest names and this list does not renders after these, alphabetically.
const SECTION_ORDER: [&str; 8] = [
    "Documentation authority",
    "Architecture and ownership",
    "Lifecycle and extension contracts",
    "Optimization",
    "User workflows",
    "API and operation reference",
    "Testing and conformance",
    "Performance and release",
];
/// Directories that hold snapshots rather than current claims.
const HISTORICAL_DIRECTORIES: [&str; 2] = ["archive/", "legacy/"];
/// Lifecycle states whose pages are still current claims.
const ACTIVE_STATUSES: [&str; 2] = ["current", "generated"];
/// Audiences outside this repository, whose pages must not carry the vocabulary
/// of how work here is produced.
const EXTERNAL_AUDIENCES: [&str; 2] = ["extension", "user"];
/// How to regenerate what this gate owns.
const REGENERATE: &str = "regenerate the navigation with `cargo_full run --bin xtask -- docs-check --write`";

/// One `[[page]]` row of the manifest.
struct Page {
    /// Path under `docs/`.
    path: String,
    /// Heading the navigation shows.
    title: String,
    /// Lifecycle state.
    status: String,
    /// Reader the page is written for.
    audience: String,
    /// Documentation owner that answers for it.
    owner: String,
    /// What the page is.
    kind: String,
    /// Reader-task section it navigates under.
    section: String,
    /// Path under `docs/` this page states the content of, or `self`.
    authority: String,
    /// Authored or generated.
    generation: String,
    /// Path under `docs/` of the generator, empty for an authored page.
    generator: String,
    /// Whether the page appears in navigation. `None` when the row omits it,
    /// which is a defect either way.
    nav: Option<bool>,
}

impl Page {
    /// Read one row, treating a missing or non-string key as empty.
    ///
    /// An empty field fails its own rule below with the sentence that names the
    /// field, which reads better than one sentence about a malformed row.
    fn from_row(row: &toml::Table) -> Self {
        let text = |key: &str| {
            row.get(key)
                .and_then(toml::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        Self {
            path: text("path"),
            title: text("title"),
            status: text("status"),
            audience: text("audience"),
            owner: text("owner"),
            kind: text("kind"),
            section: text("section"),
            authority: text("authority"),
            generation: text("generation"),
            generator: text("generator"),
            nav: row.get("nav").and_then(toml::Value::as_bool),
        }
    }

    /// Whether the page is a current claim rather than a lifecycle record.
    fn is_active(&self) -> bool {
        ACTIVE_STATUSES.contains(&self.status.as_str())
    }

    /// Whether the page navigates.
    fn navigates(&self) -> bool {
        self.nav == Some(true)
    }
}

/// Holds the documentation manifest to the pages, the navigation and the links
/// on disk.
pub struct DocsCheck;

impl Gate for DocsCheck {
    fn name(&self) -> &'static str {
        "docs-check"
    }

    fn help(&self) -> &'static str {
        "Hold the manifest-backed documentation lifecycle, generated navigation and public links to the tree; --write regenerates the navigation"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let docs = ctx.root.join(DOCS);
        let mut report = Report::clean();

        let Some((owners, pages)) = load_manifest(&ctx.root, &mut report)? else {
            return Ok(report);
        };
        let published = published_pages(&ctx.root, &docs)?;
        for finding in validate(&docs, &owners, &pages, &published) {
            report.find(finding);
        }
        if report.count() > 0 {
            // The manifest is the authority for everything below, so a manifest
            // that does not hold is not a tree to render navigation from.
            return Ok(report);
        }

        let rendered = [
            (SUMMARY, render_summary(&pages)),
            (INDEX, render_index(&owners, &pages)),
        ];
        if ctx.write {
            for (path, content) in &rendered {
                write_file(&ctx.root.join(path), content)?;
            }
            report.note("wrote the summary and the authority index".to_string());
        } else {
            for (path, content) in &rendered {
                if read_text(&ctx.root, path)?.as_str() != content.as_str() {
                    report.find(Finding::in_file(
                        *path,
                        "the generated navigation disagrees with the documentation manifest",
                        REGENERATE,
                    ));
                }
            }
        }

        // Links are resolved after the navigation is written, because the
        // navigation is generated: a stale summary linking a page the manifest
        // no longer holds used to report a broken link and then refuse to write
        // the summary that no longer links it.
        let links = link_findings(&ctx.root, &pages, &mut report)?;
        report.note(format!(
            "{} published page(s), {links} outbound link(s)",
            pages.len()
        ));
        Ok(report)
    }
}

/// The owner table and the page rows, or `None` when the manifest cannot be
/// judged row by row.
///
/// A manifest that does not parse is a gate that could not run. A manifest that
/// parses and declares the wrong shape is a finding: the tree is wrong, not the
/// gate.
fn load_manifest(
    root: &Path,
    report: &mut Report,
) -> Result<Option<(BTreeMap<String, String>, Vec<Page>)>, GateError> {
    let text = read_text(root, MANIFEST)?;
    let document: toml::Table = toml::from_str(&text).map_err(|error| {
        GateError::new(
            format!("`{MANIFEST}` does not parse as TOML: {error}"),
            "fix the syntax the parser names; the documentation surface is declared here",
        )
    })?;

    if document.get("version").and_then(toml::Value::as_integer) != Some(MANIFEST_VERSION) {
        report.find(Finding::in_file(
            MANIFEST,
            format!("the manifest does not declare `version = {MANIFEST_VERSION}`"),
            "state the schema version this gate reads",
        ));
        return Ok(None);
    }

    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    let owner_rows = document
        .get("owner")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if owner_rows.is_empty() {
        report.find(Finding::in_file(
            MANIFEST,
            "the manifest declares no documentation owner",
            "declare one [[owner]] per party that answers for a page, with the source it answers from",
        ));
        return Ok(None);
    }
    for (index, row) in owner_rows.iter().enumerate() {
        let row = row.as_table();
        let id = row
            .and_then(|table| table.get("id"))
            .and_then(toml::Value::as_str)
            .unwrap_or_default();
        let authority = row
            .and_then(|table| table.get("authority"))
            .and_then(toml::Value::as_str)
            .unwrap_or_default();
        if id.is_empty() || authority.is_empty() {
            report.find(Finding::in_file(
                MANIFEST,
                format!("owner row {} declares no id or no authority", index + 1),
                "give every owner an id and the source it answers from",
            ));
            continue;
        }
        if owners.insert(id.to_string(), authority.to_string()).is_some() {
            report.find(Finding::in_file(
                MANIFEST,
                format!("duplicate documentation owner: {id}"),
                "one owner per id; two rows for one id give a page two authorities",
            ));
        }
    }

    let Some(page_rows) = document.get("page").and_then(toml::Value::as_array) else {
        report.find(Finding::in_file(
            MANIFEST,
            "the manifest declares no page",
            "declare one [[page]] per published document under docs/",
        ));
        return Ok(None);
    };
    let pages: Vec<Page> = page_rows
        .iter()
        .filter_map(toml::Value::as_table)
        .map(Page::from_row)
        .collect();
    if pages.len() != page_rows.len() {
        report.find(Finding::in_file(
            MANIFEST,
            "a [[page]] entry is not a table",
            "declare every page as a [[page]] table",
        ));
        return Ok(None);
    }
    Ok(Some((owners, pages)))
}

/// Every published Markdown page under `docs/`, relative to `docs/`.
///
/// `SUMMARY.md` is navigation rather than a page, so it carries no row.
fn published_pages(root: &Path, docs: &Path) -> Result<BTreeSet<String>, GateError> {
    let mut found: BTreeSet<String> = BTreeSet::new();
    for entry in tree_walk::pruned(docs, BUILD_OUTPUT_AND_VCS) {
        let entry = entry.map_err(|error| {
            GateError::new(
                format!("could not walk `{DOCS}`: {error}"),
                "check the documentation tree is readable",
            )
        })?;
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().is_none_or(|kind| kind != "md") {
            continue;
        }
        let Ok(relative) = path.strip_prefix(docs) else {
            continue;
        };
        let relative = relative.to_string_lossy().replace('\\', "/");
        if relative != "SUMMARY.md" {
            found.insert(relative);
        }
    }
    let candidates: BTreeSet<String> = found
        .iter()
        .map(|page| format!("{DOCS}/{page}"))
        .collect();
    let ignored = ignored_paths(root, &candidates)?;
    Ok(found
        .into_iter()
        .filter(|page| !ignored.contains(&format!("{DOCS}/{page}")))
        .collect())
}

/// Every finding the manifest rows make against themselves and the tree.
fn validate(
    docs: &Path,
    owners: &BTreeMap<String, String>,
    pages: &[Page],
    published: &BTreeSet<String>,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (owner, authority) in owners {
        if !docs.join(authority).exists() {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("documentation owner {owner}: authority does not exist: {authority}"),
                "point the owner at the source it answers from, or drop the owner",
            ));
        }
    }

    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for page in pages {
        if !seen.insert(page.path.as_str()) {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("duplicate DOCS.toml page: {}", page.path),
                "one row per page; two rows give one document two lifecycles",
            ));
        }
    }
    let declared: BTreeSet<&str> = pages.iter().map(|page| page.path.as_str()).collect();
    for path in published {
        if !declared.contains(path.as_str()) {
            findings.push(Finding::in_file(
                format!("{DOCS}/{path}"),
                "unclassified documentation page",
                "declare the page in docs/DOCS.toml with its lifecycle, audience, owner and authority, or delete it",
            ));
        }
    }
    for path in &declared {
        if !published.contains(*path) {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("DOCS.toml names missing page: {path}"),
                "write the page, or drop its row; a classified page nobody can open is not documentation",
            ));
        }
    }

    for page in pages {
        findings.extend(page_findings(docs, owners, page));
    }
    findings
}

/// Every finding one page row makes.
fn page_findings(docs: &Path, owners: &BTreeMap<String, String>, page: &Page) -> Vec<Finding> {
    let mut findings = Vec::new();
    let path = page.path.as_str();
    let mut find = |message: String, fix: &str| {
        findings.push(Finding::in_file(MANIFEST, message, fix));
    };

    if !STATUSES.contains(&page.status.as_str()) {
        find(
            format!("{path}: invalid lifecycle `{}`", page.status),
            "declare one of current, generated, superseded or archived",
        );
    }
    if !AUDIENCES.contains(&page.audience.as_str()) {
        find(
            format!("{path}: invalid or missing audience `{}`", page.audience),
            "name the reader: user, extension, contributor or release",
        );
    }
    if !owners.contains_key(&page.owner) {
        find(
            format!(
                "{path}: unknown or deleted documentation owner `{}`",
                page.owner
            ),
            "point the page at a declared [[owner]], or declare that owner",
        );
    }
    if !KINDS.contains(&page.kind.as_str()) {
        find(
            format!("{path}: invalid or missing document kind `{}`", page.kind),
            "state what the page is; the kinds are listed in the gate",
        );
    }
    if page.section.is_empty() {
        find(
            format!("{path}: missing reader-task section"),
            "name the task a reader is doing when they open it; the section is the navigation heading",
        );
    }
    if page.title.is_empty() {
        find(
            format!("{path}: missing title"),
            "state the heading the navigation shows",
        );
    }
    if !GENERATIONS.contains(&page.generation.as_str()) {
        find(
            format!(
                "{path}: invalid or missing generation mode `{}`",
                page.generation
            ),
            "declare generation = \"manual\" for an authored page or \"generated\" for one a generator writes",
        );
    }
    if page.authority.is_empty() {
        find(
            format!("{path}: missing authority source"),
            "name the source the page states, or `self` when the page is the source",
        );
    } else if page.authority != "self" && !docs.join(&page.authority).exists() {
        find(
            format!("{path}: authority source does not exist: {}", page.authority),
            "point the page at a source that exists, or make the page its own authority",
        );
    }
    if HISTORICAL_DIRECTORIES
        .iter()
        .any(|directory| path.starts_with(directory))
        && page.status != "archived"
    {
        find(
            format!("{path}: historical directories require archived lifecycle"),
            "archive the page, or move it out of the historical directory",
        );
    }
    if matches!(page.status.as_str(), "archived" | "superseded") && page.nav != Some(false) {
        find(
            format!("{path}: inactive pages must set nav = false"),
            "keep lifecycle records out of navigation; a reader following the summary must reach current claims",
        );
    }
    if page.is_active() && path.ends_with(".md") && page.nav != Some(true) {
        find(
            format!("{path}: active Markdown pages must set nav = true"),
            "navigate every current page; a page nothing links to is a page nobody reads",
        );
    }
    if page.status == "generated" && page.generation != "generated" {
        find(
            format!("{path}: generated lifecycle requires generated ownership"),
            "declare generation = \"generated\" beside the generated lifecycle",
        );
    }
    if page.status != "generated" && page.generation == "generated" {
        find(
            format!("{path}: generated ownership requires generated lifecycle"),
            "declare status = \"generated\" beside the generated ownership",
        );
    }
    if page.generation == "generated" {
        if page.generator.is_empty() {
            find(
                format!("{path}: generated page must name one generator"),
                "name the source file that writes the page",
            );
        } else if !docs.join(&page.generator).exists() {
            find(
                format!("{path}: generator does not exist: {}", page.generator),
                "point the page at the generator that writes it today",
            );
        }
        if page.authority == "self" {
            find(
                format!("{path}: generated page cannot be its own authority"),
                "name the source the generator reads",
            );
        }
    } else if !page.generator.is_empty() {
        find(
            format!("{path}: manual page cannot name a generator"),
            "drop the generator, or declare the page generated and let the generator write it",
        );
    }

    if page.is_active()
        && EXTERNAL_AUDIENCES.contains(&page.audience.as_str())
        && page.generation == "manual"
    {
        if let Ok(content) = fs::read_to_string(docs.join(path)) {
            for marker in MARKERS {
                if (marker.matches)(&content) {
                    findings.push(Finding::in_file(
                        format!("{DOCS}/{path}"),
                        format!("{} document leaks {}", page.audience, marker.label),
                        "state the product fact without the process that produced it; a reader outside this repository cannot act on it",
                    ));
                }
            }
        }
    }
    findings
}

/// One vocabulary of internal process a published page must not carry.
struct Marker {
    /// What the reader is being shown, named in the finding.
    label: &'static str,
    /// Whether a document carries it.
    matches: fn(&str) -> bool,
}

/// Every internal-process vocabulary this gate rejects.
const MARKERS: &[Marker] = &[
    Marker {
        label: "local planning URI",
        matches: leaks_planning_uri,
    },
    Marker {
        label: "execution backlog",
        matches: leaks_backlog,
    },
    Marker {
        label: "agent execution process",
        matches: leaks_agent_process,
    },
    Marker {
        label: "internal phase identifier",
        matches: leaks_phase_identifier,
    },
];

/// A `local://` link resolves only in the session that wrote it.
fn leaks_planning_uri(content: &str) -> bool {
    content.to_ascii_lowercase().contains("local://")
}

/// The execution backlog is not tracked and names work nobody outside has.
fn leaks_backlog(content: &str) -> bool {
    contains_word(content, "BACKLOG.md", true)
}

/// How work here is produced is not a product fact.
fn leaks_agent_process(content: &str) -> bool {
    ["subagent", "agent swarm", "worktree protocol"]
        .iter()
        .any(|phrase| contains_word(content, phrase, false))
}

/// A numbered phase, slice or tranche names a plan a reader cannot see.
fn leaks_phase_identifier(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    for keyword in ["phase", "slice", "tranche"] {
        let mut at = 0;
        while let Some(found) = lower[at..].find(keyword) {
            let start = at + found;
            let end = start + keyword.len();
            at = end;
            if !word_bounded(&lower, start, end) {
                continue;
            }
            let rest = &lower[end..];
            let spaced = rest.trim_start_matches([' ', '\t', '\n', '\r']);
            if spaced.len() == rest.len() {
                continue;
            }
            let identifier = spaced.strip_prefix(|first: char| first.is_ascii_alphabetic());
            let digits = identifier.unwrap_or(spaced);
            let width = digits
                .find(|character: char| !character.is_ascii_digit())
                .unwrap_or(digits.len());
            if width == 0 {
                continue;
            }
            let after = digits[width..].chars().next();
            if after.is_none_or(|character| !is_word_character(character)) {
                return true;
            }
        }
    }
    false
}

/// Whether `needle` appears in `haystack` bounded by non-word characters.
fn contains_word(haystack: &str, needle: &str, case_sensitive: bool) -> bool {
    let (haystack, needle) = if case_sensitive {
        (haystack.to_string(), needle.to_string())
    } else {
        (
            haystack.to_ascii_lowercase(),
            needle.to_ascii_lowercase(),
        )
    };
    let mut at = 0;
    while let Some(found) = haystack[at..].find(&needle) {
        let start = at + found;
        let end = start + needle.len();
        at = end;
        if word_bounded(&haystack, start, end) {
            return true;
        }
    }
    false
}

/// Whether the span `start..end` of `text` has a non-word character on each
/// side, which is what a regex word boundary asserts.
fn word_bounded(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    before.is_none_or(|character| !is_word_character(character))
        && after.is_none_or(|character| !is_word_character(character))
}

/// Whether a character is one a word boundary sits beside.
fn is_word_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// Findings for every outbound link in the published navigation, and how many
/// links were judged.
fn link_findings(
    root: &Path,
    pages: &[Page],
    report: &mut Report,
) -> Result<usize, GateError> {
    let mut documents: Vec<String> = vec![SUMMARY.to_string()];
    documents.extend(
        pages
            .iter()
            .filter(|page| page.navigates())
            .map(|page| format!("{DOCS}/{}", page.path)),
    );

    let mut sites: Vec<(String, u32, String, String)> = Vec::new();
    for document in &documents {
        let path = root.join(document);
        let Ok(content) = read_text_bounded(&path, MAX_DOCUMENT_BYTES, "documentation link") else {
            continue;
        };
        let base = Path::new(document)
            .parent()
            .map(|parent| parent.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        for (line, raw) in markdown_links(&content) {
            let Some(resolved) = resolve_link(&base, &raw) else {
                continue;
            };
            sites.push((document.clone(), line, raw, resolved));
        }
    }

    let candidates: BTreeSet<String> = sites
        .iter()
        .map(|(_, _, _, resolved)| resolved.clone())
        .filter(|resolved| !resolved.starts_with(".."))
        .collect();
    let ignored = ignored_paths(root, &candidates)?;

    for (document, line, raw, resolved) in &sites {
        if resolved.starts_with("..") {
            report.find(Finding::at(
                document.clone(),
                *line,
                format!("link [{raw}] escapes the repository root"),
                "repoint the link inside the repository; a path above the root resolves for nobody",
            ));
        } else if !root.join(resolved).exists() {
            report.find(Finding::at(
                document.clone(),
                *line,
                format!("link [{raw}] names no such path: {resolved}"),
                "repoint the link at a published document, or drop the whole claim; deleting the link syntax and leaving the sentence promising it is not a fix",
            ));
        } else if ignored.contains(resolved) {
            report.find(Finding::at(
                document.clone(),
                *line,
                format!("link [{raw}] resolves to {resolved}, which the repository excludes"),
                "publish the target, or state inline that it is not published; a link only the author can follow is worse than no link",
            ));
        }
    }
    Ok(sites.len())
}

/// Every `[text](target)` link in `content`, with the 1-based line it sits on.
///
/// Nested brackets end the candidate, and a target carrying whitespace or a
/// parenthesis is not a link this gate can resolve.
fn markdown_links(content: &str) -> Vec<(u32, String)> {
    let mut links = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let bytes = line.as_bytes();
        let mut at = 0;
        while at < bytes.len() {
            if bytes[at] != b'[' {
                at += 1;
                continue;
            }
            let mut close = at + 1;
            while close < bytes.len() && bytes[close] != b']' && bytes[close] != b'[' {
                close += 1;
            }
            if close >= bytes.len() || bytes[close] != b']' || close + 1 >= bytes.len() {
                at += 1;
                continue;
            }
            if bytes[close + 1] != b'(' {
                at = close + 1;
                continue;
            }
            let start = close + 2;
            let mut end = start;
            while end < bytes.len()
                && !matches!(bytes[end], b')' | b'(' | b' ' | b'\t')
            {
                end += 1;
            }
            if end >= bytes.len() || bytes[end] != b')' || end == start {
                at = close + 1;
                continue;
            }
            links.push((index as u32 + 1, line[start..end].to_string()));
            at = end + 1;
        }
    }
    links
}

/// The repository-relative path a link resolves to, or `None` when the link
/// leaves the repository or names an anchor in the page itself.
fn resolve_link(base: &str, raw: &str) -> Option<String> {
    for scheme in ["http://", "https://", "mailto:"] {
        if raw.starts_with(scheme) {
            return None;
        }
    }
    let target = raw.split('#').next().unwrap_or_default();
    if target.is_empty() {
        return None;
    }
    let joined = match target.strip_prefix('/') {
        Some(rooted) => rooted.to_string(),
        None => format!("{base}/{target}"),
    };
    let normalized = normalize(&joined);
    (!normalized.is_empty()).then_some(normalized)
}

/// Collapse `.` and `..` textually.
///
/// `Path::canonicalize` cannot be used: the whole point is to classify targets
/// that do not exist, and a path that does not exist cannot be canonicalized.
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => match parts.last() {
                Some(last) if *last != ".." => {
                    parts.pop();
                }
                _ => parts.push(".."),
            },
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// Which of `candidates` the repository excludes.
///
/// Outside a work tree there is no ignore data, so nothing is excluded. A
/// `git check-ignore` that fails for any other reason is a gate that could not
/// run: reporting every link as published would be worse than saying so.
fn ignored_paths(root: &Path, candidates: &BTreeSet<String>) -> Result<BTreeSet<String>, GateError> {
    if candidates.is_empty() || !is_work_tree(root) {
        return Ok(BTreeSet::new());
    }
    let mut child = Command::new("git")
        .args(["check-ignore", "--stdin"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            GateError::new(
                format!("could not run `git check-ignore`: {error}"),
                "install git, which the gate needs to tell a published target from an excluded one",
            )
        })?;
    if let Some(stdin) = child.stdin.as_mut() {
        let joined = candidates.iter().cloned().collect::<Vec<_>>().join("\n");
        stdin.write_all(joined.as_bytes()).map_err(|error| {
            GateError::new(
                format!("could not send paths to `git check-ignore`: {error}"),
                "check the checkout is a readable git work tree",
            )
        })?;
    }
    let output = child.wait_with_output().map_err(|error| {
        GateError::new(
            format!("could not read `git check-ignore`: {error}"),
            "check the checkout is a readable git work tree",
        )
    })?;
    match output.status.code() {
        Some(0 | 1) => Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_string)
            .collect()),
        code => Err(GateError::new(
            format!(
                "`git check-ignore` exited {}: {}",
                code.unwrap_or(-1),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "make the ignore query answerable; without it the gate cannot tell a published target from an excluded one",
        )),
    }
}

/// Whether `root` is inside a git work tree.
fn is_work_tree(root: &Path) -> bool {
    Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// The reading order, grouped by reader task.
fn render_summary(pages: &[Page]) -> String {
    let mut groups: BTreeMap<&str, Vec<&Page>> = BTreeMap::new();
    for page in pages {
        if page.navigates() && page.path != "INDEX.md" {
            groups.entry(page.section.as_str()).or_default().push(page);
        }
    }
    let mut lines = vec![
        "<!-- Generated from docs/DOCS.toml by xtask docs-check. -->".to_string(),
        "# Summary".to_string(),
        String::new(),
        "- [Documentation authority and lifecycle](INDEX.md)".to_string(),
    ];
    let mut sections: Vec<&str> = SECTION_ORDER.to_vec();
    sections.extend(
        groups
            .keys()
            .copied()
            .filter(|section| !SECTION_ORDER.contains(section)),
    );
    for section in sections {
        let Some(members) = groups.get(section) else {
            continue;
        };
        let mut members = members.clone();
        members.sort_by(|left, right| {
            (&left.title, &left.path).cmp(&(&right.title, &right.path))
        });
        lines.extend([String::new(), format!("# {section}"), String::new()]);
        for page in members {
            lines.push(format!("- [{}]({})", page.title, page.path));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

/// The authority and lifecycle table.
fn render_index(owners: &BTreeMap<String, String>, pages: &[Page]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for page in pages {
        *counts.entry(page.status.as_str()).or_default() += 1;
    }
    let mut lines = vec![
        "<!-- Generated from docs/DOCS.toml by xtask docs-check. Do not edit. -->".to_string(),
        "# Documentation Authority and Lifecycle".to_string(),
        String::new(),
        "Source: [`docs/DOCS.toml`](DOCS.toml).".to_string(),
        String::new(),
        "Each active page declares its audience, owner, authority source, kind, and".to_string(),
        "generation mode. Generated pages also declare the generator. Superseded and".to_string(),
        "archived pages remain lifecycle evidence and are excluded from navigation.".to_string(),
        String::new(),
        "## Documentation owners".to_string(),
        String::new(),
        "| Owner | Authority |".to_string(),
        "| --- | --- |".to_string(),
    ];
    for (owner, authority) in owners {
        lines.push(format!("| `{owner}` | [`{authority}`]({authority}) |"));
    }
    lines.extend([
        String::new(),
        "## Lifecycle counts".to_string(),
        String::new(),
    ]);
    for status in ["current", "generated", "superseded", "archived"] {
        lines.push(format!(
            "- {status}: {}.",
            counts.get(status).copied().unwrap_or_default()
        ));
    }
    lines.extend([
        String::new(),
        "## Pages".to_string(),
        String::new(),
        "| Status | Audience | Owner | Kind | Page | Authority | Generation |".to_string(),
        "| --- | --- | --- | --- | --- | --- | --- |".to_string(),
    ]);
    let mut sorted: Vec<&Page> = pages.iter().collect();
    sorted.sort_by(|left, right| left.path.cmp(&right.path));
    for page in sorted {
        let authority = if page.authority == "self" {
            "self".to_string()
        } else {
            format!("[{0}]({0})", page.authority)
        };
        let generation = if page.generator.is_empty() {
            page.generation.clone()
        } else {
            format!("{}: [{1}]({1})", page.generation, page.generator)
        };
        lines.push(format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {authority} | {generation} |",
            page.status, page.audience, page.owner, page.kind, page.path
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

/// Read one document under the checkout root, under the read bound.
fn read_text(root: &Path, relative: &str) -> Result<String, GateError> {
    read_text_bounded(&root.join(relative), MAX_DOCUMENT_BYTES, "documentation").map_err(|error| {
        GateError::new(
            format!("could not read `{relative}`: {error}"),
            "restore the document; the documentation surface is declared and rendered from it",
        )
    })
}

/// Write one generated document, creating the directory it lives in.
fn write_file(path: &Path, content: &str) -> Result<(), GateError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            GateError::new(
                format!("could not create `{}`: {error}", parent.display()),
                "check the checkout is writable",
            )
        })?;
    }
    fs::write(path, content).map_err(|error| {
        GateError::new(
            format!("could not write `{}`: {error}", path.display()),
            "check the checkout is writable",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use tempfile::TempDir;

    /// A page row with coherent defaults, overridden field by field.
    fn page(path: &str) -> Page {
        Page {
            path: path.to_string(),
            title: path.to_string(),
            status: "current".to_string(),
            audience: "user".to_string(),
            owner: "docs".to_string(),
            kind: "guide".to_string(),
            section: "User workflows".to_string(),
            authority: "self".to_string(),
            generation: "manual".to_string(),
            generator: String::new(),
            nav: Some(true),
        }
    }

    /// One owner whose authority exists in the fixture.
    fn owners() -> BTreeMap<String, String> {
        BTreeMap::from([("docs".to_string(), "owner.md".to_string())])
    }

    /// A docs directory holding the owner authority.
    fn fixture() -> TempDir {
        let temporary = TempDir::new().expect("a temporary directory");
        fs::create_dir_all(temporary.path().join("docs")).expect("a docs directory");
        fs::write(temporary.path().join("docs/owner.md"), "# Owner\n").expect("the authority");
        temporary
    }

    /// Every message a validation run produced.
    fn messages(findings: &[Finding]) -> String {
        findings
            .iter()
            .map(|finding| finding.message.clone())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// WHY: the coherent case must produce nothing, or every negative below
    /// would pass by rejecting everything.
    #[test]
    fn a_coherent_manifest_reports_nothing() {
        let temporary = fixture();
        let docs = temporary.path().join("docs");
        fs::write(docs.join("guide.md"), "# Guide\n").expect("a page");
        fs::write(docs.join("generated.md"), "# Generated\n").expect("a page");
        fs::write(docs.join("source.toml"), "version = 1\n").expect("an authority");
        fs::write(docs.join("generator.rs"), "// generator\n").expect("a generator");
        let mut second = page("generated.md");
        second.status = "generated".to_string();
        second.audience = "extension".to_string();
        second.authority = "source.toml".to_string();
        second.generation = "generated".to_string();
        second.generator = "generator.rs".to_string();
        let published = BTreeSet::from(["guide.md".to_string(), "generated.md".to_string()]);

        let findings = validate(&docs, &owners(), &[page("guide.md"), second], &published);

        assert!(findings.is_empty(), "{}", messages(&findings));
    }

    /// WHY: each of these is a way the manifest and the tree drift apart, and
    /// each was a real defect once. One fixture proves the whole class stays
    /// covered; a per-rule fixture rots into a list nobody extends.
    #[test]
    fn every_authority_and_lifecycle_drift_is_reported() {
        let temporary = fixture();
        let docs = temporary.path().join("docs");
        fs::write(docs.join("declared.md"), "# Declared\n").expect("a page");
        fs::write(docs.join("unclassified.md"), "# Unclassified\n").expect("a page");
        fs::create_dir_all(docs.join("archive")).expect("a historical directory");
        fs::write(docs.join("archive/old.md"), "# Old\n").expect("a snapshot");
        let mut archived = page("declared.md");
        archived.status = "archived".to_string();
        let mut broken = page("declared.md");
        broken.status = "generated".to_string();
        broken.owner = "removed-owner".to_string();
        broken.authority = "missing.toml".to_string();
        broken.generation = "generated".to_string();
        broken.generator = "missing.rs".to_string();
        let published = BTreeSet::from([
            "declared.md".to_string(),
            "unclassified.md".to_string(),
            "archive/old.md".to_string(),
        ]);

        let findings = validate(
            &docs,
            &owners(),
            &[archived, broken, page("archive/old.md")],
            &published,
        );
        let reported = messages(&findings);

        for expected in [
            "duplicate DOCS.toml page",
            "unclassified documentation page",
            "inactive pages must set nav = false",
            "unknown or deleted documentation owner",
            "authority source does not exist",
            "generator does not exist",
            "historical directories require archived",
        ] {
            assert!(
                reported.contains(expected)
                    || findings
                        .iter()
                        .any(|finding| finding.message.contains(expected)),
                "no finding for {expected}: {reported}"
            );
        }
    }

    /// WHY: a page a generator writes must name the generator, or the tree has
    /// a generated document nobody can regenerate.
    #[test]
    fn a_generated_page_without_a_generator_is_reported() {
        let temporary = fixture();
        let docs = temporary.path().join("docs");
        fs::write(docs.join("generated.md"), "# Generated\n").expect("a page");
        let mut row = page("generated.md");
        row.status = "generated".to_string();
        row.authority = "owner.md".to_string();
        row.generation = "generated".to_string();

        let findings = validate(
            &docs,
            &owners(),
            &[row],
            &BTreeSet::from(["generated.md".to_string()]),
        );

        assert!(
            messages(&findings).contains("must name one generator"),
            "{}",
            messages(&findings)
        );
    }

    /// WHY: the vocabulary of how work here is produced resolves to nothing for
    /// a reader outside this repository, and naming a private plan in a
    /// published page is a disclosure as well as a dead reference.
    #[test]
    fn an_external_page_may_not_carry_internal_process() {
        let temporary = fixture();
        let docs = temporary.path().join("docs");
        fs::write(
            docs.join("public.md"),
            "# Public\n\nRead BACKLOG.md during Phase 3.\n",
        )
        .expect("a page");
        let mut row = page("public.md");
        row.audience = "extension".to_string();

        let findings = validate(
            &docs,
            &owners(),
            &[row],
            &BTreeSet::from(["public.md".to_string()]),
        );
        let reported = messages(&findings);

        assert!(reported.contains("leaks execution backlog"), "{reported}");
        assert!(
            reported.contains("leaks internal phase identifier"),
            "{reported}"
        );
    }

    /// WHY: the same words are how contributors describe their own work, so the
    /// rule is about the audience and not about the words.
    #[test]
    fn a_contributor_page_may_carry_internal_process() {
        let temporary = fixture();
        let docs = temporary.path().join("docs");
        fs::write(
            docs.join("contributor.md"),
            "# Contributor\n\nUpdate BACKLOG.md before Phase 3.\n",
        )
        .expect("a page");
        let mut row = page("contributor.md");
        row.audience = "contributor".to_string();

        let findings = validate(
            &docs,
            &owners(),
            &[row],
            &BTreeSet::from(["contributor.md".to_string()]),
        );

        assert!(findings.is_empty(), "{}", messages(&findings));
    }

    /// WHY: a word-bounded marker must not fire on a longer word that contains
    /// it, or the gate teaches writers to avoid ordinary prose.
    #[test]
    fn a_marker_inside_a_longer_word_is_not_a_leak() {
        assert!(!leaks_phase_identifier("the phased rollout 3 was fine"));
        assert!(!leaks_agent_process("the subagentless design"));
        assert!(!leaks_backlog("see BACKLOG.mdx"));
        assert!(leaks_phase_identifier("finish Tranche B2 first"));
        assert!(leaks_planning_uri("read LOCAL://plan.md"));
    }

    /// WHY: link classification is the whole gate; a normalizer that resolved a
    /// escaping path to a relative one would report every leak as published.
    #[test]
    fn a_link_resolves_against_its_own_directory() {
        assert_eq!(
            resolve_link("docs/testing", "../ARCHITECTURE.md").as_deref(),
            Some("docs/ARCHITECTURE.md")
        );
        assert_eq!(
            resolve_link("docs", "/README.md").as_deref(),
            Some("README.md")
        );
        assert_eq!(
            resolve_link("docs", "../../../STANDARD.md").as_deref(),
            Some("../../STANDARD.md")
        );
        assert_eq!(resolve_link("docs", "#heading"), None);
        assert_eq!(resolve_link("docs", "https://example.invalid/x"), None);
        assert_eq!(
            resolve_link("docs", "CLI.md#usage").as_deref(),
            Some("docs/CLI.md")
        );
    }

    /// WHY: the extractor decides what the link gate can see. A target carrying
    /// a space or a parenthesis is not a path this gate can resolve, and a
    /// bracket nested in the label ends the candidate rather than swallowing the
    /// rest of the line as a target.
    #[test]
    fn only_resolvable_link_targets_are_extracted() {
        let content =
            "see [one](a.md) and [two](b.md)\n![img](c.png)\n[skip](x y.md)\n[label [x]](d.md)\n";
        assert_eq!(
            markdown_links(content),
            vec![
                (1, "a.md".to_string()),
                (1, "b.md".to_string()),
                (2, "c.png".to_string()),
            ]
        );
    }

    /// WHY: the summary is the reading order, so a section the order list does
    /// not name must still render, and `INDEX.md` must appear once at the top
    /// rather than twice.
    #[test]
    fn the_summary_renders_known_sections_first_and_the_index_once() {
        let mut first = page("CLI.md");
        first.title = "CLI".to_string();
        let mut later = page("ODD.md");
        later.title = "Odd".to_string();
        later.section = "Zebra section".to_string();
        let mut index = page("INDEX.md");
        index.section = "Documentation authority".to_string();

        let summary = render_summary(&[later, first, index]);

        let authority = summary.find("# Zebra section").expect("the odd section");
        let workflows = summary.find("# User workflows").expect("the known section");
        assert!(workflows < authority, "{summary}");
        assert_eq!(summary.matches("(INDEX.md)").count(), 1, "{summary}");
        assert!(summary.ends_with("\n"), "{summary}");
    }

    /// A manifest and tree the whole gate accepts, holding one page plus the two
    /// generated navigation documents.
    fn tree(temporary: &TempDir) -> PathBuf {
        let root = temporary.path().to_path_buf();
        let docs = root.join(DOCS);
        fs::create_dir_all(&docs).expect("a docs directory");
        fs::write(docs.join("guide.md"), "# Guide\n").expect("a page");
        fs::write(docs.join("gen.rs"), "// generator\n").expect("a generator");
        fs::write(
            docs.join("DOCS.toml"),
            "version = 2\n\
             \n\
             [[owner]]\n\
             id = \"docs\"\n\
             authority = \"guide.md\"\n\
             \n\
             [[page]]\n\
             path = \"guide.md\"\n\
             title = \"Guide\"\n\
             status = \"current\"\n\
             audience = \"user\"\n\
             owner = \"docs\"\n\
             kind = \"guide\"\n\
             section = \"User workflows\"\n\
             authority = \"self\"\n\
             generation = \"manual\"\n\
             nav = true\n\
             \n\
             [[page]]\n\
             path = \"INDEX.md\"\n\
             title = \"Documentation authority and lifecycle\"\n\
             status = \"generated\"\n\
             audience = \"contributor\"\n\
             owner = \"docs\"\n\
             kind = \"governance\"\n\
             section = \"Documentation authority\"\n\
             authority = \"DOCS.toml\"\n\
             generation = \"generated\"\n\
             generator = \"gen.rs\"\n\
             nav = true\n",
        )
        .expect("the manifest");
        root
    }

    /// WHY: the navigation is generated, and a link inside it used to block the
    /// write that would have fixed it. Deleting a page from the manifest left the
    /// stale summary linking a file that no longer exists, the link finding
    /// returned before the render, and `--write` reported the broken link instead
    /// of replacing the document that carried it. Nothing could regenerate the
    /// navigation without hand-editing the generated file first.
    #[test]
    fn stale_navigation_does_not_block_its_own_regeneration() {
        let temporary = TempDir::new().expect("a temporary directory");
        let root = tree(&temporary);
        fs::write(
            root.join(SUMMARY),
            "# Summary\n\n- [Gone](gone.md)\n- [Guide](guide.md)\n",
        )
        .expect("a stale summary");
        fs::write(root.join(INDEX), "# Stale\n").expect("a stale index");

        let checked = DocsCheck
            .run(&GateCtx::new(root.clone(), Vec::new()))
            .expect("the gate runs");
        let reported = messages(&checked.findings);
        assert!(
            reported.contains("names no such path: docs/gone.md"),
            "the dead link is still reported: {reported}"
        );
        assert!(
            reported.contains("disagrees with the documentation manifest"),
            "and so is the drift the same run can see: {reported}"
        );

        let written = DocsCheck
            .run(&GateCtx::new(root.clone(), vec!["--write".to_string()]))
            .expect("the gate runs");
        assert!(
            written.findings.is_empty(),
            "writing the navigation resolves both: {}",
            messages(&written.findings)
        );
        let summary = fs::read_to_string(root.join(SUMMARY)).expect("the written summary");
        assert!(!summary.contains("gone.md"), "{summary}");
        assert!(summary.contains("[Guide](guide.md)"), "{summary}");

        let again = DocsCheck
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate runs");
        assert!(
            again.findings.is_empty(),
            "and the written tree is clean: {}",
            messages(&again.findings)
        );
    }
}
