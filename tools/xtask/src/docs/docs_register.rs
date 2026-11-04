//! The `docs-register` gate: the register every authored Markdown page is
//! written in, what a contributor-facing page may not name, and which pages the
//! documentation manifest has to account for.
//!
//! `docs-check` judges the pages under `docs/`: their lifecycle rows, the
//! generated navigation, and the links between them. This gate judges the
//! sentences, across the whole checkout rather than one directory, and the one
//! part of the surface that sits outside `docs/`: the Markdown at the
//! repository root, which is the first thing a reader opens and the only
//! documentation the manifest could not reach.
//!
//! Three rules:
//!
//! 1. No authored page carries a phrase the register rejects. The phrases are
//!    read from `docs/REGISTER.toml` at run time, so the rule is data and the
//!    gate is the engine.
//! 2. No contributor-facing page names this checkout's build configuration. A
//!    reader who has just cloned the repository can act on none of it, and a
//!    page that teaches an override teaches a reader to break the wrapper. The
//!    names come from `docs/REGISTER.toml` and from the files that set them,
//!    `cargo_full` and `.cargo/config.toml`, so an export added there is
//!    covered without an edit to either.
//! 3. Every Markdown page at the repository root is declared in
//!    `docs/DOCS.toml` as a `[[root_page]]`, and every such row names a page
//!    that exists. `docs-check` already answers this for `docs/`, so this rule
//!    covers the root and nothing else: one defect must not be reported twice.
//!
//! The roster is derived, not listed. It is every Markdown file the checkout
//! publishes, less the pages a generator writes and the pages the manifest
//! records as history. A generator's output is corrected in its generator, and
//! rewriting an archived snapshot to today's register would falsify the record.
//! Inside a page that survives that filter, a section that names the generator
//! writing it is skipped the same way, which is how a crate README carrying one
//! generated section is judged on the prose a person wrote.

use std::collections::{BTreeMap, BTreeSet};

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};
use crate::release::release_docs::GENERATED_RELEASE_DOCUMENTS;

/// The register this gate reads its phrases from.
const REGISTER: &str = "docs/REGISTER.toml";
/// The manifest that declares the published surface.
const MANIFEST: &str = "docs/DOCS.toml";
/// Where a manifest page path is rooted.
const DOCS: &str = "docs";
/// Register schema this gate reads.
const REGISTER_VERSION: i64 = 1;
/// The wrapper that declares the build environment.
const WRAPPER: &str = "cargo_full";
/// The cargo configuration this checkout carries.
const CARGO_CONFIG: &str = ".cargo/config.toml";
/// Shortest environment variable name read as configuration rather than prose.
///
/// `CC` is two characters and occurs inside ordinary sentences. A name that
/// short cannot be told from prose, and a rule that cannot tell is a rule that
/// fires on correct pages until someone deletes it.
const MIN_VARIABLE_LENGTH: usize = 3;
/// Lifecycle state whose pages record what was true rather than what is.
const ARCHIVED: &str = "archived";
/// Generation mode whose pages a generator writes.
const GENERATED: &str = "generated";
/// The audience whose pages are read by someone working on this repository.
const CONTRIBUTOR: &str = "contributor";

/// How a phrase is found in a line.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Match {
    /// Bounded by non-word characters.
    Word,
    /// Anywhere in the line.
    Text,
    /// Opening a line or a sentence.
    SentenceStart,
}

impl Match {
    /// The mode a register row names, or `None` when it names no known mode.
    fn parse(value: &str) -> Option<Self> {
        match value {
            "word" => Some(Self::Word),
            "text" => Some(Self::Text),
            "sentence-start" => Some(Self::SentenceStart),
            _ => None,
        }
    }
}

/// One set of phrases rejected for one reason.
struct Group {
    /// What the reader is being shown, named in the finding.
    label: String,
    /// The corrective action.
    fix: String,
    /// How each phrase is found.
    mode: Match,
    /// The phrases, lowercased.
    phrases: Vec<String>,
}

impl Group {
    /// Every phrase in this group that `lowered` carries.
    ///
    /// `openings` is where a sentence begins on the line, computed once by the
    /// caller because every group asks the same line the same question.
    fn hits(&self, lowered: &str, openings: &[usize]) -> Vec<&str> {
        self.phrases
            .iter()
            .filter(|phrase| match self.mode {
                Match::Word => scan::contains_word(lowered, phrase),
                Match::Text => lowered.contains(phrase.as_str()),
                Match::SentenceStart => openings
                    .iter()
                    .any(|start| opens_with(lowered, *start, phrase)),
            })
            .map(String::as_str)
            .collect()
    }
}

/// The register: what no page may say, and what a contributor page may not name.
struct Register {
    /// Phrase groups every authored page is judged against.
    banned: Vec<Group>,
    /// Phrase groups a contributor-facing page is judged against.
    host_local: Vec<Group>,
}

/// Holds every authored page to the documentation register.
pub struct DocsRegister;

impl crate::gate::GateBehavior for DocsRegister {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let register = load_register(&tree)?;
        let manifest = tree.read_toml(MANIFEST)?;

        let authored = authored_pages(&tree, &manifest);
        let contributor = contributor_pages(&tree, &manifest, &authored);
        let host_local = host_local_names(&tree, &register)?;

        let mut report = Report::clean();
        report.cover_complete("authored pages", authored.len());
        for page in &authored {
            let text = tree.read(page)?;
            let judged = authored_lines(&text, &tree);
            for (index, line) in text.lines().enumerate() {
                if !judged[index] {
                    continue;
                }
                let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
                let lowered = line.to_ascii_lowercase();
                let openings = sentence_openings(&lowered);
                for group in &register.banned {
                    for phrase in group.hits(&lowered, &openings) {
                        report.find(Finding::at(
                            page,
                            number,
                            format!("{}: `{phrase}`", group.label),
                            group.fix.clone(),
                        ));
                    }
                }
                if !contributor.contains(page) {
                    continue;
                }
                for (phrase, fix) in &host_local {
                    if scan::contains_word(&lowered, phrase) {
                        report.find(Finding::at(
                            page,
                            number,
                            format!("host-local build configuration: `{phrase}`"),
                            fix.clone(),
                        ));
                    }
                }
            }
        }
        for finding in root_page_findings(&tree, &manifest) {
            report.find(finding);
        }

        report.note(format!(
            "{} authored page(s), {} of them contributor-facing, {} register phrase(s), {} host-local name(s)",
            authored.len(),
            contributor.len(),
            register
                .banned
                .iter()
                .map(|group| group.phrases.len())
                .sum::<usize>(),
            host_local.len(),
        ));
        Ok(report)
    }
}

/// The register, or a gate that could not run.
///
/// An empty register is a failure rather than a clean report: a rule with no
/// phrase judges nothing, and reads as coverage while judging nothing.
fn load_register(tree: &Tree) -> Result<Register, GateError> {
    let document = tree.read_toml(REGISTER)?;
    if document.get("version").and_then(toml::Value::as_integer) != Some(REGISTER_VERSION) {
        return Err(GateError::new(
            format!("`{REGISTER}` does not declare `version = {REGISTER_VERSION}`"),
            "state the schema version this gate reads",
        ));
    }
    let banned = load_groups(&document, "banned", None)?;
    let host_local = load_groups(&document, "host_local", Some(Match::Word))?;
    if banned.is_empty() || host_local.is_empty() {
        return Err(GateError::new(
            format!("`{REGISTER}` declares no phrase to reject"),
            "declare the register before asking a page to hold to it; a rule with no phrase judges nothing",
        ));
    }
    Ok(Register { banned, host_local })
}

/// Every group under `key`, with `fixed` overriding the row's own match mode.
fn load_groups(
    document: &toml::Table,
    key: &str,
    fixed: Option<Match>,
) -> Result<Vec<Group>, GateError> {
    let rows = document
        .get(key)
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut groups = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let row = row.as_table().ok_or_else(|| {
            GateError::new(
                format!("`{REGISTER}` [[{key}]] entry {} is not a table", index + 1),
                "declare every group as a table with a label, a fix and its phrases",
            )
        })?;
        let text = |field: &str| crate::toml_text::string_field(row, field);
        let label = text("label");
        let fix = text("fix");
        let mode = match fixed {
            Some(mode) => mode,
            None => Match::parse(&text("match")).ok_or_else(|| {
                GateError::new(
                    format!("`{REGISTER}` group `{label}` names no known match mode"),
                    "declare match = \"word\", \"text\" or \"sentence-start\"",
                )
            })?,
        };
        let phrases: Vec<String> = row
            .get("phrases")
            .and_then(toml::Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_ascii_lowercase)
                    .collect()
            })
            .unwrap_or_default();
        if label.is_empty() || fix.is_empty() || phrases.is_empty() {
            return Err(GateError::new(
                format!("`{REGISTER}` [[{key}]] entry {} is incomplete", index + 1),
                "give every group a label, the corrective action, and at least one phrase",
            ));
        }
        groups.push(Group {
            label,
            fix,
            mode,
            phrases,
        });
    }
    Ok(groups)
}

/// Every Markdown page in the checkout whose prose a person wrote.
fn authored_pages(tree: &Tree, manifest: &toml::Table) -> BTreeSet<String> {
    let mut excluded: BTreeSet<String> = GENERATED_RELEASE_DOCUMENTS
        .iter()
        .map(|path| (*path).to_string())
        .collect();
    for row in manifest_pages(manifest) {
        let path = crate::toml_text::string_field(row, "path");
        if path.is_empty() {
            continue;
        }
        if crate::toml_text::string_field(row, "generation") == GENERATED
            || crate::toml_text::string_field(row, "status") == ARCHIVED
        {
            excluded.insert(format!("{DOCS}/{path}"));
        }
    }
    tree.paths()
        .iter()
        .filter(|path| path.extension().is_some_and(|kind| kind == "md"))
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .filter(|path| !excluded.contains(path))
        .collect()
}

/// Every authored page a person working on this repository reads.
///
/// The repository root is contributor-facing whatever a row says, because the
/// root is what a reader opens first and no manifest row reaches it.
fn contributor_pages(
    tree: &Tree,
    manifest: &toml::Table,
    authored: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut pages: BTreeSet<String> = root_markdown(tree);
    for row in manifest_pages(manifest) {
        if crate::toml_text::string_field(row, "audience") == CONTRIBUTOR {
            let path = crate::toml_text::string_field(row, "path");
            if !path.is_empty() {
                pages.insert(format!("{DOCS}/{path}"));
            }
        }
    }
    pages
        .into_iter()
        .filter(|page| authored.contains(page))
        .collect()
}

/// Every Markdown file at the repository root.
fn root_markdown(tree: &Tree) -> BTreeSet<String> {
    tree.paths()
        .iter()
        .filter(|path| path.extension().is_some_and(|kind| kind == "md"))
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .filter(|path| !path.contains('/'))
        .collect()
}

/// The `[[page]]` rows of the manifest.
fn manifest_pages(manifest: &toml::Table) -> impl Iterator<Item = &toml::Table> {
    manifest
        .get("page")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(toml::Value::as_table)
}

/// Findings for the repository-root pages and the rows that declare them.
fn root_page_findings(tree: &Tree, manifest: &toml::Table) -> Vec<Finding> {
    let audiences: BTreeSet<String> = manifest_pages(manifest)
        .map(|row| crate::toml_text::string_field(row, "audience"))
        .filter(|audience| !audience.is_empty())
        .collect();
    let rows = manifest
        .get("root_page")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();

    let mut findings = Vec::new();
    let mut declared: BTreeSet<String> = BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        let Some(row) = row.as_table() else {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("root page entry {} is not a table", index + 1),
                "declare every root page as a [[root_page]] table",
            ));
            continue;
        };
        let text = |field: &str| crate::toml_text::string_field(row, field);
        let path = text("path");
        if path.is_empty() {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("root page entry {} declares no path", index + 1),
                "name the Markdown file at the repository root the row accounts for",
            ));
            continue;
        }
        if !declared.insert(path.clone()) {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("duplicate root page: {path}"),
                "one row per page; two rows give one document two audiences",
            ));
        }
        if text("title").is_empty() {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("root page {path} declares no title"),
                "state what a reader opens the page for",
            ));
        }
        let audience = text("audience");
        if !audiences.contains(&audience) {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("root page {path} declares the unknown audience `{audience}`"),
                "name an audience the published pages already use; a new audience is declared on a page first",
            ));
        }
    }

    let present = root_markdown(tree);
    for path in &present {
        if !declared.contains(path) {
            findings.push(Finding::in_file(
                path,
                "repository-root page the documentation manifest does not declare",
                "declare it in docs/DOCS.toml as a [[root_page]] with its title and audience, or delete it",
            ));
        }
    }
    for path in &declared {
        if !present.contains(path) {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("root page row names a page the root does not hold: {path}"),
                "write the page, or drop its row; a declared page nobody can open is not documentation",
            ));
        }
    }
    findings
}

/// Every name this checkout's build configuration sets, with what to do about
/// a page that names one.
///
/// The register carries the names no file in the checkout spells, and the two
/// files that declare the build carry the rest. Reading them is what keeps the
/// rule current: an export added to the wrapper is covered by the next run.
fn host_local_names(
    tree: &Tree,
    register: &Register,
) -> Result<BTreeMap<String, String>, GateError> {
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    for group in &register.host_local {
        for phrase in &group.phrases {
            names.insert(phrase.clone(), group.fix.clone());
        }
    }

    let derived = "the wrapper at the workspace root declares the build environment once; a page that names what it sets is naming one machine's configuration";
    if tree.has(WRAPPER) {
        for line in tree.read(WRAPPER)?.lines() {
            let Some(rest) = line.trim_start().strip_prefix("export ") else {
                continue;
            };
            let name = rest.split('=').next().unwrap_or_default().trim();
            if name.len() >= MIN_VARIABLE_LENGTH {
                names.insert(name.to_ascii_lowercase(), derived.to_string());
            }
        }
    }
    if tree.has(CARGO_CONFIG) {
        names.insert(CARGO_CONFIG.to_ascii_lowercase(), derived.to_string());
        let document = tree.read_toml(CARGO_CONFIG)?;
        let environment = document
            .get("env")
            .and_then(toml::Value::as_table)
            .map(toml::Table::keys)
            .into_iter()
            .flatten();
        for name in environment {
            if name.len() >= MIN_VARIABLE_LENGTH {
                names.insert(name.to_ascii_lowercase(), derived.to_string());
            }
        }
    }
    Ok(names)
}

/// Which lines of a page a person wrote, by zero-based index.
///
/// A level-2 section that names the generator writing it is the generator's
/// output, not the author's prose, and correcting it here would be corrected
/// away by the next regeneration. The claim has to resolve: the section names a
/// path the checkout holds, so a section cannot excuse itself by citing a
/// generator that does not exist.
fn authored_lines(text: &str, tree: &Tree) -> Vec<bool> {
    let lines: Vec<&str> = text.lines().collect();
    let mut judged = vec![true; lines.len()];
    let mut start = 0;
    while start < lines.len() {
        let mut end = start + 1;
        while end < lines.len() && !lines[end].starts_with("## ") {
            end += 1;
        }
        if lines[start..end]
            .iter()
            .any(|line| names_a_generator(line, tree))
        {
            judged[start..end].fill(false);
        }
        start = end;
    }
    judged
}

/// Whether `line` says a generator in this checkout writes what surrounds it.
fn names_a_generator(line: &str, tree: &Tree) -> bool {
    let lowered = line.to_ascii_lowercase();
    if !lowered.contains("is generated by") && !lowered.contains("is generated from") {
        return false;
    }
    line.split('`')
        .skip(1)
        .step_by(2)
        .flat_map(str::split_whitespace)
        .any(|token| {
            tree.has(token)
                || tree.absolute(token).exists()
                || crate::subcommands::find(token).is_some()
        })
}

/// Byte offsets in `lowered` where a sentence begins.
///
/// A line opens one, after whatever markdown marks it as a heading, a list item
/// or a quote, and so does every position after a terminated sentence.
fn sentence_openings(lowered: &str) -> Vec<usize> {
    let mut openings = Vec::new();
    if let Some((offset, _)) = lowered
        .char_indices()
        .find(|(_, character)| !is_line_marker(*character))
    {
        openings.push(offset);
    }
    let bytes = lowered.as_bytes();
    for (offset, byte) in bytes.iter().enumerate() {
        if !matches!(byte, b'.' | b'!' | b'?') {
            continue;
        }
        let after = lowered[offset + 1..]
            .char_indices()
            .find(|(_, character)| !character.is_whitespace())
            .map(|(index, _)| offset + 1 + index);
        if let Some(after) = after {
            if after > offset + 1 {
                openings.push(after);
            }
        }
    }
    openings
}

/// Whether a character only marks the shape of a line rather than opening it.
fn is_line_marker(character: char) -> bool {
    character.is_whitespace()
        || character.is_ascii_digit()
        || matches!(
            character,
            '#' | '>' | '-' | '*' | '+' | '|' | ')' | '.' | '_'
        )
}

/// Whether the sentence at `start` opens with `phrase` as whole words.
fn opens_with(lowered: &str, start: usize, phrase: &str) -> bool {
    let Some(rest) = lowered.get(start..) else {
        return false;
    };
    let Some(after) = rest.strip_prefix(phrase) else {
        return false;
    };
    after
        .chars()
        .next()
        .is_none_or(|character| !character.is_alphanumeric() && character != '_')
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::TempDir;

    use super::{opens_with, sentence_openings, DocsRegister, Match};
    use crate::gate::{GateBehavior, GateCtx, Report};
    use crate::gates::fixture_checkout::checkout;

    /// A register carrying one group per match mode and one host-local group.
    const REGISTER_FIXTURE: &str = r#"
version = 1

[[banned]]
label = "hype"
match = "word"
fix = "state the measured fact"
phrases = ["blazing", "simply"]

[[banned]]
label = "the documentation as its own subject"
match = "sentence-start"
fix = "open the sentence with the thing the page is about"
phrases = ["this document", "this section"]

[[banned]]
label = "an em dash"
match = "text"
fix = "split the sentence"
phrases = ["\u2014"]

[[host_local]]
label = "compiler cache configuration"
fix = "a compiler cache is local configuration outside the repository"
phrases = ["sccache"]
"#;

    /// A manifest with one page, so the audience vocabulary is derived, and one
    /// row per root page the fixture carries.
    const MANIFEST_FIXTURE: &str = r#"
version = 2

[[owner]]
id = "architecture"
authority = "ARCHITECTURE.md"

[[page]]
path = "ARCHITECTURE.md"
title = "Architecture"
status = "current"
audience = "contributor"
owner = "architecture"
kind = "architecture"
section = "Architecture and ownership"
authority = "self"
generation = "manual"
nav = true

[[page]]
path = "guide.md"
title = "Guide"
status = "current"
audience = "user"
owner = "architecture"
kind = "guide"
section = "User workflows"
authority = "self"
generation = "manual"
nav = true

[[root_page]]
path = "README.md"
title = "Readme"
audience = "user"

[[root_page]]
path = "CONTRIBUTING.md"
title = "Contributing"
audience = "contributor"
"#;

    /// A wrapper that exports one build variable, for the derived half of the
    /// host-local rule.
    const WRAPPER_FIXTURE: &str =
        "#!/usr/bin/env bash\nexport CARGO_TARGET_DIR=\"/elsewhere\"\nexec cargo \"$@\"\n";

    /// The tree every test starts from: clean, and judged by all three rules.
    fn fixture() -> (TempDir, std::path::PathBuf) {
        checkout(&[
            ("docs/REGISTER.toml", REGISTER_FIXTURE),
            ("docs/DOCS.toml", MANIFEST_FIXTURE),
            ("docs/ARCHITECTURE.md", "# Architecture\n\nThe layers.\n"),
            ("docs/guide.md", "# Guide\n\nCompile a graph.\n"),
            ("README.md", "# vyre\n\nA compiler.\n"),
            ("CONTRIBUTING.md", "# Contributing\n\nRun the gates.\n"),
            ("crate/GUIDE.md", "# Guide\n\nCall the builder.\n"),
            ("cargo_full", WRAPPER_FIXTURE),
        ])
    }

    fn run(root: &Path) -> Report {
        DocsRegister
            .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
            .expect("the gate runs against a fixture checkout")
    }

    fn messages(report: &Report) -> String {
        report
            .findings
            .iter()
            .map(|finding| format!("{} {}", finding.named_file(), finding.message))
            .collect::<Vec<_>>()
            .join("; ")
    }

    fn write(root: &Path, path: &str, text: &str) {
        let target = root.join(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).expect("the fixture directory");
        }
        fs::write(target, text).expect("the fixture file");
    }

    /// WHY: every red case below is only evidence if the tree it was injected
    /// into was green, and a gate that reports on a clean fixture cannot tell
    /// an injection from its own noise.
    #[test]
    fn a_clean_checkout_reports_nothing() {
        let (_temporary, root) = fixture();

        let report = run(&root);

        assert!(report.findings.is_empty(), "{}", messages(&report));
    }

    /// WHY: the register is judged across the whole checkout, not one
    /// directory. A page under a crate is prose a reader receives.
    #[test]
    fn a_banned_phrase_in_any_page_is_reported() {
        let (_temporary, root) = fixture();
        write(&root, "crate/GUIDE.md", "# Guide\n\nA blazing fast walk.\n");

        let report = run(&root);

        assert!(
            messages(&report).contains("hype: `blazing`"),
            "{}",
            messages(&report)
        );
        assert_eq!(report.findings[0].line, Some(3));
    }

    /// WHY: a sentence whose subject is the page tells a reader nothing about
    /// the subject, and the mode has to see a sentence rather than a substring.
    #[test]
    fn the_documentation_as_its_own_subject_is_reported() {
        let (_temporary, root) = fixture();
        write(
            &root,
            "crate/GUIDE.md",
            "# Guide\n\nThis section lists the operations.\n\nEverything in this section holds.\n",
        );

        let report = run(&root);
        let reported = messages(&report);

        assert!(reported.contains("its own subject"), "{reported}");
        assert_eq!(
            report.findings.len(),
            1,
            "the phrase mid-sentence is not the subject of the sentence: {reported}"
        );
    }

    /// WHY: an em dash is a character, so the rule that catches it cannot be
    /// the one that needs word boundaries.
    #[test]
    fn an_em_dash_is_reported() {
        let (_temporary, root) = fixture();
        write(
            &root,
            "crate/GUIDE.md",
            "# Guide\n\nOne call\u{2014}one program.\n",
        );

        let report = run(&root);

        assert!(
            messages(&report).contains("an em dash"),
            "{}",
            messages(&report)
        );
    }

    /// WHY: a phrase bounded by word characters must not fire inside a longer
    /// word, or the rule teaches writers to avoid ordinary prose.
    #[test]
    fn a_bounded_phrase_does_not_fire_inside_a_longer_word() {
        let (_temporary, root) = fixture();
        write(
            &root,
            "crate/GUIDE.md",
            "# Guide\n\nSimplyfied is not simply.\n",
        );

        let report = run(&root);

        assert_eq!(report.findings.len(), 1, "{}", messages(&report));
    }

    /// WHY: this is the clause that keeps one machine's build off the page a
    /// new contributor reads. Both halves are proved: the name the register
    /// carries, and the name derived from the wrapper that exports it.
    #[test]
    fn a_contributor_page_naming_host_local_configuration_is_reported() {
        let (_temporary, root) = fixture();
        write(
            &root,
            "CONTRIBUTING.md",
            "# Contributing\n\nInstall sccache.\n\nSet CARGO_TARGET_DIR yourself.\n",
        );

        let report = run(&root);
        let reported = messages(&report);

        assert!(reported.contains("`sccache`"), "{reported}");
        assert!(reported.contains("`cargo_target_dir`"), "{reported}");
    }

    /// WHY: the rule is about who reads the page, not about the words. A page
    /// outside the contributor surface may name what the build sets, and a rule
    /// that fires there would be reporting the tooling's own documentation.
    #[test]
    fn a_page_outside_the_contributor_surface_may_name_the_build() {
        let (_temporary, root) = fixture();
        write(
            &root,
            "crate/GUIDE.md",
            "# Guide\n\nThe wrapper sets CARGO_TARGET_DIR and sccache is optional.\n",
        );

        let report = run(&root);

        assert!(report.findings.is_empty(), "{}", messages(&report));
    }

    /// WHY: a page at the root is the first thing a reader opens and no
    /// `[[page]]` path can reach it, so an undeclared one is documentation with
    /// no owner and no audience.
    #[test]
    fn a_root_page_the_manifest_does_not_declare_is_reported() {
        let (_temporary, root) = fixture();
        write(&root, "NOTES.md", "# Notes\n\nSomething.\n");

        let report = run(&root);
        let reported = messages(&report);

        assert!(
            reported.contains("repository-root page the documentation manifest does not declare"),
            "{reported}"
        );
        assert!(reported.contains("NOTES.md"), "{reported}");
    }

    /// WHY: the roster is derived from the tree, so a row that outlived its
    /// page has to fail; otherwise the manifest records a surface nobody has.
    #[test]
    fn a_root_page_row_naming_no_page_is_reported() {
        let (_temporary, root) = fixture();
        let manifest = format!(
            "{MANIFEST_FIXTURE}\n[[root_page]]\npath = \"GONE.md\"\ntitle = \"Gone\"\naudience = \"user\"\n"
        );
        write(&root, "docs/DOCS.toml", &manifest);

        let report = run(&root);

        assert!(
            messages(&report)
                .contains("root page row names a page the root does not hold: GONE.md"),
            "{}",
            messages(&report)
        );
    }

    /// WHY: a root row that names an audience no page uses invents a reader.
    #[test]
    fn a_root_page_row_with_an_unknown_audience_is_reported() {
        let (_temporary, root) = fixture();
        write(&root, "NOTES.md", "# Notes\n\nSomething.\n");
        let manifest = format!(
            "{MANIFEST_FIXTURE}\n[[root_page]]\npath = \"NOTES.md\"\ntitle = \"Notes\"\naudience = \"nobody\"\n"
        );
        write(&root, "docs/DOCS.toml", &manifest);

        let report = run(&root);

        assert!(
            messages(&report).contains("unknown audience `nobody`"),
            "{}",
            messages(&report)
        );
    }

    /// WHY: correcting a generated page is corrected away by the next
    /// regeneration, so the roster has to exclude it. The manifest is the only
    /// place that says which page a generator writes.
    #[test]
    fn a_generated_page_is_not_judged() {
        let (_temporary, root) = fixture();
        let manifest = format!(
            "{MANIFEST_FIXTURE}\n[[page]]\npath = \"generated.md\"\ntitle = \"Generated\"\nstatus = \"generated\"\naudience = \"contributor\"\nowner = \"architecture\"\nkind = \"reference\"\nsection = \"Architecture and ownership\"\nauthority = \"ARCHITECTURE.md\"\ngeneration = \"generated\"\ngenerator = \"../cargo_full\"\nnav = true\n"
        );
        write(&root, "docs/DOCS.toml", &manifest);
        write(
            &root,
            "docs/generated.md",
            "# Generated\n\nBlazing output.\n",
        );

        let report = run(&root);

        assert!(report.findings.is_empty(), "{}", messages(&report));
    }

    /// WHY: a crate README carries one generated section and authored prose
    /// around it. The section is skipped only because it names a generator the
    /// checkout holds, so a page cannot excuse itself by inventing one.
    #[test]
    fn a_generated_section_is_skipped_and_an_invented_generator_is_not() {
        let (_temporary, root) = fixture();
        write(
            &root,
            "crate/GUIDE.md",
            "# Guide\n\n## Commands\n\nThis section is generated from `cargo_full`.\n\nBlazing output.\n",
        );

        let honoured = run(&root);
        assert!(honoured.findings.is_empty(), "{}", messages(&honoured));

        write(
            &root,
            "crate/GUIDE.md",
            "# Guide\n\n## Commands\n\nThis section is generated from `no/such/generator.py`.\n\nBlazing output.\n",
        );

        let invented = run(&root);
        assert!(
            messages(&invented).contains("hype: `blazing`"),
            "{}",
            messages(&invented)
        );
    }

    /// WHY: a register with no phrase judges nothing while reading as coverage.
    /// That has to be a gate that could not run, never a clean report.
    #[test]
    fn an_empty_register_is_a_gate_that_could_not_run() {
        let (_temporary, root) = fixture();
        write(&root, "docs/REGISTER.toml", "version = 1\n");

        let error = DocsRegister
            .run(&GateCtx::new(root.clone(), Vec::new()))
            .expect_err("an empty register cannot judge a page");

        assert!(
            error.to_string().contains("declares no phrase to reject"),
            "{error}"
        );
    }

    /// WHY: the match mode decides what the phrase means, so a row that names
    /// no known mode is a rule nobody can read, not a rule that matches nothing.
    #[test]
    fn an_unknown_match_mode_is_a_gate_that_could_not_run() {
        let (_temporary, root) = fixture();
        write(
            &root,
            "docs/REGISTER.toml",
            "version = 1\n\n[[banned]]\nlabel = \"hype\"\nmatch = \"regex\"\nfix = \"state the fact\"\nphrases = [\"blazing\"]\n\n[[host_local]]\nlabel = \"cache\"\nfix = \"drop it\"\nphrases = [\"sccache\"]\n",
        );

        let error = DocsRegister
            .run(&GateCtx::new(root.clone(), Vec::new()))
            .expect_err("an unreadable match mode cannot judge a page");

        assert!(error.to_string().contains("no known match mode"), "{error}");
    }

    /// WHY: a sentence opening is what separates the subject rule from a
    /// substring search, and markdown marks the opening in several ways.
    #[test]
    fn a_sentence_opens_after_markdown_marks_and_after_a_full_stop() {
        assert!(opens_with(
            "- this document states",
            sentence_openings("- this document states")[0],
            "this document"
        ));
        let quoted = "> **this page** holds";
        assert!(sentence_openings(quoted).iter().any(|start| opens_with(
            quoted,
            *start,
            "this page"
        )));
        let second = "the walk is bounded. this section lists it";
        assert!(sentence_openings(second).iter().any(|start| opens_with(
            second,
            *start,
            "this section"
        )));
        let middle = "everything in this section holds";
        assert!(!sentence_openings(middle).iter().any(|start| opens_with(
            middle,
            *start,
            "this section"
        )));
    }

    /// WHY: the modes are the register's vocabulary, and one the gate cannot
    /// parse silently disables a group.
    #[test]
    fn every_declared_match_mode_parses_and_nothing_else_does() {
        for mode in ["word", "text", "sentence-start"] {
            assert!(Match::parse(mode).is_some(), "{mode}");
        }
        assert!(Match::parse("substring").is_none());
    }
}
