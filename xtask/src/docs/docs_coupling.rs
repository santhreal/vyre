//! The `docs-coupling` gate: whether an authored page still points at code that
//! exists, and whether a change to that code arrived with the page.
//!
//! `docs-check` judges the manifest, the navigation and the links between pages.
//! This gate judges the other direction: the subject. An authored page declares
//! `covers`, the source paths it states the content of, and both sides of the
//! coupling are derived from that declaration at run time rather than from a
//! table here.
//!
//! Four rules, in the order they fail:
//!
//! 1. An authored page that is a current claim declares a non-empty `covers`.
//!    Without it nothing can hold the page to the code, and the page becomes
//!    prose nobody has to maintain.
//! 2. Every `covers` entry matches a path that exists. A page that claims to
//!    describe a deleted module describes nothing.
//! 3. Every repository path an authored page cites in code resolves. A document
//!    that says a thing lives in a file that does not exist is worse than
//!    silence, because a reader stops looking rather than looking elsewhere.
//! 4. A change to a covered path arrives with its page, and with a changelog
//!    fragment. This is the only rule that reads a diff; with no diff to read it
//!    contributes nothing, so a clean tree is clean.
//!
//! The diff comes from `--base REF` or `GITHUB_BASE_REF`, comparing the merge
//! base with `HEAD`. Without either, it is the worktree plus the index, which is
//! what a local caller has before committing. A push event has neither a base
//! ref nor an unclean tree, so it reads an empty diff and rules 1 to 3 answer on
//! their own.
//!
//! A base ref this checkout does not hold, which is what a shallow clone gives a
//! pull request, is one finding naming the ref and the fetch that fixes it. It is
//! not a gate that could not run: that reports a whole workflow red with a
//! message about git, and it is not a silent pass either, because rule 4 would
//! then be unenforceable in exactly the environment it exists for.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::output_arg::read_text_bounded;

/// The manifest that declares every page and what it covers.
const MANIFEST: &str = "docs/DOCS.toml";
/// Where a page path is rooted.
const DOCS: &str = "docs";
/// Where a changelog fragment lives.
const FRAGMENTS: &str = "release/changes/unreleased";
/// Bound on any one document this gate reads.
const MAX_DOCUMENT_BYTES: u64 = 8_388_608;
/// Lifecycle states whose pages are still current claims.
const ACTIVE_STATUSES: [&str; 2] = ["current", "generated"];
/// Directories that hold snapshots rather than current claims.
const HISTORICAL_DIRECTORIES: [&str; 2] = ["archive/", "legacy/"];
/// Characters that mark a cited path as a template rather than a path.
const TEMPLATE_MARKERS: [char; 7] = ['<', '>', '{', '}', '|', '$', '*'];
/// URL schemes a citation may open with, which are not repository paths.
const SCHEMES: [&str; 6] = ["http:", "https:", "mailto:", "git@", "docs.rs", "www."];

/// One authored page and the code it answers for.
struct Covering {
    /// Path under `docs/`.
    page: String,
    /// Source paths the page states the content of, as written.
    covers: Vec<String>,
}

/// Holds an authored page to the code it names, and to the diff that changes it.
pub struct DocsCoupling;

impl Gate for DocsCoupling {
    fn name(&self) -> &'static str {
        "docs-coupling"
    }

    fn help(&self) -> &'static str {
        "Whether an authored page still names code that exists, and whether a change to that code arrived with the page; --base REF compares against that ref"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let pages = load_covering(&ctx.root, &mut report)?;
        if pages.is_empty() {
            return Ok(report);
        }
        let tree = tracked_paths(&ctx.root)?;

        for page in &pages {
            report
                .findings
                .extend(covers_findings(page, &tree, &ctx.root));
        }
        let extensions = extension_vocabulary(&tree);
        let roots = top_level_directories(&tree);
        for page in &pages {
            report.findings.extend(citation_findings(
                &ctx.root,
                &page.page,
                &tree,
                &extensions,
                &roots,
            )?);
        }

        let declared = format!(
            "{} authored page(s), {} covers entr(ies)",
            pages.len(),
            pages.iter().map(|page| page.covers.len()).sum::<usize>()
        );
        match changed_paths(&ctx.root, ctx.flag("--base"))? {
            Diff::Read(changed) => {
                report.note(format!("{declared}, {} changed path(s)", changed.len()));
                report
                    .findings
                    .extend(coupling_findings(&pages, &changed, &ctx.root));
            }
            Diff::UnreachableBase(reference) => {
                report.note(declared);
                report.find(Finding::new(
                    format!("`{reference}` is not in this checkout, so no diff can be compared against it"),
                    "fetch the base ref before running this gate: `actions/checkout` with `fetch-depth: 0`, or pass `--base REF` naming a ref this checkout holds",
                ));
            }
        }
        Ok(report)
    }
}

/// Every authored page that is a current claim, with what it declares it covers.
///
/// A generated page is skipped: its generator gate already fails when the
/// artifact and the tree disagree, and requiring a second declaration would make
/// two owners of one rule.
fn load_covering(root: &Path, report: &mut Report) -> Result<Vec<Covering>, GateError> {
    let text = read_document(root, MANIFEST)?;
    let document: toml::Table = toml::from_str(&text).map_err(|error| {
        GateError::new(
            format!("`{MANIFEST}` does not parse as TOML: {error}"),
            "fix the syntax the parser names; the documentation surface is declared here",
        )
    })?;
    let rows = document
        .get("page")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if rows.is_empty() {
        return Err(GateError::new(
            format!("`{MANIFEST}` declares no page"),
            "declare the documentation surface before asking whether it is coupled",
        ));
    }

    let mut pages = Vec::new();
    for row in rows {
        let Some(row) = row.as_table() else {
            continue;
        };
        let field = |key: &str| crate::toml_text::string_field(row, key);
        let page = field("path");
        let status = field("status");
        let generation = field("generation");
        if page.is_empty()
            || !ACTIVE_STATUSES.contains(&status.as_str())
            || generation != "manual"
            || HISTORICAL_DIRECTORIES
                .iter()
                .any(|directory| page.starts_with(directory))
        {
            continue;
        }
        let covers: Vec<String> = row
            .get("covers")
            .and_then(toml::Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if covers.is_empty() {
            report.find(Finding::in_file(
                MANIFEST,
                format!("`{page}` is an authored current page and declares no `covers`"),
                "add `covers = [...]` naming the source paths the page states the content of",
            ));
            continue;
        }
        pages.push(Covering { page, covers });
    }
    Ok(pages)
}

/// Findings for a `covers` entry that matches nothing in the tree.
fn covers_findings(page: &Covering, tree: &BTreeSet<String>, root: &Path) -> Vec<Finding> {
    page.covers
        .iter()
        .filter(|pattern| !matches_any(pattern, tree))
        .map(|pattern| {
            Finding::in_file(
                Path::new(DOCS).join(&page.page),
                format!("`covers` entry `{pattern}` matches no path in the tree"),
                "name a path that exists, or delete the entry with the claim that needed it",
            )
            .relative_to(root)
        })
        .collect()
}

/// Findings for a repository path the page cites in code and the tree does not
/// hold.
///
/// Only code is read. Prose names a module the way a reader says it out loud,
/// and markdown link targets are resolved by `docs-check` against the page they
/// sit on, so reading either here would report the same defect twice under two
/// resolutions.
fn citation_findings(
    root: &Path,
    page: &str,
    tree: &BTreeSet<String>,
    extensions: &BTreeSet<String>,
    roots: &BTreeSet<String>,
) -> Result<Vec<Finding>, GateError> {
    let relative = format!("{DOCS}/{page}");
    let text = read_document(root, &relative)?;
    let mut findings = Vec::new();
    let mut seen = BTreeSet::new();
    for (line, candidate) in code_citations(&text) {
        if !looks_like_repository_path(&candidate, extensions, roots) {
            continue;
        }
        if tree.contains(&candidate) || root.join(&candidate).exists() {
            continue;
        }
        if !seen.insert(candidate.clone()) {
            continue;
        }
        findings.push(Finding::at(
            &relative,
            line,
            format!("cites `{candidate}`, which does not exist"),
            "name the path that holds the thing today, or delete the claim",
        ));
    }
    Ok(findings)
}

/// Findings for a covered path that changed without its page or a fragment.
fn coupling_findings(
    pages: &[Covering],
    changed: &BTreeSet<String>,
    root: &Path,
) -> Vec<Finding> {
    if changed.is_empty() {
        return Vec::new();
    }
    let mut findings = Vec::new();
    let mut any_covered = false;
    for page in pages {
        let page_path = format!("{DOCS}/{}", page.page);
        let page_changed = changed.contains(&page_path);
        let mut hits: BTreeMap<&str, Vec<&String>> = BTreeMap::new();
        for pattern in &page.covers {
            let matched: Vec<&String> = changed
                .iter()
                .filter(|path| glob_matches(pattern, path))
                .collect();
            if !matched.is_empty() {
                hits.insert(pattern.as_str(), matched);
            }
        }
        if hits.is_empty() {
            continue;
        }
        any_covered = true;
        if page_changed {
            continue;
        }
        let named = hits
            .values()
            .flatten()
            .map(|path| path.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        findings.push(
            Finding::in_file(
                &page_path,
                format!("{named} changed and the page that covers it did not"),
                "state the new behaviour on the page, or narrow its `covers` to what it still answers for",
            )
            .relative_to(root),
        );
    }
    if any_covered
        && !changed
            .iter()
            .any(|path| path.starts_with(FRAGMENTS) && path.ends_with(".toml"))
    {
        findings.push(Finding::new(
            "a covered source path changed and no changelog fragment did",
            format!("add one fragment under `{FRAGMENTS}/` carrying `category` and `text`"),
        ));
    }
    findings
}

/// Every path the repository tracks, relative to the checkout root.
fn tracked_paths(root: &Path) -> Result<BTreeSet<String>, GateError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("could not run `git ls-files`: {error}"),
                "run this gate inside a git work tree; the tracked set is the oracle",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            "`git ls-files` failed".to_string(),
            "run this gate inside a git work tree; the tracked set is the oracle",
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut paths: BTreeSet<String> = text
        .split('\0')
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
        .collect();
    if paths.is_empty() {
        return Err(GateError::new(
            "the repository tracks no file".to_string(),
            "run this gate inside a populated checkout",
        ));
    }
    let directories: Vec<String> = paths
        .iter()
        .flat_map(|path| {
            let mut prefixes = Vec::new();
            let mut at = 0;
            while let Some(offset) = path[at..].find('/') {
                at += offset;
                prefixes.push(path[..at].to_string());
                at += 1;
            }
            prefixes
        })
        .collect();
    paths.extend(directories);
    Ok(paths)
}

/// Extensions that occur among the tracked files, lowercased and without the dot.
fn extension_vocabulary(tree: &BTreeSet<String>) -> BTreeSet<String> {
    tree.iter()
        .filter_map(|path| Path::new(path).extension()?.to_str())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// First path component of every tracked path, which is what a repository-rooted
/// citation opens with.
fn top_level_directories(tree: &BTreeSet<String>) -> BTreeSet<String> {
    tree.iter()
        .filter_map(|path| path.split('/').next())
        .filter(|component| !component.is_empty())
        .map(str::to_string)
        .collect()
}

/// Every token inside an inline code span or a fenced block, with its line.
fn code_citations(text: &str) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    let mut fenced = false;
    for (index, line) in text.lines().enumerate() {
        let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            for token in line.split(|character: char| character.is_whitespace()) {
                out.push((number, token.to_string()));
            }
            continue;
        }
        for span in inline_spans(line) {
            for token in span.split(|character: char| character.is_whitespace()) {
                out.push((number, token.to_string()));
            }
        }
    }
    out
}

/// The contents of every backtick-delimited span on one line.
fn inline_spans(line: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('`') else {
            break;
        };
        spans.push(&rest[..close]);
        rest = &rest[close + 1..];
    }
    spans
}

/// Whether a token is a citation of a path inside this repository.
fn looks_like_repository_path(
    token: &str,
    extensions: &BTreeSet<String>,
    roots: &BTreeSet<String>,
) -> bool {
    let token = token.trim_matches(|character: char| {
        matches!(
            character,
            '`' | '"' | '\'' | '(' | ')' | '[' | ']' | ',' | ';' | ':' | '.'
        )
    });
    if token.len() < 3 || !token.contains('/') || token.starts_with('/') {
        return false;
    }
    if token.contains(TEMPLATE_MARKERS) || token.contains("..") {
        return false;
    }
    if SCHEMES.iter().any(|scheme| token.starts_with(scheme)) {
        return false;
    }
    let Some(first) = token.split('/').next() else {
        return false;
    };
    if !roots.contains(first) {
        return false;
    }
    let named_extension = Path::new(token)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match named_extension {
        Some(extension) => extensions.contains(&extension),
        None => true,
    }
}

/// What the diff comparison produced.
enum Diff {
    /// The comparison ran; these are the paths it touched.
    Read(BTreeSet<String>),
    /// The base ref this checkout was told to compare against is not present, so
    /// rule 4 has nothing to read. A shallow checkout is the usual cause and it
    /// is an environment fact, not a defect in the tree, so it is reported as one
    /// actionable finding rather than as a gate that could not run: a crashed
    /// gate takes a whole workflow red with a message about git.
    UnreachableBase(String),
}

/// Every path the diff touches, relative to the checkout root.
///
/// With a base ref the comparison is the merge base with `HEAD`, which is the
/// set a pull request proposes. Without one it is the index plus the worktree,
/// which is the set a local caller is about to commit.
fn changed_paths(root: &Path, base: Option<&str>) -> Result<Diff, GateError> {
    let base = base
        .map(str::to_string)
        .or_else(|| std::env::var("GITHUB_BASE_REF").ok())
        .filter(|reference| !reference.is_empty());
    let Some(reference) = base else {
        let mut paths = diff_names(root, &["diff", "--name-only", "HEAD"])?;
        paths.extend(diff_names(root, &["diff", "--name-only", "--cached", "HEAD"])?);
        return Ok(Diff::Read(paths));
    };
    let remote = format!("origin/{reference}");
    if !ref_exists(root, &remote) {
        return Ok(Diff::UnreachableBase(remote));
    }
    let range = format!("{remote}...HEAD");
    Ok(Diff::Read(diff_names(
        root,
        &["diff", "--name-only", &range],
    )?))
}

/// Whether this checkout holds the named ref.
///
/// `rev-parse --verify` is asked rather than inferred from a failing `diff`,
/// because a diff can fail for reasons a fetch does not fix and the two
/// answers need different fix lines.
fn ref_exists(root: &Path, reference: &str) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--verify", "--quiet"])
        .arg(format!("{reference}^{{commit}}"))
        .output()
        .is_ok_and(|output| output.status.success())
}

/// One `git diff` invocation, as a set of paths.
fn diff_names(root: &Path, arguments: &[&str]) -> Result<BTreeSet<String>, GateError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("could not run `git {}`: {error}", arguments.join(" ")),
                "run this gate inside a git work tree, or pass `--base REF` naming a fetched ref",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "`git {}` failed: {}",
                arguments.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "fetch the base ref before comparing against it",
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect())
}

/// Whether a `covers` pattern matches any path in the set.
fn matches_any(pattern: &str, paths: &BTreeSet<String>) -> bool {
    paths.iter().any(|path| glob_matches(pattern, path))
}

/// Whether `pattern` matches `path`, with `**` crossing separators and `*`
/// matching within one component.
fn glob_matches(pattern: &str, path: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix("/**") {
        if path == prefix || path.starts_with(&format!("{prefix}/")) {
            return true;
        }
    }
    segment_matches(
        &pattern.split('/').collect::<Vec<_>>(),
        &path.split('/').collect::<Vec<_>>(),
    )
}

/// Component-wise glob match, where `**` consumes any number of components.
fn segment_matches(pattern: &[&str], path: &[&str]) -> bool {
    match (pattern.first(), path.first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(&"**"), _) => {
            if segment_matches(&pattern[1..], path) {
                return true;
            }
            !path.is_empty() && segment_matches(pattern, &path[1..])
        }
        (Some(_), None) => false,
        (Some(head), Some(component)) => {
            component_matches(head, component) && segment_matches(&pattern[1..], &path[1..])
        }
    }
}

/// Whether one pattern component matches one path component, `*` spanning any
/// run of characters inside it.
fn component_matches(pattern: &str, component: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else {
        return pattern == component;
    };
    if !component.starts_with(first) {
        return false;
    }
    let mut at = first.len();
    let parts: Vec<&str> = parts.collect();
    if parts.is_empty() {
        return at == component.len();
    }
    let (last, middle) = parts.split_last().expect("Fix: parts is not empty.");
    for part in middle {
        match component[at..].find(part) {
            Some(offset) => at += offset + part.len(),
            None => return false,
        }
    }
    component.len() >= at + last.len() && component[at..].ends_with(last)
}

/// Read one document under the checkout root, under the read bound.
fn read_document(root: &Path, relative: &str) -> Result<String, GateError> {
    read_text_bounded(&root.join(relative), MAX_DOCUMENT_BYTES, "documentation").map_err(|error| {
        GateError::new(
            format!("could not read `{relative}`: {error}"),
            "restore the document; the manifest declares it as a current page",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the glob is the whole coupling. A `covers` entry that matches
    /// nothing silently exempts a page, and one that matches too much drags an
    /// unrelated page into every diff. Both directions are asserted here rather
    /// than through a fixture tree, because the pattern language is the contract.
    #[test]
    fn double_star_crosses_separators_and_single_star_does_not() {
        assert!(glob_matches("vyre-libs/src/parsing/**", "vyre-libs/src/parsing/go/lex.rs"));
        assert!(glob_matches("vyre-libs/src/parsing/**", "vyre-libs/src/parsing"));
        assert!(!glob_matches("vyre-libs/src/parsing/**", "vyre-libs/src/graph/mod.rs"));
        assert!(glob_matches("vyre-megakernel/src/*.rs", "vyre-megakernel/src/cost.rs"));
        assert!(!glob_matches(
            "vyre-megakernel/src/*.rs",
            "vyre-megakernel/src/target/mod.rs"
        ));
        assert!(glob_matches("Cargo.toml", "Cargo.toml"));
        assert!(!glob_matches("Cargo.toml", "vyre/Cargo.toml"));
    }

    /// WHY: rule 3 reads tokens out of code, so what counts as a path decides
    /// whether it reports real broken pointers or noise. A template placeholder,
    /// a URL, a bare word with a slash in it and a relative escape are each a
    /// false positive that would get the rule muted.
    #[test]
    fn a_citation_is_a_rooted_path_with_a_known_extension() {
        let extensions: BTreeSet<String> = ["rs", "toml", "md"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let roots: BTreeSet<String> = ["docs", "vyre-libs", "release"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let cited = |token: &str| looks_like_repository_path(token, &extensions, &roots);

        assert!(cited("docs/ARCHITECTURE.md"));
        assert!(cited("vyre-libs/src/parsing"));
        assert!(cited("release/changes/unreleased/"));
        assert!(!cited("vyre-libs/src/<domain>/<primitive>.rs"));
        assert!(!cited("https://docs.rs/vyre"));
        assert!(!cited("../lego-block-rule.md"));
        assert!(!cited("producer/consumer"));
        assert!(!cited("/etc/passwd"));
        assert!(!cited("docs/ARCHITECTURE.rst"));
    }

    /// WHY: only code is read, so a fenced block, an inline span and prose must
    /// be told apart. Reading prose reported every slash a sentence contained;
    /// reading nothing made the rule unreachable.
    #[test]
    fn citations_come_from_fences_and_inline_spans_only() {
        let text = "prose naming docs/nowhere.md\n\n```sh\nread docs/inside.md\n```\n\nspan `docs/span.md` here\n";
        let found: Vec<String> = code_citations(text)
            .into_iter()
            .map(|(_, token)| token)
            .collect();
        assert!(found.iter().any(|token| token == "docs/inside.md"));
        assert!(found.iter().any(|token| token == "docs/span.md"));
        assert!(
            !found.iter().any(|token| token == "docs/nowhere.md"),
            "Fix: prose is not a citation, got {found:?}"
        );
    }

    /// WHY: rule 4 is the coupling itself. A covered path that changed without
    /// its page must be one finding, the same change with the page must be
    /// clean, and an empty diff must contribute nothing, which is what makes a
    /// push event green.
    #[test]
    fn a_covered_change_needs_its_page_and_a_fragment() {
        let pages = vec![Covering {
            page: "reference/wire-format.md".to_string(),
            covers: vec!["vyre-foundation/src/serial/wire/**".to_string()],
        }];
        let root = Path::new("/nonexistent");

        let changed: BTreeSet<String> = ["vyre-foundation/src/serial/wire/framing/mod.rs"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let findings = coupling_findings(&pages, &changed, root);
        assert_eq!(findings.len(), 2, "Fix: expected the page and the fragment, got {findings:?}");

        let with_both: BTreeSet<String> = [
            "vyre-foundation/src/serial/wire/framing/mod.rs",
            "docs/reference/wire-format.md",
            "release/changes/unreleased/a-thing.toml",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        assert!(coupling_findings(&pages, &with_both, root).is_empty());

        assert!(coupling_findings(&pages, &BTreeSet::new(), root).is_empty());
    }

    /// WHY: rule 4 needs history, and a shallow checkout is the environment a
    /// pull request actually runs in. An unreachable base must read as one
    /// actionable finding: a `GateError` here takes the whole workflow red over
    /// a fetch depth, and an empty diff would make the rule silently
    /// unenforceable in the only place it matters.
    #[test]
    fn an_unreachable_base_ref_is_a_finding_and_not_a_crash() {
        let root = std::env::current_dir().expect("Fix: the test runs inside the checkout.");
        let root = root.ancestors().find(|path| path.join(".git").exists()).unwrap_or(&root);
        match changed_paths(root, Some("no-such-base-ref-cbb0a1")) {
            Ok(Diff::UnreachableBase(reference)) => {
                assert_eq!(reference, "origin/no-such-base-ref-cbb0a1");
            }
            other => panic!("Fix: an unreachable base must report itself, got {:?}", other.is_ok()),
        }
        assert!(matches!(changed_paths(root, None), Ok(Diff::Read(_))));
    }

    /// WHY: an uncovered change must not demand a fragment. Editing a test or a
    /// workflow is not an observable product change, and reporting one there
    /// trains a reader to add an empty fragment to pass the gate.
    #[test]
    fn an_uncovered_change_demands_nothing() {
        let pages = vec![Covering {
            page: "reference/wire-format.md".to_string(),
            covers: vec!["vyre-foundation/src/serial/wire/**".to_string()],
        }];
        let changed: BTreeSet<String> = ["xtask/src/main.rs"].into_iter().map(str::to_string).collect();
        assert!(coupling_findings(&pages, &changed, Path::new("/nonexistent")).is_empty());
    }
}
