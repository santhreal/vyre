//! Every path a published document names resolves to a published path.
//!
//! A code span and a command example are claims: a reader copies them. A claim
//! naming a file the checkout does not carry, or one an ignore rule excludes,
//! resolves to nothing, and nothing said so while this rule lived in a Python
//! script that no workflow invoked. `.github/CI_REQUIRED.md` required two
//! workflows that were never files.
//!
//! The document set is derived from the tree at run time: the root README, every
//! page under `docs/`, every workspace member README, and every Markdown file at
//! the repository root or under `.github/`. A new root document joins on the
//! commit that adds it. Pages the docs manifest marks archived or superseded are
//! out, and so is anything under `docs/archive/` or `docs/legacy/`, because a
//! superseded page is a record rather than a claim.
//!
//! `CHANGELOG.md` and `docs/release/` are exempt as a kind, not as a list. A
//! changelog entry naming a file a later commit deleted is correct history, and
//! rewriting it to satisfy a reference check would falsify the record.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Documents that record what happened rather than what the tree holds.
const HISTORICAL_DOCUMENTS: &[&str] = &["CHANGELOG.md", "docs/release/"];

/// Path prefixes that make a token a workspace-root-relative claim.
const ROOT_PREFIXES: &[&str] = &[
    ".github/",
    "Cargo.toml",
    "CHANGELOG.md",
    "README.md",
    "consumer/",
    "docs/",
    "libs/",
    "release/",
    "scripts/",
    "tools/",
];

/// Path prefixes a crate README resolves against its own directory.
const CRATE_RELATIVE_PREFIXES: &[&str] = &[
    "api/",
    "benches/",
    "examples/",
    "hardware/",
    "pipeline/",
    "rules/",
    "src/",
    "tests/",
];

/// Suffixes that make a slash-bearing token a path rather than prose.
const PATH_SUFFIXES: &[&str] = &[
    ".c", ".h", ".json", ".md", ".py", ".rs", ".sh", ".toml", ".txt", ".vir", ".wgsl",
];

/// Flags whose next argument names an output, which need not pre-exist.
const OUTPUT_FLAGS: &[&str] = &[
    "--emit",
    "--out",
    "--out-dir",
    "--output",
    "--output-dir",
    "--write",
    "-o",
];

/// Fence languages whose body is a command example.
const COMMAND_LANGUAGES: &[&str] = &["console", "bash", "sh", "shell"];

/// The one document whose whole body is a generated table of other documents.
const GENERATED_INDEX: &str = "docs/INDEX.md";

/// What an unresolved reference costs, and how to close it.
const FIX: &str = "publish the referenced input, correct the path, or delete the claim; an output destination belongs behind --output or --write, which this gate does not follow";

/// Where one reference was written and what it said.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Reference {
    document: String,
    line: u32,
    raw: String,
    resolved: String,
    source: &'static str,
}

/// Every path-like code span and command input resolves to a published path.
pub struct DocsReferences;

impl Gate for DocsReferences {
    fn name(&self) -> &'static str {
        "docs-references"
    }

    fn help(&self) -> &'static str {
        "Hold every path a published document names in a code span or a command example to a published path"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let documents = documents(&tree)?;
        let mut references = BTreeSet::new();
        for document in &documents {
            collect(&tree, document, &mut references)?;
        }
        let mut report = Report::clean();
        for reference in &references {
            let Some(problem) = judge(&tree, reference) else {
                continue;
            };
            report.find(Finding::at(
                reference.document.clone(),
                reference.line,
                problem,
                FIX,
            ));
        }
        report.note(format!(
            "{} path-like code span(s) and command input(s) across {} document(s)",
            references.len(),
            documents.len()
        ));
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        Ok(report)
    }
}

/// What is wrong with one reference, or `None` when it resolves.
fn judge(tree: &Tree, reference: &Reference) -> Option<String> {
    if reference.resolved == OUTSIDE {
        return Some(format!(
            "{} `{}` names a path outside this repository",
            reference.source, reference.raw
        ));
    }
    if reference.resolved.contains('*') || reference.resolved.contains('?') {
        return if glob_matches(tree, &reference.resolved) {
            None
        } else {
            Some(format!(
                "{} `{}` expands to `{}`, which matches nothing published",
                reference.source, reference.raw, reference.resolved
            ))
        };
    }
    match resolution(tree, &reference.resolved) {
        Resolution::Listed => None,
        Resolution::Excluded => Some(format!(
            "{} `{}` resolves to `{}`, which an ignore rule excludes from the checkout",
            reference.source, reference.raw, reference.resolved
        )),
        Resolution::Missing => Some(format!(
            "{} `{}` resolves to `{}`, which the checkout does not carry",
            reference.source, reference.raw, reference.resolved
        )),
    }
}

/// The sentinel a reference that leaves the repository resolves to.
const OUTSIDE: &str = "\0outside";

/// What the tree holds at one repository-relative path.
enum Resolution {
    /// The checkout publishes it.
    Listed,
    /// Something is there, but the checkout does not publish it.
    Excluded,
    /// Nothing is there.
    Missing,
}

fn resolution(tree: &Tree, relative: &str) -> Resolution {
    if tree.has(relative) {
        return Resolution::Listed;
    }
    let absolute = tree.absolute(relative);
    if absolute.is_dir() {
        let prefix = format!("{}/", relative.trim_end_matches('/'));
        if tree
            .paths()
            .iter()
            .any(|path| path.to_string_lossy().starts_with(&prefix))
        {
            return Resolution::Listed;
        }
        return Resolution::Excluded;
    }
    if absolute.exists() {
        return Resolution::Excluded;
    }
    Resolution::Missing
}

/// Whether any published path matches a glob.
fn glob_matches(tree: &Tree, pattern: &str) -> bool {
    tree.paths()
        .iter()
        .any(|path| crate::gates::scan::glob_match(pattern, path.to_string_lossy().as_ref()))
}

/// Every document whose claims this gate reads.
fn documents(tree: &Tree) -> Result<Vec<PathBuf>, GateError> {
    let inactive = inactive_pages(tree)?;
    let mut candidates: BTreeSet<PathBuf> = BTreeSet::new();
    if tree.has("README.md") {
        candidates.insert(PathBuf::from("README.md"));
    }
    for member in tree.members()? {
        let readme = format!("{member}/README.md");
        if tree.has(&readme) {
            candidates.insert(PathBuf::from(readme));
        }
    }
    for path in tree.paths() {
        if path.extension().and_then(|value| value.to_str()) != Some("md") {
            continue;
        }
        let text = path.to_string_lossy().to_string();
        let is_root_document = !text.contains('/');
        let is_github = text.starts_with(".github/");
        let is_docs = text.starts_with("docs/");
        if is_root_document || is_github || is_docs {
            candidates.insert(path.clone());
        }
    }
    let mut documents = Vec::new();
    for path in candidates {
        let text = path.to_string_lossy().to_string();
        if inactive.contains(&text) {
            continue;
        }
        let archived = (text.starts_with("docs/archive/") || text.starts_with("docs/legacy/"))
            && path.file_name().and_then(|name| name.to_str()) != Some("README.md");
        if archived {
            continue;
        }
        if HISTORICAL_DOCUMENTS
            .iter()
            .any(|prefix| text.starts_with(prefix))
        {
            continue;
        }
        documents.push(path);
    }
    Ok(documents)
}

/// Pages the docs manifest classifies as archived or superseded.
fn inactive_pages(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let manifest = "docs/DOCS.toml";
    if !tree.has(manifest) {
        return Ok(BTreeSet::new());
    }
    let table = tree.read_toml(manifest)?;
    let mut inactive = BTreeSet::new();
    let Some(pages) = table.get("page").and_then(toml::Value::as_array) else {
        return Ok(inactive);
    };
    for page in pages {
        let status = page.get("status").and_then(toml::Value::as_str);
        if status != Some("archived") && status != Some("superseded") {
            continue;
        }
        if let Some(path) = page.get("path").and_then(toml::Value::as_str) {
            inactive.insert(format!("docs/{path}"));
        }
    }
    Ok(inactive)
}

/// Every reference one document makes.
fn collect(
    tree: &Tree,
    document: &Path,
    references: &mut BTreeSet<Reference>,
) -> Result<(), GateError> {
    let relative = document.to_string_lossy().to_string();
    let text = tree.read(document)?;
    if relative != GENERATED_INDEX {
        for (line, span) in code_spans(&text) {
            if let Some(resolved) = resolve(tree, document, &span, false) {
                references.insert(Reference {
                    document: relative.clone(),
                    line,
                    raw: span,
                    resolved,
                    source: "code span",
                });
            }
        }
    }
    for (line, command) in command_lines(&text) {
        for token in command_path_tokens(tree, &command) {
            if let Some(resolved) = resolve(tree, document, &token, true) {
                references.insert(Reference {
                    document: relative.clone(),
                    line,
                    raw: token,
                    resolved,
                    source: "command",
                });
            }
        }
    }
    Ok(())
}

/// Every single-backtick code span, with the line it starts on.
///
/// A run of two or more backticks opens a literal or a fence rather than a span,
/// so it is skipped whole: pairing across it would read a fence body as a path.
fn code_spans(text: &str) -> Vec<(u32, String)> {
    let bytes = text.as_bytes();
    let mut runs = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'`' {
            index += 1;
            continue;
        }
        let start = index;
        while index < bytes.len() && bytes[index] == b'`' {
            index += 1;
        }
        runs.push((start, index - start));
    }
    let mut spans = Vec::new();
    let mut cursor = 0;
    while cursor + 1 < runs.len() {
        let (open, open_len) = runs[cursor];
        if open_len != 1 {
            cursor += 1;
            continue;
        }
        let (close, close_len) = runs[cursor + 1];
        let body = &text[open + 1..close];
        if close_len != 1 || body.is_empty() || body.contains('\n') {
            cursor += 1;
            continue;
        }
        spans.push((line_of(text, open), body.to_string()));
        cursor += 2;
    }
    spans
}

/// One-based line number of a byte offset.
fn line_of(text: &str, offset: usize) -> u32 {
    let count = text[..offset].bytes().filter(|byte| *byte == b'\n').count() + 1;
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// Every logical command line inside a command fence, with its line number.
///
/// A trailing backslash continues the command, so a multi-line invocation is one
/// claim rather than several fragments.
fn command_lines(text: &str) -> Vec<(u32, String)> {
    let mut commands = Vec::new();
    let mut in_fence = false;
    let mut logical = String::new();
    let mut logical_line = 0;
    for (index, raw) in text.lines().enumerate() {
        let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
        if let Some(rest) = raw.trim_start().strip_prefix("```") {
            if in_fence {
                if !logical.is_empty() {
                    commands.push((logical_line, std::mem::take(&mut logical)));
                }
                in_fence = false;
            } else {
                let language = rest.trim().to_ascii_lowercase();
                in_fence = COMMAND_LANGUAGES.contains(&language.as_str());
            }
            continue;
        }
        if !in_fence {
            continue;
        }
        let mut line = raw.trim();
        if let Some(rest) = line.strip_prefix("$ ").or_else(|| line.strip_prefix("> ")) {
            line = rest.trim_start();
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if logical.is_empty() {
            logical_line = number;
        }
        let continues = raw.trim_end().ends_with('\\');
        let fragment = if continues {
            line.trim_end_matches('\\').trim_end()
        } else {
            line
        };
        if logical.is_empty() {
            logical.push_str(fragment);
        } else {
            logical.push(' ');
            logical.push_str(fragment);
        }
        if continues {
            continue;
        }
        commands.push((logical_line, std::mem::take(&mut logical)));
    }
    if !logical.is_empty() {
        commands.push((logical_line, logical));
    }
    commands
}

/// Split a command line the way a POSIX shell would, dropping a comment tail.
fn shell_split(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut has_token = false;
    let mut glyphs = line.chars().peekable();
    while let Some(glyph) = glyphs.next() {
        match glyph {
            character if character.is_whitespace() => {
                if has_token {
                    tokens.push(std::mem::take(&mut current));
                    has_token = false;
                }
            }
            '#' if !has_token => break,
            '\'' => {
                has_token = true;
                for quoted in glyphs.by_ref() {
                    if quoted == '\'' {
                        break;
                    }
                    current.push(quoted);
                }
            }
            '"' => {
                has_token = true;
                while let Some(quoted) = glyphs.next() {
                    match quoted {
                        '"' => break,
                        '\\' => {
                            if let Some(escaped) = glyphs.next() {
                                current.push(escaped);
                            }
                        }
                        other => current.push(other),
                    }
                }
            }
            '\\' => {
                has_token = true;
                if let Some(escaped) = glyphs.next() {
                    current.push(escaped);
                }
            }
            other => {
                has_token = true;
                current.push(other);
            }
        }
    }
    if has_token {
        tokens.push(current);
    }
    tokens
}

/// Every token of a command line that claims an input path.
fn command_path_tokens(tree: &Tree, line: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut skip_next = false;
    for token in shell_split(line) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if OUTPUT_FLAGS.contains(&token.as_str()) {
            skip_next = true;
            continue;
        }
        if OUTPUT_FLAGS
            .iter()
            .any(|flag| token.starts_with(&format!("{flag}=")))
        {
            continue;
        }
        let relative_prefix = token.starts_with("./") || token.starts_with("../");
        if token.starts_with('-') || (token.contains('=') && !relative_prefix) {
            continue;
        }
        if is_path_candidate(tree, &token) {
            paths.push(token);
        }
    }
    paths
}

/// The path a raw token claims, with a fragment, a line selector and trailing
/// punctuation removed.
fn path_token(raw: &str) -> String {
    let trimmed = raw
        .trim()
        .trim_matches(|glyph| glyph == '\'' || glyph == '"')
        .trim_end_matches(['.', ',', ';', ':']);
    let without_fragment = trimmed.split('#').next().unwrap_or("");
    strip_line_selector(without_fragment).to_string()
}

/// Drop a trailing `:LINE` or `:LINE-LINE` selector.
fn strip_line_selector(token: &str) -> &str {
    let Some(colon) = token.rfind(':') else {
        return token;
    };
    let tail = &token[colon + 1..];
    if tail.is_empty() {
        return token;
    }
    let mut parts = tail.split('-');
    let start = parts.next().unwrap_or("");
    if start.is_empty() || !start.bytes().all(|byte| byte.is_ascii_digit()) {
        return token;
    }
    match parts.next() {
        None => &token[..colon],
        Some(end) if !end.is_empty() && end.bytes().all(|byte| byte.is_ascii_digit()) => {
            if parts.next().is_some() {
                token
            } else {
                &token[..colon]
            }
        }
        Some(_) => token,
    }
}

/// Whether the token's first segment names something at the repository root.
fn has_existing_root_prefix(tree: &Tree, token: &str) -> bool {
    let Some((first, _)) = token.split_once('/') else {
        return false;
    };
    first != "." && first != ".." && tree.absolute(first).exists()
}

/// Whether a token is a path claim rather than prose, a URL or Rust syntax.
fn is_path_candidate(tree: &Tree, raw: &str) -> bool {
    let token = path_token(raw);
    if token.is_empty() || token.chars().any(char::is_whitespace) {
        return false;
    }
    if token.starts_with("http://") || token.starts_with("https://") || token.starts_with("mailto:")
    {
        return false;
    }
    if token.contains("::")
        || token.starts_with('$')
        || token.contains(['<', '>', '{', '}', '@'])
        || token.starts_with("///")
    {
        return false;
    }
    if matches!(token.as_str(), "." | ".." | "./" | "../") {
        return false;
    }
    if token.starts_with("./") || token.starts_with("../") || token.starts_with('/') {
        return true;
    }
    if ROOT_PREFIXES.iter().any(|prefix| token.starts_with(prefix)) {
        return true;
    }
    if token.contains('/') && PATH_SUFFIXES.iter().any(|suffix| token.ends_with(suffix)) {
        return true;
    }
    has_existing_root_prefix(tree, &token)
}

/// The repository-relative path a token resolves to, or `None` when the token is
/// not a path claim at all.
fn resolve(tree: &Tree, document: &Path, raw: &str, from_command: bool) -> Option<String> {
    let token = path_token(raw);
    if !is_path_candidate(tree, &token) {
        return None;
    }
    let absolute_claim = token.starts_with('/');
    let document_parent = document.parent().unwrap_or(Path::new(""));
    let candidate = if absolute_claim {
        PathBuf::from(&token)
    } else if document.file_name().and_then(|name| name.to_str()) == Some("README.md")
        && document_parent != Path::new("")
        && CRATE_RELATIVE_PREFIXES
            .iter()
            .any(|prefix| token.starts_with(prefix))
    {
        tree.absolute(document_parent).join(&token)
    } else if ROOT_PREFIXES.iter().any(|prefix| token.starts_with(prefix))
        || has_existing_root_prefix(tree, &token)
    {
        tree.absolute(&token)
    } else if from_command && token.starts_with("./") {
        tree.absolute(&token[2..])
    } else {
        tree.absolute(document_parent).join(&token)
    };
    let normalized = normalize(&candidate);
    match normalized.strip_prefix(normalize(tree.root())) {
        Ok(relative) => Some(relative.to_string_lossy().replace('\\', "/")),
        Err(_) if absolute_claim => None,
        Err(_) => Some(OUTSIDE.to_string()),
    }
}

/// Resolve `.` and `..` lexically. Symlinks are irrelevant to a claim about
/// which published path a document names.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_backtick_span_is_read_and_a_double_one_is_not() {
        let spans = code_spans("see `docs/a.md` and ``literal`` then `b.rs`\n");
        let bodies: Vec<&str> = spans.iter().map(|(_, body)| body.as_str()).collect();
        assert_eq!(bodies, vec!["docs/a.md", "b.rs"]);
        assert_eq!(spans[0].0, 1);
    }

    #[test]
    fn a_fence_body_is_never_read_as_a_code_span() {
        let text = "text\n```bash\ncargo run\n```\n";
        assert!(code_spans(text).is_empty(), "got {:?}", code_spans(text));
    }

    #[test]
    fn a_continued_command_is_one_logical_line() {
        let text = "```bash\ncargo run \\\n  --manifest-path a/Cargo.toml\n```\n";
        assert_eq!(
            command_lines(text),
            vec![(2, "cargo run --manifest-path a/Cargo.toml".to_string())]
        );
    }

    #[test]
    fn a_non_command_fence_contributes_nothing() {
        assert!(command_lines("```toml\npath = \"a/b.rs\"\n```\n").is_empty());
    }

    #[test]
    fn a_line_selector_and_a_fragment_are_stripped_but_a_colon_in_prose_is_not() {
        assert_eq!(path_token("docs/a.md:12-40"), "docs/a.md");
        assert_eq!(path_token("docs/a.md#heading"), "docs/a.md");
        assert_eq!(path_token("`docs/a.md`,"), "docs/a.md");
        assert_eq!(path_token("http://example.com:80"), "http://example.com:80");
    }

    #[test]
    fn a_shell_split_drops_a_comment_and_keeps_a_quoted_token() {
        assert_eq!(
            shell_split("cargo run 'a b.rs' \"c.rs\" # tail"),
            vec!["cargo", "run", "a b.rs", "c.rs"]
        );
    }

    #[test]
    fn an_output_flag_argument_is_not_an_input_claim() {
        let tree = Tree::open(&crate::checkout::checkout_root()).expect("the checkout lists");
        assert_eq!(
            command_path_tokens(&tree, "xtask catalog --output docs/generated/absent.toml"),
            Vec::<String>::new()
        );
        assert_eq!(
            command_path_tokens(&tree, "xtask catalog --output=docs/generated/absent.toml"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn a_parent_segment_is_resolved_lexically() {
        assert_eq!(normalize(Path::new("/a/b/../c/./d")), PathBuf::from("/a/c/d"));
    }
}
