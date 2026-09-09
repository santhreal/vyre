//! `cargo xtask tracked-modules` - every declared module is in the commit.
//!
//! # The defect this exists for
//!
//! A module declaration and the file it names are two paths, and a commit can
//! carry one without the other. `vyre-spec/src/lib.rs` was committed with
//! `pub mod resource_capability;` while `vyre-spec/src/resource_capability.rs`
//! stayed untracked. The working tree compiled, every local gate was green, and
//! the pushed tree failed with E0583 on the root of the crate graph. Every
//! workflow job inherited that failure, so the branch showed twenty red lanes
//! for one missing file, and the same commit hid a second instance in
//! `vyre-runtime`.
//!
//! # Why no other check catches it
//!
//! Every other gate reads the working tree, where the file is present. A build
//! answers the question only for a checkout that lacks the file, which is
//! remote CI, after the push, at the cost of the whole matrix. Nothing in this
//! tree consulted the index before this gate.
//!
//! # What it judges
//!
//! Tracked source only. An untracked new file that nothing declares is ordinary
//! in-flight work and says nothing about the commit; the defect is a
//! declaration that is committed and unsatisfied. The two halves of a partial
//! commit are reported separately, because a file that exists and is untracked
//! is a staging mistake while a file that is absent is a broken tree, and the
//! corrective action differs.
//!
//! # Where it is deliberately blind
//!
//! `#[path]` resolution depends on whether the declaring file is a crate root,
//! and the cargo target list decides that. Rather than model the target graph,
//! a declaration is accepted when any of its candidate paths is tracked. That
//! can miss a declaration whose file is tracked at the wrong path, and it
//! cannot produce a false finding, which is the tradeoff a zero baseline
//! requires.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, Report};

/// Cap on one source file read while scanning declarations.
const MAX_SOURCE_BYTES: u64 = 4_194_304;

/// Rejects a commit whose module declarations outrun its tracked files.
pub struct TrackedModules;

impl crate::gate::GateBehavior for TrackedModules {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let root = crate::checkout::checkout_root();
        let tracked = tracked_files(&root)?;
        let sources: Vec<PathBuf> = tracked
            .iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
            .cloned()
            .collect();
        if sources.len() < 2 {
            return Err(GateError::new(
                format!(
                    "tracked-modules found {} tracked Rust file(s), which means the index was \
                     not read rather than a tree with one source file",
                    sources.len()
                ),
                "run it inside a git checkout of the workspace",
            ));
        }

        let mut loaded = Vec::with_capacity(sources.len());
        for source in sources {
            if let Some(text) = read_capped(&root.join(&source)) {
                loaded.push((source, text));
            }
        }
        let root_for_exists = root.clone();
        Ok(judge(&loaded, &tracked, &move |path| {
            root_for_exists.join(path).is_file()
        }))
    }
}

/// Judges one set of loaded sources against the index.
///
/// The filesystem arrives as a predicate so the decision is testable without a
/// checkout: whether an unsatisfied declaration is a staging mistake or a
/// broken tree is the only thing the disk is consulted for.
fn judge(
    files: &[(PathBuf, String)],
    tracked: &BTreeSet<PathBuf>,
    exists: &dyn Fn(&Path) -> bool,
) -> Report {
    let mut report = Report::clean();
    let mut declarations = 0usize;
    for (source, text) in files {
        for declaration in declarations_in(text) {
            declarations += 1;
            let candidates = candidate_paths(source, &declaration);
            if candidates.iter().any(|path| tracked.contains(path)) {
                continue;
            }
            let present = candidates.iter().find(|path| exists(path));
            let name = &declaration.name;
            report.find(Finding::at(
                source.clone(),
                declaration.line,
                match present {
                    Some(_) => format!(
                        "declares `mod {name}` and the file that satisfies it exists but is not \
                         tracked, so the commit does not carry it"
                    ),
                    None => format!("declares `mod {name}` and no tracked file provides it"),
                },
                match present {
                    Some(path) => format!(
                        "stage it by name in the same commit as this declaration: git add -- {}",
                        path.display()
                    ),
                    None => format!(
                        "write {} or delete the declaration; a declaration with no file is a \
                         build failure for every checkout but this one",
                        candidates
                            .first()
                            .map(|path| path.display().to_string())
                            .unwrap_or_else(|| name.clone())
                    ),
                },
            ));
        }
    }
    report.cover_complete("module declarations", declarations);
    report.note(format!(
        "{declarations} declaration(s) over {} tracked Rust file(s)",
        files.len()
    ));
    report
}

/// One non-inline module declaration, the inline modules that enclose it, and
/// the path attribute that redirects it.
#[derive(Debug, PartialEq, Eq)]
struct Declaration {
    name: String,
    line: u32,
    path_attribute: Option<String>,
    /// Inline `mod` blocks around the declaration, outermost first. Each one
    /// adds a directory segment to the file the declaration resolves to.
    enclosing: Vec<String>,
}

/// Every path in the index, repository-relative.
///
/// An empty or failed listing is an error rather than a clean report: a gate
/// that reports zero findings because it read nothing certifies what it never
/// checked.
fn tracked_files(root: &Path) -> Result<BTreeSet<PathBuf>, GateError> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "tracked-modules could not run git in {}: {error}",
                    root.display()
                ),
                "install git, or run the gate inside a checkout",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "git ls-files failed in {}: {}",
                root.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "run the gate inside a git checkout of the workspace",
        ));
    }
    Ok(output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .map(|entry| PathBuf::from(String::from_utf8_lossy(entry).into_owned()))
        .collect())
}

/// Reads one source file, skipping anything past the cap.
fn read_capped(path: &Path) -> Option<String> {
    let length = std::fs::metadata(path).ok()?.len();
    if length > MAX_SOURCE_BYTES {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

/// Every `mod NAME;` in one file, with the inline modules that enclose it and
/// the `#[path]` attribute that redirects it.
///
/// Comment depth and string state are carried across lines, so neither a
/// commented-out region nor a declaration embedded in a source sample is
/// judged. This tree holds several of the latter: a gate that scans for module
/// declarations keeps whole `mod` samples in string constants, and a code
/// generator emits `mod` lines as data.
///
/// An inline `mod NAME {` declares no file of its own, but it does add a path
/// segment to every declaration inside it. `vyre-foundation/src/lib.rs`
/// declares `mod ir_inner { pub(crate) mod model; }`, and that `model` lives at
/// `src/ir_inner/model`, not `src/model`.
fn declarations_in(text: &str) -> Vec<Declaration> {
    let mut found = Vec::new();
    let mut pending_path: Option<String> = None;
    let mut state = ScanState::default();
    let mut enclosing: Vec<(String, usize)> = Vec::new();
    let mut brace_depth = 0usize;
    for (index, raw) in text.lines().enumerate() {
        let line = strip_noncode(raw, &mut state);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // The attribute's value is a string, so it is read from the raw line.
        // Reaching here proves the line is code rather than sample text.
        if trimmed.starts_with("#[path") {
            if let Some(value) = path_attribute(raw.trim()) {
                pending_path = Some(value);
                continue;
            }
        }
        let opened = inline_module_name(trimmed);
        match module_name(trimmed) {
            Some(name) => found.push(Declaration {
                name,
                line: index as u32 + 1,
                path_attribute: pending_path.take(),
                enclosing: enclosing.iter().map(|(name, _)| name.clone()).collect(),
            }),
            // Any statement other than an attribute or doc line ends the
            // attribute's reach, so a stray `#[path]` cannot redirect a
            // declaration further down the file.
            None => {
                if !trimmed.starts_with("#[") && !trimmed.starts_with("///") {
                    pending_path = None;
                }
            }
        }
        let opens = line.matches('{').count();
        let closes = line.matches('}').count();
        brace_depth = (brace_depth + opens).saturating_sub(closes);
        if let Some(name) = opened {
            enclosing.push((name, brace_depth));
        }
        while enclosing
            .last()
            .is_some_and(|(_, depth)| *depth > brace_depth)
        {
            enclosing.pop();
        }
    }
    found
}

/// The name an inline `mod NAME {` opens.
fn inline_module_name(trimmed: &str) -> Option<String> {
    let opener = trimmed.strip_suffix('{')?.trim_end();
    module_name(&format!("{opener};"))
}

/// Which literal the scanner is inside of when a line ends.
enum StringKind {
    /// A `"..."` literal, where a backslash escapes the next character.
    Normal,
    /// An `r#"..."#` literal, closed by a quote and this many hashes.
    Raw(usize),
}

/// Comment and literal state carried from one line to the next.
#[derive(Default)]
struct ScanState {
    block_depth: usize,
    string: Option<StringKind>,
}

/// Returns the code on one line, with comments and literal contents removed.
fn strip_noncode(line: &str, state: &mut ScanState) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut index = 0usize;
    while index < chars.len() {
        match &state.string {
            Some(StringKind::Normal) => {
                if chars[index] == '\\' {
                    index += 2;
                    continue;
                }
                if chars[index] == '"' {
                    state.string = None;
                }
                index += 1;
                continue;
            }
            Some(StringKind::Raw(hashes)) => {
                let closes = chars[index] == '"'
                    && (1..=*hashes).all(|offset| chars.get(index + offset) == Some(&'#'));
                if closes {
                    index += hashes + 1;
                    state.string = None;
                    continue;
                }
                index += 1;
                continue;
            }
            None => {}
        }
        if state.block_depth > 0 {
            if chars[index] == '*' && chars.get(index + 1) == Some(&'/') {
                state.block_depth -= 1;
                index += 2;
                continue;
            }
            index += 1;
            continue;
        }
        if chars[index] == '/' && chars.get(index + 1) == Some(&'*') {
            state.block_depth += 1;
            index += 2;
            continue;
        }
        if chars[index] == '/' && chars.get(index + 1) == Some(&'/') {
            break;
        }
        if let Some(hashes) = raw_string_open(&chars, index, &out) {
            state.string = Some(StringKind::Raw(hashes));
            index += hashes + 2;
            continue;
        }
        if chars[index] == '"' {
            state.string = Some(StringKind::Normal);
            index += 1;
            continue;
        }
        // A character literal can hold a quote, which would otherwise open a
        // string that never closes and blind the rest of the file.
        if chars[index] == '\'' {
            if let Some(width) = char_literal_width(&chars, index) {
                index += width;
                continue;
            }
        }
        out.push(chars[index]);
        index += 1;
    }
    out
}

/// The hash count of a raw string starting at `index`, when one starts there.
fn raw_string_open(chars: &[char], index: usize, emitted: &str) -> Option<usize> {
    if chars[index] != 'r' {
        return None;
    }
    // An `r` that continues an identifier is not a literal prefix.
    if emitted
        .chars()
        .next_back()
        .is_some_and(|previous| previous.is_alphanumeric() || previous == '_')
    {
        return None;
    }
    let hashes = chars[index + 1..]
        .iter()
        .take_while(|character| **character == '#')
        .count();
    (chars.get(index + 1 + hashes) == Some(&'"')).then_some(hashes)
}

/// The width of a character literal at `index`, when one is there.
///
/// A lifetime looks the same up to its first character, so the closing quote is
/// what distinguishes them.
fn char_literal_width(chars: &[char], index: usize) -> Option<usize> {
    if chars.get(index + 1) == Some(&'\\') {
        let mut width = 3;
        while chars.get(index + width) != Some(&'\'') && index + width < chars.len() {
            width += 1;
        }
        return (chars.get(index + width) == Some(&'\'')).then_some(width + 1);
    }
    (chars.get(index + 2) == Some(&'\'')).then_some(3)
}

/// The value of a `#[path = "..."]` attribute.
fn path_attribute(trimmed: &str) -> Option<String> {
    let rest = trimmed.strip_prefix("#[path")?;
    let rest = rest.trim_start().strip_prefix('=')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// The module name a non-inline declaration names.
fn module_name(trimmed: &str) -> Option<String> {
    let mut rest = trimmed;
    // An attribute may share the line with the declaration it modifies.
    if let Some(close) = rest.rfind("] ") {
        rest = rest[close + 2..].trim_start();
    }
    for visibility in ["pub(crate) ", "pub(super) ", "pub(in crate) ", "pub "] {
        if let Some(stripped) = rest.strip_prefix(visibility) {
            rest = stripped.trim_start();
        }
    }
    let rest = rest.strip_prefix("mod ")?;
    let name = rest.strip_suffix(';')?.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(name.to_string())
}

/// Every repository-relative path that could satisfy one declaration.
///
/// Both the crate-root resolution and the module-directory resolution are
/// offered because the cargo target list decides which applies, and accepting
/// either can only under-report.
fn candidate_paths(source: &Path, declaration: &Declaration) -> Vec<PathBuf> {
    let parent = source.parent().unwrap_or(Path::new(""));
    let stem = source
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("");
    let module_dir = if matches!(stem, "lib" | "main" | "mod") {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    let mut candidates = Vec::new();
    for root in [parent.to_path_buf(), module_dir] {
        // Each inline `mod` block between the file and the declaration adds a
        // directory segment.
        let mut root = root;
        for segment in &declaration.enclosing {
            root = root.join(segment);
        }
        match &declaration.path_attribute {
            Some(attribute) => candidates.push(normalize(root.join(attribute))),
            None => {
                candidates.push(normalize(root.join(format!("{}.rs", declaration.name))));
                candidates.push(normalize(root.join(&declaration.name).join("mod.rs")));
            }
        }
    }
    candidates
}

/// Resolves `.` and `..` lexically.
///
/// A `#[path]` attribute routinely climbs out of `src`, and the index holds the
/// resolved path, so an unresolved one matches nothing and every such
/// declaration would be reported as missing.
fn normalize(path: PathBuf) -> PathBuf {
    let mut parts: Vec<std::ffi::OsString> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                parts.pop();
            }
            other => parts.push(other.as_os_str().to_os_string()),
        }
    }
    parts.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracked(paths: &[&str]) -> BTreeSet<PathBuf> {
        paths.iter().map(PathBuf::from).collect()
    }

    fn nothing_on_disk(_: &Path) -> bool {
        false
    }

    /// The exact defect: a committed declaration whose file was never staged.
    #[test]
    fn an_untracked_file_behind_a_declaration_is_a_finding() {
        let files = vec![(
            PathBuf::from("vyre-spec/src/lib.rs"),
            "pub mod resource_capability;\n".to_string(),
        )];
        let report = judge(&files, &tracked(&["vyre-spec/src/lib.rs"]), &|path| {
            path == Path::new("vyre-spec/src/resource_capability.rs")
        });
        let findings = report.findings;
        assert_eq!(findings.len(), 1);
        assert!(
            findings[0].message.contains("exists but is not tracked"),
            "a file present on disk and absent from the index is a staging mistake: {}",
            findings[0].message
        );
        assert!(
            findings[0]
                .fix
                .contains("git add -- vyre-spec/src/resource_capability.rs"),
            "the fix names the path to stage: {}",
            findings[0].fix
        );
    }

    /// A declaration with no file anywhere is a different defect and says so.
    #[test]
    fn a_declaration_with_no_file_at_all_is_reported_as_a_broken_tree() {
        let files = vec![(
            PathBuf::from("vyre-spec/src/lib.rs"),
            "mod gone;\n".to_string(),
        )];
        let report = judge(
            &files,
            &tracked(&["vyre-spec/src/lib.rs"]),
            &nothing_on_disk,
        );
        let findings = report.findings;
        assert_eq!(findings.len(), 1);
        assert!(
            findings[0].message.contains("no tracked file provides it"),
            "{}",
            findings[0].message
        );
    }

    /// A declaration inside a string constant is sample text, not a claim.
    ///
    /// This tree has three of them in one gate and one in a code generator, so
    /// a scanner that reads them is red on a tree that is right.
    #[test]
    fn a_declaration_inside_a_string_literal_is_not_judged() {
        for source in [
            "const SAMPLE: &str = \"mod embedded;\";\n",
            "const SAMPLE: &str = r#\"mod embedded;\"#;\n",
            "const SAMPLE: &str = \"pub mod a;\\npub mod b;\\n\";\n",
            "write(\"mod embedded;\\n\");\n",
        ] {
            assert!(
                declarations_in(source).is_empty(),
                "sample text judged as a declaration: {source}"
            );
        }
    }

    /// A literal spanning lines hides every declaration until it closes.
    #[test]
    fn a_multiline_literal_hides_the_declarations_inside_it() {
        let source = "const SAMPLE: &str = r#\"\nmod inside_one;\nmod inside_two;\n\"#;\nmod \
                      after;\n";
        let found = declarations_in(source);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].name, "after");
    }

    /// A quote inside a character literal must not open a string.
    #[test]
    fn a_quote_in_a_character_literal_does_not_blind_the_scanner() {
        let found = declarations_in("const Q: char = '\"';\nmod after;\n");
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].name, "after");
    }

    /// A `#[path]` attribute that climbs out of `src` resolves to the index.
    #[test]
    fn a_path_attribute_that_climbs_resolves_against_the_index() {
        let files = vec![(
            PathBuf::from("vyre-spec/src/lib.rs"),
            "#[path = \"../tests/internal/mod.rs\"]\nmod tests;\n".to_string(),
        )];
        let report = judge(
            &files,
            &tracked(&["vyre-spec/src/lib.rs", "vyre-spec/tests/internal/mod.rs"]),
            &nothing_on_disk,
        );
        assert!(
            report.findings.is_empty(),
            "an unresolved `..` matches nothing in the index: {:?}",
            report
                .findings
                .iter()
                .map(|finding| finding.fix.as_str())
                .collect::<Vec<_>>()
        );
    }

    /// An inline block adds a directory segment to what its children resolve
    /// to, which is the shape `vyre-foundation/src/lib.rs` uses.
    #[test]
    fn an_inline_block_adds_a_directory_segment_to_its_children() {
        let files = vec![(
            PathBuf::from("vyre-foundation/src/lib.rs"),
            "mod ir_inner {\n    pub(crate) mod model;\n}\npub mod after;\n".to_string(),
        )];
        let satisfied = judge(
            &files,
            &tracked(&[
                "vyre-foundation/src/lib.rs",
                "vyre-foundation/src/ir_inner/model/mod.rs",
                "vyre-foundation/src/after.rs",
            ]),
            &nothing_on_disk,
        );
        assert!(
            satisfied.findings.is_empty(),
            "a child of an inline block resolves under that block: {:?}",
            satisfied
                .findings
                .iter()
                .map(|finding| finding.fix.as_str())
                .collect::<Vec<_>>()
        );

        // The segment is real: the same file one level up does not satisfy it,
        // and the declaration after the block is back at the top level.
        let wrong_level = judge(
            &files,
            &tracked(&[
                "vyre-foundation/src/lib.rs",
                "vyre-foundation/src/model/mod.rs",
                "vyre-foundation/src/ir_inner/after.rs",
            ]),
            &nothing_on_disk,
        );
        assert_eq!(wrong_level.findings.len(), 2, "{:?}", wrong_level.findings);
    }

    /// A tracked file satisfies its declaration under either resolution.
    #[test]
    fn a_tracked_file_satisfies_its_declaration() {
        let files = vec![
            (
                PathBuf::from("vyre-spec/src/lib.rs"),
                "pub mod present;\npub mod nested;\n".to_string(),
            ),
            (
                PathBuf::from("vyre-spec/src/deep.rs"),
                "pub mod inner;\n".to_string(),
            ),
        ];
        let report = judge(
            &files,
            &tracked(&[
                "vyre-spec/src/lib.rs",
                "vyre-spec/src/deep.rs",
                "vyre-spec/src/present.rs",
                "vyre-spec/src/nested/mod.rs",
                "vyre-spec/src/deep/inner.rs",
            ]),
            &nothing_on_disk,
        );
        assert!(
            report.findings.is_empty(),
            "{:?}",
            report
                .findings
                .iter()
                .map(|finding| finding.message.as_str())
                .collect::<Vec<_>>()
        );
    }

    /// The form the test trees actually use: a sibling file behind `#[path]`.
    #[test]
    fn a_path_attribute_redirects_to_the_sibling_file() {
        let files = vec![(
            PathBuf::from("vyre-spec/tests/all_tests.rs"),
            "/// Integration tests.\n#[path = \"resource_capability_contracts.rs\"]\npub mod \
             resource_capability_contracts;\n"
                .to_string(),
        )];
        let satisfied = judge(
            &files,
            &tracked(&[
                "vyre-spec/tests/all_tests.rs",
                "vyre-spec/tests/resource_capability_contracts.rs",
            ]),
            &nothing_on_disk,
        );
        assert!(satisfied.findings.is_empty());

        let unsatisfied = judge(
            &files,
            &tracked(&["vyre-spec/tests/all_tests.rs"]),
            &nothing_on_disk,
        );
        assert_eq!(unsatisfied.findings.len(), 1);
    }

    /// An inline module declares no file, so it is not a declaration.
    #[test]
    fn an_inline_module_is_not_judged() {
        assert!(declarations_in("mod tests {\n    fn t() {}\n}\n").is_empty());
        assert!(declarations_in("#[cfg(test)]\nmod tests {}\n").is_empty());
    }

    /// A commented-out declaration is text, not a claim about the tree.
    #[test]
    fn a_commented_declaration_is_not_judged() {
        assert!(declarations_in("// mod dead;\n").is_empty());
        assert!(declarations_in("/* mod dead;\n   mod also_dead; */\n").is_empty());
        assert!(declarations_in("//! mod documented;\n").is_empty());
        assert_eq!(declarations_in("mod live; // mod dead;\n").len(), 1);
    }

    /// Every visibility form and a same-line attribute reach the same name.
    #[test]
    fn every_declaration_form_yields_the_name() {
        for source in [
            "mod alpha;",
            "pub mod alpha;",
            "pub(crate) mod alpha;",
            "pub(super) mod alpha;",
            "#[cfg(test)] mod alpha;",
            "#[cfg(feature = \"x\")] pub mod alpha;",
        ] {
            let found = declarations_in(source);
            assert_eq!(found.len(), 1, "no declaration found in {source}");
            assert_eq!(found[0].name, "alpha", "wrong name from {source}");
        }
    }

    /// A `#[path]` attribute reaches only the declaration that follows it.
    #[test]
    fn a_path_attribute_does_not_carry_past_an_intervening_statement() {
        let found = declarations_in("#[path = \"a.rs\"]\nuse std::io;\npub mod b;\n");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path_attribute, None);
    }

    /// Coverage counts declarations, so a scan that read nothing cannot pass as
    /// a clean tree.
    #[test]
    fn coverage_counts_the_declarations_judged() {
        let files = vec![(
            PathBuf::from("a/src/lib.rs"),
            "pub mod one;\npub mod two;\n".to_string(),
        )];
        let report = judge(
            &files,
            &tracked(&["a/src/lib.rs", "a/src/one.rs", "a/src/two.rs"]),
            &nothing_on_disk,
        );
        assert!(report.findings.is_empty());
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("2 declaration(s)")),
            "{:?}",
            report.notes
        );
        assert!(
            report.coverage.iter().any(|coverage| coverage.is_closed()
                && coverage.discovered == 2
                && coverage.judged == 2),
            "the judged universe is the declaration set, so a scan that read nothing \
             cannot report a closed cover: {:?}",
            report.coverage
        );
    }
}
