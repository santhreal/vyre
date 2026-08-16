//! The tracked-source scanner every scanning gate reads the tree through.
//!
//! Nine shell gates each carried their own ripgrep invocation. Most ended in
//! `2>/dev/null || true`, which turns a failed search into an empty result, and
//! an empty result is what those gates read as a clean tree. The inventory scan
//! now owned by `hot-path-inventory` passed on every possible tree for two
//! years: it asked for `-P`, that ripgrep build has no PCRE2, every invocation
//! errored, and the error went to `/dev/null`.
//!
//! Here a failed scan is a `GateError` and cannot be mistaken for a clean tree,
//! because the type does not allow it. A scan path that does not exist is also a
//! `GateError`: a rule scanning nothing reports success forever.
//!
//! Tracked files only. A count taken over whatever is on disk moves with
//! untracked scratch, which makes a ratchet disagree between a development tree
//! and CI.
//!
//! There are two scanners in this workspace on purpose, and collapsing them
//! would undo what this one is for. `structure_gate::source_scan` walks the
//! filesystem with `walkdir`, prunes build output, and skips a file it cannot
//! read, which is right for a contract test that judges whatever source is
//! present. This one lists from `git ls-files` and makes a missing scan path a
//! hard error, which is right for a ratchet whose number must mean the same
//! thing in a development tree and in CI. The two split their work: the file set
//! and the ratchet engine belong here, and masking comments and string literals
//! and finding the end of a brace-delimited block belong to `source_scan`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::GateError;

/// Largest file any gate reads.
pub const MAX_FILE_BYTES: u64 = 16_777_216;

/// One matching line.
pub struct Hit {
    /// Repository-relative path of the file the line came from.
    pub file: PathBuf,
    /// One-based line number.
    pub line: u32,
    /// The line, trimmed.
    pub text: String,
}

/// The set of files the repository will carry, and reads against them.
///
/// The set is what git would commit: tracked files plus untracked files no
/// ignore rule excludes.
pub struct Tree {
    root: PathBuf,
    paths: Vec<PathBuf>,
    absent: Vec<PathBuf>,
}

impl Tree {
    /// List the tree, or fail naming what could not be listed.
    pub fn open(root: &Path) -> Result<Self, GateError> {
        let listing = Command::new("git")
            .arg("-C")
            .arg(root)
            .args([
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ])
            .output()
            .map_err(|error| {
                GateError::new(
                    format!("cannot list the tree: {error}"),
                    "install git, or run this gate inside a git checkout of the repository",
                )
            })?;
        if !listing.status.success() {
            return Err(GateError::new(
                format!(
                    "cannot list the tree: git ls-files exited {}",
                    listing.status
                ),
                "run this gate inside a git checkout of the repository",
            ));
        }
        let mut paths = Vec::new();
        let mut absent = Vec::new();
        for entry in listing.stdout.split(|byte| *byte == 0) {
            if entry.is_empty() {
                continue;
            }
            let relative = PathBuf::from(String::from_utf8_lossy(entry).as_ref());
            if root.join(&relative).is_file() {
                paths.push(relative);
            } else {
                absent.push(relative);
            }
        }
        paths.sort();
        absent.sort();
        Ok(Self {
            root: root.to_path_buf(),
            paths,
            absent,
        })
    }

    #[must_use]
    /// The workspace root every relative path resolves against.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Every listed file, repository-relative and sorted.
    #[must_use]
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Tracked files absent from the working tree.
    ///
    /// A parallel edit can delete a tracked file, and a fresh CI checkout never
    /// has one, so this is a development-tree condition rather than a rule
    /// failure. It is still reported as a note: a file dropped from a scan
    /// without a word is how a gate quietly stops covering what it names.
    #[must_use]
    pub fn absent(&self) -> &[PathBuf] {
        &self.absent
    }

    /// The note a gate carries when the scan was narrower than the tree claims.
    #[must_use]
    pub fn absence_note(&self) -> Option<String> {
        if self.absent.is_empty() {
            return None;
        }
        Some(format!(
            "{} tracked file(s) absent from the working tree, not scanned: {}",
            self.absent.len(),
            self.absent
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    /// Whether a repository-relative path is listed and present.
    #[must_use]
    pub fn has(&self, relative: &str) -> bool {
        self.paths.iter().any(|path| path == Path::new(relative))
    }

    /// Whether anything exists at a repository-relative path, tracked or not.
    #[must_use]
    pub fn exists(&self, relative: &str) -> bool {
        self.root.join(relative).exists()
    }

    #[must_use]
    /// An absolute path for a repository-relative one.
    pub fn absolute(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.root.join(relative)
    }

    /// Read one file under the read bound.
    pub fn read(&self, relative: impl AsRef<Path>) -> Result<String, GateError> {
        let relative = relative.as_ref();
        let path = self.root.join(relative);
        let size = fs::metadata(&path)
            .map_err(|error| {
                GateError::new(
                    format!("cannot inspect `{}`: {error}", relative.display()),
                    "restore the file the gate reads, or repoint the gate at the path it moved to",
                )
            })?
            .len();
        if size > MAX_FILE_BYTES {
            return Err(GateError::new(
                format!(
                    "`{}` is {size} bytes, above the {MAX_FILE_BYTES}-byte read bound",
                    relative.display()
                ),
                "split the file, or raise the bound in xtask/src/gates/scan.rs deliberately",
            ));
        }
        fs::read_to_string(&path).map_err(|error| {
            GateError::new(
                format!("cannot read `{}`: {error}", relative.display()),
                "restore the file as valid UTF-8 text",
            )
        })
    }

    /// Read and parse one TOML file.
    pub fn read_toml(&self, relative: impl AsRef<Path>) -> Result<toml::Table, GateError> {
        let relative = relative.as_ref();
        let text = self.read(relative)?;
        toml::from_str::<toml::Table>(&text).map_err(|error| {
            GateError::new(
                format!("cannot parse TOML `{}`: {error}", relative.display()),
                "repair the manifest so it parses",
            )
        })
    }

    /// Listed files under the given roots with one of the given extensions.
    ///
    /// A root that does not exist is fatal. Every ratchet here names the paths
    /// it covers, and three of them scored zero for a year because the code had
    /// moved out from under a path nobody rechecked.
    pub fn scope(&self, roots: &[&str], extensions: &[&str]) -> Result<Vec<PathBuf>, GateError> {
        for root in roots {
            if !self.root.join(root).exists() {
                return Err(GateError::new(
                    format!("scan path does not exist: {root}"),
                    "repoint the rule at the path the code moved to, or delete the rule; \
                     a rule scanning nothing reports success forever",
                ));
            }
        }
        let mut files = Vec::new();
        for path in &self.paths {
            let matches_extension = extensions.is_empty()
                || extensions.iter().any(|extension| {
                    path.extension().and_then(|value| value.to_str()) == Some(*extension)
                });
            if !matches_extension {
                continue;
            }
            if roots.iter().any(|root| under(path, root)) {
                files.push(path.clone());
            }
        }
        Ok(files)
    }

    /// Rust sources under the given roots.
    pub fn rust(&self, roots: &[&str]) -> Result<Vec<PathBuf>, GateError> {
        self.scope(roots, &["rs"])
    }

    /// Every Rust source in the tree.
    #[must_use]
    pub fn all_rust(&self) -> Vec<PathBuf> {
        self.paths
            .iter()
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("rs"))
            .cloned()
            .collect()
    }

    /// Every workspace member directory, from `workspace.members`.
    ///
    /// Globs are expanded against the listed tree rather than the filesystem, so
    /// a member set and a scan set can never disagree.
    pub fn members(&self) -> Result<Vec<String>, GateError> {
        let manifest = self.read_toml("Cargo.toml")?;
        let declared = manifest
            .get("workspace")
            .and_then(|workspace| workspace.get("members"))
            .and_then(toml::Value::as_array)
            .ok_or_else(|| {
                GateError::new(
                    "the root Cargo.toml declares no workspace.members array",
                    "declare workspace.members as an array of member directories",
                )
            })?;
        let mut members = BTreeSet::new();
        for entry in declared {
            let entry = entry.as_str().ok_or_else(|| {
                GateError::new(
                    "workspace.members holds a non-string entry",
                    "declare every member as a string path",
                )
            })?;
            if entry.contains('*') {
                let prefix = entry.trim_end_matches('*').trim_end_matches('/');
                for path in &self.paths {
                    if path.file_name().and_then(|name| name.to_str()) != Some("Cargo.toml") {
                        continue;
                    }
                    let Some(parent) = path.parent().and_then(Path::to_str) else {
                        continue;
                    };
                    if parent.starts_with(prefix) && parent != prefix {
                        members.insert(parent.to_string());
                    }
                }
            } else {
                members.insert(entry.to_string());
            }
        }
        Ok(members.into_iter().collect())
    }

    /// Every workspace member with its manifest, in member order.
    pub fn member_manifests(&self) -> Result<Vec<Member>, GateError> {
        let mut members = Vec::new();
        for path in self.members()? {
            let manifest = format!("{path}/Cargo.toml");
            let table = self.read_toml(&manifest)?;
            let name = table
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                .ok_or_else(|| {
                    GateError::new(
                        format!("workspace member `{path}` declares no package.name"),
                        "declare package.name in the member manifest",
                    )
                })?
                .to_string();
            members.push(Member {
                path,
                name,
                manifest: table,
            });
        }
        Ok(members)
    }

    /// Every line of every named file that satisfies the predicate.
    pub fn hits(
        &self,
        files: &[PathBuf],
        predicate: impl Fn(&str) -> bool,
    ) -> Result<Vec<Hit>, GateError> {
        let mut hits = Vec::new();
        for file in files {
            let text = self.read(file)?;
            for (index, line) in text.lines().enumerate() {
                if predicate(line) {
                    hits.push(Hit {
                        file: file.clone(),
                        line: u32::try_from(index + 1).unwrap_or(u32::MAX),
                        text: line.trim().to_string(),
                    });
                }
            }
        }
        Ok(hits)
    }
}

/// One workspace member.
pub struct Member {
    /// Repository-relative directory of the member.
    pub path: String,
    /// Package name from the member manifest.
    pub name: String,
    /// The member manifest, parsed.
    pub manifest: toml::Table,
}

impl Member {
    /// Whether Cargo would publish this member.
    #[must_use]
    pub fn publishable(&self) -> bool {
        let Some(publish) = self
            .manifest
            .get("package")
            .and_then(|package| package.get("publish"))
        else {
            return true;
        };
        match publish {
            toml::Value::Boolean(value) => *value,
            toml::Value::Array(registries) => !registries.is_empty(),
            _ => true,
        }
    }

    /// The member's declared feature names.
    #[must_use]
    pub fn features(&self) -> Vec<String> {
        self.manifest
            .get("features")
            .and_then(toml::Value::as_table)
            .map(|features| features.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// Whether one published path matches a glob, with `*` and `?` confined to one
/// path segment and `**` spanning any number of them.
///
/// Two gates need this: a document may name a set of pages rather than one page,
/// and an optimization lane names the paths it may write as globs. Matching
/// against the listed tree rather than the filesystem is what makes both answers
/// the same in a development tree and in CI.
#[must_use]
pub fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    segments_match(&pattern, &path)
}

fn segments_match(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((head, rest)) if *head == "**" => {
            (0..=path.len()).any(|skip| segments_match(rest, &path[skip..]))
        }
        Some((head, rest)) => match path.split_first() {
            Some((first, tail)) if segment_match(head, first) => segments_match(rest, tail),
            _ => false,
        },
    }
}

/// Match one glob segment against one path segment.
///
/// A reachability table over the segment rather than backtracking, so a pattern
/// with several stars costs the product of the two lengths instead of exploding.
fn segment_match(pattern: &str, name: &str) -> bool {
    let name: Vec<char> = name.chars().collect();
    let mut reachable = vec![false; name.len() + 1];
    reachable[0] = true;
    for glyph in pattern.chars() {
        if glyph == '*' {
            let mut seen = false;
            for cell in &mut reachable {
                seen |= *cell;
                *cell = seen;
            }
            continue;
        }
        let mut next = vec![false; name.len() + 1];
        for (index, in_name) in name.iter().enumerate() {
            if reachable[index] && (glyph == '?' || glyph == *in_name) {
                next[index + 1] = true;
            }
        }
        reachable = next;
    }
    reachable[name.len()]
}

/// One scanning rule: a scope, a line predicate, and a reviewed allowance.
///
/// Every shell ratchet had this shape and reimplemented it, which is how three
/// of them ended up scoring zero against paths the code had moved out of.
pub struct Rule<'r> {
    /// Repository-relative roots the rule covers. A root that does not exist is
    /// fatal, because a rule scanning nothing reports success forever.
    pub roots: &'r [&'r str],
    /// Paths inside the scope the rule does not cover, such as test trees. A skip
    /// that leaves nothing behind is fatal for the same reason a missing root is:
    /// the rule scans no file and reports success forever.
    pub skip: &'r dyn Fn(&Path) -> bool,
    /// What a matching line looks like.
    pub line: &'r dyn Fn(&str) -> bool,
    /// Files whose occurrences a reviewer has signed off, each documenting its
    /// own bound or its init-only nature in the module.
    pub reviewed: &'r [&'r str],
    /// An occurrence form that is reviewed wherever it appears, such as a hit
    /// inside a doc comment.
    pub reviewed_line: Option<&'r dyn Fn(&str) -> bool>,
    /// A second condition on the whole statement the line sits in. An occurrence
    /// counts only when this also holds, which is how a rule judges a method
    /// chain whose bound is spelled on an earlier line of the same expression.
    pub statement: Option<&'r dyn Fn(&str) -> bool>,
    /// What an occurrence means.
    pub message: &'r str,
    /// The corrective action for an occurrence.
    pub fix: &'r str,
    /// What an occurrence nobody signed off means, and what to do about it.
    pub unreviewed_message: &'r str,
    /// The corrective action for an occurrence nobody signed off.
    pub unreviewed_fix: &'r str,
}

/// Run one scanning rule.
///
/// An occurrence is one finding: the pinned number is the ratchet, so a new
/// occurrence raises the count and fails. An occurrence nobody reviewed is a
/// second finding on the same line, because two independent things are wrong and
/// one pinned number has to move when either does. Without the second finding an
/// occurrence could move out of a reviewed file into the hot path without
/// changing the total, which is the case `--strict` used to cover through a
/// caller nothing invoked.
pub fn ratchet(tree: &Tree, rule: &Rule<'_>) -> Result<crate::gate::Report, GateError> {
    use crate::gate::{Finding, Report};

    let in_scope = tree.rust(rule.roots)?;
    let in_scope_count = in_scope.len();
    let files: Vec<PathBuf> = in_scope
        .into_iter()
        .filter(|path| !(rule.skip)(path))
        .collect();
    if files.is_empty() {
        return Err(GateError::new(
            format!(
                "the rule under {} skips all {in_scope_count} file(s) in scope",
                rule.roots.join(", ")
            ),
            "narrow the skip predicate or move the roots to where the subject now lives; a rule \
             that scans no file reports success forever",
        ));
    }
    let mut hits = tree.hits(&files, |line| (rule.line)(line))?;
    if let Some(predicate) = rule.statement {
        hits = retain_by_statement(tree, hits, predicate)?;
    }
    let mut report = Report::clean();
    if let Some(note) = tree.absence_note() {
        report.note(note);
    }
    report.note(format!(
        "scanned {} file(s) under {}",
        files.len(),
        rule.roots.join(", ")
    ));
    let mut used = vec![false; rule.reviewed.len()];
    for hit in &hits {
        let mut reviewed = false;
        for (index, entry) in rule.reviewed.iter().enumerate() {
            if hit.file == Path::new(entry) {
                used[index] = true;
                reviewed = true;
            }
        }
        if !reviewed {
            if let Some(predicate) = rule.reviewed_line {
                reviewed = predicate(&hit.text);
            }
        }
        report.find(Finding::at(
            hit.file.clone(),
            hit.line,
            format!("{}: {}", rule.message, hit.text),
            rule.fix.to_string(),
        ));
        if !reviewed {
            report.find(Finding::at(
                hit.file.clone(),
                hit.line,
                format!("{}: {}", rule.unreviewed_message, hit.text),
                rule.unreviewed_fix.to_string(),
            ));
        }
    }
    for (index, entry) in rule.reviewed.iter().enumerate() {
        if !used[index] {
            report.find(Finding::in_file(
                *entry,
                format!("reviewed exemption `{entry}` matches no occurrence"),
                "delete the stale entry; it reserves an exemption nothing uses",
            ));
        }
    }
    Ok(report)
}

/// Keep the hits whose whole statement satisfies `predicate`.
///
/// Hits arrive grouped by file, so one file's text is held while its hits are
/// judged and dropped when the next file starts.
fn retain_by_statement(
    tree: &Tree,
    hits: Vec<Hit>,
    predicate: &dyn Fn(&str) -> bool,
) -> Result<Vec<Hit>, GateError> {
    let mut kept = Vec::with_capacity(hits.len());
    let mut current: Option<(PathBuf, String)> = None;
    for hit in hits {
        let text = match &current {
            Some((file, text)) if *file == hit.file => text,
            _ => {
                let text = tree.read(&hit.file)?;
                &current.insert((hit.file.clone(), text)).1
            }
        };
        let lines: Vec<&str> = text.lines().collect();
        let index = (hit.line as usize).saturating_sub(1);
        if predicate(&statement_at(&lines, index)) {
            kept.push(hit);
        }
    }
    Ok(kept)
}

/// The whole statement the line at `index` belongs to, joined into one line.
///
/// A method chain spans lines, so a rule that judges a call by what the rest of
/// its chain carries cannot read one line. The statement starts after the nearest
/// earlier line whose code closes a statement, opens or closes a block, or holds
/// no code at all, and ends at the first line from `index` that closes one.
#[must_use]
pub fn statement_at(lines: &[&str], index: usize) -> String {
    fn closes_statement(line: &str) -> bool {
        let code = scan_code(line).code;
        let code = code.trim_end();
        code.is_empty()
            || code.ends_with(';')
            || code.ends_with('{')
            || code.ends_with('}')
            || code.ends_with(',')
    }

    if lines.is_empty() {
        return String::new();
    }
    let index = index.min(lines.len() - 1);
    let mut start = index;
    while start > 0 && !closes_statement(lines[start - 1]) {
        start -= 1;
    }
    let mut end = index;
    while end + 1 < lines.len() && !closes_statement(lines[end]) {
        end += 1;
    }
    lines[start..=end]
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether a path sits under a test or bench tree, which is not a dispatch path.
#[must_use]
pub fn is_test_tree(path: &Path) -> bool {
    let path = path.to_string_lossy();
    path.contains("/test/")
        || path.contains("/tests/")
        || path.contains("/benches/")
        || path.contains("/fuzz/")
}

/// Whether a line begins a comment, which quotes a symbol rather than calling it.
#[must_use]
pub fn is_comment(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*')
}

/// The text with every string and char literal blanked, comments left intact.
///
/// A detector's own pattern table is source that contains every shape it looks
/// for, so a rule that reads raw lines reports itself. Blanking literals removes
/// that class without an exemption naming the detector. Comments stay because
/// several rules read prose, and the span walk belongs to
/// `structure_gate::opaque_span`, which is the one owner of what is not code.
///
/// Byte length and line structure are preserved, so a line number taken from the
/// masked text names the same line in the file.
#[must_use]
pub fn mask_literals(text: &str) -> String {
    let mut masked = String::with_capacity(text.len());
    let mut at = 0;
    while at < text.len() {
        if !text.is_char_boundary(at) {
            at += 1;
            continue;
        }
        match structure_gate::opaque_span(text, at) {
            Some(span) if span > 0 => {
                let mut end = (at + span).min(text.len());
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }
                let piece = &text[at..end];
                if piece.starts_with("//") || piece.starts_with("/*") {
                    masked.push_str(piece);
                } else {
                    for character in piece.chars() {
                        if character == '\n' {
                            masked.push('\n');
                        } else {
                            // One space per byte, so a masked offset still names its own line.
                            for _ in 0..character.len_utf8() {
                                masked.push(' ');
                            }
                        }
                    }
                }
                at = end;
            }
            _ => {
                let character = text[at..].chars().next().unwrap_or(' ');
                masked.push(character);
                at += character.len_utf8();
            }
        }
    }
    masked
}

/// Whether a repository-relative path sits at or under a root.
#[must_use]
pub fn under(path: &Path, root: &str) -> bool {
    let path = path.to_string_lossy();
    if root.is_empty() || root == "." {
        return true;
    }
    path == root || path.starts_with(&format!("{root}/"))
}

/// Whether a line contains any of the needles.
#[must_use]
pub fn contains_any(line: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| line.contains(needle))
}

/// Which needle a line contains, for a finding that names the rule it broke.
#[must_use]
pub fn first_of<'n>(line: &str, needles: &[&'n str]) -> Option<&'n str> {
    needles
        .iter()
        .find(|needle| line.contains(**needle))
        .copied()
}

/// Whether a line contains the needle as a whole identifier.
///
/// A substring search for `stable` also matches `unstable`, which is how the CI
/// matrix gate reported a toolchain axis it had lost.
#[must_use]
pub fn contains_word(line: &str, needle: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(at) = line[from..].find(needle) {
        let start = from + at;
        let end = start + needle.len();
        let before_is_word = start > 0 && is_word_byte(bytes[start - 1]);
        let after_is_word = end < bytes.len() && is_word_byte(bytes[end]);
        if !before_is_word && !after_is_word {
            return true;
        }
        from = start + 1;
        if from >= line.len() {
            break;
        }
    }
    false
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Lines of a file with their one-based numbers, for gates that need position.
#[must_use]
pub fn numbered(text: &str) -> Vec<(u32, &str)> {
    text.lines()
        .enumerate()
        .map(|(index, line)| (u32::try_from(index + 1).unwrap_or(u32::MAX), line))
        .collect()
}

/// Which lines belong to a test-only item, by 0-based index.
///
/// The scan always meant to exclude test code: it skipped a line that WAS the
/// `#[cfg(test)]` attribute, and called that "intentional dev-only lines, not
/// runtime cost". An attribute annotates the item that follows it, so skipping
/// the attribute line excluded one line and nothing else. Every `panic!`,
/// `format!` and `.to_string()` inside a `mod tests` body was reported as
/// runtime hot-path cost, including panics that exist only to destructure a
/// fixture and which the heatmap then weighted twelve times per kLOC. The same
/// bodies inflated `count_code_lines`, so the per-kLOC denominator was wrong in
/// the opposite direction and a file's real density read lower than it is.
///
/// An item with a body runs to the line that closes it. An item without one
/// (`#[cfg(test)] use super::*;`) runs to its terminating semicolon.
#[must_use]
pub fn cfg_test_lines(lines: &[&str]) -> Vec<bool> {
    let mut test_only = vec![false; lines.len()];
    let mut depth = 0i32;
    let mut index = 0usize;
    while index < lines.len() {
        let scan = scan_code(lines[index]);
        if !is_test_only_attribute(scan.code) {
            depth += scan.brace_delta;
            index += 1;
            continue;
        }
        let outer_depth = depth;
        let mut opened = false;
        while index < lines.len() {
            let line = scan_code(lines[index]);
            depth += line.brace_delta;
            opened |= line.brace_delta > 0;
            test_only[index] = true;
            index += 1;
            if opened && depth <= outer_depth {
                break;
            }
            if !opened && line.code.trim_end().ends_with(';') {
                break;
            }
        }
    }
    test_only
}

/// Whether an attribute puts its item in the test harness and nowhere else.
///
/// `#[test]` is the plainest case: the item exists only under `cargo test`, and
/// a `#[test]` function written beside shipped code in `src/` is as much test
/// code as one inside `mod tests`.
///
/// For `#[cfg(..)]` the predicate decides. `test` and `all(test, feature = "x")`
/// cannot compile outside the harness. `any(test, feature = "cpu-parity")` is a
/// different claim: it compiles in a release build with that feature on, and the
/// 864 items carrying it in this workspace are shipped code. `not(test)` is the
/// opposite of a test item.
#[must_use]
pub fn is_test_only_attribute(code: &str) -> bool {
    let trimmed = code.trim();
    if trimmed == "#[test]" || trimmed.ends_with("::test]") {
        return true;
    }
    let Some(predicate) = code
        .trim()
        .strip_prefix("#[cfg(")
        .and_then(|rest| rest.strip_suffix(")]"))
    else {
        return false;
    };
    let predicate = predicate.trim();
    if predicate == "test" {
        return true;
    }
    predicate
        .strip_prefix("all(")
        .and_then(|rest| rest.strip_suffix(')'))
        .is_some_and(|inner| inner.split(',').any(|term| term.trim() == "test"))
}

/// One line's runtime code and the nesting it contributes.
pub struct CodeScan<'a> {
    /// Everything before a line comment.
    pub code: &'a str,
    /// The line comment, from its `//` to the end of the line, or empty.
    pub comment: &'a str,
    /// `{` minus `}`, counting only braces that sit in code.
    pub brace_delta: i32,
    /// `(` minus `)`, counting only parentheses that sit in code.
    pub paren_delta: i32,
}

/// Split `line` into runtime code and comment, and count its nesting.
///
/// Braces inside a string or character literal do not open or close a block,
/// and `//` inside one does not start a comment. A `'` that opens a lifetime
/// rather than a character literal is left alone: taking `&'a str` for the
/// start of a literal used to swallow the rest of the line, so a `//` after a
/// lifetime read as code and a `{` after one moved the block depth.
#[must_use]
pub fn scan_code(line: &str) -> CodeScan<'_> {
    let bytes = line.as_bytes();
    let mut brace_delta = 0i32;
    let mut paren_delta = 0i32;
    let mut in_string = false;
    let mut in_char = false;
    let mut escaped = false;
    let mut index = 0usize;

    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        if in_string {
            if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if in_char {
            if byte == b'\\' {
                escaped = true;
            } else if byte == b'\'' {
                in_char = false;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'\'' if opens_char_literal(bytes, index) => in_char = true,
            b'/' if bytes.get(index + 1) == Some(&b'/') => {
                return CodeScan {
                    code: &line[..index],
                    comment: &line[index..],
                    brace_delta,
                    paren_delta,
                };
            }
            b'{' => brace_delta += 1,
            b'}' => brace_delta -= 1,
            b'(' => paren_delta += 1,
            b')' => paren_delta -= 1,
            _ => {}
        }
        index += 1;
    }
    CodeScan {
        code: line,
        comment: "",
        brace_delta,
        paren_delta,
    }
}

/// True when the `'` at `index` opens a character literal rather than a
/// lifetime, judged by whether a closing `'` follows within one escape
/// sequence.
fn opens_char_literal(bytes: &[u8], index: usize) -> bool {
    let limit = (index + 5).min(bytes.len());
    bytes[index + 1..limit].contains(&b'\'')
}

/// Which lines construct or handle an error, by 0-based index.
///
/// A measured path is the one a successful call takes. An error message is
/// built once, on the way out, by code that has already decided the call
/// failed, and this workspace requires that message to carry context and a
/// fix, so every such message allocates. Counting those allocations as
/// per-dispatch cost set two rules against each other: the CUDA dispatch
/// surface read as nineteen allocations on the launch path when every one of
/// them was a `format!` inside a `return Err`, and the only way to meet the
/// budget was to make the errors say less.
///
/// A construction runs from the line that opens it to the line where the
/// nesting it opened closes, so a message that spans six lines is one
/// exclusion and not six.
#[must_use]
pub fn error_construction_lines(lines: &[&str]) -> Vec<bool> {
    const OPENERS: &[&str] = &["Err(", "map_err(", "ok_or_else(", "ok_or("];
    let mut on_error_path = vec![false; lines.len()];
    let mut depth = 0i32;
    let mut index = 0usize;
    while index < lines.len() {
        let scan = scan_code(lines[index]);
        let delta = scan.brace_delta + scan.paren_delta;
        if !contains_any(scan.code, OPENERS) {
            depth += delta;
            index += 1;
            continue;
        }
        let outer_depth = depth;
        while index < lines.len() {
            let line = scan_code(lines[index]);
            depth += line.brace_delta + line.paren_delta;
            on_error_path[index] = true;
            index += 1;
            if depth <= outer_depth {
                break;
            }
        }
    }
    on_error_path
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a glob whose `*` crossed a path separator would let one document
    /// reference or one lane write-glob claim a whole subtree, which is the
    /// scope-that-does-not-exist defect both readers of this exist to catch.
    #[test]
    fn a_glob_star_stays_inside_one_path_segment_and_a_double_star_does_not() {
        assert!(glob_match("docs/*.md", "docs/a.md"));
        assert!(!glob_match("docs/*.md", "docs/nested/a.md"));
        assert!(glob_match("docs/**/*.md", "docs/nested/deeper/a.md"));
        assert!(glob_match("docs/**", "docs/a.md"));
        assert!(glob_match("a?c.rs", "abc.rs"));
        assert!(!glob_match("a?c.rs", "ac.rs"));
        assert!(glob_match(
            "vyre-*/src/**/*.rs",
            "vyre-libs/src/graph/mod.rs"
        ));
        assert!(!glob_match("vyre-*/src/**/*.rs", "vyre-libs/tests/a.rs"));
    }

    /// WHY: a substring search for a word is the defect that let the CI matrix
    /// gate pass with a missing axis, because `stable` matches `unstable`.
    #[test]
    fn a_word_search_does_not_match_inside_a_longer_word() {
        assert!(contains_word("toolchain: stable", "stable"));
        assert!(!contains_word("toolchain: unstable", "stable"));
        assert!(!contains_word("toolchain: stables", "stable"));
        assert!(contains_word("[stable, beta]", "stable"));
    }

    /// WHY: `under` decides what a ratchet covers. A prefix comparison without
    /// the separator counts `vyre-libs-extra/src/a.rs` as part of `vyre-libs`,
    /// which silently widens a pinned count.
    #[test]
    fn a_scope_root_matches_only_its_own_directory() {
        assert!(under(Path::new("vyre-libs/src/a.rs"), "vyre-libs"));
        assert!(under(Path::new("vyre-libs/src/a.rs"), "vyre-libs/src"));
        assert!(!under(Path::new("vyre-libs-extra/src/a.rs"), "vyre-libs"));
        assert!(under(Path::new("anything"), "."));
    }

    /// WHY: the whole point of this module is that a failed scan cannot read as
    /// a clean tree. A scope naming a path that does not exist has to be an
    /// error, not an empty file list.
    #[test]
    fn a_missing_scan_path_is_an_error_and_not_an_empty_scan() {
        let root = std::env::temp_dir();
        let tree = Tree {
            root: root.clone(),
            paths: vec![PathBuf::from("src/a.rs")],
            absent: Vec::new(),
        };
        let error = tree
            .rust(&["a-directory-that-does-not-exist"])
            .expect_err("a missing scan path is fatal");
        assert!(error.message.contains("scan path does not exist"));
    }

    /// WHY: a production rule that must skip test code has to know where the test
    /// item ends. A nested module inside it, or a brace inside a string, must not
    /// end the extent early, and nothing above the attribute may be swept in.
    #[test]
    fn a_cfg_test_item_covers_itself_and_nothing_above_it() {
        let text = "fn production() {}\n#[cfg(test)]\nmod tests {\n    mod inner {\n        fn helper() {}\n    }\n    fn probe() {\n        let brace = \"}\";\n    }\n}\nfn after() {}\n";
        let lines: Vec<&str> = text.lines().collect();
        let flags = cfg_test_lines(&lines);
        assert_eq!(
            flags,
            vec![false, true, true, true, true, true, true, true, true, true, false],
            "the extent is the attribute through the item's closing brace"
        );
    }

    /// WHY: this workspace marks test code five ways, and only three of them mean
    /// test-only. `any(test, feature = "cpu-parity")` sits on 864 items that ship
    /// whenever that feature is on, so reading it as test code would hide real
    /// production panics and real hot-path cost from every rule built on this
    /// scan. A bare `#[test]` beside shipped code is the opposite case: it never
    /// ships, wherever it is written. The attribute, not the word `test`, decides.
    #[test]
    fn only_an_attribute_that_requires_test_marks_test_code() {
        for attribute in [
            "#[cfg(test)]",
            "    #[cfg(all(test, feature = \"subgroup-ops\"))]",
            "#[cfg(all(feature = \"gpu\", test))]",
            "#[test]",
            "#[tokio::test]",
        ] {
            let lines = vec![attribute, "mod tests {", "}", "fn after() {}"];
            assert_eq!(
                cfg_test_lines(&lines),
                vec![true, true, true, false],
                "{attribute} gates its item on the harness"
            );
        }
        for attribute in [
            "#[cfg(any(test, feature = \"cpu-parity\"))]",
            "#[cfg(not(test))]",
            "#[cfg(feature = \"test-utils\")]",
            "#[cfg_attr(test, allow(dead_code))]",
        ] {
            let lines = vec![attribute, "mod shipped {", "}", "fn after() {}"];
            assert_eq!(
                cfg_test_lines(&lines),
                vec![false, false, false, false],
                "{attribute} still compiles outside the harness"
            );
        }
    }

    /// WHY: a bound spelled on an earlier line of the same method chain is the
    /// whole reason the read-all rule had a module-wide exemption, under which an
    /// unbounded read added beside the bounded one reported nothing. The statement
    /// is the unit that carries the bound, so it is the unit the rule reads.
    ///
    /// What this does not catch: a bound applied to a different value in the same
    /// statement, or one applied in a helper the statement calls.
    #[test]
    fn a_statement_carries_every_line_of_its_own_chain() {
        let lines = vec![
            "    let mut bytes = Vec::new();",
            "    Read::by_ref(&mut file)",
            "        .take(limit)",
            "        .read_to_end(&mut bytes)?;",
            "    file.read_to_end(&mut bytes)?;",
        ];

        assert_eq!(
            statement_at(&lines, 3),
            "Read::by_ref(&mut file) .take(limit) .read_to_end(&mut bytes)?;"
        );
        assert_eq!(statement_at(&lines, 4), "file.read_to_end(&mut bytes)?;");
        assert_eq!(statement_at(&lines, 0), "let mut bytes = Vec::new();");
        assert_eq!(statement_at(&[], 7), "");
    }

    /// WHY: the statement filter is what turns the read-all rule from a
    /// per-module waiver into a per-expression judgement, so both directions are
    /// proved: a chain that carries its bound is not an occurrence, and one that
    /// does not is, in the same file.
    #[test]
    fn a_statement_filter_keeps_only_the_occurrences_its_chain_condemns() {
        let (_directory, root) = crate::gates::fixture_checkout::checkout(&[(
            "src/read.rs",
            "fn bounded(file: &mut File, bytes: &mut Vec<u8>) {\n    Read::by_ref(file)\n        .take(64)\n        .read_to_end(bytes);\n}\nfn unbounded(file: &mut File, bytes: &mut Vec<u8>) {\n    file.read_to_end(bytes);\n}\n",
        )]);

        let tree = Tree::open(&root).expect("fixture tree");
        let report = ratchet(
            &tree,
            &Rule {
                roots: &["src"],
                skip: &|_| false,
                line: &|line| line.contains("read_to_end"),
                reviewed: &[],
                reviewed_line: None,
                statement: Some(&|statement| !statement.contains(".take(")),
                message: "unbounded read",
                fix: "cap the read",
                unreviewed_message: "unbounded read nobody signed off",
                unreviewed_fix: "cap the read",
            },
        )
        .expect("rule runs");

        let lines: Vec<Option<u32>> = report.findings.iter().map(|finding| finding.line).collect();
        assert_eq!(
            lines,
            vec![Some(7), Some(7)],
            "only the chain without a bound counts"
        );
    }

    /// WHY: a missing root was already fatal because a rule scanning nothing
    /// reports success forever, and a skip predicate that excludes every file in
    /// scope produces the same empty scan with an existing root. That is how a
    /// rule survives the directory it guards being renamed under the exemption:
    /// the root still exists, every file now matches the skip, and the pinned
    /// count stays at zero. Both directions are proved on the same fixture.
    ///
    /// What this does not catch: a skip that leaves one file whose subject moved
    /// into the files it excludes.
    #[test]
    fn a_rule_that_skips_every_file_in_scope_is_fatal() {
        let (_directory, root) = crate::gates::fixture_checkout::checkout(&[(
            "src/read.rs",
            "fn unbounded(file: &mut File, bytes: &mut Vec<u8>) {\n    file.read_to_end(bytes);\n}\n",
        )]);
        let tree = Tree::open(&root).expect("fixture tree");
        let rule = |skip: &'static dyn Fn(&Path) -> bool| Rule {
            roots: &["src"],
            skip,
            line: &|line| line.contains("read_to_end"),
            reviewed: &[],
            reviewed_line: None,
            statement: None,
            message: "unbounded read",
            fix: "cap the read",
            unreviewed_message: "unbounded read nobody signed off",
            unreviewed_fix: "cap the read",
        };

        let error = ratchet(&tree, &rule(&|_| true)).expect_err("an empty scan is fatal");
        assert_eq!(
            error.message, "the rule under src skips all 1 file(s) in scope",
            "the error names the scope and how many files the skip removed"
        );

        let report = ratchet(&tree, &rule(&|_| false)).expect("one file left in scope");
        assert_eq!(
            report.findings.len(),
            2,
            "the same rule reports the occurrence once the skip stops covering it"
        );
    }
}
