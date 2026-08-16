//! Every filesystem path cited inside release evidence resolves on disk.
//!
//! An artifact that is internally consistent and cites deleted code certifies
//! against fiction, and nothing else reads the paths inside one. The oracle here
//! is stat, plus git's ignore status: a cited path that exists but is gitignored
//! reaches no other reader, so it is unverifiable and is reported separately.
//!
//! The extension vocabulary is derived from the tree at run time. The first `.cu`
//! file added to the workspace extends this gate in the same commit, with nobody
//! editing this file, and a version string or an op id ends in nothing the tree
//! uses as an extension so neither reads as a citation.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Where release evidence lives.
const EVIDENCE_DIR: &str = "release/evidence";

/// One string leaf that looks like a filesystem path.
struct Citation {
    /// The artifact that carries it.
    artifact: PathBuf,
    /// The route to the string, so two citations on one object stay distinct.
    location: String,
    /// The cited value.
    path: String,
}

/// Cited paths in release evidence resolve, and reach a reader.
pub struct EvidencePaths;

impl Gate for EvidencePaths {
    fn name(&self) -> &'static str {
        "evidence-paths"
    }

    fn help(&self) -> &'static str {
        "paths cited inside release evidence that are missing or gitignored"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        if !tree.exists(EVIDENCE_DIR) {
            return Err(GateError::new(
                format!("{EVIDENCE_DIR} does not exist"),
                "generate release evidence before judging it, or repoint this gate",
            ));
        }

        let extensions = extension_vocabulary(&tree);
        if extensions.is_empty() {
            return Err(GateError::new(
                "no file extension occurs anywhere in the tree, so the citation vocabulary \
                 cannot be derived",
                "point the gate at the workspace being certified",
            ));
        }

        let mut report = Report::clean();
        let mut citations = Vec::new();
        let mut artifacts = tree
            .scope(&[EVIDENCE_DIR], &["json"])?
            .into_iter()
            .collect::<Vec<_>>();
        artifacts.sort();
        for artifact in &artifacts {
            let text = tree.read(artifact)?;
            let document: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
                GateError::new(
                    format!("{} is not valid JSON: {error}", artifact.display()),
                    "regenerate the artifact with its owning release-evidence command",
                )
            })?;
            collect(&document, &mut Vec::new(), artifact, &extensions, &mut citations);
        }
        report.note(format!(
            "{} citation(s) across {} artifact(s)",
            citations.len(),
            artifacts.len()
        ));

        let mut present = BTreeSet::new();
        for citation in &citations {
            match resolve(&tree, citation) {
                Some(resolved) => {
                    present.insert(resolved);
                }
                None => report.find(Finding::in_file(
                    citation.artifact.clone(),
                    format!(
                        "{} cites a path that is not on disk: {}",
                        citation.location, citation.path
                    ),
                    "regenerate the artifact from the current tree with its owning \
                     release-evidence command, or delete the citation if the evidence is \
                     obsolete; never hand-edit a generated artifact",
                )),
            }
        }

        for path in ignored_paths(&present)? {
            report.find(Finding::in_file(
                path,
                "the cited path exists but is gitignored, so no other reader can verify it",
                "commit the path, or stop citing it; evidence must cite what reaches a reader",
            ));
        }

        Ok(report)
    }
}

/// Every conformance test an invariant descriptor cites exists.
///
/// A broken pointer means a test was renamed or deleted without updating the
/// invariant that cites it. Recording the pointer as known-missing is not an
/// option: an invariant that cites nothing asserts nothing.
pub struct InvariantPaths;

impl Gate for InvariantPaths {
    fn name(&self) -> &'static str {
        "invariant-paths"
    }

    fn help(&self) -> &'static str {
        "conformance tests cited by invariant descriptors that do not exist"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        const INVARIANTS: &str = "vyre-spec/src/invariants.rs";
        /// The path in a doc comment showing the citation form, not a citation.
        const EXAMPLE: &str = "conform/tests/<file>.rs";

        let tree = Tree::open(&ctx.root)?;
        let text = tree.read(INVARIANTS)?;
        let mut report = Report::clean();
        let mut cited = BTreeSet::new();
        for (line, number) in cited_conform_paths(&text) {
            if line == EXAMPLE {
                continue;
            }
            if !cited.insert(line.clone()) {
                continue;
            }
            if !tree.absolute(&line).is_file() {
                report.find(Finding::at(
                    INVARIANTS,
                    number,
                    format!("the invariant cites a conformance test that does not exist: {line}"),
                    "restore the test, or delete the invariant entry; a citation of nothing \
                     documents a broken pointer instead of asserting the invariant",
                ));
            }
        }
        report.note(format!("{} cited conformance test(s)", cited.len()));
        Ok(report)
    }
}

/// Conformance test paths cited in the text, with the line each sits on.
///
/// A match counts only where the path begins. `conform/` also occurs inside
/// `conform/vyre-conform/tests/...`, and reading from there invented the
/// citation `conform/tests/invariants.rs`, which no tree has ever held: the
/// gate then reported a broken pointer for every descriptor that spelled its
/// path correctly.
fn cited_conform_paths(text: &str) -> Vec<(String, u32)> {
    let mut found = Vec::new();
    for (number, line) in crate::gates::scan::numbered(text) {
        let mut from = 0;
        while let Some(at) = line[from..].find("conform/") {
            let start = from + at;
            let end = line[start..]
                .find(|character| character == ':' || character == '"')
                .map_or(line.len(), |offset| start + offset);
            if begins_a_path(line, start) {
                let candidate = &line[start..end];
                if let Some(cut) = candidate.find(".rs") {
                    found.push((candidate[..cut + 3].to_string(), number));
                }
            }
            from = end.max(start + 1);
        }
    }
    found
}

/// Whether the offset starts a path rather than sitting inside a longer one.
fn begins_a_path(line: &str, start: usize) -> bool {
    match line[..start].chars().next_back() {
        None => true,
        Some(character) => {
            !(character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '/' | '.'))
        }
    }
}

/// Extensions that occur among the tree's own files, lowercased.
fn extension_vocabulary(tree: &Tree) -> BTreeSet<String> {
    tree.paths()
        .iter()
        .filter_map(|path| path.extension().and_then(|value| value.to_str()))
        .filter(|extension| extension.chars().all(|c| c.is_ascii_alphanumeric()))
        .map(str::to_ascii_lowercase)
        .collect()
}

/// Every string leaf shaped like a path, at any depth.
///
/// Shape is three conditions: no whitespace, a last component whose extension the
/// tree uses, and either a slash or a nearest enclosing key named `path`, because
/// a bare sibling filename is cited that way. Restricting this to a `findings`
/// array, or to members of a top-level array, or to the key `path`, each left
/// most of the citations unread.
fn collect(
    value: &serde_json::Value,
    route: &mut Vec<String>,
    artifact: &Path,
    extensions: &BTreeSet<String>,
    out: &mut Vec<Citation>,
) {
    match value {
        serde_json::Value::String(text) => {
            let key = route
                .iter()
                .rev()
                .find(|step| !step.starts_with('['))
                .map(|step| step.trim_start_matches('.'))
                .unwrap_or_default();
            if looks_like_path(text, key, extensions) {
                out.push(Citation {
                    artifact: artifact.to_path_buf(),
                    location: route.join("").trim_start_matches('.').to_string(),
                    path: text.clone(),
                });
            }
        }
        serde_json::Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                route.push(format!("[{index}]"));
                collect(item, route, artifact, extensions, out);
                route.pop();
            }
        }
        serde_json::Value::Object(fields) => {
            for (name, item) in fields {
                route.push(format!(".{name}"));
                collect(item, route, artifact, extensions, out);
                route.pop();
            }
        }
        _ => {}
    }
}

/// Whether a string leaf is a citation.
fn looks_like_path(text: &str, key: &str, extensions: &BTreeSet<String>) -> bool {
    if text.is_empty() || text.chars().any(char::is_whitespace) {
        return false;
    }
    if !(text.contains('/') || key == "path") {
        return false;
    }
    let last = text.trim_end_matches('/').rsplit('/').next().unwrap_or(text);
    let Some((_, extension)) = last.rsplit_once('.') else {
        return false;
    };
    extensions.contains(&extension.to_ascii_lowercase())
}

/// The cited path on disk, under the three live conventions.
///
/// The artifact-directory fallback is not decoration: without it the release gate
/// log reports three citations of its own siblings as missing, and a gate with
/// false positives gets muted.
fn resolve(tree: &Tree, citation: &Citation) -> Option<PathBuf> {
    let cited = Path::new(&citation.path);
    if cited.is_absolute() {
        return cited.exists().then(|| cited.to_path_buf());
    }
    let from_root = tree.absolute(cited);
    if from_root.exists() {
        return Some(from_root);
    }
    let beside = tree
        .absolute(&citation.artifact)
        .parent()
        .map(|directory| directory.join(cited))?;
    beside.exists().then_some(beside)
}

/// Which of the present paths git would ignore.
///
/// Ignore status is per repository and the citations span more than one checkout,
/// so the paths are grouped by their repository before asking. Outside a checkout
/// the question is not answerable and the path is left alone rather than guessed
/// at.
fn ignored_paths(present: &BTreeSet<PathBuf>) -> Result<Vec<PathBuf>, GateError> {
    let mut by_repository: BTreeMap<PathBuf, Vec<&PathBuf>> = BTreeMap::new();
    let mut repository_of: BTreeMap<PathBuf, Option<PathBuf>> = BTreeMap::new();
    for path in present {
        let Some(directory) = path.parent() else {
            continue;
        };
        let repository = repository_of
            .entry(directory.to_path_buf())
            .or_insert_with(|| toplevel(directory));
        if let Some(repository) = repository.clone() {
            by_repository.entry(repository).or_default().push(path);
        }
    }

    let mut ignored = Vec::new();
    for (repository, paths) in by_repository {
        let mut child = Command::new("git")
            .arg("-C")
            .arg(&repository)
            .args(["check-ignore", "--stdin"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| {
                GateError::new(
                    format!("cannot run git check-ignore in {}: {error}", repository.display()),
                    "install git, or run the gate inside a checkout",
                )
            })?;
        {
            let stdin = child.stdin.as_mut().ok_or_else(|| {
                GateError::new(
                    "git check-ignore did not accept input",
                    "run the gate inside a checkout",
                )
            })?;
            for path in &paths {
                writeln!(stdin, "{}", path.display()).map_err(|error| {
                    GateError::new(
                        format!("cannot write to git check-ignore: {error}"),
                        "run the gate inside a checkout",
                    )
                })?;
            }
        }
        let output = child.wait_with_output().map_err(|error| {
            GateError::new(
                format!("git check-ignore failed in {}: {error}", repository.display()),
                "run the gate inside a checkout",
            )
        })?;
        // One means nothing matched, which is the clean answer. Anything above
        // one is git failing, and a failure that reads as clean is the defect
        // this gate exists to catch.
        match output.status.code() {
            Some(0 | 1) => {}
            other => {
                return Err(GateError::new(
                    format!(
                        "git check-ignore exited with {:?} in {}",
                        other,
                        repository.display()
                    ),
                    "resolve the git failure; an unanswerable ignore status must not read \
                     as a clean tree",
                ))
            }
        }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if !line.is_empty() {
                ignored.push(PathBuf::from(line));
            }
        }
    }
    ignored.sort();
    ignored.dedup();
    Ok(ignored)
}

/// The repository a directory belongs to, if any.
fn toplevel(directory: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(["rev-parse", "--show-toplevel"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!text.is_empty()).then(|| PathBuf::from(text))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::Command;

    use tempfile::TempDir;

    use super::*;
    use crate::gates::fixture_checkout;

    fn vocabulary() -> BTreeSet<String> {
        ["rs", "toml", "json", "md"]
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// WHY: the shape rule is what keeps the gate off version strings, op ids and
    /// schema ids. If any of those read as a citation the gate reports paths that
    /// were never paths, and a gate with false positives gets muted.
    #[test]
    fn a_version_or_op_id_is_not_a_citation() {
        let extensions = vocabulary();
        assert!(!looks_like_path("1.2.0", "version", &extensions));
        assert!(!looks_like_path(
            "vyre-primitives::hardware",
            "op",
            &extensions
        ));
        assert!(!looks_like_path(
            "vyre-conform-input-envelope-v1",
            "schema",
            &extensions
        ));
    }

    /// WHY: the conformance crate lives at `conform/vyre-conform`, so every
    /// correct citation contains `conform/` twice. Reading from the second
    /// occurrence invented `conform/tests/invariants.rs`, a path no tree has
    /// held, and the gate reported a broken pointer for a descriptor that was
    /// right. One citation per cited path, from where the path starts.
    #[test]
    fn a_nested_crate_path_is_read_once_from_its_start() {
        let text = "        \"conform/vyre-conform/tests/invariants.rs::deterministic\",\n";
        assert_eq!(
            cited_conform_paths(text),
            vec![("conform/vyre-conform/tests/invariants.rs".to_string(), 1)]
        );
    }

    /// WHY: two descriptors on one line, and the second must still be read.
    #[test]
    fn every_citation_on_a_line_is_read() {
        let text = "a \"conform/vyre-conform/tests/one.rs::x\" b \"conform/vyre-conform/tests/two.rs::y\"\n";
        let cited: Vec<String> = cited_conform_paths(text)
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        assert_eq!(
            cited,
            [
                "conform/vyre-conform/tests/one.rs".to_string(),
                "conform/vyre-conform/tests/two.rs".to_string()
            ]
        );
    }

    /// WHY: the two citation forms are a path with a slash and a bare filename
    /// under the key `path`. Dropping either one is how the shell versions of this
    /// rule read four percent of the defect.
    #[test]
    fn both_citation_forms_are_read() {
        let extensions = vocabulary();
        assert!(looks_like_path("vyre-foundation/src/lib.rs", "source", &extensions));
        assert!(looks_like_path("backend-matrix.json", "path", &extensions));
        assert!(!looks_like_path("backend-matrix.json", "name", &extensions));
    }

    /// WHY: an extension the tree does not use is not a citation. This is what
    /// makes the vocabulary a run-time property of the tree rather than a
    /// hardcoded list that stops covering the next file type someone adds.
    #[test]
    fn an_unknown_extension_is_not_a_citation() {
        let extensions = vocabulary();
        assert!(!looks_like_path("kernels/reduce.cu", "source", &extensions));
        let mut extended = extensions.clone();
        extended.insert("cu".to_string());
        assert!(looks_like_path("kernels/reduce.cu", "source", &extended));
    }

    /// WHY: a citation has to stay addressable however deeply a schema nests it,
    /// and two citations on one object must not collapse into one location. The
    /// route is the whole point of the report being actionable.
    #[test]
    fn nested_citations_keep_distinct_routes() {
        let document: serde_json::Value = serde_json::from_str(
            r#"{"findings":[{"path":"a.rs","manifest":"b/c.toml"}],"root":"d/e.md"}"#,
        )
        .expect("the fixture parses");
        let mut found = Vec::new();
        collect(
            &document,
            &mut Vec::new(),
            Path::new("release/evidence/x.json"),
            &vocabulary(),
            &mut found,
        );
        let mut routes: Vec<String> = found.iter().map(|c| c.location.clone()).collect();
        routes.sort();
        assert_eq!(
            routes,
            vec![
                "findings[0].manifest".to_string(),
                "findings[0].path".to_string(),
                "root".to_string()
            ]
        );
    }

    /// A git checkout holding one evidence artifact, one Rust file and one
    /// manifest.
    ///
    /// Neither of those two is decoration. The citation vocabulary is the set of
    /// extensions the tree itself uses, so in a tree carrying no `.rs` and no
    /// `.toml` file a citation of either reads as prose and the gate reports
    /// nothing, for the right reason and with nothing proved.
    fn checkout(artifact: &str) -> (TempDir, PathBuf) {
        fixture_checkout::checkout(&[
            ("keep.rs", "fn keep() {}\n"),
            ("Cargo.toml", "[workspace]\n"),
            (&format!("{EVIDENCE_DIR}/artifact.json"), artifact),
        ])
    }

    /// Run one git step in `root`, failing the test when git does.
    fn git(root: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(root)
            .status()
            .expect("git is available");
        assert!(status.success(), "the fixture git step failed: {args:?}");
    }

    /// Every finding, as the file it names followed by its message.
    fn messages(report: &Report) -> String {
        report
            .findings
            .iter()
            .map(|finding| format!("{} {}", finding.named_file(), finding.message))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// WHY: the passing direction is what a stale artifact produces trivially, so
    /// the gate is only worth its run time if the failing direction is pinned. A
    /// refactor that stopped discovering citations, mis-parsed the JSON or
    /// resolved every path to some fallback would leave the green run green. The
    /// report has to name the artifact, the route and the path, because a
    /// citation nobody can locate is not actionable.
    #[test]
    fn a_citation_that_does_not_resolve_is_reported() {
        let (_temporary, root) = checkout(
            r#"{"findings":[{"path":"definitely/not/on/disk/anywhere.rs"}]}"#,
        );

        let report = EvidencePaths
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate runs");
        let reported = messages(&report);
        assert!(
            reported.contains("artifact.json"),
            "the artifact is named: {reported}"
        );
        assert!(
            reported.contains("findings[0].path"),
            "and the route inside it: {reported}"
        );
        assert!(
            reported.contains("definitely/not/on/disk/anywhere.rs"),
            "and the path that resolves to nothing: {reported}"
        );
    }

    /// WHY: existence and reachability are different questions with different
    /// oracles, `stat` and `git check-ignore`, and this branch never fires on the
    /// real tree, where no cited path is ignored. Without a fixture it would be
    /// code nobody has watched work and everybody trusts. The tracked file pins
    /// the other half: a committed path matching an ignore pattern is already in
    /// public history and must stay clean, which is why the question is asked of
    /// git's index rather than of the pattern.
    #[test]
    fn a_cited_path_that_is_gitignored_is_reported() {
        let (_temporary, root) = checkout("{}");
        fs::write(root.join(".gitignore"), "generated.rs\ntracked.rs\n")
            .expect("an ignore rule");
        fs::write(root.join("generated.rs"), "fn generated() {}\n").expect("an ignored file");
        fs::write(root.join("tracked.rs"), "fn tracked() {}\n").expect("a tracked file");
        git(&root, &["add", "-f", ".gitignore", "tracked.rs", "keep.rs"]);
        git(
            &root,
            &[
                "-c",
                "user.email=gate@vyre.test",
                "-c",
                "user.name=gate",
                "commit",
                "-qm",
                "fixture",
            ],
        );
        fs::write(
            root.join(EVIDENCE_DIR).join("artifact.json"),
            format!(
                r#"{{"files":[{{"path":"{}"}},{{"path":"{}"}}]}}"#,
                root.join("tracked.rs").display(),
                root.join("generated.rs").display()
            ),
        )
        .expect("an evidence artifact");

        let report = EvidencePaths
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate runs");
        let reported = messages(&report);
        assert!(
            reported.contains("gitignored"),
            "the reason is reachability, not absence: {reported}"
        );
        assert!(
            reported.contains("generated.rs"),
            "and the ignored path is named: {reported}"
        );
        assert!(
            !reported.contains("tracked.rs"),
            "a committed path stays clean however it is ignored: {reported}"
        );
    }

    /// WHY: discovery has been narrowed twice and both narrowings hid live
    /// defects. Reading one top-level array of objects with a `path` field hid 81
    /// of 634 citations, and reading the key `path` and nothing else hid 2775
    /// more under `manifest`, `artifact`, `workflow` and bare array members; nine
    /// of those were dead. So one absent path sits at each placement a narrower
    /// filter missed, each must be reported at its full route, and the count is
    /// asserted so a filter that reads the first and stops is red.
    #[test]
    fn a_citation_is_read_at_every_placement() {
        let (_temporary, root) = checkout(
            concat!(
                r#"{"path":"absent/on/the/root/object.rs","#,
                r#""subject":{"path":"absent/under/an/object.rs"},"#,
                r#""groups":[{"rows":[{"path":"absent/in/a/nested/array.rs"}]}],"#,
                r#""crate":{"manifest":"absent/crate/Cargo.toml"},"#,
                r#""sources":["absent/array/member.rs"]}"#
            ),
        );

        let report = EvidencePaths
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate runs");
        let reported = messages(&report);
        for (route, path) in [
            ("path", "absent/on/the/root/object.rs"),
            ("subject.path", "absent/under/an/object.rs"),
            ("groups[0].rows[0].path", "absent/in/a/nested/array.rs"),
            ("crate.manifest", "absent/crate/Cargo.toml"),
            ("sources[0]", "absent/array/member.rs"),
        ] {
            assert!(
                reported.contains(&format!("{route} cites a path that is not on disk: {path}")),
                "the citation at `{route}` is located: {reported}"
            );
        }
        assert_eq!(report.count(), 5, "every placement counts: {reported}");
        assert!(
            report.notes.iter().any(|note| note.contains("5 citation(s)")),
            "and the note states what was read: {:?}",
            report.notes
        );
    }

    /// WHY: the discovery rule is a shape rule, so it needs a floor on the other
    /// side. A gate with false positives gets muted, and muting this one restores
    /// the state it was built to end. Version strings, schema ids, operation ids,
    /// fingerprints, recorded commands and ratios all live beside real citations
    /// and none of them names a file. The separator is the extension vocabulary
    /// the tree itself uses, so a loosening that starts matching any dotted token
    /// turns this red.
    #[test]
    fn a_string_that_names_no_file_is_not_a_citation() {
        let (_temporary, root) = checkout(
            concat!(
                r#"{"version":"1.2.0","#,
                r#""schema_id":"vyre-conform-input-envelope-v1","#,
                r#""op":"vyre-primitives::hardware::subgroup_shuffle","#,
                r#""fingerprint":"source-tree-v1:f42685f0","#,
                r#""command":"git grep -nE \"trait CpuOp\" -- absent/crate/src","#,
                r#""ratio":"0.0/1.0"}"#
            ),
        );

        let report = EvidencePaths
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate runs");
        assert_eq!(
            report.count(),
            0,
            "no string here names a file: {}",
            messages(&report)
        );
        assert!(
            report.notes.iter().any(|note| note.contains("0 citation(s)")),
            "and none was read as one: {:?}",
            report.notes
        );
    }
}
