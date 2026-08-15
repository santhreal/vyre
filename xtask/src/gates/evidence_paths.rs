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
fn cited_conform_paths(text: &str) -> Vec<(String, u32)> {
    let mut found = Vec::new();
    for (number, line) in crate::gates::scan::numbered(text) {
        let mut from = 0;
        while let Some(at) = line[from..].find("conform/") {
            let start = from + at;
            let end = line[start..]
                .find(|character| character == ':' || character == '"')
                .map_or(line.len(), |offset| start + offset);
            let candidate = &line[start..end];
            if let Some(cut) = candidate.find(".rs") {
                found.push((candidate[..cut + 3].to_string(), number));
            }
            from = start + "conform/".len();
        }
    }
    found
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
                .map(String::as_str)
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
    use super::*;

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
}
