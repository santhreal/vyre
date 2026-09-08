//! Every integration-test file runs, and every test case runs once.
//!
//! Integration tests in this workspace are grouped: a package declares one
//! `[[test]]` harness per required-feature set and each former target is a
//! module of one. That is what keeps roughly ninety test binaries in a tree that
//! would otherwise link over thirteen hundred, and it moves the wiring out of
//! cargo's discovery and into the manifest, where a mistake is silent.
//!
//! Three mistakes are possible and none of them fails a build. A file added
//! under `tests/` that no harness declares compiles nowhere and runs nowhere:
//! with `autotests = false` cargo does not pick it up, nothing references it,
//! and the suite stays green while the assertions in it are gone. A package that
//! leaves autodiscovery on gets the opposite: the file links its own binary as
//! well as running inside a harness, so one test reports twice and the grouping
//! is undone one file at a time. A file declaring test cases that two harnesses
//! both include runs its cases twice under two target names, which is the same
//! double count reached the long way round.
//!
//! The file set is read from the tree on every run and the target set from the
//! manifests, so a test added tomorrow is judged by the same rule rather than by
//! a list somebody has to remember to extend.
//!
//! Two kinds of file under `tests/` are not integration-test files and are
//! excluded, both derived from source rather than named here. Unit-test material
//! a `src/` module includes through `#[path]` is compiled by the library target;
//! `test-material-placement` owns where that may live. A fixture a test feeds to
//! a compiler harness by glob, as `trybuild` does with `tests/ui/*.rs`, is data
//! read at run time and is never a module of anything.
//!
//! # What it does not catch
//!
//! Reachability is resolved through `mod` declarations in source text, so a
//! module reached by `include!` is not followed. A target whose root sits in a
//! subdirectory of `tests/` is credited with that whole subdirectory, because
//! rustc resolves those declarations and rejects the ones with no file.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{Member, Tree};

/// The grouping decision record, which states which files stay their own target.
const DECISIONS: &str = "xtask/test-harness-isolation.toml";

/// Every integration-test file is declared by a cargo test target, once.
pub struct TestTargetMembership;

impl crate::gate::GateBehavior for TestTargetMembership {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }

        let isolated = isolated(&tree)?;
        let members = tree.member_manifests()?;
        let mut packages_with_tests = 0_usize;
        let mut files_judged = 0_usize;
        for member in &members {
            let directory = format!("{}/tests", member.path);
            let all = rust_files_under(&tree, &directory);
            if all.is_empty() {
                continue;
            }
            let excluded = excluded(&tree, member, &directory)?;
            let files: Vec<String> = all
                .into_iter()
                .filter(|path| !excluded.contains(path))
                .collect();
            if files.is_empty() {
                continue;
            }
            packages_with_tests += 1;
            files_judged += files.len();
            judge(&tree, member, &directory, &files, &isolated, &mut report)?;
        }
        report.cover_complete("integration test files", files_judged);
        report.note(format!(
            "{packages_with_tests} package(s) carry integration tests"
        ));
        if packages_with_tests == 0 {
            return Err(GateError::new(
                "no workspace member carries an integration test",
                "run this gate inside the workspace checkout; a scan over an empty test set \
                 reports success forever",
            ));
        }
        Ok(report)
    }
}

/// Every judgment one package's test directory carries.
fn judge(
    tree: &Tree,
    member: &Member,
    directory: &str,
    files: &[String],
    isolated: &BTreeSet<String>,
    report: &mut Report,
) -> Result<(), GateError> {
    let manifest = format!("{}/Cargo.toml", member.path);
    let discovers = member
        .manifest
        .get("package")
        .and_then(|package| package.get("autotests"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(true);
    if discovers {
        report.find(Finding::at(
            PathBuf::from(&manifest),
            1,
            format!(
                "`{}` carries integration tests and leaves autotests on",
                member.name
            ),
            "declare `autotests = false` under [package] and give every test file a target; \
             discovery links a second binary for a file a harness already runs",
        ));
    }

    for entry in member
        .manifest
        .get("test")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let Some(name) = entry.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let Some(path) = entry.get("path").and_then(toml::Value::as_str) else {
            report.find(Finding::at(
                PathBuf::from(&manifest),
                1,
                format!("test target `{name}` declares no path"),
                "give every [[test]] row an explicit `path`; a grouped harness is not discovered",
            ));
            continue;
        };
        if !tree.has(&format!("{}/{path}", member.path)) {
            report.find(Finding::at(
                PathBuf::from(&manifest),
                1,
                format!("test target `{name}` points at `{path}`, which the checkout lacks"),
                "repoint the row at the harness file, or delete the row",
            ));
        }
    }

    let ownership = ownership(tree, member);
    let owners = &ownership.owners;

    for file in files {
        let claimants = owners.get(file).map(Vec::as_slice).unwrap_or_default();
        if claimants.is_empty() {
            report.find(Finding::at(
                PathBuf::from(file),
                1,
                "no test target declares this file".to_string(),
                format!(
                    "declare it as a module of the harness for its required features in \
                     {directory}, or give it a [[test]] row in {manifest}; a file no target \
                     names compiles nowhere and runs nowhere"
                ),
            ));
        } else if claimants.len() > 1 && declares_cases(tree, file)? {
            report.find(Finding::at(
                PathBuf::from(file),
                1,
                format!(
                    "{} test targets declare this file: {}",
                    claimants.len(),
                    claimants.join(", ")
                ),
                "keep a file that declares test cases in one target; a shared fixture \
                 declares none of its own and may be included by several",
            ));
        }
    }

    for path in isolated {
        if !path.starts_with(&format!("{directory}/")) {
            continue;
        }
        let shared = owners.get(path).into_iter().flatten().any(|name| {
            !ownership
                .targets
                .iter()
                .any(|target| target.name == *name && dedicated_to(target, path))
        });
        if shared {
            report.find(Finding::at(
                PathBuf::from(path),
                1,
                "an isolated test is compiled into a shared harness".to_string(),
                format!(
                    "give it a [[test]] row of its own in {manifest} and remove the module \
                     declaration, or change its row in {DECISIONS} to `grouped`"
                ),
            ));
        }
    }
    Ok(())
}

/// Which cargo test target compiles which file, for one package.
///
/// Every gate asking what a workflow's `--test` selector reaches needs this, and
/// asking it of the manifest alone stopped answering when the files became
/// modules of a harness rather than targets of their own.
pub struct Ownership {
    /// Every declared test target of the package.
    pub targets: Vec<TestTarget>,
    /// The targets that compile each file, by tree-relative path.
    pub owners: BTreeMap<String, Vec<String>>,
}

/// One declared `[[test]]` target.
pub struct TestTarget {
    /// The name a `--test` argument spells.
    pub name: String,
    /// Tree-relative path of the target root.
    pub root: String,
    /// Features the manifest requires to compile it.
    pub required_features: BTreeSet<String>,
}

/// Every target a package declares, with the files each one compiles.
#[must_use]
pub fn ownership(tree: &Tree, member: &Member) -> Ownership {
    let mut targets = Vec::new();
    let mut owners: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in member
        .manifest
        .get("test")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let Some(name) = entry.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let relative = entry
            .get("path")
            .and_then(toml::Value::as_str)
            .map_or_else(|| format!("tests/{name}.rs"), str::to_string);
        let root = format!("{}/{relative}", member.path);
        if !tree.has(&root) {
            continue;
        }
        for reached in reachable(tree, &root) {
            owners.entry(reached).or_default().push(name.to_string());
        }
        targets.push(TestTarget {
            name: name.to_string(),
            root,
            required_features: crate::toml_text::string_array(entry.get("required-features"))
                .into_iter()
                .collect(),
        });
    }
    Ownership { targets, owners }
}

/// Every `.rs` file under a directory, at any depth.
fn rust_files_under(tree: &Tree, directory: &str) -> Vec<String> {
    let prefix = format!("{directory}/");
    tree.paths()
        .iter()
        .filter_map(|path| path.to_str())
        .filter(|path| path.starts_with(&prefix) && path.ends_with(".rs"))
        .map(str::to_string)
        .collect()
}

/// Every file under `tests/` that is not an integration-test file.
///
/// Both classes are read out of the package's own source: material a `src/`
/// module includes, and a fixture directory a test hands to a compiler harness
/// as a glob.
fn excluded(tree: &Tree, member: &Member, directory: &str) -> Result<BTreeSet<String>, GateError> {
    let mut excluded = BTreeSet::new();
    let source = format!("{}/src", member.path);
    let included_prefix = format!("{directory}/");
    for file in rust_files_under(tree, &source) {
        let text = tree.read(&file)?;
        let base = parent_of(&file);
        for path in attribute_paths(&text) {
            let resolved = join(&base, &path);
            if resolved.starts_with(&included_prefix) {
                excluded.extend(reachable(tree, &resolved));
            }
        }
    }
    for file in rust_files_under(tree, directory) {
        let text = tree.read(&file)?;
        for glob in fixture_globs(&text) {
            let Some(prefix) = glob.split('*').next() else {
                continue;
            };
            let prefix = join(&member.path, prefix);
            for candidate in rust_files_under(tree, prefix.trim_end_matches('/')) {
                excluded.insert(candidate);
            }
        }
    }
    Ok(excluded)
}

/// Whether a file declares a test case rustc will link into a target.
///
/// A `#[test]` inside a `macro_rules!` body declares nothing on its own: the
/// case exists once per expansion, in whichever file invokes the macro. Counting
/// those would report every shared assertion macro as a duplicated test.
fn declares_cases(tree: &Tree, path: &str) -> Result<bool, GateError> {
    let text = tree.read(path)?;
    let mut depth = 0_i32;
    let mut in_macro = false;
    for line in text.lines() {
        if !in_macro && line.contains("macro_rules!") {
            in_macro = true;
            depth = 0;
        }
        if in_macro {
            depth += i32::try_from(line.matches('{').count()).unwrap_or(0);
            depth -= i32::try_from(line.matches('}').count()).unwrap_or(0);
            if depth <= 0 {
                in_macro = false;
            }
            continue;
        }
        let trimmed = line.trim();
        if trimmed.starts_with("#[test]") || trimmed.starts_with("#[tokio::test]") {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Every file a test target compiles, the root included.
///
/// Module declarations are followed through `#[path = "..."]` when one is given
/// and through cargo's own layout when it is not: `mod name;` beside a root is
/// `name.rs` or `name/mod.rs`, resolved against the declaring file's directory.
fn reachable(tree: &Tree, root: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut queue = vec![root.to_string()];
    // A target root in a subdirectory of `tests/` compiles that subdirectory's
    // module tree. rustc resolves those declarations and rejects one with no
    // file, so crediting the tree cannot hide a file that never compiles.
    if let Some(directory) = enclosing_module_directory(root) {
        seen.extend(rust_files_under(tree, &directory));
    }
    while let Some(current) = queue.pop() {
        if !seen.insert(current.clone()) && current != root {
            continue;
        }
        let Ok(text) = tree.read(&current) else {
            continue;
        };
        let base = parent_of(&current);
        for declared in declarations(&text) {
            let candidates = match declared {
                Declaration::Explicit(path) => vec![join(&base, &path)],
                Declaration::Named(name) => vec![
                    join(&base, &format!("{name}.rs")),
                    join(&base, &format!("{name}/mod.rs")),
                ],
            };
            for candidate in candidates {
                if tree.has(&candidate) {
                    if !seen.contains(&candidate) {
                        queue.push(candidate);
                    }
                    break;
                }
            }
        }
    }
    seen
}

/// The directory a target root owns, when the root is not directly under `tests`.
fn enclosing_module_directory(root: &str) -> Option<String> {
    let directory = parent_of(root);
    if directory.rsplit('/').next() == Some("tests") {
        return None;
    }
    Some(directory)
}

/// Whether a target exists for one isolated file rather than for a group.
///
/// The file itself as the target root is the plain case. A root inside a
/// subdirectory of `tests/` is the other: such a target compiles that
/// subdirectory and nothing else, so an isolated file under it shares a process
/// only with its own module family. That is what isolation asks for, and a
/// fixture whose registration must reach several files in one pin has no other
/// shape available. A root directly under `tests/` is a grouped harness and
/// never satisfies an isolated row, whatever it declares.
fn dedicated_to(target: &TestTarget, path: &str) -> bool {
    if target.root == *path {
        return true;
    }
    enclosing_module_directory(&target.root)
        .is_some_and(|directory| path.starts_with(&format!("{directory}/")))
}

/// One module declaration, by explicit path or by name.
enum Declaration {
    /// `#[path = "..."] mod name;`
    Explicit(String),
    /// `mod name;`
    Named(String),
}

/// Every module declaration in one file, in source order.
fn declarations(text: &str) -> Vec<Declaration> {
    let mut declarations = Vec::new();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(path) = attribute_path(line) {
            pending = Some(path);
            continue;
        }
        if line.starts_with("//") {
            continue;
        }
        let Some(name) = module_name(line) else {
            if !line.is_empty() && !line.starts_with('#') {
                pending = None;
            }
            continue;
        };
        match pending.take() {
            Some(path) => declarations.push(Declaration::Explicit(path)),
            None => declarations.push(Declaration::Named(name)),
        }
    }
    declarations
}

/// Every path a `#[path = "..."]` attribute in the text names.
fn attribute_paths(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| attribute_path(line.trim()))
        .collect()
}

/// The path a `#[path = "..."]` attribute names.
fn attribute_path(line: &str) -> Option<String> {
    let rest = line.strip_prefix("#[path")?.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Every `tests/`-relative glob a string literal in the text names.
///
/// This is how a compile-fail harness selects its cases, and those files are
/// input data rather than modules.
fn fixture_globs(text: &str) -> Vec<String> {
    let mut globs = Vec::new();
    for piece in text.split('"').skip(1).step_by(2) {
        if piece.starts_with("tests/") && piece.contains('*') && piece.ends_with(".rs") {
            globs.push(piece.to_string());
        }
    }
    globs
}

/// The module a `mod name;` line declares, if the line is one.
fn module_name(line: &str) -> Option<String> {
    let rest = line.strip_suffix(';')?;
    let rest = rest
        .strip_prefix("pub mod ")
        .or_else(|| rest.strip_prefix("mod "))
        .or_else(|| rest.strip_prefix("pub(crate) mod "))
        .or_else(|| rest.strip_prefix("pub(super) mod "))?
        .trim();
    if rest.is_empty()
        || !rest
            .chars()
            .all(|letter| letter.is_alphanumeric() || letter == '_')
    {
        return None;
    }
    Some(rest.to_string())
}

/// The directory holding a file, in the tree's own spelling.
fn parent_of(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default()
}

/// A path joined onto a directory, with `.` and `..` segments resolved.
fn join(base: &str, relative: &str) -> String {
    let mut segments: Vec<&str> = base.split('/').filter(|part| !part.is_empty()).collect();
    for part in relative.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    segments.join("/")
}

/// Every test file the decision record marks `isolated`.
fn isolated(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let table = tree.read_toml(DECISIONS)?;
    let mut isolated = BTreeSet::new();
    for row in table
        .get("test")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let Some(path) = row.get("path").and_then(toml::Value::as_str) else {
            continue;
        };
        if row.get("grouping").and_then(toml::Value::as_str) == Some("isolated") {
            isolated.insert(path.to_string());
        }
    }
    Ok(isolated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateBehavior;
    use crate::gates::fixture_checkout::{checkout, files as reported, messages};

    /// The workspace manifest a fixture checkout needs to have members.
    const WORKSPACE: &str = "[workspace]\nmembers = [\"pkg\"]\n";

    /// A package manifest with the grouping policy applied.
    const GROUPED: &str = "[package]\nname = \"pkg\"\nversion = \"0.1.0\"\nautotests = false\n\n\
                           [[test]]\nname = \"all_tests\"\npath = \"tests/all_tests.rs\"\n";

    /// An empty decision record.
    const NO_DECISIONS: &str = "schema_version = 1\n";

    /// WHY: this is the failure the gate exists for. A test file no harness
    /// declares does not fail a build under `autotests = false`; it stops
    /// running, and every other test keeps passing. The neighbour the harness
    /// does declare must stay unreported, or the gate reports the layout it was
    /// given rather than the gap in it.
    #[test]
    fn a_test_file_no_target_declares_is_reported_and_a_declared_one_is_not() {
        let (_directory, root) = checkout(&[
            ("Cargo.toml", WORKSPACE),
            ("pkg/Cargo.toml", GROUPED),
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"declared.rs\"]\nmod declared;\n",
            ),
            ("pkg/tests/declared.rs", "#[test]\nfn runs() {}\n"),
            ("pkg/tests/orphan.rs", "#[test]\nfn never_runs() {}\n"),
            (DECISIONS, NO_DECISIONS),
        ]);

        let report = TestTargetMembership
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert_eq!(
            reported(&report),
            ["pkg/tests/orphan.rs"],
            "only the file no target declares is reported: {:?}",
            messages(&report)
        );
    }

    /// WHY: autodiscovery left on is the other half of the same contract. Every
    /// file here has a target, so the only thing wrong is that cargo links a
    /// second binary for each of them and reports every grouped case twice.
    #[test]
    fn a_package_that_leaves_autotests_on_is_reported() {
        let discovering = GROUPED.replace("autotests = false\n", "");
        let (_directory, root) = checkout(&[
            ("Cargo.toml", WORKSPACE),
            ("pkg/Cargo.toml", &discovering),
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"declared.rs\"]\nmod declared;\n",
            ),
            ("pkg/tests/declared.rs", "#[test]\nfn runs() {}\n"),
            (DECISIONS, NO_DECISIONS),
        ]);

        let report = TestTargetMembership
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert_eq!(report.findings.len(), 1, "{:?}", report.findings);
        assert!(
            report.findings[0].message.contains("leaves autotests on"),
            "{:?}",
            report.findings[0]
        );
    }

    /// WHY: a file with test cases in two harnesses runs them twice under two
    /// target names, and a selector that names one target silently runs the
    /// other's copy too. A shared fixture, including one whose only `#[test]`
    /// sits inside an assertion macro, is the legitimate case and must stay
    /// unreported: reporting it would make grouping impossible.
    #[test]
    fn a_case_file_in_two_harnesses_is_reported_and_a_macro_fixture_is_not() {
        let two = "[package]\nname = \"pkg\"\nversion = \"0.1.0\"\nautotests = false\n\n\
                   [[test]]\nname = \"all_tests\"\npath = \"tests/all_tests.rs\"\n\n\
                   [[test]]\nname = \"all_tests_extra\"\npath = \"tests/all_tests_extra.rs\"\n\
                   required-features = [\"extra\"]\n";
        let suite =
            "#[path = \"fixture.rs\"]\nmod fixture;\n#[path = \"shared.rs\"]\nmod shared;\n";
        let macro_fixture =
            "macro_rules! case {\n    ($name:ident) => {\n        #[test]\n        \
                            fn $name() {}\n    };\n}\n";
        let (_directory, root) = checkout(&[
            ("Cargo.toml", WORKSPACE),
            ("pkg/Cargo.toml", two),
            ("pkg/tests/all_tests.rs", suite),
            ("pkg/tests/all_tests_extra.rs", suite),
            ("pkg/tests/fixture.rs", macro_fixture),
            ("pkg/tests/shared.rs", "#[test]\nfn runs_twice() {}\n"),
            (DECISIONS, NO_DECISIONS),
        ]);

        let report = TestTargetMembership
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert_eq!(
            reported(&report),
            ["pkg/tests/shared.rs"],
            "the fixture whose cases exist only per expansion is not a duplicate: {:?}",
            messages(&report)
        );
    }

    /// WHY: the decision record is the only place stating that a test cannot
    /// share a process. Grouping one anyway lets the mutation it was isolated
    /// for reach every other test in the binary, and nothing else in the tree
    /// notices.
    #[test]
    fn an_isolated_test_compiled_into_a_shared_harness_is_reported() {
        let decisions = "schema_version = 1\n\n[[test]]\npath = \"pkg/tests/mutates.rs\"\n\
                         grouping = \"isolated\"\nreason = \"writes the process environment\"\n";
        let (_directory, root) = checkout(&[
            ("Cargo.toml", WORKSPACE),
            ("pkg/Cargo.toml", GROUPED),
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"mutates.rs\"]\nmod mutates;\n",
            ),
            (
                "pkg/tests/mutates.rs",
                "#[test]\nfn runs() {\n    std::env::set_var(\"FLAG\", \"1\");\n}\n",
            ),
            (DECISIONS, decisions),
        ]);

        let report = TestTargetMembership
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert_eq!(report.findings.len(), 1, "{:?}", report.findings);
        assert!(
            report.findings[0].message.contains("shared harness"),
            "{:?}",
            report.findings[0]
        );
    }

    /// WHY: isolation is about which tests share a process, and a target rooted
    /// in a subdirectory of `tests/` compiles that subdirectory and nothing
    /// else. An isolated fixture there shares a binary only with its own module
    /// family, which is the shape a registration reaching several files in one
    /// pin requires. Reporting it would leave the isolated row with no legal
    /// arrangement at all, and the pin would go back into the shared harness.
    #[test]
    fn an_isolated_test_in_its_own_directory_target_is_not_reported() {
        let manifest = "[package]\nname = \"pkg\"\nversion = \"0.1.0\"\nautotests = false\n\n\
                        [[test]]\nname = \"all_tests\"\npath = \"tests/all_tests.rs\"\n\n\
                        [[test]]\nname = \"pin\"\npath = \"tests/pin/main.rs\"\n";
        let decisions = "schema_version = 1\n\n[[test]]\npath = \"pkg/tests/pin/fixture.rs\"\n\
                         grouping = \"isolated\"\nreason = \"registers a fixture op\"\n";
        let (_directory, root) = checkout(&[
            ("Cargo.toml", WORKSPACE),
            ("pkg/Cargo.toml", manifest),
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"other.rs\"]\nmod other;\n",
            ),
            ("pkg/tests/other.rs", "#[test]\nfn runs() {}\n"),
            (
                "pkg/tests/pin/main.rs",
                "mod fixture;\n\n#[test]\nfn pinned() {}\n",
            ),
            (
                "pkg/tests/pin/fixture.rs",
                "inventory::submit! { Op::new() }\n",
            ),
            (DECISIONS, decisions),
        ]);

        let report = TestTargetMembership
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert!(report.findings.is_empty(), "{:?}", report.findings);
    }

    /// WHY: a target root in a subdirectory is one binary for its whole module
    /// tree, which is what grouping wants, so its members are not orphans. This
    /// is the shape `xtask/tests/tree_contracts` has, and reporting its fifty
    /// members would make the gate unusable exactly where it is needed.
    #[test]
    fn a_directory_target_covers_its_own_module_tree() {
        let manifest = "[package]\nname = \"pkg\"\nversion = \"0.1.0\"\nautotests = false\n\n\
                        [[test]]\nname = \"contracts\"\npath = \"tests/contracts/main.rs\"\n";
        let (_directory, root) = checkout(&[
            ("Cargo.toml", WORKSPACE),
            ("pkg/Cargo.toml", manifest),
            ("pkg/tests/contracts/main.rs", "mod one;\nmod two;\n"),
            ("pkg/tests/contracts/one.rs", "#[test]\nfn a() {}\n"),
            ("pkg/tests/contracts/two.rs", "#[test]\nfn b() {}\n"),
            (DECISIONS, NO_DECISIONS),
        ]);

        let report = TestTargetMembership
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert!(report.findings.is_empty(), "{:?}", report.findings);
    }

    /// WHY: not everything under `tests/` is an integration-test file. Unit-test
    /// material a `src/` module includes is compiled by the library target, and a
    /// compile-fail case selected by glob is input data. Reporting either as an
    /// orphan would produce a gate nobody can keep at zero, which is how a gate
    /// stops being read.
    #[test]
    fn unit_test_material_and_a_glob_fixture_are_not_integration_files() {
        let (_directory, root) = checkout(&[
            ("Cargo.toml", WORKSPACE),
            ("pkg/Cargo.toml", GROUPED),
            (
                "pkg/src/lib.rs",
                "#[cfg(test)]\n#[path = \"../tests/internal/mod.rs\"]\nmod internal;\n",
            ),
            ("pkg/tests/internal/mod.rs", "mod inner;\n"),
            ("pkg/tests/internal/inner.rs", "#[test]\nfn unit() {}\n"),
            (
                "pkg/tests/all_tests.rs",
                "#[path = \"cases.rs\"]\nmod cases;\n",
            ),
            (
                "pkg/tests/cases.rs",
                "#[test]\nfn ui() {\n    let cases = trybuild::TestCases::new();\n    \
                 cases.compile_fail(\"tests/ui/*.rs\");\n}\n",
            ),
            ("pkg/tests/ui/rejected.rs", "fn main() {}\n"),
            (DECISIONS, NO_DECISIONS),
        ]);

        let report = TestTargetMembership
            .run(&GateCtx::new(root, Vec::new()))
            .expect("Fix: the gate must read the fixture tree; check the fixture git step");
        assert!(report.findings.is_empty(), "{:?}", report.findings);
    }
}
