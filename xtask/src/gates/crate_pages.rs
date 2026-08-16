//! The `crate-pages` gate: every crate states its boundary and its module map.
//!
//! A crate boundary lived in two places a reader had to already know about:
//! `docs/architecture/crates.md` and `docs/CRATE_OWNERSHIP.toml`. Someone
//! opening a crate directory saw a `README.md` that described what the crate
//! does and nothing that said what it must never hold, which is the half of a
//! boundary that decides where new code goes. Sixteen crates had no page of
//! their own at all.
//!
//! Every workspace member carries a `SPEC.md` and a `README.md`. The roster is
//! the `workspace.members` array read at run time, so a crate added to the
//! workspace is a crate this gate demands pages for, and no list here goes
//! stale. Two directories outside the member set carry a `README.md` for the
//! same reason: they are built by CI and neither is a workspace member.
//!
//! Presence is the weakest half. A page that exists and states nothing is what
//! a presence-only rule produces, so the content is judged too:
//!
//! 1. A `SPEC.md` that never names what the crate must not contain is red. A
//!    boundary with only an inclusion half admits everything.
//! 2. A `SPEC.md` that names a workspace crate as an outbound edge the manifest
//!    does not declare is red, in that direction: a page claiming an edge the
//!    build does not have describes a different tree.
//! 3. A `README.md` module map naming a path the crate does not carry is red.
//!    That is the claim that rots first, because a module moves and the map
//!    does not.

use std::collections::BTreeSet;


use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Directories that carry a `README.md` without being workspace members.
///
/// Both are built by a workflow, and neither is reachable from the workspace
/// manifest, so nothing else in the tree demands a word about what they are.
const NON_MEMBER_PAGES: &[&str] = &["examples/external_backend_extension", "fuzz"];

/// The heading a boundary page states its exclusion half under.
const EXCLUSION_HEADING: &str = "## Must never contain";

/// The heading a boundary page states its outbound edges under.
const EDGES_HEADING: &str = "## What crosses its edges";

/// The line the outbound half of the edge section starts at.
const OUTBOUND_LEAD: &str = "Out of this crate, into:";

/// The line the inbound half starts at, which this gate does not judge.
const INBOUND_LEAD: &str = "Into this crate, from:";

/// Findings for one member's pages.
fn member_findings(tree: &Tree, path: &str, name: &str, members: &[String]) -> Vec<Finding> {
    let mut findings = Vec::new();
    let spec = format!("{path}/SPEC.md");
    let readme = format!("{path}/README.md");
    if !tree.exists(&spec) {
        findings.push(Finding::in_file(
            &spec,
            format!("crate `{name}` states no boundary"),
            "add SPEC.md naming what the crate owns, what it must never contain, what crosses its edges and the gates that enforce it",
        ));
    } else if let Ok(text) = tree.read(&spec) {
        findings.extend(spec_findings(&spec, name, &text, members, tree));
    }
    if !tree.exists(&readme) {
        findings.push(Finding::in_file(
            &readme,
            format!("crate `{name}` has no page describing it"),
            "add README.md with one paragraph, the module map, the entry points and how to run its tests",
        ));
    } else if let Ok(text) = tree.read(&readme) {
        findings.extend(module_map_findings(tree, path, &readme, &text));
    }
    findings
}

/// Findings for one boundary page.
fn spec_findings(
    spec: &str,
    name: &str,
    text: &str,
    members: &[String],
    tree: &Tree,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    if !text.contains(EXCLUSION_HEADING) {
        findings.push(Finding::in_file(
            spec,
            format!("`{name}` states no exclusion, so its boundary admits everything"),
            format!("add a `{EXCLUSION_HEADING}` section naming what the crate must not hold"),
        ));
    } else if section(text, EXCLUSION_HEADING).trim().is_empty() {
        findings.push(Finding::in_file(
            spec,
            format!("`{name}` carries an empty exclusion section"),
            "state what the crate must not hold, or delete the heading",
        ));
    }
    let declared = manifest_edges(tree, name, members);
    for (line, claimed) in claimed_edges(text, members, name) {
        if !declared.contains(&claimed) {
            findings.push(Finding::at(
                spec,
                line,
                format!("`{name}` claims an outbound edge to `{claimed}` its manifest does not declare"),
                "declare the dependency, or take the edge off the page; a claimed edge the build does not have describes a different tree",
            ));
        }
    }
    findings
}

/// Every workspace crate the member manifest depends on.
///
/// A `[target.'cfg(...)'.dependencies]` table is a dependency like any other:
/// `vyre-bench` declares `vyre-driver-cuda` that way, and a reader of the three
/// unconditional tables alone reported the page as claiming an edge the build
/// does have.
fn manifest_edges(tree: &Tree, name: &str, members: &[String]) -> BTreeSet<String> {
    let Ok(manifests) = tree.member_manifests() else {
        return BTreeSet::new();
    };
    let Some(member) = manifests.iter().find(|member| member.name == name) else {
        return BTreeSet::new();
    };
    let mut edges = BTreeSet::new();
    let mut tables: Vec<&toml::Table> = vec![&member.manifest];
    if let Some(targets) = member.manifest.get("target").and_then(toml::Value::as_table) {
        tables.extend(targets.values().filter_map(toml::Value::as_table));
    }
    for table in tables {
        for kind in ["dependencies", "dev-dependencies", "build-dependencies"] {
            let Some(rows) = table.get(kind).and_then(toml::Value::as_table) else {
                continue;
            };
            for key in rows.keys() {
                if members.iter().any(|member| member == key) {
                    edges.insert(key.clone());
                }
            }
        }
    }
    edges
}

/// Every workspace crate the outbound half of the edge section names.
///
/// The section states both directions, and only the outbound one is the crate's
/// own claim: the inbound rows describe the dependents' manifests, which those
/// crates' pages answer for. Reading both directions here reported every
/// dependent of every crate as a false edge.
fn claimed_edges(text: &str, members: &[String], name: &str) -> Vec<(u32, String)> {
    let mut claimed = Vec::new();
    let mut seen = BTreeSet::new();
    let mut outbound = false;
    for (line, body) in numbered_section(text, EDGES_HEADING) {
        let trimmed = body.trim();
        if trimmed.starts_with(OUTBOUND_LEAD) {
            outbound = true;
            continue;
        }
        if trimmed.starts_with(INBOUND_LEAD) {
            outbound = false;
            continue;
        }
        if !outbound {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("- `") else {
            continue;
        };
        let Some((package, _)) = rest.split_once('`') else {
            continue;
        };
        if package == name || !members.iter().any(|member| member == package) {
            continue;
        }
        if seen.insert(package.to_string()) {
            claimed.push((line, package.to_string()));
        }
    }
    claimed
}

/// Findings for the module map of one page.
///
/// A path is read only from a bullet whose code span looks like a file or a
/// directory inside the crate, so a bullet naming a type or a command is prose
/// this rule does not judge.
fn module_map_findings(tree: &Tree, crate_path: &str, readme: &str, text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (line, body) in crate::gates::scan::numbered(text) {
        let trimmed = body.trim();
        let Some(rest) = trimmed.strip_prefix("- `") else {
            continue;
        };
        let Some((span, _)) = rest.split_once('`') else {
            continue;
        };
        if !is_module_path(span) {
            continue;
        }
        let relative = format!("{crate_path}/{}", span.trim_end_matches('/'));
        if !tree.exists(&relative) {
            findings.push(Finding::at(
                readme,
                line,
                format!("the module map names `{span}`, which the crate does not carry"),
                "point the row at the module as it stands, or delete the row; a map of a tree that moved sends every reader to the wrong file",
            ));
        }
    }
    findings
}

/// Whether a code span in a bullet names a module of the crate.
///
/// Only a path under one of the crate's own source roots is judged. A crate
/// page also cites workspace files such as `docs/CRATE_OWNERSHIP.toml` and
/// names generated outputs such as `scorecard.md`, and both resolve against the
/// repository root rather than the crate: `docs-references` already reads
/// those, and reading them here reported a citation as a missing module.
fn is_module_path(span: &str) -> bool {
    if span.contains(' ') || span.contains("::") {
        return false;
    }
    ["src/", "tests/", "benches/", "examples/", "fuzz_targets/"]
        .iter()
        .any(|root| span.starts_with(root))
}

/// The body of one `##` section, empty when the heading is absent.
fn section<'a>(text: &'a str, heading: &str) -> &'a str {
    let Some(start) = text.find(heading) else {
        return "";
    };
    let body = &text[start + heading.len()..];
    match body.find("\n## ") {
        Some(end) => &body[..end],
        None => body,
    }
}

/// Every line of one `##` section, with its line number in the whole document.
fn numbered_section(text: &str, heading: &str) -> Vec<(u32, String)> {
    let mut rows = Vec::new();
    let mut inside = false;
    for (line, body) in crate::gates::scan::numbered(text) {
        if body.trim_end() == heading {
            inside = true;
            continue;
        }
        if inside && body.starts_with("## ") {
            break;
        }
        if inside {
            rows.push((line, body.to_string()));
        }
    }
    rows
}

/// Hold every crate to a boundary page and a module map.
pub struct CratePages;

impl Gate for CratePages {
    fn name(&self) -> &'static str {
        "crate-pages"
    }

    fn help(&self) -> &'static str {
        "Hold every workspace member to a SPEC.md stating its boundary and a README.md whose module map resolves"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let members = tree.member_manifests()?;
        let names: Vec<String> = members.iter().map(|member| member.name.clone()).collect();
        let mut report = Report::clean();
        for member in &members {
            for finding in member_findings(&tree, &member.path, &member.name, &names) {
                report.find(finding);
            }
        }
        for path in NON_MEMBER_PAGES {
            let readme = format!("{path}/README.md");
            if !tree.exists(&readme) {
                report.find(Finding::in_file(
                    &readme,
                    format!("`{path}` is built by CI and says nothing about what it is"),
                    "add README.md with one paragraph, what it holds and how to run it",
                ));
                continue;
            }
            let text = tree.read(&readme)?;
            for finding in module_map_findings(&tree, path, &readme, &text) {
                report.find(finding);
            }
        }
        report.note(format!(
            "{} workspace member(s) and {} non-member director(ies) judged",
            members.len(),
            NON_MEMBER_PAGES.len()
        ));
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::fixture_checkout::checkout;
    use std::path::Path;

    const SPEC: &str = "# demo\n\n## Owns\n\nThings.\n\n## Must never contain\n\nA device.\n\n## What crosses its edges\n\nOut of this crate, into:\n\n- `other` over the `seam` seam, public: a reason.\n";
    const README: &str = "# demo\n\nA crate.\n\n## Modules\n\n- `src/lib.rs`: the crate.\n";

    fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        let files = [
            (
                "Cargo.toml",
                "[workspace]\nmembers = [\"demo\", \"other\"]\n",
            ),
            (
                "demo/Cargo.toml",
                "[package]\nname = \"demo\"\n\n[dependencies]\nother = { path = \"../other\" }\n",
            ),
            ("demo/src/lib.rs", "pub fn demo() {}\n"),
            ("demo/SPEC.md", SPEC),
            ("demo/README.md", README),
            ("other/Cargo.toml", "[package]\nname = \"other\"\n"),
            ("other/src/lib.rs", "pub fn other() {}\n"),
            (
                "other/SPEC.md",
                "# other\n\n## Owns\n\nThings.\n\n## Must never contain\n\nA device.\n",
            ),
            ("other/README.md", "# other\n\nA crate.\n"),
            ("examples/external_backend_extension/README.md", "# example\n"),
            ("fuzz/README.md", "# fuzz\n"),
        ];
        checkout(&files)
    }

    fn findings(root: &Path) -> Vec<Finding> {
        CratePages
            .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
            .expect("Fix: the gate must run against the fixture")
            .findings
    }

    fn messages(findings: &[Finding]) -> String {
        findings
            .iter()
            .map(|finding| finding.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// WHY: the clean case has to be silent, or every injection below proves
    /// nothing. The roster comes from `workspace.members`, so this also fixes
    /// that a member with both pages is judged and passes.
    #[test]
    fn a_tree_where_every_member_states_its_boundary_is_silent() {
        let (_temporary, root) = fixture();
        assert!(findings(&root).is_empty(), "{}", messages(&findings(&root)));
    }

    /// WHY: the roster is derived, so adding a member must make the suite red
    /// until that member has pages. A hardcoded list would go stale in silence,
    /// which is the same failure as having no gate.
    #[test]
    fn a_member_added_to_the_workspace_needs_pages() {
        let (_temporary, root) = fixture();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"demo\", \"other\", \"third\"]\n",
        )
        .expect("Fix: the fixture manifest must be writable");
        std::fs::create_dir_all(root.join("third/src")).expect("Fix: the fixture must be writable");
        std::fs::write(
            root.join("third/Cargo.toml"),
            "[package]\nname = \"third\"\n",
        )
        .expect("Fix: the fixture must be writable");
        std::fs::write(root.join("third/src/lib.rs"), "pub fn third() {}\n")
            .expect("Fix: the fixture must be writable");

        let found = findings(&root);
        assert_eq!(found.len(), 2, "{}", messages(&found));
        assert!(messages(&found).contains("states no boundary"));
        assert!(messages(&found).contains("no page describing it"));
    }

    /// WHY: this is the clause presence alone cannot carry. A page that lists
    /// what a crate owns and never says what it must not hold is the half of a
    /// boundary that decides nothing, and it is the half that gets dropped.
    #[test]
    fn a_boundary_page_with_no_exclusion_is_reported() {
        let (_temporary, root) = fixture();
        std::fs::write(
            root.join("demo/SPEC.md"),
            "# demo\n\n## Owns\n\nThings.\n\n## What crosses its edges\n\nNothing.\n",
        )
        .expect("Fix: the fixture must be writable");

        let found = findings(&root);
        assert_eq!(found.len(), 1, "{}", messages(&found));
        assert!(found[0].message.contains("states no exclusion"));
    }

    /// WHY: an exclusion heading with nothing under it satisfies a substring
    /// check and states no rule, so the empty case is judged apart from the
    /// missing one.
    #[test]
    fn an_empty_exclusion_section_is_reported() {
        let (_temporary, root) = fixture();
        std::fs::write(
            root.join("demo/SPEC.md"),
            "# demo\n\n## Owns\n\nThings.\n\n## Must never contain\n\n## What crosses its edges\n\nNothing.\n",
        )
        .expect("Fix: the fixture must be writable");

        let found = findings(&root);
        assert_eq!(found.len(), 1, "{}", messages(&found));
        assert!(found[0].message.contains("empty exclusion section"));
    }

    /// WHY: a page may only claim an edge the build has. The direction matters:
    /// an edge the manifest declares and the page omits is an incomplete page,
    /// not a false claim, and this rule reports only the false claim.
    #[test]
    fn an_edge_the_manifest_does_not_declare_is_reported() {
        let (_temporary, root) = fixture();
        std::fs::write(
            root.join("other/SPEC.md"),
            "# other\n\n## Owns\n\nThings.\n\n## Must never contain\n\nA device.\n\n## What crosses its edges\n\nOut of this crate, into:\n\n- `demo` over the `seam` seam, public: a reason.\n",
        )
        .expect("Fix: the fixture must be writable");

        let found = findings(&root);
        assert_eq!(found.len(), 1, "{}", messages(&found));
        assert!(found[0].message.contains("claims an outbound edge to `demo`"));
        assert_eq!(found[0].line, Some(15));
    }

    /// WHY: the module map is the claim that rots first, because a module moves
    /// and the map does not. A bullet that names no path must stay unjudged, or
    /// the rule reports prose.
    #[test]
    fn a_module_map_naming_a_module_the_crate_lacks_is_reported() {
        let (_temporary, root) = fixture();
        std::fs::write(
            root.join("demo/README.md"),
            "# demo\n\nA crate.\n\n## Modules\n\n- `src/gone.rs`: departed.\n- `DemoType`: a type, not a path.\n",
        )
        .expect("Fix: the fixture must be writable");

        let found = findings(&root);
        assert_eq!(found.len(), 1, "{}", messages(&found));
        assert!(found[0].message.contains("`src/gone.rs`"));
    }

    /// WHY: both directories are built by a workflow and neither is a workspace
    /// member, so the member roster cannot reach them and a reader who opens one
    /// has nothing to read.
    #[test]
    fn a_non_member_directory_with_no_page_is_reported() {
        let (_temporary, root) = fixture();
        std::fs::remove_file(root.join("fuzz/README.md")).expect("Fix: the fixture is writable");

        let found = findings(&root);
        assert_eq!(found.len(), 1, "{}", messages(&found));
        assert!(found[0].message.contains("fuzz"));
    }
}
