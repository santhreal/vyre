//! What the manifests must say about each other.
//!
//! Three rules, all read from tracked manifests and none of them running cargo.
//! Every manifest on disk belongs to the workspace or declares its own. Every
//! path edge and every inherited key resolves to something that exists. And a
//! publishable crate's dependency on a member carries a version when that member
//! is published and carries none when it is not.
//!
//! Cargo is deliberately absent. The failure the path rule detects is a workspace
//! cargo cannot load, so a cargo-based check cannot run while the defect is
//! present: `5826591fad` deleted a crate tree and left a sibling depending on it,
//! and every cargo command failed from a clean checkout for several commits with
//! nothing saying why.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Dependency tables a manifest can declare, at the top level or under a target.
const DEP_TABLES: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];

/// Every manifest on disk is a workspace member or its own workspace root.
pub struct WorkspaceMembership;

impl Gate for WorkspaceMembership {
    fn name(&self) -> &'static str {
        "workspace-membership"
    }

    fn help(&self) -> &'static str {
        "every Cargo.toml is a workspace member or its own workspace root"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let declared: BTreeSet<String> = tree.members()?.into_iter().collect();
        if declared.is_empty() {
            return Err(GateError::new(
                "the root manifest declares no workspace members",
                "declare workspace.members; a membership scan over an empty roster \
                 reports success forever",
            ));
        }
        let excluded = excluded_directories(&tree)?;
        let mut counted = 0usize;

        for manifest in manifests(&tree) {
            let Some(directory) = manifest_directory(&manifest) else {
                continue;
            };
            counted += 1;
            if declared.contains(&directory) || excluded.contains(&directory) {
                continue;
            }
            let table = tree.read_toml(&manifest)?;
            if table.contains_key("workspace") {
                continue;
            }
            report.find(Finding::in_file(
                &manifest,
                "crate is in neither workspace.members nor workspace.exclude and \
                 declares no [workspace] of its own",
                "add the directory to workspace.members, or give the crate its own \
                 [workspace] table so cargo treats it as separate",
            ));
        }

        report.note(format!("{counted} manifest(s) accounted for"));
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        Ok(report)
    }
}

/// Every path edge and inherited key resolves to a tracked manifest or table entry.
pub struct PathDepsResolve;

impl Gate for PathDepsResolve {
    fn name(&self) -> &'static str {
        "path-deps-resolve"
    }

    fn help(&self) -> &'static str {
        "path dependencies, members, patches and workspace inheritance all resolve"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let manifests = manifests(&tree);
        if manifests.is_empty() {
            return Err(GateError::new(
                "no tracked Cargo.toml found",
                "run this gate inside the workspace checkout; a manifest scan over an \
                 empty set reports success forever",
            ));
        }
        let tracked: BTreeSet<PathBuf> = manifests.iter().cloned().collect();
        let root_table = tree.read_toml("Cargo.toml")?;
        let root_workspace = root_table
            .get("workspace")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| {
                GateError::new(
                    "the root Cargo.toml declares no [workspace] table",
                    "restore the workspace table; every inheritance edge resolves against it",
                )
            })?;
        let workspace_deps = keys_of(root_workspace, "dependencies");
        let workspace_package = keys_of(root_workspace, "package");
        if workspace_deps.is_empty() {
            return Err(GateError::new(
                "[workspace.dependencies] is empty",
                "restore the table; the inheritance half of this rule would scan nothing",
            ));
        }
        let mut edges = 0usize;

        for manifest in &manifests {
            let table = tree.read_toml(manifest)?;
            let mut edge = Resolver {
                tree: &tree,
                tracked: &tracked,
                manifest,
                lines: dep_lines(&tree.read(manifest)?),
                edges: 0,
            };

            for (prefix, host) in dependency_hosts(&table) {
                for kind in DEP_TABLES {
                    for (key, spec) in entries(&host, kind) {
                        let table_name = format!("{prefix}{kind}");
                        if let Some(path) = spec.get("path").and_then(toml::Value::as_str) {
                            edge.path_edge(&mut report, &table_name, &key, path);
                        }
                        if spec.get("workspace").and_then(toml::Value::as_bool) == Some(true) {
                            edge.inherited(
                                &mut report,
                                &table_name,
                                &key,
                                &workspace_deps,
                                "workspace.dependencies",
                            );
                        }
                    }
                }
            }

            for (field, spec) in entries(&table, "package") {
                if spec.get("workspace").and_then(toml::Value::as_bool) == Some(true) {
                    edge.inherited(
                        &mut report,
                        "package",
                        &field,
                        &workspace_package,
                        "workspace.package",
                    );
                }
            }

            if let Some(workspace) = table.get("workspace").and_then(toml::Value::as_table) {
                for (key, spec) in entries(workspace, "dependencies") {
                    if let Some(path) = spec.get("path").and_then(toml::Value::as_str) {
                        edge.path_edge(&mut report, "workspace.dependencies", &key, path);
                    }
                }
                let declared = workspace
                    .get("members")
                    .and_then(toml::Value::as_array)
                    .map_or(&[][..], Vec::as_slice);
                for member in declared {
                    let Some(member) = member.as_str() else {
                        report.find(Finding::in_file(
                            manifest,
                            "workspace.members holds a non-string entry",
                            "declare every member as a string path",
                        ));
                        continue;
                    };
                    edge.member_edge(&mut report, &manifests, member);
                }
                for (registry, spec) in entries(workspace, "patch") {
                    edge.patch_edges(&mut report, &registry, &spec);
                }
            }

            for (registry, spec) in entries(&table, "patch") {
                edge.patch_edges(&mut report, &registry, &spec);
            }
            edges += edge.edges;
        }

        report.note(format!(
            "{edges} manifest edge(s) across {} tracked manifest(s)",
            manifests.len()
        ));
        Ok(report)
    }
}

/// One manifest's edges, checked against the tracked manifest set.
///
/// The edge count is carried here rather than in a closure so both halves of the
/// rule, paths and inheritance, report against the same total. A rule that
/// counts nothing cannot say it scanned anything.
struct Resolver<'r> {
    tree: &'r Tree,
    tracked: &'r BTreeSet<PathBuf>,
    manifest: &'r Path,
    lines: BTreeMap<(String, String), u32>,
    edges: usize,
}

impl Resolver<'_> {
    /// The line a key is written on, when the text form located it.
    fn line(&self, table: &str, key: &str) -> Option<u32> {
        self.lines
            .get(&(table.to_string(), key.to_string()))
            .copied()
    }

    /// A `path = ` edge, which must name a tracked manifest inside the tree.
    fn path_edge(&mut self, report: &mut Report, table: &str, key: &str, raw: &str) {
        self.edges += 1;
        let line = self.line(table, key);
        let label = format!("{table}.{key}");
        let Some(target) = resolved_manifest(self.manifest, raw) else {
            report.find(finding(
                self.manifest,
                line,
                format!("{label} path `{raw}` escapes the repository"),
                "express the path relative to the manifest, inside this repository",
            ));
            return;
        };
        if self.tracked.contains(&target) {
            return;
        }
        let state = if self.tree.exists(&target.to_string_lossy()) {
            "exists but is untracked"
        } else {
            "does not exist"
        };
        report.find(finding(
            self.manifest,
            line,
            format!(
                "{label} path `{raw}` names `{}`, which {state}",
                target.display()
            ),
            "delete the entry, or restore the member it names; a manifest naming a \
             missing member makes every cargo command fail from a clean checkout",
        ));
    }

    /// A `workspace = true` edge, which must have a root table entry to inherit.
    fn inherited(
        &mut self,
        report: &mut Report,
        table: &str,
        key: &str,
        available: &BTreeSet<String>,
        origin: &str,
    ) {
        self.edges += 1;
        if available.contains(key) {
            return;
        }
        report.find(finding(
            self.manifest,
            self.line(table, key),
            format!("{table}.{key} sets `workspace = true` but `{origin}.{key}` is absent"),
            "add the root entry, or declare the dependency locally; cargo cannot load the \
             workspace at all while this dangles",
        ));
    }

    /// A `workspace.members` entry, literal or a pattern.
    fn member_edge(&mut self, report: &mut Report, manifests: &[PathBuf], member: &str) {
        self.edges += 1;
        if member.contains('*') || member.contains('?') || member.contains('[') {
            let prefix = member.trim_end_matches('*').trim_end_matches('/');
            let matched = manifests.iter().any(|candidate| {
                manifest_directory(candidate)
                    .is_some_and(|directory| directory.starts_with(prefix))
            });
            if !matched {
                report.find(Finding::in_file(
                    self.manifest,
                    format!("workspace.members pattern `{member}` matches no Cargo.toml"),
                    "delete the pattern, or restore the members it was written for",
                ));
            }
            return;
        }
        let Some(target) = resolved_manifest(self.manifest, member) else {
            report.find(Finding::in_file(
                self.manifest,
                format!("workspace.members `{member}` escapes the repository"),
                "express the member path relative to the manifest",
            ));
            return;
        };
        if !self.tracked.contains(&target) {
            report.find(Finding::in_file(
                self.manifest,
                format!(
                    "workspace.members `{member}` names `{}`, which is not a tracked manifest",
                    target.display()
                ),
                "delete the member, or restore the crate it names",
            ));
        }
    }

    /// Every `path = ` edge under one `[patch.<registry>]` table.
    fn patch_edges(&mut self, report: &mut Report, registry: &str, spec: &toml::Table) {
        for (key, entry) in spec {
            let Some(path) = entry
                .as_table()
                .and_then(|entry| entry.get("path"))
                .and_then(toml::Value::as_str)
            else {
                continue;
            };
            self.path_edge(report, &format!("patch.{registry}"), key, path);
        }
    }
}

/// A publishable crate's dependency on a member carries a version exactly when
/// that member is published.
pub struct InternalDepVersions;

impl Gate for InternalDepVersions {
    fn name(&self) -> &'static str {
        "internal-dep-versions"
    }

    fn help(&self) -> &'static str {
        "internal dependencies of publishable crates carry a version, and \
         dependencies on unpublishable members carry none"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let members = tree.member_manifests()?;
        let roster: BTreeMap<String, bool> = members
            .iter()
            .map(|member| (member.name.clone(), member.publishable()))
            .collect();
        let published = roster.values().filter(|value| **value).count();
        let unpublished = roster.len() - published;
        if published == 0 || unpublished == 0 {
            return Err(GateError::new(
                format!(
                    "derived {published} publishable and {unpublished} unpublishable members"
                ),
                "one half of this rule would scan nothing; check that member manifests \
                 declare package.publish as they mean it",
            ));
        }
        let root_table = tree.read_toml("Cargo.toml")?;
        let workspace_deps = root_table
            .get("workspace")
            .and_then(|workspace| workspace.get("dependencies"))
            .and_then(toml::Value::as_table)
            .cloned()
            .unwrap_or_default();
        let mut edges = 0usize;

        for member in members.iter().filter(|member| member.publishable()) {
            let manifest = PathBuf::from(format!("{}/Cargo.toml", member.path));
            let lines = dep_lines(&tree.read(&manifest)?);
            for (prefix, host) in dependency_hosts(&member.manifest) {
                for kind in DEP_TABLES {
                    for (key, spec) in entries(&host, kind) {
                        let table_name = format!("{prefix}{kind}");
                        let package = target_package(&key, &spec, &workspace_deps);
                        let Some(publishable) = roster.get(&package) else {
                            continue;
                        };
                        edges += 1;
                        let line = lines
                            .get(&(table_name.clone(), key.clone()))
                            .copied();
                        let named = if package == key {
                            key.clone()
                        } else {
                            format!("{key} (package = {package})")
                        };
                        let inherited =
                            spec.get("workspace").and_then(toml::Value::as_bool) == Some(true);
                        let source = if inherited {
                            "[workspace.dependencies]".to_string()
                        } else {
                            format!("[{table_name}]")
                        };
                        match has_version(&key, &spec, &workspace_deps) {
                            None => report.find(finding(
                                &manifest,
                                line,
                                format!(
                                    "{named} sets `workspace = true` but \
                                     [workspace.dependencies] has no `{key}` entry"
                                ),
                                "add the entry, or declare the dependency locally",
                            )),
                            Some(true) if !publishable => report.find(finding(
                                &manifest,
                                line,
                                format!(
                                    "{named} carries a version through {source}, but {package} \
                                     is `publish = false`"
                                ),
                                "make it path-only; no registry can satisfy a version \
                                 requirement on an unpublishable crate, so packaging the \
                                 depender fails here",
                            )),
                            Some(false)
                                if *publishable && !table_name.ends_with("dev-dependencies") =>
                            {
                                report.find(finding(
                                    &manifest,
                                    line,
                                    format!(
                                        "{named} is path-only through {source}, but {package} \
                                         is published"
                                    ),
                                    "give it both `version` and `path`, or inherit from a \
                                     versioned workspace entry; path-only blocks publishing \
                                     from resolving siblings on the registry",
                                ));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let root_lines = dep_lines(&tree.read("Cargo.toml")?);
        for (key, spec) in workspace_deps.iter() {
            let spec_table = spec.as_table().cloned().unwrap_or_default();
            let package = target_package(key, &spec_table, &workspace_deps);
            let Some(publishable) = roster.get(&package) else {
                continue;
            };
            edges += 1;
            let line = root_lines
                .get(&("workspace.dependencies".to_string(), key.clone()))
                .copied();
            let named = if package == *key {
                key.clone()
            } else {
                format!("{key} (package = {package})")
            };
            let versioned = match spec {
                toml::Value::String(_) => true,
                other => other
                    .as_table()
                    .is_some_and(|table| table.contains_key("version")),
            };
            if *publishable && !versioned {
                report.find(finding(
                    Path::new("Cargo.toml"),
                    line,
                    format!("[workspace.dependencies] {named} is path-only, but {package} is published"),
                    "add `version`; every member inheriting this entry would publish \
                     unresolvable",
                ));
            } else if !publishable && versioned {
                report.find(finding(
                    Path::new("Cargo.toml"),
                    line,
                    format!(
                        "[workspace.dependencies] {named} carries a version, but {package} is \
                         `publish = false`"
                    ),
                    "make the entry path-only; every member inheriting it inherits a \
                     requirement no registry can satisfy",
                ));
            }
        }

        report.note(format!(
            "{edges} internal dependency edge(s) from {published} publishable member(s), \
             {unpublished} member(s) must stay path-only"
        ));
        Ok(report)
    }
}

/// Every tracked `Cargo.toml`, repository-relative.
fn manifests(tree: &Tree) -> Vec<PathBuf> {
    tree.paths()
        .iter()
        .filter(|path| path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml"))
        .cloned()
        .collect()
}

/// The directory a manifest sits in, or `None` for the root manifest.
fn manifest_directory(manifest: &Path) -> Option<String> {
    let parent = manifest.parent()?;
    if parent.as_os_str().is_empty() {
        return None;
    }
    Some(parent.to_string_lossy().replace('\\', "/"))
}

/// Directories the root manifest holds out of the workspace deliberately.
fn excluded_directories(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let table = tree.read_toml("Cargo.toml")?;
    Ok(table
        .get("workspace")
        .and_then(|workspace| workspace.get("exclude"))
        .and_then(toml::Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default())
}

/// The top-level table plus every `[target.<triple>]` table, with the prefix a
/// finding names them by.
pub(crate) fn dependency_hosts(table: &toml::Table) -> Vec<(String, toml::Table)> {
    let mut hosts = vec![(String::new(), table.clone())];
    if let Some(targets) = table.get("target").and_then(toml::Value::as_table) {
        for (triple, host) in targets {
            if let Some(host) = host.as_table() {
                hosts.push((format!("target.{triple}."), host.clone()));
            }
        }
    }
    hosts
}

/// One dependency table's entries, each normalised to a table so a bare version
/// string and a full specification read the same way.
pub(crate) fn entries(table: &toml::Table, name: &str) -> Vec<(String, toml::Table)> {
    let Some(inner) = table.get(name).and_then(toml::Value::as_table) else {
        return Vec::new();
    };
    inner
        .iter()
        .map(|(key, value)| {
            let spec = match value {
                toml::Value::Table(spec) => spec.clone(),
                toml::Value::String(version) => {
                    let mut spec = toml::Table::new();
                    spec.insert(
                        "version".to_string(),
                        toml::Value::String(version.clone()),
                    );
                    spec
                }
                _ => toml::Table::new(),
            };
            (key.clone(), spec)
        })
        .collect()
}

/// The key set of one sub-table of the workspace table.
fn keys_of(workspace: &toml::Table, name: &str) -> BTreeSet<String> {
    workspace
        .get(name)
        .and_then(toml::Value::as_table)
        .map(|table| table.keys().cloned().collect())
        .unwrap_or_default()
}

/// The manifest a path edge points at, or `None` when the path leaves the tree.
///
/// Resolution is lexical rather than filesystem-based, so a path naming a
/// deleted directory still reports where it pointed instead of failing to
/// canonicalise.
fn resolved_manifest(manifest: &Path, raw: &str) -> Option<PathBuf> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(parent) = manifest.parent() {
        for component in parent.components() {
            if let Component::Normal(part) = component {
                parts.push(part.to_string_lossy().to_string());
            }
        }
    }
    for component in Path::new(raw).components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
            Component::ParentDir => {
                if parts.pop().is_none() {
                    return None;
                }
            }
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    parts.push("Cargo.toml".to_string());
    Some(parts.iter().collect())
}

/// The package a dependency key names, following a `package =` rename through
/// workspace inheritance.
pub(crate) fn target_package(key: &str, spec: &toml::Table, workspace_deps: &toml::Table) -> String {
    if let Some(renamed) = spec.get("package").and_then(toml::Value::as_str) {
        return renamed.to_string();
    }
    if spec.get("workspace").and_then(toml::Value::as_bool) == Some(true) {
        if let Some(renamed) = workspace_deps
            .get(key)
            .and_then(toml::Value::as_table)
            .and_then(|inherited| inherited.get("package"))
            .and_then(toml::Value::as_str)
        {
            return renamed.to_string();
        }
    }
    key.to_string()
}

/// Whether a dependency carries a version, or `None` when it inherits from a
/// workspace entry that does not exist.
fn has_version(key: &str, spec: &toml::Table, workspace_deps: &toml::Table) -> Option<bool> {
    if spec.get("workspace").and_then(toml::Value::as_bool) == Some(true) {
        return match workspace_deps.get(key) {
            None => None,
            Some(toml::Value::String(_)) => Some(true),
            Some(inherited) => Some(
                inherited
                    .as_table()
                    .is_some_and(|table| table.contains_key("version")),
            ),
        };
    }
    Some(spec.contains_key("version"))
}

/// A finding that carries a line number when the key was locatable.
fn finding(
    manifest: &Path,
    line: Option<u32>,
    message: impl Into<String>,
    fix: impl Into<String>,
) -> Finding {
    match line {
        Some(line) => Finding::at(manifest, line, message, fix),
        None => Finding::in_file(manifest, message, fix),
    }
}

/// Map each `(table, key)` pair to the line its key is written on.
///
/// A dependency read out of a parsed table has no position, and a finding a
/// reader cannot jump to costs them the search.
pub(crate) fn dep_lines(text: &str) -> BTreeMap<(String, String), u32> {
    let mut located = BTreeMap::new();
    let mut table = String::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && !trimmed.starts_with("[[") {
            if let Some(end) = trimmed.find(']') {
                table = trimmed[1..end].trim().to_string();
            }
            continue;
        }
        if trimmed.starts_with('#') || !trimmed.contains('=') {
            continue;
        }
        let key = trimmed
            .split('=')
            .next()
            .unwrap_or_default()
            .trim()
            .trim_matches('"')
            .split('.')
            .next()
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
        located.entry((table.clone(), key)).or_insert(number);
    }
    located
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the path resolver runs against manifests naming directories that no
    /// longer exist, which is the defect it detects, so it must not depend on the
    /// filesystem. It must also refuse a path that leaves the repository, because
    /// a member outside the tree is not something a checkout can build.
    #[test]
    fn path_resolution_is_lexical_and_bounded() {
        let manifest = Path::new("vyre-libs/Cargo.toml");
        assert_eq!(
            resolved_manifest(manifest, "../vyre-foundation"),
            Some(PathBuf::from("vyre-foundation/Cargo.toml"))
        );
        assert_eq!(
            resolved_manifest(manifest, "gone/crate"),
            Some(PathBuf::from("vyre-libs/gone/crate/Cargo.toml"))
        );
        assert_eq!(resolved_manifest(manifest, "../../outside"), None);
        assert_eq!(resolved_manifest(manifest, "/absolute"), None);
    }

    /// WHY: the version rule keys on the real package, so a rename must follow
    /// through both the local table and the inherited one. A rename read as the
    /// key name silently drops the edge out of the scan.
    #[test]
    fn a_rename_names_the_package_it_points_at() {
        let mut workspace = toml::Table::new();
        let mut inherited = toml::Table::new();
        inherited.insert(
            "package".to_string(),
            toml::Value::String("vyre-foundation".to_string()),
        );
        workspace.insert("foundation".to_string(), toml::Value::Table(inherited));

        let mut local = toml::Table::new();
        local.insert(
            "package".to_string(),
            toml::Value::String("vyre-driver".to_string()),
        );
        assert_eq!(target_package("driver", &local, &workspace), "vyre-driver");

        let mut inheriting = toml::Table::new();
        inheriting.insert("workspace".to_string(), toml::Value::Boolean(true));
        assert_eq!(
            target_package("foundation", &inheriting, &workspace),
            "vyre-foundation"
        );
        assert_eq!(
            target_package("plain", &toml::Table::new(), &workspace),
            "plain"
        );
    }

    /// WHY: `workspace = true` against a table entry that does not exist is a
    /// workspace cargo cannot load, and it is a different answer from "carries no
    /// version". Collapsing the two loses the load failure.
    #[test]
    fn inheritance_without_an_entry_is_neither_versioned_nor_unversioned() {
        let mut workspace = toml::Table::new();
        let mut versioned = toml::Table::new();
        versioned.insert(
            "version".to_string(),
            toml::Value::String("0.7.2".to_string()),
        );
        workspace.insert("present".to_string(), toml::Value::Table(versioned));

        let mut inheriting = toml::Table::new();
        inheriting.insert("workspace".to_string(), toml::Value::Boolean(true));
        assert_eq!(has_version("present", &inheriting, &workspace), Some(true));
        assert_eq!(has_version("missing", &inheriting, &workspace), None);

        let mut path_only = toml::Table::new();
        path_only.insert("path".to_string(), toml::Value::String("../x".to_string()));
        assert_eq!(has_version("x", &path_only, &workspace), Some(false));
    }

    /// WHY: a bare version string and a full specification are the same edge, and
    /// the earlier shell rule skipped the string form, so a crate declaring
    /// `vyre-foundation = "0.7.2"` was unchecked.
    #[test]
    fn a_bare_version_string_is_a_dependency_entry() {
        let table: toml::Table = toml::from_str(
            "[dependencies]\nvyre-foundation = \"0.7.2\"\nvyre-driver = { path = \"../d\" }\n",
        )
        .expect("literal parses");
        let found = entries(&table, "dependencies");
        assert_eq!(found.len(), 2);
        let foundation = found
            .iter()
            .find(|(key, _)| key == "vyre-foundation")
            .expect("entry present");
        assert!(foundation.1.contains_key("version"));
    }

    /// WHY: findings carry line numbers so a reader can jump to the key. The
    /// locator reads text rather than a parsed table, so it must track the table
    /// header and take the first occurrence of a key.
    #[test]
    fn keys_are_located_under_their_table() {
        let text = "[package]\nname = \"x\"\n\n[dependencies]\nvyre-libs = { path = \"../l\" }\n\
                    \n[target.'cfg(unix)'.dependencies]\nlibc = \"0.2\"\n";
        let lines = dep_lines(text);
        assert_eq!(
            lines.get(&("dependencies".to_string(), "vyre-libs".to_string())),
            Some(&5)
        );
        assert_eq!(
            lines.get(&(
                "target.'cfg(unix)'.dependencies".to_string(),
                "libc".to_string()
            )),
            Some(&8)
        );
        assert_eq!(
            lines.get(&("package".to_string(), "name".to_string())),
            Some(&2)
        );
    }
}
