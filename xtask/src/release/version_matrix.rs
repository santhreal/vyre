//! Hold the release version story to the manifests, lockfile and release docs.

use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::artifact_gate::Inspection;
use crate::manifest_walk;
use crate::release::release_train;

/// The two artifacts this gate owns, relative to the workspace root.
const MATRIX: &str = "release/evidence/version/version-matrix.json";
const TAG_PLAN: &str = "release/evidence/version/release-tag-plan.json";

/// The command an operator must see green before creating either release tag.
///
/// It used to name `version-matrix --output release/evidence/version/version-matrix.json`, a
/// flag that no longer exists because the artifact path is owned by the gate.
const REQUIRED_GATE_BEFORE_TAG: &str = "cargo_full run --bin xtask -- version-matrix && \
     cargo_full run --bin xtask -- vyre-release-gate && scripts/apply-branch-protection.sh main";

const MAX_VERSION_EVIDENCE_TEXT_BYTES: u64 = 8_388_608;

#[derive(Debug, Serialize)]
struct VersionMatrix {
    schema_version: u32,
    requested_vyre_release: &'static str,
    tag_story: ReleaseTagStory,
    required_release_packages: Vec<String>,
    missing_required_release_packages: Vec<String>,
    crates: Vec<CrateVersion>,
    dependency_hints: Vec<DependencyVersionHint>,
    lockfile_packages: Vec<LockfilePackageVersion>,
    release_doc_tag_findings: Vec<ReleaseDocTagFinding>,
    release_note_token_findings: Vec<ReleaseNoteTokenFinding>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReleaseTagStory {
    vyre_rc_tag: &'static str,
    vyre_tag: &'static str,
    policy: &'static str,
    required_in_release_notes: Vec<&'static str>,
    required_in_packaging: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct ReleaseTagPlan<'a> {
    schema_version: u32,
    vyre_rc_tag: &'a str,
    vyre_tag: &'a str,
    tag_creation_order: Vec<&'a str>,
    required_gate_before_rc_tag: &'a str,
    required_gate_before_tag: &'a str,
    version_matrix_blocker_count: usize,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CrateVersion {
    package: String,
    version: String,
    manifest: String,
    release_group: &'static str,
    publishable: bool,
}

#[derive(Debug, Serialize)]
struct DependencyVersionHint {
    manifest: String,
    dependency: String,
    version: String,
    expected: &'static str,
    release_group: &'static str,
}

#[derive(Debug, Serialize)]
struct LockfilePackageVersion {
    lockfile: String,
    package: String,
    version: String,
    expected: &'static str,
    release_group: &'static str,
}

#[derive(Debug, Serialize)]
struct ReleaseDocTagFinding {
    path: String,
    line: usize,
    text: String,
}

#[derive(Debug, Serialize)]
struct ReleaseNoteTokenFinding {
    path: String,
    issue: String,
}

crate::artifact_gate! {
    /// Holds the version-story evidence to the manifests, lockfile and release docs.
    VersionMatrixGate,
    name: "version-matrix",
    help: "Regenerate release/evidence/version/version-matrix.json and release/evidence/version/release-tag-plan.json from \
       the workspace manifests, Cargo.lock and the release docs, and report each line the \
       committed copies disagree on. Proves every publishable crate carries the version the \
       release train declares, that every required release package is present at its expected \
       version, that pinned dependency and lockfile versions match, that no release doc gives \
       a bare tag command, and that release notes carry no stale version token. Proves nothing \
       about what is published on a registry: every fact here is read from this checkout.",
    inspect: |ctx| inspect(&ctx.root),
}

/// The version story the tree tells, and the two artifacts recording it.
fn inspect(vyre_root: &Path) -> Inspection {
    let mut inspection = Inspection::new();
    let matrix = build_matrix(vyre_root);
    for blocker in &matrix.blockers {
        inspection.blocked(
            MATRIX,
            blocker.clone(),
            "Correct the version, dependency pin, lockfile entry or release doc the sentence \
             names. The blocker is recorded in the artifact as well, so committing it does not \
             clear this finding.",
        );
    }
    inspection.generates(MATRIX, &matrix);
    inspection.generates(TAG_PLAN, &release_tag_plan(&matrix));
    inspection
}

/// Every version fact this gate reads out of the checkout.
fn build_matrix(vyre_root: &Path) -> VersionMatrix {
    let mut crates = Vec::new();
    let mut collection_blockers = Vec::new();
    collect_workspace_versions(vyre_root, "vyre", &mut crates, &mut collection_blockers);
    crates.sort_by(|left, right| left.package.cmp(&right.package));
    let missing_required_release_packages = missing_required_release_packages(&crates);
    let mut dependency_hints = Vec::new();
    collect_workspace_dependency_hints(vyre_root, &mut dependency_hints, &mut collection_blockers);
    dependency_hints.sort_by(|left, right| {
        left.manifest
            .cmp(&right.manifest)
            .then(left.dependency.cmp(&right.dependency))
    });
    let mut lockfile_packages = Vec::new();
    collect_lockfile_versions(
        &vyre_root.join("Cargo.lock"),
        &mut lockfile_packages,
        &mut collection_blockers,
    );
    lockfile_packages.sort_by(|left, right| {
        left.lockfile
            .cmp(&right.lockfile)
            .then(left.package.cmp(&right.package))
    });

    let mut blockers = Vec::new();
    blockers.extend(collection_blockers);
    for krate in &crates {
        if !krate.publishable {
            continue;
        }
        match krate.release_group {
            "vyre" if krate.version != release_train::vyre_version() => blockers.push(format!(
                "{} is version {}, requested Vyre release is {}",
                krate.package,
                krate.version,
                release_train::vyre_version()
            )),
            _ => {}
        }
    }
    blockers.extend(
        missing_required_release_packages
            .iter()
            .map(|package| format!("missing required release package `{package}`")),
    );
    for hint in &dependency_hints {
        if hint.version != hint.expected {
            blockers.push(format!(
                "{} dependency `{}` is version {}, expected {} for {} release",
                hint.manifest, hint.dependency, hint.version, hint.expected, hint.release_group
            ));
        }
    }
    for package in &lockfile_packages {
        if package.version != package.expected {
            blockers.push(format!(
                "{} lock package `{}` is version {}, expected {} for {} release",
                package.lockfile,
                package.package,
                package.version,
                package.expected,
                package.release_group
            ));
        }
    }
    let (release_doc_tag_findings, doc_scan_blockers) = scan_bare_release_tags(vyre_root);
    blockers.extend(doc_scan_blockers);
    for finding in &release_doc_tag_findings {
        blockers.push(format!(
            "{}:{} uses an ambiguous bare release tag command `{}`",
            finding.path, finding.line, finding.text
        ));
    }
    let release_note_token_findings = scan_release_note_tokens(vyre_root);
    for finding in &release_note_token_findings {
        blockers.push(format!(
            "{} has a release-note version issue: {}",
            finding.path, finding.issue
        ));
    }

    VersionMatrix {
        schema_version: 2,
        requested_vyre_release: release_train::vyre_version(),
        tag_story: release_tag_story(),
        required_release_packages: release_train::required_release_packages()
            .into_iter()
            .map(|(package, version, _)| format!("{package}@{version}"))
            .collect(),
        missing_required_release_packages,
        crates,
        dependency_hints,
        lockfile_packages,
        release_doc_tag_findings,
        release_note_token_findings,
        blockers,
    }
}

fn missing_required_release_packages(crates: &[CrateVersion]) -> Vec<String> {
    release_train::required_release_packages()
        .into_iter()
        .filter_map(|(required_package, expected_version, expected_group)| {
            let present = crates.iter().any(|krate| {
                krate.package == required_package
                    && krate.version == expected_version
                    && krate.release_group == expected_group
            });
            (!present).then(|| format!("{required_package}@{expected_version}:{expected_group}"))
        })
        .collect()
}

/// The tag plan the version story implies, recorded beside the matrix.
fn release_tag_plan<'a>(matrix: &'a VersionMatrix) -> ReleaseTagPlan<'a> {
    let tag_story = &matrix.tag_story;
    ReleaseTagPlan {
        schema_version: 2,
        vyre_rc_tag: tag_story.vyre_rc_tag,
        vyre_tag: tag_story.vyre_tag,
        tag_creation_order: release_train::tag_creation_order().to_vec(),
        required_gate_before_rc_tag: REQUIRED_GATE_BEFORE_TAG,
        required_gate_before_tag: REQUIRED_GATE_BEFORE_TAG,
        version_matrix_blocker_count: matrix.blockers.len(),
        blockers: matrix.blockers.clone(),
    }
}

fn scan_bare_release_tags(vyre_root: &Path) -> (Vec<ReleaseDocTagFinding>, Vec<String>) {
    let mut findings = Vec::new();
    let mut blockers = Vec::new();
    let bare_tag = format!("v{}", release_train::vyre_version());
    let bare_rc_tag = format!("{bare_tag}-rc.1");
    for path in release_doc_paths(vyre_root) {
        let text = match read_text_bounded(&path) {
            Ok(text) => text,
            Err(error) => {
                blockers.push(format!(
                    "failed to read release doc `{}` for tag scan: {error}",
                    path.display()
                ));
                continue;
            }
        };
        for (line_index, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if is_bare_release_tag_command(trimmed, &bare_tag, &bare_rc_tag) {
                findings.push(ReleaseDocTagFinding {
                    path: path.display().to_string(),
                    line: line_index + 1,
                    text: trimmed.to_string(),
                });
            }
        }
    }
    (findings, blockers)
}

fn is_bare_release_tag_command(line: &str, bare_tag: &str, bare_rc_tag: &str) -> bool {
    let mut words = line.split_whitespace();
    let tag = match (words.next(), words.next()) {
        (Some("git"), Some("tag")) => words.next(),
        (Some("git"), Some("push")) if words.next() == Some("origin") => words.next(),
        (Some("gh"), Some("release")) if words.next() == Some("create") => words.next(),
        _ => None,
    };
    tag.is_some_and(|tag| tag == bare_tag || tag == bare_rc_tag)
}

/// The release prose a reader is sent to for this version.
///
/// This resolved `docs/release/v<version>.md`, a per-version notes page the
/// mdbook carried. The book is gone and nothing regenerates that page, so every
/// run opened with an unreadable-document finding and the token scan judged a
/// file that does not exist. The changelog is the surviving release-notes
/// surface and `xtask release-docs` writes its identity preamble from the
/// same release train these tokens come from.
fn current_release_notes_path(vyre_root: &Path) -> PathBuf {
    vyre_root.join("CHANGELOG.md")
}

fn scan_release_note_tokens(vyre_root: &Path) -> Vec<ReleaseNoteTokenFinding> {
    let mut findings = Vec::new();
    for path in [
        vyre_root.join("release/evidence/docs/release-notes.md"),
        vyre_root.join("release/evidence/docs/release-notes-version-story.md"),
        current_release_notes_path(vyre_root),
    ] {
        let text = match read_text_bounded(&path) {
            Ok(text) => text,
            Err(error) => {
                findings.push(ReleaseNoteTokenFinding {
                    path: path.display().to_string(),
                    issue: format!("required release-note document unreadable: {error}"),
                });
                continue;
            }
        };
        for required in release_tag_story().required_in_release_notes {
            if !text.contains(required) {
                findings.push(ReleaseNoteTokenFinding {
                    path: path.display().to_string(),
                    issue: format!("missing required token `{required}`"),
                });
            }
        }
        append_release_version_issues(&path, &text, &mut findings);
    }
    for path in [vyre_root.join("release/evidence/docs/crate-metadata-proof.md")] {
        match read_text_bounded(&path) {
            Ok(text) => append_release_version_issues(&path, &text, &mut findings),
            Err(error) => findings.push(ReleaseNoteTokenFinding {
                path: path.display().to_string(),
                issue: format!("required release evidence document unreadable: {error}"),
            }),
        }
    }
    findings
}

fn append_release_version_issues(
    path: &Path,
    text: &str,
    findings: &mut Vec<ReleaseNoteTokenFinding>,
) {
    for line in text.lines() {
        for issue in release_note_version_issues(line, release_train::vyre_version()) {
            findings.push(ReleaseNoteTokenFinding {
                path: path.display().to_string(),
                issue,
            });
        }
    }
}

fn release_note_version_issues(line: &str, vyre_version: &str) -> Vec<String> {
    const VYRE_DECLARATION: &str = "- Vyre release:";
    const PACKAGE_DECLARATION: &str = "- Required version-matrix packages:";

    let trimmed = line.trim();
    let mut issues = Vec::new();
    if let Some(rest) = trimmed.strip_prefix(VYRE_DECLARATION) {
        match rest
            .split_once('`')
            .and_then(|(_, rest)| rest.split_once('`'))
        {
            Some((quoted, _)) if quoted != vyre_version => issues.push(format!(
                "Vyre release has `{quoted}`, expected `{vyre_version}`"
            )),
            None => issues.push("Vyre release declaration has no quoted version".to_string()),
            _ => {}
        }
    }
    if !trimmed.starts_with(PACKAGE_DECLARATION) {
        return issues;
    }
    for token in trimmed.split('`').skip(1).step_by(2) {
        let Some((package, version)) = token.split_once('@') else {
            continue;
        };
        let Some((expected, _)) = expected_dependency_version(package) else {
            continue;
        };
        if version != expected {
            issues.push(format!(
                "`{package}` declares version `{version}`, expected `{expected}`"
            ));
        }
    }
    issues
}

/// Documents scanned for a bare, ambiguous release tag command.
///
/// Three of the five entries here were one release runbook twice and a
/// checklist beside it, all deleted with the book, so the scan spent its
/// blockers reporting that it could not read them and never reached a document
/// that could carry a bad tag command. The duplicate also double-counted every
/// finding in that one file.
fn release_doc_paths(vyre_root: &Path) -> Vec<PathBuf> {
    vec![
        vyre_root.join("README.md"),
        vyre_root.join("CONTRIBUTING.md"),
        current_release_notes_path(vyre_root),
        vyre_root.join("release/evidence/docs/release-notes.md"),
        vyre_root.join("release/evidence/docs/release-notes-version-story.md"),
    ]
}

fn release_tag_story() -> ReleaseTagStory {
    ReleaseTagStory {
        vyre_rc_tag: release_train::vyre_rc_tag(),
        vyre_tag: release_train::vyre_tag(),
        policy: release_train::tag_policy(),
        required_in_release_notes: release_train::required_release_note_tokens(),
        required_in_packaging: release_train::required_packaging_steps(),
    }
}

fn collect_lockfile_versions(
    path: &Path,
    packages: &mut Vec<LockfilePackageVersion>,
    blockers: &mut Vec<String>,
) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "failed to read lockfile `{}`: {error}",
                path.display()
            ));
            return;
        }
    };
    let value = match toml::from_str::<toml::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            blockers.push(format!(
                "failed to parse lockfile `{}`: {error}",
                path.display()
            ));
            return;
        }
    };
    let Some(entries) = value.get("package").and_then(toml::Value::as_array) else {
        return;
    };
    for entry in entries {
        let Some(table) = entry.as_table() else {
            continue;
        };
        let Some(name) = table.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let Some((expected, release_group)) = expected_dependency_version(name) else {
            continue;
        };
        let Some(version) = table.get("version").and_then(toml::Value::as_str) else {
            continue;
        };
        packages.push(LockfilePackageVersion {
            lockfile: path.display().to_string(),
            package: name.to_string(),
            version: version.to_string(),
            expected,
            release_group,
        });
    }
}

fn collect_workspace_dependency_hints(
    root: &Path,
    hints: &mut Vec<DependencyVersionHint>,
    blockers: &mut Vec<String>,
) {
    let root_manifest = root.join("Cargo.toml");
    collect_one_dependency_hints(&root_manifest, hints, blockers);
    let text = match read_text_bounded(&root_manifest) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "failed to read workspace manifest `{}` for dependency hints: {error}",
                root_manifest.display()
            ));
            return;
        }
    };
    let value = match toml::from_str::<toml::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            blockers.push(format!(
                "failed to parse workspace manifest `{}` for dependency hints: {error}",
                root_manifest.display()
            ));
            return;
        }
    };
    let Some(members) = value
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
    else {
        return;
    };
    for member in members {
        let Some(member) = member.as_str() else {
            continue;
        };
        if member.contains('*') {
            continue;
        }
        collect_one_dependency_hints(&root.join(member).join("Cargo.toml"), hints, blockers);
    }
}

fn collect_one_dependency_hints(
    path: &Path,
    hints: &mut Vec<DependencyVersionHint>,
    blockers: &mut Vec<String>,
) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "failed to read dependency manifest `{}`: {error}",
                path.display()
            ));
            return;
        }
    };
    let value = match toml::from_str::<toml::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            blockers.push(format!(
                "failed to parse dependency manifest `{}`: {error}",
                path.display()
            ));
            return;
        }
    };
    // Only NORMAL dependencies constrain release-version coherence. `dev-` and
    // `build-dependencies` are non-public and non-transitive to consumers, and
    // internal ones are deliberately pinned to an already-published version to
    // break the publish cycle (a crate that is published BEFORE its dev-dep
    // cannot require the not-yet-published release version). Requiring those to
    // equal the release version is a false positive that would break the very
    // publish order package-readiness validates, so they are excluded here.
    collect_dependency_table(path, "dependencies", &value, hints);
    if let Some(workspace) = value.get("workspace") {
        collect_dependency_table(path, "dependencies", workspace, hints);
    }
}

fn collect_dependency_table(
    manifest: &Path,
    table_name: &str,
    value: &toml::Value,
    hints: &mut Vec<DependencyVersionHint>,
) {
    let Some(table) = value.get(table_name).and_then(toml::Value::as_table) else {
        return;
    };
    for (dependency, spec) in table {
        let Some((expected, release_group)) = expected_dependency_version(dependency) else {
            continue;
        };
        let version = match spec {
            toml::Value::String(version) => Some(version.as_str()),
            toml::Value::Table(table) => table.get("version").and_then(toml::Value::as_str),
            _ => None,
        };
        let Some(version) = version else {
            continue;
        };
        hints.push(DependencyVersionHint {
            manifest: manifest.display().to_string(),
            dependency: dependency.clone(),
            version: version.to_string(),
            expected,
            release_group,
        });
    }
}

fn expected_dependency_version(dependency: &str) -> Option<(&'static str, &'static str)> {
    if matches!(
        dependency,
        "vyre-conform" | "vyre-bench" | "vyre-foundation-fuzz"
    ) {
        return None;
    }
    if dependency == "vyre" || dependency.starts_with("vyre-") {
        return Some((release_train::vyre_version(), "vyre"));
    }
    None
}

fn collect_workspace_versions(
    root: &Path,
    release_group: &'static str,
    versions: &mut Vec<CrateVersion>,
    blockers: &mut Vec<String>,
) {
    let workspace_version = workspace_package_version(root, blockers);
    let root_manifest = root.join("Cargo.toml");
    collect_one_version(
        &root_manifest,
        release_group,
        workspace_version.as_deref(),
        versions,
        blockers,
    );
    let text = match crate::output_arg::read_text_bounded(
        &root_manifest,
        manifest_walk::MAX_MANIFEST_BYTES,
        "version evidence",
    ) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "failed to read workspace manifest `{}` for version collection: {error}",
                root_manifest.display()
            ));
            return;
        }
    };
    let document = match toml::from_str::<toml::Table>(&text) {
        Ok(document) => document,
        Err(error) => {
            blockers.push(format!(
                "failed to parse workspace manifest `{}` for version collection: {error}",
                root_manifest.display()
            ));
            return;
        }
    };
    let Some(members) = document
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
    else {
        return;
    };
    for member in members {
        let Some(member) = member.as_str() else {
            continue;
        };
        if member.contains('*') {
            continue;
        }
        collect_one_version(
            &root.join(member).join("Cargo.toml"),
            release_group,
            workspace_version.as_deref(),
            versions,
            blockers,
        );
    }
}

fn workspace_package_version(root: &Path, blockers: &mut Vec<String>) -> Option<String> {
    crate::manifest_walk::workspace_package(root, "version evidence", blockers)?
        .get("version")
        .and_then(toml::Value::as_str)
        .map(str::to_string)
}

/// One crate's declared version, resolving `version.workspace = true`.
///
/// The read, parse and name lookup were a fourth copy of a sequence three other
/// release generators also carried. A `[package]` without a `name` used to be
/// skipped in silence here; `parse_package_manifest` reports it.
fn collect_one_version(
    path: &Path,
    release_group: &'static str,
    workspace_version: Option<&str>,
    versions: &mut Vec<CrateVersion>,
    blockers: &mut Vec<String>,
) {
    let manifest = match manifest_walk::parse_package_manifest(path, "version evidence") {
        Ok(Some(manifest)) => manifest,
        Ok(None) => return,
        Err(error) => {
            blockers.push(error);
            return;
        }
    };
    let Some(package) = manifest
        .document
        .get("package")
        .and_then(toml::Value::as_table)
    else {
        return;
    };
    let version = package
        .get("version")
        .and_then(toml::Value::as_str)
        .or_else(|| {
            package
                .get("version")
                .and_then(|value| value.get("workspace"))
                .and_then(toml::Value::as_bool)
                .filter(|workspace| *workspace)
                .and_then(|_| workspace_version)
        });
    let Some(version) = version else {
        return;
    };
    let publishable = !matches!(package.get("publish"), Some(toml::Value::Boolean(false)))
        && manifest.name != "vyre-conform";
    versions.push(CrateVersion {
        package: manifest.name,
        version: version.to_string(),
        manifest: path.display().to_string(),
        release_group,
        publishable,
    });
}

/// Read version evidence text under this generator's cap.
///
/// The bound check itself lives in `output_arg`; this only binds the cap and the
/// error context for the nine call sites below.
fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(path, MAX_VERSION_EVIDENCE_TEXT_BYTES, "version evidence")
}

#[cfg(test)]
mod tests {
    use super::{is_bare_release_tag_command, release_note_version_issues, MATRIX, TAG_PLAN};

    /// The tag gate rejects every supported command form for the active bare final tag.
    #[test]
    fn bare_final_tag_commands_are_rejected() {
        for command in [
            "git tag v0.7.0",
            "git push origin v0.7.0",
            "gh release create v0.7.0",
        ] {
            assert!(is_bare_release_tag_command(
                command,
                "v0.7.0",
                "v0.7.0-rc.1"
            ));
        }
    }

    /// The tag gate rejects every supported command form for the active bare RC tag.
    #[test]
    fn bare_rc_tag_commands_are_rejected() {
        for command in [
            "git tag v0.7.0-rc.1",
            "git push origin v0.7.0-rc.1",
            "gh release create v0.7.0-rc.1",
        ] {
            assert!(is_bare_release_tag_command(
                command,
                "v0.7.0",
                "v0.7.0-rc.1"
            ));
        }
    }

    /// Product-scoped tags and older release commands do not trigger the current bare-tag gate.
    #[test]
    fn product_scoped_and_historical_tags_are_allowed() {
        for command in [
            "git tag vyre-v0.7.0",
            "git push origin another-product-v0.1.0",
            "gh release create another-product-v0.1.0",
            "git tag v0.6.0",
        ] {
            assert!(!is_bare_release_tag_command(
                command,
                "v0.7.0",
                "v0.7.0-rc.1"
            ));
        }
    }

    /// Current release declarations and package tokens must remain contradiction-free.
    #[test]
    fn current_release_version_story_is_accepted() {
        let version = crate::release::release_train::vyre_version();
        for line in [
            format!("- Vyre release: `{version}`"),
            format!(
                "- Required version-matrix packages: `vyre@{version}`, `vyre-driver-cuda@{version}`, and `vyre-driver-wgpu@{version}`."
            ),
        ] {
            assert_eq!(
                release_note_version_issues(&line, version),
                Vec::<String>::new(),
                "Fix: the active release story must not report a contradiction for `{line}`"
            );
        }
    }

    #[test]
    fn authoritative_descriptor_declares_exact_version_matrix_artifacts() {
        let descriptor = crate::gate_metadata::descriptor_by_name("version-matrix");
        let mut expected: Vec<&str> = vec![MATRIX, TAG_PLAN];
        expected.sort_unstable();
        let mut actual: Vec<&str> = descriptor.artifacts.to_vec();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "version-matrix gate descriptor must declare exactly the canonical version evidence artifacts"
        );
    }
}
