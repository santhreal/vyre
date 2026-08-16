//! Every example crate proves the capability it demonstrates.
//!
//! An example is the only place a published surface is exercised the way a
//! consumer exercises it: from outside the workspace, against version
//! requirements, with no access to `pub(crate)`. That makes it the only check
//! on the parts of the surface nothing in the workspace can reach, and it makes
//! an example nobody builds worse than no example, because it reads as a
//! working recipe long after it stopped being one. Both of the crates this gate
//! restored had rotted while tracked: one pinned `thiserror = "=2.0.18"` after
//! the workspace moved to `=2.0.19`, so it could not resolve, and one passed
//! `Arc<[u32]>` where the error it constructs takes `Vec<u32>`, so it could not
//! compile.
//!
//! Subjects come from `examples/` at run time, so a new example is covered the
//! day it is added. A directory that carries files and no manifest is reported
//! rather than skipped, and a template placeholder this gate cannot render is
//! reported rather than substituted with something invented, so a subject
//! cannot be added in a shape the gate silently ignores.
//!
//! An example that does not build is a finding and never a `GateError`: a gate
//! error takes a whole workflow red with a message about cargo, and a broken
//! example is exactly the defect this gate exists to count.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};

/// Directory holding every subject.
const EXAMPLES: &str = "examples";

/// Tracked Rust lines one example may carry.
///
/// An example a reviewer cannot read in one sitting has stopped being a recipe
/// and become a library with one consumer.
const MAX_EXAMPLE_RUST_LINES: usize = 300;

/// Manifest name of a template that is rendered before it is built.
const TEMPLATE_MANIFEST: &str = "Cargo.toml.liquid";

/// Values this gate renders template placeholders with.
///
/// A placeholder outside this table is a finding, because rendering an unknown
/// name to anything at all produces a crate whose failure says nothing about
/// the template.
const PLACEHOLDERS: &[(&str, &str)] = &[
    ("{{crate_name}}", "vyre-example-template-probe"),
    ("{{crate_name_snake}}", "vyre_example_template_probe"),
    ("{{gh_org}}", "santhreal"),
];

/// What an example directory is, decided by the manifest it carries.
enum Subject {
    /// A standalone crate built where it sits.
    Crate,
    /// A scaffold rendered into a crate before it is built.
    Template,
}

/// Every example builds, and every capability it claims still holds.
pub struct ExampleCapability;

impl Gate for ExampleCapability {
    fn name(&self) -> &'static str {
        "example-capability"
    }

    fn help(&self) -> &'static str {
        "Build every example crate outside the workspace and run what it asserts"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tracked = tracked_example_paths(&ctx.root)?;
        let mut report = Report::clean();
        let mut subjects = 0usize;
        for (name, paths) in &tracked {
            let directory = format!("{EXAMPLES}/{name}");
            let subject = if paths.contains(&format!("{directory}/Cargo.toml")) {
                Subject::Crate
            } else if paths.contains(&format!("{directory}/{TEMPLATE_MANIFEST}")) {
                Subject::Template
            } else {
                report.find(Finding::in_file(
                    &directory,
                    format!("`{directory}` tracks {} file(s) and no manifest", paths.len()),
                    "give the example a Cargo.toml, or a Cargo.toml.liquid when it is a scaffold, so something can build it",
                ));
                continue;
            };
            subjects += 1;
            report
                .findings
                .extend(line_cap_findings(&ctx.root, &directory, paths));
            match subject {
                Subject::Crate => {
                    report
                        .findings
                        .extend(crate_findings(&ctx.root, &directory, paths));
                }
                Subject::Template => {
                    report
                        .findings
                        .extend(template_findings(&ctx.root, &directory, paths)?);
                }
            }
        }
        if subjects == 0 {
            return Err(GateError::new(
                format!("`{EXAMPLES}/` holds no example this gate can build"),
                "run this gate in a checkout that tracks the example crates",
            ));
        }
        report.note(format!("built {subjects} example subject(s)"));
        Ok(report)
    }
}

/// Tracked paths under `examples/`, grouped by the directory that owns them.
fn tracked_example_paths(root: &Path) -> Result<BTreeMap<String, Vec<String>>, GateError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--", EXAMPLES])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("could not run `git ls-files`: {error}"),
                "run this gate inside a git work tree; the tracked set is the oracle",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!("`git ls-files -- {EXAMPLES}` failed"),
            "run this gate inside a git work tree; the tracked set is the oracle",
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in text.split('\0').filter(|entry| !entry.is_empty()) {
        let Some(rest) = path.strip_prefix(&format!("{EXAMPLES}/")) else {
            continue;
        };
        let Some((name, _)) = rest.split_once('/') else {
            continue;
        };
        grouped
            .entry(name.to_string())
            .or_default()
            .push(path.to_string());
    }
    Ok(grouped)
}

/// The tracked Rust of one example, held to the review cap.
fn line_cap_findings(root: &Path, directory: &str, paths: &[String]) -> Vec<Finding> {
    let lines: usize = paths
        .iter()
        .filter(|path| path.ends_with(".rs"))
        .filter_map(|path| std::fs::read_to_string(root.join(path)).ok())
        .map(|text| text.lines().count())
        .sum();
    if lines > MAX_EXAMPLE_RUST_LINES {
        return vec![Finding::in_file(
            directory,
            format!("`{directory}` tracks {lines} Rust lines, over the {MAX_EXAMPLE_RUST_LINES} line cap"),
            "move what the example is teaching into the crate that owns it, and keep the example to the recipe",
        )];
    }
    Vec::new()
}

/// One standalone example crate: isolated, locked, and passing what it asserts.
fn crate_findings(root: &Path, directory: &str, paths: &[String]) -> Vec<Finding> {
    let manifest = format!("{directory}/Cargo.toml");
    let mut findings = workspace_isolation_findings(root, &manifest);
    if !paths.contains(&format!("{directory}/Cargo.lock")) {
        findings.push(Finding::in_file(
            &manifest,
            format!("`{directory}` tracks no Cargo.lock"),
            "commit the lockfile beside the manifest so the example builds the same versions everywhere",
        ));
        return findings;
    }
    findings.extend(cargo_findings(
        root,
        &manifest,
        &["test", "--manifest-path", &manifest, "--locked"],
    ));
    if paths.contains(&format!("{directory}/src/main.rs")) {
        findings.extend(cargo_findings(
            root,
            &manifest,
            &["run", "--manifest-path", &manifest, "--locked", "--quiet"],
        ));
    }
    findings
}

/// One template: every placeholder known, and the rendered crate passing.
fn template_findings(
    root: &Path,
    directory: &str,
    paths: &[String],
) -> Result<Vec<Finding>, GateError> {
    let manifest = format!("{directory}/{TEMPLATE_MANIFEST}");
    let mut findings = workspace_isolation_findings(root, &manifest);
    let mut rendered: BTreeMap<PathBuf, String> = BTreeMap::new();
    for path in paths {
        let Ok(text) = std::fs::read_to_string(root.join(path)) else {
            continue;
        };
        for unknown in unknown_placeholders(&text) {
            findings.push(Finding::in_file(
                path,
                format!("`{path}` uses the placeholder `{unknown}`, which this gate cannot render"),
                "teach PLACEHOLDERS in xtask/src/gates/example_capability.rs the value it renders to, or drop the placeholder",
            ));
        }
        let mut relative = PathBuf::from(path.trim_start_matches(&format!("{directory}/")));
        if relative.file_name().is_some_and(|name| name == TEMPLATE_MANIFEST) {
            relative.set_file_name("Cargo.toml");
        }
        rendered.insert(relative, render(&text));
    }
    if !findings.is_empty() {
        return Ok(findings);
    }
    let target = std::env::temp_dir().join(format!(
        "vyre-example-template-{}-{}",
        directory.replace('/', "-"),
        std::process::id()
    ));
    if target.exists() {
        std::fs::remove_dir_all(&target).map_err(|error| {
            GateError::new(
                format!("could not clear `{}`: {error}", target.display()),
                "remove the stale render directory by hand",
            )
        })?;
    }
    let Some(rendered_manifest_text) = rendered.get(Path::new("Cargo.toml")).cloned() else {
        findings.push(Finding::in_file(
            &manifest,
            format!("`{directory}` renders no Cargo.toml"),
            "name the manifest template Cargo.toml.liquid so rendering produces a manifest",
        ));
        return Ok(findings);
    };
    for (relative, text) in &rendered {
        let path = target.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                GateError::new(
                    format!("could not create `{}`: {error}", parent.display()),
                    "make the temporary directory writable",
                )
            })?;
        }
        std::fs::write(&path, text).map_err(|error| {
            GateError::new(
                format!("could not write `{}`: {error}", path.display()),
                "make the temporary directory writable",
            )
        })?;
    }
    let rendered_manifest = target.join("Cargo.toml");
    let patched = format!(
        "{rendered_manifest_text}{}",
        checkout_patch_section(root, &rendered_manifest_text)
    );
    std::fs::write(&rendered_manifest, patched).map_err(|error| {
        GateError::new(
            format!("could not write `{}`: {error}", rendered_manifest.display()),
            "make the temporary directory writable",
        )
    })?;
    let manifest_argument = rendered_manifest.to_string_lossy().into_owned();
    findings.extend(cargo_findings(
        root,
        &manifest,
        &["test", "--manifest-path", &manifest_argument],
    ));
    std::fs::remove_dir_all(&target).map_err(|error| {
        GateError::new(
            format!("could not remove `{}`: {error}", target.display()),
            "remove the render directory by hand",
        )
    })?;
    Ok(findings)
}

/// A manifest that does not declare its own workspace inherits the state the
/// example exists to prove it does not need.
fn workspace_isolation_findings(root: &Path, manifest: &str) -> Vec<Finding> {
    let Ok(text) = std::fs::read_to_string(root.join(manifest)) else {
        return vec![Finding::in_file(
            manifest,
            format!("`{manifest}` cannot be read"),
            "restore the manifest of the example crate",
        )];
    };
    if text.lines().any(|line| line.trim() == "[workspace]") {
        return Vec::new();
    }
    vec![Finding::in_file(
        manifest,
        format!("`{manifest}` declares no [workspace] of its own"),
        "add an empty [workspace] table so the example resolves against its own manifest and not the repository workspace",
    )]
}

/// Run one cargo invocation over an example and report what it says.
fn cargo_findings(root: &Path, subject: &str, arguments: &[&str]) -> Vec<Finding> {
    let cargo = crate::output_arg::cargo_runner(root);
    let invocation = format!("cargo {}", arguments.join(" "));
    let output = Command::new(&cargo)
        .args(arguments)
        .current_dir(root)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            return vec![Finding::in_file(
                subject,
                format!("`{invocation}` could not be spawned: {error}"),
                "restore the cargo_full wrapper at the workspace root",
            )];
        }
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    let mut findings: Vec<Finding> = text
        .lines()
        .filter_map(|line| line.trim().strip_prefix("test "))
        .filter_map(|rest| rest.strip_suffix(" ... FAILED"))
        .map(|name| {
            Finding::in_file(
                subject,
                format!("`{subject}` test `{name}` failed"),
                "fix the capability the example demonstrates, and never weaken the assertion to match it",
            )
        })
        .collect();
    if !output.status.success() && findings.is_empty() {
        findings.push(Finding::in_file(
            subject,
            format!(
                "`{invocation}` exited {}: {}",
                output.status.code().unwrap_or(-1),
                text.lines()
                    .filter(|line| line.contains("error"))
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            "run the same cargo command by hand and fix what the example does against the published surface",
        ));
    }
    findings
}

/// Substitute every known placeholder.
fn render(text: &str) -> String {
    let mut rendered = text.to_string();
    for (placeholder, value) in PLACEHOLDERS {
        rendered = rendered.replace(placeholder, value);
    }
    rendered
}

/// Placeholders in `text` that this gate has no value for.
fn unknown_placeholders(text: &str) -> Vec<String> {
    let mut unknown = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("{{") {
        let after = &rest[open..];
        let Some(close) = after.find("}}") else {
            break;
        };
        let placeholder = &after[..close + 2];
        if !PLACEHOLDERS
            .iter()
            .any(|(known, _)| *known == placeholder)
            && !unknown.iter().any(|found| found == placeholder)
        {
            unknown.push(placeholder.to_string());
        }
        rest = &after[close + 2..];
    }
    unknown
}

/// A `[patch.crates-io]` table pointing every dependency this checkout provides
/// at the checkout, so a rendered template is built against the tree that ships
/// it rather than against the registry.
fn checkout_patch_section(root: &Path, manifest: &str) -> String {
    let mut section = String::from("\n[patch.crates-io]\n");
    for name in dependency_names(manifest) {
        if !root.join(&name).join("Cargo.toml").is_file() {
            continue;
        }
        section.push_str(&format!(
            "{name} = {{ path = \"{}\" }}\n",
            root.join(&name).display()
        ));
    }
    section
}

/// Dependency names a manifest declares, in declaration order.
fn dependency_names(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_dependencies = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_dependencies = matches!(
                trimmed,
                "[dependencies]" | "[dev-dependencies]" | "[build-dependencies]"
            );
            continue;
        }
        if !in_dependencies {
            continue;
        }
        let Some((name, _)) = trimmed.split_once('=') else {
            continue;
        };
        let name = name.trim().to_string();
        if !name.is_empty() && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the placeholder table is the one part of this gate that a template
    /// author can outgrow. A template that introduces `{{license}}` and renders
    /// to a crate with a literal `{{license}}` in its manifest fails with a TOML
    /// error that says nothing about the template, so an unknown placeholder is
    /// named instead of substituted.
    #[test]
    fn an_unknown_placeholder_is_named() {
        let found = unknown_placeholders("name = \"{{crate_name}}\"\nlicense = \"{{license}}\"\n");

        assert_eq!(found, vec!["{{license}}".to_string()]);
    }

    /// WHY: rendering must leave nothing behind. A single unrendered
    /// `{{crate_name_snake}}` in a test file is a compile error about an unknown
    /// crate, which reads as a defect in the workspace rather than in the render.
    #[test]
    fn rendering_leaves_no_placeholder() {
        let rendered = render("use {{crate_name_snake}}::Op; // {{crate_name}} by {{gh_org}}");

        assert!(!rendered.contains("{{"), "{rendered}");
        assert!(unknown_placeholders(&rendered).is_empty());
    }

    /// WHY: every placeholder the tracked templates actually use must be
    /// renderable, and the set is read from the tree rather than restated here,
    /// so adding a placeholder to a template turns this red instead of turning
    /// the gate into one that skips its own subject.
    #[test]
    fn every_tracked_template_placeholder_has_a_value() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask sits in the workspace root")
            .to_path_buf();
        let tracked = tracked_example_paths(&root).expect("the checkout tracks its examples");
        let mut unknown = Vec::new();
        for paths in tracked.values() {
            if !paths
                .iter()
                .any(|path| path.ends_with(TEMPLATE_MANIFEST))
            {
                continue;
            }
            for path in paths {
                let Ok(text) = std::fs::read_to_string(root.join(path)) else {
                    continue;
                };
                unknown.extend(unknown_placeholders(&text));
            }
        }

        assert!(
            unknown.is_empty(),
            "tracked templates use placeholders this gate cannot render: {unknown:?}"
        );
    }

    /// WHY: the patch table is derived from the dependencies the rendered
    /// manifest declares, because a hardcoded crate list stops patching the day
    /// a template adds a dependency, and the build then silently resolves that
    /// one crate from the registry instead of from this checkout.
    #[test]
    fn dependency_names_come_from_every_dependency_table() {
        let names = dependency_names(
            "[package]\nname = \"probe\"\n\n[dependencies]\nvyre = \"0.7.2\"\nvyre-libs = { version = \"0.7.2\" }\n\n[dev-dependencies]\nvyre-reference = \"0.7.2\"\n\n[lints.rust]\nunsafe_code = \"forbid\"\n",
        );

        assert_eq!(
            names,
            vec![
                "vyre".to_string(),
                "vyre-libs".to_string(),
                "vyre-reference".to_string()
            ]
        );
    }

    /// WHY: a manifest that inherits the repository workspace resolves against
    /// workspace lints, patches and features, so it proves nothing about what a
    /// consumer outside the tree gets.
    #[test]
    fn a_manifest_without_its_own_workspace_is_a_finding() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask sits in the workspace root")
            .to_path_buf();

        let isolated = workspace_isolation_findings(&root, "examples/libs-template/Cargo.toml.liquid");
        let missing = workspace_isolation_findings(&root, "xtask/Cargo.toml");

        assert!(isolated.is_empty(), "{isolated:?}");
        assert_eq!(missing.len(), 1);
    }
}
