//! Every `cargo run` a workflow or script issues must resolve to one target.
//!
//! WHY: a package that ships more than one binary and declares no `default-run`
//! makes `cargo run -p <package>` exit 101 with "could not determine which
//! binary to run". Nothing in the workspace changes, no test fails, and no
//! review catches it, because the defect is in the shape of the manifest rather
//! than in any line of code. Adding a second file under `xtask/src/bin` failed
//! nineteen hosted jobs at their first step: six ci.yml matrix legs, the
//! registered gate sweep, the operation coverage check and the conformance
//! release gate, all of which invoke the runner that way.
//!
//! The existing sweep already asserts that every registered subcommand is named
//! by a workflow. The missing half is the other direction: that the command a
//! workflow names actually resolves and runs. Both sides are derived here at
//! run time, the invocations from `.github/workflows` and `scripts`, the binary
//! targets from every workspace member's manifest and source layout, so a new
//! workflow step or a new binary is covered the moment it lands rather than
//! when someone remembers to add it to a list.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Fewer invocations than this means the scan is broken, not the tree.
///
/// A derivation that silently yields nothing is the same defect as no gate, and
/// this one reads two directories that a reorganisation could move.
const MINIMUM_INVOCATIONS: usize = 20;

/// Fewer packages than this means the manifest walk is broken.
const MINIMUM_PACKAGES: usize = 10;

/// One `cargo run` found in a workflow or script.
#[derive(Debug, Clone)]
struct Invocation {
    origin: String,
    line: usize,
    package: Option<String>,
    binary: Option<String>,
    text: String,
}

/// The binary targets a package produces.
#[derive(Debug, Default)]
struct PackageBinaries {
    names: BTreeSet<String>,
    default_run: Option<String>,
}

#[test]
fn every_cargo_run_in_a_workflow_or_script_resolves_to_one_binary() {
    let root = structure_gate::workspace_root();
    let packages = workspace_binaries(&root);
    assert!(
        packages.len() >= MINIMUM_PACKAGES,
        "Fix: the manifest walk found {} package(s), expected at least \
         {MINIMUM_PACKAGES}. The derivation is broken, not the workspace.",
        packages.len()
    );

    let invocations = scan_invocations(&root);
    assert!(
        invocations.len() >= MINIMUM_INVOCATIONS,
        "Fix: the invocation scan found {} `cargo run` call(s) under \
         .github/workflows and scripts, expected at least \
         {MINIMUM_INVOCATIONS}. The scan is broken, not the tree.",
        invocations.len()
    );

    let mut unresolved = Vec::new();
    for invocation in &invocations {
        if let Err(reason) = resolve(invocation, &packages) {
            unresolved.push(format!(
                "{}:{}: {reason}\n      {}",
                invocation.origin,
                invocation.line,
                invocation.text.trim()
            ));
        }
    }
    unresolved.sort();

    assert!(
        unresolved.is_empty(),
        "Fix: these cargo invocations do not resolve to exactly one binary \
         target, so the step exits 101 before running anything it was written \
         to run. Declare `default-run` in the package that ships more than one \
         binary, or name the binary in the invocation:\n  {}",
        unresolved.join("\n  ")
    );
}

/// Which target an invocation names, or why it names none or several.
///
/// An invocation the scan cannot attribute to a package is an error rather
/// than a skip: a form this function does not understand is a form nobody
/// checked, which is how the defect this file exists for reached CI.
fn resolve(
    invocation: &Invocation,
    packages: &BTreeMap<String, PackageBinaries>,
) -> Result<(), String> {
    match (&invocation.package, &invocation.binary) {
        (Some(package), Some(binary)) => {
            let Some(target) = packages.get(package) else {
                return Err(format!("`-p {package}` names no workspace member"));
            };
            if target.names.contains(binary) {
                Ok(())
            } else {
                Err(format!(
                    "`{package}` ships no binary `{binary}`; it ships {:?}",
                    target.names
                ))
            }
        }
        (Some(package), None) => {
            let Some(target) = packages.get(package) else {
                return Err(format!("`-p {package}` names no workspace member"));
            };
            match target.names.len() {
                0 => Err(format!("`{package}` ships no binary at all")),
                1 => Ok(()),
                _ if target.default_run.is_some() => Ok(()),
                count => Err(format!(
                    "`{package}` ships {count} binaries {:?} and declares no \
                     `default-run`, so cargo cannot decide which to build",
                    target.names
                )),
            }
        }
        (None, Some(binary)) => {
            let owners: Vec<&String> = packages
                .iter()
                .filter(|(_, target)| target.names.contains(binary))
                .map(|(name, _)| name)
                .collect();
            match owners.len() {
                0 => Err(format!("no workspace member ships a binary `{binary}`")),
                1 => Ok(()),
                _ => Err(format!(
                    "binary `{binary}` is shipped by {owners:?}; the invocation \
                     needs `-p` to say which"
                )),
            }
        }
        (None, None) => Err(
            "names neither a package nor a binary, so what it runs depends on \
             the working directory"
                .to_string(),
        ),
    }
}

/// Every workspace member's binary targets, read from its manifest and layout.
fn workspace_binaries(root: &Path) -> BTreeMap<String, PackageBinaries> {
    let mut packages = BTreeMap::new();
    for member in structure_gate::workspace_members(root) {
        let crate_dir = root.join(&member);
        let manifest_path = crate_dir.join("Cargo.toml");
        let Ok(manifest) = std::fs::read_to_string(&manifest_path) else {
            continue;
        };
        let name = package_name(&manifest).unwrap_or(member.clone());
        let mut target = PackageBinaries {
            default_run: manifest_string(&manifest, "default-run"),
            ..PackageBinaries::default()
        };
        if crate_dir.join("src/main.rs").is_file() {
            target.names.insert(name.clone());
        }
        if let Ok(entries) = std::fs::read_dir(crate_dir.join("src/bin")) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|extension| extension == "rs") {
                    if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                        target.names.insert(stem.to_string());
                    }
                }
            }
        }
        target.names.extend(declared_bin_names(&manifest));
        packages.insert(name, target);
    }
    packages
}

/// The `name` of each `[[bin]]` section in a manifest.
fn declared_bin_names(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut inside = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            inside = trimmed == "[[bin]]";
            continue;
        }
        if inside {
            if let Some(value) = key_value(trimmed, "name") {
                names.push(value);
            }
        }
    }
    names
}

fn package_name(manifest: &str) -> Option<String> {
    let mut inside = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            inside = trimmed == "[package]";
            continue;
        }
        if inside {
            if let Some(value) = key_value(trimmed, "name") {
                return Some(value);
            }
        }
    }
    None
}

fn manifest_string(manifest: &str, key: &str) -> Option<String> {
    let mut inside = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            inside = trimmed == "[package]";
            continue;
        }
        if inside {
            if let Some(value) = key_value(trimmed, key) {
                return Some(value);
            }
        }
    }
    None
}

fn key_value(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start();
    let rest = rest.strip_prefix('=')?.trim();
    Some(rest.trim_matches('"').to_string())
}

/// Every `cargo run` under `.github/workflows` and `scripts`.
///
/// The two directories are walked rather than listed, and both the `cargo` and
/// the `./cargo_full` spellings are read, because the wrapper forwards its
/// arguments unchanged and a step written either way fails the same way.
fn scan_invocations(root: &Path) -> Vec<Invocation> {
    let mut found = Vec::new();
    for directory in [root.join(".github/workflows"), root.join("scripts")] {
        for file in
            crate::workspace_sources::sources_under(&directory, &["yml", "yaml", "sh", "py"])
        {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let origin = file
                .strip_prefix(root)
                .unwrap_or(file.as_path())
                .display()
                .to_string();
            for (index, line) in text.lines().enumerate() {
                if let Some(invocation) = parse_invocation(line) {
                    found.push(Invocation {
                        origin: origin.clone(),
                        line: index + 1,
                        ..invocation
                    });
                }
            }
        }
    }
    found
}

/// The package and binary a single `cargo run` line names, if it is one.
///
/// Only a line that reaches the `run` subcommand is returned. A line that
/// mentions cargo without running anything, such as a comment describing the
/// failure mode, is not an invocation and must not be reported as one.
fn parse_invocation(line: &str) -> Option<Invocation> {
    let trimmed = line.trim();
    if trimmed.starts_with('#') || trimmed.starts_with("//") {
        return None;
    }
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let start = tokens.iter().position(|token| {
        let bare = token.trim_start_matches("./").trim_matches('"');
        bare == "cargo" || bare == "cargo_full"
    })?;
    let rest = &tokens[start + 1..];
    let run_at = rest.iter().position(|token| *token == "run")?;
    // Anything between the program and `run` must be a flag; a word there means
    // this is prose about cargo rather than a call to it.
    if rest[..run_at].iter().any(|token| !token.starts_with('-')) {
        return None;
    }

    let mut package = None;
    let mut binary = None;
    let mut index = run_at + 1;
    while index < rest.len() {
        match rest[index] {
            "--" => break,
            "-p" | "--package" => {
                package = rest.get(index + 1).map(|value| value.to_string());
                index += 2;
            }
            "--bin" => {
                binary = rest.get(index + 1).map(|value| value.to_string());
                index += 2;
            }
            token if token.starts_with("--package=") => {
                package = token.split_once('=').map(|(_, value)| value.to_string());
                index += 1;
            }
            token if token.starts_with("--bin=") => {
                binary = token.split_once('=').map(|(_, value)| value.to_string());
                index += 1;
            }
            _ => index += 1,
        }
    }

    // A templated invocation is generated from a manifest, so the package name
    // is not in the text and there is nothing to resolve here.
    if package.as_deref().is_some_and(|name| name.contains('{'))
        || binary.as_deref().is_some_and(|name| name.contains('{'))
    {
        return None;
    }

    Some(Invocation {
        origin: String::new(),
        line: 0,
        package,
        binary,
        text: trimmed.to_string(),
    })
}
