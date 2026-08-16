//! The generated contract section of every crate README.
//!
//! A crate README is the first thing a consumer reads, and most of what it has
//! to say is already declared somewhere else: the responsibility and the
//! allowed edges in `docs/CRATE_OWNERSHIP.toml`, the feature set and the
//! version in the manifest, the release position in `release/release-train.toml`,
//! the error behavior in `docs/CRATE_GUIDES.toml`. Written by hand, all four
//! drift, and the README keeps making a claim the tree stopped honoring: the
//! retired-version rule below exists because a whole workspace of READMEs went
//! on advertising a `0.4.x` release train after the train moved.
//!
//! Everything between the two markers is rendered from those authorities. Text
//! outside them is the crate's own and is preserved.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::crate_registry::{self, CrateRecord};
use crate::gates::scan::Tree;

/// Per-crate prose the manifests do not carry.
const METADATA: &str = "docs/CRATE_GUIDES.toml";
/// The version authority every release claim is rendered from.
const RELEASE_TRAIN: &str = "release/release-train.toml";
/// Opens the generated region.
pub const BEGIN_MARKER: &str = "<!-- BEGIN GENERATED CRATE CONTRACT -->";
/// Closes the generated region.
pub const END_MARKER: &str = "<!-- END GENERATED CRATE CONTRACT -->";
/// The command that rewrites every generated region.
const WRITE_COMMAND: &str = "xtask crate-readmes --write";
/// Schema `docs/CRATE_GUIDES.toml` must declare.
const SCHEMA_VERSION: i64 = 1;
/// What a caller does about a stale or missing region.
const FIX: &str = "run `xtask crate-readmes --write`";

/// The generated contract section in every crate README.
pub struct CrateReadmes;

impl crate::gate::GateBehavior for CrateReadmes {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let records = crate_registry::load_registry(&tree, &mut report)?;
        report.cover_complete("crate readmes", records.len());
        let guides = load_guides(&tree, &records, &mut report)?;
        let versions = release_versions(&tree, &mut report)?;

        // An authority that does not describe this workspace cannot render a
        // README about it, and writing one from a broken registry publishes the
        // break. The rendered set is judged only once every input holds, which is
        // the rule the ownership pair is already rendered under.
        if !report.findings.is_empty() {
            report.note(format!(
                "{} registry row(s), none rendered: an input authority does not hold",
                records.len()
            ));
            return Ok(report);
        }

        if ctx.write {
            let cli_report = crate::docs::cli_docs::refresh_readmes(ctx)?;
            report.findings.extend(cli_report.findings);
            report.notes.extend(cli_report.notes);
            report.coverage.extend(cli_report.coverage);
            if !report.findings.is_empty() {
                return Ok(report);
            }
        }

        let mut written = 0usize;
        for record in &records {
            let Some(behavior) = guides.error_behavior(record) else {
                report.find(Finding::in_file(
                    METADATA,
                    format!(
                        "no error profile for layer `{}`, which `{}` occupies",
                        record.layer, record.package
                    ),
                    "add a [profile.<layer>] table for the layer",
                ));
                continue;
            };
            let manifest = tree.read_toml(format!("{}/Cargo.toml", record.path))?;
            let Some(status) = guides.release_status(record, &manifest, &versions, &mut report)
            else {
                continue;
            };
            let relative = format!("{}/README.md", record.path);
            report.produced(&relative);
            let existing = read_readme(&ctx.root, &relative)?;
            let contract = render_contract(
                record,
                &manifest,
                &behavior,
                &status,
                &runnable_example(&tree, record, &manifest),
                &mut report,
            );
            let Some(expected) = compose(
                &existing,
                &record.package,
                &contract,
                &relative,
                &mut report,
            ) else {
                continue;
            };
            // A retired claim inside the generated region means an authority is
            // stale, and rendering it would publish the break. One in the crate's
            // own text is a finding about that text, and the generated region is
            // still written so `--write` converges instead of leaving the
            // contract stale behind a claim it does not own.
            if let Some(claim) = retired_claim(&contract) {
                report.find(Finding::in_file(
                    &relative,
                    format!("the generated contract claims retired release `{claim}`"),
                    "state the version the release train declares",
                ));
                continue;
            }
            if let Some(claim) = retired_claim(&expected) {
                report.find(Finding::in_file(
                    &relative,
                    format!("the README claims retired release `{claim}`"),
                    "state the version the release train declares",
                ));
            }
            if ctx.write {
                fs::write(ctx.root.join(&relative), &expected).map_err(|error| {
                    GateError::new(
                        format!("cannot write `{relative}`: {error}"),
                        "make the crate directory writable",
                    )
                })?;
                written += 1;
            } else if existing != expected {
                report.find(Finding::in_file(
                    &relative,
                    "the generated crate contract does not match the authorities it is rendered from",
                    FIX,
                ));
            }
        }
        report.note(if ctx.write {
            format!("wrote {written} crate README contract(s)")
        } else {
            format!("{} crate README contract(s)", records.len())
        });
        Ok(report)
    }
}

/// The per-crate prose `docs/CRATE_GUIDES.toml` carries.
struct Guides {
    /// Error behavior per layer.
    profiles: BTreeMap<String, String>,
    /// Per-package overrides of the layer profile and the release claim.
    overrides: BTreeMap<String, toml::Table>,
}

impl Guides {
    /// The error behavior for one crate: its own override, else its layer.
    ///
    /// The layer profile is required even when a package overrides it. The
    /// override is prose for one crate and the profile is what every other crate
    /// in that layer renders, so accepting an override for a layer that declares
    /// no profile leaves the next crate in that layer with nothing to render.
    fn error_behavior(&self, record: &CrateRecord) -> Option<String> {
        let profile = self.profiles.get(&record.layer)?;
        Some(
            self.overrides
                .get(&record.package)
                .and_then(|table| table.get("error_behavior"))
                .and_then(Value::as_str)
                .map_or_else(|| profile.clone(), |text| text.trim().to_string()),
        )
    }

    /// The release claim for one crate: its own override with the release
    /// train's versions substituted, else the position the manifest declares.
    fn release_status(
        &self,
        record: &CrateRecord,
        manifest: &toml::Table,
        versions: &BTreeMap<String, String>,
        report: &mut Report,
    ) -> Option<String> {
        let version = manifest
            .get("package")
            .and_then(|package| package.get("version"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if version.is_empty() {
            report.find(Finding::in_file(
                format!("{}/Cargo.toml", record.path),
                "the manifest declares no package.version",
                "declare package.version, or inherit it from the workspace",
            ));
            return None;
        }
        let Some(template) = self
            .overrides
            .get(&record.package)
            .and_then(|table| table.get("release_status"))
        else {
            return Some(default_status(
                &record.package,
                version,
                publishable(manifest),
            ));
        };
        let template = template.as_str().unwrap_or_default().trim();
        if template.is_empty() {
            report.find(Finding::in_file(
                METADATA,
                format!(
                    "the release_status override for `{}` is empty",
                    record.package
                ),
                "write the claim, or delete the override and take the default",
            ));
            return None;
        }
        match substitute(template, versions) {
            Ok(status) => Some(status),
            Err(unknown) => {
                report.find(Finding::in_file(
                    METADATA,
                    format!(
                        "the release_status override for `{}` names `{{{unknown}}}`, which the release train does not declare",
                        record.package
                    ),
                    format!("name a version `{RELEASE_TRAIN}` declares"),
                ));
                None
            }
        }
    }
}

/// Substitute `{name}` placeholders, or name the first one that resolves to
/// nothing.
fn substitute(template: &str, versions: &BTreeMap<String, String>) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let Some(close) = rest[open..].find('}') else {
            return Err(rest[open + 1..].to_string());
        };
        let name = &rest[open + 1..open + close];
        let value = versions.get(name).ok_or_else(|| name.to_string())?;
        out.push_str(value);
        rest = &rest[open + close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Read the crate guides, rejecting a profile or override that describes no
/// crate in the tree.
fn load_guides(
    tree: &Tree,
    records: &[CrateRecord],
    report: &mut Report,
) -> Result<Guides, GateError> {
    let config = tree.read_toml(METADATA)?;
    if config.get("schema_version").and_then(Value::as_integer) != Some(SCHEMA_VERSION) {
        report.find(Finding::in_file(
            METADATA,
            format!("the crate guides do not declare schema_version = {SCHEMA_VERSION}"),
            "declare the schema version the reader expects",
        ));
    }
    let mut profiles = BTreeMap::new();
    if let Some(table) = config.get("profile").and_then(Value::as_table) {
        for (layer, profile) in table {
            match profile.get("error_behavior").and_then(Value::as_str) {
                Some(text) if !text.trim().is_empty() => {
                    profiles.insert(layer.clone(), text.trim().to_string());
                }
                _ => report.find(Finding::in_file(
                    METADATA,
                    format!("profile `{layer}` declares no non-empty `error_behavior`"),
                    "state what the layer does with an unsupported input",
                )),
            }
        }
    }
    let mut overrides = BTreeMap::new();
    if let Some(table) = config.get("package").and_then(Value::as_table) {
        for (package, value) in table {
            match value.as_table() {
                Some(table) => {
                    // An override is read the way a profile is. Prose that
                    // renders a heading with nothing under it is a finding here
                    // rather than an empty section in a published README.
                    let empty = match table.get("error_behavior") {
                        None => false,
                        Some(value) => value.as_str().unwrap_or_default().trim().is_empty(),
                    };
                    if empty {
                        report.find(Finding::in_file(
                            METADATA,
                            format!(
                                "the override for `{package}` declares an empty `error_behavior`"
                            ),
                            "state what the crate does with an unsupported input, or drop the key",
                        ));
                    }
                    overrides.insert(package.clone(), table.clone());
                }
                None => report.find(Finding::in_file(
                    METADATA,
                    format!("the override for `{package}` is not a table"),
                    "declare [package.<name>] as a table",
                )),
            }
        }
    }

    // A profile for a layer no crate occupies, or an override for a package no
    // longer in the workspace, survives a rename and reads as coverage while
    // describing nothing.
    let packages: BTreeSet<&str> = records.iter().map(|r| r.package.as_str()).collect();
    let layers: BTreeSet<&str> = records.iter().map(|r| r.layer.as_str()).collect();
    for package in overrides.keys() {
        if !packages.contains(package.as_str()) {
            report.find(Finding::in_file(
                METADATA,
                format!("`{package}` has an override and is not a workspace crate"),
                "delete the override, or restore the crate",
            ));
        }
    }
    for layer in profiles.keys() {
        if !layers.contains(layer.as_str()) {
            report.find(Finding::in_file(
                METADATA,
                format!("layer `{layer}` has an error profile and no crate occupies it"),
                "delete the profile, or record the crate that occupies the layer",
            ));
        }
    }
    Ok(Guides {
        profiles,
        overrides,
    })
}

/// Every version the release train declares, keyed as a template placeholder.
fn release_versions(
    tree: &Tree,
    report: &mut Report,
) -> Result<BTreeMap<String, String>, GateError> {
    let train = tree.read_toml(RELEASE_TRAIN)?;
    let mut versions = BTreeMap::new();
    if let Some(table) = train.get("versions").and_then(Value::as_table) {
        for (key, value) in table {
            if let Some(version) = value.as_str() {
                versions.insert(format!("{key}_version"), version.to_string());
            }
        }
    }
    if !versions.contains_key("vyre_version") {
        report.find(Finding::in_file(
            RELEASE_TRAIN,
            "the release train declares no versions.vyre",
            "declare the workspace version every crate claim renders from",
        ));
    }
    Ok(versions)
}

/// Whether cargo would publish the crate.
fn publishable(manifest: &toml::Table) -> bool {
    let Some(package) = manifest.get("package") else {
        return false;
    };
    match package.get("publish") {
        None => true,
        Some(Value::Boolean(value)) => *value,
        Some(Value::Array(registries)) => !registries.is_empty(),
        Some(_) => false,
    }
}

/// The release claim a crate with no override carries.
fn default_status(package: &str, version: &str, publishable: bool) -> String {
    if publishable {
        format!(
            "`{package}@{version}` is a publishable crate on the current Vyre release train. Publication still requires the release evidence and user-approval gates."
        )
    } else {
        format!(
            "`{package}@{version}` is workspace-internal on the current Vyre release train and is not published as a standalone crate."
        )
    }
}

/// The retired release claim the text carries, if it carries one.
///
/// `0.4.x` is the train the workspace left. A README that still names it is
/// telling a consumer to depend on a version the registry will not resolve.
///
/// A claim is a dotted number that starts with `0.4.`, so `10.4.2` is not one
/// and neither is `1.0.4.2`: both carry the retired digits inside a version of
/// another train. A trailing dotted group is part of the claim rather than a
/// boundary, because `0.4.2.1` still names the train it starts with. A trailing
/// letter or digit means the run is an identifier or a hash and not a version.
fn retired_claim(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut at = 0;
    while let Some(found) = text[at..].find("0.4.") {
        let start = at + found;
        let before_is_word = start
            .checked_sub(1)
            .is_some_and(|index| bytes[index].is_ascii_digit() || bytes[index] == b'.');
        let mut end = start + 4;
        let digits = bytes[end..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        end += digits;
        while bytes.get(end) == Some(&b'.') && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
            end += 1;
            end += bytes[end..]
                .iter()
                .take_while(|byte| byte.is_ascii_digit())
                .count();
        }
        let after_is_word = bytes.get(end).is_some_and(u8::is_ascii_alphanumeric);
        if digits > 0 && !before_is_word && !after_is_word {
            return Some(text[start..end].to_string());
        }
        at = start + 4;
    }
    None
}

/// The checked-in behavior a reader can run, and the command that runs it.
///
/// Resolution order matches what cargo would pick: an explicit `[[example]]`
/// target, then a tracked `examples/*.rs`, then a binary, then a tracked
/// integration test, then the library. The point is that the command in the
/// README resolves in a clean checkout, so an untracked example does not count.
fn runnable_example(tree: &Tree, record: &CrateRecord, manifest: &toml::Table) -> (String, String) {
    let package = &record.package;
    if let Some((name, features)) = explicit_example(manifest) {
        let arguments = if features.is_empty() {
            String::new()
        } else {
            format!(" --features {}", features.join(","))
        };
        return (
            format!("`{}/examples/{name}.rs`", record.path),
            format!("./cargo_full run -p {package} --example {name}{arguments}"),
        );
    }
    if let Some(stem) = first_stem(tree, &format!("{}/examples/", record.path)) {
        return (
            format!("`{}/examples/{stem}.rs`", record.path),
            format!("./cargo_full run -p {package} --example {stem}"),
        );
    }
    let autobins = manifest
        .get("package")
        .and_then(|package| package.get("autobins"))
        .and_then(Value::as_bool)
        != Some(false);
    if autobins && tree.has(&format!("{}/src/main.rs", record.path)) {
        return (
            format!("`{}/src/main.rs`", record.path),
            format!("./cargo_full run -p {package} -- --help"),
        );
    }
    let mut binaries: Vec<&str> = manifest
        .get("bin")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.get("name").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    binaries.sort_unstable();
    if let Some(name) = binaries.first() {
        return (
            format!("the `{name}` binary target"),
            format!("./cargo_full run -p {package} --bin {name} -- --help"),
        );
    }
    if let Some(stem) = first_stem(tree, &format!("{}/tests/", record.path)) {
        return (
            format!("`{}/tests/{stem}.rs`", record.path),
            format!("./cargo_full test -p {package} --test {stem}"),
        );
    }
    (
        format!("the `{package}` library target"),
        format!("./cargo_full test -p {package} --lib"),
    )
}

/// The alphabetically first `[[example]]` target and the features it needs.
fn explicit_example(manifest: &toml::Table) -> Option<(String, Vec<String>)> {
    let mut rows: Vec<(String, Vec<String>)> = manifest
        .get("example")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|row| {
            let name = row.get("name").and_then(Value::as_str)?.to_string();
            let mut features: Vec<String> = row
                .get("required-features")
                .and_then(Value::as_array)
                .map(|array| {
                    array
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            features.sort();
            Some((name, features))
        })
        .collect();
    rows.sort();
    rows.into_iter().next()
}

/// The alphabetically first tracked `*.rs` file directly under one directory.
fn first_stem(tree: &Tree, prefix: &str) -> Option<String> {
    let mut stems: Vec<&str> = tree
        .paths()
        .iter()
        .filter_map(|path| {
            let path = path.to_str()?;
            let rest = path.strip_prefix(prefix)?;
            rest.strip_suffix(".rs").filter(|s| !s.contains('/'))
        })
        .collect();
    stems.sort_unstable();
    stems.first().map(|stem| (*stem).to_string())
}

/// The generated region for one crate.
fn render_contract(
    record: &CrateRecord,
    manifest: &toml::Table,
    error_behavior: &str,
    release_status: &str,
    example: &(String, String),
    report: &mut Report,
) -> String {
    let features = manifest
        .get("features")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let names: Vec<String> = features.keys().cloned().collect();
    let mut defaults: Vec<String> = Vec::new();
    if let Some(declared) = features.get("default") {
        match declared.as_array() {
            Some(array) => {
                defaults = array
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect();
            }
            None => report.find(Finding::in_file(
                format!("{}/Cargo.toml", record.path),
                "features.default is not a string array",
                "declare features.default as an array of feature names",
            )),
        }
    }
    let up = "../".repeat(Path::new(&record.path).components().count());
    let guide = format!("docs/testing/{}.md", record.package);
    let (source, command) = example;
    let lines = [
        BEGIN_MARKER.to_string(),
        "## Crate contract".to_string(),
        String::new(),
        format!("This section is generated by `{WRITE_COMMAND}` from"),
        "the crate manifest, release train, ownership registry, and crate-guide metadata."
            .to_string(),
        String::new(),
        "### Purpose".to_string(),
        String::new(),
        record.responsibility.clone(),
        String::new(),
        "### Boundaries".to_string(),
        String::new(),
        format!(
            "The `{}` owner maintains this `{}` crate at `{}`.",
            record.owner, record.layer, record.path
        ),
        format!(
            "Its allowed internal production dependencies are: {}.",
            joined(&record.allowed_dependencies())
        ),
        "Any other normal or build dependency requires an ownership-registry change.".to_string(),
        String::new(),
        "### Minimal real example".to_string(),
        String::new(),
        format!("Run the checked-in behavior from {source}:"),
        String::new(),
        "```console".to_string(),
        command.clone(),
        "```".to_string(),
        String::new(),
        "### Features".to_string(),
        String::new(),
        format!("- Manifest features: {}", joined(&names)),
        format!("- Default feature members: {}", joined(&defaults)),
        String::new(),
        "### Errors and unsupported behavior".to_string(),
        String::new(),
        error_behavior.to_string(),
        String::new(),
        "### Testing".to_string(),
        String::new(),
        format!("See [`{guide}`]({up}{guide}) for the crate's test command,"),
        "hardware contract, expected skips, and failure semantics. It is generated".to_string(),
        "from `docs/testing/TESTING.toml`, which is authoritative.".to_string(),
        String::new(),
        "### Release status".to_string(),
        String::new(),
        release_status.to_string(),
        String::new(),
        "### Ownership".to_string(),
        String::new(),
        format!(
            "[`{registry}`]({up}{registry}) is authoritative for this crate's",
            registry = crate_registry::REGISTRY
        ),
        "responsibility and allowed internal edges.".to_string(),
        String::new(),
        "### License".to_string(),
        String::new(),
        "Licensed under either of".to_string(),
        String::new(),
        "- Apache License, Version 2.0, or".to_string(),
        "- MIT license".to_string(),
        String::new(),
        "at your option. See the workspace `LICENSE-APACHE` and `LICENSE-MIT` files.".to_string(),
        String::new(),
        END_MARKER.to_string(),
        String::new(),
    ];
    lines.join("\n")
}

/// A backtick-joined list, or `None` when the list is empty.
fn joined(values: &[String]) -> String {
    if values.is_empty() {
        return "None".to_string();
    }
    values
        .iter()
        .map(|value| format!("`{value}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Read one README, or the empty string when the crate has none yet.
fn read_readme(root: &Path, relative: &str) -> Result<String, GateError> {
    let path: PathBuf = root.join(relative);
    if !path.exists() {
        return Ok(String::new());
    }
    fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("cannot read `{relative}`: {error}"),
            "make the crate README readable",
        )
    })
}

/// The README with its generated region replaced, preserving the crate's own
/// text around it.
fn compose(
    existing: &str,
    package: &str,
    contract: &str,
    relative: &str,
    report: &mut Report,
) -> Option<String> {
    let begins = existing.matches(BEGIN_MARKER).count();
    let ends = existing.matches(END_MARKER).count();
    if begins != ends || begins > 1 {
        report.find(Finding::in_file(
            relative,
            format!("the README carries {begins} begin and {ends} end contract marker(s)"),
            "leave exactly one generated region, or none",
        ));
        return None;
    }
    let body = if begins == 0 {
        existing.trim_end().to_string()
    } else {
        let (before, remainder) = existing.split_once(BEGIN_MARKER)?;
        let (_, after) = remainder.split_once(END_MARKER)?;
        format!("{}\n{}", before.trim_end(), after.trim())
            .trim()
            .to_string()
    };
    let body = if body.is_empty() {
        format!(
            "# `{package}`\n\nUse this crate through the contract and checked-in example below."
        )
    } else {
        body
    };
    Some(format!("{}\n\n{contract}", body.trim_end()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the retired-train rule is the reason this gate exists, and a
    /// substring match on `0.4.` would fire on `10.4.2` and on a hash. The
    /// boundary is what makes the rule usable, so it is what gets asserted. A
    /// four-component version is a claim because it starts with the retired
    /// train; the same digits inside another train's version are not.
    #[test]
    fn a_retired_release_claim_is_recognized_only_at_a_boundary() {
        assert_eq!(
            retired_claim("vyre 0.4.12 ships"),
            Some("0.4.12".to_string())
        );
        assert_eq!(retired_claim("version 10.4.2"), None);
        assert_eq!(retired_claim("build 1.0.4.2"), None);
        assert_eq!(retired_claim("0.4.2.1"), Some("0.4.2.1".to_string()));
        assert_eq!(retired_claim("commit 0.4.2f3a"), None);
        assert_eq!(retired_claim("0.4.x"), None);
        assert_eq!(retired_claim("0.5.0"), None);
    }

    /// WHY: text outside the markers is the crate's own and a regeneration that
    /// dropped it would silently delete hand-written documentation. Composing
    /// twice must also be idempotent, or `--write` would grow the file.
    #[test]
    fn regenerating_preserves_the_crate_text_and_is_idempotent() {
        let mut report = Report::clean();
        let contract = format!("{BEGIN_MARKER}\nnew\n{END_MARKER}\n");
        let existing = format!("# `demo`\n\nhand written\n\n{BEGIN_MARKER}\nold\n{END_MARKER}\n");
        let once = compose(&existing, "demo", &contract, "demo/README.md", &mut report)
            .expect("a balanced README composes");
        assert!(once.contains("hand written"));
        assert!(!once.contains("old"));
        let twice = compose(&once, "demo", &contract, "demo/README.md", &mut report)
            .expect("the composed README composes again");
        assert_eq!(once, twice);
        assert!(report.findings.is_empty());
    }

    /// WHY: an unbalanced marker pair means the region boundary is unknown, and
    /// writing through it would either duplicate the contract or eat the
    /// crate's text. That is a finding, not a silent repair.
    #[test]
    fn an_unbalanced_marker_pair_is_a_finding() {
        let mut report = Report::clean();
        let existing = format!("{BEGIN_MARKER}\nold\n");
        assert!(compose(&existing, "demo", "", "demo/README.md", &mut report).is_none());
        assert_eq!(report.findings.len(), 1);
    }

    /// WHY: a release claim renders a version from the release train, and a
    /// placeholder nothing declares must be named rather than rendered as
    /// literal braces into a published README.
    #[test]
    fn an_unknown_release_placeholder_is_named() {
        let versions = BTreeMap::from([("vyre_version".to_string(), "0.9.1".to_string())]);
        assert_eq!(
            substitute("ships in {vyre_version}", &versions),
            Ok("ships in 0.9.1".to_string())
        );
        assert_eq!(
            substitute("ships in {conform_version}", &versions),
            Err("conform_version".to_string())
        );
    }

    /// WHY: `publish = []` and `publish = false` both mean cargo refuses to
    /// publish, and a README that called such a crate publishable would send a
    /// reader to a registry page that does not exist.
    #[test]
    fn every_form_of_a_publish_refusal_reads_as_unpublishable() {
        let cases = [
            ("", true),
            ("publish = true", true),
            ("publish = false", false),
            ("publish = []", false),
            ("publish = [\"crates-io\"]", true),
        ];
        for (declaration, expected) in cases {
            let manifest: toml::Table =
                toml::from_str(&format!("[package]\nname = \"demo\"\n{declaration}\n"))
                    .expect("the manifest parses");
            assert_eq!(publishable(&manifest), expected, "for `{declaration}`");
        }
    }
}
