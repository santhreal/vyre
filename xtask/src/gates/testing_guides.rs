//! One testing guide per workspace member, under `docs/testing/`.
//!
//! A crate's test contract is three things a reader needs at once: the commands
//! that run it, the cargo targets those commands reach, and what the crate does
//! when the hardware it wants is absent. The first two are already in the
//! manifest and the third is prose, so the guide is rendered from both rather
//! than written: a hand-written guide keeps naming a `--test` target after the
//! file is deleted, and a reader runs a command cargo rejects.
//!
//! The directory is closed in both directions. A member with no guide is a
//! finding, and a guide in the directory that no member claims is also a
//! finding, because that is the shape a crate absorption leaves behind.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use toml::Value;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::crate_registry::{self, CrateRecord};
use crate::gates::scan::Tree;

/// Per-crate testing prose the manifests do not carry.
const METADATA: &str = "docs/testing/TESTING.toml";
/// The directory this gate owns completely.
const DIRECTORY: &str = "docs/testing";
/// The command that rewrites every guide.
const WRITE_COMMAND: &str = "xtask testing-guides --write";
/// Schema `docs/testing/TESTING.toml` must declare.
const SCHEMA_VERSION: i64 = 1;
/// What a caller does about a stale or missing guide.
const FIX: &str = "run `xtask testing-guides --write`";

/// One cargo target a testing command can reach.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Target {
    /// `lib`, `bin`, `test`, `bench` or `example`.
    kind: &'static str,
    /// Target name, as cargo resolves it.
    name: String,
    /// Source path, relative to the crate root.
    source: String,
    /// Features the target needs, sorted.
    required_features: Vec<String>,
}

impl Target {
    /// The command that runs this one target.
    fn command(&self, package: &str) -> String {
        let flag = match self.kind {
            "test" => "--test",
            "bin" => "--bin",
            "example" => "--example",
            "bench" => "--bench",
            _ => return format!("./cargo_full test -p {package}"),
        };
        format!("./cargo_full test -p {package} {flag} {}", self.name)
    }
}

/// The per-crate testing guides under `docs/testing/`.
pub struct TestingGuides;

impl Gate for TestingGuides {
    fn name(&self) -> &'static str {
        "testing-guides"
    }

    fn help(&self) -> &'static str {
        "Hold one testing guide per workspace member to the manifests and docs/testing/TESTING.toml; --write regenerates them"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let records = crate_registry::load_registry(&tree, &mut report)?;
        let metadata = load_metadata(&tree, &records, &mut report)?;

        let mut expected: BTreeMap<String, String> = BTreeMap::new();
        // A row that never reached `expected` is a member whose guide would read
        // as orphaned, so only an incomplete render set suppresses the orphan
        // scan. An empty record set is the one incompleteness the loop below
        // cannot see: it renders nothing, and every guide in the directory would
        // then be reported as orphaned on the strength of a registry that
        // carried no rows. Any other finding leaves the rows it did not touch in
        // the set, so a metadata field missing for one crate no longer hides
        // every leftover guide behind it.
        let mut skipped = records.is_empty();
        for record in &records {
            let Some(fields) = metadata.resolve(record, &mut report) else {
                skipped = true;
                continue;
            };
            let manifest = tree.read_toml(format!("{}/Cargo.toml", record.path))?;
            let declared = manifest
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(Value::as_str);
            if declared != Some(record.package.as_str()) {
                report.find(Finding::in_file(
                    format!("{}/Cargo.toml", record.path),
                    format!(
                        "the manifest declares package `{}` and the registry row says `{}`",
                        declared.unwrap_or_default(),
                        record.package
                    ),
                    "name the same package in the manifest and the ownership registry",
                ));
                skipped = true;
                continue;
            }
            let targets = cargo_targets(&tree, record, &manifest, &mut report);
            let relative = format!("{DIRECTORY}/{}", guide_name(record));
            if let Some(previous) = expected.insert(
                relative.clone(),
                render_guide(record, &manifest, &targets, &fields, &mut report),
            ) {
                let _ = previous;
                report.find(Finding::in_file(
                    &relative,
                    "two workspace members render the same guide filename",
                    "give the members distinct directory names",
                ));
            }
        }

        // The directory is the whole surface, so a guide nothing renders is a
        // finding in the same pass.
        if !skipped {
            for path in tree.paths() {
                let Some(name) = path.to_str() else { continue };
                if name.starts_with(&format!("{DIRECTORY}/"))
                    && name.ends_with(".md")
                    && !name[DIRECTORY.len() + 1..].contains('/')
                    && !expected.contains_key(name)
                {
                    report.find(Finding::in_file(
                        name,
                        "no workspace member renders this guide",
                        "delete the guide, or restore the member it describes",
                    ));
                }
            }
        }

        let mut written = 0usize;
        for (relative, content) in &expected {
            let path = ctx.root.join(relative);
            if ctx.write {
                fs::write(&path, content).map_err(|error| {
                    GateError::new(
                        format!("cannot write `{relative}`: {error}"),
                        "make the testing documentation directory writable",
                    )
                })?;
                written += 1;
            } else if fs::read_to_string(&path).ok().as_deref() != Some(content.as_str()) {
                report.find(Finding::in_file(
                    relative,
                    "the testing guide does not match the manifest and metadata it is rendered from",
                    FIX,
                ));
            }
        }
        report.note(if ctx.write {
            format!("wrote {written} testing guide(s)")
        } else {
            format!("{} testing guide(s)", expected.len())
        });
        Ok(report)
    }
}

/// The guide filename one member renders, which is its directory name.
fn guide_name(record: &CrateRecord) -> String {
    let leaf = Path::new(&record.path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(record.path.as_str());
    format!("{leaf}.md")
}

/// The prose fields one guide renders.
struct Fields {
    /// What the crate needs from the machine.
    hardware: String,
    /// Which cases the suite skips, and when.
    expected_skips: String,
    /// What a failure means.
    failure_behavior: String,
    /// The kinds of test the crate carries.
    test_classes: Vec<String>,
    /// What a run leaves behind.
    evidence_outputs: Vec<String>,
    /// Commands beyond the default and all-features pair.
    commands: Vec<String>,
}

/// `docs/testing/TESTING.toml`, layered defaults then layer then package.
struct Metadata {
    /// Fields every crate starts from.
    defaults: toml::Table,
    /// Fields a layer overrides.
    profiles: BTreeMap<String, toml::Table>,
    /// Fields one package overrides.
    overrides: BTreeMap<String, toml::Table>,
}

impl Metadata {
    /// The merged fields for one crate, or `None` when a required one is
    /// missing at every level.
    fn resolve(&self, record: &CrateRecord, report: &mut Report) -> Option<Fields> {
        let Some(profile) = self.profiles.get(&record.layer) else {
            report.find(Finding::in_file(
                METADATA,
                format!(
                    "no profile for layer `{}`, which `{}` occupies",
                    record.layer, record.package
                ),
                "add a [profile.<layer>] table",
            ));
            return None;
        };
        let mut merged = self.defaults.clone();
        merged.extend(profile.clone());
        if let Some(override_table) = self.overrides.get(&record.package) {
            merged.extend(override_table.clone());
        }
        let context = format!("testing metadata for `{}`", record.package);
        let hardware = required_text(&merged, "hardware", &context, report)?;
        let expected_skips = required_text(&merged, "expected_skips", &context, report)?;
        let failure_behavior = required_text(&merged, "failure_behavior", &context, report)?;
        let test_classes = required_list(&merged, "test_classes", &context, report)?;
        let evidence_outputs = required_list(&merged, "evidence_outputs", &context, report)?;
        let commands = match merged.get("commands") {
            None => Vec::new(),
            Some(_) => required_list(&merged, "commands", &context, report)?,
        };
        Some(Fields {
            hardware,
            expected_skips,
            failure_behavior,
            test_classes,
            evidence_outputs,
            commands,
        })
    }
}

/// One required non-empty text field.
fn required_text(
    table: &toml::Table,
    field: &str,
    context: &str,
    report: &mut Report,
) -> Option<String> {
    match table.get(field).and_then(Value::as_str) {
        Some(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
        _ => {
            report.find(Finding::in_file(
                METADATA,
                format!("{context} declares no non-empty `{field}`"),
                "declare the field at the defaults, profile or package level",
            ));
            None
        }
    }
}

/// One required array of non-empty text.
fn required_list(
    table: &toml::Table,
    field: &str,
    context: &str,
    report: &mut Report,
) -> Option<Vec<String>> {
    let entries: Option<Vec<String>> = table
        .get(field)
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .map(|item| {
                    item.as_str()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or(None);
    if entries.is_none() {
        report.find(Finding::in_file(
            METADATA,
            format!("{context} declares no `{field}` array of non-empty text"),
            "declare the field at the defaults, profile or package level",
        ));
    }
    entries
}

/// Read the testing metadata, rejecting a profile or override that describes no
/// crate in the tree.
fn load_metadata(
    tree: &Tree,
    records: &[CrateRecord],
    report: &mut Report,
) -> Result<Metadata, GateError> {
    let config = tree.read_toml(METADATA)?;
    if config.get("schema_version").and_then(Value::as_integer) != Some(SCHEMA_VERSION) {
        report.find(Finding::in_file(
            METADATA,
            format!("the testing metadata does not declare schema_version = {SCHEMA_VERSION}"),
            "declare the schema version the reader expects",
        ));
    }
    let defaults = config
        .get("defaults")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_else(|| {
            report.find(Finding::in_file(
                METADATA,
                "the testing metadata declares no [defaults] table",
                "declare the fields every crate starts from",
            ));
            toml::Table::new()
        });
    let mut profiles = BTreeMap::new();
    if let Some(table) = config.get("profile").and_then(Value::as_table) {
        for (layer, profile) in table {
            match profile.as_table() {
                Some(profile) => {
                    profiles.insert(layer.clone(), profile.clone());
                }
                None => report.find(Finding::in_file(
                    METADATA,
                    format!("profile `{layer}` is not a table"),
                    "declare [profile.<layer>] as a table",
                )),
            }
        }
    }
    let mut overrides = BTreeMap::new();
    if let Some(table) = config.get("package").and_then(Value::as_table) {
        for (package, value) in table {
            match value.as_table() {
                Some(value) => {
                    overrides.insert(package.clone(), value.clone());
                }
                None => report.find(Finding::in_file(
                    METADATA,
                    format!("the override for `{package}` is not a table"),
                    "declare [package.<name>] as a table",
                )),
            }
        }
    }
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
                format!("layer `{layer}` has a profile and no crate occupies it"),
                "delete the profile, or record the crate that occupies the layer",
            ));
        }
    }
    Ok(Metadata {
        defaults,
        profiles,
        overrides,
    })
}

/// The sorted feature names a target row requires.
fn required_features(row: &Value) -> Vec<String> {
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
    features
}

/// The targets one manifest declares explicitly for a kind.
fn explicit_targets(
    manifest: &toml::Table,
    kind: &'static str,
    directory: &str,
    context: &str,
    report: &mut Report,
) -> Vec<Target> {
    let Some(rows) = manifest.get(kind) else {
        return Vec::new();
    };
    let Some(rows) = rows.as_array() else {
        report.find(Finding::in_file(
            format!("{context}/Cargo.toml"),
            format!("[[{kind}]] is not an array of tables"),
            "declare each target as a table",
        ));
        return Vec::new();
    };
    let mut targets = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let Some(name) = row.get("name").and_then(Value::as_str) else {
            report.find(Finding::in_file(
                format!("{context}/Cargo.toml"),
                format!("[[{kind}]] row {} declares no name", index + 1),
                "name every explicit cargo target",
            ));
            continue;
        };
        let source = row
            .get("path")
            .and_then(Value::as_str)
            .map_or_else(|| format!("{directory}/{name}.rs"), str::to_string);
        targets.push(Target {
            kind,
            name: name.to_string(),
            source,
            required_features: required_features(row),
        });
    }
    targets
}

/// The tracked `*.rs` files cargo would pick up implicitly from one directory.
fn implicit_targets(
    tree: &Tree,
    crate_path: &str,
    directory: &str,
    kind: &'static str,
) -> Vec<Target> {
    let prefix = format!("{crate_path}/{directory}/");
    let mut targets: Vec<Target> = tree
        .paths()
        .iter()
        .filter_map(|path| {
            let rest = path.to_str()?.strip_prefix(&prefix)?;
            let stem = rest.strip_suffix(".rs").filter(|s| !s.contains('/'))?;
            Some(Target {
                kind,
                name: stem.to_string(),
                source: format!("{directory}/{stem}.rs"),
                required_features: Vec::new(),
            })
        })
        .collect();
    targets.sort();
    targets
}

/// Whether a manifest leaves one auto-discovery field on.
fn autodiscovers(manifest: &toml::Table, field: &str) -> bool {
    manifest
        .get("package")
        .and_then(|package| package.get(field))
        .and_then(Value::as_bool)
        != Some(false)
}

/// Every cargo target a testing command can reach in one crate.
fn cargo_targets(
    tree: &Tree,
    record: &CrateRecord,
    manifest: &toml::Table,
    report: &mut Report,
) -> Vec<Target> {
    let path = record.path.as_str();
    let mut targets = Vec::new();
    let implicit_lib_name = record.package.replace('-', "_");
    match manifest.get("lib").and_then(Value::as_table) {
        Some(library) => targets.push(Target {
            kind: "lib",
            name: library
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&implicit_lib_name)
                .to_string(),
            source: library
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("src/lib.rs")
                .to_string(),
            required_features: required_features(&Value::Table(library.clone())),
        }),
        None if tree.has(&format!("{path}/src/lib.rs")) => targets.push(Target {
            kind: "lib",
            name: implicit_lib_name,
            source: "src/lib.rs".to_string(),
            required_features: Vec::new(),
        }),
        None => {}
    }

    targets.extend(explicit_targets(manifest, "bin", "src/bin", path, report));
    if autodiscovers(manifest, "autobins") {
        if tree.has(&format!("{path}/src/main.rs")) {
            targets.push(Target {
                kind: "bin",
                name: record.package.clone(),
                source: "src/main.rs".to_string(),
                required_features: Vec::new(),
            });
        }
        targets.extend(implicit_targets(tree, path, "src/bin", "bin"));
    }
    for (kind, directory, field) in [
        ("test", "tests", "autotests"),
        ("bench", "benches", "autobenches"),
        ("example", "examples", "autoexamples"),
    ] {
        targets.extend(explicit_targets(manifest, kind, directory, path, report));
        if autodiscovers(manifest, field) {
            targets.extend(implicit_targets(tree, path, directory, kind));
        }
    }

    // An explicit target and its auto-discovered twin are one target, and an
    // untracked source is not reachable in a clean checkout, so a command
    // naming it would fail for the reader this guide is written for.
    targets.retain(|target| {
        let source = format!("{path}/{}", target.source);
        !tree.exists(&source) || tree.has(&source)
    });
    targets.sort();
    targets.dedup();
    targets
}

/// One testing guide.
fn render_guide(
    record: &CrateRecord,
    manifest: &toml::Table,
    targets: &[Target],
    fields: &Fields,
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

    let mut commands = vec![format!("./cargo_full test -p {}", record.package)];
    if !names.is_empty() {
        commands.push(format!(
            "./cargo_full test -p {} --all-features",
            record.package
        ));
    }
    commands.extend(fields.commands.iter().cloned());
    let mut seen = BTreeSet::new();
    commands.retain(|command| seen.insert(command.clone()));

    let mut lines = vec![
        format!("# Testing `{}`", record.package),
        String::new(),
        "Run the default crate suite from the workspace root:".to_string(),
        String::new(),
        "```console".to_string(),
        commands[0].clone(),
        "```".to_string(),
        String::new(),
        record.responsibility.clone(),
        String::new(),
        format!(
            "The crate lives at `{}`. The `{}` owner maintains its",
            record.path, record.owner
        ),
        format!("`{}` testing contract.", record.layer),
        String::new(),
        "## Commands".to_string(),
        String::new(),
    ];
    for command in &commands {
        lines.extend([
            "```console".to_string(),
            command.clone(),
            "```".to_string(),
            String::new(),
        ]);
    }

    lines.extend(["## Feature sets".to_string(), String::new()]);
    if names.is_empty() {
        lines.push("This crate declares no Cargo features.".to_string());
    } else {
        lines.extend([
            format!("- Default feature members: {}", joined(&defaults)),
            format!("- Available manifest features: {}", joined(&names)),
            "- Use the all-features command above to compile every declared feature together."
                .to_string(),
        ]);
    }

    lines.extend([String::new(), "## Cargo targets".to_string(), String::new()]);
    if targets.is_empty() {
        lines.push(
            "Cargo declares no executable, library, test, example, or benchmark target."
                .to_string(),
        );
    } else {
        lines.extend([
            "| Kind | Target | Source | Required features | Focused command |".to_string(),
            "| --- | --- | --- | --- | --- |".to_string(),
        ]);
        for target in targets {
            lines.push(format!(
                "| `{}` | `{}` | `{}/{}` | {} | `{}` |",
                target.kind,
                target.name,
                record.path,
                target.source,
                joined(&target.required_features),
                target.command(&record.package)
            ));
        }
    }

    lines.extend([String::new(), "## Test classes".to_string(), String::new()]);
    lines.extend(fields.test_classes.iter().map(|item| format!("- {item}")));
    lines.extend([
        String::new(),
        "## Hardware requirements".to_string(),
        String::new(),
        fields.hardware.clone(),
        String::new(),
        "## Evidence outputs".to_string(),
        String::new(),
    ]);
    lines.extend(fields.evidence_outputs.iter().map(|item| {
        if item.contains('/') {
            format!("- `{item}`")
        } else {
            format!("- {item}")
        }
    }));
    lines.extend([
        String::new(),
        "## Skips and failures".to_string(),
        String::new(),
        fields.expected_skips.clone(),
        String::new(),
        fields.failure_behavior.clone(),
        String::new(),
    ]);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the guide names a focused command per target, and the flag differs
    /// by kind. A library has no focused flag at all, so emitting one would put
    /// a command in the guide that cargo rejects.
    #[test]
    fn the_focused_command_matches_the_target_kind() {
        let cases = [
            ("test", "./cargo_full test -p demo --test t"),
            ("bin", "./cargo_full test -p demo --bin t"),
            ("example", "./cargo_full test -p demo --example t"),
            ("bench", "./cargo_full test -p demo --bench t"),
            ("lib", "./cargo_full test -p demo"),
        ];
        for (kind, expected) in cases {
            let target = Target {
                kind,
                name: "t".to_string(),
                source: "src/lib.rs".to_string(),
                required_features: Vec::new(),
            };
            assert_eq!(target.command("demo"), expected, "for kind `{kind}`");
        }
    }

    /// WHY: the guide filename is the member directory name, not the package
    /// name, and the two differ across this workspace. Rendering from the
    /// package name would leave every live guide reported as orphaned.
    #[test]
    fn the_guide_filename_is_the_member_directory_name() {
        let record = CrateRecord {
            package: "vyre-conform-spec".to_string(),
            path: "conform/vyre-conform-spec".to_string(),
            owner: "conform".to_string(),
            layer: "spec".to_string(),
            responsibility: String::new(),
            dependencies: Vec::new(),
        };
        assert_eq!(guide_name(&record), "vyre-conform-spec.md");
    }

    /// WHY: metadata layers defaults, then the layer profile, then the package
    /// override, and a narrower level must win. Merging the other way would
    /// render every crate's guide with the default prose.
    #[test]
    fn a_narrower_metadata_level_wins() {
        let metadata = Metadata {
            defaults: toml::from_str(
                "hardware = \"none\"\nexpected_skips = \"none\"\nfailure_behavior = \"fail\"\ntest_classes = [\"unit\"]\nevidence_outputs = [\"none\"]\n",
            )
            .expect("the defaults parse"),
            profiles: BTreeMap::from([(
                "driver".to_string(),
                toml::from_str("hardware = \"a gpu\"\n").expect("the profile parses"),
            )]),
            overrides: BTreeMap::from([(
                "demo".to_string(),
                toml::from_str("expected_skips = \"skips without a device\"\n")
                    .expect("the override parses"),
            )]),
        };
        let record = CrateRecord {
            package: "demo".to_string(),
            path: "demo".to_string(),
            owner: "core".to_string(),
            layer: "driver".to_string(),
            responsibility: String::new(),
            dependencies: Vec::new(),
        };
        let mut report = Report::clean();
        let fields = metadata
            .resolve(&record, &mut report)
            .expect("every required field resolves");
        assert_eq!(fields.hardware, "a gpu");
        assert_eq!(fields.expected_skips, "skips without a device");
        assert_eq!(fields.failure_behavior, "fail");
        assert!(report.findings.is_empty());
    }

    /// WHY: a required field missing at every level is a finding rather than a
    /// guide rendered with an empty section, because an empty hardware section
    /// reads as "no hardware needed".
    #[test]
    fn a_field_missing_at_every_level_is_a_finding() {
        let metadata = Metadata {
            defaults: toml::Table::new(),
            profiles: BTreeMap::from([("driver".to_string(), toml::Table::new())]),
            overrides: BTreeMap::new(),
        };
        let record = CrateRecord {
            package: "demo".to_string(),
            path: "demo".to_string(),
            owner: "core".to_string(),
            layer: "driver".to_string(),
            responsibility: String::new(),
            dependencies: Vec::new(),
        };
        let mut report = Report::clean();
        assert!(metadata.resolve(&record, &mut report).is_none());
        assert_eq!(report.findings.len(), 1);
    }
}
