//! The `dep-drift` gate: workspace-managed dependency pins stay aligned.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

use toml::Value;

use crate::gate::{Finding, GateCtx, GateError, Report};

const MAX_DEP_DRIFT_MANIFEST_BYTES: u64 = 1_048_576;

/// What a manifest that disagrees with the workspace table must do about it.
const FIX: &str = "pin the dependency with `workspace = true`, or align the version with the workspace-managed dependency table";

/// Holds every manifest to the version the workspace table manages.
pub struct DepDrift;

impl crate::gate::GateBehavior for DepDrift {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let workspace_manifest = ctx.root.join("Cargo.toml");
        let workspace_text = read_text_bounded(&workspace_manifest).map_err(|error| {
            GateError::new(
                format!(
                    "cannot read the workspace manifest {}: {error}",
                    workspace_manifest.display()
                ),
                "restore the workspace manifest before running any gate",
            )
        })?;
        let workspace_toml = parse_manifest(&workspace_manifest, &workspace_text)?;

        let managed = managed_dependency_versions(&workspace_toml);
        let mut manifests = BTreeSet::new();
        let mut report = Report::clean();
        collect_manifests(&ctx.root, &mut manifests, &mut report);
        manifests.remove(&workspace_manifest);
        report.cover_complete("workspace manifests", manifests.len());

        report.note(format!(
            "{} workspace-managed dependencies across {} manifests",
            managed.len(),
            manifests.len()
        ));
        for manifest in &manifests {
            let text = read_text_bounded(manifest).map_err(|error| {
                GateError::new(
                    format!("cannot read {}: {error}", manifest.display()),
                    "make every tracked manifest readable",
                )
            })?;
            let parsed = parse_manifest(manifest, &text)?;
            collect_manifest_findings(manifest, &parsed, &managed, &ctx.root, &mut report);
        }
        Ok(report)
    }
}

fn parse_manifest(path: &Path, text: &str) -> Result<Value, GateError> {
    let table: toml::Table = toml::from_str(text).map_err(|error| {
        GateError::new(
            format!("cannot parse the manifest {}: {error}", path.display()),
            "repair the manifest syntax; a manifest this gate cannot read is a manifest it cannot judge",
        )
    })?;
    Ok(Value::Table(table))
}

fn managed_dependency_versions(workspace_toml: &Value) -> BTreeMap<String, String> {
    workspace_toml
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(Value::as_table)
        .map(|dependencies| {
            dependencies
                .iter()
                .filter_map(|(name, value)| {
                    explicit_version(value).map(|version| (name.clone(), version))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn collect_manifests(root: &Path, sink: &mut BTreeSet<PathBuf>, report: &mut Report) {
    if !root.exists() {
        report.find(Finding::in_file(
            root,
            "manifest scan root does not exist",
            "restore the directory, or stop naming it as a scan root",
        ));
        return;
    }
    for entry in crate::tree_walk::pruned(root, crate::tree_walk::BUILD_OUTPUT_AND_VCS) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.find(Finding::in_file(
                    error.path().unwrap_or(root),
                    format!("cannot read the manifest scan tree: {error}"),
                    "make every directory under the scan root readable; an unreadable directory hides every manifest under it",
                ));
                continue;
            }
        };
        if entry.file_name() == "Cargo.toml" {
            sink.insert(entry.into_path());
        }
    }
}

fn collect_manifest_findings(
    manifest_path: &Path,
    manifest: &Value,
    managed: &BTreeMap<String, String>,
    root: &Path,
    report: &mut Report,
) {
    for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
        check_dependency_table(
            manifest_path,
            section,
            manifest.get(section).and_then(Value::as_table),
            managed,
            root,
            report,
        );
    }

    if let Some(targets) = manifest.get("target").and_then(Value::as_table) {
        for (target_name, target_table) in targets {
            let target = target_table.as_table();
            for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
                check_dependency_table(
                    manifest_path,
                    &format!("target.{target_name}.{section}"),
                    target
                        .and_then(|table| table.get(section))
                        .and_then(Value::as_table),
                    managed,
                    root,
                    report,
                );
            }
        }
    }
}

fn check_dependency_table(
    manifest_path: &Path,
    section: &str,
    table: Option<&toml::map::Map<String, Value>>,
    managed: &BTreeMap<String, String>,
    root: &Path,
    report: &mut Report,
) {
    let Some(table) = table else {
        return;
    };
    for (dependency, spec) in table {
        let Some(managed_version) = managed.get(dependency) else {
            continue;
        };
        let Some(pinned_version) = explicit_version(spec) else {
            continue;
        };
        if &pinned_version != managed_version {
            report.find(
                Finding::in_file(
                    manifest_path,
                    format!(
                        "`{dependency}` in [{section}] pins `{pinned_version}` but the workspace manages `{managed_version}`"
                    ),
                    FIX,
                )
                .relative_to(root),
            );
        }
    }
}

fn explicit_version(value: &Value) -> Option<String> {
    match value {
        Value::String(version) => Some(version.clone()),
        Value::Table(table) => {
            if table
                .get("workspace")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                return None;
            }
            table
                .get("version")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        }
        _ => None,
    }
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(
        path,
        MAX_DEP_DRIFT_MANIFEST_BYTES,
        "dependency drift manifest",
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dep_drift_detects_mismatched_dependency_versions_and_ignores_workspace_inheritance() {
        let root = Path::new("/workspace");
        let mut managed = BTreeMap::new();
        managed.insert("serde".to_string(), "1.0.200".to_string());
        managed.insert("tokio".to_string(), "1.38.0".to_string());

        let mut table = toml::map::Map::new();
        table.insert("serde".to_string(), Value::String("1.0.200".to_string()));
        table.insert("tokio".to_string(), Value::String("1.30.0".to_string()));
        let mut ws_table = toml::map::Map::new();
        ws_table.insert("workspace".to_string(), Value::Boolean(true));
        table.insert("serde_json".to_string(), Value::Table(ws_table));

        let mut report = Report::clean();
        check_dependency_table(
            Path::new("crates/my_crate/Cargo.toml"),
            "dependencies",
            Some(&table),
            &managed,
            root,
            &mut report,
        );

        assert_eq!(report.findings.len(), 1);
        let finding = &report.findings[0];
        assert!(finding.message.contains(
            "`tokio` in [dependencies] pins `1.30.0` but the workspace manages `1.38.0`"
        ));
    }
}
