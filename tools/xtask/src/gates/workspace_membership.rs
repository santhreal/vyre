//! Verify `[workspace].members` matches manifests on disk and path dependencies resolve.

use std::collections::BTreeSet;
use std::path::Path;

/// Check workspace member declarations and path dependency targets.
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let members: BTreeSet<String> = crate::workspace_manifest::workspace_members(root)
        .into_iter()
        .collect();
    let excludes: BTreeSet<String> = crate::workspace_manifest::workspace_excludes(root)
        .into_iter()
        .collect();

    for member in &members {
        let manifest = root.join(member).join("Cargo.toml");
        if !manifest.is_file() {
            errors.push(format!(
                "workspace member `{member}` has no Cargo.toml on disk"
            ));
        }
    }

    for dir in crate::workspace_manifest::manifest_directories(root) {
        if excludes.contains(&dir) || excludes.iter().any(|ex| dir.starts_with(&format!("{ex}/"))) {
            continue;
        }
        if !members.contains(&dir) {
            errors.push(format!(
                "`{dir}/Cargo.toml` exists on disk but `{dir}` is absent from [workspace].members"
            ));
        }
    }

    let root_manifest = root.join("Cargo.toml");
    if let Ok(text) = crate::workspace_manifest::read_source_bounded(&root_manifest) {
        if let Ok(doc) = toml::from_str::<toml::Value>(&text) {
            if let Some(ws_deps) = doc
                .get("workspace")
                .and_then(|w| w.get("dependencies"))
                .and_then(toml::Value::as_table)
            {
                for (dep, val) in ws_deps {
                    if let Some(p) = val
                        .as_table()
                        .and_then(|t| t.get("path"))
                        .and_then(toml::Value::as_str)
                    {
                        if !root.join(p).join("Cargo.toml").is_file() {
                            errors.push(format!(
                                "[workspace.dependencies].{dep} path `{p}` has no Cargo.toml"
                            ));
                        }
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
