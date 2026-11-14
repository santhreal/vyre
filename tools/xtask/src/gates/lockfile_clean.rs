//! Verify `Cargo.lock` parses cleanly and carries at most one entry per workspace member.

use std::collections::BTreeMap;
use std::path::Path;

/// Check `Cargo.lock` consistency against workspace member versions.
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let lock_path = root.join("Cargo.lock");
    let Ok(lock_text) = crate::workspace_manifest::read_source_bounded(&lock_path) else {
        return Err(vec!["cannot read Cargo.lock".to_owned()]);
    };
    let Ok(lock_toml) = toml::from_str::<toml::Value>(&lock_text) else {
        return Err(vec!["Cargo.lock is not valid TOML".to_owned()]);
    };

    let mut member_versions: BTreeMap<String, String> = BTreeMap::new();
    let root_doc: toml::Value =
        crate::workspace_manifest::read_source_bounded(&root.join("Cargo.toml"))
            .ok()
            .and_then(|t| toml::from_str(&t).ok())
            .unwrap_or(toml::Value::Boolean(false));
    let ws_ver = root_doc
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .unwrap_or("0.8.0");

    for member in crate::workspace_manifest::workspace_members(root) {
        let Ok(text) =
            crate::workspace_manifest::read_source_bounded(&root.join(&member).join("Cargo.toml"))
        else {
            continue;
        };
        let Ok(doc) = toml::from_str::<toml::Value>(&text) else {
            continue;
        };
        let Some(pkg) = doc.get("package") else {
            continue;
        };
        let Some(name) = pkg.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let ver = pkg
            .get("version")
            .and_then(toml::Value::as_str)
            .unwrap_or(ws_ver);
        member_versions.insert(name.to_owned(), ver.to_owned());
    }

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut errors = Vec::new();
    if let Some(pkgs) = lock_toml.get("package").and_then(toml::Value::as_array) {
        for pkg in pkgs {
            let Some(name) = pkg.get("name").and_then(toml::Value::as_str) else {
                continue;
            };
            if let Some(expected_ver) = member_versions.get(name) {
                if pkg.get("source").is_none() {
                    *counts.entry(name).or_insert(0) += 1;
                    if let Some(actual_ver) = pkg.get("version").and_then(toml::Value::as_str) {
                        if actual_ver != expected_ver {
                            errors.push(format!(
                                "Cargo.lock `{name}` version {actual_ver} != manifest version {expected_ver}"
                            ));
                        }
                    }
                }
            }
        }
    }

    for (name, count) in counts {
        if count > 1 {
            errors.push(format!(
                "Cargo.lock has {count} local entries for workspace member `{name}`"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
