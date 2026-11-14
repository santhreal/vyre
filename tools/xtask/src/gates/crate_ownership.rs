//! Verify `docs/CRATE_OWNERSHIP.toml` coverage, publication classes, and package authors.

use std::collections::BTreeSet;
use std::path::Path;

const VALID_CLASSES: &[&str] = &[
    "stable-consumer-sdk",
    "extension-sdk",
    "concrete-backend",
    "internal-engine",
    "conformance-tooling",
    "private-test-support",
];

const CANONICAL_AUTHOR: &str = "Santh <64453045+santhreal@users.noreply.github.com>";

/// Check ownership entries and package metadata across workspace members.
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let root_manifest_path = root.join("Cargo.toml");
    let Ok(root_text) = crate::workspace_manifest::read_source_bounded(&root_manifest_path) else {
        return Err(vec![format!(
            "cannot read {}",
            root_manifest_path.display()
        )]);
    };
    let Ok(root_toml) = toml::from_str::<toml::Value>(&root_text) else {
        return Err(vec!["cannot parse root Cargo.toml".to_owned()]);
    };

    if let Some(ws_authors) = root_toml
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("authors"))
        .and_then(toml::Value::as_array)
    {
        if ws_authors.len() != 1 || ws_authors[0].as_str() != Some(CANONICAL_AUTHOR) {
            errors.push(format!(
                r#"Cargo.toml [workspace.package].authors must be `["{CANONICAL_AUTHOR}"]`"#
            ));
        }
    }

    let own_path = root.join("docs/CRATE_OWNERSHIP.toml");
    let Ok(own_text) = crate::workspace_manifest::read_source_bounded(&own_path) else {
        return Err(vec!["missing docs/CRATE_OWNERSHIP.toml".to_owned()]);
    };
    let Ok(own_toml) = toml::from_str::<toml::Value>(&own_text) else {
        return Err(vec!["invalid TOML in docs/CRATE_OWNERSHIP.toml".to_owned()]);
    };

    let mut owned_packages = BTreeSet::new();
    if let Some(crates) = own_toml.get("crate").and_then(toml::Value::as_array) {
        for entry in crates {
            if let Some(pkg) = entry.get("package").and_then(toml::Value::as_str) {
                owned_packages.insert(pkg);
                if entry.get("publication_class").is_some() {
                    errors.push(format!(
                        "docs/CRATE_OWNERSHIP.toml restates `publication_class` for `{pkg}`"
                    ));
                }
            }
        }
    }

    for member in crate::workspace_manifest::workspace_members(root) {
        let manifest_path = root.join(&member).join("Cargo.toml");
        let Ok(text) = crate::workspace_manifest::read_source_bounded(&manifest_path) else {
            continue;
        };
        let Ok(doc) = toml::from_str::<toml::Value>(&text) else {
            errors.push(format!("{member}/Cargo.toml: invalid TOML"));
            continue;
        };
        let Some(pkg) = doc.get("package") else {
            continue;
        };
        let Some(name) = pkg.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        if !owned_packages.contains(name) {
            errors.push(format!(
                "member `{name}` ({member}) missing from docs/CRATE_OWNERSHIP.toml"
            ));
        }
        let pub_class = pkg
            .get("metadata")
            .and_then(|m| m.get("vyre"))
            .and_then(|v| v.get("publication_class"))
            .and_then(toml::Value::as_str);
        match pub_class {
            None => errors.push(format!(
                "{member}/Cargo.toml: missing `[package.metadata.vyre].publication_class`"
            )),
            Some(cls) if !VALID_CLASSES.contains(&cls) => errors.push(format!(
                "{member}/Cargo.toml: unknown publication_class `{cls}`"
            )),
            Some(_) => {}
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
