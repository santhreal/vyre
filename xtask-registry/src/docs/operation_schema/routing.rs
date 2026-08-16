//! Which category an operation falls in, which feature enables it, and which
//! manifest has to declare that feature.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(super) fn category_from_id(id: &str) -> String {
    id.split("::")
        .nth(1)
        .or_else(|| id.split('.').next())
        .filter(|value| !value.is_empty())
        .unwrap_or("uncategorized")
        .to_string()
}

pub(super) fn feature_route(id: &str, category: &str) -> Vec<String> {
    if id.starts_with("vyre-primitives::") {
        let domain = id.split("::").nth(1).unwrap_or(category);
        let feature = if domain == "vfs" { "parsing" } else { domain };
        return vec![feature.to_string(), "inventory-registry".to_string()];
    }
    let feature = match category {
        "scan" | "matching" => "matching",
        "crypto" => "crypto",
        "math" | "optim" | "quant" => "math",
        "nn" => "nn",
        "parsing" => "parsing",
        "logical" => "logical",
        "security" => "security",
        "visual" => "visual",
        "hash" => "hash",
        "decode" => "decode",
        "rule" => "rule",
        "text" => "text",
        _ => "full",
    };
    vec![feature.to_string()]
}

pub(super) fn read_manifest_features(
    root: &Path,
    errors: &mut Vec<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut catalog = BTreeMap::new();
    for crate_name in ["vyre-driver", "vyre-primitives", "vyre-libs"] {
        let path = root.join(crate_name).join("Cargo.toml");
        let text = match super::read_text_bounded(&path) {
            Ok(value) => value,
            Err(error) => {
                errors.push(format!(
                    "read {} for operation features: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let value = match toml::from_str::<toml::Value>(&text) {
            Ok(value) => value,
            Err(error) => {
                errors.push(format!(
                    "parse {} for operation features: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let features = value
            .get("features")
            .and_then(toml::Value::as_table)
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();
        catalog.insert(crate_name.to_string(), features);
    }
    catalog
}
