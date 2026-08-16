//! Which category an operation falls in, and which manifest declares the
//! features that enable it.
//!
//! The feature route itself is read from the checkout by
//! [`super::placement`], because the crate that minted an id is not the crate
//! that holds the code.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Domain segment of one operation id.
///
/// This reads the second segment, the domain the operation was minted under,
/// not the crate prefix. A domain keeps its name when it moves between crates,
/// so this is a naming fact rather than a placement fact, and it stays correct
/// across the composition move. A registration that declares its own category
/// overrides it.
pub(super) fn category_from_id(id: &str) -> String {
    id.split("::")
        .nth(1)
        .or_else(|| id.split('.').next())
        .filter(|value| !value.is_empty())
        .unwrap_or("uncategorized")
        .to_string()
}

/// Features declared by each crate that holds an operation definition.
///
/// The crate list comes from the placements read out of the checkout, so a
/// domain that moves brings its manifest into this catalog without anyone
/// editing a list here.
pub(super) fn read_manifest_features(
    root: &Path,
    crates: &BTreeSet<String>,
    errors: &mut Vec<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut catalog = BTreeMap::new();
    for crate_name in crates {
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
