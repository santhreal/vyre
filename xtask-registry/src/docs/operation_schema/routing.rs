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

/// Crate and features that must be enabled for an operation to register.
///
/// An op id names the crate that owned the module when the id was minted, and
/// ids are frozen. The eighteen composition domains moved to `vyre-libs` and
/// kept their `vyre-primitives::` ids, so the route is read from where the
/// module lives now. `vyre-libs` registers unconditionally, so a moved domain
/// needs its own domain feature and nothing else; what is left in the intrinsic
/// crate still needs `inventory-registry` on top of its module gate.
pub(super) fn feature_route(id: &str, category: &str) -> (&'static str, Vec<String>) {
    if id.starts_with("vyre-primitives::") {
        let domain = id.split("::").nth(1).unwrap_or(category);
        if let Some(feature) = moved_domain_feature(domain) {
            return ("vyre-libs", vec![feature.to_string()]);
        }
        // `vfs` is the one module left in the intrinsic crate whose gate is not
        // its own name: `pub mod vfs` sits behind `vyre-foundation`, because the
        // resolver builds a Program and needs the IR types.
        let feature = if domain == "vfs" {
            "vyre-foundation"
        } else {
            domain
        };
        return (
            "vyre-primitives",
            vec![feature.to_string(), "inventory-registry".to_string()],
        );
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
    ("vyre-libs", vec![feature.to_string()])
}

/// `vyre-libs` feature gating the composition domain `domain`, if it moved.
///
/// Four domains are gated by the kernel feature rather than by their own name,
/// because `vyre-libs` already had a dialect feature of that name over the
/// kernels: `math`, `nn`, `matching` and `parsing` each split into a dialect
/// surface and the kernel tree the intrinsic crate used to own.
fn moved_domain_feature(domain: &str) -> Option<&'static str> {
    Some(match domain {
        "math" => "math-kernels",
        "nn" => "nn-kernels",
        "matching" => "matching-kernels",
        "parsing" => "parsing-kernels",
        "bitset" => "bitset",
        "decode" => "decode",
        "fixpoint" => "fixpoint",
        "geom" => "geom",
        "graph" => "graph",
        "hash" => "hash",
        "label" => "label",
        "nfa" => "nfa",
        "opt" => "opt",
        "predicate" => "predicate",
        "reduce" => "reduce",
        "text" => "text",
        "topology" => "topology",
        "visual" => "visual",
        _ => return None,
    })
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
