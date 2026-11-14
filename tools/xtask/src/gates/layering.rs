//! Verify acyclic workspace production dependency graph and CRATE_OWNERSHIP layer order.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Check workspace production dependency layering and acyclicity.
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let own_path = root.join("docs/CRATE_OWNERSHIP.toml");
    let Ok(own_text) = crate::workspace_manifest::read_source_bounded(&own_path) else {
        return Err(vec!["missing docs/CRATE_OWNERSHIP.toml".to_owned()]);
    };
    let Ok(own_toml) = toml::from_str::<toml::Value>(&own_text) else {
        return Err(vec!["invalid docs/CRATE_OWNERSHIP.toml".to_owned()]);
    };

    let mut layer_ranks: BTreeMap<&str, i64> = BTreeMap::new();
    if let Some(layers) = own_toml.get("layer").and_then(toml::Value::as_array) {
        for layer in layers {
            if let (Some(name), Some(rank)) = (
                layer.get("name").and_then(toml::Value::as_str),
                layer.get("rank").and_then(toml::Value::as_integer),
            ) {
                layer_ranks.insert(name, rank);
            }
        }
    }

    let mut pkg_layer: BTreeMap<&str, &str> = BTreeMap::new();
    if let Some(crates) = own_toml.get("crate").and_then(toml::Value::as_array) {
        for c in crates {
            if let (Some(pkg), Some(layer)) = (
                c.get("package").and_then(toml::Value::as_str),
                c.get("layer").and_then(toml::Value::as_str),
            ) {
                pkg_layer.insert(pkg, layer);
            }
        }
    }

    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut member_names = BTreeSet::new();

    for member in crate::workspace_manifest::workspace_members(root) {
        let manifest_path = root.join(&member).join("Cargo.toml");
        let Ok(text) = crate::workspace_manifest::read_source_bounded(&manifest_path) else {
            continue;
        };
        let Ok(doc) = toml::from_str::<toml::Value>(&text) else {
            continue;
        };
        let Some(name) = doc
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(toml::Value::as_str)
        else {
            continue;
        };
        member_names.insert(name.to_owned());
        let deps_set = edges.entry(name.to_owned()).or_default();
        collect_prod_deps(&doc, deps_set);
    }

    // Filter edges to workspace-internal production dependencies and check acyclicity + rank order.
    for (pkg, deps) in &edges {
        let src_rank = pkg_layer
            .get(pkg.as_str())
            .and_then(|l| layer_ranks.get(l))
            .copied();
        for dep in deps {
            if !member_names.contains(dep) {
                continue;
            }
            if let (Some(sr), Some(dr)) = (
                src_rank,
                pkg_layer
                    .get(dep.as_str())
                    .and_then(|l| layer_ranks.get(l))
                    .copied(),
            ) {
                if sr < dr {
                    errors.push(format!(
                        "upward dependency `{pkg}` (rank {sr}) -> `{dep}` (rank {dr})"
                    ));
                }
            }
        }
    }

    // Cycle check via DFS coloring (0 = unvisited, 1 = active, 2 = done).
    let mut state: BTreeMap<&str, u8> = BTreeMap::new();
    for pkg in &member_names {
        if state.get(pkg.as_str()).copied().unwrap_or(0) == 0 {
            dfs_cycle(pkg.as_str(), &edges, &member_names, &mut state, &mut errors);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn collect_prod_deps(doc: &toml::Value, out: &mut BTreeSet<String>) {
    for section in ["dependencies", "build-dependencies"] {
        if let Some(tbl) = doc.get(section).and_then(toml::Value::as_table) {
            for (k, v) in tbl {
                let pkg_name = v
                    .as_table()
                    .and_then(|t| t.get("package"))
                    .and_then(toml::Value::as_str)
                    .unwrap_or(k.as_str());
                out.insert(pkg_name.to_owned());
            }
        }
    }
    if let Some(targets) = doc.get("target").and_then(toml::Value::as_table) {
        for (_, tv) in targets {
            collect_prod_deps(tv, out);
        }
    }
}

fn dfs_cycle<'a>(
    node: &'a str,
    edges: &'a BTreeMap<String, BTreeSet<String>>,
    members: &'a BTreeSet<String>,
    state: &mut BTreeMap<&'a str, u8>,
    errors: &mut Vec<String>,
) {
    state.insert(node, 1);
    if let Some(deps) = edges.get(node) {
        for dep in deps {
            if !members.contains(dep) {
                continue;
            }
            match state.get(dep.as_str()).copied().unwrap_or(0) {
                1 => errors.push(format!("dependency cycle involving `{node}` -> `{dep}`")),
                0 => dfs_cycle(dep.as_str(), edges, members, state, errors),
                _ => {}
            }
        }
    }
    state.insert(node, 2);
}
