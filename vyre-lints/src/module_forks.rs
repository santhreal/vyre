//! Same-name module fork scanner.
//!
//! Exact duplicate hashes miss the common architectural failure mode:
//! two crates keep files with the same domain name and drift independently.
//! This scanner is intentionally root-scoped so callers can compare domain
//! roots such as `vyre-primitives/src/graph` and `vyre-self-substrate/src`
//! without flagging every generic `mod.rs` or `tests.rs` in the workspace.

use crate::{paths::workspace_relative, Violation, ViolationKind};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const EXEMPT_BASENAMES: &[&str] = &[
    "lib.rs",
    "main.rs",
    "mod.rs",
    "tests.rs",
    "error.rs",
    "types.rs",
];

pub fn scan_roots(roots: &[&Path]) -> Result<Vec<Violation>> {
    let mut by_basename: BTreeMap<String, Vec<ModuleHit>> = BTreeMap::new();
    for (root_index, root) in roots.iter().enumerate() {
        scan_root(root_index, root, &mut by_basename)?;
    }

    let mut violations = Vec::new();
    for (basename, mut paths) in by_basename {
        let distinct_roots = paths
            .iter()
            .map(|hit| hit.root_index)
            .collect::<BTreeSet<_>>();
        if distinct_roots.len() <= 1 {
            continue;
        }
        paths.sort_by(|a, b| a.path.cmp(&b.path));
        let joined = paths
            .iter()
            .map(|hit| hit.path.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        for hit in paths {
            violations.push(Violation {
                file: hit.path,
                line: 1,
                column: 1,
                kind: ViolationKind::ModuleFork,
                message: format!(
                    "module filename `{basename}` appears in multiple scanned roots: {joined}. Fix: pick one authority module and make the other crate a thin adapter or give the adapter a domain-specific name."
                ),
            });
        }
    }
    violations.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));
    Ok(violations)
}

#[derive(Clone, Debug)]
struct ModuleHit {
    root_index: usize,
    path: String,
}

fn scan_root(
    root_index: usize,
    root: &Path,
    by_basename: &mut BTreeMap<String, Vec<ModuleHit>>,
) -> Result<()> {
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let basename = path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("read module filename {}", path.display()))?;
        if EXEMPT_BASENAMES.contains(&basename) {
            continue;
        }
        by_basename
            .entry(basename.to_string())
            .or_default()
            .push(ModuleHit {
                root_index,
                path: workspace_relative(path),
            });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exempt_generic_module_names() {
        assert!(EXEMPT_BASENAMES.contains(&"mod.rs"));
        assert!(EXEMPT_BASENAMES.contains(&"tests.rs"));
    }
}
