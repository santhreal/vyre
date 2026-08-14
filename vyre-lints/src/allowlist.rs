//! Configuration for the `raw_ir_in_libs` lint.
//!
//! `measured_roots` names the source trees the rule applies to, and
//! `exempt_files` exempts one file path (relative to the workspace root)
//! from it. An exemption is removed when the file's migration lands.
//!
//! The roots are data rather than a hardcoded crate name because the count
//! is pinned as a ratchet. Relocating a composition domain between crates
//! would otherwise move thousands of construction sites into or out of the
//! measured set in one commit, for no semantic change, and the pin would
//! stop meaning anything. Moving a domain edits this list in the same
//! commit as the move, so the number stays comparable across it.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

const DEFAULT_MEASURED_ROOTS: &[&str] = &["vyre-libs/src"];

#[derive(Debug, Deserialize)]
struct AllowlistFile {
    /// Workspace-root-relative source trees the rule applies to.
    #[serde(default)]
    measured_roots: Vec<String>,
    /// Workspace-root-relative paths to exempt.
    #[serde(default)]
    exempt_files: Vec<String>,
}

/// Where the raw IR construction rule applies and what it exempts.
#[derive(Debug)]
pub struct Allowlist {
    roots: Vec<String>,
    paths: HashSet<String>,
}

impl Default for Allowlist {
    fn default() -> Self {
        Self {
            roots: DEFAULT_MEASURED_ROOTS.iter().map(|r| r.to_string()).collect(),
            paths: HashSet::new(),
        }
    }
}

impl Allowlist {
    /// Construct an allowlist with no exemptions.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Workspace-relative source trees the rule applies to.
    pub fn measured_roots(&self) -> &[String] {
        &self.roots
    }

    /// Return whether a workspace-relative path is exempt.
    pub fn contains(&self, workspace_relative_path: &str) -> bool {
        self.paths.contains(workspace_relative_path)
    }

    /// Return the number of exempt paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Return whether no paths are exempt.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

/// Load an allowlist from a bounded TOML file.
pub fn load(path: &Path) -> Result<Allowlist> {
    let bytes = crate::read_source_bounded(path)
        .with_context(|| format!("read allowlist {}", path.display()))?;
    let parsed: AllowlistFile =
        toml::from_str(&bytes).with_context(|| format!("parse allowlist {}", path.display()))?;
    Ok(Allowlist {
        roots: if parsed.measured_roots.is_empty() {
            DEFAULT_MEASURED_ROOTS.iter().map(|r| r.to_string()).collect()
        } else {
            parsed.measured_roots
        },
        paths: parsed.exempt_files.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_allowlist_contains_nothing() {
        let a = Allowlist::empty();
        assert!(!a.contains("vyre-libs/src/nn/attention/gqa_attention.rs"));
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
    }

    #[test]
    fn loaded_allowlist_contains_listed_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("allowlist.toml");
        std::fs::write(
            &path,
            "exempt_files = [\n  \"vyre-libs/src/nn/attention/gqa_attention.rs\",\n  \"vyre-libs/src/visual/shadow/mod.rs\",\n]\n",
        )
        .unwrap();
        let a = load(&path).unwrap();
        assert!(a.contains("vyre-libs/src/nn/attention/gqa_attention.rs"));
        assert!(a.contains("vyre-libs/src/visual/shadow/mod.rs"));
        assert!(!a.contains("vyre-libs/src/nn/other.rs"));
        assert_eq!(a.len(), 2);
    }

    /// A configured root list is what the rule applies to. This is the whole
    /// point of the field: a caller that ignored it and hardcoded a crate name
    /// would keep working, and the pinned finding count would then jump the
    /// next time a composition domain moved between crates.
    #[test]
    fn configured_measured_roots_replace_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("allowlist.toml");
        std::fs::write(
            &path,
            "measured_roots = [\n  \"vyre-libs/src\",\n  \"vyre-primitives/src/math\",\n]\nexempt_files = []\n",
        )
        .unwrap();

        let a = load(&path).unwrap();

        assert_eq!(a.measured_roots(), ["vyre-libs/src", "vyre-primitives/src/math"]);
    }

    /// An omitted list falls back to the compositions crate rather than to
    /// nothing, so a config that predates the field does not silently turn the
    /// rule into a no-op that reports zero findings.
    #[test]
    fn omitted_measured_roots_fall_back_rather_than_measuring_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("allowlist.toml");
        std::fs::write(&path, "exempt_files = []\n").unwrap();

        let a = load(&path).unwrap();

        assert_eq!(a.measured_roots(), ["vyre-libs/src"]);
        assert_eq!(Allowlist::empty().measured_roots(), ["vyre-libs/src"]);
    }

    /// The shipped configuration is what the pinned count was measured
    /// against, so assert it directly rather than trusting a future edit.
    #[test]
    fn the_shipped_configuration_declares_its_measured_roots() {
        let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("allowlist.toml");

        let a = load(&shipped).expect("shipped allowlist loads");

        assert_eq!(a.measured_roots(), ["vyre-libs/src"]);
    }

    #[test]
    fn missing_allowlist_file_errors() {
        let r = load(Path::new("/nonexistent/path/allowlist.toml"));
        assert!(
            r.is_err(),
            "load of nonexistent path must return Err, got Ok"
        );
    }

    #[test]
    fn malformed_allowlist_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("allowlist.toml");
        std::fs::write(&path, "not valid toml at all = = =").unwrap();
        let r = load(&path);
        assert!(r.is_err());
    }
}
