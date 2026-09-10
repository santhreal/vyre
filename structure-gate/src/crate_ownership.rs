//! The crate ownership registry, read from the checkout at run time.
//!
//! `docs/CRATE_OWNERSHIP.toml` is where the workspace states which subsystem
//! owns each member, where that member lives, and which layer it sits in. Any
//! rule that needs one of those facts reads them from here, so a new member is
//! covered by declaring it once rather than by editing the rules that name it.
//! A rule carrying its own copy of the roster stops covering the tree the day a
//! crate is added, and does it silently.

use std::path::Path;

use serde::Deserialize;

use crate::read_source_bounded;

/// The registry, relative to the checkout root.
pub const REGISTRY: &str = "docs/CRATE_OWNERSHIP.toml";

/// One member, as the registry declares it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrateRow {
    /// Cargo package name.
    pub package: String,
    /// Checkout-relative directory the package occupies.
    pub path: String,
    /// Named seam the package publishes to the rest of the workspace.
    pub seam: String,
    /// Architectural layer the package sits in.
    pub layer: String,
}

/// Every member the registry declares, in declaration order.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    rows: Vec<CrateRow>,
}

impl Registry {
    /// Read the registry out of the checkout at `root`.
    ///
    /// # Errors
    ///
    /// When the file is missing, unreadable, not TOML, declares no member, or
    /// carries a row without a `package`, `path`, `seam` or `layer`. Each of
    /// those leaves a caller with a roster that covers less than the tree, so it
    /// is reported instead of defaulted.
    pub fn read(root: &Path) -> Result<Self, String> {
        let path = root.join(REGISTRY);
        let text = read_source_bounded(&path)
            .map_err(|error| format!("cannot read {REGISTRY}: {error}"))?;
        Self::parse(&text)
    }

    /// Read the registry out of `text`.
    ///
    /// # Errors
    ///
    /// When the text is not TOML, declares no member, or carries a row without a
    /// `package`, `path`, `seam` or `layer`.
    pub fn parse(text: &str) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct OwnershipFile {
            #[serde(default, rename = "crate")]
            crates: Option<Vec<RawCrateEntry>>,
        }

        #[derive(Deserialize)]
        struct RawCrateEntry {
            package: Option<String>,
            path: Option<String>,
            seam: Option<String>,
            layer: Option<String>,
        }

        let file: OwnershipFile = toml::from_str(text)
            .map_err(|error| format!("{REGISTRY} is not readable as TOML: {error}"))?;
        let entries = file
            .crates
            .ok_or_else(|| format!("{REGISTRY} declares no [[crate]] entries"))?;
        let mut rows = Vec::with_capacity(entries.len());
        for entry in entries {
            let package = entry
                .package
                .ok_or_else(|| format!("{REGISTRY} has a [[crate]] entry with no `package`"))?;
            let path = entry
                .path
                .ok_or_else(|| format!("{REGISTRY} entry for `{package}` declares no `path`"))?
                .replace('\\', "/");
            let seam = entry
                .seam
                .ok_or_else(|| format!("{REGISTRY} entry for `{package}` declares no `seam`"))?;
            let layer = entry
                .layer
                .ok_or_else(|| format!("{REGISTRY} entry for `{package}` declares no `layer`"))?;
            rows.push(CrateRow {
                package,
                path,
                seam,
                layer,
            });
        }
        if rows.is_empty() {
            return Err(format!(
                "{REGISTRY} carries no member, so the roster would be empty"
            ));
        }
        Ok(Self { rows })
    }

    /// Every declared member.
    #[must_use]
    pub fn rows(&self) -> &[CrateRow] {
        &self.rows
    }

    /// The member whose directory contains the checkout-relative `path`.
    ///
    /// The longest declared directory wins, so a member nested inside another
    /// member's directory keeps the files under it.
    ///
    /// Both sides are slash-separated: the query is normalised here and the row
    /// was normalised when the registry was read. Normalising only one side let
    /// a row written with backslashes match by prefix and not exactly, so the
    /// same file was owned or unowned depending on which arm answered.
    #[must_use]
    pub fn owning_crate(&self, path: &str) -> Option<&CrateRow> {
        let path = path.replace('\\', "/");
        self.rows
            .iter()
            .filter(|row| path == row.path || path.starts_with(&format!("{}/", row.path)))
            .max_by_key(|row| row.path.len())
    }
}
