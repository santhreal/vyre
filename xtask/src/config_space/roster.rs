//! The facade roster and the declared constraints, read from the ownership record.
//!
//! `docs/CRATE_OWNERSHIP.toml` is where a crate's concern is recorded, and it
//! already carries the publication class that separates the consumer facade
//! from the domain packages it forwards to. A second file for the roster would
//! be a parallel list of the same packages, so the roster is a section of that
//! record: adding a domain is one contiguous edit beside the `[[crate]]` record
//! for the package, and `configuration-model --write` materialises the facade
//! manifest from it.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::ConfigSpaceError;

/// Path of the record the roster and the constraints live in.
pub const OWNERSHIP_RECORD_PATH: &str = "docs/CRATE_OWNERSHIP.toml";

/// One named bundle of facade features.
///
/// An aggregate forwards to no package, so it declares nothing and owns
/// nothing: it is a spelling for a set a consumer would otherwise list.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FacadeAggregate {
    /// Feature name the facade publishes.
    pub name: String,
    /// Facade features the aggregate selects.
    pub selects: Vec<String>,
}

/// One domain package the facade forwards to.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FacadeDomain {
    /// Package name.
    pub package: String,
    /// Whether the facade declares the dependency `optional = true`.
    pub optional: bool,
}

/// One domain feature the facade exposes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FacadeFeature {
    /// Feature name, identical in the facade and in the owning package.
    pub name: String,
    /// Package that declares what the name means.
    pub domain: String,
    /// Facade features this row also selects, which is the facade's curation of
    /// what a selection exposes and is not derivable from the domain manifest.
    #[serde(default)]
    pub requires: Vec<String>,
    /// Third-party optional dependencies the capability needs.
    #[serde(default)]
    pub activates: Vec<String>,
}

/// The consumer facade roster.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FacadeRoster {
    /// Facade package name.
    pub package: String,
    /// Facade features the default selection enables.
    pub default: Vec<String>,
    /// Facade features that forward to no domain package.
    #[serde(default)]
    pub local: Vec<String>,
    /// Named bundles.
    #[serde(default)]
    pub aggregate: Vec<FacadeAggregate>,
    /// Domain packages.
    #[serde(default)]
    pub domain: Vec<FacadeDomain>,
    /// Domain features.
    #[serde(default)]
    pub feature: Vec<FacadeFeature>,
}

impl FacadeRoster {
    /// The domain record for `package`.
    #[must_use]
    pub fn domain(&self, package: &str) -> Option<&FacadeDomain> {
        self.domain.iter().find(|entry| entry.package == package)
    }

    /// Every feature name the facade publishes, `default` and `full` excluded.
    #[must_use]
    pub fn published(&self) -> BTreeSet<String> {
        self.feature
            .iter()
            .map(|entry| entry.name.clone())
            .chain(self.aggregate.iter().map(|entry| entry.name.clone()))
            .chain(self.local.iter().cloned())
            .collect()
    }
}

/// One feature name many packages declare independently by convention.
///
/// A per-package convention such as a package's own device-test switch is
/// declared once per package on purpose, so the ownership rule cannot apply to
/// it. Recording it here rather than in a hardcoded array keeps the exemption
/// beside the rest of the ownership record, and the gate reports an entry that
/// no longer has two declarations so the record cannot silently rot.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SharedCapability {
    /// Feature name every package spells the same way.
    pub name: String,
    /// Why one owning package would be the wrong shape for it.
    pub reason: String,
}

/// Two features of one package that no build may enable together.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExclusiveFeatures {
    /// Package declaring both features.
    pub package: String,
    /// The two feature names.
    pub features: [String; 2],
    /// The technical constraint that makes the pair impossible.
    pub reason: String,
}

/// The declared configuration constraints.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Constraints {
    /// Feature names exempt from the one-owning-package rule.
    #[serde(default)]
    pub shared_capability: Vec<SharedCapability>,
    /// Pairs no build may enable together.
    #[serde(default)]
    pub exclusive: Vec<ExclusiveFeatures>,
}

/// The roster and the constraints, as the ownership record declares them.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
struct RecordSections {
    facade: FacadeRoster,
    #[serde(default)]
    configuration: Constraints,
}

/// Read the roster and the constraints out of the ownership record.
///
/// # Errors
///
/// Returns the reason the record could not be read or does not declare a
/// facade. A model built without the roster would report a facade that matches
/// a roster it never loaded, so there is no default to fall back to.
pub fn load(root: &Path) -> Result<(FacadeRoster, Constraints), ConfigSpaceError> {
    let path = root.join(OWNERSHIP_RECORD_PATH);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        ConfigSpaceError::ManifestParse(format!("{OWNERSHIP_RECORD_PATH}: {error}"))
    })?;
    let sections: RecordSections = toml::from_str(&text).map_err(|error| {
        ConfigSpaceError::ManifestParse(format!("{OWNERSHIP_RECORD_PATH}: {error}"))
    })?;
    Ok((sections.facade, sections.configuration))
}
