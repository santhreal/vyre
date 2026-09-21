//! The workspace feature graph, and which package declares a capability.
//!
//! A feature name that two packages declare independently is ambiguous: a
//! `cfg(feature = "visual")` in one package and the same line in another read
//! two different switches, and a consumer writing `--features visual` cannot
//! tell which semantics it selected. A facade that forwards a domain feature is
//! not a second declaration of it: the body names the owning package's feature
//! and adds nothing, so there is still one definition and one meaning.
//!
//! The two shapes are distinguishable from the manifests alone, which is what
//! this module does.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// One entry of a feature body, by what it reaches.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum BodyEntry {
    /// `dep:X`, which activates optional dependency `X` with its own defaults.
    Activates {
        /// Dependency key in the declaring manifest.
        dependency: String,
    },
    /// `X/G` or `X?/G`, which enables feature `G` of dependency `X`.
    Forwards {
        /// Dependency key in the declaring manifest.
        dependency: String,
        /// Feature of that dependency.
        feature: String,
        /// `true` for `X?/G`, which fires only once `X` is activated.
        weak: bool,
    },
    /// A bare name, which is another feature of the declaring package.
    Local {
        /// Feature of the declaring package.
        feature: String,
    },
}

impl BodyEntry {
    /// Read one `[features]` body entry.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        if let Some(dependency) = text.strip_prefix("dep:") {
            // `dep:X/G` is not cargo syntax, so the whole remainder is the key.
            return Self::Activates {
                dependency: dependency.to_string(),
            };
        }
        match text.split_once('/') {
            Some((dependency, feature)) => Self::Forwards {
                dependency: dependency.trim_end_matches('?').to_string(),
                feature: feature.to_string(),
                weak: dependency.ends_with('?'),
            },
            None => Self::Local {
                feature: text.to_string(),
            },
        }
    }

    /// The dependency this entry reaches, when it reaches one.
    #[must_use]
    pub fn dependency(&self) -> Option<&str> {
        match self {
            Self::Activates { dependency } | Self::Forwards { dependency, .. } => Some(dependency),
            Self::Local { .. } => None,
        }
    }
}

/// Where a feature body puts the capability its name states.
///
/// The four cases are exhaustive over a body, and each one is a different
/// answer to "which package decides what this name means".
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Provenance {
    /// The body reaches the same feature name in a dependency, and every local
    /// feature it names is itself forwarded or aggregated. The dependency
    /// decides what the name means; this package only exposes it.
    Forwarded,
    /// The body names only local features that are themselves forwarded or
    /// aggregated. A named bundle of other packages' capabilities, deciding
    /// nothing of its own.
    Aggregated,
    /// The body decides the capability here: an empty body a `cfg` reads, or one
    /// naming a local feature that is itself declared here.
    Declared,
    /// The body both forwards the name to a dependency and pulls in a local
    /// feature declared here, so what the name means differs by which package a
    /// build enables it from.
    Divided,
}

impl Provenance {
    /// Whether this package decides what the feature name means.
    ///
    /// A divided body decides part of it, which is why it counts: two divided
    /// bodies for one name are two meanings just as surely as two declared
    /// ones.
    #[must_use]
    pub fn declares(self) -> bool {
        match self {
            Self::Declared | Self::Divided => true,
            Self::Forwarded | Self::Aggregated => false,
        }
    }

    /// The predicate a finding states about a body of this shape.
    #[must_use]
    pub fn predicate(self) -> &'static str {
        match self {
            Self::Forwarded => "forwards the name to the package that declares it",
            Self::Aggregated => "bundles other features and declares nothing",
            Self::Declared => "declares the name here",
            Self::Divided => {
                "both forwards the name to a dependency and adds behaviour declared here, \
                 so the capability differs by which package a build enables it from"
            }
        }
    }
}

/// One package's feature table, as the configuration model reads it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PackageFeatures {
    /// Every declared feature and the parsed body it carries.
    pub features: BTreeMap<String, Vec<BodyEntry>>,
    /// Dependency keys declared `optional = true`.
    pub optional_dependencies: BTreeSet<String>,
    /// Every dependency key, optional or not.
    pub dependencies: BTreeSet<String>,
    /// Feature names any target's `required-features` names.
    pub required_features: BTreeSet<String>,
    /// `package.metadata.vyre.publication_class`.
    pub publication_class: String,
}

impl PackageFeatures {
    /// Every feature name `--features` accepts, `default` included.
    ///
    /// An optional dependency that no body names with `dep:` also publishes an
    /// implicit feature of its own key, and a build can enable it.
    #[must_use]
    pub fn enable_able(&self) -> BTreeSet<String> {
        let mut names: BTreeSet<String> = self.features.keys().cloned().collect();
        let named: BTreeSet<&str> = self
            .features
            .values()
            .flatten()
            .filter_map(|entry| match entry {
                BodyEntry::Activates { dependency } => Some(dependency.as_str()),
                BodyEntry::Forwards { .. } | BodyEntry::Local { .. } => None,
            })
            .collect();
        for dependency in &self.optional_dependencies {
            if !named.contains(dependency.as_str()) {
                names.insert(dependency.clone());
            }
        }
        names
    }
}

/// Every workspace member's feature table, keyed by package name.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FeatureGraph {
    /// Feature tables by package name.
    pub packages: BTreeMap<String, PackageFeatures>,
}

/// Cargo reserves this name for the selection a build gets without `--features`.
///
/// Every package has one, so it can never resolve to a single owning package and
/// is not a capability name at all.
pub const RESERVED_DEFAULT: &str = "default";

impl FeatureGraph {
    /// Read one member's feature table out of its parsed manifest.
    #[must_use]
    pub fn package_features(manifest: &toml::Value) -> PackageFeatures {
        let mut features = BTreeMap::new();
        if let Some(table) = manifest.get("features").and_then(toml::Value::as_table) {
            for (name, body) in table {
                let entries = body
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(toml::Value::as_str)
                    .map(BodyEntry::parse)
                    .collect();
                features.insert(name.clone(), entries);
            }
        }

        let mut dependencies = BTreeSet::new();
        let mut optional_dependencies = BTreeSet::new();
        for section in ["dependencies", "build-dependencies"] {
            let Some(table) = manifest.get(section).and_then(toml::Value::as_table) else {
                continue;
            };
            for (key, spec) in table {
                dependencies.insert(key.clone());
                if spec
                    .get("optional")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(false)
                {
                    optional_dependencies.insert(key.clone());
                }
            }
        }
        if let Some(table) = manifest
            .get("dev-dependencies")
            .and_then(toml::Value::as_table)
        {
            dependencies.extend(table.keys().cloned());
        }

        let mut required_features = BTreeSet::new();
        for section in ["test", "bench", "example", "bin"] {
            for target in manifest
                .get(section)
                .and_then(toml::Value::as_array)
                .into_iter()
                .flatten()
            {
                required_features.extend(
                    target
                        .get("required-features")
                        .and_then(toml::Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(toml::Value::as_str)
                        .map(str::to_string),
                );
            }
        }

        let publication_class = manifest
            .get("package")
            .and_then(|package| package.get("metadata"))
            .and_then(|metadata| metadata.get("vyre"))
            .and_then(|vyre| vyre.get("publication_class"))
            .and_then(toml::Value::as_str)
            .unwrap_or("internal-engine")
            .to_string();

        PackageFeatures {
            features,
            optional_dependencies,
            dependencies,
            required_features,
            publication_class,
        }
    }

    /// Where the body of `feature` in `package` puts the capability.
    ///
    /// Returns `None` when the package or the feature is not declared, which a
    /// caller reports rather than guessing at.
    #[must_use]
    pub fn provenance(&self, package: &str, feature: &str) -> Option<Provenance> {
        self.provenance_guarded(package, feature, &mut Vec::new())
    }

    fn provenance_guarded(
        &self,
        package: &str,
        feature: &str,
        visiting: &mut Vec<String>,
    ) -> Option<Provenance> {
        let body = self.packages.get(package)?.features.get(feature)?;
        if visiting.iter().any(|name| name == feature) {
            // A local cycle among this package's own features adds no reach of
            // its own, so the cycle edge contributes nothing either way.
            return Some(Provenance::Aggregated);
        }
        let forwards_this_name = body.iter().any(|entry| match entry {
            BodyEntry::Forwards {
                feature: forwarded, ..
            } => forwarded == feature,
            BodyEntry::Activates { dependency } => {
                self.dependency_default_enables(dependency, feature)
            }
            BodyEntry::Local { .. } => false,
        });

        visiting.push(feature.to_string());
        let local_declares = body
            .iter()
            .filter_map(|entry| match entry {
                BodyEntry::Local { feature } => Some(feature.as_str()),
                BodyEntry::Activates { .. } | BodyEntry::Forwards { .. } => None,
            })
            .any(|local| {
                self.provenance_guarded(package, local, visiting)
                    .is_none_or(Provenance::declares)
            });
        visiting.pop();

        let names_local = body
            .iter()
            .any(|entry| matches!(entry, BodyEntry::Local { .. }));

        Some(match (forwards_this_name, local_declares, names_local) {
            (true, true, _) => Provenance::Divided,
            (true, false, _) => Provenance::Forwarded,
            (false, false, true) => Provenance::Aggregated,
            (false, _, _) => Provenance::Declared,
        })
    }

    /// Whether activating `dependency` alone turns on its feature `feature`.
    ///
    /// `dep:X` is how a facade exposes a domain package whose `default` already
    /// names the capability, so it forwards the name just as `X/F` does.
    fn dependency_default_enables(&self, dependency: &str, feature: &str) -> bool {
        let Some(package) = self.packages.get(dependency) else {
            return false;
        };
        let mut pending: Vec<&str> = package
            .features
            .get(RESERVED_DEFAULT)
            .into_iter()
            .flatten()
            .filter_map(|entry| match entry {
                BodyEntry::Local { feature } => Some(feature.as_str()),
                BodyEntry::Activates { .. } | BodyEntry::Forwards { .. } => None,
            })
            .collect();
        let mut seen = BTreeSet::new();
        while let Some(name) = pending.pop() {
            if name == feature {
                return true;
            }
            if !seen.insert(name) {
                continue;
            }
            pending.extend(
                package
                    .features
                    .get(name)
                    .into_iter()
                    .flatten()
                    .filter_map(|entry| match entry {
                        BodyEntry::Local { feature } => Some(feature.as_str()),
                        BodyEntry::Activates { .. } | BodyEntry::Forwards { .. } => None,
                    }),
            );
        }
        false
    }

    /// Every package that decides what `feature` means, in package-name order.
    #[must_use]
    pub fn owners(&self, feature: &str) -> Vec<String> {
        if feature == RESERVED_DEFAULT {
            return Vec::new();
        }
        self.packages
            .keys()
            .filter(|package| {
                self.provenance(package, feature)
                    .is_some_and(Provenance::declares)
            })
            .cloned()
            .collect()
    }

    /// Every feature name any member declares, `default` excluded.
    #[must_use]
    pub fn feature_names(&self) -> BTreeSet<String> {
        self.packages
            .values()
            .flat_map(|package| package.features.keys())
            .filter(|name| *name != RESERVED_DEFAULT)
            .cloned()
            .collect()
    }
}
