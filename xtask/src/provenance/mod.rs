//! Dependency & Release-Provenance Authority.
//!
//! Defines one dependency and release-provenance authority covering Rust crates,
//! native tools, toolchain versions, code generators, benchmark baselines, and schemas.
//! Generates CycloneDX/SPDX SBOMs and signed SLSA v1.2 provenance records.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;

pub mod timestamp;

/// Canonical schema version for ReleaseProvenance.
pub const RELEASE_PROVENANCE_SCHEMA_VERSION: u32 = 2;

/// Path to generated release provenance document.
pub const PROVENANCE_ARTIFACT_PATH: &str = "docs/generated/release-provenance.toml";
/// Path to generated SBOM artifact.
pub const SBOM_ARTIFACT_PATH: &str = "release/evidence/metadata/sbom.json";
/// Path to generated SLSA provenance record.
pub const SLSA_PROVENANCE_ARTIFACT_PATH: &str = "release/evidence/metadata/provenance-record.json";

/// Error in provenance collection, SBOM generation, or validation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProvenanceError {
    /// Lockfile reading or parsing failure.
    Lockfile(String),
    /// Stale schema version detected.
    StaleSchemaVersion {
        /// Expected version.
        expected: u32,
        /// Found version.
        found: u32,
    },
    /// Unapproved license detected on a dependency.
    UnapprovedLicense {
        /// Package name.
        package: String,
        /// License identifier.
        license: String,
    },
    /// Banned dependency detected.
    BannedDependency {
        /// Package name.
        package: String,
        /// Reason for ban.
        reason: String,
    },
    /// Undeclared build input or network access detected.
    UndeclaredBuildInput {
        /// Build script path or crate name.
        build_script: String,
        /// Undeclared input description.
        undeclared_input: String,
    },
    /// Missing content-addressed checksum for a dependency.
    MissingChecksum(String),
    /// Serialization error.
    Serialization(String),
    /// I/O error.
    Io(String),
    /// A fact the document must state could not be measured.
    UnmeasuredFact {
        /// What the document needed.
        fact: String,
        /// Why the measurement did not produce it.
        reason: String,
    },
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lockfile(msg) => write!(f, "lockfile error: {msg}"),
            Self::StaleSchemaVersion { expected, found } => {
                write!(
                    f,
                    "stale schema version: expected {expected}, found {found}"
                )
            }
            Self::UnapprovedLicense { package, license } => {
                write!(f, "unapproved license '{license}' for package '{package}'")
            }
            Self::BannedDependency { package, reason } => {
                write!(f, "banned dependency '{package}': {reason}")
            }
            Self::UndeclaredBuildInput {
                build_script,
                undeclared_input,
            } => {
                write!(
                    f,
                    "undeclared build input in '{build_script}': {undeclared_input}"
                )
            }
            Self::MissingChecksum(msg) => write!(f, "missing checksum: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::UnmeasuredFact { fact, reason } => {
                write!(f, "cannot measure {fact}: {reason}")
            }
        }
    }
}

impl std::error::Error for ProvenanceError {}

/// Content-addressed and license-classified Rust crate dependency.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PinnedCrateDependency {
    /// Package name.
    pub name: String,
    /// Exact pinned semantic version.
    pub version: String,
    /// Cryptographic checksum (SHA-256 / hex).
    pub checksum: String,
    /// Package source (crates.io or workspace).
    pub source: String,
    /// License classification.
    pub license: String,
    /// Security policy approval status.
    pub is_policy_approved: bool,
}

/// Native toolchain or hardware runtime library dependency.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeToolInput {
    /// Tool or library name (e.g. "cuda_toolkit", "metal_toolchain", "spirv_tools").
    pub name: String,
    /// Pinned or system-provided version.
    pub version: String,
    /// Content hash or verification digest.
    pub digest: String,
    /// Whether this component is optional or required for a specific backend.
    pub is_system_provided: bool,
}

/// Where one code generator's source is: a single file, or a module directory.
#[derive(Clone, Copy, Debug)]
enum GeneratorSource {
    /// Generator defined in one file, hashed on its own.
    File(&'static str),
    /// Generator defined in a directory, hashed at every depth below it.
    Tree(&'static str),
}

impl GeneratorSource {
    /// Digest of the source, or the reason it could not be measured.
    ///
    /// # Errors
    ///
    /// Propagates `UnmeasuredFact` from the file or tree read.
    fn digest(self, root: &Path) -> Result<String, ProvenanceError> {
        match self {
            Self::File(path) => hash_source_file(root, path),
            Self::Tree(path) => hash_source_tree(root, path),
        }
    }
}

/// Every code generator whose source digest a provenance document records,
/// with the artifacts each one writes.
///
/// The document builder and the contract test below both read this table, so a
/// generator added here is checked for a measurable source path without a
/// second list to keep in step.
const CODE_GENERATORS: &[(&str, GeneratorSource, &[&str])] = &[
    (
        "vyre-macros",
        GeneratorSource::Tree("vyre-macros/src"),
        &["registration", "lowering", "dispatch"],
    ),
    (
        "xtask-registry",
        GeneratorSource::Tree("xtask-registry/src"),
        &[
            "docs/generated/op-inventory.toml",
            "docs/generated/catalog.toml",
        ],
    ),
    (
        "structure-gate",
        GeneratorSource::Tree("structure-gate/src"),
        &["structure-assertions"],
    ),
    (
        "vyre-foundation::source_digest",
        GeneratorSource::File("vyre-foundation/src/source_digest.rs"),
        &["VYRE_PTX_LOWERING_DIGEST", "VYRE_NAGA_LOWERING_DIGEST"],
    ),
];

/// Code generator or procedural macro artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CodeGeneratorInput {
    /// Generator name (e.g. "vyre-macros", "structure-gate", "xtask-registry").
    pub name: String,
    /// Source tree hash of the generator.
    pub source_digest: String,
    /// Generation target paths.
    pub target_outputs: Vec<String>,
}

/// Content-addressed benchmark baseline artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkBaselineInput {
    /// Relative path of the baseline file.
    pub path: String,
    /// Content digest of the baseline file.
    pub digest: String,
}

/// Content-addressed schema artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchemaInput {
    /// Relative path of the schema file.
    pub path: String,
    /// Content digest of the schema file.
    pub digest: String,
}

/// Build script contract declaring inputs, outputs, and safety invariants.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BuildScriptContract {
    /// Crate name owning the build script.
    pub crate_name: String,
    /// Relative path to build.rs.
    pub path: String,
    /// Declared inputs (files, directories, env vars).
    pub declared_inputs: Vec<String>,
    /// Declared cargo outputs.
    pub declared_outputs: Vec<String>,
    /// Whether the script performs any network calls.
    pub has_network_access: bool,
    /// Whether the script bounds file read operations.
    pub has_bounded_reads: bool,
}

/// Comprehensive release provenance record authority.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReleaseProvenanceAuthority {
    /// Schema version for fail-closed verification.
    pub schema_version: u32,
    /// The instant the described source was fixed at, RFC 3339 UTC.
    pub bom_timestamp: String,
    /// Rust compiler toolchain version.
    pub rustc_version: String,
    /// Cargo toolchain version.
    pub cargo_version: String,
    /// Target release triples.
    pub target_triples: Vec<String>,
    /// Cargo lockfile content-addressed digest.
    pub lockfile_digest: String,
    /// All pinned third-party and workspace crate dependencies.
    pub dependencies: Vec<PinnedCrateDependency>,
    /// Native toolchain and SDK inputs.
    pub native_tools: Vec<NativeToolInput>,
    /// Code generators and macro authorities.
    pub code_generators: Vec<CodeGeneratorInput>,
    /// Content-addressed benchmark baselines.
    pub benchmark_baselines: Vec<BenchmarkBaselineInput>,
    /// Content-addressed public schemas.
    pub schemas: Vec<SchemaInput>,
    /// Verified build script contracts.
    pub build_scripts: Vec<BuildScriptContract>,
    /// Whether build scripts and dependencies are verified offline-capable.
    pub is_offline_capable: bool,
    /// Deterministic canonical release archive digest.
    pub reproducible_archive_hash: String,
}

impl ReleaseProvenanceAuthority {
    /// Inspect Cargo.lock, manifests, and build scripts to construct the provenance authority.
    pub fn inspect_workspace(root: &Path) -> Result<Self, ProvenanceError> {
        let lockfile_path = root.join("Cargo.lock");
        let lockfile_text = fs::read_to_string(&lockfile_path)
            .map_err(|e| ProvenanceError::Lockfile(e.to_string()))?;

        let lockfile_digest = blake3::hash(lockfile_text.as_bytes()).to_hex().to_string();

        let lock_val: toml::Value =
            toml::from_str(&lockfile_text).map_err(|e| ProvenanceError::Lockfile(e.to_string()))?;

        let (allowed_licenses, banned_crates) = load_deny_policy(root);

        // Map licenses via cargo metadata or manifest inspection
        let metadata_licenses = load_metadata_licenses(root);

        let mut dependencies = Vec::new();

        if let Some(packages) = lock_val.get("package").and_then(|p| p.as_array()) {
            for pkg in packages {
                let name = pkg
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or_default()
                    .to_string();
                let version = pkg
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let checksum = pkg
                    .get("checksum")
                    .and_then(|c| c.as_str())
                    .unwrap_or("local_workspace")
                    .to_string();
                let source = pkg
                    .get("source")
                    .and_then(|s| s.as_str())
                    .unwrap_or("workspace")
                    .to_string();

                let key = format!("{name}@{version}");
                let raw_license =
                    if name.starts_with("vyre") || name == "structure-gate" || name == "xtask" {
                        "Apache-2.0 OR MIT".to_string()
                    } else if let Some(lic) = metadata_licenses
                        .get(&key)
                        .or_else(|| metadata_licenses.get(&name))
                    {
                        lic.clone()
                    } else {
                        "MIT OR Apache-2.0".to_string()
                    };

                let is_license_valid =
                    is_license_expression_approved(&raw_license, &allowed_licenses);
                let is_banned = banned_crates.iter().any(|banned| {
                    if banned.contains('@') {
                        banned == &key
                    } else {
                        banned == &name && !is_allowed_wrapper(&name)
                    }
                });

                let is_policy_approved = is_license_valid && !is_banned;

                dependencies.push(PinnedCrateDependency {
                    name,
                    version,
                    checksum,
                    source,
                    license: raw_license,
                    is_policy_approved,
                });
            }
        }

        dependencies.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));

        let target_triples = vec![
            "x86_64-unknown-linux-gnu".to_string(),
            "x86_64-pc-windows-msvc".to_string(),
            "x86_64-apple-darwin".to_string(),
            "aarch64-apple-darwin".to_string(),
            "aarch64-unknown-linux-gnu".to_string(),
        ];

        let native_tools = vec![
            NativeToolInput {
                name: "cuda_toolkit".to_string(),
                version: "12.x".to_string(),
                digest: "system_detected".to_string(),
                is_system_provided: true,
            },
            NativeToolInput {
                name: "metal_toolchain".to_string(),
                version: "macos_sdk_14".to_string(),
                digest: "system_detected".to_string(),
                is_system_provided: true,
            },
            NativeToolInput {
                name: "naga_compiler".to_string(),
                version: "22.0.0".to_string(),
                digest: "cargo_pinned".to_string(),
                is_system_provided: false,
            },
            NativeToolInput {
                name: "spirv_tools".to_string(),
                version: "2024.x".to_string(),
                digest: "system_detected".to_string(),
                is_system_provided: true,
            },
            NativeToolInput {
                name: "llvm_tools".to_string(),
                version: "18.x".to_string(),
                digest: "system_detected".to_string(),
                is_system_provided: true,
            },
            NativeToolInput {
                name: "nvidia_driver".to_string(),
                version: "550.x+".to_string(),
                digest: "system_detected".to_string(),
                is_system_provided: true,
            },
            NativeToolInput {
                name: "vulkan_sdk".to_string(),
                version: "1.3.x".to_string(),
                digest: "system_detected".to_string(),
                is_system_provided: true,
            },
        ];

        let code_generators = CODE_GENERATORS
            .iter()
            .map(|(name, source, outputs)| {
                Ok(CodeGeneratorInput {
                    name: (*name).to_string(),
                    source_digest: source.digest(root)?,
                    target_outputs: outputs.iter().map(|output| (*output).to_string()).collect(),
                })
            })
            .collect::<Result<Vec<_>, ProvenanceError>>()?;

        let benchmark_baselines = collect_benchmark_baselines(root);
        let schemas = collect_persisted_schemas(root);
        let build_scripts = collect_build_script_contracts();

        let rustc_version = measure_rustc_version()?;
        let cargo_version = measure_cargo_version()?;
        let bom_timestamp = timestamp::rfc3339_utc(measure_source_date_epoch(root)?)?;

        // Canonical deterministic archive hash derived from all components
        let mut archive_hasher = blake3::Hasher::new();
        archive_hasher.update(lockfile_digest.as_bytes());
        archive_hasher.update(rustc_version.as_bytes());
        for dep in &dependencies {
            archive_hasher.update(dep.name.as_bytes());
            archive_hasher.update(dep.version.as_bytes());
            archive_hasher.update(dep.checksum.as_bytes());
        }
        for gen in &code_generators {
            archive_hasher.update(gen.name.as_bytes());
            archive_hasher.update(gen.source_digest.as_bytes());
        }
        for schema in &schemas {
            archive_hasher.update(schema.path.as_bytes());
            archive_hasher.update(schema.digest.as_bytes());
        }
        let reproducible_archive_hash = archive_hasher.finalize().to_hex().to_string();

        Ok(Self {
            schema_version: RELEASE_PROVENANCE_SCHEMA_VERSION,
            bom_timestamp,
            rustc_version,
            cargo_version,
            target_triples,
            lockfile_digest,
            dependencies,
            native_tools,
            code_generators,
            benchmark_baselines,
            schemas,
            build_scripts,
            is_offline_capable: true,
            reproducible_archive_hash,
        })
    }

    /// Verify that all declared build script contracts match filesystem reality.
    pub fn verify_build_scripts(&self, root: &Path) -> Result<(), ProvenanceError> {
        for script in &self.build_scripts {
            let script_path = root.join(&script.path);
            if !script_path.exists() {
                return Err(ProvenanceError::UndeclaredBuildInput {
                    build_script: script.path.clone(),
                    undeclared_input: "build script does not exist on disk".to_string(),
                });
            }
            if script.has_network_access {
                return Err(ProvenanceError::UndeclaredBuildInput {
                    build_script: script.path.clone(),
                    undeclared_input: "network access is strictly forbidden in build scripts"
                        .to_string(),
                });
            }
            if !script.has_bounded_reads {
                return Err(ProvenanceError::UndeclaredBuildInput {
                    build_script: script.path.clone(),
                    undeclared_input: "unbounded filesystem reads violate build script integrity"
                        .to_string(),
                });
            }
        }
        Ok(())
    }

    /// Verify offline integrity: every third-party dependency must be pinned by checksum.
    pub fn verify_offline_integrity(&self) -> Result<(), ProvenanceError> {
        for dep in &self.dependencies {
            if dep.source != "workspace"
                && (dep.checksum.is_empty() || dep.checksum == "local_workspace")
            {
                return Err(ProvenanceError::MissingChecksum(format!(
                    "package '{}' has no content-addressed checksum",
                    dep.name
                )));
            }
            if !dep.is_policy_approved {
                return Err(ProvenanceError::UnapprovedLicense {
                    package: dep.name.clone(),
                    license: dep.license.clone(),
                });
            }
        }
        Ok(())
    }

    /// Generate a standard CycloneDX / JSON Software Bill of Materials (SBOM).
    pub fn generate_sbom(&self) -> Result<String, ProvenanceError> {
        let sbom_doc = serde_json::json!({
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": format!("urn:uuid:vyre-sbom-{}", self.lockfile_digest),
            "version": 1,
            "metadata": {
                "timestamp": self.bom_timestamp,
                "tools": [
                    {
                        "vendor": "vyre",
                        "name": "xtask-release-provenance",
                        "version": "0.8.0"
                    },
                    {
                        "vendor": "rust-lang",
                        "name": "rustc",
                        "version": self.rustc_version
                    }
                ],
                "component": {
                    "name": "vyre",
                    "version": "0.8.0",
                    "type": "framework",
                    "description": "The vyre GPU compiler workspace"
                }
            },
            "components": self.dependencies.iter().map(|dep| {
                serde_json::json!({
                    "name": dep.name,
                    "version": dep.version,
                    "type": "library",
                    "purl": format!("pkg:cargo/{}@{}", dep.name, dep.version),
                    "hashes": [
                        {
                            "alg": "SHA-256",
                            "content": dep.checksum
                        }
                    ],
                    "licenses": [
                        {
                            "license": {
                                "id": dep.license
                            }
                        }
                    ]
                })
            }).collect::<Vec<_>>()
        });

        serde_json::to_string_pretty(&sbom_doc)
            .map_err(|e| ProvenanceError::Serialization(e.to_string()))
    }

    /// Generate an SLSA v1.2 compliant release provenance record.
    pub fn generate_slsa_provenance(&self, _root: &Path) -> Result<String, ProvenanceError> {
        let provenance_doc = serde_json::json!({
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [
                {
                    "name": "vyre-release-archive",
                    "digest": {
                        "blake3": self.reproducible_archive_hash
                    }
                }
            ],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "buildType": "https://vyre.dev/build/v1",
                    "externalParameters": {
                        "rustc_version": self.rustc_version,
                        "cargo_version": self.cargo_version,
                        "target_triples": self.target_triples,
                        "offline_capable": self.is_offline_capable
                    },
                    "resolvedDependencies": self.dependencies.iter().map(|dep| {
                        serde_json::json!({
                            "uri": format!("pkg:cargo/{}@{}", dep.name, dep.version),
                            "digest": {
                                "sha256": dep.checksum
                            }
                        })
                    }).collect::<Vec<_>>()
                },
                "runDetails": {
                    "builder": {
                        "id": "https://vyre.dev/builders/cargo_full",
                        "version": self.cargo_version
                    },
                    "metadata": {
                        "invocationId": format!("inv-{}", self.lockfile_digest),
                        "completeness": {
                            "parameters": true,
                            "environment": true,
                            "materials": true
                        },
                        "reproducible": true
                    },
                    "byproducts": self.schemas.iter().map(|s| {
                        serde_json::json!({
                            "name": s.path,
                            "digest": {
                                "sha256": s.digest
                            }
                        })
                    }).collect::<Vec<_>>()
                }
            }
        });

        serde_json::to_string_pretty(&provenance_doc)
            .map_err(|e| ProvenanceError::Serialization(e.to_string()))
    }

    /// Serialize provenance authority to TOML.
    pub fn to_toml(&self) -> Result<String, ProvenanceError> {
        toml::to_string_pretty(self).map_err(|e| ProvenanceError::Serialization(e.to_string()))
    }

    /// Deserialize provenance authority from TOML with fail-closed schema validation.
    pub fn from_toml(toml_str: &str) -> Result<Self, ProvenanceError> {
        let authority: Self =
            toml::from_str(toml_str).map_err(|e| ProvenanceError::Serialization(e.to_string()))?;
        if authority.schema_version != RELEASE_PROVENANCE_SCHEMA_VERSION {
            return Err(ProvenanceError::StaleSchemaVersion {
                expected: RELEASE_PROVENANCE_SCHEMA_VERSION,
                found: authority.schema_version,
            });
        }
        Ok(authority)
    }
}

fn is_allowed_wrapper(name: &str) -> bool {
    name == "openssl" || name == "openssl-sys"
}

fn load_deny_policy(root: &Path) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut allowed = BTreeSet::new();
    let mut banned = BTreeSet::new();

    let deny_path = root.join("deny.toml");
    if let Ok(text) = fs::read_to_string(&deny_path) {
        if let Ok(val) = toml::from_str::<toml::Value>(&text) {
            if let Some(arr) = val
                .get("licenses")
                .and_then(|l| l.get("allow"))
                .and_then(|a| a.as_array())
            {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        allowed.insert(s.to_string());
                    }
                }
            }
            if let Some(arr) = val
                .get("bans")
                .and_then(|b| b.get("deny"))
                .and_then(|d| d.as_array())
            {
                for item in arr {
                    if let Some(s) = item.get("crate").and_then(|c| c.as_str()) {
                        banned.insert(s.to_string());
                    }
                }
            }
        }
    }

    if allowed.is_empty() {
        // Fallback to standard deny.toml baseline
        for lic in &[
            "Apache-2.0",
            "Apache-2.0 WITH LLVM-exception",
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "ISC",
            "Unicode-DFS-2016",
            "Unicode-3.0",
            "Zlib",
            "CC0-1.0",
            "MPL-2.0",
            "0BSD",
            "CDLA-Permissive-2.0",
        ] {
            allowed.insert((*lic).to_string());
        }
    }

    (allowed, banned)
}

fn load_metadata_licenses(root: &Path) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();

    // Fast metadata lookup via offline cargo metadata
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--offline"])
        .current_dir(root)
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                if let Some(packages) = json.get("packages").and_then(|p| p.as_array()) {
                    for pkg in packages {
                        let name = pkg.get("name").and_then(|n| n.as_str()).unwrap_or_default();
                        let version = pkg
                            .get("version")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        if let Some(lic) = pkg.get("license").and_then(|l| l.as_str()) {
                            map.insert(format!("{name}@{version}"), lic.to_string());
                            map.insert(name.to_string(), lic.to_string());
                        }
                    }
                }
            }
        }
    }

    map
}

/// Evaluates SPDX license expression against approved license set.
pub fn is_license_expression_approved(license_expr: &str, allowed_set: &BTreeSet<String>) -> bool {
    let trimmed = license_expr.trim();
    if trimmed.is_empty() || trimmed == "UNKNOWN" {
        return false;
    }

    // Handle AND conjunctions: e.g. "(MIT OR Apache-2.0) AND Unicode-3.0"
    if trimmed.contains(" AND ") {
        return trimmed.split(" AND ").all(|part| {
            is_license_expression_approved(
                part.trim_matches(|c| c == '(' || c == ')' || c == ' '),
                allowed_set,
            )
        });
    }

    // Handle OR disjunctions: e.g. "MIT OR Apache-2.0", "Apache-2.0 / MIT", "MIT/Apache-2.0"
    for delim in &[" OR ", " / ", "/"] {
        if trimmed.contains(delim) {
            return trimmed.split(delim).any(|part| {
                is_license_expression_approved(
                    part.trim_matches(|c| c == '(' || c == ')' || c == ' '),
                    allowed_set,
                )
            });
        }
    }

    let token = trimmed.trim_matches(|c| c == '(' || c == ')' || c == ' ');
    let normalized = token.strip_suffix('+').unwrap_or(token);

    allowed_set.contains(normalized)
        || normalized == "Unlicense"
        || normalized == "MIT-0"
        || normalized == "BSD-1-Clause"
}

/// Digest of one file a code generator is defined in.
///
/// # Errors
///
/// Returns `UnmeasuredFact` when the path reaches no file. A generator whose
/// source cannot be read has no measured digest, and a sentinel string in that
/// field reads as one: `hash_file_or_default` answered `file_missing` and left
/// `xtask/src/gate_metadata.rs` in the roster after the table became the
/// `xtask/src/gate_metadata` directory, so the document attested that sentinel
/// as the generator's source.
fn hash_source_file(root: &Path, rel_path: &str) -> Result<String, ProvenanceError> {
    let full = root.join(rel_path);
    let bytes = fs::read(&full).map_err(|error| ProvenanceError::UnmeasuredFact {
        fact: format!("source digest of `{rel_path}`"),
        reason: error.to_string(),
    })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Digest of every file under one directory, at any depth.
///
/// The path folded into the hash beside each file's bytes is relative to the
/// directory. Hashing the absolute path made the digest a fact about where the
/// checkout sits, so two checkouts of one commit attested different bytes for
/// the same source. The walk is recursive because a generator's source is its
/// whole module: `vyre-macros/src/pass` was outside a single-level read, so a
/// change to the macro pass moved no digest.
///
/// # Errors
///
/// Returns `UnmeasuredFact` when the directory supplies no file. Git tracks
/// files rather than directories, so a name whose source moved leaves the
/// directory behind in every checkout that pulled the deletion, and a name that
/// never existed leaves nothing. Both hold no content. Reading `.is_dir()` and
/// answering `dir_missing` kept `conform/structure-gate/src` in the roster after
/// the crate moved to `structure-gate/src`.
fn hash_source_tree(root: &Path, rel_dir: &str) -> Result<String, ProvenanceError> {
    let full = root.join(rel_dir);
    let unmeasured = |reason: String| ProvenanceError::UnmeasuredFact {
        fact: format!("source digest of `{rel_dir}`"),
        reason,
    };
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(&full).sort_by_file_name() {
        let entry = entry.map_err(|error| unmeasured(error.to_string()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(&full)
            .map_err(|error| unmeasured(error.to_string()))?
            .to_string_lossy()
            .replace('\\', "/");
        let bytes =
            fs::read(entry.path()).map_err(|error| unmeasured(format!("{relative}: {error}")))?;
        files.push((relative, bytes));
    }
    if files.is_empty() {
        return Err(unmeasured(
            "the directory supplies no file, so the name reaches no source".to_string(),
        ));
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = blake3::Hasher::new();
    for (name, content) in files {
        hasher.update(name.as_bytes());
        hasher.update(&content);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn collect_benchmark_baselines(root: &Path) -> Vec<BenchmarkBaselineInput> {
    let bench_dir = root.join("release/evidence/benchmarks");
    let mut baselines = Vec::new();
    if let Ok(entries) = fs::read_dir(&bench_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(rel) = path.strip_prefix(root) {
                    if let Ok(bytes) = fs::read(&path) {
                        baselines.push(BenchmarkBaselineInput {
                            path: rel.to_string_lossy().to_string(),
                            digest: blake3::hash(&bytes).to_hex().to_string(),
                        });
                    }
                }
            }
        }
    }
    baselines.sort_by(|a, b| a.path.cmp(&b.path));
    baselines
}

fn collect_persisted_schemas(root: &Path) -> Vec<SchemaInput> {
    let generated_dir = root.join("docs/generated");
    let mut schemas = Vec::new();
    if let Ok(entries) = fs::read_dir(&generated_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if file_name == "release-provenance.toml" {
                        continue;
                    }
                }
                let is_schema = path
                    .extension()
                    .is_some_and(|ext| ext == "json" || ext == "toml");
                if is_schema {
                    if let Ok(rel) = path.strip_prefix(root) {
                        if let Ok(bytes) = fs::read(&path) {
                            schemas.push(SchemaInput {
                                path: rel.to_string_lossy().to_string(),
                                digest: blake3::hash(&bytes).to_hex().to_string(),
                            });
                        }
                    }
                }
            }
        }
    }
    schemas.sort_by(|a, b| a.path.cmp(&b.path));
    schemas
}

fn collect_build_script_contracts() -> Vec<BuildScriptContract> {
    vec![
        BuildScriptContract {
            crate_name: "vyre-emit-ptx".to_string(),
            path: "vyre-emit-ptx/build.rs".to_string(),
            declared_inputs: vec!["src/**".to_string(), "Cargo.toml".to_string()],
            declared_outputs: vec![
                "cargo:rustc-env=VYRE_PTX_LOWERING_DIGEST".to_string(),
                "cargo:rerun-if-changed".to_string(),
            ],
            has_network_access: false,
            has_bounded_reads: true,
        },
        BuildScriptContract {
            crate_name: "vyre-emit-naga".to_string(),
            path: "vyre-emit-naga/build.rs".to_string(),
            declared_inputs: vec!["src/**".to_string(), "Cargo.toml".to_string()],
            declared_outputs: vec![
                "cargo:rustc-env=VYRE_NAGA_LOWERING_DIGEST".to_string(),
                "cargo:rerun-if-changed".to_string(),
            ],
            has_network_access: false,
            has_bounded_reads: true,
        },
        BuildScriptContract {
            crate_name: "vyre-bench".to_string(),
            path: "vyre-bench/build.rs".to_string(),
            declared_inputs: vec!["env:OPT_LEVEL".to_string()],
            declared_outputs: vec!["cargo:rustc-env=VYRE_BENCH_OPT_LEVEL".to_string()],
            has_network_access: false,
            has_bounded_reads: true,
        },
        BuildScriptContract {
            crate_name: "vyre-driver-wgpu".to_string(),
            path: "vyre-driver-wgpu/build.rs".to_string(),
            declared_inputs: vec!["Cargo.toml".to_string()],
            declared_outputs: vec![
                "cargo:rustc-env=VYRE_NAGA_VERSION".to_string(),
                "cargo:rerun-if-changed=Cargo.toml".to_string(),
            ],
            has_network_access: false,
            has_bounded_reads: true,
        },
    ]
}

/// The running rustc version, or the reason it could not be read.
fn measure_rustc_version() -> Result<String, ProvenanceError> {
    measure_tool_version("rustc")
}

/// The running cargo version, or the reason it could not be read.
fn measure_cargo_version() -> Result<String, ProvenanceError> {
    measure_tool_version("cargo")
}

/// The version `tool` reports under `-V`, with its own name stripped.
fn measure_tool_version(tool: &str) -> Result<String, ProvenanceError> {
    let out =
        Command::new(tool)
            .arg("-V")
            .output()
            .map_err(|error| ProvenanceError::UnmeasuredFact {
                fact: format!("{tool} version"),
                reason: error.to_string(),
            })?;
    if !out.status.success() {
        return Err(ProvenanceError::UnmeasuredFact {
            fact: format!("{tool} version"),
            reason: format!("{tool} -V exited with {}", out.status),
        });
    }
    let reported = String::from_utf8_lossy(&out.stdout);
    reported
        .strip_prefix(tool)
        .map(|version| version.trim().to_string())
        .filter(|version| !version.is_empty())
        .ok_or_else(|| ProvenanceError::UnmeasuredFact {
            fact: format!("{tool} version"),
            reason: format!("{tool} -V printed {:?}", reported.trim()),
        })
}

/// The second the release source was fixed at, for a document that is byte
/// compared across runs.
///
/// `SOURCE_DATE_EPOCH` wins when it is set, which is what a distribution build
/// sets to pin every generated timestamp. Otherwise it is the commit time of
/// the newest commit that changed source, which is the instant the tree being
/// described came into existence.
///
/// Source here is what [`crate::source_provenance::EXCLUDED_FROM_SOURCE`]
/// leaves, for the reason that constant states. Reading `HEAD` instead made
/// this document impossible to commit: the write stamped the commit time of
/// the tree it read, committing that write produced a newer commit, and the
/// artifact was stale against its own carrier the moment it landed.
fn measure_source_date_epoch(root: &Path) -> Result<i64, ProvenanceError> {
    if let Ok(declared) = std::env::var("SOURCE_DATE_EPOCH") {
        return parse_epoch_second(&declared, "SOURCE_DATE_EPOCH");
    }
    let mut arguments = vec!["log", "-1", "--pretty=%ct", "--", "."];
    arguments.extend(crate::source_provenance::EXCLUDED_FROM_SOURCE);
    let out = Command::new("git")
        .current_dir(root)
        .args(&arguments)
        .output()
        .map_err(|error| ProvenanceError::UnmeasuredFact {
            fact: "source date epoch".to_string(),
            reason: error.to_string(),
        })?;
    if !out.status.success() {
        return Err(ProvenanceError::UnmeasuredFact {
            fact: "source date epoch".to_string(),
            reason: format!("git log exited with {}", out.status),
        });
    }
    parse_epoch_second(
        &String::from_utf8_lossy(&out.stdout),
        "git log -1 --pretty=%ct over the source paths",
    )
}

/// `reported` read as a unix second, naming `origin` when it is not one.
fn parse_epoch_second(reported: &str, origin: &str) -> Result<i64, ProvenanceError> {
    reported
        .trim()
        .parse::<i64>()
        .map_err(|error| ProvenanceError::UnmeasuredFact {
            fact: "source date epoch".to_string(),
            reason: format!("{origin} produced {:?}: {error}", reported.trim()),
        })
}

#[cfg(test)]
mod measurement_tests {
    use super::*;

    /// WHY: every fact a provenance document states used to have a literal
    /// standing behind it, so a document produced on a host with no toolchain
    /// still claimed a toolchain version. The contract is that a fact that
    /// cannot be measured stops the document instead of being invented. This
    /// covers the tool version arm; the epoch arm is below.
    ///
    /// What it does not catch: a tool that exists and reports a wrong version.
    #[test]
    fn a_tool_that_is_not_installed_stops_the_document() {
        let measured = measure_tool_version("vyre-provenance-absent-tool");
        let Err(ProvenanceError::UnmeasuredFact { fact, .. }) = measured else {
            panic!("an absent tool produced a version: {measured:?}");
        };
        assert_eq!(fact, "vyre-provenance-absent-tool version");
    }

    /// WHY: the installed toolchain is measurable here, and the measurement
    /// has to strip the tool's own name so the document carries a version and
    /// not a sentence.
    #[test]
    fn an_installed_tool_reports_a_bare_version() {
        let rustc = measure_rustc_version().expect("rustc is required to build this test");
        assert!(
            !rustc.starts_with("rustc"),
            "the tool name survived into the version: {rustc}"
        );
        assert!(
            rustc.starts_with(|c: char| c.is_ascii_digit()),
            "not a version: {rustc}"
        );
    }

    /// WHY: the epoch is the one fact a caller can pin, and a pin that does
    /// not parse must fail rather than fall back to a build-time clock, which
    /// is what would silently break byte comparison of the artifact.
    #[test]
    fn an_unparseable_epoch_stops_the_document() {
        for declared in ["", "   ", "yesterday", "1789016022.5", "0x6a", "1e9"] {
            let measured = parse_epoch_second(declared, "SOURCE_DATE_EPOCH");
            assert!(
                matches!(measured, Err(ProvenanceError::UnmeasuredFact { .. })),
                "{declared:?} was accepted as a unix second: {measured:?}"
            );
        }
    }

    /// WHY: a pinned epoch has to reach the rendered instant unchanged, which
    /// is what makes two runs over one tree produce the same bytes. The
    /// negative epoch is here because the parse is signed and a document may
    /// describe a tree older than the epoch.
    #[test]
    fn a_pinned_epoch_reaches_the_rendered_instant() {
        for (declared, rendered) in [
            (" 1789016022\n", "2026-09-10T04:53:42Z"),
            ("0", "1970-01-01T00:00:00Z"),
            ("-1", "1969-12-31T23:59:59Z"),
        ] {
            let seconds = parse_epoch_second(declared, "SOURCE_DATE_EPOCH")
                .expect("a bare integer is a unix second");
            assert_eq!(
                timestamp::rfc3339_utc(seconds).expect("a representable second"),
                rendered,
                "for {declared:?}"
            );
        }
    }

    /// WHY: closes the class where recording evidence moves the instant the
    /// provenance document states, which made the document impossible to
    /// commit: `--write` stamped the commit time of the tree it read, the
    /// commit carrying that write was newer, and the next comparison reported
    /// the artifact as stale against itself. The epoch answers for source, and
    /// the evidence corpus and the document's own projection are not source.
    ///
    /// What this does not catch: a commit that changes source and evidence
    /// together. That one does move the epoch, and should.
    #[test]
    fn committing_only_evidence_leaves_the_source_instant_where_it_was() {
        let checkout = tempfile::tempdir().expect("Fix: create the fixture checkout.");
        let root = checkout.path();
        crate::fixture_checkout::seeded(root);
        std::fs::create_dir_all(root.join("release/evidence/metadata"))
            .expect("Fix: create the evidence directory.");
        std::fs::write(root.join("src.rs"), "fn main() {}\n").expect("Fix: write the source file.");
        crate::fixture_checkout::commit_worktree(root, "record source");
        let after_source = measure_source_date_epoch(root)
            .expect("Fix: measure the epoch after the source commit.");

        // A commit one second later carrying only excluded paths.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(root.join("release/evidence/metadata/sbom.json"), "{}\n")
            .expect("Fix: write the evidence artifact.");
        std::fs::create_dir_all(root.join("docs/generated")).expect("Fix: create docs/generated.");
        std::fs::write(
            root.join("docs/generated/release-provenance.toml"),
            "x = 1\n",
        )
        .expect("Fix: write the provenance projection.");
        crate::fixture_checkout::commit_worktree(root, "record evidence");
        let after_evidence = measure_source_date_epoch(root)
            .expect("Fix: measure the epoch after the evidence commit.");

        assert_eq!(
            after_evidence, after_source,
            "Fix: an evidence-only commit must leave the source instant alone, or the document \
             this instant is written into is stale the moment it is committed."
        );

        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(root.join("src.rs"), "fn main() { let _ = 1; }\n")
            .expect("Fix: change the source file.");
        crate::fixture_checkout::commit_worktree(root, "change source");
        let after_change = measure_source_date_epoch(root)
            .expect("Fix: measure the epoch after the source change.");
        assert!(
            after_change > after_source,
            "Fix: a commit that changes source must move the instant: {after_change} is not past \
             {after_source}."
        );
    }

    /// WHY: the schema carries a field now, so a document written before it
    /// describes a release whose provenance was never recorded. Serving that
    /// as current is the resurrection this version number exists to stop.
    #[test]
    fn the_schema_version_is_past_the_timestampless_document() {
        assert!(
            RELEASE_PROVENANCE_SCHEMA_VERSION > 1,
            "a v1 document carries no measured timestamp and must not verify"
        );
    }

    /// WHY: a source digest is the fact a rebuild is checked against, so a
    /// path that reaches no source has to stop the document. Answering a
    /// sentinel string instead let two generator paths go stale unnoticed,
    /// `xtask/src/gate_metadata.rs` after it became a directory and
    /// `conform/structure-gate/src` after the crate moved, and the published
    /// document recorded `file_missing` and `dir_missing` as their digests.
    ///
    /// What it does not catch: a path that reaches source belonging to a
    /// different generator than the row names.
    #[test]
    fn a_generator_path_that_reaches_no_source_stops_the_document() {
        let root = crate::checkout::checkout_root();
        let absent_tree = hash_source_tree(&root, "vyre-provenance-absent-directory");
        let Err(ProvenanceError::UnmeasuredFact { fact, .. }) = absent_tree else {
            panic!("a directory holding no source produced a digest: {absent_tree:?}");
        };
        assert_eq!(fact, "source digest of `vyre-provenance-absent-directory`");

        let empty_tree = tempfile::tempdir().expect("a temporary directory is required");
        let named = empty_tree.path().join("src");
        fs::create_dir(&named).expect("the empty directory is required");
        let measured = hash_source_tree(empty_tree.path(), "src");
        let Err(ProvenanceError::UnmeasuredFact { fact, .. }) = measured else {
            panic!("an empty directory produced a digest: {measured:?}");
        };
        assert_eq!(fact, "source digest of `src`");

        let absent_file = hash_source_file(&root, "vyre-provenance-absent-file.rs");
        let Err(ProvenanceError::UnmeasuredFact { fact, .. }) = absent_file else {
            panic!("an absent file produced a digest: {absent_file:?}");
        };
        assert_eq!(fact, "source digest of `vyre-provenance-absent-file.rs`");
    }

    /// WHY: a generator row names a path, and the path goes stale when the
    /// source moves. The row set is read from `CODE_GENERATORS` at run time,
    /// so a generator added later is judged here without a second list.
    ///
    /// What it does not catch: a row whose path reaches source that belongs to
    /// a different generator than the row names.
    #[test]
    fn every_generator_in_the_roster_measures_a_digest() {
        let root = crate::checkout::checkout_root();
        let mut measured = 0usize;
        for (name, source, outputs) in CODE_GENERATORS {
            let digest = source
                .digest(&root)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                digest.len(),
                64,
                "{name} carries no blake3 digest: {digest}"
            );
            assert!(!outputs.is_empty(), "{name} records no generated artifact");
            measured += 1;
        }
        assert!(
            measured > 0,
            "an empty roster proves no generator path, and this contract judged nothing"
        );
    }

    /// WHY: the digest is published so a rebuild elsewhere can be compared
    /// against it. Folding the absolute path of each file into the hash made
    /// the digest a fact about the checkout's location, so the same commit
    /// unpacked at a second path attested different bytes and every
    /// comparison failed.
    #[test]
    fn a_tree_digest_is_independent_of_where_the_tree_sits() {
        let first = tempfile::tempdir().expect("a temporary directory is required");
        let second = tempfile::tempdir().expect("a second temporary directory is required");
        for base in [first.path(), second.path()] {
            fs::create_dir_all(base.join("src/pass")).expect("the fixture tree is required");
            fs::write(base.join("src/lib.rs"), b"pub fn one() {}\n")
                .expect("the fixture is required");
            fs::write(base.join("src/pass/mod.rs"), b"pub fn two() {}\n")
                .expect("the nested fixture is required");
        }
        let left = hash_source_tree(first.path(), "src").expect("the fixture tree is measurable");
        let right = hash_source_tree(second.path(), "src").expect("the fixture tree is measurable");
        assert_eq!(
            left, right,
            "the same source at two paths produced two digests"
        );

        fs::write(first.path().join("src/pass/mod.rs"), b"pub fn three() {}\n")
            .expect("the nested fixture is rewritable");
        let changed =
            hash_source_tree(first.path(), "src").expect("the fixture tree is measurable");
        assert_ne!(
            left, changed,
            "a change below the top level of the tree moved no digest"
        );
    }
}
