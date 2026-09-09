//! One typed configuration schema partitioned by semantic compile input, operational policy, diagnostics, and credentials.
//!
//! WHY: closes the class "scattered environment variable reads and uncoordinated CLI flags
//! create contradictory runtime policies or bypass artifact cache identity".
//! Establishes strict resolution precedence: Explicit -> CLI -> TOML -> Env injection points.
//! Behavior-affecting options contribute to `behavior_hash`; secrets are redacted.

use core::fmt;
use std::collections::BTreeMap;
use std::format;
use std::string::String;

/// Top-level partition for configuration fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ConfigPartition {
    /// Semantic compile inputs that directly affect IR, optimization, or target code.
    SemanticCompileInput,
    /// Operational policies (memory limits, concurrency, queue depths, timeouts).
    OperationalPolicy,
    /// Diagnostic, telemetry, tracing, and logging controls.
    DiagnosticControls,
    /// Deployment credentials, tokens, and authentication secrets.
    Credentials,
}

impl ConfigPartition {
    /// All partitions.
    pub const ALL: &'static [Self] = &[
        Self::SemanticCompileInput,
        Self::OperationalPolicy,
        Self::DiagnosticControls,
        Self::Credentials,
    ];

    /// Stable string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SemanticCompileInput => "semantic_compile_input",
            Self::OperationalPolicy => "operational_policy",
            Self::DiagnosticControls => "diagnostic_controls",
            Self::Credentials => "credentials",
        }
    }
}

impl fmt::Display for ConfigPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Field data type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigType {
    /// Boolean flag.
    Bool,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// 64-bit float.
    F64,
    /// String value.
    String,
    /// Filesystem path string.
    Path,
}

/// Identity and caching impact of a configuration field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityImpact {
    /// Affects compile IR and target artifact identity.
    AffectsCompileIdentity,
    /// Affects cache lookup and reuse.
    AffectsCacheIdentity,
    /// Operational policy only; does not change emitted artifact bytes.
    OperationalOnly,
}

/// Secrecy classification of a configuration field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigSecrecy {
    /// Public configuration field, visible in receipts, logs, and diagnostics.
    Public,
    /// Sensitive secret/credential, strictly redacted in inspection, receipts, and logs.
    Secret,
}

/// Mutability lifecycle of a configuration field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigMutability {
    /// Immutable once resolved for the process.
    ImmutableAtRuntime,
    /// Dynamic between requests/sessions.
    MutableBetweenRequests,
}

/// Dynamic value for configuration fields.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigValue {
    /// Boolean.
    Bool(bool),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// 64-bit float.
    F64(f64),
    /// String.
    String(String),
}

impl ConfigValue {
    /// Render value as string with secrecy redaction if needed.
    #[must_use]
    pub fn display_redacted(&self, secrecy: ConfigSecrecy) -> String {
        if secrecy == ConfigSecrecy::Secret {
            return String::from("[REDACTED]");
        }
        match self {
            Self::Bool(b) => format!("{b}"),
            Self::U32(v) => format!("{v}"),
            Self::U64(v) => format!("{v}"),
            Self::F64(v) => format!("{v}"),
            Self::String(s) => s.clone(),
        }
    }
}

/// Declarative schema entry for one configuration option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigFieldDef {
    /// Unique dot-separated key (e.g. "compile.opt_level", "runtime.max_queue_depth").
    pub key: &'static str,
    /// Owning partition.
    pub partition: ConfigPartition,
    /// Data type.
    pub field_type: ConfigType,
    /// Owner crate or subsystem name.
    pub owner: &'static str,
    /// Default value as static string.
    pub default_raw: &'static str,
    /// Human-readable bounds description.
    pub bounds_desc: &'static str,
    /// Impact on identity and caching.
    pub identity_impact: IdentityImpact,
    /// Secrecy classification.
    pub secrecy: ConfigSecrecy,
    /// Mutability classification.
    pub mutability: ConfigMutability,
    /// Whether changing this field requires full process restart.
    pub requires_restart: bool,
    /// Documentation help text.
    pub help_text: &'static str,
}

/// The complete declarative configuration registry.
pub const CANONICAL_CONFIG_FIELDS: &[ConfigFieldDef] = &[
    // Semantic compile inputs
    ConfigFieldDef {
        key: "compile.opt_level",
        partition: ConfigPartition::SemanticCompileInput,
        field_type: ConfigType::U32,
        owner: "vyre-foundation",
        default_raw: "2",
        bounds_desc: "0..=3",
        identity_impact: IdentityImpact::AffectsCompileIdentity,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::ImmutableAtRuntime,
        requires_restart: false,
        help_text: "Compiler optimization level (0=none, 1=basic, 2=standard, 3=aggressive).",
    },
    ConfigFieldDef {
        key: "compile.target_backend",
        partition: ConfigPartition::SemanticCompileInput,
        field_type: ConfigType::String,
        owner: "vyre-foundation",
        default_raw: "auto",
        bounds_desc: "auto|<registered backend id>",
        identity_impact: IdentityImpact::AffectsCompileIdentity,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::ImmutableAtRuntime,
        requires_restart: false,
        help_text: "Target GPU backend for lowering and code generation.",
    },
    ConfigFieldDef {
        key: "compile.enable_fma",
        partition: ConfigPartition::SemanticCompileInput,
        field_type: ConfigType::Bool,
        owner: "vyre-foundation",
        default_raw: "true",
        bounds_desc: "true|false",
        identity_impact: IdentityImpact::AffectsCompileIdentity,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::ImmutableAtRuntime,
        requires_restart: false,
        help_text: "Enable fused multiply-add synthesis in the optimizer.",
    },
    // Operational policies
    ConfigFieldDef {
        key: "runtime.max_queue_depth",
        partition: ConfigPartition::OperationalPolicy,
        field_type: ConfigType::U32,
        owner: "vyre-runtime",
        default_raw: "1024",
        bounds_desc: "1..=65536",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::MutableBetweenRequests,
        requires_restart: false,
        help_text: "Maximum capacity of the resident submission work queue.",
    },
    ConfigFieldDef {
        key: "runtime.execution_timeout_ms",
        partition: ConfigPartition::OperationalPolicy,
        field_type: ConfigType::U64,
        owner: "vyre-runtime",
        default_raw: "5000",
        bounds_desc: "100..=3600000",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::MutableBetweenRequests,
        requires_restart: false,
        help_text: "Hard execution timeout in milliseconds before triggering device reset.",
    },
    ConfigFieldDef {
        key: "runtime.cache_dir",
        partition: ConfigPartition::OperationalPolicy,
        field_type: ConfigType::Path,
        owner: "vyre-runtime",
        default_raw: ".cache/vyre",
        bounds_desc: "valid directory path",
        identity_impact: IdentityImpact::AffectsCacheIdentity,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::ImmutableAtRuntime,
        requires_restart: true,
        help_text: "Persistent disk cache directory for compiled megakernels.",
    },
    // Diagnostic controls
    ConfigFieldDef {
        key: "diag.trace_level",
        partition: ConfigPartition::DiagnosticControls,
        field_type: ConfigType::String,
        owner: "vyre-foundation",
        default_raw: "info",
        bounds_desc: "off|error|warn|info|debug|trace",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::MutableBetweenRequests,
        requires_restart: false,
        help_text: "Causal diagnostic and execution event tracing verbosity level.",
    },
    ConfigFieldDef {
        key: "diag.perf_sampling_rate",
        partition: ConfigPartition::DiagnosticControls,
        field_type: ConfigType::F64,
        owner: "vyre-bench",
        default_raw: "0.01",
        bounds_desc: "0.0..=1.0",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::MutableBetweenRequests,
        requires_restart: false,
        help_text: "Fraction of dispatches sampled for fine-grained GPU timing.",
    },
    // Credentials
    ConfigFieldDef {
        key: "credentials.fleet_auth_token",
        partition: ConfigPartition::Credentials,
        field_type: ConfigType::String,
        owner: "vyre-runtime",
        default_raw: "",
        bounds_desc: "hex/base64 string or empty",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Secret,
        mutability: ConfigMutability::ImmutableAtRuntime,
        requires_restart: true,
        help_text: "Secret authentication bearer token for multi-tenant cluster admission.",
    },
];

/// Configuration resolution layer source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfigLayer {
    /// Process-level explicit library code assignment (highest precedence).
    ExplicitLibrary = 0,
    /// CLI argument override (second highest).
    CliOverride = 1,
    /// Versioned TOML configuration file (third).
    TomlFile = 2,
    /// Process environment variable injection point (lowest; only for named credentials/deployment).
    EnvironmentInjection = 3,
    /// Compiled default fallback.
    DefaultFallback = 4,
}

/// Resolved effective configuration record.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedConfiguration {
    values: BTreeMap<String, (ConfigValue, ConfigLayer)>,
}

impl ResolvedConfiguration {
    /// Initialize with defaults.
    #[must_use]
    pub fn new_with_defaults() -> Self {
        let mut values = BTreeMap::new();
        for def in CANONICAL_CONFIG_FIELDS {
            let val = match def.field_type {
                ConfigType::Bool => ConfigValue::Bool(def.default_raw == "true"),
                ConfigType::U32 => ConfigValue::U32(def.default_raw.parse().unwrap_or(0)),
                ConfigType::U64 => ConfigValue::U64(def.default_raw.parse().unwrap_or(0)),
                ConfigType::F64 => ConfigValue::F64(def.default_raw.parse().unwrap_or(0.0)),
                ConfigType::String | ConfigType::Path => {
                    ConfigValue::String(String::from(def.default_raw))
                }
            };
            values.insert(String::from(def.key), (val, ConfigLayer::DefaultFallback));
        }
        Self { values }
    }

    /// Apply an override with explicit layer precedence.
    ///
    /// # Errors
    ///
    /// Returns error if key is unknown or type is invalid.
    pub fn apply_override(
        &mut self,
        key: &str,
        value: ConfigValue,
        layer: ConfigLayer,
    ) -> Result<(), String> {
        let def = CANONICAL_CONFIG_FIELDS
            .iter()
            .find(|d| d.key == key)
            .ok_or_else(|| {
                format!(
                    "Fix: unknown configuration key '{key}'. Run with --help to view registered configuration schema keys."
                )
            })?;

        // Type validation
        let type_valid = matches!(
            (&value, def.field_type),
            (ConfigValue::Bool(_), ConfigType::Bool)
                | (ConfigValue::U32(_), ConfigType::U32)
                | (ConfigValue::U64(_), ConfigType::U64)
                | (ConfigValue::F64(_), ConfigType::F64)
                | (ConfigValue::String(_), ConfigType::String | ConfigType::Path)
        );
        if !type_valid {
            return Err(format!(
                "Fix: type mismatch for configuration key '{key}'; expected {:?}",
                def.field_type
            ));
        }

        // Apply only if new layer has higher or equal precedence (lower numeric enum value)
        if let Some((_, current_layer)) = self.values.get(key) {
            if layer > *current_layer {
                return Ok(()); // Lower precedence layer cannot overwrite higher precedence
            }
        }

        self.values.insert(String::from(key), (value, layer));
        Ok(())
    }

    /// Retrieve a configuration value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.values.get(key).map(|(v, _)| v)
    }

    /// Return redacted view mapping key to formatted string.
    #[must_use]
    pub fn redacted_view(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for (k, (v, _)) in &self.values {
            let secrecy = CANONICAL_CONFIG_FIELDS
                .iter()
                .find(|d| d.key == k.as_str())
                .map_or(ConfigSecrecy::Public, |d| d.secrecy);
            out.insert(k.clone(), v.display_redacted(secrecy));
        }
        out
    }

    /// Compute 32-byte cryptographic behavior hash over all identity/cache-affecting fields.
    #[must_use]
    pub fn behavior_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"VYRE_CONFIG_BEHAVIOR_V1");

        for def in CANONICAL_CONFIG_FIELDS {
            if def.identity_impact != IdentityImpact::OperationalOnly {
                if let Some((val, _)) = self.values.get(def.key) {
                    hasher.update(def.key.as_bytes());
                    match val {
                        ConfigValue::Bool(b) => hasher.update(&[if *b { 1 } else { 0 }]),
                        ConfigValue::U32(v) => hasher.update(&v.to_le_bytes()),
                        ConfigValue::U64(v) => hasher.update(&v.to_le_bytes()),
                        ConfigValue::F64(v) => hasher.update(&v.to_le_bytes()),
                        ConfigValue::String(s) => hasher.update(s.as_bytes()),
                    };
                }
            }
        }

        *hasher.finalize().as_bytes()
    }
}
