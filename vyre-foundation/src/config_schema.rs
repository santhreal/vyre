//! One typed configuration schema partitioned by semantic compile input, operational policy, diagnostics, and credentials.
//!
//! WHY: closes the class "scattered environment variable reads and uncoordinated CLI flags
//! create contradictory runtime policies or bypass artifact cache identity".
//! Establishes strict resolution precedence: Explicit -> CLI -> TOML -> Env injection points -> Defaults.
//! Behavior-affecting options contribute to `behavior_hash`; secrets are redacted.

use core::fmt;
use std::collections::BTreeMap;
use std::format;
use std::string::String;
use std::vec::Vec;

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

    /// Render unredacted raw value as string.
    #[must_use]
    pub fn display_raw(&self) -> String {
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

impl ConfigFieldDef {
    /// Validate a value against this field's type and bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if the value has the wrong type or violates bounds.
    pub fn validate_value(&self, value: &ConfigValue) -> Result<(), String> {
        match (value, self.field_type) {
            (ConfigValue::Bool(_), ConfigType::Bool) => Ok(()),
            (ConfigValue::U32(v), ConfigType::U32) => {
                if self.key == "compile.opt_level" && *v > 3 {
                    return Err(format!(
                        "Fix: configuration key '{}' value '{v}' violates bounds: {}",
                        self.key, self.bounds_desc
                    ));
                }
                if self.key == "compile.planar_rewrite_batch_threshold" && (*v == 0 || *v > 1024) {
                    return Err(format!(
                        "Fix: configuration key '{}' value '{v}' violates bounds: {}",
                        self.key, self.bounds_desc
                    ));
                }
                if self.key == "runtime.max_queue_depth" && (*v == 0 || *v > 65536) {
                    return Err(format!(
                        "Fix: configuration key '{}' value '{v}' violates bounds: {}",
                        self.key, self.bounds_desc
                    ));
                }
                Ok(())
            }
            (ConfigValue::U64(v), ConfigType::U64) => {
                if self.key == "runtime.execution_timeout_ms" && (*v < 100 || *v > 3600000) {
                    return Err(format!(
                        "Fix: configuration key '{}' value '{v}' violates bounds: {}",
                        self.key, self.bounds_desc
                    ));
                }
                if self.key == "runtime.device_wait_timeout_ms" && (*v < 1000 || *v > 300000) {
                    return Err(format!(
                        "Fix: configuration key '{}' value '{v}' violates bounds: {}",
                        self.key, self.bounds_desc
                    ));
                }
                Ok(())
            }
            (ConfigValue::F64(v), ConfigType::F64) => {
                if self.key == "diag.perf_sampling_rate" && (*v < 0.0 || *v > 1.0 || v.is_nan()) {
                    return Err(format!(
                        "Fix: configuration key '{}' value '{v}' violates bounds: {}",
                        self.key, self.bounds_desc
                    ));
                }
                Ok(())
            }
            (ConfigValue::String(s), ConfigType::String | ConfigType::Path) => {
                if self.key == "compile.target_backend" {
                    // A backend id is spelled by the driver crate that registers it, so this
                    // layer checks the selector shape and never a roster of concrete names.
                    let well_formed = s.len() <= 64
                        && s.starts_with(|c: char| c.is_ascii_lowercase())
                        && s.chars().all(|c| {
                            c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_'
                        });
                    if !well_formed {
                        return Err(format!(
                            "Fix: configuration key '{}' value '{s}' violates bounds: {}",
                            self.key, self.bounds_desc
                        ));
                    }
                }
                if self.key == "diag.trace_level" {
                    let allowed = ["off", "error", "warn", "info", "debug", "trace"];
                    if !allowed.contains(&s.as_str()) {
                        return Err(format!(
                            "Fix: configuration key '{}' value '{s}' violates bounds: {}",
                            self.key, self.bounds_desc
                        ));
                    }
                }
                Ok(())
            }
            _ => Err(format!(
                "Fix: type mismatch for configuration key '{}'; expected {:?}",
                self.key, self.field_type
            )),
        }
    }
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
        bounds_desc: "auto|reference|registered backend id",
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
    ConfigFieldDef {
        key: "compile.planar_rewrite_batch_threshold",
        partition: ConfigPartition::SemanticCompileInput,
        field_type: ConfigType::U32,
        owner: "vyre-foundation",
        default_raw: "64",
        bounds_desc: "1..=1024",
        identity_impact: IdentityImpact::AffectsCompileIdentity,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::ImmutableAtRuntime,
        requires_restart: false,
        help_text: "Threshold for batching planar graph rewrite rules in the pass engine.",
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
    ConfigFieldDef {
        key: "runtime.device_wait_timeout_ms",
        partition: ConfigPartition::OperationalPolicy,
        field_type: ConfigType::U64,
        owner: "vyre-driver",
        default_raw: "10000",
        bounds_desc: "1000..=300000",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::MutableBetweenRequests,
        requires_restart: false,
        help_text: "Timeout in milliseconds waiting for device context acquisition.",
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
    ConfigFieldDef {
        key: "diag.dump_primary_text",
        partition: ConfigPartition::DiagnosticControls,
        field_type: ConfigType::Bool,
        owner: "vyre-driver",
        default_raw: "false",
        bounds_desc: "true|false",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Public,
        mutability: ConfigMutability::MutableBetweenRequests,
        requires_restart: false,
        help_text: "Dump the emitted primary text source for target diagnostics.",
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
    ConfigFieldDef {
        key: "credentials.conform_worker_secret",
        partition: ConfigPartition::Credentials,
        field_type: ConfigType::String,
        owner: "vyre-conform",
        default_raw: "",
        bounds_desc: "hex/base64 string or empty",
        identity_impact: IdentityImpact::OperationalOnly,
        secrecy: ConfigSecrecy::Secret,
        mutability: ConfigMutability::ImmutableAtRuntime,
        requires_restart: true,
        help_text: "Secret token for worker process authentication in conformance sweeps.",
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
    /// Process environment variable injection point (fourth; only for named credentials/deployment).
    EnvironmentInjection = 3,
    /// Compiled default fallback (lowest).
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

        def.validate_value(&value)?;

        // Apply only if new layer has higher or equal precedence (lower numeric enum value)
        if let Some((_, current_layer)) = self.values.get(key) {
            if layer > *current_layer {
                return Ok(()); // Lower precedence layer cannot overwrite higher precedence
            }
        }

        self.values.insert(String::from(key), (value, layer));
        Ok(())
    }

    /// Parse and apply a versioned TOML configuration string.
    ///
    /// # Errors
    ///
    /// Returns error if TOML is malformed, schema_version is missing/unsupported,
    /// unknown fields are encountered, or values violate bounds.
    pub fn apply_toml_str(&mut self, toml_str: &str) -> Result<(), String> {
        let table: toml::Table = toml::from_str(toml_str)
            .map_err(|e| format!("Fix: malformed TOML configuration: {e}"))?;
        self.apply_toml_table(&table)
    }

    /// Apply a parsed TOML configuration table.
    ///
    /// # Errors
    ///
    /// Returns error if schema_version is missing/unsupported,
    /// unknown fields are encountered, or values violate bounds.
    pub fn apply_toml_table(&mut self, table: &toml::Table) -> Result<(), String> {
        let version = table
            .get("schema_version")
            .and_then(|v| match v {
                toml::Value::Integer(i) => Some(*i),
                _ => None,
            })
            .ok_or_else(|| {
                String::from("Fix: TOML configuration must declare integer `schema_version = 1`.")
            })?;
        if version != 1 {
            return Err(format!(
                "Fix: unsupported TOML configuration schema_version {version}, expected 1."
            ));
        }

        // Traverse tables: both flat keys ("compile.opt_level" = 3) and nested tables ([compile] opt_level = 3)
        let mut flattened = Vec::new();
        for (k, v) in table {
            if k == "schema_version" {
                continue;
            }
            if let toml::Value::Table(sub_table) = v {
                for (sub_k, sub_v) in sub_table {
                    flattened.push((format!("{k}.{sub_k}"), sub_v));
                }
            } else {
                flattened.push((k.clone(), v));
            }
        }

        for (key, val) in flattened {
            let def = CANONICAL_CONFIG_FIELDS
                .iter()
                .find(|d| d.key == key.as_str())
                .ok_or_else(|| {
                    format!(
                        "Fix: unknown configuration key '{key}' in TOML configuration file. Run with --help to view registered configuration schema keys."
                    )
                })?;

            let config_val = match (val, def.field_type) {
                (toml::Value::Boolean(b), ConfigType::Bool) => ConfigValue::Bool(*b),
                (toml::Value::Integer(i), ConfigType::U32) => {
                    if *i < 0 || *i > u32::MAX as i64 {
                        return Err(format!(
                            "Fix: integer out of u32 range for configuration key '{key}': {i}"
                        ));
                    }
                    ConfigValue::U32(*i as u32)
                }
                (toml::Value::Integer(i), ConfigType::U64) => {
                    if *i < 0 {
                        return Err(format!(
                            "Fix: integer out of u64 range for configuration key '{key}': {i}"
                        ));
                    }
                    ConfigValue::U64(*i as u64)
                }
                (toml::Value::Float(f), ConfigType::F64) => ConfigValue::F64(*f),
                (toml::Value::String(s), ConfigType::String | ConfigType::Path) => {
                    ConfigValue::String(s.clone())
                }
                _ => {
                    return Err(format!(
                        "Fix: type mismatch for configuration key '{key}' in TOML; expected {:?}",
                        def.field_type
                    ));
                }
            };

            def.validate_value(&config_val)?;
            self.apply_override(&key, config_val, ConfigLayer::TomlFile)?;
        }
        Ok(())
    }

    /// Parse and apply a CLI override argument in `key=value` or `key:value` format.
    ///
    /// # Errors
    ///
    /// Returns error if format is invalid, key is unknown, or value violates bounds.
    pub fn apply_cli_arg(&mut self, key: &str, raw_value: &str) -> Result<(), String> {
        let def = CANONICAL_CONFIG_FIELDS
            .iter()
            .find(|d| d.key == key)
            .ok_or_else(|| {
                format!(
                    "Fix: unknown configuration key '{key}'. Run with --help to view registered configuration schema keys."
                )
            })?;

        let val = match def.field_type {
            ConfigType::Bool => {
                let b = match raw_value {
                    "true" | "1" | "yes" | "on" => true,
                    "false" | "0" | "no" | "off" => false,
                    _ => {
                        return Err(format!(
                            "Fix: invalid boolean value '{raw_value}' for configuration key '{key}'; expected true|false"
                        ));
                    }
                };
                ConfigValue::Bool(b)
            }
            ConfigType::U32 => {
                let v = raw_value.parse::<u32>().map_err(|e| {
                    format!(
                        "Fix: invalid u32 value '{raw_value}' for configuration key '{key}': {e}"
                    )
                })?;
                ConfigValue::U32(v)
            }
            ConfigType::U64 => {
                let v = raw_value.parse::<u64>().map_err(|e| {
                    format!(
                        "Fix: invalid u64 value '{raw_value}' for configuration key '{key}': {e}"
                    )
                })?;
                ConfigValue::U64(v)
            }
            ConfigType::F64 => {
                let v = raw_value.parse::<f64>().map_err(|e| {
                    format!(
                        "Fix: invalid f64 value '{raw_value}' for configuration key '{key}': {e}"
                    )
                })?;
                ConfigValue::F64(v)
            }
            ConfigType::String | ConfigType::Path => ConfigValue::String(String::from(raw_value)),
        };

        def.validate_value(&val)?;
        self.apply_override(key, val, ConfigLayer::CliOverride)
    }

    /// Apply an explicit library value (highest precedence).
    ///
    /// # Errors
    ///
    /// Returns error if key is unknown or value violates bounds/type.
    pub fn apply_explicit(&mut self, key: &str, value: ConfigValue) -> Result<(), String> {
        let def = CANONICAL_CONFIG_FIELDS
            .iter()
            .find(|d| d.key == key)
            .ok_or_else(|| {
                format!(
                    "Fix: unknown configuration key '{key}'. Run with --help to view registered configuration schema keys."
                )
            })?;
        def.validate_value(&value)?;
        self.apply_override(key, value, ConfigLayer::ExplicitLibrary)
    }

    /// Apply a named environment variable injection point converted once at the process boundary.
    ///
    /// # Errors
    ///
    /// Returns error if the environment variable is not a recognized injection point,
    /// or if parsing/bounds fail.
    pub fn apply_named_env_var(&mut self, env_name: &str, raw_val: &str) -> Result<(), String> {
        let canonical_key = match env_name {
            "VYRE_FLEET_AUTH_TOKEN" => "credentials.fleet_auth_token",
            "VYRE_CONFORM_WORKER_SECRET" => "credentials.conform_worker_secret",
            "VYRE_BACKEND" => "compile.target_backend",
            "VYRE_OPT_LEVEL" => "compile.opt_level",
            "VYRE_ENABLE_FMA" => "compile.enable_fma",
            "VYRE_PLANAR_REWRITE_BATCH_THRESHOLD" => "compile.planar_rewrite_batch_threshold",
            "VYRE_MAX_QUEUE_DEPTH" => "runtime.max_queue_depth",
            "VYRE_EXECUTION_TIMEOUT_MS" => "runtime.execution_timeout_ms",
            "VYRE_DEVICE_WAIT_TIMEOUT_MS" => "runtime.device_wait_timeout_ms",
            "VYRE_CACHE_DIR" => "runtime.cache_dir",
            "VYRE_TRACE_LEVEL" | "VYRE_TRACE" => "diag.trace_level",
            "VYRE_PERF_SAMPLING_RATE" => "diag.perf_sampling_rate",
            "VYRE_DUMP_PRIMARY_TEXT" => "diag.dump_primary_text",
            _ => {
                return Err(format!(
                    "Fix: unrecognized environment variable injection point '{env_name}'."
                ));
            }
        };

        let def = CANONICAL_CONFIG_FIELDS
            .iter()
            .find(|d| d.key == canonical_key)
            .ok_or_else(|| {
                format!(
                    "Fix: unknown configuration key '{canonical_key}' for env var '{env_name}'."
                )
            })?;

        let val = match def.field_type {
            ConfigType::Bool => {
                let b = match raw_val {
                    "true" | "1" | "yes" | "on" => true,
                    "false" | "0" | "no" | "off" => false,
                    _ => {
                        return Err(format!(
                            "Fix: invalid boolean value '{raw_val}' for env var '{env_name}'; expected true|false"
                        ));
                    }
                };
                ConfigValue::Bool(b)
            }
            ConfigType::U32 => {
                let v = raw_val.parse::<u32>().map_err(|e| {
                    format!("Fix: invalid u32 value '{raw_val}' for env var '{env_name}': {e}")
                })?;
                ConfigValue::U32(v)
            }
            ConfigType::U64 => {
                let v = raw_val.parse::<u64>().map_err(|e| {
                    format!("Fix: invalid u64 value '{raw_val}' for env var '{env_name}': {e}")
                })?;
                ConfigValue::U64(v)
            }
            ConfigType::F64 => {
                let v = raw_val.parse::<f64>().map_err(|e| {
                    format!("Fix: invalid f64 value '{raw_val}' for env var '{env_name}': {e}")
                })?;
                ConfigValue::F64(v)
            }
            ConfigType::String | ConfigType::Path => ConfigValue::String(String::from(raw_val)),
        };

        def.validate_value(&val)?;
        self.apply_override(canonical_key, val, ConfigLayer::EnvironmentInjection)
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

    /// Compute 32-byte cryptographic artifact identity over base IR identity and all semantic compile inputs.
    ///
    /// Operational policy, diagnostic controls, and credentials do not affect this identity.
    #[must_use]
    pub fn compile_artifact_identity(&self, base_ir_identity: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"VYRE_ARTIFACT_IDENTITY_V1");
        hasher.update(base_ir_identity);

        for def in CANONICAL_CONFIG_FIELDS {
            if def.identity_impact == IdentityImpact::AffectsCompileIdentity {
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

    /// Format diagnostic summary of resolved configuration with secrets strictly redacted.
    #[must_use]
    pub fn format_diagnostic(&self) -> String {
        let mut out = String::from("Vyre Resolved Configuration:\n");
        for def in CANONICAL_CONFIG_FIELDS {
            if let Some((val, layer)) = self.values.get(def.key) {
                let rendered = val.display_redacted(def.secrecy);
                out.push_str(&format!(
                    "  {} = {} (source: {:?}, partition: {})\n",
                    def.key, rendered, layer, def.partition
                ));
            }
        }
        out
    }
}

/// Generate the configuration reference markdown table from canonical schema definitions.
#[must_use]
pub fn render_configuration_reference_markdown() -> String {
    let mut out = String::from("# Configuration Reference\n\n");
    out.push_str(
        "Authoritative schema for Vyre compile, runtime, diagnostic, and credential options.\n\n",
    );
    out.push_str("| Key | Partition | Type | Owner | Default | Bounds | Secrecy | Description |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");

    for def in CANONICAL_CONFIG_FIELDS {
        out.push_str(&format!(
            "| `{}` | `{}` | `{:?}` | `{}` | `{}` | `{}` | `{:?}` | {} |\n",
            def.key,
            def.partition.as_str(),
            def.field_type,
            def.owner,
            def.default_raw,
            def.bounds_desc,
            def.secrecy,
            def.help_text,
        ));
    }
    out
}

/// Generate CLI configuration options help text.
#[must_use]
pub fn render_cli_help() -> String {
    let mut out = String::from("Configuration Options:\n");
    for def in CANONICAL_CONFIG_FIELDS {
        out.push_str(&format!(
            "  --{} <{:?}> (default: '{}')\n      {}\n",
            def.key.replace('.', "-").replace('_', "-"),
            def.field_type,
            def.default_raw,
            def.help_text
        ));
    }
    out
}
