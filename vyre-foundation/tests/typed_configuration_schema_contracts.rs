//! Tests for typed configuration schema, partition bounds, precedence resolution, and secret redaction.
//!
//! WHY: proves Row 123:
//! - Precedence order (Explicit > CLI > TOML > Environment > Default) is strictly enforced.
//! - Secret fields are redacted from display and receipts.
//! - Behavior-affecting fields alter `behavior_hash`, while operational/diagnostic fields do not.
//! - Unknown keys and type mismatches fail with actionable corrective diagnostics.

use vyre_foundation::config_schema::{
    ConfigLayer, ConfigPartition, ConfigSecrecy, ConfigType, ConfigValue, ResolvedConfiguration,
    CANONICAL_CONFIG_FIELDS,
};

#[test]
fn configuration_schema_covers_all_partitions() {
    for partition in ConfigPartition::ALL {
        let count = CANONICAL_CONFIG_FIELDS
            .iter()
            .filter(|f| f.partition == *partition)
            .count();
        assert!(
            count > 0,
            "Fix: ConfigPartition '{partition}' must have at least one registered field."
        );
    }
}

#[test]
fn configuration_precedence_explicit_overrides_cli_and_toml() {
    let mut config = ResolvedConfiguration::new_with_defaults();

    // TOML sets opt_level = 1
    config
        .apply_override("compile.opt_level", ConfigValue::U32(1), ConfigLayer::TomlFile)
        .expect("Fix: TOML override must succeed.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(1)));

    // CLI overrides TOML with opt_level = 3
    config
        .apply_override("compile.opt_level", ConfigValue::U32(3), ConfigLayer::CliOverride)
        .expect("Fix: CLI override must succeed over TOML.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(3)));

    // Explicit library code overrides CLI with opt_level = 2
    config
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(2),
            ConfigLayer::ExplicitLibrary,
        )
        .expect("Fix: ExplicitLibrary override must succeed over CLI.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(2)));

    // Lower precedence TOML cannot overwrite ExplicitLibrary
    config
        .apply_override("compile.opt_level", ConfigValue::U32(0), ConfigLayer::TomlFile)
        .expect("Fix: Lower precedence layer cannot fail, but must not overwrite.");
    assert_eq!(
        config.get("compile.opt_level"),
        Some(&ConfigValue::U32(2)),
        "Fix: lower precedence layer must not overwrite higher precedence value."
    );
}

#[test]
fn secret_credentials_are_redacted_in_inspection_views() {
    let mut config = ResolvedConfiguration::new_with_defaults();

    let mut secret_count = 0;
    let mut public_count = 0;

    for def in CANONICAL_CONFIG_FIELDS {
        match def.secrecy {
            ConfigSecrecy::Secret => {
                secret_count += 1;
                let secret_probe = match def.field_type {
                    ConfigType::Bool => ConfigValue::Bool(true),
                    ConfigType::U32 => ConfigValue::U32(99999),
                    ConfigType::U64 => ConfigValue::U64(99999),
                    ConfigType::F64 => ConfigValue::F64(999.99),
                    ConfigType::String | ConfigType::Path => {
                        ConfigValue::String("unredacted_sensitive_secret_payload_xyz".into())
                    }
                };

                config
                    .apply_override(def.key, secret_probe.clone(), ConfigLayer::ExplicitLibrary)
                    .unwrap_or_else(|e| panic!("Fix: override for secret field {} failed: {e}", def.key));

                let view = config.redacted_view();
                let redacted_val = view
                    .get(def.key)
                    .unwrap_or_else(|| panic!("Fix: field {} missing from redacted_view", def.key));

                assert_eq!(
                    redacted_val, "[REDACTED]",
                    "Fix: secret field '{}' must be strictly redacted in inspection view.",
                    def.key
                );
                assert_eq!(
                    secret_probe.display_redacted(ConfigSecrecy::Secret),
                    "[REDACTED]",
                    "Fix: display_redacted with ConfigSecrecy::Secret must output [REDACTED]."
                );
            }
            ConfigSecrecy::Public => {
                public_count += 1;
                let view = config.redacted_view();
                let public_val = view
                    .get(def.key)
                    .unwrap_or_else(|| panic!("Fix: field {} missing from redacted_view", def.key));

                assert_ne!(
                    public_val, "[REDACTED]",
                    "Fix: public field '{}' must not be redacted in inspection view.",
                    def.key
                );
            }
        }
    }

    assert!(
        secret_count > 0,
        "Fix: at least one ConfigSecrecy::Secret field must be registered in CANONICAL_CONFIG_FIELDS."
    );
    assert!(
        public_count > 0,
        "Fix: at least one ConfigSecrecy::Public field must be registered in CANONICAL_CONFIG_FIELDS."
    );
}

#[test]
fn behavior_hash_changes_only_on_identity_affecting_fields() {
    let mut config1 = ResolvedConfiguration::new_with_defaults();
    let hash_default = config1.behavior_hash();

    // Changing operational or diagnostic field does not alter behavior_hash
    config1
        .apply_override(
            "diag.trace_level",
            ConfigValue::String("debug".into()),
            ConfigLayer::ExplicitLibrary,
        )
        .unwrap();
    config1
        .apply_override(
            "runtime.max_queue_depth",
            ConfigValue::U32(2048),
            ConfigLayer::ExplicitLibrary,
        )
        .unwrap();
    assert_eq!(
        config1.behavior_hash(),
        hash_default,
        "Fix: operational and diagnostic changes must not alter behavior_hash."
    );

    // Changing compile.opt_level alters behavior_hash
    config1
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(3),
            ConfigLayer::ExplicitLibrary,
        )
        .unwrap();
    assert_ne!(
        config1.behavior_hash(),
        hash_default,
        "Fix: compile optimization level must alter behavior_hash."
    );
}

#[test]
fn unknown_keys_and_type_mismatches_fail_with_actionable_errors() {
    let mut config = ResolvedConfiguration::new_with_defaults();

    let err_unknown = config
        .apply_override(
            "nonexistent.key",
            ConfigValue::Bool(true),
            ConfigLayer::CliOverride,
        )
        .expect_err("Fix: unknown configuration key must fail closed.");
    assert!(err_unknown.contains("Fix:"));

    let err_type = config
        .apply_override(
            "compile.opt_level",
            ConfigValue::Bool(true), // Expected U32
            ConfigLayer::CliOverride,
        )
        .expect_err("Fix: type mismatch must fail closed.");
    assert!(err_type.contains("Fix:"));
}
