//! Tests for typed configuration schema, partition bounds, precedence resolution, and secret redaction.
//!
//! WHY: proves Row 123:
//! - Precedence order (Explicit > CLI > TOML > Environment > Default) is strictly enforced.
//! - Secret fields are strictly redacted from display, inspection views, diagnostics, and receipts.
//! - Behavior-affecting and semantic compile inputs alter artifact identity; operational policy, diagnostics, and credentials do not.
//! - Unknown keys, schema version mismatches, and bounds violations fail with actionable corrective diagnostics.
//! - Source-derived field set closure validates that every field declares owner, bounds, default, identity effect, and secrecy.

use vyre_foundation::{
    render_cli_help, render_configuration_reference_markdown, ConfigFieldDef, ConfigLayer,
    ConfigMutability, ConfigPartition, ConfigSecrecy, ConfigType, ConfigValue, IdentityImpact,
    ResolvedConfiguration, CANONICAL_CONFIG_FIELDS,
};

#[test]
fn configuration_schema_covers_all_partitions() {
    for partition in ConfigPartition::ALL {
        let count = CANONICAL_CONFIG_FIELDS
            .iter()
            .filter(|def| def.partition == *partition)
            .count();
        assert!(
            count > 0,
            "Fix: configuration schema must declare at least one field for partition '{partition}'."
        );
    }
}

#[test]
fn configuration_precedence_explicit_overrides_cli_and_toml() {
    let mut config = ResolvedConfiguration::new_with_defaults();
    assert_eq!(
        config.get("compile.opt_level"),
        Some(&ConfigValue::U32(2)),
        "Default opt_level must be 2"
    );

    // 1. TOML sets opt_level = 1 (TOML beats Default)
    config
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(1),
            ConfigLayer::TomlFile,
        )
        .expect("Fix: TOML override must succeed over default.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(1)));

    // 2. CLI overrides TOML with opt_level = 3 (CLI beats TOML)
    config
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(3),
            ConfigLayer::CliOverride,
        )
        .expect("Fix: CLI override must succeed over TOML.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(3)));

    // 3. Explicit library code overrides CLI with opt_level = 0 (Explicit beats CLI)
    config
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(0),
            ConfigLayer::ExplicitLibrary,
        )
        .expect("Fix: ExplicitLibrary override must succeed over CLI.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(0)));

    // 4. Lower precedence TOML cannot overwrite ExplicitLibrary
    config
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(2),
            ConfigLayer::TomlFile,
        )
        .expect("Fix: Lower precedence layer must return Ok without overwriting.");
    assert_eq!(
        config.get("compile.opt_level"),
        Some(&ConfigValue::U32(0)),
        "Fix: ExplicitLibrary value must remain authoritative against TOML."
    );

    // 5. Lower precedence CLI cannot overwrite ExplicitLibrary
    config
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(3),
            ConfigLayer::CliOverride,
        )
        .expect("Fix: Lower precedence CLI must return Ok without overwriting.");
    assert_eq!(
        config.get("compile.opt_level"),
        Some(&ConfigValue::U32(0)),
        "Fix: ExplicitLibrary value must remain authoritative against CLI."
    );

    // 6. Lower precedence Environment injection cannot overwrite ExplicitLibrary
    config
        .apply_override(
            "compile.opt_level",
            ConfigValue::U32(1),
            ConfigLayer::EnvironmentInjection,
        )
        .expect("Fix: Lower precedence Environment must return Ok without overwriting.");
    assert_eq!(
        config.get("compile.opt_level"),
        Some(&ConfigValue::U32(0)),
        "Fix: ExplicitLibrary value must remain authoritative against Environment."
    );
}

#[test]
fn secret_credentials_are_strictly_redacted_in_all_views_and_diagnostics() {
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
                        ConfigValue::String("secret_fleet_bearer_token_xyz_never_leak".into())
                    }
                };

                config
                    .apply_override(def.key, secret_probe.clone(), ConfigLayer::ExplicitLibrary)
                    .unwrap_or_else(|e| {
                        panic!("Fix: override for secret field {} failed: {e}", def.key)
                    });

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

    // Verify diagnostic summary never contains raw secret content
    let diagnostic = config.format_diagnostic();
    assert!(
        !diagnostic.contains("secret_fleet_bearer_token_xyz_never_leak"),
        "Fix: diagnostic output must never contain raw secret material."
    );
    assert!(
        diagnostic.contains("credentials.fleet_auth_token = [REDACTED]"),
        "Fix: diagnostic output must display [REDACTED] for secret fields."
    );
}

#[test]
fn artifact_identity_isolates_operational_policy_from_semantic_compile_inputs() {
    let config_base = ResolvedConfiguration::new_with_defaults();
    let base_ir = b"program_graph_identity_vir0_op123_test_fixture";
    let id_base = config_base.compile_artifact_identity(base_ir);

    // Mutating operational policy, diagnostic controls, and credentials leaves artifact identity identical
    let mut config_op = config_base.clone();
    config_op
        .apply_explicit("runtime.max_queue_depth", ConfigValue::U32(8192))
        .expect("override max_queue_depth");
    config_op
        .apply_explicit("runtime.execution_timeout_ms", ConfigValue::U64(10000))
        .expect("override execution_timeout_ms");
    config_op
        .apply_explicit(
            "runtime.cache_dir",
            ConfigValue::String("/tmp/test_cache".into()),
        )
        .expect("override cache_dir");
    config_op
        .apply_explicit("diag.trace_level", ConfigValue::String("trace".into()))
        .expect("override trace_level");
    config_op
        .apply_explicit("diag.perf_sampling_rate", ConfigValue::F64(0.5))
        .expect("override perf_sampling_rate");
    config_op
        .apply_explicit(
            "credentials.fleet_auth_token",
            ConfigValue::String("token_abc".into()),
        )
        .expect("override fleet_auth_token");

    let id_op = config_op.compile_artifact_identity(base_ir);
    assert_eq!(
        id_base, id_op,
        "Fix: mutating operational policy, diagnostic controls, or credentials must not change compile artifact identity."
    );

    // Mutating semantic compile input (opt_level) changes artifact identity
    let mut config_semantic = config_base.clone();
    config_semantic
        .apply_explicit("compile.opt_level", ConfigValue::U32(3))
        .expect("override opt_level");
    let id_semantic = config_semantic.compile_artifact_identity(base_ir);
    assert_ne!(
        id_base, id_semantic,
        "Fix: changing compile.opt_level must change compile artifact identity."
    );

    // Mutating semantic compile input (enable_fma) changes artifact identity
    let mut config_fma = config_base.clone();
    config_fma
        .apply_explicit("compile.enable_fma", ConfigValue::Bool(false))
        .expect("override enable_fma");
    let id_fma = config_fma.compile_artifact_identity(base_ir);
    assert_ne!(
        id_base, id_fma,
        "Fix: changing compile.enable_fma must change compile artifact identity."
    );
    assert_ne!(
        id_semantic, id_fma,
        "Fix: distinct compile options must produce distinct artifact identities."
    );
}

#[test]
fn source_derived_field_set_closure_and_bounds_validation() {
    let known_crates = [
        "vyre-foundation",
        "vyre-runtime",
        "vyre-bench",
        "vyre-driver",
        "vyre-driver-cuda",
        "vyre-driver-wgpu",
        "vyre-conform",
    ];

    for def in CANONICAL_CONFIG_FIELDS {
        // 1. Key format
        assert!(
            def.key.contains('.'),
            "Fix: configuration key '{}' must be dot-separated.",
            def.key
        );

        // 2. Owner crate
        assert!(
            known_crates.contains(&def.owner),
            "Fix: configuration key '{}' owner '{}' must be a recognized workspace crate.",
            def.key,
            def.owner
        );

        // 3. Bounds & help text
        assert!(
            !def.bounds_desc.is_empty(),
            "Fix: configuration key '{}' must declare non-empty bounds_desc.",
            def.key
        );
        assert!(
            !def.help_text.is_empty(),
            "Fix: configuration key '{}' must declare non-empty help_text.",
            def.key
        );

        // 4. Default raw must parse and satisfy bounds
        let default_val = match def.field_type {
            ConfigType::Bool => ConfigValue::Bool(def.default_raw == "true"),
            ConfigType::U32 => ConfigValue::U32(
                def.default_raw
                    .parse()
                    .unwrap_or_else(|_| panic!("Fix: invalid u32 default for {}", def.key)),
            ),
            ConfigType::U64 => ConfigValue::U64(
                def.default_raw
                    .parse()
                    .unwrap_or_else(|_| panic!("Fix: invalid u64 default for {}", def.key)),
            ),
            ConfigType::F64 => ConfigValue::F64(
                def.default_raw
                    .parse()
                    .unwrap_or_else(|_| panic!("Fix: invalid f64 default for {}", def.key)),
            ),
            ConfigType::String | ConfigType::Path => {
                ConfigValue::String(String::from(def.default_raw))
            }
        };
        def.validate_value(&default_val)
            .unwrap_or_else(|e| panic!("Fix: default value for {} violates bounds: {e}", def.key));

        // 5. Partition and identity/secrecy alignment
        match def.partition {
            ConfigPartition::SemanticCompileInput => {
                assert_eq!(
                    def.identity_impact,
                    IdentityImpact::AffectsCompileIdentity,
                    "Fix: semantic compile input '{}' must affect compile identity.",
                    def.key
                );
                assert_eq!(
                    def.secrecy,
                    ConfigSecrecy::Public,
                    "Fix: semantic compile input '{}' must be public.",
                    def.key
                );
            }
            ConfigPartition::Credentials => {
                assert_eq!(
                    def.secrecy,
                    ConfigSecrecy::Secret,
                    "Fix: credential '{}' must be classified as Secret.",
                    def.key
                );
                assert_eq!(
                    def.identity_impact,
                    IdentityImpact::OperationalOnly,
                    "Fix: credential '{}' must be OperationalOnly.",
                    def.key
                );
            }
            ConfigPartition::OperationalPolicy | ConfigPartition::DiagnosticControls => {
                assert_ne!(
                    def.identity_impact,
                    IdentityImpact::AffectsCompileIdentity,
                    "Fix: operational/diagnostic field '{}' must not affect compile identity.",
                    def.key
                );
            }
            _ => {}
        }
    }

    // Verify synthetic invalid field definition is rejected by bounds validation
    let bad_opt_field = ConfigFieldDef {
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
        help_text: "Optimization level.",
    };
    let bad_val = ConfigValue::U32(99);
    let err = bad_opt_field
        .validate_value(&bad_val)
        .expect_err("out of bounds must fail");
    assert!(err.contains("Fix:"));
    assert!(err.contains("violates bounds: 0..=3"));
}

#[test]
fn versioned_toml_loading_and_rejection_rules() {
    let mut config = ResolvedConfiguration::new_with_defaults();

    // 1. Valid versioned TOML
    let valid_toml = r#"
schema_version = 1

[compile]
opt_level = 3
enable_fma = false

[runtime]
max_queue_depth = 4096
"#;
    config
        .apply_toml_str(valid_toml)
        .expect("Fix: valid versioned TOML must succeed.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(3)));
    assert_eq!(
        config.get("compile.enable_fma"),
        Some(&ConfigValue::Bool(false))
    );
    assert_eq!(
        config.get("runtime.max_queue_depth"),
        Some(&ConfigValue::U32(4096))
    );

    // 2. Missing schema_version fails
    let no_version_toml = r#"
[compile]
opt_level = 1
"#;
    let err_no_ver = config
        .apply_toml_str(no_version_toml)
        .expect_err("missing schema_version must fail");
    assert!(err_no_ver.contains("Fix:"));
    assert!(err_no_ver.contains("schema_version = 1"));

    // 3. Unsupported schema_version fails
    let bad_version_toml = r#"
schema_version = 2
"#;
    let err_bad_ver = config
        .apply_toml_str(bad_version_toml)
        .expect_err("unsupported schema_version must fail");
    assert!(err_bad_ver.contains("Fix:"));
    assert!(err_bad_ver.contains("unsupported TOML configuration schema_version 2"));

    // 4. Unknown key in TOML fails
    let unknown_key_toml = r#"
schema_version = 1
[compile]
unknown_option_abc = 42
"#;
    let err_unknown = config
        .apply_toml_str(unknown_key_toml)
        .expect_err("unknown key must fail");
    assert!(err_unknown.contains("Fix:"));
    assert!(err_unknown.contains("unknown configuration key 'compile.unknown_option_abc'"));

    // 5. Out of bounds value in TOML fails
    let out_of_bounds_toml = r#"
schema_version = 1
[compile]
opt_level = 10
"#;
    let err_bounds = config
        .apply_toml_str(out_of_bounds_toml)
        .expect_err("out of bounds value must fail");
    assert!(err_bounds.contains("Fix:"));
    assert!(err_bounds.contains("violates bounds: 0..=3"));
}

#[test]
fn cli_overrides_and_bounds_validation() {
    let mut config = ResolvedConfiguration::new_with_defaults();

    // 1. Valid CLI overrides
    config
        .apply_cli_arg("compile.opt_level", "3")
        .expect("Fix: valid CLI override must succeed.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(3)));

    config
        .apply_cli_arg("compile.enable_fma", "false")
        .expect("Fix: valid CLI bool override must succeed.");
    assert_eq!(
        config.get("compile.enable_fma"),
        Some(&ConfigValue::Bool(false))
    );

    // 2. Unknown CLI key fails
    let err_unknown = config
        .apply_cli_arg("nonexistent.setting", "42")
        .expect_err("unknown CLI key must fail");
    assert!(err_unknown.contains("Fix:"));
    assert!(err_unknown.contains("unknown configuration key 'nonexistent.setting'"));

    // 3. Out of bounds CLI value fails
    let err_bounds = config
        .apply_cli_arg("compile.opt_level", "5")
        .expect_err("out of bounds CLI value must fail");
    assert!(err_bounds.contains("Fix:"));
    assert!(err_bounds.contains("violates bounds: 0..=3"));

    // 4. Invalid integer format fails
    let err_parse = config
        .apply_cli_arg("compile.opt_level", "invalid_num")
        .expect_err("invalid integer format must fail");
    assert!(err_parse.contains("Fix:"));
}

#[test]
fn named_environment_variable_injection_boundary() {
    let mut config = ResolvedConfiguration::new_with_defaults();

    // 1. Valid named env var injection
    config
        .apply_named_env_var("VYRE_OPT_LEVEL", "3")
        .expect("Fix: named env var injection must succeed.");
    assert_eq!(config.get("compile.opt_level"), Some(&ConfigValue::U32(3)));

    config
        .apply_named_env_var("VYRE_FLEET_AUTH_TOKEN", "bearer_token_123")
        .expect("Fix: secret credential env var injection must succeed.");
    assert_eq!(
        config.get("credentials.fleet_auth_token"),
        Some(&ConfigValue::String("bearer_token_123".into()))
    );

    // 2. Unrecognized environment variable fails
    let err_unrecognized = config
        .apply_named_env_var("RANDOM_UNREGISTERED_ENV_VAR", "value")
        .expect_err("unrecognized env var must fail");
    assert!(err_unrecognized.contains("Fix:"));
    assert!(err_unrecognized.contains("unrecognized environment variable injection point"));
}

#[test]
fn generated_help_and_configuration_reference() {
    let reference_md = render_configuration_reference_markdown();
    assert!(
        reference_md.contains("# Configuration Reference"),
        "Reference markdown must contain header"
    );
    assert!(
        reference_md.contains("compile.opt_level"),
        "Reference markdown must list compile.opt_level"
    );
    assert!(
        reference_md.contains("credentials.fleet_auth_token"),
        "Reference markdown must list credentials.fleet_auth_token"
    );

    let cli_help = render_cli_help();
    assert!(
        cli_help.contains("Configuration Options:"),
        "CLI help must contain header"
    );
    assert!(
        cli_help.contains("--compile-opt-level"),
        "CLI help must list --compile-opt-level"
    );
    assert!(
        cli_help.contains("--runtime-max-queue-depth"),
        "CLI help must list --runtime-max-queue-depth"
    );
}
