//! Tree contracts for Dependency & Release-Provenance Authority (Row 116).

use xtask::checkout::checkout_root;
use xtask::provenance::*;

#[test]
fn release_provenance_authority_captures_lockfile_and_dependencies() {
    let root = checkout_root();
    let authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");

    assert_eq!(authority.schema_version, RELEASE_PROVENANCE_SCHEMA_VERSION);
    assert!(!authority.lockfile_digest.is_empty());
    assert!(!authority.dependencies.is_empty());
    assert!(authority.is_offline_capable);
    assert!(!authority.reproducible_archive_hash.is_empty());

    // Verify each dependency is classified and pinned
    for dep in &authority.dependencies {
        assert!(!dep.name.is_empty());
        assert!(!dep.version.is_empty());
        assert!(!dep.license.is_empty());
        assert!(
            dep.is_policy_approved,
            "dependency '{}@{}' with license '{}' must be policy-approved",
            dep.name, dep.version, dep.license
        );
    }
}

#[test]
fn adding_unapproved_dependency_or_missing_license_turns_suite_red() {
    let root = checkout_root();
    let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");

    // 1. Dependency with no license turns suite red
    let unclassified_dep = PinnedCrateDependency {
        name: "unclassified-crate".to_string(),
        version: "0.1.0".to_string(),
        checksum: "abcdef0123456789".to_string(),
        source: "registry+https://github.com/rust-lang/crates.io-index".to_string(),
        license: "UNKNOWN".to_string(),
        is_policy_approved: false,
    };
    authority.dependencies.push(unclassified_dep);

    let err = authority.verify_offline_integrity().unwrap_err();
    assert!(matches!(err, ProvenanceError::UnapprovedLicense { .. }));

    // 2. Dependency with unapproved non-permissive license (e.g. AGPL-3.0) turns suite red
    let mut authority2 = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");
    let agpl_dep = PinnedCrateDependency {
        name: "agpl-crate".to_string(),
        version: "1.0.0".to_string(),
        checksum: "1234567890abcdef".to_string(),
        source: "registry+https://github.com/rust-lang/crates.io-index".to_string(),
        license: "AGPL-3.0-only".to_string(),
        is_policy_approved: false,
    };
    authority2.dependencies.push(agpl_dep);

    let err2 = authority2.verify_offline_integrity().unwrap_err();
    assert!(matches!(err2, ProvenanceError::UnapprovedLicense { package, license } if package == "agpl-crate" && license == "AGPL-3.0-only"));
}

#[test]
fn unpinned_dependency_checksum_is_refused() {
    let root = checkout_root();
    let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");

    // Unpinned third-party crate missing checksum
    authority.dependencies.push(PinnedCrateDependency {
        name: "unpinned-crate".to_string(),
        version: "2.0.0".to_string(),
        checksum: String::new(),
        source: "registry+https://github.com/rust-lang/crates.io-index".to_string(),
        license: "MIT".to_string(),
        is_policy_approved: true,
    });

    let err = authority.verify_offline_integrity().unwrap_err();
    assert!(matches!(&err, ProvenanceError::MissingChecksum(msg) if msg.contains("unpinned-crate")));
}

#[test]
fn unpinned_or_network_reading_build_input_is_refused_by_name() {
    let root = checkout_root();
    let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");

    authority.build_scripts.push(BuildScriptContract {
        crate_name: "vyre-remote-fetch".to_string(),
        path: "vyre-bench/build.rs".to_string(),
        declared_inputs: vec!["https://malicious.example.com/payload.bin".to_string()],
        declared_outputs: vec!["cargo:rustc-env=PAYLOAD".to_string()],
        has_network_access: true,
        has_bounded_reads: true,
    });

    let err = authority.verify_build_scripts(&root).unwrap_err();
    match err {
        ProvenanceError::UndeclaredBuildInput { build_script, undeclared_input } => {
            assert_eq!(build_script, "vyre-bench/build.rs");
            assert!(undeclared_input.contains("network access is strictly forbidden"));
        }
        other => panic!("expected UndeclaredBuildInput error, got {other:?}"),
    }
}

#[test]
fn unbounded_build_script_reads_are_refused() {
    let root = checkout_root();
    let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");

    authority.build_scripts.push(BuildScriptContract {
        crate_name: "vyre-unbounded".to_string(),
        path: "vyre-driver-wgpu/build.rs".to_string(),
        declared_inputs: vec!["Cargo.toml".to_string()],
        declared_outputs: vec![],
        has_network_access: false,
        has_bounded_reads: false,
    });

    let err = authority.verify_build_scripts(&root).unwrap_err();
    match err {
        ProvenanceError::UndeclaredBuildInput { build_script, undeclared_input } => {
            assert_eq!(build_script, "vyre-driver-wgpu/build.rs");
            assert!(undeclared_input.contains("unbounded filesystem reads"));
        }
        other => panic!("expected UndeclaredBuildInput error, got {other:?}"),
    }
}

#[test]
fn sbom_and_slsa_provenance_generation_succeeds() {
    let root = checkout_root();
    let authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");

    // 1. CycloneDX SBOM generation
    let sbom_json = authority.generate_sbom().expect("sbom generation");
    let sbom_val: serde_json::Value = serde_json::from_str(&sbom_json).expect("valid sbom json");
    assert_eq!(sbom_val["bomFormat"], "CycloneDX");
    assert_eq!(sbom_val["specVersion"], "1.5");
    assert!(sbom_val["components"].as_array().unwrap().len() > 100);

    // Verify metadata and tools
    assert!(sbom_val["metadata"]["tools"].as_array().unwrap().len() >= 2);

    // 2. SLSA v1.2 Provenance generation
    let slsa_json = authority.generate_slsa_provenance(&root).expect("slsa provenance generation");
    let slsa_val: serde_json::Value = serde_json::from_str(&slsa_json).expect("valid slsa json");
    assert_eq!(slsa_val["_type"], "https://in-toto.io/Statement/v1");
    assert_eq!(slsa_val["predicateType"], "https://slsa.dev/provenance/v1");
    assert_eq!(
        slsa_val["predicate"]["buildDefinition"]["buildType"],
        "https://vyre.dev/build/v1"
    );
    assert!(slsa_val["predicate"]["buildDefinition"]["resolvedDependencies"]
        .as_array()
        .unwrap()
        .len()
        > 100);
}

#[test]
fn stale_release_provenance_schema_fails_closed() {
    let root = checkout_root();
    let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");
    authority.schema_version = 99; // Stale schema version

    let toml = authority.to_toml().expect("to toml");
    let err = ReleaseProvenanceAuthority::from_toml(&toml).unwrap_err();
    assert!(matches!(
        err,
        ProvenanceError::StaleSchemaVersion {
            expected: 1,
            found: 99
        }
    ));
}

#[test]
fn all_workspace_build_scripts_satisfy_safety_contracts() {
    let root = checkout_root();
    let authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");

    // Workspace build scripts must pass verification on actual checkout
    authority
        .verify_build_scripts(&root)
        .expect("all workspace build scripts must be verified safe and bounded");
    authority
        .verify_offline_integrity()
        .expect("offline integrity check must pass for workspace");
}
