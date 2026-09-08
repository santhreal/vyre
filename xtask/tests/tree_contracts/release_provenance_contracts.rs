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

    // Verify each dependency is classified and pinned
    for dep in &authority.dependencies {
        assert!(!dep.name.is_empty());
        assert!(!dep.version.is_empty());
        assert!(!dep.license.is_empty());
        assert!(dep.is_policy_approved);
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
    assert!(sbom_val["components"].as_array().unwrap().len() > 10);

    // 2. SLSA v1.2 Provenance generation
    let slsa_json = authority.generate_slsa_provenance(&root).expect("slsa provenance generation");
    let slsa_val: serde_json::Value = serde_json::from_str(&slsa_json).expect("valid slsa json");
    assert_eq!(slsa_val["_type"], "https://in-toto.io/Statement/v1");
    assert_eq!(slsa_val["predicateType"], "https://slsa.dev/provenance/v1");
}

#[test]
fn stale_release_provenance_schema_fails_closed() {
    let root = checkout_root();
    let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
        .expect("provenance authority generation");
    authority.schema_version = 99; // Stale schema version

    let toml = authority.to_toml().expect("to toml");
    let err = ReleaseProvenanceAuthority::from_toml(&toml).unwrap_err();
    assert!(matches!(err, ProvenanceError::StaleSchemaVersion { expected: 1, found: 99 }));
}
