//! Contract tests for `vyre_driver::Target` historical contract and fail-closed deserialization.

use vyre_driver::Target;

#[test]
fn target_historical_variants_and_aot_target_ids() {
    assert_eq!(Target::Ptx.aot_target_id(), "secondary_text");
    assert_eq!(Target::SecondaryText.aot_target_id(), "secondary_text");
    assert_eq!(Target::SpirV.aot_target_id(), "spv");
    assert_eq!(Target::PrimaryText.aot_target_id(), "primary_text");
    assert_eq!(Target::PrimaryBinary.aot_target_id(), "primary_binary");
    assert_eq!(Target::NativeModule.aot_target_id(), "native_module");
    assert_eq!(
        Target::ReferenceBackend.aot_target_id(),
        "reference_backend"
    );
    assert_eq!(
        Target::Extension("my_backend").aot_target_id(),
        "my_backend"
    );
}

#[test]
fn target_extension_matches_aot_target_id() {
    let targets = [
        Target::Ptx,
        Target::SpirV,
        Target::PrimaryText,
        Target::PrimaryBinary,
        Target::SecondaryText,
        Target::NativeModule,
        Target::ReferenceBackend,
        Target::Extension("custom_ext"),
    ];

    for t in targets {
        assert_eq!(t.extension(), t.aot_target_id());
    }
}

#[test]
fn target_serialization() {
    assert_eq!(
        serde_json::to_string(&Target::Ptx).unwrap(),
        "\"secondary_text\""
    );
    assert_eq!(serde_json::to_string(&Target::SpirV).unwrap(), "\"spv\"");
    assert_eq!(
        serde_json::to_string(&Target::PrimaryText).unwrap(),
        "\"primary_text\""
    );
    assert_eq!(
        serde_json::to_string(&Target::PrimaryBinary).unwrap(),
        "\"primary_binary\""
    );
    assert_eq!(
        serde_json::to_string(&Target::NativeModule).unwrap(),
        "\"native_module\""
    );
    assert_eq!(
        serde_json::to_string(&Target::ReferenceBackend).unwrap(),
        "\"reference_backend\""
    );
    assert_eq!(
        serde_json::to_string(&Target::Extension("custom")).unwrap(),
        "\"custom\""
    );
}

#[test]
fn target_deserialization_known_variants() {
    assert_eq!(
        serde_json::from_str::<Target>("\"ptx\"").unwrap(),
        Target::Ptx
    );
    assert_eq!(
        serde_json::from_str::<Target>("\"secondary_text\"").unwrap(),
        Target::Ptx
    );
    assert_eq!(
        serde_json::from_str::<Target>("\"spv\"").unwrap(),
        Target::SpirV
    );
    assert_eq!(
        serde_json::from_str::<Target>("\"spirv\"").unwrap(),
        Target::SpirV
    );
    assert_eq!(
        serde_json::from_str::<Target>("\"primary_text\"").unwrap(),
        Target::PrimaryText
    );
    assert_eq!(
        serde_json::from_str::<Target>("\"primary_binary\"").unwrap(),
        Target::PrimaryBinary
    );
    assert_eq!(
        serde_json::from_str::<Target>("\"native_module\"").unwrap(),
        Target::NativeModule
    );
    assert_eq!(
        serde_json::from_str::<Target>("\"reference_backend\"").unwrap(),
        Target::ReferenceBackend
    );
}

#[test]
fn target_deserialization_fails_closed_on_unknown() {
    let err = serde_json::from_str::<Target>("\"unknown_backend\"").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unsupported target"));
    assert!(msg.contains("Fix:"));
}
