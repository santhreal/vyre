//! Contract tests for `vyre_driver::Target` contract and fail-closed deserialization.

use vyre_driver::Target;

#[test]
fn target_variants_and_aot_target_ids() {
    assert_eq!(Target::PrimaryText.aot_target_id(), "primary_text");
    assert_eq!(Target::PrimaryBinary.aot_target_id(), "primary_binary");
    assert_eq!(Target::SecondaryText.aot_target_id(), "secondary_text");
    assert_eq!(Target::SecondaryBinary.aot_target_id(), "secondary_binary");
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
        Target::PrimaryText,
        Target::PrimaryBinary,
        Target::SecondaryText,
        Target::SecondaryBinary,
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
        serde_json::to_string(&Target::PrimaryText).unwrap(),
        r#""primary_text""#
    );
    assert_eq!(
        serde_json::to_string(&Target::PrimaryBinary).unwrap(),
        r#""primary_binary""#
    );
    assert_eq!(
        serde_json::to_string(&Target::SecondaryText).unwrap(),
        r#""secondary_text""#
    );
    assert_eq!(
        serde_json::to_string(&Target::SecondaryBinary).unwrap(),
        r#""secondary_binary""#
    );
    assert_eq!(
        serde_json::to_string(&Target::NativeModule).unwrap(),
        r#""native_module""#
    );
    assert_eq!(
        serde_json::to_string(&Target::ReferenceBackend).unwrap(),
        r#""reference_backend""#
    );
    assert_eq!(
        serde_json::to_string(&Target::Extension("custom")).unwrap(),
        r#""custom""#
    );
}

#[test]
fn target_deserialization_known_variants() {
    assert_eq!(
        serde_json::from_str::<Target>(r#""primary_text""#).unwrap(),
        Target::PrimaryText
    );
    assert_eq!(
        serde_json::from_str::<Target>(r#""primary_binary""#).unwrap(),
        Target::PrimaryBinary
    );
    assert_eq!(
        serde_json::from_str::<Target>(r#""secondary_text""#).unwrap(),
        Target::SecondaryText
    );
    assert_eq!(
        serde_json::from_str::<Target>(r#""secondary_binary""#).unwrap(),
        Target::SecondaryBinary
    );
    assert_eq!(
        serde_json::from_str::<Target>(r#""native_module""#).unwrap(),
        Target::NativeModule
    );
    assert_eq!(
        serde_json::from_str::<Target>(r#""reference_backend""#).unwrap(),
        Target::ReferenceBackend
    );
}

#[test]
fn target_deserialization_fails_closed_on_unknown() {
    let err = serde_json::from_str::<Target>(r#""unknown_backend""#).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unsupported target"));
    assert!(msg.contains("Fix:"));
}
