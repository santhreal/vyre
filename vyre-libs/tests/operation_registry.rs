//! Canonical semantic operation registry coverage for library compositions.

use vyre_foundation::operation::{OperationRegistry, OperationTier};

#[test]
fn library_fixtures_are_canonical_semantic_registrations() {
    let entries: Vec<_> = vyre_libs::operation_catalog::all_entries().collect();
    let registered_library_ids: Vec<_> = OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Library)
        .map(|entry| entry.id)
        .collect();
    let catalog_ids: Vec<_> = entries.iter().map(|entry| entry.id).collect();
    assert_eq!(
        catalog_ids, registered_library_ids,
        "Fix: the library operation view must include every canonical library registration exactly once"
    );
    assert!(
        !entries.is_empty(),
        "Fix: linked library features must register at least one operation"
    );

    for entry in entries {
        assert_eq!(entry.tier, OperationTier::Library, "{}", entry.id);
        assert_eq!(
            OperationRegistry::global()
                .get(entry.id)
                .expect("Fix: fixture view must resolve through the canonical registry")
                .id,
            entry.id
        );
        let program = entry
            .program()
            .expect("Fix: library compositions must provide a neutral Program builder");
        assert_eq!(program.entry_op_id(), Some(entry.id), "{}", entry.id);
    }

    let tolerances = [
        ("vyre-libs::nn::softmax", 1),
        ("vyre-libs::nn::attention", 4),
        ("vyre-libs::nn::gqa_attention", 4),
        ("vyre-libs::nn::layer_norm", 1),
        ("vyre-libs::nn::silu", 1),
        ("vyre-libs::nn::logit_softcap", 2),
        ("vyre-libs::nn::rms_norm", 2),
        ("vyre-libs::nn::rms_norm_linear", 2),
        ("vyre-libs::math::fft::fft_convolve_circular_complex", 4),
        ("vyre-libs::math::linalg::matmul_strassen_2x2", 32),
        ("vyre-libs::optim::newton_schulz_5step", 64),
        ("vyre-libs::optim::ema_apply", 1),
        ("vyre-libs::optim::muoneq_r", 8),
    ];
    for (id, expected) in tolerances {
        assert_eq!(
            OperationRegistry::global()
                .get(id)
                .expect("Fix: tolerance owner must be registered")
                .tolerance(),
            expected,
            "{id}"
        );
    }
    assert!(OperationRegistry::global()
        .get("unknown-operation")
        .is_none());
}

/// Every linked registration is usable through the registry alone.
///
/// WHY: these three assertions lived in `vyre-test-support`, whose own binary
/// links nothing that registers, so they ran against an empty registry and the
/// two tests that called them could not pass. They belong wherever the
/// registrations are, and the roster is the registry itself rather than a
/// hardcoded floor, so a dialect added tomorrow is checked without an edit.
#[test]
fn every_registration_carries_a_version_and_a_way_to_build_itself() {
    let mut checked = 0usize;
    for entry in OperationRegistry::global().iter() {
        assert!(!entry.id.is_empty(), "Fix: a registration has an empty id.");
        assert!(
            entry.semantic_version > 0,
            "Fix: operation `{}` registers semantic_version 0; version a registration from 1.",
            entry.id
        );
        assert!(
            entry.build.is_some() || entry.signature.is_some(),
            "Fix: operation `{}` registers neither a builder nor a signature, so nothing can be done with it through the registry.",
            entry.id
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "Fix: the registry is empty in a binary that links vyre-libs, so inventory submissions are not reaching the link."
    );
}
