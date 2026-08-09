//! Canonical semantic operation registry coverage for library compositions.

use vyre_foundation::operation::{OperationRegistry, OperationTier};

#[test]
fn library_fixtures_are_canonical_semantic_registrations() {
    let entries: Vec<_> = vyre_libs::fixture_catalog::all_entries().collect();
    assert!(
        entries.len() >= 160,
        "Fix: every library fixture must submit one canonical operation registration"
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
