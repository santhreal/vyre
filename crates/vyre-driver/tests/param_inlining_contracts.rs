//! Contracts for `vyre_driver::param_inlining`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::param_inlining::{
    decide_param_inlining, ParamInliningDecision, ParamInliningPolicy,
};

#[test]
fn small_aligned_payload_inlines_under_large_inline_default() {
    let policy = ParamInliningPolicy::large_inline_default();
    let decision = decide_param_inlining(64, policy);
    assert_eq!(decision, ParamInliningDecision::Inline { padded_bytes: 64 });
    assert!(decision.is_inline());
}

#[test]
fn payload_at_inline_ceiling_still_inlines() {
    let policy = ParamInliningPolicy::large_inline_default();
    // 3 KiB is exactly the budget; must inline.
    let decision = decide_param_inlining(3 * 1024, policy);
    assert_eq!(
        decision,
        ParamInliningDecision::Inline {
            padded_bytes: 3 * 1024
        }
    );
}

#[test]
fn payload_above_inline_ceiling_falls_back_to_uniform() {
    let policy = ParamInliningPolicy::large_inline_default();
    let decision = decide_param_inlining(3 * 1024 + 1, policy);
    // 3073 -> padded to 3076 -> > 3072 -> UniformBuffer.
    assert_eq!(decision, ParamInliningDecision::UniformBuffer);
}

#[test]
fn unaligned_payload_pads_when_allowed() {
    let policy = ParamInliningPolicy::large_inline_default();
    // 17 -> pad to 20 (next multiple of 4).
    let decision = decide_param_inlining(17, policy);
    assert_eq!(decision, ParamInliningDecision::Inline { padded_bytes: 20 });
}

#[test]
fn unaligned_payload_falls_back_when_padding_disallowed() {
    let policy = ParamInliningPolicy {
        max_inline_bytes: 64,
        align_bytes: 4,
        allow_padding_to_align: false,
    };
    let decision = decide_param_inlining(17, policy);
    assert_eq!(decision, ParamInliningDecision::UniformBuffer);
}

#[test]
fn padded_size_must_also_fit_under_ceiling() {
    let policy = ParamInliningPolicy {
        max_inline_bytes: 16,
        align_bytes: 8,
        allow_padding_to_align: true,
    };
    // 13 -> pad to 16 -> exactly fits.
    assert_eq!(
        decide_param_inlining(13, policy),
        ParamInliningDecision::Inline { padded_bytes: 16 }
    );
    // 17 -> pad to 24 -> exceeds 16.
    assert_eq!(
        decide_param_inlining(17, policy),
        ParamInliningDecision::UniformBuffer
    );
}

#[test]
fn disabled_policy_always_uses_uniform() {
    let policy = ParamInliningPolicy::disabled();
    assert_eq!(
        decide_param_inlining(0, policy),
        ParamInliningDecision::UniformBuffer
    );
    assert_eq!(
        decide_param_inlining(8, policy),
        ParamInliningDecision::UniformBuffer
    );
    assert_eq!(
        decide_param_inlining(1024, policy),
        ParamInliningDecision::UniformBuffer
    );
}

#[test]
fn small_inline_default_inlines_tiny_payloads_only() {
    let policy = ParamInliningPolicy::small_inline_default();
    // 64-byte payload fits the conservative 128-byte small-inline default.
    assert_eq!(
        decide_param_inlining(64, policy),
        ParamInliningDecision::Inline { padded_bytes: 64 }
    );
    // 256 bytes exceeds the conservative small-inline default.
    assert_eq!(
        decide_param_inlining(256, policy),
        ParamInliningDecision::UniformBuffer
    );
}

#[test]
fn zero_byte_payload_inlines_with_zero_padded_bytes() {
    let policy = ParamInliningPolicy::large_inline_default();
    // Zero-byte payloads are degenerate but must take the inline path
    // because there's literally nothing to upload  -  uniform buffer
    // for zero bytes is wasteful.
    assert_eq!(
        decide_param_inlining(0, policy),
        ParamInliningDecision::Inline { padded_bytes: 0 }
    );
}

#[test]
fn zero_align_policy_falls_back_safely() {
    // Defensive: a zero alignment policy must not crash; falls back
    // to uniform buffer instead of attempting unsound packing.
    let policy = ParamInliningPolicy {
        max_inline_bytes: 1024,
        align_bytes: 0,
        allow_padding_to_align: true,
    };
    assert_eq!(
        decide_param_inlining(64, policy),
        ParamInliningDecision::UniformBuffer
    );
}

#[test]
fn adversarial_padding_overflow_cannot_inline() {
    let policy = ParamInliningPolicy {
        max_inline_bytes: u32::MAX,
        align_bytes: 256,
        allow_padding_to_align: true,
    };
    assert_eq!(
        decide_param_inlining(u32::MAX - 1, policy),
        ParamInliningDecision::UniformBuffer
    );
}
