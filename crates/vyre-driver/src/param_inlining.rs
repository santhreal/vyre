//! D7 substrate: push-constant / tiny-param inlining policy.
//!
//! When a dispatch's per-launch params buffer is small enough, it can be
//! inlined into backend launch metadata instead of allocating a uniform
//! buffer, uploading bytes, binding, and synchronising. Avoiding that
//! 4-step path costs microseconds per launch on short kernels and is
//! pure win when the params fit.
//!
//! This module owns the *decision*: given a backend's inline budget and
//! a payload size, should the dispatcher inline? It does **not** own the
//! per-backend mechanics; those live in the concrete drivers and consume
//! this policy.

/// Per-backend inline-params policy. Built from live capability probes
/// so neutral runtime code can pick the inline path without knowing the
/// concrete backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamInliningPolicy {
    /// Maximum payload bytes the backend can accept inline. Set to `0`
    /// to disable inlining entirely and force the uniform-buffer path.
    pub max_inline_bytes: u32,
    /// Required alignment of the inline payload, in bytes. A payload
    /// whose size is not a multiple of `align_bytes` cannot be inlined
    /// unless padding is allowed.
    pub align_bytes: u32,
    /// Whether the policy permits the dispatcher to round payload size
    /// up to the next `align_bytes` multiple before inlining. When
    /// false, only naturally-aligned payloads inline; oversize after
    /// padding is still rejected.
    pub allow_padding_to_align: bool,
}

impl ParamInliningPolicy {
    /// Conservative large-inline default: 3 KiB inline budget,
    /// 4-byte alignment, padding allowed. Concrete drivers with larger
    /// native launch-metadata budgets should override this from live
    /// capability probes.
    #[must_use]
    pub const fn large_inline_default() -> Self {
        Self {
            max_inline_bytes: 3 * 1024,
            align_bytes: 4,
            allow_padding_to_align: true,
        }
    }

    /// Conservative small-inline default: 128 B inline budget,
    /// 4-byte alignment, padding allowed. Concrete drivers should
    /// override this from live capability probes when more inline
    /// launch metadata is available.
    #[must_use]
    pub const fn small_inline_default() -> Self {
        Self {
            max_inline_bytes: 128,
            align_bytes: 4,
            allow_padding_to_align: true,
        }
    }

    /// Construct a policy that disables inlining. Useful for backends
    /// whose probed limit is zero or for benchmark sweeps that need to
    /// exclude the inline path.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            max_inline_bytes: 0,
            align_bytes: 4,
            allow_padding_to_align: false,
        }
    }
}

/// Decision returned by [`decide_param_inlining`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamInliningDecision {
    /// Payload fits inline; dispatcher should pack it into launch args
    /// or push constants.
    Inline {
        /// Reserved size for the inlined payload (always `>= bytes_len`,
        /// possibly rounded up to `align_bytes`).
        padded_bytes: u32,
    },
    /// Payload does not fit; dispatcher must allocate a uniform buffer
    /// and bind it.
    UniformBuffer,
}

impl ParamInliningDecision {
    /// Whether this decision is the inline path (helper for predicates).
    #[must_use]
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline { .. })
    }
}

/// Decide how to deliver a `bytes_len`-byte param payload under
/// `policy`. Returns [`ParamInliningDecision::UniformBuffer`] when the
/// payload exceeds the inline budget (after optional padding) or when
/// inlining is disabled.
#[must_use]
pub fn decide_param_inlining(bytes_len: u32, policy: ParamInliningPolicy) -> ParamInliningDecision {
    if policy.max_inline_bytes == 0 {
        return ParamInliningDecision::UniformBuffer;
    }
    if policy.align_bytes == 0 {
        // Defensive  -  a zero alignment is meaningless; treat as
        // uniform-buffer-only to avoid undefined-behaviour packing.
        return ParamInliningDecision::UniformBuffer;
    }

    let needs_padding = bytes_len % policy.align_bytes != 0;
    let padded_bytes = if needs_padding {
        if !policy.allow_padding_to_align {
            return ParamInliningDecision::UniformBuffer;
        }
        // Round up to the next align_bytes multiple exactly. Overflow
        // cannot inline, because the padded payload is larger than any
        // representable backend inline budget.
        let remainder = bytes_len % policy.align_bytes;
        let padding = policy.align_bytes - remainder;
        let padded = u64::from(bytes_len) + u64::from(padding);
        if padded > u64::from(policy.max_inline_bytes) {
            return ParamInliningDecision::UniformBuffer;
        }
        padded as u32
    } else {
        bytes_len
    };

    if padded_bytes <= policy.max_inline_bytes {
        ParamInliningDecision::Inline { padded_bytes }
    } else {
        ParamInliningDecision::UniformBuffer
    }
}
