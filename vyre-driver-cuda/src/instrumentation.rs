//! Cached CUDA instrumentation environment controls.
//!
//! Dispatch and lowering paths are hot enough that optional diagnostics cannot
//! repeatedly query process environment variables. This module centralizes those
//! controls behind one-time reads.

use std::sync::OnceLock;

pub(crate) const CUDA_CANONICAL_PREEMIT_ENV: &str = "VYRE_CUDA_CANONICAL_PREEMIT";
pub(crate) const CUDA_DESCRIPTOR_REWRITES_ENV: &str = "VYRE_CUDA_DESCRIPTOR_REWRITES";

pub(crate) fn cuda_stage_trace_enabled() -> bool {
    cached_flag("VYRE_CUDA_STAGE_TRACE", &CUDA_STAGE_TRACE)
}

pub(crate) fn cuda_profiler_ranges_enabled() -> bool {
    cached_flag("VYRE_CUDA_NVTX_RANGES", &CUDA_NVTX_RANGES)
        || cached_flag("VYRE_CUDA_PROFILE_RANGES", &CUDA_PROFILE_RANGES)
}

pub(crate) fn cuda_descriptor_audit_enabled() -> bool {
    cached_flag("VYRE_CUDA_DESCRIPTOR_AUDIT", &CUDA_DESCRIPTOR_AUDIT)
}

pub(crate) fn cuda_canonical_preemit_enabled() -> bool {
    cached_enabled_default_true(CUDA_CANONICAL_PREEMIT_ENV, &CUDA_CANONICAL_PREEMIT_DISABLED)
}

pub(crate) fn cuda_descriptor_rewrites_enabled() -> bool {
    cached_enabled_default_true(
        CUDA_DESCRIPTOR_REWRITES_ENV,
        &CUDA_DESCRIPTOR_REWRITES_DISABLED,
    )
}

pub(crate) fn cuda_graph_replay_enabled() -> bool {
    cached_enabled_default_true("VYRE_CUDA_GRAPH_REPLAY", &CUDA_GRAPH_REPLAY_DISABLED)
}

pub(crate) fn cuda_dispatch_validation_enabled() -> bool {
    cached_enabled_default_true(
        "VYRE_CUDA_VALIDATE_DISPATCH",
        &CUDA_VALIDATE_DISPATCH_DISABLED,
    )
}

pub(crate) fn cuda_resident_borrowed_fallback_enabled() -> bool {
    cached_flag(
        "VYRE_CUDA_RESIDENT_BORROWED_FALLBACK",
        &CUDA_RESIDENT_BORROWED_FALLBACK,
    )
}

pub(crate) fn cuda_resident_sync_before_launch_enabled() -> bool {
    cached_flag(
        "VYRE_CUDA_RESIDENT_SYNC_BEFORE_LAUNCH",
        &CUDA_RESIDENT_SYNC_BEFORE_LAUNCH,
    )
}

static CUDA_STAGE_TRACE: OnceLock<bool> = OnceLock::new();
static CUDA_NVTX_RANGES: OnceLock<bool> = OnceLock::new();
static CUDA_PROFILE_RANGES: OnceLock<bool> = OnceLock::new();
static CUDA_DESCRIPTOR_AUDIT: OnceLock<bool> = OnceLock::new();
static CUDA_CANONICAL_PREEMIT_DISABLED: OnceLock<bool> = OnceLock::new();
static CUDA_DESCRIPTOR_REWRITES_DISABLED: OnceLock<bool> = OnceLock::new();
static CUDA_GRAPH_REPLAY_DISABLED: OnceLock<bool> = OnceLock::new();
static CUDA_VALIDATE_DISPATCH_DISABLED: OnceLock<bool> = OnceLock::new();
static CUDA_RESIDENT_BORROWED_FALLBACK: OnceLock<bool> = OnceLock::new();
static CUDA_RESIDENT_SYNC_BEFORE_LAUNCH: OnceLock<bool> = OnceLock::new();

fn cached_flag(name: &'static str, slot: &OnceLock<bool>) -> bool {
    *slot.get_or_init(|| std::env::var_os(name).is_some())
}

fn cached_enabled_default_true(name: &'static str, disabled_slot: &OnceLock<bool>) -> bool {
    !*disabled_slot.get_or_init(|| {
        std::env::var(name)
            .map(|value| env_value_disables_default_true(&value))
            .unwrap_or(false)
    })
}

fn env_value_disables_default_true(value: &str) -> bool {
    matches!(value, "0" | "false" | "FALSE" | "off" | "OFF")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_true_env_parser_accepts_only_explicit_false_values() {
        for value in ["0", "false", "FALSE", "off", "OFF"] {
            assert!(
                env_value_disables_default_true(value),
                "Fix: CUDA default-on instrumentation knob value `{value}` must disable the feature."
            );
        }
        for value in ["", "1", "true", "TRUE", "on", "ON", "False", "Off"] {
            assert!(
                !env_value_disables_default_true(value),
                "Fix: CUDA default-on instrumentation knob value `{value}` must not accidentally disable the feature."
            );
        }
    }

    #[test]
    fn cached_flags_are_stable_after_first_read() {
        let slot = OnceLock::new();
        assert!(!cached_flag("VYRE_CUDA_TEST_UNSET_FLAG", &slot));
        assert!(!cached_flag("VYRE_CUDA_TEST_UNSET_FLAG", &slot));
    }
}
