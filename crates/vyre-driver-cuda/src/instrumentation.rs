//! Cached CUDA instrumentation environment controls.
//!
//! Dispatch and lowering paths are hot enough that optional diagnostics cannot
//! repeatedly query process environment variables. This module centralizes those
//! controls behind one-time reads.

use std::sync::LazyLock;
use std::time::Duration;

pub(crate) const CUDA_RESIDENT_BORROWED_FALLBACK_ENV: &str = "VYRE_CUDA_RESIDENT_BORROWED_FALLBACK";
pub(crate) const CUDA_ALLOW_BORROWED_FALLBACK_ENV: &str = "VYRE_CUDA_ALLOW_BORROWED_FALLBACK";

/// Environment control over the bounded device-wait ceiling, in whole seconds.
pub(crate) const CUDA_DEVICE_WAIT_TIMEOUT_ENV: &str = "VYRE_CUDA_DEVICE_WAIT_TIMEOUT_SECS";

/// Ceiling on one CUDA device wait when nothing overrides it.
///
/// Five minutes is far above any conformance, parity or benchmark dispatch in
/// this workspace, and short enough that a wait which will never finish is
/// reported instead of being mistaken for a long-running job.
const CUDA_DEVICE_WAIT_TIMEOUT_SECS_DEFAULT: u64 = 300;

/// Ceiling every bounded CUDA device wait is measured against.
pub(crate) fn cuda_device_wait_timeout() -> Duration {
    *CUDA_DEVICE_WAIT_TIMEOUT
}

static CUDA_DEVICE_WAIT_TIMEOUT: LazyLock<Duration> = LazyLock::new(|| {
    let Ok(value) = std::env::var(CUDA_DEVICE_WAIT_TIMEOUT_ENV) else {
        return Duration::from_secs(CUDA_DEVICE_WAIT_TIMEOUT_SECS_DEFAULT);
    };
    match parse_device_wait_timeout_secs(&value) {
        Some(seconds) => Duration::from_secs(seconds),
        None => {
            eprintln!(
                "Fix: {CUDA_DEVICE_WAIT_TIMEOUT_ENV}=`{value}` is not a positive whole number of seconds; the bounded CUDA device wait keeps its {CUDA_DEVICE_WAIT_TIMEOUT_SECS_DEFAULT}s ceiling."
            );
            Duration::from_secs(CUDA_DEVICE_WAIT_TIMEOUT_SECS_DEFAULT)
        }
    }
});

/// Whole positive seconds in `value`, or `None` for anything else.
fn parse_device_wait_timeout_secs(value: &str) -> Option<u64> {
    value
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|seconds| *seconds > 0)
}

pub(crate) fn cuda_stage_trace_enabled() -> bool {
    *CUDA_STAGE_TRACE
}

pub(crate) fn cuda_profiler_ranges_enabled() -> bool {
    *CUDA_PROFILER_RANGES
}

pub(crate) fn cuda_descriptor_audit_enabled() -> bool {
    *CUDA_DESCRIPTOR_AUDIT
}

pub(crate) fn cuda_graph_replay_enabled() -> bool {
    *CUDA_GRAPH_REPLAY
}

pub(crate) fn cuda_dispatch_validation_enabled() -> bool {
    *CUDA_VALIDATE_DISPATCH
}

pub(crate) fn cuda_resident_borrowed_fallback_enabled() -> bool {
    CUDA_RESIDENT_BORROWED_FALLBACK_POLICY.enabled
}

pub(crate) fn cuda_resident_sync_before_launch_enabled() -> bool {
    *CUDA_RESIDENT_SYNC_BEFORE_LAUNCH
}

static CUDA_STAGE_TRACE: LazyLock<bool> = LazyLock::new(|| env_flag("VYRE_CUDA_STAGE_TRACE"));
static CUDA_PROFILER_RANGES: LazyLock<bool> =
    LazyLock::new(|| env_flag("VYRE_CUDA_NVTX_RANGES") || env_flag("VYRE_CUDA_PROFILE_RANGES"));
static CUDA_DESCRIPTOR_AUDIT: LazyLock<bool> =
    LazyLock::new(|| env_flag("VYRE_CUDA_DESCRIPTOR_AUDIT"));
static CUDA_GRAPH_REPLAY: LazyLock<bool> =
    LazyLock::new(|| env_enabled_default_true("VYRE_CUDA_GRAPH_REPLAY"));
static CUDA_VALIDATE_DISPATCH: LazyLock<bool> =
    LazyLock::new(|| env_enabled_default_true("VYRE_CUDA_VALIDATE_DISPATCH"));
static CUDA_RESIDENT_BORROWED_FALLBACK_POLICY: LazyLock<ResidentBorrowedFallbackPolicy> =
    LazyLock::new(compute_resident_borrowed_fallback_policy);
static CUDA_RESIDENT_SYNC_BEFORE_LAUNCH: LazyLock<bool> =
    LazyLock::new(|| env_flag("VYRE_CUDA_RESIDENT_SYNC_BEFORE_LAUNCH"));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentBorrowedFallbackPolicy {
    enabled: bool,
}

fn compute_resident_borrowed_fallback_policy() -> ResidentBorrowedFallbackPolicy {
    if !env_flag(CUDA_RESIDENT_BORROWED_FALLBACK_ENV) {
        return ResidentBorrowedFallbackPolicy { enabled: false };
    }

    #[cfg(debug_assertions)]
    {
        ResidentBorrowedFallbackPolicy { enabled: true }
    }

    #[cfg(not(debug_assertions))]
    {
        if env_value_enables_explicit_true(
            &std::env::var(CUDA_ALLOW_BORROWED_FALLBACK_ENV).unwrap_or_default(),
        ) {
            eprintln!(
                "\n\
                 ================================================================================\n\
                 WARNING: VYRE_CUDA_ALLOW_BORROWED_FALLBACK=1\n\
                 CUDA resident dispatch is using the host-buffer BORROWED FALLBACK path.\n\
                 This path downloads resident buffers, runs dispatch_borrowed, and re-uploads.\n\
                 Release performance and parity evidence are INVALID while this escape hatch is on.\n\
                 Unset {CUDA_RESIDENT_BORROWED_FALLBACK_ENV} and {CUDA_ALLOW_BORROWED_FALLBACK_ENV} \
                 before collecting release metrics.\n\
                 ================================================================================\n"
            );
            return ResidentBorrowedFallbackPolicy { enabled: true };
        }

        eprintln!(
            "\n\
             ERROR: {CUDA_RESIDENT_BORROWED_FALLBACK_ENV} is set but resident borrowed fallback \
             is refused on release builds.\n\
             Fix: unset {CUDA_RESIDENT_BORROWED_FALLBACK_ENV} and use native CUDA resident dispatch, \
             or set {CUDA_ALLOW_BORROWED_FALLBACK_ENV}=1 only for local debugging (invalidates release perf).\n"
        );
        ResidentBorrowedFallbackPolicy { enabled: false }
    }
}

// The only production caller is the release-build refusal above: a debug build
// enables the fallback without consulting the explicit-true override.
#[cfg(any(not(debug_assertions), test))]
fn env_value_enables_explicit_true(value: &str) -> bool {
    matches!(value, "1" | "true" | "TRUE" | "on" | "ON")
}

fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn env_enabled_default_true(name: &str) -> bool {
    std::env::var(name).map_or(true, |value| !env_value_disables_default_true(&value))
}

fn env_value_disables_default_true(value: &str) -> bool {
    matches!(value, "0" | "false" | "FALSE" | "off" | "OFF")
}

// Inline: covers `env_value_disables_default_true`,
// `env_value_enables_explicit_true`, `parse_device_wait_timeout_secs`, which no integration test
// can name.
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
    fn device_wait_ceiling_accepts_only_positive_whole_seconds() {
        assert_eq!(parse_device_wait_timeout_secs("1"), Some(1));
        assert_eq!(parse_device_wait_timeout_secs(" 45 "), Some(45));
        assert_eq!(
            parse_device_wait_timeout_secs(&u64::MAX.to_string()),
            Some(u64::MAX)
        );
        // Zero would make every device wait fail on its first query, and a
        // negative, fractional or non-numeric value has no ceiling to apply, so
        // each keeps the compiled-in default rather than silently disabling the
        // bound.
        for value in ["0", "-1", "1.5", "", "  ", "300s", "many"] {
            assert_eq!(
                parse_device_wait_timeout_secs(value),
                None,
                "Fix: CUDA device-wait ceiling `{value}` must be refused instead of replacing the bound."
            );
        }
    }

    #[test]
    fn explicit_true_env_parser_accepts_only_release_escape_hatch_values() {
        for value in ["1", "true", "TRUE", "on", "ON"] {
            assert!(
                env_value_enables_explicit_true(value),
                "Fix: CUDA resident borrowed-fallback allow value `{value}` must enable the escape hatch."
            );
        }
        for value in ["", "0", "false", "FALSE", "off", "OFF", "yes"] {
            assert!(
                !env_value_enables_explicit_true(value),
                "Fix: CUDA resident borrowed-fallback allow value `{value}` must not accidentally enable the escape hatch."
            );
        }
    }
}
