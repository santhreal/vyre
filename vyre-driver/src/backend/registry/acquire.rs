//! Backend selection and acquisition policy.

use std::collections::HashSet;
use std::time::Instant;
use vyre_foundation::ir::OpId;

use super::inventory_streams::{
    registered_backend, registered_backend_dispatches, registered_backend_precedence,
    registered_backends, BackendRegistration,
};
use crate::allocation::reserve_vec_to_capacity;
use crate::backend::{default_supported_ops, BackendError, VyreBackend};
use crate::{DeviceProfile, DeviceTimingQuality};

fn backend_dispatches_result(id: &str) -> Result<bool, BackendError> {
    registered_backend_dispatches(id)
}

/// Return whether the named backend submitted `dispatches: true`.
///
/// # Errors
///
/// Returns the validated registry startup error when providers conflict.
pub fn backend_dispatches(id: &str) -> Result<bool, BackendError> {
    backend_dispatches_result(id)
}

fn backend_precedence_result(id: &str) -> Result<u32, BackendError> {
    registered_backend_precedence(id)
}

/// Look up a backend's submitted precedence. Returns `u32::MAX` for backends
/// that did not submit a `BackendPrecedence` entry.
///
/// # Errors
///
/// Returns the validated registry startup error when providers conflict.
pub fn backend_precedence(id: &str) -> Result<u32, BackendError> {
    backend_precedence_result(id)
}

fn registered_backends_by_precedence_slice_result(
) -> Result<&'static [&'static BackendRegistration], BackendError> {
    static SORTED: std::sync::LazyLock<Result<Box<[&'static BackendRegistration]>, BackendError>> =
        std::sync::LazyLock::new(|| {
            let registrations = registered_backends()?;
            let mut keyed = Vec::new();
            reserve_vec_to_capacity(
                &mut keyed,
                registrations.len(),
                "Vyre backend registry",
                "precedence key slot",
                "reduce linked backend inventory or split registry initialization",
            )?;
            for registration in registrations {
                keyed.push((
                    backend_precedence_result(registration.id)?,
                    registration.id,
                    registration,
                ));
            }
            keyed.sort_unstable_by(|left, right| {
                left.0.cmp(&right.0).then_with(|| left.1.cmp(right.1))
            });
            Ok(keyed
                .into_iter()
                .map(|(_, _, registration)| registration)
                .collect())
        });
    match &*SORTED {
        Ok(sorted) => Ok(sorted),
        Err(error) => Err(error.clone()),
    }
}

/// Return every registered backend sorted by precedence (low rank first).
#[must_use]
pub fn registered_backends_by_precedence_slice(
) -> Result<&'static [&'static BackendRegistration], BackendError> {
    registered_backends_by_precedence_slice_result()
}

/// Return every registered backend sorted by precedence (low rank first).
/// Prefer [`registered_backends_by_precedence_slice`] on hot paths.
#[must_use]
pub fn registered_backends_by_precedence() -> Result<Vec<&'static BackendRegistration>, BackendError>
{
    Ok(registered_backends_by_precedence_slice()?.to_vec())
}

fn registration_for_id(id: &str) -> Result<Option<&'static BackendRegistration>, BackendError> {
    registered_backend(id)
}
/// Resolve the immutable registration for one linked backend identifier.
///
/// # Errors
///
/// Returns [`BackendError`] when registry initialization fails or no linked
/// backend registered `id`.
pub fn backend_registration(id: &str) -> Result<&'static BackendRegistration, BackendError> {
    registration_for_id(id)?.ok_or_else(|| {
        BackendError::new(format!(
            "backend `{id}` is not linked into this binary. Fix: link the concrete driver crate that registers this backend or choose one of the registered backend ids."
        ))
    })
}

/// Construct the registered backend with the requested stable identifier.
///
/// # Errors
///
/// Returns [`BackendError`] when no linked backend registered `id`, or when
/// the selected backend factory cannot initialize on this host.
pub fn acquire(id: &str) -> Result<Box<dyn VyreBackend>, BackendError> {
    let Some(registration) = registration_for_id(id)? else {
        return Err(BackendError::new(format!(
            "backend `{id}` is not linked into this binary. Fix: link the concrete driver crate that registers this backend or choose one of the registered backend ids."
        )));
    };
    registration.acquire()
}

/// Construct the highest-precedence linked backend that declares live dispatch.
/// The preferred runtime path is GPU-only: CPU reference backends remain
/// available through [`acquire`] for explicit conformance/oracle use, but are
/// never selected implicitly.
///
/// # Errors
///
/// Returns [`BackendError`] when no dispatch-capable backend is linked or every
/// matching backend factory fails on this host.
pub fn acquire_preferred_dispatch_backend() -> Result<Box<dyn VyreBackend>, BackendError> {
    let registrations = registered_backends_by_precedence_slice_result()?;
    let mut failures = Vec::new();
    reserve_vec_to_capacity(
        &mut failures,
        registrations.len(),
        "preferred backend acquisition",
        "failure detail slot",
        "reduce linked backend inventory or request a backend by id",
    )?;
    let mut skipped_reference_oracles = Vec::new();
    reserve_vec_to_capacity(
        &mut skipped_reference_oracles,
        registrations.len(),
        "preferred backend acquisition",
        "reference-skip slot",
        "reduce linked backend inventory or request a backend by id",
    )?;
    let mut candidates = Vec::new();
    reserve_vec_to_capacity(
        &mut candidates,
        registrations.len(),
        "preferred backend acquisition",
        "selection candidate slot",
        "reduce linked backend inventory or request a backend by id",
    )?;
    for registration in registrations {
        if !backend_dispatches_result(registration.id)? {
            continue;
        }
        if registration.reference_oracle {
            skipped_reference_oracles.push(registration.id);
            continue;
        }
        let started = Instant::now();
        match registration.acquire() {
            Ok(backend) => {
                let profile = backend.device_profile();
                let facts = BackendSelectionFacts {
                    id: registration.id,
                    precedence: backend_precedence_result(registration.id)?,
                    capability_score: backend_capability_score(profile),
                    timing_quality_score: timing_quality_score(profile.timing_quality),
                    memory_bandwidth_gbps: profile.mem_bw_gbps,
                    acquisition_ns: elapsed_ns_saturating(started),
                };
                candidates.push(PreferredBackendCandidate { facts, backend });
            }
            Err(error) => {
                tracing::trace!(
                    "acquire_preferred_dispatch_backend: failed to initialize backend `{}`: {}",
                    registration.id,
                    error
                );
                failures.push(format!("{}: {error}", registration.id))
            }
        }
    }
    if !candidates.is_empty() {
        candidates.sort_unstable_by(|left, right| {
            compare_backend_selection_facts(left.facts, right.facts)
        });
        let selected = candidates.remove(0);
        tracing::trace!(
            "acquire_preferred_dispatch_backend: selected backend `{}` with capability_score={} timing_quality_score={} precedence={} acquisition_ns={}",
            selected.facts.id,
            selected.facts.capability_score,
            selected.facts.timing_quality_score,
            selected.facts.precedence,
            selected.facts.acquisition_ns
        );
        return Ok(selected.backend);
    }
    let detail = if !failures.is_empty() {
        failures.join("; ")
    } else if !skipped_reference_oracles.is_empty() {
        format!(
            "only reference oracle backend(s) were available: {}",
            skipped_reference_oracles.join(", ")
        )
    } else {
        "no dispatch-capable backend is linked into this binary".to_string()
    };
    Err(BackendError::new(format!(
        "no usable GPU dispatch backend is available ({detail}). Fix: link a dispatch-capable GPU backend driver crate and repair the GPU driver probe; the CPU reference backend is explicit conformance-oracle infrastructure only."
    )))
}

struct PreferredBackendCandidate {
    facts: BackendSelectionFacts,
    backend: Box<dyn VyreBackend>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BackendSelectionFacts {
    id: &'static str,
    precedence: u32,
    capability_score: u32,
    timing_quality_score: u32,
    memory_bandwidth_gbps: u32,
    acquisition_ns: u64,
}

fn compare_backend_selection_facts(
    left: BackendSelectionFacts,
    right: BackendSelectionFacts,
) -> std::cmp::Ordering {
    right
        .capability_score
        .cmp(&left.capability_score)
        .then_with(|| right.timing_quality_score.cmp(&left.timing_quality_score))
        .then_with(|| right.memory_bandwidth_gbps.cmp(&left.memory_bandwidth_gbps))
        .then_with(|| left.precedence.cmp(&right.precedence))
        .then_with(|| left.acquisition_ns.cmp(&right.acquisition_ns))
        .then_with(|| left.id.cmp(right.id))
}

fn backend_capability_score(profile: DeviceProfile) -> u32 {
    let mut score = 0_u32;
    if profile.max_storage_buffer_binding_size > 0 {
        score += 16;
    }
    if profile.max_invocations_per_workgroup >= 64 {
        score += 8;
    }
    if profile.max_invocations_per_workgroup >= 256 {
        score += 4;
    }
    if profile.subgroup_size > 0 {
        score += 8;
    }
    if profile.supports_subgroup_ops {
        score += 8;
    }
    if profile.has_subgroup_shuffle {
        score += 4;
    }
    if profile.has_shared_memory {
        score += 4;
    }
    if profile.supports_f16 {
        score += 2;
    }
    if profile.supports_bf16 {
        score += 2;
    }
    if profile.supports_tensor_cores {
        score += 4;
    }
    if profile.supports_indirect_dispatch {
        score += 2;
    }
    if profile.supports_device_timestamps {
        score += 2;
    }
    if profile.supports_hardware_counters {
        score += 2;
    }
    score
}

fn timing_quality_score(quality: DeviceTimingQuality) -> u32 {
    match quality {
        DeviceTimingQuality::HostOnly => 0,
        DeviceTimingQuality::HostEnqueueWait => 1,
        DeviceTimingQuality::DeviceTimestamps => 2,
        DeviceTimingQuality::HardwareCounters => 3,
    }
}

fn elapsed_ns_saturating(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Core operation support set used by backends during migration.
#[must_use]
pub fn core_supported_ops() -> &'static HashSet<OpId> {
    default_supported_ops()
}

// Inline: covers `BackendSelectionFacts`, `backend_dispatches_result`, `backend_precedence_result`,
// `compare_backend_selection_facts` and 2 more items this module keeps private, which no
// integration test can name.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_backend_registry_queries_return_absent_without_panicking() {
        let missing_id = "__vyre_missing_backend__";

        match backend_dispatches_result(missing_id) {
            Ok(dispatches) => assert!(!dispatches),
            Err(error) => panic!("registry dispatch query failed: {error}"),
        }

        match backend_precedence_result(missing_id) {
            Ok(precedence) => assert_eq!(precedence, u32::MAX),
            Err(error) => panic!("registry precedence query failed: {error}"),
        }

        match registration_for_id(missing_id) {
            Ok(registration) => assert!(registration.is_none()),
            Err(error) => panic!("registry id query failed: {error}"),
        }
    }

    #[test]
    fn public_wrappers_return_absent_defaults_for_missing_backend() {
        let missing_id = "__vyre_missing_backend__";
        assert!(!backend_dispatches(missing_id).expect("valid backend registry"));
        assert_eq!(
            backend_precedence(missing_id).expect("valid backend registry"),
            u32::MAX
        );
    }

    #[test]
    fn preferred_selection_uses_typed_capability_before_precedence() {
        let weak_low_rank = BackendSelectionFacts {
            id: "low-rank",
            precedence: 1,
            capability_score: 1,
            timing_quality_score: 0,
            memory_bandwidth_gbps: 0,
            acquisition_ns: 10,
        };
        let strong_high_rank = BackendSelectionFacts {
            id: "high-rank",
            precedence: 50,
            capability_score: 2,
            timing_quality_score: 0,
            memory_bandwidth_gbps: 0,
            acquisition_ns: 10,
        };

        assert_eq!(
            compare_backend_selection_facts(strong_high_rank, weak_low_rank),
            std::cmp::Ordering::Less,
            "Fix: implicit backend selection must prefer typed capability evidence before falling back to BackendPrecedence rank."
        );
    }

    #[test]
    fn preferred_selection_uses_measured_acquisition_cost_as_tie_breaker() {
        let faster = BackendSelectionFacts {
            id: "faster",
            precedence: 10,
            capability_score: 8,
            timing_quality_score: 1,
            memory_bandwidth_gbps: 100,
            acquisition_ns: 5,
        };
        let slower = BackendSelectionFacts {
            id: "slower",
            acquisition_ns: 50,
            ..faster
        };

        assert_eq!(
            compare_backend_selection_facts(faster, slower),
            std::cmp::Ordering::Less,
            "Fix: implicit backend selection must use measured acquisition cost once typed capability and precedence facts tie."
        );
    }

    #[test]
    fn preferred_selection_keeps_precedence_as_tie_breaker() {
        let higher_precedence = BackendSelectionFacts {
            id: "rank-a",
            precedence: 5,
            capability_score: 8,
            timing_quality_score: 1,
            memory_bandwidth_gbps: 100,
            acquisition_ns: 5,
        };
        let lower_precedence = BackendSelectionFacts {
            id: "rank-b",
            precedence: 10,
            ..higher_precedence
        };

        assert_eq!(
            compare_backend_selection_facts(higher_precedence, lower_precedence),
            std::cmp::Ordering::Less,
            "Fix: BackendPrecedence must remain the deterministic tie-breaker after typed capability facts."
        );
    }
}
