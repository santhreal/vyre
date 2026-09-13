//! Discovery and filtering of semantic-execution-capable registered backends.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex};

use vyre_driver_reference::is_oracle_executor_id;
use vyre_foundation::failure_domain::reclaim_poisoned_mutex;
use vyre_registry_link::backend::live_backend_registry;

/// Find a linked backend registration by ID, or find the default backend if "auto".
///
/// # Errors
///
/// Returns an error message if the registry fails or the backend ID is not linked.
pub fn backend_registration(
    backend_id: &str,
) -> Result<&'static vyre_driver::BackendRegistration, String> {
    let registrations = live_backend_registry()
        .map_err(|error| format!("backend registry startup failed: {error}"))?;
    let requested = if backend_id == "auto" {
        let mut requested = None;
        for registration in registrations {
            if supports_semantic_execution(registration) {
                requested = Some(registration.id);
                break;
            }
        }
        requested.ok_or_else(|| {
            "no semantic-execution-capable backend is linked into this binary. Fix: link a concrete driver crate that registers compiler and materializer facets.".to_string()
        })?
    } else {
        backend_id
    };
    registrations
        .iter()
        .find(|registration| registration.id == requested)
        .ok_or_else(|| {
            let linked: Vec<&str> = registrations
                .iter()
                .map(|registration| registration.id)
                .collect();
            format!(
                "unknown backend `{requested}`. This binary registers {linked:?}. Fix: link the concrete driver crate that registers `{requested}`, or name one of the registered ids."
            )
        })
}

/// Enumerate all registered backends supporting semantic execution.
///
/// # Errors
///
/// Returns an error message if the registry fails to initialize.
pub fn semantic_execution_backends(
) -> Result<Vec<&'static vyre_driver::BackendRegistration>, String> {
    let registrations = live_backend_registry()
        .map_err(|error| format!("backend registry startup failed: {error}"))?;
    Ok(registrations
        .iter()
        .filter(|backend| supports_semantic_execution(backend))
        .collect())
}

fn supports_semantic_execution(backend: &vyre_driver::BackendRegistration) -> bool {
    admits_semantic_execution(
        backend.reference_oracle,
        backend.target_compiler.is_some(),
        backend.materializer.is_some(),
    )
}

/// A backend admits semantic execution when it registers both facets and is
/// not the conformance oracle.
///
/// The oracle is excluded whatever facets it registers. Proving it would
/// compare `vyre-reference` against itself, which certifies nothing, so the
/// exclusion cannot depend on the oracle happening to lack a facet today.
const fn admits_semantic_execution(
    reference_oracle: bool,
    has_target_compiler: bool,
    has_materializer: bool,
) -> bool {
    !reference_oracle && has_target_compiler && has_materializer
}

/// One registered backend this host cannot execute, and what refused it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnavailableBackend {
    /// Stable backend identifier.
    pub id: &'static str,
    /// The acquisition refusal, verbatim.
    pub reason: String,
}

/// Split `backends` into those this host can acquire and those it cannot,
/// keeping every acquisition open.
///
/// Registration is a property of the binary and availability is a property of
/// the host, and `--backend all` used to conflate them: the `gpu` feature links
/// the Metal and SPIR-V registrations on every platform so that naming one
/// gives a real refusal instead of `unknown backend`, and an `all` run on Linux
/// then proved Metal, wrote one failed pair per operation for a framework that
/// is not on the machine, and refused the certificate. 353 rows saying
/// `unsupported feature Apple Metal.framework native runtime` is not a defect
/// report, and it buried the rows that were.
///
/// Acquisition is the question, because a backend that cannot be acquired
/// cannot execute anything. The refusal is carried rather than dropped: a
/// certificate states which backends it covers and why it covers no more.
///
/// The live handles are returned rather than dropped because probing is not
/// free of consequence. A backend that brings up a vendor runtime tears it down
/// again when its handle drops, and unloading a Vulkan ICD out from under the
/// process-wide instance another backend already holds loses that instance's
/// devices: probe-then-drop turned every later wgpu route into `Parent device
/// is lost` on a host whose GPU was idle. The caller holds them until the run
/// ends.
#[must_use]
pub fn partition_by_host_availability(
    backends: &[&'static vyre_driver::BackendRegistration],
) -> (
    Vec<(
        &'static vyre_driver::BackendRegistration,
        Box<dyn vyre_driver::VyreBackend>,
    )>,
    Vec<UnavailableBackend>,
) {
    let mut live = Vec::with_capacity(backends.len());
    let mut unavailable = Vec::new();
    for &backend in backends {
        match backend.acquire() {
            Ok(handle) => live.push((backend, handle)),
            Err(error) => unavailable.push(UnavailableBackend {
                id: backend.id,
                reason: error.to_string(),
            }),
        }
    }
    (live, unavailable)
}

/// Registered target facts, the digest a receipt carries for them, and what
/// resolving them has cost.
struct RegisteredTargetFacts {
    facts: vyre_megakernel::DeviceFacts,
    digest: String,
    /// Acquisitions of this backend the cache has performed. One is the whole
    /// contract; the field exists so that is observable rather than inferred.
    acquisitions: u64,
}

/// Target facts per backend id, resolved on first use.
///
/// Compile facts are immutable for a registration: they describe the device the
/// artifact will run on, and the compiler selects against them. Reading them
/// still costs an acquisition, and the proof path asks for them per case and
/// twice more per case pair, so a 353-operation run acquired every backend more
/// than a thousand times to recompute one constant.
///
/// That is not free even where a driver shares its runtime. A wgpu acquisition
/// creates its own device, which is what lets device-loss recovery replace one
/// backend's device without disturbing another's, and sixty-four held at once
/// open 629 descriptors. Hundreds of them exhaust the process, and the first
/// thing to fail is whichever driver next asks the loader for a file.
static TARGET_FACTS: LazyLock<Mutex<BTreeMap<&'static str, Arc<RegisteredTargetFacts>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Read the registered target facts for `registration`, acquiring it at most
/// once per process.
///
/// # Errors
///
/// Returns the acquisition refusal when this host cannot acquire the backend.
pub fn registered_target_facts(
    registration: &'static vyre_driver::BackendRegistration,
) -> Result<vyre_megakernel::DeviceFacts, vyre_driver::BackendError> {
    Ok(registered_facts(registration)?.facts)
}

/// Digest the registered target facts for `backend_id`, or `None` when this
/// host has no such backend to ask.
///
/// A receipt carries this digest so a verifier can reject a pair whose two
/// sides compiled against different device facts.
#[must_use]
pub fn target_facts_digest(backend_id: &str) -> Option<String> {
    let registration = backend_registration(backend_id).ok()?;
    Some(registered_facts(registration).ok()?.digest.clone())
}

/// Acquisitions of `backend_id` this cache has performed since the process
/// started, or zero for a backend it has never resolved.
///
/// The count is the cache's contract stated as a number: proving a case reads
/// target facts, and reading them must not acquire a backend. A run that
/// acquires per case holds hundreds of acquisitions open across its workers and
/// exhausts the process's file descriptors, and the first thing to fail is
/// whichever driver next asks the loader for a file. Nothing in the run's
/// output distinguishes that from a host with no GPU.
#[must_use]
pub fn target_facts_acquisitions(backend_id: &str) -> u64 {
    reclaim_poisoned_mutex(
        &TARGET_FACTS,
        "the conformance target-facts cache",
        "the per-backend registered target facts",
    )
    .get(backend_id)
    .map_or(0, |entry| entry.acquisitions)
}

fn registered_facts(
    registration: &'static vyre_driver::BackendRegistration,
) -> Result<Arc<RegisteredTargetFacts>, vyre_driver::BackendError> {
    // A refusal is not cached. Acquisition failure is a property of the host at
    // that moment, and a backend that comes back has to be usable without
    // restarting the process.
    let mut cache = reclaim_poisoned_mutex(
        &TARGET_FACTS,
        "the conformance target-facts cache",
        "the per-backend registered target facts",
    );
    if let Some(cached) = cache.get(registration.id) {
        return Ok(Arc::clone(cached));
    }
    let acquisitions = cache
        .get(registration.id)
        .map_or(0, |entry| entry.acquisitions);
    let facts = registration.acquire()?.device_profile().compile_facts();
    let digest = blake3::hash(format!("{facts:?}").as_bytes())
        .to_hex()
        .to_string();
    let entry = Arc::new(RegisteredTargetFacts {
        facts,
        digest,
        acquisitions: acquisitions + 1,
    });
    cache.insert(registration.id, Arc::clone(&entry));
    Ok(entry)
}

/// Filter a list of backends by a selector string ("all", backend ID, or comma-separated list).
///
/// # Errors
///
/// Returns an error message if the filter matches no registered backend.
pub fn select_backends(
    all_backends: &[&'static vyre_driver::BackendRegistration],
    filter: &str,
) -> Result<Vec<&'static vyre_driver::BackendRegistration>, String> {
    if filter == "all" {
        return Ok(all_backends.to_vec());
    }
    let selected = all_backends
        .iter()
        .copied()
        .filter(|backend| backend.id == filter)
        .collect::<Vec<_>>();
    if !selected.is_empty() {
        return Ok(selected);
    }
    let known = all_backends
        .iter()
        .map(|backend| backend.id)
        .collect::<Vec<_>>()
        .join(", ");
    let fix =
        format!("Fix: pass `--backend all` or one semantic-execution-capable backend id: {known}.");
    if is_oracle_executor_id(filter) {
        return Err(format!(
            "`{filter}` is the reference oracle, not a backend, so proving against it would \
             certify the reference executor against itself. {fix}"
        ));
    }
    let registrations = live_backend_registry()
        .map_err(|error| format!("backend registry startup failed: {error}"))?;
    let Some(registration) = registrations
        .iter()
        .find(|registration| registration.id == filter)
    else {
        return Err(format!("unknown backend `{filter}`. {fix}"));
    };
    if registration.reference_oracle {
        return Err(format!(
            "the selected backend set only contains reference dispatch backends: `{filter}` is the \
             reference oracle, so proving against it would certify the reference executor against \
             itself. {fix}"
        ));
    }
    Err(format!(
        "backend `{filter}` registers no semantic execution facets, so it cannot execute a program \
         against vyre-reference. {fix}"
    ))
}

// Inline: covers the crate-private admission rule, which no integration test
// can reach and whose variant space no registered backend covers.
#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the refusal message a caller sees proves only which branch printed
    /// it. The class is every combination of the three facts the rule reads, and
    /// the one that matters is a reference oracle that registers both facets:
    /// today no registered oracle does, so a dropped flag check would leave
    /// every message-level test green while making the oracle certifiable.
    #[test]
    fn only_a_non_oracle_with_both_facets_admits_semantic_execution() {
        for reference_oracle in [false, true] {
            for has_target_compiler in [false, true] {
                for has_materializer in [false, true] {
                    let admitted = admits_semantic_execution(
                        reference_oracle,
                        has_target_compiler,
                        has_materializer,
                    );
                    let expected = !reference_oracle && has_target_compiler && has_materializer;
                    assert_eq!(
                        admitted, expected,
                        "oracle={reference_oracle} compiler={has_target_compiler} \
                         materializer={has_materializer}"
                    );
                    if reference_oracle {
                        assert!(
                            !admitted,
                            "a reference oracle with compiler={has_target_compiler} and \
                             materializer={has_materializer} must never be admitted"
                        );
                    }
                }
            }
        }
    }

    /// WHY: the case above proves the rule, not that the tree has an oracle for
    /// it to exclude. Without a registered oracle the exclusion is unreachable,
    /// so this reads the live registry and requires that no entry claims to be
    /// one. The `reg.id != "cpu-ref"` half of the condition is gone: naming the
    /// id the interpreter used to register under judged one spelling, and the
    /// execution-domain closure in
    /// `vyre-driver-reference/tests/production_registry_execution_domain.rs`
    /// judges every entry.
    #[test]
    fn no_registered_backend_is_a_reference_oracle() {
        let registrations = live_backend_registry().expect("Fix: backend registry must start");
        for reg in registrations {
            assert!(
                !reg.reference_oracle,
                "Fix: backend `{}` is an oracle and must not appear in the VyreBackend registry",
                reg.id
            );
        }
    }
}
