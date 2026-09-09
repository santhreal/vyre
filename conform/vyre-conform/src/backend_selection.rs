//! Discovery and filtering of semantic-execution-capable registered backends.

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
    if filter == "cpu-ref" || filter == "reference" || filter == "oracle" {
        return Err(format!(
            "the selected backend set only contains reference dispatch backends: `{filter}` is the \
             reference oracle, so proving against it would certify the reference executor against \
             itself. {fix}"
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
    /// it to exclude. Without a registered oracle the exclusion is unreachable
    /// WHY: Row 87 requires that no reference oracle appears as a VyreBackend
    /// in the backend registry.
    #[test]
    fn no_registered_backend_is_a_reference_oracle() {
        let registrations = live_backend_registry().expect("Fix: backend registry must start");
        for reg in registrations {
            assert!(
                !reg.reference_oracle && reg.id != "cpu-ref",
                "Fix: backend `{}` is an oracle and must not appear in the VyreBackend registry",
                reg.id
            );
        }
    }
}
