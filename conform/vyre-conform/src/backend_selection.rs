//! Discovery and filtering of semantic-execution-capable registered backends.

use vyre_registry_link::backend::live_backend_registry;

pub(crate) fn backend_registration(
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
            format!(
                "unknown backend `{requested}`. Fix: link a concrete driver crate that registers this backend id."
            )
        })
}

pub(crate) fn semantic_execution_backends(
) -> Result<Vec<&'static vyre_driver::BackendRegistration>, String> {
    let registrations = live_backend_registry()
        .map_err(|error| format!("backend registry startup failed: {error}"))?;
    Ok(registrations
        .iter()
        .filter(|backend| supports_semantic_execution(backend))
        .collect())
}

fn supports_semantic_execution(backend: &vyre_driver::BackendRegistration) -> bool {
    !backend.reference_oracle && backend.target_compiler.is_some() && backend.materializer.is_some()
}

pub(crate) fn select_backends(
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
    if selected.is_empty() {
        let known = all_backends
            .iter()
            .map(|backend| backend.id)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "unknown or non-semantic backend `{filter}`. Fix: pass `--backend all` or one semantic-execution-capable backend id: {known}."
        ));
    }
    Ok(selected)
}
