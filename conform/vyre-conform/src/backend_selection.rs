//! Discovery and filtering of dispatch-capable registered backends.

use vyre_driver::backend_dispatches;
use vyre_registry_link::backend::live_backend_registry;

pub(crate) fn backend_registration(
    backend_id: &str,
) -> Result<&'static vyre_driver::BackendRegistration, String> {
    let registrations = live_backend_registry()
        .map_err(|error| format!("backend registry startup failed: {error}"))?;
    let requested = if backend_id == "auto" {
        let mut requested = None;
        for registration in registrations {
            if backend_dispatches(registration.id)
                .map_err(|error| format!("backend registry startup failed: {error}"))?
            {
                requested = Some(registration.id);
                break;
            }
        }
        requested.ok_or_else(|| {
            "no dispatch-capable backend is linked into this binary. Fix: link a concrete driver crate that registers compiler and materializer facets.".to_string()
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

pub(crate) fn dispatch_capable_backends(
) -> Result<Vec<&'static vyre_driver::BackendRegistration>, String> {
    let registrations = live_backend_registry()
        .map_err(|error| format!("backend registry startup failed: {error}"))?;
    let mut backends = Vec::new();
    for backend in registrations {
        if backend_dispatches(backend.id)
            .map_err(|error| format!("backend registry startup failed: {error}"))?
        {
            backends.push(backend);
        }
    }
    Ok(backends)
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
            "unknown or non-dispatch backend `{filter}`. Fix: pass `--backend all` or one dispatch-capable backend id: {known}."
        ));
    }
    Ok(selected)
}
