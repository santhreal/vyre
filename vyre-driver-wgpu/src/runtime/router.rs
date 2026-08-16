//! Backend auto-picker.
//!
//! `BackendRouter` walks `inventory::iter::<BackendRegistration>`,
//! filters out registered backends that cannot dispatch or are CPU
//! reference oracles, and picks the best executable GPU backend available
//! by precedence. Override via `VYRE_BACKEND=<id>`. The router is
//! intentionally stateless: backend precedence lives in inventory
//! registrations and adapter-specific persistence belongs to the backend
//! cache layer, not routing.
//!
//! Precedence (high → low):
//!
//! 1. `VYRE_BACKEND=<id>`  -  if set and the backend is registered,
//!    wins only when the backend is registered, executable, and GPU-backed.
//! 2. `cuda`  -  when an NVIDIA/CUDA backend is linked, registered, and executable.
//! 3. `wgpu`  -  portable GPU backend after CUDA.
//! 4. `spirv`  -  when the SPIR-V backend is registered.
//!
//! `BackendRouter::pick()` returns the selected backend id on success,
//! or a structured `BackendError` when no executable backend is linked.

use std::env;

use vyre_driver::{backend_dispatches, registered_backends_by_precedence_slice};
use vyre_driver::{BackendError, BackendRegistration};
use vyre_foundation::ir::Program;

/// How to source the forced-backend override.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Override<'a> {
    /// Read `VYRE_BACKEND` from the process environment.
    FromEnv,
    /// Use the explicit override regardless of environment.
    Explicit(&'a str),
    /// No override  -  router runs on precedence alone.
    None,
}

const OVERRIDE_ENV: &str = "VYRE_BACKEND";

/// Routing decision produced by the backend auto-picker.
#[derive(Debug, Clone)]
pub struct RouterDecision {
    /// The selected backend id.
    pub backend: &'static str,
    /// Reason the decision fell to this backend.
    pub reason: Reason,
}

/// How the decision was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Reason {
    /// `VYRE_BACKEND=<id>` forced the selection.
    EnvOverride,
    /// Highest-precedence registered backend that covers the
    /// Program's dialects.
    Precedence,
}

/// Backend auto-picker.
///
/// Constructed with [`BackendRouter::new`]; queries the runtime
/// inventory on demand so newly-registered backends participate
/// without router rebuild.
#[derive(Default)]
pub struct BackendRouter;

impl BackendRouter {
    /// New router.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Pick the best-available backend for `_program`.
    ///
    /// # Errors
    ///
    /// Returns `BackendError` when:
    ///
    /// * `VYRE_BACKEND` is set to a backend id that is not
    ///   registered.
    /// * No executable registered GPU backend is found. Vyre reports this as
    ///   a linkage or driver-visibility error instead of routing into
    ///   reference evaluation.
    pub fn pick(&self, program: &Program) -> Result<RouterDecision, BackendError> {
        self.pick_with_override(program, Override::FromEnv)
    }

    /// Pick with an explicit override source  -  the testable form of
    /// [`pick`](Self::pick).
    ///
    /// # Errors
    ///
    /// Same conditions as [`pick`](Self::pick).
    pub fn pick_with_override(
        &self,
        _program: &Program,
        source: Override<'_>,
    ) -> Result<RouterDecision, BackendError> {
        let registered = vyre_driver::registered_backends()?;

        let forced: Option<String> = match source {
            Override::FromEnv => env::var(OVERRIDE_ENV).ok(),
            Override::Explicit(s) => Some(s.to_owned()),
            Override::None => None,
        };
        if let Some(forced) = forced {
            let forced = forced.trim();
            if !forced.is_empty() {
                let hit = registered.iter().find_map(|registration| {
                    (registration.id == forced && !registration.reference_oracle)
                        .then_some(registration)
                });
                let hit = match hit {
                    Some(registration) if backend_dispatches(registration.id)? => {
                        Some(registration)
                    }
                    _ => None,
                };
                return match hit {
                    Some(reg) => Ok(RouterDecision {
                        backend: reg.id,
                        reason: Reason::EnvOverride,
                    }),
                    None => Err(BackendError::new(format!(
                        "VYRE_BACKEND={forced} is not an executable registered GPU backend. Fix: link CUDA/WGPU or unset VYRE_BACKEND; cpu-ref/reference are explicit conformance oracles, not runtime router targets."
                    ))),
                };
            }
        }

        // V7-EXT-021: precedence comes from the BackendPrecedence inventory
        // submitted by each backend crate, not a hardcoded driver-side table.
        // Walk backends in precedence order and return the first hit.
        for registration in registered_backends_by_precedence_slice()? {
            if registered
                .iter()
                .any(|candidate| candidate.id == registration.id)
                && backend_dispatches(registration.id)?
                && !registration.reference_oracle
            {
                return Ok(RouterDecision {
                    backend: registration.id,
                    reason: Reason::Precedence,
                });
            }
        }

        Err(BackendError::new(
            "no executable GPU backend is registered. Fix: link vyre-driver-cuda or vyre-driver-wgpu into the binary and verify the GPU driver probe succeeds.",
        ))
    }

    /// Enumerate every registered backend in precedence order. Inventory-driven
    /// per V7-EXT-021  -  backends without a submitted `BackendPrecedence`
    /// trail every backend that has one (rank `u32::MAX`).
    /// # Errors
    ///
    /// Returns the validated registry startup error when providers conflict.
    pub fn enumerate_by_precedence() -> Result<Vec<&'static BackendRegistration>, BackendError> {
        Ok(registered_backends_by_precedence_slice()?.to_vec())
    }
}
