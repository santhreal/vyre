//! Inventory streams contributed by linked backend crates.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use vyre_foundation::ir::OpId;
use vyre_foundation::operation::{TargetId, TargetOperationFacet};

use super::grid_sync_split::wrap_grid_sync_split;
use crate::backend::{ArtifactMaterializer, BackendError, VyreBackend};
use vyre_megakernel::TargetCompiler;

struct RegisteredOperationSupport {
    id: &'static str,
    operations: &'static HashSet<OpId>,
}

impl crate::backend::Backend for RegisteredOperationSupport {
    fn id(&self) -> &'static str {
        self.id
    }

    fn version(&self) -> &'static str {
        "registered-target-compiler"
    }

    fn supported_ops(&self) -> &HashSet<OpId> {
        self.operations
    }
}

/// One backend constructor contributed by a linked backend crate.
///
/// Backend construction can fail (missing GPU adapter, unsupported driver),
/// so the factory returns a [`BackendError`] rather than panicking. Callers
/// iterate [`registered_backends`] and skip backends whose factory fails on
/// this host.
#[derive(Clone)]
pub struct BackendRegistration {
    /// Stable backend identifier, matching [`VyreBackend::id`].
    pub id: &'static str,
    /// Validated target identity owned by the concrete backend crate.
    pub target_id: TargetId,
    /// Stable target-payload format identity owned by the concrete backend.
    ///
    /// This is `None` only when the registration has no target compiler.
    pub payload_format: Option<&'static str>,
    /// Whether this backend is an explicit conformance oracle rather than an
    /// eligible production autoroute target.
    pub reference_oracle: bool,
    /// Factory that constructs the backend implementation.
    ///
    /// Returns `Err(BackendError)` when the backend cannot initialize on
    /// this host. The error message must include a `Fix:` remediation section
    /// per the frozen `BackendError` contract.
    pub factory: fn() -> Result<Box<dyn VyreBackend>, BackendError>,
    /// Language-level IR operation IDs accepted by raw backend dispatch.
    pub supported_ops: fn() -> &'static HashSet<OpId>,
    /// Canonical semantic operation IDs supported by the target compiler.
    ///
    /// This owner-local projection is the target facet submission. The shared
    /// driver joins it with `OperationRegistry` and never infers semantic
    /// support from language-level node capability.
    pub semantic_operations: fn() -> &'static HashSet<OpId>,
    /// Pure compiler facet for this backend's immutable target payload.
    pub target_compiler: Option<fn() -> Result<Box<dyn TargetCompiler>, BackendError>>,
    /// Device acquisition and immutable payload materialization facet.
    pub materializer: Option<fn() -> Result<Box<dyn ArtifactMaterializer>, BackendError>>,
}

impl BackendRegistration {
    /// Construct this registered backend through the shared driver boundary.
    ///
    /// This preserves the raw factory ABI while ensuring registration-based
    /// callers receive the same dispatch wrapper as [`crate::backend::acquire`]
    /// and [`crate::backend::acquire_preferred_dispatch_backend`].
    ///
    /// # Errors
    ///
    /// Returns the backend factory error when the concrete backend cannot
    /// initialize on this host.
    pub fn acquire(&self) -> Result<Box<dyn VyreBackend>, BackendError> {
        (self.factory)().map(wrap_grid_sync_split)
    }

    /// Acquire this backend's pure target compiler facet.
    ///
    /// # Errors
    ///
    /// Returns an explicit unsupported-feature error when the linked backend
    /// does not provide native target compilation, or a contract error when
    /// the constructed compiler disagrees with its registered payload format.
    pub fn target_compiler(&self) -> Result<Box<dyn TargetCompiler>, BackendError> {
        let factory = self
            .target_compiler
            .ok_or_else(|| BackendError::UnsupportedFeature {
                name: "registered target compiler; Fix: link a backend crate that registers native artifact compilation instead of passing a raw Program".to_string(),
                backend: self.id.to_string(),
            })?;
        let expected = self.payload_format.ok_or_else(|| {
            BackendError::new(format!(
                "backend `{}` registers a target compiler without a payload format. Fix: declare the concrete owner-local payload format in BackendRegistration.",
                self.id
            ))
        })?;
        let compiler = factory()?;
        if compiler.format().identity() != expected {
            return Err(BackendError::new(format!(
                "backend `{}` registers payload format `{expected}` but constructed compiler format `{}`. Fix: keep the concrete target registration and compiler format identical.",
                self.id,
                compiler.format().identity()
            )));
        }
        Ok(compiler)
    }

    /// Acquire this backend's device materializer facet.
    ///
    /// # Errors
    ///
    /// Returns an explicit unsupported-feature error when no native
    /// materializer is registered, or the concrete device acquisition error.
    pub fn materializer(&self) -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
        self.materializer
            .ok_or_else(|| BackendError::UnsupportedFeature {
                name: "registered artifact materializer; Fix: link the backend's native materializer instead of recompiling a raw Program at dispatch".to_string(),
                backend: self.id.to_string(),
            })?()
    }
}

inventory::collect!(BackendRegistration);

/// Return target compiler facets keyed by canonical semantic operation identity.
///
/// A compiler-capable backend contributes a facet when the canonical neutral
/// program contains only operation IDs advertised by that backend.
///
/// # Errors
///
/// Returns [`BackendError`] when backend registry startup fails or a concrete
/// target advertises an unknown semantic operation.
pub fn registered_target_operation_facets() -> Result<&'static [TargetOperationFacet], BackendError>
{
    static FACETS: LazyLock<Result<Arc<[TargetOperationFacet]>, BackendError>> = LazyLock::new(
        || {
            let backends = registered_backends()?;
            let mut facets = Vec::new();
            let facet_count = backends.iter().fold(0usize, |count, backend| {
                count.saturating_add((backend.semantic_operations)().len())
            });
            crate::allocation::reserve_vec_to_capacity(
                &mut facets,
                facet_count,
                "Vyre target facet registry",
                "target operation facet",
                "reduce linked target operation declarations",
            )?;
            for backend in backends
                .iter()
                .filter(|backend| backend.target_compiler.is_some())
            {
                for operation_id in (backend.semantic_operations)() {
                    let operation =
                        vyre_foundation::operation::OperationRegistry::global()
                            .get(operation_id)
                            .ok_or_else(|| {
                                BackendError::new(format!(
                                    "target `{}` advertises unknown semantic operation `{operation_id}`. Fix: submit one canonical OperationRegistration or remove the stale target facet.",
                                    backend.target_id
                                ))
                            })?;
                    if operation.program().is_some() {
                        facets.push(TargetOperationFacet {
                            operation_id: operation.id,
                            target_id: backend.target_id.clone(),
                            version: 1,
                        });
                    }
                }
            }
            facets.sort_unstable_by(|left, right| {
                (left.operation_id, &left.target_id).cmp(&(right.operation_id, &right.target_id))
            });
            for pair in facets.windows(2) {
                if pair[0].operation_id == pair[1].operation_id
                    && pair[0].target_id == pair[1].target_id
                {
                    return Err(BackendError::new(format!(
                        "duplicate target facet for operation `{}` and target `{}`. Fix: keep one concrete-driver semantic operation declaration per target.",
                        pair[0].operation_id, pair[0].target_id
                    )));
                }
            }
            Ok(Arc::from(facets))
        },
    );
    match &*FACETS {
        Ok(facets) => Ok(facets.as_ref()),
        Err(error) => Err(error.clone()),
    }
}

/// Per-backend precedence rank registered alongside its
/// [`BackendRegistration`]. Lower rank wins in router selection.
///
/// Conventional ranks are backend-owned. A backend that does not submit a
/// `BackendPrecedence` entry is treated as `u32::MAX`.
pub struct BackendPrecedence {
    /// Backend identifier; must match the corresponding
    /// [`BackendRegistration::id`].
    pub id: &'static str,
    /// Sort key. Lower means higher priority.
    pub rank: u32,
}

inventory::collect!(BackendPrecedence);

/// Backend capability declaration: whether a backend owns a live dispatch
/// stack on this host.
pub struct BackendCapability {
    /// Backend identifier; must match the corresponding
    /// [`BackendRegistration::id`].
    pub id: &'static str,
    /// `true` when this backend's `dispatch` can execute a Program and return
    /// real outputs; `false` when the backend is emission-only.
    pub dispatches: bool,
}

inventory::collect!(BackendCapability);

/// Immutable validated view over linked backend registrations and metadata.
struct BackendRegistry {
    registrations: Arc<[BackendRegistration]>,
    capabilities: Arc<[(&'static str, bool)]>,
    precedence: Arc<[(&'static str, u32)]>,
}

impl BackendRegistry {
    fn build() -> Result<Self, BackendError> {
        let registration_count = inventory::iter::<BackendRegistration>.into_iter().count();
        let mut registrations = Vec::new();
        crate::allocation::reserve_vec_to_capacity(
            &mut registrations,
            registration_count,
            "Vyre backend registry",
            "backend registration",
            "reduce linked backend inventory",
        )?;
        registrations.extend(inventory::iter::<BackendRegistration>.into_iter().cloned());
        registrations.sort_unstable_by(|left, right| left.id.cmp(right.id));
        for registration in &registrations {
            validate_registration(registration)?;
        }
        for pair in registrations.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(BackendError::new(format!(
                    "duplicate backend registration `{}`. Fix: keep one concrete provider for each backend id.",
                    pair[0].id
                )));
            }
        }

        let mut targets = Vec::new();
        crate::allocation::reserve_vec_to_capacity(
            &mut targets,
            registrations.len(),
            "Vyre backend registry",
            "target identity",
            "reduce linked backend inventory",
        )?;
        targets.extend(
            registrations
                .iter()
                .map(|registration| (registration.target_id.as_str(), registration.id)),
        );
        targets.sort_unstable();
        for pair in targets.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(BackendError::new(format!(
                    "target `{}` is claimed by backend providers `{}` and `{}`. Fix: keep one concrete provider for each target identity.",
                    pair[0].0, pair[0].1, pair[1].1
                )));
            }
        }

        let capabilities = freeze_capabilities(&registrations)?;
        let precedence = freeze_precedence(&registrations)?;
        Ok(Self {
            registrations: Arc::from(registrations),
            capabilities,
            precedence,
        })
    }

    fn registration(&self, id: &str) -> Option<&BackendRegistration> {
        self.registrations
            .binary_search_by_key(&id, |registration| registration.id)
            .ok()
            .map(|index| &self.registrations[index])
    }

    fn dispatches(&self, id: &str) -> bool {
        self.capabilities
            .binary_search_by_key(&id, |(backend_id, _)| *backend_id)
            .ok()
            .is_some_and(|index| self.capabilities[index].1)
    }

    fn precedence(&self, id: &str) -> u32 {
        self.precedence
            .binary_search_by_key(&id, |(backend_id, _)| *backend_id)
            .ok()
            .map_or(u32::MAX, |index| self.precedence[index].1)
    }
}

fn validate_registration(registration: &BackendRegistration) -> Result<(), BackendError> {
    validate_registry_identity("backend", registration.id)?;
    if let Some(format) = registration.payload_format {
        validate_registry_identity("target payload format", format)?;
    }
    if registration.target_compiler.is_some() != registration.payload_format.is_some() {
        return Err(BackendError::new(format!(
            "backend `{}` must register its target compiler and payload format together. Fix: provide both target fields or leave both absent.",
            registration.id
        )));
    }
    Ok(())
}

fn validate_registry_identity(kind: &str, identity: &str) -> Result<(), BackendError> {
    if identity.is_empty() || identity.trim() != identity {
        return Err(BackendError::new(format!(
            "{kind} identity `{identity}` is empty or whitespace-padded. Fix: declare a stable non-empty identity without surrounding whitespace."
        )));
    }
    Ok(())
}

fn freeze_capabilities(
    registrations: &[BackendRegistration],
) -> Result<Arc<[(&'static str, bool)]>, BackendError> {
    let count = inventory::iter::<BackendCapability>.into_iter().count();
    let mut entries = Vec::new();
    crate::allocation::reserve_vec_to_capacity(
        &mut entries,
        count,
        "Vyre backend registry",
        "dispatch capability",
        "reduce linked backend capability declarations",
    )?;
    entries.extend(
        inventory::iter::<BackendCapability>
            .into_iter()
            .map(|entry| (entry.id, entry.dispatches)),
    );
    entries.sort_unstable_by_key(|entry| entry.0);
    validate_metadata_ids(registrations, &entries, "dispatch capability")?;
    Ok(Arc::from(entries))
}

fn freeze_precedence(
    registrations: &[BackendRegistration],
) -> Result<Arc<[(&'static str, u32)]>, BackendError> {
    let count = inventory::iter::<BackendPrecedence>.into_iter().count();
    let mut entries = Vec::new();
    crate::allocation::reserve_vec_to_capacity(
        &mut entries,
        count,
        "Vyre backend registry",
        "backend precedence",
        "reduce linked backend precedence declarations",
    )?;
    entries.extend(
        inventory::iter::<BackendPrecedence>
            .into_iter()
            .map(|entry| (entry.id, entry.rank)),
    );
    entries.sort_unstable_by_key(|entry| entry.0);
    validate_metadata_ids(registrations, &entries, "backend precedence")?;
    Ok(Arc::from(entries))
}

fn validate_metadata_ids<T>(
    registrations: &[BackendRegistration],
    entries: &[(&'static str, T)],
    kind: &str,
) -> Result<(), BackendError> {
    for entry in entries {
        validate_registry_identity(kind, entry.0)?;
        if registrations
            .binary_search_by_key(&entry.0, |registration| registration.id)
            .is_err()
        {
            return Err(BackendError::new(format!(
                "{kind} metadata names unregistered backend `{}`. Fix: submit one BackendRegistration with the same id or delete the orphaned metadata.",
                entry.0
            )));
        }
    }
    for pair in entries.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(BackendError::new(format!(
                "duplicate {kind} metadata for backend `{}`. Fix: keep one owner-local metadata submission per backend.",
                pair[0].0
            )));
        }
    }
    Ok(())
}

fn backend_registry() -> Result<&'static BackendRegistry, BackendError> {
    static REGISTRY: LazyLock<Result<BackendRegistry, BackendError>> =
        LazyLock::new(BackendRegistry::build);
    match &*REGISTRY {
        Ok(registry) => Ok(registry),
        Err(error) => Err(error.clone()),
    }
}

/// Return all backend registrations linked into the current binary.
///
/// Registrations are sorted by stable backend identity. The first call freezes
/// one owned registry; subsequent calls return the same immutable slice.
///
/// # Errors
///
/// Returns [`BackendError`] when provider identities conflict, metadata is
/// orphaned or duplicated, a compiler/format pair is incomplete, or registry
/// allocation fails.
pub fn registered_backends() -> Result<&'static [BackendRegistration], BackendError> {
    Ok(backend_registry()?.registrations.as_ref())
}

pub(super) fn registered_backend(
    id: &str,
) -> Result<Option<&'static BackendRegistration>, BackendError> {
    Ok(backend_registry()?.registration(id))
}

pub(super) fn registered_backend_dispatches(id: &str) -> Result<bool, BackendError> {
    Ok(backend_registry()?.dispatches(id))
}

pub(super) fn registered_backend_precedence(id: &str) -> Result<u32, BackendError> {
    Ok(backend_registry()?.precedence(id))
}
