//! Shared validation caches and launch-geometry checks for concrete drivers.

use std::collections::HashSet;
use std::hash::BuildHasherDefault;

use rustc_hash::FxHasher;
use vyre_foundation::ir::{OpId, Program};
use vyre_foundation::validate::{BackendValidationCapabilities, ValidationOptions};

use crate::{BackendError, DispatchConfig, VyreBackend};

/// Default successful-validation hash entries retained per backend instance.
pub const DEFAULT_VALIDATION_HASH_ENTRIES: usize = 8192;
/// Default VSA fingerprints retained per backend instance.
pub const DEFAULT_VALIDATION_VSA_ENTRIES: usize = 2048;
/// Default VSA shard count.
pub const DEFAULT_VALIDATION_VSA_SHARDS: usize = 64;

type ValidationSet = dashmap::DashSet<blake3::Hash, BuildHasherDefault<FxHasher>>;

/// Successful-program validation cache shared by concrete drivers.
pub struct ValidationCache {
    hashes: ValidationSet,
    vsa_hashes: ValidationSet,
    max_hash_entries: usize,
    max_vsa_entries: usize,
    vsa_shards: usize,
}

impl std::fmt::Debug for ValidationCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ValidationCache")
            .field("hashes", &self.hashes.len())
            .field("vsa_hashes", &self.vsa_hashes.len())
            .field("vsa_shards", &self.vsa_shards)
            .field("max_hash_entries", &self.max_hash_entries)
            .field("max_vsa_entries", &self.max_vsa_entries)
            .finish()
    }
}

impl Default for ValidationCache {
    fn default() -> Self {
        Self::new(
            DEFAULT_VALIDATION_HASH_ENTRIES,
            DEFAULT_VALIDATION_VSA_ENTRIES,
            DEFAULT_VALIDATION_VSA_SHARDS,
        )
    }
}

impl ValidationCache {
    /// Create a validation cache with bounded hash and VSA storage.
    #[must_use]
    pub fn new(max_hash_entries: usize, max_vsa_entries: usize, vsa_shards: usize) -> Self {
        let shard_count = vsa_shards.max(1);
        Self {
            hashes: dashmap::DashSet::with_hasher(BuildHasherDefault::<FxHasher>::default()),
            vsa_hashes: dashmap::DashSet::with_capacity_and_hasher(
                max_vsa_entries.max(1),
                BuildHasherDefault::<FxHasher>::default(),
            ),
            max_hash_entries: max_hash_entries.max(1),
            max_vsa_entries: max_vsa_entries.max(1),
            vsa_shards: shard_count,
        }
    }

    /// Compute the validation hash for a program.
    #[must_use]
    pub fn program_hash(program: &Program) -> blake3::Hash {
        blake3::Hash::from(program.fingerprint())
    }

    /// Return whether a validation hash is cached.
    #[must_use]
    pub fn contains_hash(&self, hash: &blake3::Hash) -> bool {
        self.hashes.contains(hash)
    }

    /// Remember a successful validation hash.
    pub fn remember_hash(&self, hash: blake3::Hash) {
        if self.hashes.len() >= self.max_hash_entries {
            self.hashes.clear();
        }
        self.hashes.insert(hash);
    }

    /// Remember a successful validation hash and its VSA fingerprint.
    ///
    /// # Errors
    ///
    /// Returns if a VSA shard lock is poisoned.
    pub fn remember_success(&self, hash: blake3::Hash, vsa: &[u32]) -> Result<(), BackendError> {
        self.remember_hash(hash);
        if self.vsa_hashes.len() >= self.max_vsa_entries {
            self.vsa_hashes.clear();
        }
        self.vsa_hashes.insert(vsa_words_hash(vsa));
        Ok(())
    }

    /// Clear cached validation state.
    ///
    /// # Errors
    ///
    /// Returns if a VSA shard lock is poisoned.
    pub fn clear(&self) -> Result<(), BackendError> {
        self.hashes.clear();
        self.vsa_hashes.clear();
        Ok(())
    }

    /// Validate `program` once, memoizing the complete backend contract.
    ///
    /// This is the shared driver validation path: foundation invariants,
    /// backend supported-op coverage, program capability requirements, and
    /// VSA cache insertion all happen in one place. Concrete drivers supply
    /// only their actual capability values.
    ///
    /// # Errors
    ///
    /// Returns when validation fails or a VSA shard lock is poisoned.
    pub fn get_or_validate(
        &self,
        program: &Program,
        validation_options: ValidationOptions<'_>,
        supported_ops: &HashSet<OpId>,
        caps: ProgramValidationCaps,
    ) -> Result<(), BackendError> {
        let hash = Self::program_hash(program);
        if self.contains_hash(&hash) || program.is_validated_on(caps.backend_id) {
            self.remember_hash(hash);
            return Ok(());
        }

        validate_program_contract(program, validation_options, supported_ops, caps)?;

        let vsa = crate::launch::program_vsa_fingerprint_words(program);
        self.remember_success(hash, &vsa)?;
        program.mark_validated_on(caps.backend_id);
        Ok(())
    }

    /// Validate `program` against a concrete backend and cache successful
    /// results.
    ///
    /// This is the canonical driver-owned validation-cache entry point for
    /// backends that implement both the runtime backend contract and the
    /// foundation capability-validation contract.
    ///
    /// # Errors
    ///
    /// Returns when validation fails or cache mutation fails.
    pub fn get_or_validate_backend<B>(
        &self,
        program: &Program,
        backend: &B,
    ) -> Result<(), BackendError>
    where
        B: VyreBackend + BackendValidationCapabilities,
    {
        let validation_options = ValidationOptions::default().with_backend(backend);
        self.get_or_validate(
            program,
            validation_options,
            backend.supported_ops(),
            ProgramValidationCaps::from_backend(backend),
        )
    }
}

/// Concrete backend capability values needed for shared program validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramValidationCaps {
    /// Stable backend identifier used in diagnostics and validation stamps.
    pub backend_id: &'static str,
    /// Native subgroup operations are available and lowered.
    pub supports_subgroup_ops: bool,
    /// IEEE binary16 buffers/operations are lowered.
    pub supports_f16: bool,
    /// Bfloat16 buffers/operations are lowered.
    pub supports_bf16: bool,
    /// Indirect dispatch is lowered.
    pub supports_indirect_dispatch: bool,
    /// Distributed collective communication nodes are lowered.
    pub supports_distributed_collectives: bool,
    /// `Node::Trap` is lowered with backend-visible trap semantics.
    pub supports_trap_propagation: bool,
    /// A whole-grid barrier is lowered natively, as a cooperative launch.
    pub supports_grid_sync: bool,
    /// The shared registry wrapper may emulate a whole-grid barrier for this
    /// backend by splitting one program into sequential host dispatches.
    ///
    /// Held separately from `supports_grid_sync` because the two answer
    /// different questions and a backend can refuse the emulation while
    /// lowering nothing natively. Validation refuses only when both are false.
    pub allows_host_grid_sync_split: bool,
    /// Maximum supported workgroup dimensions.
    pub max_workgroup_size: [u32; 3],
}

impl ProgramValidationCaps {
    /// Snapshot capability values from a `VyreBackend` trait object.
    #[must_use]
    pub fn from_backend(backend: &dyn VyreBackend) -> Self {
        Self {
            backend_id: backend.id(),
            supports_subgroup_ops: backend.supports_subgroup_ops(),
            supports_f16: backend.supports_f16(),
            supports_bf16: backend.supports_bf16(),
            supports_indirect_dispatch: backend.supports_indirect_dispatch(),
            supports_distributed_collectives: backend.supports_distributed_collectives(),
            // Every backend in this workspace leaves a host-readable record
            // when a lane traps, so a trapping launch refuses instead of
            // returning wrong data. A backend that cannot must stop reporting
            // this here rather than at its own call site.
            supports_trap_propagation: true,
            supports_grid_sync: backend.supports_grid_sync(),
            allows_host_grid_sync_split: backend.allows_host_grid_sync_split(),
            max_workgroup_size: backend.max_workgroup_size(),
        }
    }

    /// The same facts in the shape the foundation capability check takes.
    ///
    /// This is the only place a backend's advertisement is mapped onto
    /// [`vyre_foundation::program_caps::BackendSupport`]. Written out at each
    /// call site instead, eight same-typed values get re-spelled per caller
    /// and a transposition reads as plausible code.
    #[must_use]
    pub fn support(&self) -> vyre_foundation::program_caps::BackendSupport {
        vyre_foundation::program_caps::BackendSupport {
            subgroup_ops: self.supports_subgroup_ops,
            half_precision: self.supports_f16,
            brain_float: self.supports_bf16,
            indirect_dispatch: self.supports_indirect_dispatch,
            trap_propagation: self.supports_trap_propagation,
            distributed_collectives: self.supports_distributed_collectives,
            // Either route runs the barrier. Which one is chosen happens after
            // validation has established that one exists.
            grid_sync: self.supports_grid_sync || self.allows_host_grid_sync_split,
            max_workgroup_size: self.max_workgroup_size,
        }
    }
}

/// Validate a program against backend-neutral and backend-reported contracts.
///
/// # Errors
///
/// Returns when foundation validation, supported-op validation, or required
/// capability checks fail.
pub fn validate_program_contract(
    program: &Program,
    validation_options: ValidationOptions<'_>,
    supported_ops: &HashSet<OpId>,
    caps: ProgramValidationCaps,
) -> Result<(), BackendError> {
    let lowered_program = if caps.supports_distributed_collectives {
        None
    } else {
        vyre_foundation::transform::collectives::lower_single_rank_collectives(program).map_err(
            |error| BackendError::InvalidProgram {
                fix: error.to_string(),
            },
        )?
    };
    let program = lowered_program.as_ref().unwrap_or(program);
    let report = vyre_foundation::validate::validate_with_options(program, validation_options);
    if let Some(source) = report.errors.into_iter().next() {
        return Err(BackendError::Validation { source });
    }

    validate_supported_ops(program, caps.backend_id, supported_ops)
        .map_err(|source| BackendError::Validation { source })?;

    let required = vyre_foundation::program_caps::scan(program);
    vyre_foundation::program_caps::check_backend_capabilities(
        caps.backend_id,
        &caps.support(),
        &required,
    )
    .map_err(|error| BackendError::InvalidProgram {
        fix: error.to_string(),
    })
}

fn validate_supported_ops(
    program: &Program,
    backend_id: &'static str,
    supported_ops: &HashSet<OpId>,
) -> Result<(), vyre_foundation::validate::ValidationError> {
    struct SupportedOpsBackend<'a> {
        id: &'static str,
        ops: &'a HashSet<OpId>,
    }

    impl crate::backend::Backend for SupportedOpsBackend<'_> {
        fn id(&self) -> &'static str {
            self.id
        }

        fn version(&self) -> &'static str {
            env!("CARGO_PKG_VERSION")
        }

        fn supported_ops(&self) -> &HashSet<OpId> {
            self.ops
        }
    }

    crate::backend::validation::validate_program(
        program,
        &SupportedOpsBackend {
            id: backend_id,
            ops: supported_ops,
        },
    )
}

/// Launch-geometry limits reported by a concrete driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchGeometryLimits {
    /// Backend name used in diagnostics.
    pub backend: &'static str,
    /// Maximum invocations in one workgroup or block.
    pub max_threads_per_block: u32,
    /// Maximum workgroup or block dimensions (x, y, z).
    pub max_block_dim: [u32; 3],
    /// Maximum workgroup count per grid dimension.
    pub max_grid_dim: [u32; 3],
    /// Maximum threads the device keeps resident on one compute unit, the
    /// hardware block a workgroup is resident on, or `0` when the backend does
    /// not probe this number.
    ///
    /// This is the budget that decides how many whole workgroups fit on one
    /// unit, and the division is integral: a workgroup width that does not
    /// divide it strands the remainder on every unit for the launch's whole
    /// lifetime. Backends that leave this `0` opt out of every residency-aware
    /// decision rather than receiving one derived from a guessed budget.
    pub max_threads_per_sm: u32,
}

impl LaunchGeometryLimits {
    /// Whole workgroups of `workgroup_threads` that stay resident on one
    /// compute unit, or `None` when this backend reports no per-unit budget.
    #[must_use]
    pub fn blocks_per_compute_unit(&self, workgroup_threads: u32) -> Option<u32> {
        (self.max_threads_per_sm != 0 && workgroup_threads != 0)
            .then(|| blocks_per_compute_unit(self.max_threads_per_sm, workgroup_threads))
    }

    /// Threads that stay resident on one compute unit at `workgroup_threads`
    /// wide, or `None` when this backend reports no per-unit budget.
    #[must_use]
    pub fn resident_threads_per_compute_unit(&self, workgroup_threads: u32) -> Option<u32> {
        (self.max_threads_per_sm != 0 && workgroup_threads != 0)
            .then(|| resident_threads_per_compute_unit(self.max_threads_per_sm, workgroup_threads))
    }
}

/// Whole workgroups of `workgroup_threads` that fit one compute unit's thread
/// budget.
///
/// This is the single definition of the residency division in the workspace.
/// Cooperative launch preflight and cold-start launch-width selection both
/// route through it, because two independent copies of this arithmetic
/// had already drifted apart once. The division is integral by hardware: a
/// unit hosts whole workgroups only.
///
/// Threads are the only ceiling modelled here. Hardware also caps blocks per
/// unit independently, and a backend reports that cap separately, so at narrow
/// widths
/// the real block count is lower than this returns and the shortfall can be
/// large: where the device caps blocks at 24, a 32-wide group against a
/// 1536-thread budget measures 24 blocks and 768 resident threads, half what
/// this function's 48 blocks and 1536 threads predict. A caller that ranks
/// widths from widest downward never reaches that regime. A caller that
/// answers "does this declared width fit", such as a cooperative launch
/// preflight, does, and must clamp by the device-reported block cap before
/// admitting a grid.
#[must_use]
pub fn blocks_per_compute_unit(max_threads_per_unit: u32, workgroup_threads: u32) -> u32 {
    if workgroup_threads == 0 {
        return 0;
    }
    max_threads_per_unit / workgroup_threads
}

/// Threads resident on one compute unit at `workgroup_threads` wide, under the
/// per-unit thread budget alone.
///
/// Equal to `blocks_per_compute_unit(..) * workgroup_threads`, so it is at most
/// `max_threads_per_unit` and falls short of it by exactly the slots the
/// integral division strands. A width of 1024 against a 1536-thread budget
/// resolves to one block and 1024 resident threads, leaving 512 slots per unit
/// unusable for the launch's duration. The block-count caveat on
/// [`blocks_per_compute_unit`] applies here too.
#[must_use]
pub fn resident_threads_per_compute_unit(max_threads_per_unit: u32, workgroup_threads: u32) -> u32 {
    blocks_per_compute_unit(max_threads_per_unit, workgroup_threads)
        .saturating_mul(workgroup_threads)
}

/// Validate workgroup and grid dimensions against backend launch limits.
///
/// # Errors
///
/// Returns when dimensions are zero, overflow the invocation product, exceed
/// workgroup limits, or exceed per-axis grid limits.
pub fn validate_launch_geometry(
    workgroup: [u32; 3],
    grid: [u32; 3],
    limits: LaunchGeometryLimits,
) -> Result<(), BackendError> {
    if workgroup.contains(&0) || grid.contains(&0) {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} workgroup and grid dimensions must all be non-zero.",
                limits.backend
            ),
        });
    }
    let threads = workgroup[0]
        .checked_mul(workgroup[1])
        .and_then(|xy| xy.checked_mul(workgroup[2]))
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} workgroup dimensions overflowed u32; reduce workgroup_override.",
                limits.backend
            ),
        })?;
    if threads > limits.max_threads_per_block {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} workgroup has {threads} threads but device max is {}.",
                limits.backend, limits.max_threads_per_block
            ),
        });
    }
    for (axis, &dim) in workgroup.iter().enumerate() {
        if dim > limits.max_block_dim[axis] {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {} workgroup axis {axis} requested {} threads but device max is {}.",
                    limits.backend, dim, limits.max_block_dim[axis]
                ),
            });
        }
    }
    for (axis, &dim) in grid.iter().enumerate() {
        if dim > limits.max_grid_dim[axis] {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {} grid axis {axis} requested {} workgroups but device max is {}.",
                    limits.backend, dim, limits.max_grid_dim[axis]
                ),
            });
        }
    }
    Ok(())
}

/// Validate a program's effective workgroup shape against a backend's reported limits.
///
/// This is the shared pre-dispatch gate for callers that have a `VyreBackend`
/// trait object but have not entered a concrete driver yet.
///
/// # Errors
///
/// Returns when any workgroup axis is zero, exceeds the backend's per-axis
/// limit, or when total invocations exceed the backend's workgroup limit.
pub fn validate_program_for_backend(
    backend: &dyn VyreBackend,
    program: &Program,
    config: &DispatchConfig,
) -> Result<(), BackendError> {
    let workgroup = config
        .workgroup_override
        .unwrap_or(program.workgroup_size());
    let max_axes = backend.max_workgroup_size();
    if workgroup.contains(&0) {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: backend `{}` cannot dispatch zero-sized workgroup dimensions; set positive workgroup sizes.",
                backend.id()
            ),
        });
    }
    for (axis, &dim) in workgroup.iter().enumerate() {
        if dim > max_axes[axis] {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: backend `{}` workgroup axis {axis} requested {} but max is {}.",
                    backend.id(),
                    dim,
                    max_axes[axis]
                ),
            });
        }
    }
    let invocations = workgroup[0]
        .checked_mul(workgroup[1])
        .and_then(|xy| xy.checked_mul(workgroup[2]))
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: backend `{}` workgroup dimensions overflowed u32; reduce workgroup size.",
                backend.id()
            ),
        })?;
    let max_invocations = backend.max_compute_invocations_per_workgroup();
    if invocations > max_invocations {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: backend `{}` workgroup has {invocations} invocations but max is {max_invocations}.",
                backend.id()
            ),
        });
    }
    if let Some(grid) = config.grid_override {
        let max_workgroups = backend.max_compute_workgroups_per_dimension();
        if grid.contains(&0) {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: backend `{}` cannot dispatch zero-sized grid dimensions; set positive grid_override values.",
                    backend.id()
                ),
            });
        }
        for (axis, &dim) in grid.iter().enumerate() {
            if dim > max_workgroups {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: backend `{}` grid_override axis {axis} requested {} workgroups but max is {}.",
                        backend.id(),
                        dim,
                        max_workgroups
                    ),
                });
            }
        }
    }
    Ok(())
}

fn vsa_words_hash(words: &[u32]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(words.len() as u64).to_le_bytes());
    for word in words {
        hasher.update(&word.to_le_bytes());
    }
    hasher.finalize()
}

// Inline: the cache cases read `ValidationCache::vsa_hashes`, a private field, so
// no integration test can observe what the cache actually remembered.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_cache_records_vsa_without_lock_shards() {
        let cache = ValidationCache::new(8, 8, 4);
        let hash = blake3::hash(b"program");
        cache
            .remember_success(hash, &[1, 2, 3, 4])
            .expect("Fix: lock-free VSA cache insertion must not fail");

        assert!(cache.contains_hash(&hash));
        assert_eq!(cache.vsa_hashes.len(), 1);
        assert!(format!("{cache:?}").contains("vsa_hashes"));
    }

    #[test]
    fn validation_cache_bounds_vsa_hashes_by_clear() {
        let cache = ValidationCache::new(8, 2, 4);
        for i in 0..3u32 {
            cache
                .remember_success(blake3::hash(&i.to_le_bytes()), &[i])
                .expect("Fix: VSA cache insertion must stay infallible");
        }
        assert!(
            cache.vsa_hashes.len() <= 2,
            "Fix: bounded VSA cache must not grow past max entries"
        );
    }

    /// The residency division is integral and both of its edges are pinned,
    /// because this arithmetic now has exactly one definition and the native
    /// cooperative launch preflight reads it.
    ///
    /// A zero width has no meaningful block count and yields zero rather than
    /// dividing. A width wider than the whole per-unit budget hosts no block at
    /// all, so it also yields zero: that is a launch the caller must reject,
    /// not one silently rounded up to a single block. Both match what
    /// `cooperative_thread_residency_block_limit` did before the arithmetic
    /// moved here, and a factoring that quietly changed either edge would be
    /// worse than the duplicate it replaced.
    #[test]
    fn residency_division_is_integral_at_both_edges() {
        assert_eq!(blocks_per_compute_unit(1536, 0), 0);
        assert_eq!(resident_threads_per_compute_unit(1536, 0), 0);
        assert_eq!(blocks_per_compute_unit(1536, 2048), 0);
        assert_eq!(resident_threads_per_compute_unit(1536, 2048), 0);
        assert_eq!(blocks_per_compute_unit(0, 256), 0);
        assert_eq!(resident_threads_per_compute_unit(0, 256), 0);

        assert_eq!(blocks_per_compute_unit(1536, 1024), 1);
        assert_eq!(
            resident_threads_per_compute_unit(1536, 1024),
            1024,
            "Fix: 1024 wide against a 1536-thread unit strands 512 slots. The truncation is the whole point of pinning this."
        );
        assert_eq!(blocks_per_compute_unit(1536, 256), 6);
        assert_eq!(resident_threads_per_compute_unit(1536, 256), 1536);
    }

    /// A backend that reports no per-unit thread budget answers `unknown`, so
    /// no residency-aware decision can be derived from a number it never gave.
    #[test]
    fn unreported_per_unit_budget_answers_unknown_rather_than_zero() {
        let reported = LaunchGeometryLimits {
            backend: "reported",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        };
        let unreported = LaunchGeometryLimits {
            max_threads_per_sm: 0,
            ..reported
        };

        assert_eq!(reported.blocks_per_compute_unit(256), Some(6));
        assert_eq!(reported.resident_threads_per_compute_unit(256), Some(1536));
        assert_eq!(reported.blocks_per_compute_unit(0), None);
        assert_eq!(unreported.blocks_per_compute_unit(256), None);
        assert_eq!(unreported.resident_threads_per_compute_unit(256), None);
    }
}
