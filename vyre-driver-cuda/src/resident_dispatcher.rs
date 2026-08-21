//! CUDA-resident `ProgramDispatcher`. The module holds no optimizer pass; it
//! holds the residency, pooling and static-upload caching that the pass
//! pipeline dispatches through.
//!
//! Implements the persistent surface of `ProgramDispatcher`: alloc
//! once, upload once, dispatch many times against the same resident
//! buffers, read back at the end. This bypasses the per-call sync
//! overhead the borrowed `dispatch` API has, which is the dominant
//! cost on the optimizer's multi-pass pipeline at small input sizes.
//!
//! CUDA is the persistent optimizer release path. Non-CUDA dispatchers must
//! select their explicit borrowed-dispatch route through capability probing;
//! they must not masquerade as resident execution or silently degrade a CUDA
//! residency contract.

use std::cell::RefCell;

use rustc_hash::FxHashMap;
use vyre_driver::accounting::checked_add_u64_lazy;
use vyre_driver::input_identity::{domain_separated_exact_input_key, ExactInputKey};
use vyre_driver::DispatchConfig;
use vyre_foundation::program_dispatch::{DispatchError, ResidentDispatchStep, ResidentReadRange};

use crate::backend::output_range::CudaOutputReadback;
use crate::backend::staging_reserve::reserve_vec;
use crate::backend::{CudaBackend, CudaResidentBuffer, CudaResidentDispatchStep};
use crate::resident_dispatcher_trait::resident_usize_to_u64;

const CUDA_RESIDENT_POOL_BUDGET_DENOMINATOR: u64 = 32;

pub(crate) struct StaticUploadCacheEntry {
    pub(crate) handles: Vec<CudaResidentBuffer>,
    pub(crate) bytes: u64,
}

pub(crate) fn reserve_resident_vec<T>(
    vec: &mut Vec<T>,
    capacity: usize,
    field: &'static str,
) -> Result<(), DispatchError> {
    reserve_vec(vec, capacity, field)
        .map_err(|error| DispatchError::BackendError(error.to_string()))
}

/// Optimizer dispatcher backed by CUDA-resident buffers.
///
/// Holds a borrow on a live [`CudaBackend`]. All `ProgramDispatcher`
/// trait methods route through the backend's resident-buffer surface
/// when the persistent path applies; the borrowed `dispatch` method
/// still exists for transitions between passes that haven't been
/// converted to persistent yet.
///
/// **Persistent buffer pool.** `free_resident` does NOT actually free
/// the underlying CUDA allocation  -  it returns the handle to a
/// per-byte-len free list owned by this dispatcher. Subsequent
/// `alloc_resident` calls with the same `byte_len` reuse a pooled
/// handle in O(1), bypassing the ~3-5ms CUDA `cuMemAlloc`
/// round-trip. Real alloc/free fires only on size-class misses or
/// when the dispatcher is dropped (see `Drop` impl). For a multi-
/// pass optimizer that allocates 14+ buffers per pipeline run, this
/// drops alloc cost from ~50ms/run on the first call to ~µs on
/// every subsequent call.
pub struct CudaProgramDispatcher<'a> {
    pub(crate) backend: &'a CudaBackend,
    /// `local id -> owned handle` for resident buffers we allocated.
    ///
    /// The `ProgramDispatcher` contract passes bare `u64` ids across its
    /// boundary, so the real owner-qualified handle is retained here and
    /// looked up again on the way back in. A handle is never rebuilt from a
    /// bare id: that is the fabrication this table exists to prevent.
    pub(crate) sizes: RefCell<FxHashMap<u64, CudaResidentBuffer>>,
    /// Per-`byte_len` free list. `free_resident` pushes onto the list
    /// instead of calling the backend; `alloc_resident` pops first
    /// before falling back to a real allocation.
    pub(crate) free_pool: RefCell<FxHashMap<usize, Vec<CudaResidentBuffer>>>,
    /// Bytes currently retained by `free_pool`.
    pub(crate) pooled_bytes: RefCell<u64>,
    /// Content-addressed immutable resident payloads. These handles stay live
    /// across optimizer calls so warmed CUDA runs skip repeated H2D upload of
    /// graph/arena buffers.
    pub(crate) static_upload_cache: RefCell<FxHashMap<ExactInputKey, StaticUploadCacheEntry>>,
    /// Bytes currently retained by `static_upload_cache`.
    pub(crate) static_cached_bytes: RefCell<u64>,
    /// Hard cap for idle resident handles retained by this dispatcher.
    pub(crate) max_pooled_bytes: u64,
    /// Hard cap for immutable resident payload handles retained by this
    /// dispatcher.
    pub(crate) max_static_cached_bytes: u64,
}

impl<'a> CudaProgramDispatcher<'a> {
    /// Wrap a live `CudaBackend` for use as an `ProgramDispatcher`.
    pub fn new(backend: &'a CudaBackend) -> Self {
        Self::with_pool_budget(
            backend,
            cuda_resident_pool_budget_bytes(backend.device_memory_bytes()),
        )
    }

    fn with_pool_budget(backend: &'a CudaBackend, max_pooled_bytes: u64) -> Self {
        Self {
            backend,
            sizes: RefCell::new(FxHashMap::default()),
            free_pool: RefCell::new(FxHashMap::default()),
            pooled_bytes: RefCell::new(0),
            static_upload_cache: RefCell::new(FxHashMap::default()),
            static_cached_bytes: RefCell::new(0),
            max_pooled_bytes,
            max_static_cached_bytes: max_pooled_bytes,
        }
    }

    #[cfg(all(test, feature = "device-tests"))]
    fn new_with_pool_budget_for_tests(backend: &'a CudaBackend, max_pooled_bytes: u64) -> Self {
        Self::with_pool_budget(backend, max_pooled_bytes)
    }

    pub(crate) fn device_feature_cache_key(&self) -> u64 {
        (u64::from(self.backend.ptx_target_sm()) << 32)
            | u64::from(self.backend.pipeline_feature_flags().bits())
    }

    pub(crate) fn resolve(&self, id: u64) -> Result<CudaResidentBuffer, DispatchError> {
        let sizes = self.sizes.borrow();
        sizes.get(&id).copied().ok_or_else(|| {
            DispatchError::Rejected(format!(
                "Fix: CUDA optimizer dispatcher received unknown resident handle id {id}; \
                 every id must come from this dispatcher's `alloc_resident`."
            ))
        })
    }

    pub(crate) fn resolve_many(
        &self,
        ids: &[u64],
    ) -> Result<Vec<CudaResidentBuffer>, DispatchError> {
        let mut handles = Vec::new();
        reserve_resident_vec(&mut handles, ids.len(), "resident handle")?;
        for &id in ids {
            handles.push(self.resolve(id)?);
        }
        Ok(handles)
    }

    pub(crate) fn resolve_uploads<'b>(
        &self,
        uploads: &[(u64, &'b [u8])],
    ) -> Result<Vec<(CudaResidentBuffer, &'b [u8])>, DispatchError> {
        let mut concrete = Vec::new();
        reserve_resident_vec(&mut concrete, uploads.len(), "optimizer upload")?;
        for &(id, bytes) in uploads {
            concrete.push((self.resolve(id)?, bytes));
        }
        Ok(concrete)
    }

    pub(crate) fn resolve_read_ranges(
        &self,
        ranges: &[ResidentReadRange],
    ) -> Result<(Vec<CudaResidentBuffer>, Vec<CudaOutputReadback>), DispatchError> {
        let mut handles = Vec::new();
        reserve_resident_vec(&mut handles, ranges.len(), "optimizer readback handle")?;
        let mut readbacks = Vec::new();
        reserve_resident_vec(&mut readbacks, ranges.len(), "optimizer readback range")?;
        for range in ranges {
            handles.push(self.resolve(range.handle_id)?);
            readbacks.push(CudaOutputReadback {
                device_offset: range.byte_offset,
                byte_len: range.byte_len,
            });
        }
        Ok((handles, readbacks))
    }

    pub(crate) fn resolve_clears(
        &self,
        clears: &[(u64, usize)],
    ) -> Result<Vec<CudaResidentBuffer>, DispatchError> {
        let mut handles = Vec::new();
        reserve_resident_vec(&mut handles, clears.len(), "optimizer clear handle")?;
        for &(id, byte_len) in clears {
            let handle = self.resolve(id)?;
            if handle.byte_len != byte_len {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: CUDA optimizer resident clear for handle {id} expected full buffer length {} but caller requested {byte_len}.",
                    handle.byte_len
                )));
            }
            handles.push(handle);
        }
        Ok(handles)
    }

    pub(crate) fn resolve_fills(
        &self,
        fills: &[(u64, usize, u8)],
    ) -> Result<Vec<(CudaResidentBuffer, u8)>, DispatchError> {
        let mut resolved = Vec::new();
        reserve_resident_vec(&mut resolved, fills.len(), "optimizer fill handle")?;
        for &(id, byte_len, value) in fills {
            let handle = self.resolve(id)?;
            if handle.byte_len != byte_len {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: CUDA optimizer resident fill for handle {id} expected full buffer length {} but caller requested {byte_len}.",
                    handle.byte_len
                )));
            }
            resolved.push((handle, value));
        }
        Ok(resolved)
    }

    pub(crate) fn resolve_step_handles(
        &self,
        steps: &[ResidentDispatchStep<'_>],
        field: &'static str,
    ) -> Result<Vec<Vec<CudaResidentBuffer>>, DispatchError> {
        let mut resolved_step_handles = Vec::new();
        reserve_resident_vec(&mut resolved_step_handles, steps.len(), field)?;
        for step in steps {
            resolved_step_handles.push(self.resolve_many(step.handle_ids)?);
        }
        Ok(resolved_step_handles)
    }

    pub(crate) fn build_cuda_steps<'step, 'program>(
        &self,
        steps: &'step [ResidentDispatchStep<'program>],
        resolved_step_handles: &'step [Vec<CudaResidentBuffer>],
        field: &'static str,
    ) -> Result<Vec<CudaResidentDispatchStep<'step>>, DispatchError>
    where
        'program: 'step,
    {
        let mut cuda_steps = Vec::new();
        reserve_resident_vec(&mut cuda_steps, steps.len(), field)?;
        for (step, handles) in steps.iter().zip(resolved_step_handles.iter()) {
            let mut config = DispatchConfig::default();
            config.grid_override = step.grid_override;
            cuda_steps.push(CudaResidentDispatchStep {
                program: step.program,
                handles,
                config,
            });
        }
        Ok(cuda_steps)
    }

    /// Drain the per-size free pool and return all pooled handles to
    /// the backend. Called from `Drop` so the CUDA context isn't
    /// leaking allocations after the dispatcher is gone.
    fn drain_pool(&self) {
        let mut pool = self.free_pool.borrow_mut();
        let mut sizes = self.sizes.borrow_mut();
        for (_byte_len, handles) in pool.drain() {
            for handle in handles {
                sizes.remove(&handle.handle.id());
                let _ = self.backend.free_resident(handle);
            }
        }
        *self.pooled_bytes.borrow_mut() = 0;
    }

    fn drain_static_upload_cache(&self) {
        let mut cache = self.static_upload_cache.borrow_mut();
        let mut sizes = self.sizes.borrow_mut();
        for (_key, entry) in cache.drain() {
            for handle in entry.handles {
                sizes.remove(&handle.handle.id());
                let _ = self.backend.free_resident(handle);
            }
        }
        *self.static_cached_bytes.borrow_mut() = 0;
    }

    fn evict_one_pooled_resident(&self) -> Result<bool, DispatchError> {
        let mut pool = self.free_pool.borrow_mut();
        let Some(byte_len) = pool
            .iter()
            .filter(|(_, handles)| !handles.is_empty())
            .map(|(byte_len, _)| *byte_len)
            .max()
        else {
            return Ok(false);
        };
        let Some(handles) = pool.get_mut(&byte_len) else {
            return Ok(false);
        };
        let Some(handle) = handles.pop() else {
            return Ok(false);
        };
        drop(pool);
        {
            let mut pooled_bytes = self.pooled_bytes.borrow_mut();
            let handle_bytes =
                resident_usize_to_u64(handle.byte_len, "resident pool evicted handle bytes")?;
            *pooled_bytes = pooled_bytes.checked_sub(handle_bytes).ok_or_else(|| {
                DispatchError::BackendError(
                    "CUDA optimizer resident pool byte accounting underflowed during eviction"
                        .to_string(),
                )
            })?;
        }
        self.backend
            .free_resident(handle)
            .map_err(|e| DispatchError::BackendError(e.to_string()))?;
        Ok(true)
    }

    pub(crate) fn evict_until_resident_pool_has_room(
        &self,
        incoming_bytes: u64,
    ) -> Result<bool, DispatchError> {
        if incoming_bytes > self.max_pooled_bytes {
            return Ok(false);
        }
        while *self.pooled_bytes.borrow() > self.max_pooled_bytes - incoming_bytes {
            if !self.evict_one_pooled_resident()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(crate) fn static_upload_cache_key(
        &self,
        cache_domain: u64,
        payloads: &[&[u8]],
    ) -> Result<ExactInputKey, DispatchError> {
        domain_separated_exact_input_key(
            b"vyre.cuda.optimizer.static-upload.v1",
            cache_domain,
            self.device_feature_cache_key(),
            payloads,
        )
        .map_err(|error| DispatchError::BackendError(error.to_string()))
    }

    pub(crate) fn static_payload_bytes(&self, payloads: &[&[u8]]) -> Result<u64, DispatchError> {
        let mut bytes = 0_u64;
        for payload in payloads {
            let payload_bytes = resident_usize_to_u64(payload.len(), "static payload byte total")?;
            bytes = checked_add_u64_lazy(bytes, payload_bytes, || {
                DispatchError::BackendError(
                    "CUDA optimizer static payload byte accounting overflowed".to_string(),
                )
            })?;
        }
        Ok(bytes)
    }

    fn evict_one_static_upload_cache_entry(&self) -> Result<bool, DispatchError> {
        let Some(key) = self
            .static_upload_cache
            .borrow()
            .iter()
            .max_by_key(|(_, entry)| entry.bytes)
            .map(|(key, _)| *key)
        else {
            return Ok(false);
        };
        let Some(entry) = self.static_upload_cache.borrow_mut().remove(&key) else {
            return Ok(false);
        };
        {
            let mut cached_bytes = self.static_cached_bytes.borrow_mut();
            *cached_bytes = cached_bytes.checked_sub(entry.bytes).ok_or_else(|| {
                DispatchError::BackendError(
                    "CUDA optimizer static cache byte accounting underflowed during eviction"
                        .to_string(),
                )
            })?;
        }
        let mut sizes = self.sizes.borrow_mut();
        for handle in entry.handles {
            sizes.remove(&handle.handle.id());
            self.backend
                .free_resident(handle)
                .map_err(|e| DispatchError::BackendError(e.to_string()))?;
        }
        Ok(true)
    }

    pub(crate) fn evict_until_static_upload_cache_has_room(
        &self,
        incoming_bytes: u64,
    ) -> Result<bool, DispatchError> {
        if incoming_bytes > self.max_static_cached_bytes {
            return Ok(false);
        }
        while *self.static_cached_bytes.borrow() > self.max_static_cached_bytes - incoming_bytes {
            if !self.evict_one_static_upload_cache_entry()? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

// Inline: `vyre_driver_cuda::resident_dispatcher` is `pub(crate)`, so no integration test can reach
// what this suite exercises.
#[cfg(all(test, feature = "device-tests"))]
#[allow(clippy::items_after_test_module)]
mod tests {
    use vyre_foundation::program_dispatch::ProgramDispatcher;

    use super::CudaProgramDispatcher;
    use crate::backend::CudaBackend;

    #[test]
    fn cuda_resident_pool_enforces_byte_budget() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
        let baseline = backend.resident_allocated_bytes();

        {
            let dispatcher = CudaProgramDispatcher::new_with_pool_budget_for_tests(&backend, 64);
            let first = dispatcher
                .alloc_resident(64)
                .expect("Fix: first resident optimizer allocation must succeed.");
            let second = dispatcher
                .alloc_resident(64)
                .expect("Fix: second resident optimizer allocation must succeed.");
            assert_eq!(
                backend.resident_allocated_bytes(),
                baseline + 128,
                "Fix: live CUDA resident accounting must include both active optimizer buffers."
            );

            dispatcher
                .free_resident(first)
                .expect("Fix: freeing first optimizer buffer into the pool must succeed.");
            assert_eq!(
                backend.resident_allocated_bytes(),
                baseline + 128,
                "Fix: one active buffer plus one pooled buffer should remain resident."
            );

            dispatcher
                .free_resident(second)
                .expect("Fix: freeing second optimizer buffer must respect the pool budget.");
            assert_eq!(
                backend.resident_allocated_bytes(),
                baseline + 64,
                "Fix: optimizer resident pool must evict excess idle buffers instead of pinning unbounded VRAM."
            );
        }

        assert_eq!(
            backend.resident_allocated_bytes(),
            baseline,
            "Fix: dropping the optimizer dispatcher must release every retained resident buffer."
        );
    }

    #[test]
    fn cuda_resident_static_upload_cache_skips_warm_h2d_and_releases_on_drop() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
        let baseline = backend.resident_allocated_bytes();
        let payload_a = vec![0xA5_u8; 256];
        let payload_b = vec![0x5A_u8; 512];
        let expected_h2d = (payload_a.len() + payload_b.len()) as u64;

        {
            let dispatcher =
                CudaProgramDispatcher::new_with_pool_budget_for_tests(&backend, expected_h2d * 4);
            backend.reset_telemetry();
            let cold = dispatcher
                .acquire_resident_static_uploads(0x4355_4441_5354_4154, &[&payload_a, &payload_b])
                .expect("Fix: cold static resident upload must allocate and upload.");
            assert!(
                !cold.cache_hit,
                "Fix: first static resident acquisition cannot claim a cache hit."
            );
            assert!(
                cold.retained_by_dispatcher,
                "Fix: cacheable CUDA static resident handles must stay owned by the dispatcher."
            );
            dispatcher
                .release_resident_static_uploads(cold)
                .expect("Fix: releasing a retained static resident set must be a no-op.");
            assert_eq!(
                backend.telemetry_snapshot().host_to_device_bytes,
                expected_h2d,
                "Fix: cold static resident acquisition must upload each immutable payload exactly once."
            );

            backend.reset_telemetry();
            let warm = dispatcher
                .acquire_resident_static_uploads(0x4355_4441_5354_4154, &[&payload_a, &payload_b])
                .expect("Fix: warm static resident acquisition must reuse device buffers.");
            assert!(
                warm.cache_hit,
                "Fix: identical immutable CUDA payloads must be served from resident cache."
            );
            dispatcher
                .release_resident_static_uploads(warm)
                .expect("Fix: releasing a warm retained static resident set must be a no-op.");
            assert_eq!(
                backend.telemetry_snapshot().host_to_device_bytes,
                0,
                "Fix: warm static resident acquisition must not re-upload immutable payloads."
            );
        }

        assert_eq!(
            backend.resident_allocated_bytes(),
            baseline,
            "Fix: dropping the CUDA optimizer dispatcher must release retained static-cache buffers."
        );
    }

    #[test]
    fn cuda_resident_clear_uses_device_memset_not_h2d_upload() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
        let dispatcher = CudaProgramDispatcher::new_with_pool_budget_for_tests(&backend, 4096);
        let handle = dispatcher
            .alloc_resident(64)
            .expect("Fix: resident clear test allocation must succeed.");
        dispatcher
            .upload_resident(handle, &[0xFF_u8; 64])
            .expect("Fix: resident clear test seed upload must succeed.");

        backend.reset_telemetry();
        let mut outputs = Vec::new();
        dispatcher
            .clear_upload_resident_many_sequence_read_many_into(
                &[(handle, 64)],
                &[],
                &[],
                &[handle],
                &mut outputs,
            )
            .expect("Fix: CUDA resident clear+read sequence must succeed.");
        assert_eq!(
            backend.telemetry_snapshot().host_to_device_bytes,
            0,
            "Fix: CUDA resident clears must use device memset instead of H2D zero uploads."
        );
        assert_eq!(
            outputs,
            vec![vec![0_u8; 64]],
            "Fix: CUDA resident clear must zero every byte before readback."
        );

        backend.reset_telemetry();
        dispatcher
            .fill_upload_resident_many_sequence_read_many_into(
                &[(handle, 64, 0xA5)],
                &[],
                &[],
                &[handle],
                &mut outputs,
            )
            .expect("Fix: CUDA resident fill+read sequence must succeed.");
        assert_eq!(
            backend.telemetry_snapshot().host_to_device_bytes,
            0,
            "Fix: CUDA resident fills must use device memset instead of H2D byte-pattern uploads."
        );
        assert_eq!(
            outputs,
            vec![vec![0xA5_u8; 64]],
            "Fix: CUDA resident fill must write the requested byte pattern before readback."
        );

        dispatcher
            .free_resident(handle)
            .expect("Fix: resident clear test handle must return to the pool.");
    }
}

fn cuda_resident_pool_budget_bytes(total_memory_bytes: u64) -> u64 {
    total_memory_bytes / CUDA_RESIDENT_POOL_BUDGET_DENOMINATOR
}

impl<'a> Drop for CudaProgramDispatcher<'a> {
    fn drop(&mut self) {
        self.drain_static_upload_cache();
        self.drain_pool();
    }
}
