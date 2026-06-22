//! Resident-buffer dispatch for [`GpuLiteralSet`] region-presence scans.
//!
//! # Why this exists
//!
//! [`GpuLiteralSet::scan_presence_by_region`](super::literal_set::GpuLiteralSet::scan_presence_by_region)
//! and its async sibling issue every dispatch through `dispatch_borrowed`, which
//! re-encodes and **re-uploads seven immutable tables on every call**: the DFA
//! transition / output-offset / output-record tables, the per-pattern length
//! table, and the three suffix-prefilter masks (end mask, 2-gram mask, 3-gram
//! bloom). For a large detector set the DFA transition table alone is
//! `state_count × 256` u32s — multiple MiB — and it is *identical* across every
//! file of a corpus. A consumer that scans many coalesced batches (keyhog's
//! phase-1 layout) pays that multi-MiB host→device transfer once per batch even
//! though only the haystack and the per-region presence output actually change.
//!
//! [`ResidentPresencePipeline`] uploads those seven tables **once** into
//! backend-resident resources and keeps them resident for the session lifetime.
//! Each [`scan_into`](ResidentPresencePipeline::scan_into) then transfers only the
//! per-file haystack (a ranged upload into the resident haystack buffer) and
//! zeroes the used prefix of the resident presence buffer, dispatches against the
//! resident tables, and reads back the per-region presence bitmap — the per-scan
//! transfer drops from `O(tables + haystack)` to `O(haystack + region rows)`.
//! This is the region-presence counterpart of
//! [`RulePipeline::prepare_resident`](super::mega_scan::RulePipeline::prepare_resident)
//! (the regex/NFA mega-scan path) and of
//! [`GpuLiteralSet::prepare_presence_by_region_dispatch`](super::literal_set::GpuLiteralSet::prepare_presence_by_region_dispatch)
//! (the backend-neutral single-shot prepared payload).
//!
//! The decoded bitmap is byte-identical to
//! [`GpuLiteralSet::scan_presence_by_region`]'s return (bit `p` of region `r`'s
//! row is set iff pattern `p`'s literal occurs in region `r`), so a consumer can
//! swap the borrowed path for a resident session without changing any
//! post-processing — proven by the GPU parity test in the integration suite and
//! the host-orchestration unit test below.
//!
//! # The `max_regions` cap
//!
//! The resident program is built for `max_regions` coalesced files: that count
//! sizes the resident presence buffer (binding 6) and the kernel's
//! `ceil_log2(max_regions)` region binary-search width. The ACTUAL per-scan region
//! count is read dynamically from `buf_len(region_starts)`, so one resident
//! session serves any batch with `region_count <= max_regions`. A batch that
//! exceeds the cap is rejected **loudly** (it would index past the resident
//! presence buffer); the caller re-dispatches it through the per-batch-sized
//! borrowed [`GpuLiteralSet::scan_presence_by_region`] — never a silent truncation.
//!
//! # Backend support
//!
//! Resident dispatch requires a backend that implements the resident half of the
//! [`VyreBackend`] contract (`allocate_resident`, `upload_resident*`,
//! `dispatch_resident_timed`). The wgpu and CUDA backends do; the CPU reference
//! does not. [`GpuLiteralSet::prepare_resident_presence`] surfaces the backend's
//! `UnsupportedFeature` error **loudly** — the caller must handle it explicitly
//! (fail closed, or a loud/recorded fallback), never degrade silently.

use vyre::{BackendError, VyreBackend};
use vyre_driver::Resource;
use vyre_foundation::ir::Program;

use super::dispatch_io;
use super::literal_set::GpuLiteralSet;

const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Number of buffer bindings in the region-presence program (see
/// [`super::literal_set::GpuLiteralSet::build_presence_by_region_dispatch`]).
const PRESENCE_BY_REGION_BINDINGS: usize = 12;

/// A [`GpuLiteralSet`] with its immutable region-presence tables uploaded into
/// backend-resident resources, ready for repeated low-overhead scans.
///
/// Construct with [`GpuLiteralSet::prepare_resident_presence`]. The session owns
/// nine resident resources (haystack, the seven immutable tables, and the
/// read-write presence buffer); call [`free`](Self::free) to release them, or drop
/// the session and let the backend reclaim them when its device context is torn
/// down.
///
/// The session is `Send + Sync`: the resident handles are opaque ids and all
/// mutation happens through the borrowed `backend`, so a single session can be
/// shared across scan threads (each thread supplies its own packing scratch).
#[derive(Debug)]
pub struct ResidentPresencePipeline {
    /// Region-presence program sized for `max_regions` coalesced files.
    program: Program,
    /// Resident haystack buffer, sized to `haystack_capacity` padded bytes.
    haystack: Resource,
    /// Resident DFA transition table (immutable, uploaded once).
    transitions: Resource,
    /// Resident DFA output-offset table (immutable, uploaded once).
    output_offsets: Resource,
    /// Resident DFA output-record table (immutable, uploaded once).
    output_records: Resource,
    /// Resident per-pattern length table (immutable, uploaded once).
    pattern_lengths: Resource,
    /// Resident per-region presence buffer (read-write; used prefix reset per scan).
    presence: Resource,
    /// Resident suffix prefilter end mask (immutable, uploaded once).
    candidate_end_mask: Resource,
    /// Resident suffix prefilter 2-gram mask (immutable, uploaded once).
    candidate_suffix2_mask: Resource,
    /// Resident suffix prefilter 3-gram bloom (immutable, uploaded once).
    candidate_suffix3_bloom: Resource,
    /// Padded byte capacity of the resident haystack buffer.
    haystack_capacity: usize,
    /// Largest coalesced-file count this session's presence buffer was sized for.
    max_regions: u32,
    /// Pattern count (bit width of each per-region presence row).
    pattern_count: u32,
    /// Presence bitmap `u32` words per region.
    presence_words: u32,
    /// Program workgroup X extent, for the per-scan byte-scan dispatch geometry.
    workgroup_x: u32,
}

// SAFETY mirror of the `ResidentRulePipeline`/`GpuLiteralSet` contract: `Resource`
// handles are plain ids and `Program` is `Send + Sync`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ResidentPresencePipeline>;
};

impl GpuLiteralSet {
    /// Upload this matcher's immutable region-presence tables into backend-resident
    /// resources and return a [`ResidentPresencePipeline`] for repeated scans.
    ///
    /// `haystack_capacity_bytes` is the largest coalesced haystack the session will
    /// scan (e.g. the consumer's batch cap); the resident haystack buffer is
    /// allocated once at that padded size and every scan uploads only its real
    /// bytes. `max_regions` is the largest coalesced-file count the session will
    /// scan; it sizes the resident presence buffer and the kernel's region
    /// binary-search width, and caps decoded regions.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the backend does not support resident
    /// resources, when `max_regions` is zero, or when allocation / upload of the
    /// resident tables fails. The caller must handle this loudly (fail closed or a
    /// recorded fallback) — never degrade silently.
    pub fn prepare_resident_presence(
        &self,
        backend: &dyn VyreBackend,
        haystack_capacity_bytes: usize,
        max_regions: u32,
    ) -> Result<ResidentPresencePipeline, BackendError> {
        let tables = self.resident_presence_tables(max_regions)?;

        let haystack_capacity = dispatch_io::haystack_padded_u32_byte_len(haystack_capacity_bytes)?;
        let haystack = backend.allocate_resident(haystack_capacity)?;

        // The seven immutable tables: allocate + upload ONCE.
        let transitions = allocate_and_upload(backend, &tables.transitions)?;
        let output_offsets = allocate_and_upload(backend, &tables.output_offsets)?;
        let output_records = allocate_and_upload(backend, &tables.output_records)?;
        let pattern_lengths = allocate_and_upload(backend, &tables.pattern_lengths)?;
        let candidate_end_mask = allocate_and_upload(backend, &tables.candidate_end_mask)?;
        let candidate_suffix2_mask =
            allocate_and_upload(backend, &tables.candidate_suffix2_mask)?;
        let candidate_suffix3_bloom =
            allocate_and_upload(backend, &tables.candidate_suffix3_bloom)?;

        // The read-write presence buffer: sized for the full max_regions capacity,
        // zeroed per scan over the used prefix only.
        let presence_capacity_words = (max_regions as usize)
            .checked_mul(tables.presence_words as usize)
            .ok_or_else(|| {
                BackendError::new(format!(
                    "resident region-presence capacity {max_regions} regions × {} words/region overflows host usize. Fix: lower max_regions or shard the pattern set.",
                    tables.presence_words
                ))
            })?;
        let presence_capacity_bytes = presence_capacity_words
            .checked_mul(U32_BYTES)
            .ok_or_else(|| {
                BackendError::new(
                    "resident region-presence presence-buffer byte capacity overflows host usize. Fix: lower max_regions or shard the pattern set.".to_string(),
                )
            })?;
        let presence = backend.allocate_resident(presence_capacity_bytes)?;

        Ok(ResidentPresencePipeline {
            program: tables.program,
            haystack,
            transitions,
            output_offsets,
            output_records,
            pattern_lengths,
            presence,
            candidate_end_mask,
            candidate_suffix2_mask,
            candidate_suffix3_bloom,
            haystack_capacity,
            max_regions,
            pattern_count: tables.pattern_count,
            presence_words: tables.presence_words,
            workgroup_x: tables.workgroup_x,
        })
    }
}

/// Allocate a resident buffer sized to `bytes` and upload them once.
fn allocate_and_upload(backend: &dyn VyreBackend, bytes: &[u8]) -> Result<Resource, BackendError> {
    let resource = backend.allocate_resident(bytes.len())?;
    backend.upload_resident(&resource, bytes)?;
    Ok(resource)
}

impl ResidentPresencePipeline {
    /// Scan `haystack` (a coalesced batch with ascending `region_starts` beginning
    /// at 0) against the resident pipeline, decoding the per-region presence bitmap
    /// into caller-owned `out`. Equivalent to
    /// [`GpuLiteralSet::scan_presence_by_region`] but with the immutable tables
    /// already resident (no per-scan table transfer).
    ///
    /// `region_base` is added to every candidate position before the region binary
    /// search; pass `0` for a single-dispatch scan (see
    /// [`GpuLiteralSet::scan_presence_by_region_with_scratch`] for the sharded
    /// meaning). `scratch` reuses the packed-haystack / presence-reset staging
    /// buffer across calls; pass a per-thread `Vec` that lives as long as the scan
    /// loop.
    ///
    /// On return, `out` holds `region_starts.len() × presence_words` packed `u32`
    /// words: bit `p` of region `r`'s row is set iff pattern `p` occurs in region
    /// `r`.
    ///
    /// # Errors
    /// Returns [`BackendError`] when `region_starts` is empty / does not begin at 0,
    /// when `region_count` exceeds the session's `max_regions` cap, when `haystack`
    /// exceeds the session's haystack capacity, or on upload / dispatch / readback
    /// failure. On any error `out` is left cleared (no partial bitmap).
    pub fn scan_into(
        &self,
        backend: &dyn VyreBackend,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        out: &mut Vec<u32>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        out.clear();

        let region_count = u32::try_from(region_starts.len()).map_err(|_| {
            BackendError::new(
                "resident region-presence: region count exceeds u32 GPU ABI".to_string(),
            )
        })?;
        if region_count == 0 {
            return Err(BackendError::new(
                "resident region-presence: region_starts must be non-empty. Fix: pass one start offset per coalesced file, beginning with 0.".to_string(),
            ));
        }
        if region_starts[0] != 0 {
            return Err(BackendError::new(
                "resident region-presence: region_starts[0] must be 0 (the kernel binary-search lower bound). Fix: the first coalesced file must start at offset 0.".to_string(),
            ));
        }
        if region_count > self.max_regions {
            return Err(BackendError::new(format!(
                "resident region-presence batch has {region_count} regions but the session was prepared for at most {}. Fix: raise max_regions in prepare_resident_presence, or dispatch this batch through the per-batch-sized borrowed GpuLiteralSet::scan_presence_by_region (a larger cap would index past the resident presence buffer).",
                self.max_regions
            )));
        }

        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "ResidentPresencePipeline::scan",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;

        // (1) Stage the haystack into the resident buffer (real bytes only; the
        // kernel bounds its cursor with haystack_len so the stale tail is never
        // read).
        dispatch_io::pack_haystack_u32_into(haystack, scratch)?;
        if scratch.len() > self.haystack_capacity {
            return Err(BackendError::new(format!(
                "ResidentPresencePipeline haystack is {} packed byte(s) but the resident buffer holds {}. Fix: raise haystack_capacity_bytes in prepare_resident_presence or shard the haystack.",
                scratch.len(),
                self.haystack_capacity
            )));
        }
        backend.upload_resident_at(&self.haystack, 0, scratch)?;

        // (2) Zero the USED prefix of the resident presence buffer (binding 6 is
        // OR-accumulated by the kernel, so it must arrive zeroed). Rows beyond
        // region_count are never written (the kernel bounds the region index by
        // buf_len(region_starts)) and never read, so only the used prefix needs
        // clearing — the resident analogue of `ResidentRulePipeline`'s 4-byte
        // counter reset. Reusing `scratch` is safe: `upload_resident_at` copies the
        // source synchronously (wgpu `Queue::write_buffer` into the staging belt,
        // CUDA H2D memcpy), so the buffer is free to repurpose the instant the
        // haystack upload above returns.
        let used_words = (region_count as usize)
            .checked_mul(self.presence_words as usize)
            .ok_or_else(|| {
                BackendError::new(
                    "resident region-presence used-word count overflows host usize. Fix: lower the region count or shard the pattern set.".to_string(),
                )
            })?;
        let reset_bytes = used_words.checked_mul(U32_BYTES).ok_or_else(|| {
            BackendError::new(
                "resident region-presence presence-reset byte count overflows host usize. Fix: lower the region count or shard the pattern set.".to_string(),
            )
        })?;
        scratch.clear();
        scratch.resize(reset_bytes, 0);
        backend.upload_resident_at(&self.presence, 0, scratch)?;

        // (3) Bind in program order. The immutable tables and the haystack /
        // presence buffers are resident; the three small per-scan control buffers
        // (haystack_len, region_starts, region_base) stay Borrowed — they are tiny
        // and change every scan, so host replication is cheaper than a resident
        // round-trip (matching `ResidentRulePipeline`'s control-buffer policy).
        let resources = [
            self.haystack.clone(),                                  // 0: haystack
            self.transitions.clone(),                              // 1: transitions
            self.output_offsets.clone(),                           // 2: output_offsets
            self.output_records.clone(),                           // 3: output_records
            self.pattern_lengths.clone(),                          // 4: pattern_lengths
            Resource::Borrowed(haystack_len.to_le_bytes().to_vec()), // 5: haystack_len
            self.presence.clone(),                                 // 6: presence (read_write)
            self.candidate_end_mask.clone(),                       // 7: candidate_end_mask
            self.candidate_suffix2_mask.clone(),                   // 8: candidate_suffix2_mask
            self.candidate_suffix3_bloom.clone(),                  // 9: candidate_suffix3_bloom
            Resource::Borrowed(dispatch_io::u32_words_as_le_bytes(region_starts).into_owned()), // 10: region_starts
            Resource::Borrowed(region_base.to_le_bytes().to_vec()), // 11: region_base
        ];
        debug_assert_eq!(resources.len(), PRESENCE_BY_REGION_BINDINGS);

        let config = dispatch_io::byte_scan_dispatch_config(haystack_len, self.workgroup_x);
        let timed = backend.dispatch_resident_timed(&self.program, &resources, &config)?;

        // The presence buffer is the program's only ReadWrite storage, returned at
        // output index 0 — identical decode to `scan_presence_by_region`.
        let presence_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            0,
            "ResidentPresencePipeline presence buffer",
        )?;
        out.extend(
            presence_bytes
                .chunks_exact(4)
                .take(used_words)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        // Fail CLOSED on a short readback: a presence resource that returns fewer
        // than the used words would otherwise hand back a silently truncated bitmap
        // (some regions reported clean that were never scanned — Law 10).
        if out.len() != used_words {
            let returned = out.len();
            out.clear();
            return Err(BackendError::new(format!(
                "ResidentPresencePipeline presence readback returned {returned} u32 word(s) but the {region_count}-region scan needs {used_words}. Fix: ensure the backend reads back the full binding-6 presence resource."
            )));
        }
        Ok(())
    }

    /// Largest coalesced-file count this session's presence buffer was sized for.
    #[must_use]
    pub fn max_regions(&self) -> u32 {
        self.max_regions
    }

    /// Pattern count (bit width of each per-region presence row).
    #[must_use]
    pub fn pattern_count(&self) -> u32 {
        self.pattern_count
    }

    /// Presence bitmap `u32` words per region.
    #[must_use]
    pub fn presence_words(&self) -> u32 {
        self.presence_words
    }

    /// Padded byte capacity of the resident haystack buffer.
    #[must_use]
    pub fn haystack_capacity(&self) -> usize {
        self.haystack_capacity
    }

    /// Release every resident resource this session owns.
    ///
    /// Call this before the backend's device context is dropped to reclaim the
    /// resident allocations eagerly; otherwise they are reclaimed when the backend
    /// tears down. The session is consumed.
    ///
    /// # Errors
    /// Returns the first [`BackendError`] from freeing a resource; remaining
    /// resources are still attempted.
    pub fn free(self, backend: &dyn VyreBackend) -> Result<(), BackendError> {
        let mut first_err = None;
        for resource in [
            self.haystack,
            self.transitions,
            self.output_offsets,
            self.output_records,
            self.pattern_lengths,
            self.presence,
            self.candidate_end_mask,
            self.candidate_suffix2_mask,
            self.candidate_suffix3_bloom,
        ] {
            if let Err(error) = backend.free_resident(resource) {
                first_err.get_or_insert(error);
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use vyre::DispatchConfig as Config;
    use vyre_driver::TimedDispatchResult;

    // pattern_id order matches the integration corpus: key=0 .. api=7.
    const LITERALS: &[&[u8]] = &[
        b"key", b"token", b"secret", b"AKIA", b"ghp_", b"sk_live_", b"password", b"api",
    ];

    /// Mock backend that records resident traffic and returns a canned presence
    /// buffer, so the host orchestration (seven-table-upload-once, per-scan haystack
    /// staging + presence reset, 12-binding dispatch, decode) is validated without a
    /// GPU. Real GPU resident-vs-borrowed parity is asserted in the integration
    /// suite where a live wgpu backend is available. `VyreBackend` requires
    /// `Send + Sync`, so the counters use atomics / `Mutex`.
    struct MockResidentBackend {
        next_id: AtomicU64,
        /// (handle_id, byte_len) for every allocate_resident call.
        allocations: Mutex<Vec<(u64, usize)>>,
        /// Number of full uploads (immutable table uploads) seen.
        full_uploads: AtomicUsize,
        /// Number of ranged uploads (haystack stage + presence reset) seen.
        ranged_uploads: AtomicUsize,
        /// Byte lengths of every ranged upload, in order (haystack, reset, ...).
        ranged_upload_lens: Mutex<Vec<usize>>,
        /// Canned presence-buffer bytes returned at output index 0.
        presence_buffer: Vec<u8>,
    }

    impl MockResidentBackend {
        fn new(presence_buffer: Vec<u8>) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                allocations: Mutex::new(Vec::new()),
                full_uploads: AtomicUsize::new(0),
                ranged_uploads: AtomicUsize::new(0),
                ranged_upload_lens: Mutex::new(Vec::new()),
                presence_buffer,
            }
        }
    }

    impl vyre::backend::private::Sealed for MockResidentBackend {}

    impl VyreBackend for MockResidentBackend {
        fn id(&self) -> &'static str {
            "mock-resident-presence"
        }

        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _config: &Config,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("resident path does not use borrowed dispatch")
        }

        fn allocate_resident(&self, byte_len: usize) -> Result<Resource, BackendError> {
            let handle = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.allocations
                .lock()
                .expect("mock allocations mutex")
                .push((handle, byte_len));
            Ok(Resource::Resident(handle))
        }

        fn upload_resident(&self, _resource: &Resource, _bytes: &[u8]) -> Result<(), BackendError> {
            self.full_uploads.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn upload_resident_at(
            &self,
            _resource: &Resource,
            _dst_offset_bytes: usize,
            bytes: &[u8],
        ) -> Result<(), BackendError> {
            self.ranged_uploads.fetch_add(1, Ordering::Relaxed);
            self.ranged_upload_lens
                .lock()
                .expect("mock ranged-upload mutex")
                .push(bytes.len());
            Ok(())
        }

        fn free_resident(&self, _resource: Resource) -> Result<(), BackendError> {
            Ok(())
        }

        fn dispatch_resident_timed(
            &self,
            _program: &Program,
            resources: &[Resource],
            config: &Config,
        ) -> Result<TimedDispatchResult, BackendError> {
            // Contract checks the consumer relies on:
            assert_eq!(
                resources.len(),
                PRESENCE_BY_REGION_BINDINGS,
                "region-presence binds twelve buffers"
            );
            // The seven immutable tables + the haystack + the presence buffer are
            // resident; only the three tiny control buffers stay borrowed.
            for resident_idx in [0usize, 1, 2, 3, 4, 6, 7, 8, 9] {
                assert!(
                    matches!(resources[resident_idx], Resource::Resident(_)),
                    "binding {resident_idx} must be resident, not re-uploaded"
                );
            }
            for borrowed_idx in [5usize, 10, 11] {
                assert!(
                    matches!(resources[borrowed_idx], Resource::Borrowed(_)),
                    "binding {borrowed_idx} (a per-scan control buffer) must be borrowed"
                );
            }
            assert!(
                config.grid_override.is_some(),
                "resident region-presence scan must supply a byte-scan grid override"
            );
            Ok(TimedDispatchResult {
                outputs: vec![self.presence_buffer.clone()],
                wall_ns: 0,
                device_ns: None,
                enqueue_ns: None,
                wait_ns: None,
            })
        }
    }

    /// Decode one single-word region row into the set of pattern ids whose bit is set.
    fn present_ids(word: u32, pattern_count: u32) -> BTreeSet<u32> {
        (0..pattern_count).filter(|&p| (word >> p) & 1 == 1).collect()
    }

    #[test]
    fn prepare_uploads_tables_once_then_scans_transfer_only_haystack_and_reset() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let pattern_count = LITERALS.len() as u32;
        assert_eq!(pattern_count, 8);
        // 8 patterns -> 1 presence word/region. max_regions = 4 -> capacity 4 words.
        // Canned presence: planted prefix [row0,row1,row2] + a stale 4th word the
        // 3-region decode must ignore.
        let row0 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 7); // {key,token,secret,AKIA,api}
        let row1 = (1 << 4) | (1 << 5) | (1 << 6); // {ghp_,sk_live_,password}
        let row2 = 0u32; // {}
        let stale = 0xDEAD_BEEFu32;
        let mut canned = Vec::new();
        for w in [row0, row1, row2, stale] {
            canned.extend_from_slice(&w.to_le_bytes());
        }
        let backend = MockResidentBackend::new(canned);

        let session = matcher
            .prepare_resident_presence(&backend, 4096, 4)
            .expect("mock backend supports resident allocation");

        // Nine resident allocations: haystack + 7 immutable tables + presence.
        assert_eq!(
            backend.allocations.lock().unwrap().len(),
            9,
            "haystack + 7 immutable tables + presence buffer"
        );
        // The presence buffer (last allocation) is sized for max_regions × words × 4
        // = 4 × 1 × 4 = 16 bytes.
        assert_eq!(
            backend.allocations.lock().unwrap()[8].1,
            4 * 1 * U32_BYTES,
            "presence buffer sized for max_regions capacity"
        );
        // The seven immutable tables are uploaded exactly once, at prepare time.
        assert_eq!(
            backend.full_uploads.load(Ordering::Relaxed),
            7,
            "seven immutable tables uploaded once each"
        );
        assert_eq!(backend.ranged_uploads.load(Ordering::Relaxed), 0);

        // A 3-region coalesced batch (regions at 0 / 7 / 12; first start == 0).
        let haystack = b"aaa\nbbbb\nccc\n";
        let region_starts = [0u32, 4, 9];
        let mut out = Vec::new();
        let mut scratch = Vec::new();
        for _ in 0..3 {
            session
                .scan_into(&backend, haystack, &region_starts, 0, &mut out, &mut scratch)
                .expect("resident region-presence scan decodes canned bitmap");
        }

        // Decode parity: the canned planted prefix surfaces; the stale 4th word is
        // never observed by the 3-region decode.
        assert_eq!(out, vec![row0, row1, row2], "3 regions × 1 word, stale tail ignored");
        assert_eq!(present_ids(out[0], pattern_count), BTreeSet::from([0, 1, 2, 3, 7]));
        assert_eq!(present_ids(out[1], pattern_count), BTreeSet::from([4, 5, 6]));
        assert_eq!(present_ids(out[2], pattern_count), BTreeSet::new());

        // No further full uploads after prepare; each scan does exactly two ranged
        // uploads (haystack stage + presence reset) — the tables never move again.
        assert_eq!(
            backend.full_uploads.load(Ordering::Relaxed),
            7,
            "immutable tables re-uploaded mid-loop"
        );
        assert_eq!(
            backend.ranged_uploads.load(Ordering::Relaxed),
            6,
            "3 scans × (haystack stage + presence reset)"
        );
        // Every presence reset uploads exactly used_words × 4 = 3 × 1 × 4 = 12 bytes
        // (the used prefix only, not the full 16-byte capacity).
        let reset_lens: Vec<usize> = backend
            .ranged_upload_lens
            .lock()
            .unwrap()
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .collect();
        assert_eq!(
            reset_lens,
            vec![12, 12, 12],
            "each presence reset zeroes only the 3-region used prefix"
        );
    }

    #[test]
    fn scan_rejects_region_count_over_the_max_regions_cap() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        // Canned presence is irrelevant; the cap guard fires before dispatch.
        let backend = MockResidentBackend::new(vec![0u8; 4]);
        let session = matcher
            .prepare_resident_presence(&backend, 4096, 2)
            .expect("prepare with a 2-region cap");

        let haystack = b"a\nb\nc\n";
        let region_starts = [0u32, 2, 4]; // 3 regions > cap of 2
        let mut out = vec![999];
        let mut scratch = Vec::new();
        let err = session
            .scan_into(&backend, haystack, &region_starts, 0, &mut out, &mut scratch)
            .expect_err("a batch over the resident region cap must error, not truncate");
        assert!(
            err.to_string().contains("session was prepared for at most 2") && out.is_empty(),
            "cap error must name the limit and expose no partial bitmap: {err}"
        );
        // The over-cap batch must never reach the device.
        assert_eq!(
            backend.ranged_uploads.load(Ordering::Relaxed),
            0,
            "rejected batch must not stage any resident upload"
        );
    }

    #[test]
    fn scan_rejects_haystack_larger_than_resident_capacity() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let backend = MockResidentBackend::new(vec![0u8; 4]);
        let session = matcher
            .prepare_resident_presence(&backend, 8, 4)
            .expect("prepare with an 8-byte haystack capacity");

        let mut out = Vec::new();
        let mut scratch = Vec::new();
        let region_starts = [0u32];
        let err = session
            .scan_into(&backend, &[b'a'; 64], &region_starts, 0, &mut out, &mut scratch)
            .expect_err("64-byte haystack must not fit an 8-byte resident buffer");
        assert!(
            err.to_string().contains("resident buffer holds") && out.is_empty(),
            "capacity error must name the limit and expose no stale bitmap: {err}"
        );
    }

    #[test]
    fn prepare_rejects_zero_max_regions() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let backend = MockResidentBackend::new(vec![0u8; 4]);
        let err = matcher
            .prepare_resident_presence(&backend, 4096, 0)
            .expect_err("max_regions = 0 cannot size the presence buffer");
        assert!(
            err.to_string().contains("max_regions must be >= 1"),
            "zero-cap error must explain the cause: {err}"
        );
    }
}
