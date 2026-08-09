//! Authenticated resident execution for literal-set region-presence scans.
//!
//! [`ResidentPresencePipeline`] compiles one canonical presence artifact,
//! materializes it for a registered backend, uploads immutable DFA and prefilter
//! tables once, and reuses the resulting artifact instance across submissions.
//! Per scan it uploads the haystack and controls, clears the used presence prefix,
//! submits typed resident bindings, and decodes the compiler-owned output.

use vyre_driver::{BackendError, Resource, TimedDispatchResult};

use super::dispatch_io;
use super::literal_set::{decode_presence_words_into, GpuLiteralSet};

const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Number of buffer bindings in the region-presence program (see
/// [`super::literal_set::GpuLiteralSet::build_presence_by_region_dispatch`]).
const PRESENCE_BY_REGION_BINDINGS: usize = 12;

/// An authenticated region-presence artifact with immutable tables in resident resources.
///
/// Construct with [`GpuLiteralSet::prepare_resident_presence`]. Call
/// [`free`](Self::free) to release the twelve resident allocations eagerly.
/// The artifact session and opaque resource handles are `Send + Sync`.
pub struct ResidentPresencePipeline {
    artifact: crate::artifact_session::ScanArtifactSession,
    resource_names: Vec<String>,
    haystack: Resource,
    transitions: Resource,
    output_offsets: Resource,
    output_records: Resource,
    pattern_lengths: Resource,
    presence: Resource,
    candidate_end_mask: Resource,
    candidate_suffix2_mask: Resource,
    candidate_suffix3_bloom: Resource,
    haystack_len_buf: Resource,
    region_starts_buf: Resource,
    region_base_buf: Resource,
    haystack_capacity: usize,
    max_regions: u32,
    pattern_count: u32,
    presence_words: u32,
    workgroup_x: u32,
}

// Artifact sessions and resident resource handles are `Send + Sync`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ResidentPresencePipeline>;
};

impl GpuLiteralSet {
    /// Compile and materialize a region-presence artifact for `backend_id`, then
    /// upload this matcher's immutable tables into resident resources.
    ///
    /// `haystack_capacity_bytes` is the largest coalesced haystack accepted by
    /// the session. `max_regions` sizes the presence and region-start resources.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when validation, registration, compilation,
    /// materialization, allocation, or upload fails.
    pub fn prepare_resident_presence(
        &self,
        backend_id: &str,
        haystack_capacity_bytes: usize,
        max_regions: u32,
    ) -> Result<ResidentPresencePipeline, BackendError> {
        let tables = self.resident_presence_tables(max_regions)?;
        let resource_names = tables
            .program
            .buffers()
            .iter()
            .map(|buffer| buffer.name().to_string())
            .collect::<Vec<_>>();
        if resource_names.len() != PRESENCE_BY_REGION_BINDINGS {
            return Err(BackendError::new(format!(
                "resident presence artifact declares {} resources, expected {PRESENCE_BY_REGION_BINDINGS}. Fix: keep the presence Program ABI synchronized with ResidentPresencePipeline.",
                resource_names.len()
            )));
        }
        let registration = vyre_driver::backend::backend_registration(backend_id)?;
        let artifact =
            crate::artifact_session::ScanArtifactSession::compile(&tables.program, registration)
                .map_err(crate::artifact_session::as_backend_error)?;
        let haystack_capacity = dispatch_io::haystack_padded_u32_byte_len(haystack_capacity_bytes)?;
        let haystack = artifact
            .allocate_resident(haystack_capacity)
            .map_err(crate::artifact_session::as_backend_error)?;
        let transitions = allocate_and_upload(&artifact, &tables.transitions)?;
        let output_offsets = allocate_and_upload(&artifact, &tables.output_offsets)?;
        let output_records = allocate_and_upload(&artifact, &tables.output_records)?;
        let pattern_lengths = allocate_and_upload(&artifact, &tables.pattern_lengths)?;
        let candidate_end_mask = allocate_and_upload(&artifact, &tables.candidate_end_mask)?;
        let candidate_suffix2_mask =
            allocate_and_upload(&artifact, &tables.candidate_suffix2_mask)?;
        let candidate_suffix3_bloom =
            allocate_and_upload(&artifact, &tables.candidate_suffix3_bloom)?;
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
        let presence = artifact
            .allocate_resident(presence_capacity_bytes)
            .map_err(crate::artifact_session::as_backend_error)?;
        let region_starts_capacity_bytes =
            (max_regions as usize).checked_mul(U32_BYTES).ok_or_else(|| {
                BackendError::new(
                    "resident region-presence region-starts byte capacity overflows host usize. Fix: lower max_regions.".to_string(),
                )
            })?;
        let haystack_len_buf = artifact
            .allocate_resident(U32_BYTES)
            .map_err(crate::artifact_session::as_backend_error)?;
        let region_starts_buf = artifact
            .allocate_resident(region_starts_capacity_bytes)
            .map_err(crate::artifact_session::as_backend_error)?;
        let region_base_buf = artifact
            .allocate_resident(U32_BYTES)
            .map_err(crate::artifact_session::as_backend_error)?;

        Ok(ResidentPresencePipeline {
            artifact,
            resource_names,
            haystack,
            transitions,
            output_offsets,
            output_records,
            pattern_lengths,
            presence,
            candidate_end_mask,
            candidate_suffix2_mask,
            candidate_suffix3_bloom,
            haystack_len_buf,
            region_starts_buf,
            region_base_buf,
            haystack_capacity,
            max_regions,
            pattern_count: tables.pattern_count,
            presence_words: tables.presence_words,
            workgroup_x: tables.workgroup_x,
        })
    }
}

/// Allocate a resident buffer sized to `bytes` and upload them once.
fn allocate_and_upload(
    artifact: &crate::artifact_session::ScanArtifactSession,
    bytes: &[u8],
) -> Result<Resource, BackendError> {
    let resource = artifact
        .allocate_resident(bytes.len())
        .map_err(crate::artifact_session::as_backend_error)?;
    artifact
        .upload_resident(&resource, bytes)
        .map_err(crate::artifact_session::as_backend_error)?;
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
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        out: &mut Vec<u32>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        self.scan_into_timed(haystack, region_starts, region_base, out, scratch)?;
        Ok(())
    }

    /// Like [`scan_into`](Self::scan_into) but returns the dispatch's
    /// [`TimedDispatchResult`] so a consumer or benchmark can attribute the
    /// per-scan cost between the GPU kernel (`device_ns`) and host-side
    /// staging/readback (`wall_ns - device_ns`). The decoded per-region presence
    /// bitmap is written to `out` identically to [`scan_into`](Self::scan_into);
    /// the returned result's `outputs` are the same raw presence bytes already
    /// decoded into `out`.
    ///
    /// # Errors
    /// Same as [`scan_into`](Self::scan_into).
    pub fn scan_into_timed(
        &self,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        out: &mut Vec<u32>,
        scratch: &mut Vec<u8>,
    ) -> Result<TimedDispatchResult, BackendError> {
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
        self.artifact
            .upload_resident_at(&self.haystack, 0, scratch)
            .map_err(crate::artifact_session::as_backend_error)?;

        // (2) Zero the USED prefix of the resident presence buffer (binding 6 is
        // OR-accumulated by the kernel, so it must arrive zeroed). Rows beyond
        // region_count are never written (the kernel bounds the region index by
        // buf_len(region_starts)) and never read, so only the used prefix needs
        // clearing, the resident analogue of `ResidentScanSession`'s 4-byte
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
        self.artifact
            .upload_resident_at(&self.presence, 0, scratch)
            .map_err(crate::artifact_session::as_backend_error)?;

        // (3) Stage the three per-scan control buffers. They MUST be resident, not
        // borrowed: the CUDA resident dispatch resolves every binding to a resident
        // handle and rejects a borrowed mix (`cuda_compiled_persistent_borrowed_resource`),
        // so an all-resident dispatch is the only form portable across wgpu AND CUDA
        // (a downstream consumer's backend). haystack_len and region_base are one u32 each.
        self.artifact
            .upload_resident_at(&self.haystack_len_buf, 0, &haystack_len.to_le_bytes())
            .map_err(crate::artifact_session::as_backend_error)?;
        self.artifact
            .upload_resident_at(&self.region_base_buf, 0, &region_base.to_le_bytes())
            .map_err(crate::artifact_session::as_backend_error)?;

        // region_starts is a FIXED `max_regions`-sized resident buffer so its
        // `buf_len`: the kernel's live region count, does not change with the
        // batch. The real starts fill [0, region_count); the tail
        // [region_count, max_regions) is padded with `u32::MAX`, a sentinel strictly
        // greater than any candidate position (positions are bounded by the scan
        // size << u32::MAX), so the region binary search never maps a hit to a
        // padding row. Those rows stay untouched and are never decoded, the result
        // for the real regions is identical to a `region_count`-length region_starts.
        // Reusing `scratch` is safe (synchronous upload copy, as above).
        scratch.clear();
        let region_starts_words = self.max_regions as usize;
        scratch.reserve(region_starts_words.saturating_mul(U32_BYTES));
        for &start in region_starts {
            scratch.extend_from_slice(&start.to_le_bytes());
        }
        for _ in (region_count as usize)..region_starts_words {
            scratch.extend_from_slice(&u32::MAX.to_le_bytes());
        }
        self.artifact
            .upload_resident_at(&self.region_starts_buf, 0, scratch)
            .map_err(crate::artifact_session::as_backend_error)?;

        let resources = [
            (self.resource_names[0].as_str(), &self.haystack),
            (self.resource_names[1].as_str(), &self.transitions),
            (self.resource_names[2].as_str(), &self.output_offsets),
            (self.resource_names[3].as_str(), &self.output_records),
            (self.resource_names[4].as_str(), &self.pattern_lengths),
            (self.resource_names[5].as_str(), &self.haystack_len_buf),
            (self.resource_names[6].as_str(), &self.presence),
            (self.resource_names[7].as_str(), &self.candidate_end_mask),
            (self.resource_names[8].as_str(), &self.candidate_suffix2_mask),
            (self.resource_names[9].as_str(), &self.candidate_suffix3_bloom),
            (self.resource_names[10].as_str(), &self.region_starts_buf),
            (self.resource_names[11].as_str(), &self.region_base_buf),
        ];
        let grid = dispatch_io::byte_scan_dispatch_config(haystack_len, self.workgroup_x)
            .grid_override
            .ok_or_else(|| {
                BackendError::new("resident presence geometry omitted its invocation grid")
            })?;
        let timed = self
            .artifact
            .submit_resident_timed(&resources, grid)
            .map_err(crate::artifact_session::as_backend_error)?;

        // The presence buffer is the program's only ReadWrite storage, returned at
        // output index 0 (identical decode to `scan_presence_by_region`).
        let presence_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            0,
            "ResidentPresencePipeline presence buffer",
        )?;
        // The single region-presence wire decoder (shared with the sync / async /
        // prepared / fused paths in literal_set), filling the caller's `out`.
        decode_presence_words_into(presence_bytes, used_words, out);
        // Fail CLOSED on a short readback: a presence resource that returns fewer
        // than the used words would otherwise hand back a silently truncated bitmap
        // (some regions reported clean that were never scanned. Law 10).
        if out.len() != used_words {
            let returned = out.len();
            out.clear();
            return Err(BackendError::new(format!(
                "ResidentPresencePipeline presence readback returned {returned} u32 word(s) but the {region_count}-region scan needs {used_words}. Fix: ensure the backend reads back the full binding-6 presence resource."
            )));
        }
        Ok(timed)
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
    pub fn free(self) -> Result<(), BackendError> {
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
            self.haystack_len_buf,
            self.region_starts_buf,
            self.region_base_buf,
        ] {
            if let Err(error) = self.artifact.free_resident(resource) {
                first_err.get_or_insert_with(|| crate::artifact_session::as_backend_error(error));
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}
