//! Resident-buffer dispatch for [`ScanSession`] (the regex/NFA mega-scan path).
//!
//! [`ScanSession::scan`](crate::session::ScanSession::scan) transfers immutable
//! NFA tables for each borrowed submission. [`ResidentScanSession`] compiles one
//! canonical artifact, materializes it for a registered backend, uploads the
//! transition and epsilon tables once, and reuses that artifact instance across
//! submissions.
//!
//! Each scan uploads the haystack, resets the hit counter, updates two control
//! values, submits typed resident bindings, and decodes the canonical hit buffer.
//! The match wire format remains `(pattern_id, start, end)` triples after a u32
//! count prefix.
//!
//! Materialization, allocation, upload, submission, and release all use one
//! artifact materializer generation. Unsupported resident operations fail
//! through the registered backend rather than selecting a raw fallback route.

use vyre_driver::{BackendError, Resource};
use vyre_foundation::match_result::Match;

use super::dispatch_io;
use super::session::{hit_buffer_byte_len, ScanSession};

/// An authenticated scan artifact with immutable NFA tables in resident resources.
///
/// Construct with [`ScanSession::prepare_resident`]. Call [`free`](Self::free)
/// to release its resident allocations eagerly.
pub struct ResidentScanSession {
    artifact: crate::artifact_session::ScanArtifactSession,
    resource_names: Vec<String>,
    haystack: Resource,
    transition: Resource,
    epsilon: Resource,
    hits: Resource,
    haystack_len_buf: Resource,
    max_scan_bytes_buf: Resource,
    haystack_capacity: usize,
    max_matches: u32,
}

// Artifact sessions and resident resource handles are `Send + Sync`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ResidentScanSession>;
};

impl ScanSession {
    /// Compile and materialize this pipeline for `backend_id`, then upload its
    /// immutable NFA tables into resident resources.
    ///
    /// `haystack_capacity_bytes` is the largest haystack accepted by this
    /// session. `max_matches` sizes the resident hit buffer and caps decoding.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when backend registration, compilation,
    /// materialization, allocation, or upload fails.
    pub fn prepare_resident(
        &self,
        backend_id: &str,
        haystack_capacity_bytes: usize,
        max_matches: u32,
    ) -> Result<ResidentScanSession, BackendError> {
        let haystack_capacity = dispatch_io::haystack_padded_u32_byte_len(haystack_capacity_bytes)?;
        let program = self.compiled.program.clone();
        if let Some(input_decl) = program.buffers().iter().find(|decl| decl.binding == 0) {
            if let Some(required_bytes) = input_decl.static_byte_len().map_err(BackendError::new)? {
                if haystack_capacity < required_bytes {
                    return Err(BackendError::new(format!(
                        "ResidentScanSession::prepare_resident: the NFA program's input buffer statically declares {required_bytes} byte(s), but the resident haystack capacity is {haystack_capacity}. Fix: raise haystack_capacity_bytes or rebuild the ScanSession with a smaller input length."
                    )));
                }
            }
        }
        let resource_names = program
            .buffers()
            .iter()
            .map(|buffer| buffer.name().to_string())
            .collect::<Vec<_>>();
        if resource_names.len() != 6 {
            return Err(BackendError::new(format!(
                "resident NFA artifact declares {} resources, expected 6. Fix: keep the NFA Program ABI synchronized with ResidentScanSession.",
                resource_names.len()
            )));
        }
        let registration = vyre_driver::backend::backend_registration(backend_id)?;
        let artifact = crate::artifact_session::ScanArtifactSession::compile(&program, registration)
            .map_err(crate::artifact_session::as_backend_error)?;
        let haystack = artifact
            .allocate_resident(haystack_capacity)
            .map_err(crate::artifact_session::as_backend_error)?;
        let transition_bytes = dispatch_io::u32_words_as_le_bytes(&self.compiled.transition_table);
        let transition = artifact
            .allocate_resident(transition_bytes.len())
            .map_err(crate::artifact_session::as_backend_error)?;
        artifact
            .upload_resident(&transition, transition_bytes.as_ref())
            .map_err(crate::artifact_session::as_backend_error)?;
        let epsilon_bytes = dispatch_io::u32_words_as_le_bytes(&self.compiled.epsilon_table);
        let epsilon = artifact
            .allocate_resident(epsilon_bytes.len())
            .map_err(crate::artifact_session::as_backend_error)?;
        artifact
            .upload_resident(&epsilon, epsilon_bytes.as_ref())
            .map_err(crate::artifact_session::as_backend_error)?;
        let hits = artifact
            .allocate_resident(hit_buffer_byte_len(max_matches)?)
            .map_err(crate::artifact_session::as_backend_error)?;
        let haystack_len_buf = artifact
            .allocate_resident(std::mem::size_of::<u32>())
            .map_err(crate::artifact_session::as_backend_error)?;
        let max_scan_bytes_buf = artifact
            .allocate_resident(std::mem::size_of::<u32>())
            .map_err(crate::artifact_session::as_backend_error)?;

        Ok(ResidentScanSession {
            artifact,
            resource_names,
            haystack,
            transition,
            epsilon,
            hits,
            haystack_len_buf,
            max_scan_bytes_buf,
            haystack_capacity,
            max_matches,
        })
    }
}

impl ResidentScanSession {
    /// Scan `haystack` against the resident pipeline, decoding matches into
    /// caller-owned `matches`. Equivalent to [`ScanSession::scan`] but with the
    /// NFA tables already resident (no per-scan table transfer).
    ///
    /// `scratch` reuses the packed-haystack staging buffer across calls; pass a
    /// per-thread `Vec` that lives as long as the scan loop.
    ///
    /// Walks every workgroup to end-of-haystack (`max_scan_bytes = u32::MAX`),
    /// matching [`ScanSession::scan`]. Use [`scan_bounded_into`](Self::scan_bounded_into)
    /// to cap per-workgroup work to the longest possible match length.
    ///
    /// # Errors
    /// Returns [`BackendError`] on upload, dispatch, or readback failure, or
    /// when `haystack` exceeds the session's configured capacity.
    pub fn scan_into(
        &self,
        haystack: &[u8],
        matches: &mut Vec<Match>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        self.scan_bounded_into(haystack, u32::MAX, matches, scratch)
    }

    /// Per-workgroup-bounded resident scan. See [`ScanSession::scan_bounded`]
    /// for the bound's semantics (O(N × max_scan_bytes) instead of O(N²)).
    ///
    /// # Errors
    /// Same as [`scan_into`](Self::scan_into).
    pub fn scan_bounded_into(
        &self,
        haystack: &[u8],
        max_scan_bytes: u32,
        matches: &mut Vec<Match>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        matches.clear();
        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "ResidentScanSession::scan",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;
        dispatch_io::pack_haystack_u32_into(haystack, scratch)?;
        if scratch.len() > self.haystack_capacity {
            return Err(BackendError::new(format!(
                "ResidentScanSession haystack is {} packed byte(s) but the resident buffer holds {}. Fix: raise haystack_capacity_bytes in prepare_resident or shard the haystack.",
                scratch.len(),
                self.haystack_capacity
            )));
        }
        self.artifact
            .upload_resident_at(&self.haystack, 0, scratch)
            .map_err(crate::artifact_session::as_backend_error)?;
        self.artifact
            .upload_resident_at(&self.hits, 0, &0u32.to_le_bytes())
            .map_err(crate::artifact_session::as_backend_error)?;
        self.artifact
            .upload_resident_at(&self.haystack_len_buf, 0, &haystack_len.to_le_bytes())
            .map_err(crate::artifact_session::as_backend_error)?;
        self.artifact
            .upload_resident_at(
                &self.max_scan_bytes_buf,
                0,
                &max_scan_bytes.to_le_bytes(),
            )
            .map_err(crate::artifact_session::as_backend_error)?;
        let resources = [
            (self.resource_names[0].as_str(), &self.haystack),
            (self.resource_names[1].as_str(), &self.transition),
            (self.resource_names[2].as_str(), &self.epsilon),
            (self.resource_names[3].as_str(), &self.hits),
            (self.resource_names[4].as_str(), &self.haystack_len_buf),
            (self.resource_names[5].as_str(), &self.max_scan_bytes_buf),
        ];
        let timed = self
            .artifact
            .submit_resident_timed(&resources)
            .map_err(crate::artifact_session::as_backend_error)?;
        let hit_bytes =
            dispatch_io::try_output_bytes(&timed.outputs, 0, "ResidentScanSession hit buffer")?;
        let count = dispatch_io::try_read_u32_prefix(hit_bytes, "ResidentScanSession hit buffer")?;
        dispatch_io::try_unpack_match_triples_capped_into(
            &hit_bytes[4..],
            count,
            self.max_matches,
            "ResidentScanSession hit buffer",
            matches,
        )
    }

    /// The match cap this session's resident hit buffer was sized for.
    #[must_use]
    pub fn max_matches(&self) -> u32 {
        self.max_matches
    }

    /// Padded byte capacity of the resident haystack buffer.
    #[must_use]
    pub fn haystack_capacity(&self) -> usize {
        self.haystack_capacity
    }

    /// Release every resident resource this session owns.
    ///
    /// The owning artifact materializer remains alive through cleanup. The
    /// session is consumed.
    ///
    /// # Errors
    /// Returns the first [`BackendError`] from freeing a resource; remaining
    /// resources are still attempted.
    pub fn free(self) -> Result<(), BackendError> {
        let mut first_err = None;
        for resource in [
            self.haystack,
            self.transition,
            self.epsilon,
            self.hits,
            self.haystack_len_buf,
            self.max_scan_bytes_buf,
        ] {
            if let Err(error) = self.artifact.free_resident(resource) {
                first_err.get_or_insert_with(|| crate::artifact_session::as_backend_error(error));
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}
