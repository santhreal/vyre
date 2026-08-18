//! Program dispatch seam - the boundary between code that builds a vyre
//! `Program` and a backend that can run one.
//!
//! A caller encodes its own data into buffers, builds a `Program` that computes
//! the answer, and asks a [`crate::program_dispatch::ProgramDispatcher`] to run it. The returned bytes
//! are the result. Foundation owns the seam because it is stated entirely in
//! `Program` and buffer terms and because every layer above needs it: the pass
//! engine replays optimizer passes through it, the composition library
//! runs its device-resident solvers through it, and each concrete backend
//! implements it.
//!
//! The seam sits below the composition library, which needs it as much as the
//! pass engine does, and it is stated without reference to either. The host-side
//! byte marshalling that goes with it is `vyre_libs::dispatch_buffers`, which
//! cannot live here because it delegates to `vyre_primitives::wire`.

mod resident;

pub use self::resident::{
    DispatchError, ResidentDispatchStep, ResidentReadRange, ResidentStaticBufferSet,
};

use crate::ir::{BufferAccess, BufferDecl, Program};

/// The buffers a [`ProgramDispatcher`] returns, in declared order.
///
/// The trait contract is "the declared outputs in the same canonical order", which
/// means every writable storage buffer. Workgroup scratch is never a dispatch
/// output, so it is excluded.
///
/// This is the single owner of that rule. A dispatcher that hardcodes its own
/// output list instead of deriving it from the program silently disagrees with the
/// program the moment a buffer is added or removed. That is exactly how a
/// three-output oracle came to be paired with a two-output DCE program: the oracle
/// asserted a layout the program no longer had, and nothing compared the two.
#[must_use]
pub fn declared_dispatch_outputs(program: &Program) -> Vec<&BufferDecl> {
    program
        .buffers()
        .iter()
        .filter(|decl| {
            matches!(
                decl.access(),
                BufferAccess::ReadWrite | BufferAccess::WriteOnly
            )
        })
        .collect()
}

/// Run a vyre Program with byte inputs, return byte outputs in the
/// Program's declared output order.
///
/// This is the canonical dispatch boundary. Every implementation is a driver
/// crate; the reference oracle is the only one that evaluates on the host.
pub trait ProgramDispatcher {
    /// Dispatch `program` with the given byte inputs (one `Vec<u8>`
    /// per declared input buffer in canonical buffer order). Returns
    /// the declared outputs in the same canonical order.
    ///
    /// `grid_override` lets parallel kernels dispatch enough
    /// workgroups to cover their input. `None` means "use the
    /// backend's default grid" (typically `[1, 1, 1]`), which is what
    /// sequential single-thread Programs want. Parallel passes
    /// compute `Some([ceil(work/wg_x), 1, 1])` based on the input
    /// size and their declared workgroup_size.
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError>;

    /// Whether this dispatcher supports the persistent-resident path.
    /// Default: false, overridden by a dispatcher that keeps device buffers
    /// alive across launches. The orchestrator uses this to decide whether to
    /// take the persistent fast-path (encode arena once, upload once, dispatch
    /// many, read back once) or the non-resident per-call path.
    fn supports_persistent(&self) -> bool {
        false
    }

    /// Device/lowering feature bits that affect reusable plan identity.
    ///
    /// Backends with feature-dependent lowering must override this so
    /// self-substrate plan caches cannot replay a Program shape prepared for a
    /// different hardware/lowering capability set. Test-only and reference
    /// dispatchers keep the zero default because they do not specialize plans by
    /// device.
    fn device_feature_cache_key(&self) -> u64 {
        0
    }

    /// Allocate a backend-resident buffer. Returns an opaque u64
    /// handle. Callers must `free_resident` to release.
    fn alloc_resident(&self, _byte_len: usize) -> Result<u64, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: this dispatcher does not implement the persistent path; \
             use `dispatch` instead, or wire the resident-buffer methods."
                .to_string(),
        ))
    }

    /// Allocate a logical group of resident buffers and roll back partial state
    /// if any allocation fails.
    fn alloc_resident_many(&self, byte_lens: &[usize]) -> Result<Vec<u64>, DispatchError> {
        let mut handles = Vec::new();
        handles.try_reserve(byte_lens.len()).map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: reserve resident handle group before allocation; requested {} buffer(s): {error}.",
                byte_lens.len()
            ))
        })?;
        for (index, &byte_len) in byte_lens.iter().enumerate() {
            match self.alloc_resident(byte_len) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    let allocation_error = error.to_string();
                    if let Err(free_error) = free_resident_handles(
                        self,
                        &handles,
                        "resident grouped allocation rollback",
                    ) {
                        return Err(DispatchError::BackendError(format!(
                            "Fix: resident grouped allocation failed at buffer {index} after {} partial allocation(s): {allocation_error}; rollback also failed: {free_error}.",
                            handles.len()
                        )));
                    }
                    return Err(error);
                }
            }
        }
        Ok(handles)
    }

    /// Upload host bytes into a resident buffer.
    fn upload_resident(&self, _handle: u64, _bytes: &[u8]) -> Result<(), DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: dispatcher does not implement upload_resident.".to_string(),
        ))
    }

    /// Upload several resident buffers with one backend fence when supported.
    fn upload_resident_many(&self, uploads: &[(u64, &[u8])]) -> Result<(), DispatchError> {
        for &(handle, bytes) in uploads {
            self.upload_resident(handle, bytes)?;
        }
        Ok(())
    }

    /// Acquire resident handles for immutable payloads.
    ///
    /// Portable default behavior allocates and uploads exactly like
    /// `alloc_resident` + `upload_resident_many`, then returns
    /// `retained_by_dispatcher = false` so release frees the buffers. A
    /// dispatcher that content-addresses immutable optimizer buffers overrides
    /// this and skips the host-to-device traffic on warmed identical programs.
    fn acquire_resident_static_uploads(
        &self,
        _cache_domain: u64,
        payloads: &[&[u8]],
    ) -> Result<ResidentStaticBufferSet, DispatchError> {
        let mut byte_lens = Vec::new();
        byte_lens.try_reserve(payloads.len()).map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: reserve resident static byte lengths before upload; requested {} payload(s): {error}.",
                payloads.len()
            ))
        })?;
        for payload in payloads {
            byte_lens.push(payload.len());
        }
        let handles = self.alloc_resident_many(&byte_lens)?;

        let mut uploads = Vec::new();
        uploads.try_reserve(payloads.len()).map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: reserve resident static upload storage before upload; requested {} payload(s): {error}.",
                payloads.len()
            ))
        })?;
        for (&handle, &payload) in handles.iter().zip(payloads.iter()) {
            uploads.push((handle, payload));
        }

        if let Err(error) = self.upload_resident_many(&uploads) {
            let upload_error = error.to_string();
            if let Err(free_error) =
                free_resident_handles(self, &handles, "resident static upload rollback")
            {
                return Err(DispatchError::BackendError(format!(
                    "Fix: resident static upload failed after allocating {} buffer(s): {upload_error}; rollback also failed: {free_error}.",
                    handles.len()
                )));
            }
            return Err(error);
        }

        Ok(ResidentStaticBufferSet {
            handles,
            cache_hit: false,
            retained_by_dispatcher: false,
        })
    }

    /// Release a static resident buffer set acquired from
    /// [`Self::acquire_resident_static_uploads`].
    fn release_resident_static_uploads(
        &self,
        set: ResidentStaticBufferSet,
    ) -> Result<(), DispatchError> {
        if set.retained_by_dispatcher {
            return Ok(());
        }
        for handle in set.handles {
            self.free_resident(handle)?;
        }
        Ok(())
    }

    /// Download a resident buffer's current contents to host bytes.
    fn read_resident(&self, _handle: u64) -> Result<Vec<u8>, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: dispatcher does not implement read_resident.".to_string(),
        ))
    }

    /// Download several resident buffers with one backend fence when supported.
    fn read_resident_many(&self, handles: &[u64]) -> Result<Vec<Vec<u8>>, DispatchError> {
        handles
            .iter()
            .map(|&handle| self.read_resident(handle))
            .collect()
    }

    /// Download selected byte ranges from resident buffers.
    fn read_resident_ranges(
        &self,
        ranges: &[ResidentReadRange],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut outputs = Vec::new();
        self.read_resident_ranges_into(ranges, &mut outputs)?;
        Ok(outputs)
    }

    /// Download selected byte ranges from resident buffers into caller-owned
    /// byte slots.
    fn read_resident_ranges_into(
        &self,
        ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        let mut unique_handles = Vec::new();
        unique_handles.try_reserve(ranges.len()).map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: reserve resident ranged-read handle dedupe storage before dispatch; requested {} range(s): {error}.",
                ranges.len()
            ))
        })?;
        let mut range_handle_indices = Vec::new();
        range_handle_indices
            .try_reserve(ranges.len())
            .map_err(|error| {
                DispatchError::BackendError(format!(
                    "Fix: reserve resident ranged-read index storage before dispatch; requested {} range(s): {error}.",
                    ranges.len()
                ))
            })?;
        for range in ranges {
            if let Some(index) = unique_handles
                .iter()
                .position(|&handle| handle == range.handle_id)
            {
                range_handle_indices.push(index);
            } else {
                let index = unique_handles.len();
                unique_handles.push(range.handle_id);
                range_handle_indices.push(index);
            }
        }
        let full_buffers = self.read_resident_many(&unique_handles)?;
        if full_buffers.len() != unique_handles.len() {
            return Err(DispatchError::BackendError(format!(
                "Fix: resident ranged-read batch returned {} buffer(s) for {} unique handle(s).",
                full_buffers.len(),
                unique_handles.len()
            )));
        }
        if outputs.len() < ranges.len() {
            outputs
                .try_reserve(ranges.len() - outputs.len())
                .map_err(|error| {
                    DispatchError::BackendError(format!(
                        "Fix: reserve resident ranged-read output storage before dispatch; requested {} range(s): {error}.",
                        ranges.len()
                    ))
                })?;
            outputs.resize_with(ranges.len(), Vec::new);
        } else {
            outputs.truncate(ranges.len());
        }
        for ((range, &buffer_index), output) in ranges
            .iter()
            .zip(range_handle_indices.iter())
            .zip(outputs.iter_mut())
        {
            let full = full_buffers.get(buffer_index).ok_or_else(|| {
                DispatchError::BackendError(format!(
                    "Fix: resident ranged-read handle index {buffer_index} missing from {} readback buffer(s).",
                    full_buffers.len()
                ))
            })?;
            let end = range
                .byte_offset
                .checked_add(range.byte_len)
                .ok_or_else(|| {
                    DispatchError::BadInputs(format!(
                    "Fix: resident read range for handle {} overflows usize at offset {} len {}.",
                    range.handle_id, range.byte_offset, range.byte_len
                ))
                })?;
            if end > full.len() {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: resident read range for handle {} requested bytes [{}..{}) but buffer readback has {} bytes.",
                    range.handle_id,
                    range.byte_offset,
                    end,
                    full.len()
                )));
            }
            output.clear();
            output.extend_from_slice(&full[range.byte_offset..end]);
        }
        Ok(())
    }

    /// Free a resident buffer previously returned by `alloc_resident`.
    fn free_resident(&self, _handle: u64) -> Result<(), DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: dispatcher does not implement free_resident.".to_string(),
        ))
    }

    /// Dispatch a Program against resident-buffer handles. Each
    /// handle is referenced from the Program's declared buffer in the
    /// same canonical buffer order. RW buffers are not read back  -
    /// caller invokes `read_resident` once at end of pipeline.
    fn dispatch_resident(
        &self,
        _program: &Program,
        _handles: &[u64],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: dispatcher does not implement dispatch_resident.".to_string(),
        ))
    }

    /// Dispatch an ordered sequence of resident-buffer Programs.
    ///
    /// Default implementation preserves correctness by fencing each step
    /// through `dispatch_resident`. A dispatcher with an ordered queue overrides
    /// this to enqueue the whole dependent chain and synchronize once.
    fn dispatch_resident_sequence(
        &self,
        steps: &[ResidentDispatchStep<'_>],
    ) -> Result<(), DispatchError> {
        for step in steps {
            self.dispatch_resident(step.program, step.handle_ids, step.grid_override)?;
        }
        Ok(())
    }

    /// Dispatch an ordered resident sequence and read selected resident buffers.
    ///
    /// The value-returning convenience path delegates to the caller-owned
    /// output variant so a dispatcher with an ordered queue keeps kernels and
    /// readbacks behind one host fence.
    fn dispatch_resident_sequence_read_many(
        &self,
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut outputs = Vec::new();
        self.upload_resident_many_sequence_read_many_into(&[], steps, read_handles, &mut outputs)?;
        Ok(outputs)
    }

    /// Dispatch an ordered resident sequence and read selected byte ranges.
    fn dispatch_resident_sequence_read_ranges(
        &self,
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut outputs = Vec::new();
        self.upload_resident_many_sequence_read_ranges_into(&[], steps, read_ranges, &mut outputs)?;
        Ok(outputs)
    }

    /// Upload resident buffers, dispatch an ordered resident sequence, then
    /// read selected resident buffers.
    ///
    /// The value-returning convenience path delegates to the caller-owned
    /// output variant. Its portable default fences at each boundary; an ordered
    /// queue override keeps uploads, kernels, and readbacks behind one fence.
    fn upload_resident_many_sequence_read_many(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut outputs = Vec::new();
        self.upload_resident_many_sequence_read_many_into(
            uploads,
            steps,
            read_handles,
            &mut outputs,
        )?;
        Ok(outputs)
    }

    /// Upload resident buffers, dispatch an ordered resident sequence, then
    /// read selected byte ranges.
    fn upload_resident_many_sequence_read_ranges(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut outputs = Vec::new();
        self.upload_resident_many_sequence_read_ranges_into(
            uploads,
            steps,
            read_ranges,
            &mut outputs,
        )?;
        Ok(outputs)
    }

    /// Same contract as [`Self::upload_resident_many_sequence_read_many`],
    /// but writes readbacks into caller-owned byte slots.
    fn upload_resident_many_sequence_read_many_into(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        self.upload_resident_many(uploads)?;
        self.dispatch_resident_sequence(steps)?;
        let readbacks = self.read_resident_many(read_handles)?;
        if outputs.len() < readbacks.len() {
            outputs.resize_with(readbacks.len(), Vec::new);
        } else {
            outputs.truncate(readbacks.len());
        }
        for (slot, readback) in outputs.iter_mut().zip(readbacks) {
            slot.clear();
            slot.extend_from_slice(&readback);
        }
        Ok(())
    }

    /// Same contract as [`Self::upload_resident_many_sequence_read_many_into`],
    /// but first clears full resident buffers to zero.
    ///
    /// A dispatcher without device-side fill emulates clears as zero-byte
    /// payload uploads and still pays one upload/sequence/read boundary. One
    /// with device-side fill overrides this to enqueue the fills ahead of the
    /// explicit uploads and kernels, which keeps scratch initialization off the
    /// host bus without adding a fence.
    fn clear_upload_resident_many_sequence_read_many_into(
        &self,
        clears: &[(u64, usize)],
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        if clears.is_empty() {
            return self.upload_resident_many_sequence_read_many_into(
                uploads,
                steps,
                read_handles,
                outputs,
            );
        }
        let mut fills = Vec::new();
        fills.try_reserve(clears.len()).map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: reserve resident clear fill descriptors before dispatch; requested {} clear(s): {error}.",
                clears.len()
            ))
        })?;
        for &(handle, byte_len) in clears {
            fills.push((handle, byte_len, 0));
        }
        self.fill_upload_resident_many_sequence_read_many_into(
            &fills,
            uploads,
            steps,
            read_handles,
            outputs,
        )
    }

    /// Same contract as
    /// [`Self::clear_upload_resident_many_sequence_read_many_into`], but fills
    /// each resident buffer with an arbitrary byte value.
    fn fill_upload_resident_many_sequence_read_many_into(
        &self,
        fills: &[(u64, usize, u8)],
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        if fills.is_empty() {
            return self.upload_resident_many_sequence_read_many_into(
                uploads,
                steps,
                read_handles,
                outputs,
            );
        }

        with_staged_fill_uploads(
            fills,
            uploads,
            "resident fill payloads",
            "resident fill/upload payloads",
            |combined_uploads| {
                self.upload_resident_many_sequence_read_many_into(
                    combined_uploads,
                    steps,
                    read_handles,
                    outputs,
                )
            },
        )
    }

    /// Same contract as [`Self::upload_resident_many_sequence_read_ranges_into`],
    /// but fills resident buffers first. A dispatcher with device-side fill
    /// overrides this to use it plus compact readback range copies on one queue.
    fn fill_upload_resident_many_sequence_read_ranges_into(
        &self,
        fills: &[(u64, usize, u8)],
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        if fills.is_empty() {
            return self.upload_resident_many_sequence_read_ranges_into(
                uploads,
                steps,
                read_ranges,
                outputs,
            );
        }

        with_staged_fill_uploads(
            fills,
            uploads,
            "resident range-fill payloads",
            "resident range-fill/upload payloads",
            |combined_uploads| {
                self.upload_resident_many_sequence_read_ranges_into(
                    combined_uploads,
                    steps,
                    read_ranges,
                    outputs,
                )
            },
        )
    }

    /// Same contract as [`Self::upload_resident_many_sequence_read_ranges`],
    /// but writes compact readbacks into caller-owned byte slots.
    fn upload_resident_many_sequence_read_ranges_into(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        self.upload_resident_many(uploads)?;
        self.dispatch_resident_sequence(steps)?;
        self.read_resident_ranges_into(read_ranges, outputs)
    }
}

fn free_resident_handles<D: ProgramDispatcher + ?Sized>(
    dispatcher: &D,
    handles: &[u64],
    context: &str,
) -> Result<(), DispatchError> {
    for (index, &handle) in handles.iter().enumerate() {
        dispatcher.free_resident(handle).map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: {context} failed to free resident handle {handle} at index {index}: {error}."
            ))
        })?;
    }
    Ok(())
}

fn with_staged_fill_uploads<R>(
    fills: &[(u64, usize, u8)],
    uploads: &[(u64, &[u8])],
    fill_context: &'static str,
    combined_context: &'static str,
    run: impl FnOnce(&[(u64, &[u8])]) -> Result<R, DispatchError>,
) -> Result<R, DispatchError> {
    let mut fill_payloads = Vec::new();
    fill_payloads.try_reserve(fills.len()).map_err(|error| {
        DispatchError::BackendError(format!(
            "Fix: reserve {fill_context} before dispatch; requested {} fill(s): {error}.",
            fills.len()
        ))
    })?;
    for &(_, byte_len, value) in fills {
        fill_payloads.push(vec![value; byte_len]);
    }

    let mut combined_uploads = Vec::new();
    combined_uploads
        .try_reserve(fills.len() + uploads.len())
        .map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: reserve {combined_context} before dispatch; requested {} fill(s) and {} upload(s): {error}.",
                fills.len(),
                uploads.len()
            ))
        })?;
    for ((handle, _, _), fill) in fills.iter().zip(fill_payloads.iter()) {
        combined_uploads.push((*handle, fill.as_slice()));
    }
    combined_uploads.extend_from_slice(uploads);

    run(&combined_uploads)
}
#[cfg(test)]
mod tests;
