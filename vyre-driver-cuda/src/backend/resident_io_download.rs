use smallvec::SmallVec;
use vyre_driver::{BackendError, OutputBuffers};

use super::dispatch::CudaBackend;
use super::output_range::CudaOutputReadback;
use super::pinned_allocations::HostTransferAllocations;
use super::resident::{CudaResidentBuffer, ResidentViewCache};
use super::resident_io::{add_resident_copy_count, add_resident_copy_slots, ResidentStreamFailure};
use super::resident_readback_fusion::{
    fuse_resident_readback_copies, resident_readback_copy, validate_fused_resident_readbacks,
    FusedResidentReadbacks, ResidentReadbackCopy, ResidentReadbackView,
};
use super::staging_reserve::{
    clear_vec_slots, reserve_smallvec, reserve_vec, reserved_vec, resize_vec_slots,
};
use crate::numeric::CUDA_NUMERIC;

pub(crate) fn clear_resident_copy_outputs(
    copies: &[ResidentReadbackCopy],
    outputs: &mut OutputBuffers,
) -> Result<(), BackendError> {
    resize_vec_slots(outputs, copies.len(), "readback output")?;
    clear_vec_slots(outputs);
    Ok(())
}

pub(crate) fn reserve_borrowed_resident_readback_outputs(
    views: &[ResidentReadbackView],
    outputs: &mut [&mut Vec<u8>],
) -> Result<(), BackendError> {
    if views.len() != outputs.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident readback materialization expected {} caller output slot(s) but received {}. Rebuild the resident readback fusion plan before copying bytes.",
                views.len(),
                outputs.len()
            ),
        });
    }
    for (view, output) in views.iter().zip(outputs.iter_mut()) {
        reserve_vec(
            *output,
            view.byte_len,
            "borrowed resident readback output bytes",
        )?;
    }
    Ok(())
}

pub(crate) fn reserve_resident_readback_outputs(
    views: &[ResidentReadbackView],
    outputs: &mut OutputBuffers,
) -> Result<(), BackendError> {
    let existing_slots_to_copy = outputs.len().min(views.len());
    if outputs.len() < views.len() {
        reserve_vec(outputs, views.len(), "resident readback output slots")?;
    }
    for (view, output) in views
        .iter()
        .take(existing_slots_to_copy)
        .zip(outputs.iter_mut())
    {
        reserve_vec(output, view.byte_len, "resident readback output bytes")?;
    }
    let mut appended_outputs = reserved_vec(
        views.len() - existing_slots_to_copy,
        "resident readback appended output slots",
    )?;
    for view in views.iter().skip(existing_slots_to_copy) {
        appended_outputs.push(reserved_vec(
            view.byte_len,
            "resident readback appended output bytes",
        )?);
    }
    outputs.truncate(views.len());
    outputs.extend(appended_outputs);
    Ok(())
}

pub(crate) fn next_resident_readback_view(
    views: &[ResidentReadbackView],
    view_index: &mut usize,
    total_copy_slots: usize,
) -> Result<ResidentReadbackView, BackendError> {
    let view = views
        .get(*view_index)
        .copied()
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident fused batched readback ran out of output views while reserving {total_copy_slots} copy slot(s). Rebuild the resident readback fusion plan before collecting outputs.",
            ),
        })?;
    *view_index += 1;
    Ok(view)
}

pub(crate) fn reserve_resident_readback_batch_outputs(
    copy_batches: &[SmallVec<[ResidentReadbackCopy; 8]>],
    views: &[ResidentReadbackView],
    outputs: &mut Vec<OutputBuffers>,
) -> Result<(), BackendError> {
    let existing_batches_to_copy = outputs.len().min(copy_batches.len());
    if outputs.len() < copy_batches.len() {
        reserve_vec(outputs, copy_batches.len(), "batched readback output slots")?;
    }

    let mut appended_items_by_batch = reserved_vec(
        existing_batches_to_copy,
        "batched readback appended item groups",
    )?;
    let mut appended_batches = reserved_vec(
        copy_batches.len() - existing_batches_to_copy,
        "batched readback appended output batches",
    )?;
    let total_copy_slots = views.len();
    let mut view_index = 0usize;

    for (copies, batch_outputs) in copy_batches
        .iter()
        .take(existing_batches_to_copy)
        .zip(outputs.iter_mut())
    {
        let existing_items_to_copy = batch_outputs.len().min(copies.len());
        if batch_outputs.len() < copies.len() {
            reserve_vec(batch_outputs, copies.len(), "batched readback item slots")?;
        }
        for output in batch_outputs.iter_mut().take(existing_items_to_copy) {
            let view = next_resident_readback_view(views, &mut view_index, total_copy_slots)?;
            reserve_vec(output, view.byte_len, "batched readback item bytes")?;
        }
        let mut appended_items = reserved_vec(
            copies.len() - existing_items_to_copy,
            "batched readback appended item slots",
        )?;
        for _ in existing_items_to_copy..copies.len() {
            let view = next_resident_readback_view(views, &mut view_index, total_copy_slots)?;
            appended_items.push(reserved_vec(
                view.byte_len,
                "batched readback appended item bytes",
            )?);
        }
        appended_items_by_batch.push(appended_items);
    }

    for copies in copy_batches.iter().skip(existing_batches_to_copy) {
        let mut batch_outputs =
            reserved_vec(copies.len(), "batched readback appended batch item slots")?;
        for _ in copies {
            let view = next_resident_readback_view(views, &mut view_index, total_copy_slots)?;
            batch_outputs.push(reserved_vec(
                view.byte_len,
                "batched readback appended batch item bytes",
            )?);
        }
        appended_batches.push(batch_outputs);
    }

    if view_index != views.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident fused batched readback produced {} output view(s) for {view_index} consumed slot(s). Keep resident readback fusion cardinality-preserving before collecting outputs.",
                views.len()
            ),
        });
    }

    outputs.truncate(copy_batches.len());
    for ((copies, batch_outputs), appended_items) in copy_batches
        .iter()
        .take(existing_batches_to_copy)
        .zip(outputs.iter_mut())
        .zip(appended_items_by_batch)
    {
        batch_outputs.truncate(copies.len());
        batch_outputs.extend(appended_items);
    }
    outputs.extend(appended_batches);
    Ok(())
}

impl CudaBackend {
    /// Download bytes from an existing CUDA-resident buffer.
    pub fn download_resident(&self, handle: CudaResidentBuffer) -> Result<Vec<u8>, BackendError> {
        let byte_len = self.resident_store.view(handle)?.byte_len;
        let mut bytes = reserved_vec(byte_len, "resident download output bytes")?;
        self.download_resident_into(handle, &mut bytes)?;
        Ok(bytes)
    }

    /// Download several full CUDA-resident buffers with one stream fence.
    pub fn download_resident_many(
        &self,
        handles: &[CudaResidentBuffer],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut outputs = reserved_vec(handles.len(), "resident output")?;
        self.download_resident_many_into(handles, &mut outputs)?;
        Ok(outputs)
    }

    /// Download several full CUDA-resident buffers into caller-owned output
    /// slots with one stream fence.
    pub fn download_resident_many_into(
        &self,
        handles: &[CudaResidentBuffer],
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        let mut copies = SmallVec::<[ResidentReadbackCopy; 8]>::new();
        reserve_smallvec(&mut copies, handles.len(), "full readback copy")?;
        let mut expected_copy_count = 0usize;
        let mut resident_view_cache = ResidentViewCache::new();
        reserve_smallvec(
            &mut resident_view_cache,
            handles.len(),
            "resident full-readback view cache",
        )?;
        for &handle in handles {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident full-readback view cache",
            )?;
            copies.push(resident_readback_copy(
                "resident full readback",
                handle.handle,
                buffer,
                0,
                buffer.byte_len,
            )?);
            if buffer.byte_len != 0 {
                add_resident_copy_count(&mut expected_copy_count, "full readback")?;
            }
        }
        if expected_copy_count == 0 {
            return clear_resident_copy_outputs(&copies, outputs);
        }
        self.download_resident_fused_copies_many_into(&copies, outputs)
    }

    /// Download bytes from an existing CUDA-resident buffer into caller-owned
    /// storage.
    pub fn download_resident_into(
        &self,
        handle: CudaResidentBuffer,
        bytes: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        let byte_len = self.resident_store.view(handle)?.byte_len;
        self.download_resident_range_into(handle, 0, byte_len, bytes)
    }

    /// Download a byte range from an existing CUDA-resident buffer.
    pub fn download_resident_range(
        &self,
        handle: CudaResidentBuffer,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let mut bytes = reserved_vec(byte_len, "resident ranged download output bytes")?;
        self.download_resident_range_into(handle, byte_offset, byte_len, &mut bytes)?;
        Ok(bytes)
    }

    /// Download a byte range from an existing CUDA-resident buffer into
    /// caller-owned storage.
    pub fn download_resident_range_into(
        &self,
        handle: CudaResidentBuffer,
        byte_offset: usize,
        byte_len: usize,
        bytes: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        self.download_resident_ranges_into(&[(handle, byte_offset, byte_len)], &mut [bytes])
    }

    /// Download selected byte ranges from resident buffers into caller-owned
    /// output slots with one stream fence.
    pub fn download_resident_ranges_into(
        &self,
        ranges: &[(CudaResidentBuffer, usize, usize)],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        if ranges.len() != outputs.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident ranged batch download expected matching range/output counts but got {} range(s) and {} output(s).",
                    ranges.len(),
                    outputs.len()
                ),
            });
        }
        let mut copies = SmallVec::<[ResidentReadbackCopy; 8]>::new();
        reserve_smallvec(&mut copies, ranges.len(), "ranged readback copy")?;
        let mut expected_copy_count = 0usize;
        let mut resident_view_cache = ResidentViewCache::new();
        reserve_smallvec(
            &mut resident_view_cache,
            ranges.len(),
            "resident ranged-readback view cache",
        )?;
        for &(handle, byte_offset, byte_len) in ranges {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident ranged-readback view cache",
            )?;
            copies.push(resident_readback_copy(
                "resident ranged batch download",
                handle.handle,
                buffer,
                byte_offset,
                byte_len,
            )?);
            if byte_len != 0 {
                add_resident_copy_count(&mut expected_copy_count, "ranged readback")?;
            }
        }
        let fused_readbacks = fuse_resident_readback_copies(&copies)?;
        validate_fused_resident_readbacks(
            &fused_readbacks,
            copies.len(),
            "resident ranged batch download",
        )?;
        reserve_borrowed_resident_readback_outputs(&fused_readbacks.views, outputs)?;
        if expected_copy_count == 0 {
            for output in outputs.iter_mut() {
                output.clear();
            }
            return Ok(());
        }
        let (host_transfers, copy_count) =
            self.stage_fused_resident_readbacks_to_host(&fused_readbacks)?;
        for (view, output) in fused_readbacks.views.iter().zip(outputs.iter_mut()) {
            host_transfers.collect_output_range_into(
                view.copy_slot,
                view.byte_offset,
                view.byte_len,
                output,
            )?;
        }
        self.record_resident_readback_telemetry(
            &fused_readbacks,
            copy_count,
            "resident readback operation count",
        )?;
        Ok(())
    }

    /// Download selected byte ranges from several CUDA-resident buffers with one stream fence.
    pub(crate) fn download_resident_readbacks_many(
        &self,
        handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut outputs = reserved_vec(handles.len(), "resident readback output")?;
        self.download_resident_readbacks_many_into(handles, readbacks, &mut outputs)?;
        Ok(outputs)
    }

    /// Download selected byte ranges from several CUDA-resident buffers into
    /// caller-owned output slots with one stream fence.
    pub(crate) fn download_resident_readbacks_many_into(
        &self,
        handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        if handles.len() != readbacks.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident readback expected matching handle/range counts but got {} handle(s) and {} range(s).",
                    handles.len(),
                    readbacks.len()
                ),
            });
        }
        let mut copies = SmallVec::<[ResidentReadbackCopy; 8]>::new();
        reserve_smallvec(&mut copies, handles.len(), "readback copy")?;
        let mut resident_view_cache = ResidentViewCache::new();
        reserve_smallvec(
            &mut resident_view_cache,
            handles.len(),
            "resident readback view cache",
        )?;
        for (&handle, readback) in handles.iter().zip(readbacks.iter()) {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident readback view cache",
            )?;
            copies.push(resident_readback_copy(
                "resident readback",
                handle.handle,
                buffer,
                readback.device_offset,
                readback.byte_len,
            )?);
        }
        self.download_resident_fused_copies_many_into(&copies, outputs)
    }

    pub(crate) fn stage_fused_resident_readbacks_to_host(
        &self,
        fused_readbacks: &FusedResidentReadbacks,
    ) -> Result<(HostTransferAllocations, usize), BackendError> {
        self.warmup()?;
        let mut host_transfers = HostTransferAllocations::with_capacity(
            std::sync::Arc::clone(&self.host_pool),
            fused_readbacks.non_empty_copy_count,
            fused_readbacks.copies.len(),
        )?;
        let copy_count = match self.with_resident_stream_classified(|stream| {
            let mut copy_count = 0usize;
            for copy in &fused_readbacks.copies {
                let dst = host_transfers.push_output(copy.byte_len)?;
                if copy.byte_len != 0 {
                    // SAFETY: FFI to libcuda.so. Source pointer/range was
                    // validated against the resident allocation before staging;
                    // the pinned host destination remains owned until the stream
                    // fence completes.
                    unsafe {
                        super::copy::d2h_async_checked(dst, copy.src, copy.byte_len, stream.raw())?;
                    }
                    copy_count += 1;
                }
            }
            if copy_count != 0 {
                stream.synchronize()?;
                self.telemetry.record_sync_point();
            }
            Ok::<usize, BackendError>(copy_count)
        }) {
            Ok(copy_count) => copy_count,
            Err(ResidentStreamFailure::Completed(error)) => return Err(error),
            Err(ResidentStreamFailure::CompletionUnproven(error)) => {
                std::mem::forget(host_transfers);
                return Err(error);
            }
        };
        Ok((host_transfers, copy_count))
    }

    pub(crate) fn record_resident_readback_telemetry(
        &self,
        fused_readbacks: &FusedResidentReadbacks,
        copy_count: usize,
        operation_count_label: &str,
    ) -> Result<(), BackendError> {
        self.telemetry
            .record_device_to_host_readback(fused_readbacks.bytes);
        self.telemetry.record_device_readback_operations(
            CUDA_NUMERIC.usize_to_u64(copy_count, operation_count_label)?,
        );
        Ok(())
    }

    fn download_resident_fused_copies_many_into(
        &self,
        copies: &[ResidentReadbackCopy],
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        let fused_readbacks = fuse_resident_readback_copies(copies)?;
        validate_fused_resident_readbacks(
            &fused_readbacks,
            copies.len(),
            "resident fused readback",
        )?;
        reserve_resident_readback_outputs(&fused_readbacks.views, outputs)?;
        if fused_readbacks.non_empty_copy_count == 0 {
            clear_vec_slots(outputs);
            return Ok(());
        }
        let (host_transfers, copy_count) =
            self.stage_fused_resident_readbacks_to_host(&fused_readbacks)?;
        for (view, output) in fused_readbacks.views.iter().zip(outputs.iter_mut()) {
            host_transfers.collect_output_range_into(
                view.copy_slot,
                view.byte_offset,
                view.byte_len,
                output,
            )?;
        }
        self.record_resident_readback_telemetry(
            &fused_readbacks,
            copy_count,
            "resident fused readback operation count",
        )?;
        Ok(())
    }

    fn download_resident_fused_copy_batches_many_into(
        &self,
        copy_batches: &[SmallVec<[ResidentReadbackCopy; 8]>],
        total_copy_slots: usize,
        outputs: &mut Vec<OutputBuffers>,
    ) -> Result<(), BackendError> {
        let mut flat_copies = SmallVec::<[ResidentReadbackCopy; 8]>::new();
        reserve_smallvec(
            &mut flat_copies,
            total_copy_slots,
            "flat fused batch readback copy",
        )?;
        for copies in copy_batches {
            flat_copies.extend(copies.iter().copied());
        }

        let fused_readbacks = fuse_resident_readback_copies(&flat_copies)?;
        validate_fused_resident_readbacks(
            &fused_readbacks,
            total_copy_slots,
            "resident fused batched readback",
        )?;
        reserve_resident_readback_batch_outputs(copy_batches, &fused_readbacks.views, outputs)?;
        if fused_readbacks.non_empty_copy_count == 0 {
            for batch_outputs in outputs.iter_mut() {
                clear_vec_slots(batch_outputs);
            }
            return Ok(());
        }

        let (host_transfers, copy_count) =
            self.stage_fused_resident_readbacks_to_host(&fused_readbacks)?;

        let mut fused_views = fused_readbacks.views.iter().copied();
        for (copies, batch_outputs) in copy_batches.iter().zip(outputs.iter_mut()) {
            for output in batch_outputs {
                let view = fused_views.next().ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident fused batched readback ran out of output views while materializing {} batch(es). Rebuild the resident readback fusion plan before collecting outputs.",
                        copy_batches.len()
                    ),
                })?;
                host_transfers.collect_output_range_into(
                    view.copy_slot,
                    view.byte_offset,
                    view.byte_len,
                    output,
                )?;
            }
        }
        if fused_views.next().is_some() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident fused batched readback produced more output views than {} requested copy slot(s). Keep resident readback fusion cardinality-preserving before collecting outputs.",
                    total_copy_slots
                ),
            });
        }
        self.record_resident_readback_telemetry(
            &fused_readbacks,
            copy_count,
            "resident fused batched readback operation count",
        )?;
        Ok(())
    }

    /// Download selected byte ranges from several resident-output batches into
    /// caller-owned output storage with one stream fence.
    pub(crate) fn download_resident_readback_batches_many_into(
        &self,
        handle_batches: &[SmallVec<[CudaResidentBuffer; 8]>],
        readback_batches: &[SmallVec<[CudaOutputReadback; 8]>],
        outputs: &mut Vec<OutputBuffers>,
    ) -> Result<(), BackendError> {
        if handle_batches.len() != readback_batches.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident batch readback expected matching batch counts but got {} handle batch(es) and {} range batch(es).",
                    handle_batches.len(),
                    readback_batches.len()
                ),
            });
        }
        let mut copy_batches = SmallVec::<[SmallVec<[ResidentReadbackCopy; 8]>; 8]>::new();
        reserve_smallvec(&mut copy_batches, handle_batches.len(), "readback batch")?;
        let mut total_copy_slots = 0usize;
        for (batch_index, (handles, readbacks)) in handle_batches
            .iter()
            .zip(readback_batches.iter())
            .enumerate()
        {
            if handles.len() != readbacks.len() {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident batch readback item {batch_index} expected matching handle/range counts but got {} handle(s) and {} range(s).",
                        handles.len(),
                        readbacks.len()
                    ),
                });
            }
            let mut copies = SmallVec::<[ResidentReadbackCopy; 8]>::new();
            reserve_smallvec(&mut copies, handles.len(), "batched readback copy")?;
            add_resident_copy_slots(&mut total_copy_slots, handles.len(), "batch readback")?;
            let mut resident_view_cache = ResidentViewCache::new();
            reserve_smallvec(
                &mut resident_view_cache,
                handles.len(),
                "resident batched-readback view cache",
            )?;
            for (&handle, readback) in handles.iter().zip(readbacks.iter()) {
                let buffer = self.resident_store.view_cached(
                    handle,
                    &mut resident_view_cache,
                    "resident batched-readback view cache",
                )?;
                copies.push(resident_readback_copy(
                    "resident batch readback",
                    handle.handle,
                    buffer,
                    readback.device_offset,
                    readback.byte_len,
                )?);
            }
            copy_batches.push(copies);
        }
        self.download_resident_fused_copy_batches_many_into(
            &copy_batches,
            total_copy_slots,
            outputs,
        )
    }
}
