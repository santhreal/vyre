use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use vyre_driver::BackendError;

use crate::buffer::staging::StagingBufferPool;

/// Readback copy and map request submitted without waiting for GPU completion.
pub(crate) enum PendingGpuBufferReadback {
    Ready,
    Mapping {
        readback: wgpu::Buffer,
        read_len: u64,
        readback_usage: wgpu::BufferUsages,
        pool: Option<StagingBufferPool>,
        submission: wgpu::SubmissionIndex,
        receiver: crossbeam_channel::Receiver<Result<(), wgpu::BufferAsyncError>>,
        ready: Arc<std::sync::atomic::AtomicBool>,
        trim_start: usize,
        visible_len: usize,
    },
}

impl PendingGpuBufferReadback {
    pub(crate) fn is_ready(&self, device: &wgpu::Device) -> bool {
        match self {
            Self::Ready => true,
            Self::Mapping { ready, .. } => {
                if crate::runtime::device::poll_device_once(device).is_err() {
                    return false;
                }
                ready.load(Ordering::Acquire)
            }
        }
    }

    pub(crate) fn await_into(
        self,
        device: &wgpu::Device,
        deadline: Option<Instant>,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        let Self::Mapping {
            readback,
            read_len,
            readback_usage,
            pool,
            submission,
            receiver,
            trim_start,
            visible_len,
            ..
        } = self
        else {
            out.clear();
            return Ok(());
        };
        let mapping = if let Some(deadline) = deadline {
            let mut backoff = crate::wait_backoff::AdaptiveWaitBackoff::from_micros(64, 2, 50, 5);
            loop {
                crate::runtime::device::poll_device_once(device)?;
                match receiver.try_recv() {
                    Ok(result) => break result,
                    Err(crossbeam_channel::TryRecvError::Empty) => {}
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        return Err(BackendError::new(
                            "persistent buffer readback channel closed before completion. Fix: keep the GPU device alive until readback completes.",
                        ));
                    }
                }
                let now = Instant::now();
                if now >= deadline {
                    return Err(BackendError::new(
                        "dispatch cancelled after DispatchConfig.timeout before readback completed. Fix: raise DispatchConfig.timeout or split the program into smaller chunks.",
                    ));
                }
                backoff.idle_for(deadline.saturating_duration_since(now));
            }
        } else {
            crate::runtime::device::poll_device_wait_for(device, submission)?;
            receiver
                .recv_timeout(std::time::Duration::from_secs(30))
                .map_err(|source| {
                    BackendError::new(format!(
                        "persistent buffer readback callback did not complete after submission wait: {source}. Fix: keep the GPU device alive and inspect driver callback progress."
                    ))
                })?
        };
        mapping.map_err(|source| {
            BackendError::new(format!(
                "persistent buffer readback mapping failed: {source:?}. Fix: use COPY_SRC handles and MAP_READ staging buffers."
            ))
        })?;
        let slice = readback.slice(0..read_len);
        let mapped = slice.get_mapped_range();
        let trim_end = trim_start.checked_add(visible_len).ok_or_else(|| {
            BackendError::new(format!(
                "persistent buffer range trim overflows usize at offset {trim_start} len {visible_len}. Fix: split the buffer before readback."
            ))
        })?;
        let visible = &mapped[trim_start..trim_end];
        if out.len() == visible_len {
            out.copy_from_slice(visible);
        } else {
            vyre_foundation::allocation::reserve_exact_cleared(out, visible_len).map_err(
                |source| {
                    BackendError::new(format!(
                        "persistent buffer readback could not reserve {visible_len} output bytes exactly: {source}. Fix: lower max_output_bytes or stream readback in smaller shards."
                    ))
                },
            )?;
            out.extend_from_slice(visible);
        }
        drop(mapped);
        readback.unmap();
        if let Some(pool) = pool {
            pool.release(readback, read_len, readback_usage);
        }
        Ok(())
    }
}
