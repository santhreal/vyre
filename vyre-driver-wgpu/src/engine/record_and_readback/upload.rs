use crate::numeric::WGPU_NUMERIC;
use vyre_driver::BackendError;

pub(super) fn pool_backend_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::new(format!(
        "GPU buffer pool acquisition failed: {error}. Fix: restart the process if the pool lock was poisoned, or reduce concurrent dispatch pressure."
    ))
}

pub(super) fn write_padded_input(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    binding_name: &str,
    bytes: &[u8],
    size: usize,
) -> Result<Option<(u64, u64)>, BackendError> {
    // wgpu treats a write past the destination's end as a validation error and
    // raises it on the device error scope, which surfaces as a PROCESS ABORT out
    // of `Queue::write_buffer` rather than as a `Result` a caller can handle. A
    // library must not take the host down over a buffer declaration, so the
    // overrun is caught here and returned as a vyre error instead.
    if bytes.len() > size {
        return Err(BackendError::new(format!(
            "dispatch supplied {} input byte(s) for binding `{binding_name}`, whose device buffer is {size} byte(s), so the upload would overrun it. Fix: match the input length to the buffer's declared .with_count(n), or declare the buffer runtime-sized (no .with_count) and let the backend size it from these bytes.",
            bytes.len()
        )));
    }
    let zero_start = crate::padded_upload::write_padded_prefix(
        queue,
        buffer,
        bytes,
        "padded input tail offset",
    )?;

    if size > zero_start {
        Ok(Some((
            WGPU_NUMERIC.usize_to_u64(zero_start, "padded input zero-fill start")?,
            WGPU_NUMERIC.usize_to_u64(size - zero_start, "padded input zero-fill length")?,
        )))
    } else {
        Ok(None)
    }
}
