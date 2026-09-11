use vyre_driver::BackendError;

pub(crate) fn aligned_len(len: usize) -> Result<u64, BackendError> {
    let padded = aligned_len_usize(len, "GPU buffer length")?;
    u64::try_from(padded).map_err(|source| {
        BackendError::new(format!(
            "GPU buffer length {padded} cannot fit u64: {source}. Fix: split the dispatch input."
        ))
    })
}

pub(crate) fn aligned_len_u64(len: u64, label: &'static str) -> Result<u64, BackendError> {
    crate::numeric::WGPU_NUMERIC.align_up_u64(len, 4, 4, label)
}

fn aligned_len_usize(len: usize, label: &'static str) -> Result<usize, BackendError> {
    crate::numeric::WGPU_NUMERIC.align_up_usize(len, 4, 4, label)
}

pub(crate) fn write_padded(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    bytes: &[u8],
    allocation_len: u64,
) -> Result<(), BackendError> {
    crate::padded_upload::write_padded_and_zero_fill(queue, buffer, bytes, allocation_len)
}
