use vyre_driver::BackendError;
use vyre_foundation::ir::BufferDecl;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaOutputReadback {
    pub(crate) device_offset: usize,
    pub(crate) byte_len: usize,
}

pub(crate) fn cuda_output_readback(
    buffer: &BufferDecl,
    full_size: usize,
) -> Result<CudaOutputReadback, BackendError> {
    let full_size_u64 = full_size as u64;
    let range = buffer.output_byte_range().unwrap_or(0..full_size_u64);
    if range.start > range.end || range.end > full_size_u64 {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA output `{}` declares byte range {:?} outside its {full_size}-byte buffer.",
                buffer.name(),
                range
            ),
        });
    }
    let device_offset = usize::try_from(range.start).map_err(|_| BackendError::InvalidProgram {
        fix: format!("Fix: CUDA output `{}` start offset {} exceeds usize.", buffer.name(), range.start),
    })?;
    let byte_len = usize::try_from(range.end - range.start).map_err(|_| BackendError::InvalidProgram {
        fix: format!("Fix: CUDA output `{}` byte length exceeds usize.", buffer.name()),
    })?;
    Ok(CudaOutputReadback {
        device_offset,
        byte_len,
    })
}

pub(crate) fn cuda_output_readback_for_binding(
    buffers: &[BufferDecl],
    buffer_index: usize,
    binding_name: &str,
    full_size: usize,
    context: &'static str,
) -> Result<CudaOutputReadback, BackendError> {
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA {context} expected program buffer index {buffer_index} for binding `{binding_name}` but only {} buffer(s) were declared. Rebuild the binding plan before launch.",
                buffers.len()
            ),
        })?;
    cuda_output_readback(buffer, full_size)
}
