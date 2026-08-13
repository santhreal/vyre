use crate::api::case::BenchError;

pub(crate) use vyre_primitives::wire::pack_f32_slice as f32_bytes;

pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;
pub(crate) fn u32_input_bytes<const N: usize>(inputs: [&[u32]; N]) -> Vec<Vec<u8>> {
    inputs.into_iter().map(u32_bytes).collect()
}

pub(crate) fn decode_u64_words(bytes: &[u8], context: &str) -> Result<Vec<u64>, BenchError> {
    if bytes.len() % 8 != 0 {
        return Err(BenchError::CorrectnessViolation(format!(
            "{context} metric payload length {} is not divisible by 8",
            bytes.len()
        )));
    }
    Ok(vyre_primitives::wire::decode_u64_le_bytes_all(bytes))
}

/// Read one little-endian `u32` at a word index, bounds- and overflow-checked.
///
/// `context` names the buffer in the error, so a case that reads several
/// buffers still says which one was short.
pub(crate) fn read_word(bytes: &[u8], word_index: u32, context: &str) -> Result<u32, BenchError> {
    let (offset, end) = word_range(word_index).ok_or_else(|| {
        BenchError::CorrectnessViolation(format!("{context} word index {word_index} overflowed usize"))
    })?;
    let word = bytes.get(offset..end).ok_or_else(|| {
        BenchError::CorrectnessViolation(format!(
            "{context} word {word_index} is outside output buffer"
        ))
    })?;
    vyre_primitives::wire::read_u32_le_word(word, 0, context)
        .map_err(BenchError::CorrectnessViolation)
}

/// Write one little-endian `u32` at a word index, bounds- and overflow-checked.
pub(crate) fn write_word(
    bytes: &mut [u8],
    word_index: u32,
    value: u32,
    context: &str,
) -> Result<(), BenchError> {
    let (offset, end) = word_range(word_index).ok_or_else(|| {
        BenchError::ExecutionFailed(format!("{context} word index {word_index} overflowed usize"))
    })?;
    let slot = bytes.get_mut(offset..end).ok_or_else(|| {
        BenchError::ExecutionFailed(format!(
            "{context} word {word_index} is outside output buffer"
        ))
    })?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn word_range(word_index: u32) -> Option<(usize, usize)> {
    let offset = usize::try_from(word_index).ok()?.checked_mul(4)?;
    Some((offset, offset.checked_add(4)?))
}

/// Bytes per nanosecond, which is gigabytes per second. Zero elapsed time
/// reports zero rather than an infinity that would poison a report.
pub(crate) fn gb_per_second(bytes: u64, nanos: u64) -> f64 {
    if nanos == 0 {
        return 0.0;
    }
    bytes as f64 / nanos as f64
}

/// Units per second scaled by 1000, saturating instead of overflowing.
pub(crate) fn rate_per_second_x1000(units: u64, nanos: u64) -> u64 {
    if nanos == 0 {
        return 0;
    }
    ((u128::from(units) * 1_000_000_000_000u128) / u128::from(nanos)).min(u128::from(u64::MAX))
        as u64
}
