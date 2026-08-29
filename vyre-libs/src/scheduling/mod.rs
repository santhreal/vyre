//! Scheduling, fusion, batching, and dispatch-strategy compositions.

#[cfg(test)]
pub mod branch_compaction;
#[cfg(test)]
pub mod frontier_partitioning;
pub mod frontier_typed_ir;
#[cfg(test)]
pub mod multi_corpus_batching;
pub mod planar_rewrite_pass_scheduler;
#[cfg(test)]
pub mod polyhedral_fusion;
pub mod spectral_schedule;
pub mod submodular_cache_eviction;

pub(crate) fn checked_product_count(
    left: u32,
    right: u32,
    left_name: &str,
    right_name: &str,
    context: &str,
) -> Result<usize, vyre_megakernel::SemanticExecutionError> {
    if left == 0 || right == 0 {
        return Err(vyre_megakernel::SemanticExecutionError::InvalidRequest(
            format!(
                "Fix: {context} requires {left_name} > 0 and {right_name} > 0, got {left_name}={left}, {right_name}={right}."
            ),
        ));
    }
    (left as usize)
        .checked_mul(right as usize)
        .ok_or_else(|| {
            vyre_megakernel::SemanticExecutionError::InvalidRequest(format!(
                "Fix: {context} {left_name}*{right_name} overflows usize for {left_name}={left}, {right_name}={right}."
            ))
        })
}

pub(crate) fn checked_square_cells(
    n: u32,
    context: &str,
) -> Result<usize, vyre_megakernel::SemanticExecutionError> {
    if n == 0 {
        return Err(vyre_megakernel::SemanticExecutionError::InvalidRequest(
            format!("Fix: {context} requires n > 0."),
        ));
    }
    (n as usize).checked_mul(n as usize).ok_or_else(|| {
        vyre_megakernel::SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} n*n overflows usize for n={n}."
        ))
    })
}

pub(crate) fn decode_u32_output_exact(
    bytes: &[u8],
    expected_words: usize,
    context: &str,
    out: &mut Vec<u32>,
) -> Result<(), vyre_megakernel::SemanticExecutionError> {
    let expected_bytes = expected_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            vyre_megakernel::SemanticExecutionError::Backend(format!(
                "Fix: {context} output byte count overflowed usize."
            ))
        })?;
    if bytes.len() != expected_bytes {
        return Err(vyre_megakernel::SemanticExecutionError::Backend(format!(
            "Fix: {context} expected {expected_bytes} output bytes, got {}.",
            bytes.len()
        )));
    }
    vyre_primitives::wire::unpack_u32_slice_into(bytes, expected_words, context, out)
        .map_err(vyre_megakernel::SemanticExecutionError::Backend)
}
