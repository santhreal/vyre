//! Bitset, provenance, matroid and fingerprint encoding compositions.

#[cfg(test)]
pub(crate) mod bitset_compression;
pub mod bitset_mask_algebra;
pub mod bitset_summary;
pub mod bitset_transform_pipeline;
pub mod matching_diagnostic_compaction;
pub mod matroid_exact_megakernel;
#[cfg(test)]
pub(crate) mod matroid_megakernel_scheduler;
#[cfg(any(
    feature = "nn-activation",
    feature = "nn-linear",
    feature = "nn-norm",
    feature = "nn-attention"
))]
pub mod nn_attention_paging;
pub mod parsing_dispatch_pipeline;
pub mod reduce_dispatch_pipeline;
pub mod scallop_provenance;
pub mod scallop_provenance_wide;
pub mod vsa_fingerprint;

pub(crate) fn decode_first_output(
    outputs: &[Vec<u8>],
    words: usize,
    context: &'static str,
    out: &mut Vec<u32>,
) -> Result<(), vyre_megakernel::SemanticExecutionError> {
    if outputs.is_empty() {
        return Err(vyre_megakernel::SemanticExecutionError::Backend(format!(
            "Fix: {context} expected at least one output buffer, got 0."
        )));
    }
    crate::dispatch_buffers::decode_u32_output_exact(&outputs[0], words, context, out)
}
