//! Data-structure, provenance, and substrate encoding modules.

pub mod bitset_compression;
pub mod bitset_mask_algebra;
pub mod bitset_summary;
pub mod bitset_transform_pipeline;
pub mod matching_diagnostic_compaction;
pub mod matroid_exact_megakernel;
pub mod matroid_megakernel_scheduler;
pub mod nn_attention_paging;
pub mod parsing_dispatch_pipeline;
pub mod reduce_dispatch_pipeline;
pub mod reduction_metrics;
pub mod scallop_provenance;
pub mod scallop_provenance_wide;
pub mod vsa_fingerprint;

pub(crate) fn decode_first_output(
    outputs: &[Vec<u8>],
    words: usize,
    context: &'static str,
    out: &mut Vec<u32>,
) -> Result<(), crate::optimizer::dispatcher::DispatchError> {
    if outputs.is_empty() {
        return Err(crate::optimizer::dispatcher::DispatchError::BackendError(
            format!("Fix: {context} expected at least one output buffer, got 0."),
        ));
    }
    crate::dispatch_buffers::decode_u32_output_exact(&outputs[0], words, context, out)
}
