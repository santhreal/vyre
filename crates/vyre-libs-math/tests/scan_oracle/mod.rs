//! The scan oracle math tests compare against.
//!
//! The deterministic word generators this module carried are one contract with
//! one owner, `vyre_test_support::word_corpora`. Eleven test trees held a copy
//! and they had drifted apart, so a seed no longer named one sequence.

pub(crate) fn prefix_scan_cpu_ref(
    input: &[u32],
    kind: vyre_libs_math::math::prefix_scan::ScanKind,
) -> Vec<u32> {
    match kind {
        vyre_libs_math::math::prefix_scan::ScanKind::InclusiveSum => {
            vyre_reference::composition_witness::inclusive_prefix_sum_witness(input)
        }
        vyre_libs_math::math::prefix_scan::ScanKind::ExclusiveSum => {
            vyre_reference::composition_witness::exclusive_prefix_sum_witness(input)
        }
    }
}
