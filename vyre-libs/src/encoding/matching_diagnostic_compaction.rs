//! Matching diagnostic compaction via `vyre-primitives::matching`.
//!
//! Self-substrate diagnostics and pass traces produce raw spans, brace-pair
//! links, and pattern-id regions. This module keeps that pipeline resident:
//! compile the DFA once, match brackets on-device, sort region triples, then
//! emit dedup survivor flags for stream compaction.

use crate::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
};
use crate::pattern::{
    bracket_match, dedup_regions_flag_program, region_sort_program, RegionTriple,
    BRACKET_KIND_CLOSE, BRACKET_KIND_OPEN, BRACKET_KIND_OTHER,
};
use crate::plumbing::host::scratch::reserve_vec_capacity;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Caller-owned dispatch scratch for matching diagnostic compaction.
#[derive(Debug, Default)]
pub struct MatchingDiagnosticCompactionGpuScratch {
    inputs: Vec<Vec<u8>>,
    pids: Vec<u32>,
    starts: Vec<u32>,
    ends: Vec<u32>,
    decoded_pids: Vec<u32>,
    decoded_starts: Vec<u32>,
    decoded_ends: Vec<u32>,
    decoded_regions: Vec<RegionTriple>,
}

/// Match diagnostic brace tokens through the bracket-match primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when `kinds.len()` or `max_depth` exceeds
/// the primitive index space, semantic execution fails, or readback is malformed.
pub fn bracket_pairs_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    kinds: &[u32],
    max_depth: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut scratch = MatchingDiagnosticCompactionGpuScratch::default();
    let mut out = Vec::new();
    bracket_pairs_via_with_scratch_into(
        dispatcher,
        policy,
        kinds,
        max_depth,
        &mut scratch,
        &mut out,
    )?;
    Ok(out)
}

/// Match diagnostic brace tokens through the bracket-match primitive using
/// caller-owned scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn bracket_pairs_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    kinds: &[u32],
    max_depth: u32,
    scratch: &mut MatchingDiagnosticCompactionGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, matching_diagnostic_compaction_calls};
    bump(&matching_diagnostic_compaction_calls);

    let n = checked_len(kinds.len(), "bracket_pairs_via")?;
    let max_depth_usize = usize::try_from(max_depth).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: bracket_pairs_via max_depth={max_depth} does not fit usize scratch sizing."
        ))
    })?;
    let program = bracket_match("kinds", "stack", "match_pairs", n, max_depth);
    // Input-consuming buffers ONLY: `kinds` ReadOnly(0) + `stack` plain-ReadWrite(1). `match_pairs`
    // is `BufferDecl::output`(2), backend-allocated, so it consumes NO dispatch input (the kernel
    // initializes every entry to BRACKET_MATCH_NONE itself). Passing a seed slot for it would over-feed the
    // real backend's strict input count (`inputs.len() == input_indices.len()`).
    ensure_input_slots(&mut scratch.inputs, 2);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], kinds);
    write_zero_bytes(
        &mut scratch.inputs[1],
        max_depth_usize * std::mem::size_of::<u32>(),
    );
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs[..2],
        policy,
    )
    .map(|output| output.outputs)?;
    // Writable buffers are returned in binding order `[stack(1), match_pairs(2)]`, so the match-pairs
    // result is outputs[1]. NOT outputs[0], which is the `stack` scratch.
    decode_output_at(&outputs, 1, kinds.len(), "bracket_pairs_via", out)
}

/// Sort diagnostic region triples by `(pid, start, end)` through the primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the region count is zero or too large,
/// semantic execution fails, or readback is malformed.
pub fn sort_regions_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    regions: &[RegionTriple],
) -> Result<Vec<RegionTriple>, SemanticExecutionError> {
    let mut scratch = MatchingDiagnosticCompactionGpuScratch::default();
    let mut out = Vec::new();
    sort_regions_via_with_scratch_into(dispatcher, policy, regions, &mut scratch, &mut out)?;
    Ok(out)
}

/// Sort diagnostic region triples through the primitive using caller-owned
/// staging and output storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the region count is zero or too large,
/// semantic execution fails, or readback is malformed.
pub fn sort_regions_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    regions: &[RegionTriple],
    scratch: &mut MatchingDiagnosticCompactionGpuScratch,
    out: &mut Vec<RegionTriple>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, matching_diagnostic_compaction_calls};
    bump(&matching_diagnostic_compaction_calls);

    let count = checked_nonzero_len(regions.len(), "sort_regions_via")?;
    split_regions_into(
        regions,
        &mut scratch.pids,
        &mut scratch.starts,
        &mut scratch.ends,
    )?;
    let program = region_sort_program(
        "pids",
        "starts",
        "ends",
        "pids_out",
        "starts_out",
        "ends_out",
        count,
    );
    ensure_input_slots(&mut scratch.inputs, 6);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], &scratch.pids);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], &scratch.starts);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], &scratch.ends);
    for slot in 3..=5 {
        write_zero_bytes(
            &mut scratch.inputs[slot],
            regions.len() * std::mem::size_of::<u32>(),
        );
    }
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    decode_region_outputs_into(&outputs, regions.len(), "sort_regions_via", scratch, out)
}

/// Emit dedup survivor flags for sorted region triples through the primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the region count is too large, semantic
/// execution fails, or readback is malformed.
pub fn dedup_region_survivor_flags_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    sorted_regions: &[RegionTriple],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut scratch = MatchingDiagnosticCompactionGpuScratch::default();
    let mut out = Vec::new();
    dedup_region_survivor_flags_via_with_scratch_into(
        dispatcher,
        policy,
        sorted_regions,
        &mut scratch,
        &mut out,
    )?;
    Ok(out)
}

/// Emit dedup survivor flags through the primitive using caller-owned staging.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the region count is too large, semantic
/// execution fails, or readback is malformed.
pub fn dedup_region_survivor_flags_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    sorted_regions: &[RegionTriple],
    scratch: &mut MatchingDiagnosticCompactionGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, matching_diagnostic_compaction_calls};
    bump(&matching_diagnostic_compaction_calls);

    if sorted_regions.is_empty() {
        out.clear();
        return Ok(());
    }
    let count = checked_len(sorted_regions.len(), "dedup_region_survivor_flags_via")?;
    split_regions_into(
        sorted_regions,
        &mut scratch.pids,
        &mut scratch.starts,
        &mut scratch.ends,
    )?;
    let program = dedup_regions_flag_program("pids", "starts", "ends", "survivors", count);
    // Input-consuming buffers ONLY: pids/starts/ends ReadOnly(0-2). `survivors` is
    // `BufferAccess::WriteOnly`(3), backend-allocated, consumes NO dispatch input; passing a zero
    // slot for it would over-feed the real backend's strict input count.
    ensure_input_slots(&mut scratch.inputs, 3);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], &scratch.pids);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], &scratch.starts);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], &scratch.ends);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs[..3],
        policy,
    )
    .map(|output| output.outputs)?;
    decode_first_output(
        &outputs,
        sorted_regions.len(),
        "dedup_region_survivor_flags_via",
        out,
    )
}

/// Build a compact fixture token stream for one nested diagnostic block.
#[must_use]
pub fn nested_diagnostic_brace_fixture() -> Vec<u32> {
    vec![
        BRACKET_KIND_OPEN,
        BRACKET_KIND_OTHER,
        BRACKET_KIND_OPEN,
        BRACKET_KIND_CLOSE,
        BRACKET_KIND_CLOSE,
    ]
}

#[cfg(test)]
fn split_regions(regions: &[RegionTriple]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut pids = Vec::with_capacity(regions.len());
    let mut starts = Vec::with_capacity(regions.len());
    let mut ends = Vec::with_capacity(regions.len());
    split_regions_into(regions, &mut pids, &mut starts, &mut ends)
        .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - test fixture region split should reserve output columns");
    (pids, starts, ends)
}

fn split_regions_into(
    regions: &[RegionTriple],
    pids: &mut Vec<u32>,
    starts: &mut Vec<u32>,
    ends: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    pids.clear();
    starts.clear();
    ends.clear();
    reserve_vec_capacity(pids, regions.len(), "diagnostic region pids")?;
    reserve_vec_capacity(starts, regions.len(), "diagnostic region starts")?;
    reserve_vec_capacity(ends, regions.len(), "diagnostic region ends")?;
    for region in regions {
        pids.push(region.pid);
        starts.push(region.start);
        ends.push(region.end);
    }
    Ok(())
}

fn checked_len(len: usize, context: &'static str) -> Result<u32, SemanticExecutionError> {
    u32::try_from(len).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} received {len} items, which exceeds the u32 GPU index space."
        ))
    })
}

fn checked_nonzero_len(len: usize, context: &'static str) -> Result<u32, SemanticExecutionError> {
    let count = checked_len(len, context)?;
    if count == 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires at least one region."
        )));
    }
    Ok(count)
}

fn decode_region_outputs_into(
    outputs: &[Vec<u8>],
    count: usize,
    context: &'static str,
    scratch: &mut MatchingDiagnosticCompactionGpuScratch,
    out: &mut Vec<RegionTriple>,
) -> Result<(), SemanticExecutionError> {
    if outputs.len() < 3 {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {context} expected three output buffers, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(&outputs[0], count, context, &mut scratch.decoded_pids)?;
    decode_u32_output_exact(&outputs[1], count, context, &mut scratch.decoded_starts)?;
    decode_u32_output_exact(&outputs[2], count, context, &mut scratch.decoded_ends)?;
    scratch.decoded_regions.clear();
    reserve_vec_capacity(&mut scratch.decoded_regions, count, context)?;
    for index in 0..count {
        scratch.decoded_regions.push(RegionTriple::new(
            scratch.decoded_pids[index],
            scratch.decoded_starts[index],
            scratch.decoded_ends[index],
        ));
    }
    out.clear();
    out.extend_from_slice(&scratch.decoded_regions);
    Ok(())
}

fn decode_first_output(
    outputs: &[Vec<u8>],
    words: usize,
    context: &'static str,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    decode_output_at(outputs, 0, words, context, out)
}

/// Decode the writable buffer at `index` in the backend's canonical (binding-order) output list.
///
/// The real backend returns EVERY writable buffer (plain-ReadWrite/InputOutput + WriteOnly + output)
/// in binding order, so a consumer whose intended result is NOT the first writable buffer must decode
/// the correct index (e.g. `bracket_match` binds `stack` ReadWrite(1) before `match_pairs` output(2),
/// so the pairs are `outputs[1]`).
fn decode_output_at(
    outputs: &[Vec<u8>],
    index: usize,
    words: usize,
    context: &'static str,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let buffer = outputs.get(index).ok_or_else(|| {
        SemanticExecutionError::Backend(format!(
            "Fix: {context} expected at least {} output buffer(s), got {}.",
            index + 1,
            outputs.len()
        ))
    })?;
    decode_u32_output_exact(buffer, words, context, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use crate::pattern::{
        dfa_compile as compile_diagnostic_dfa,
        dfa_compile_with_budget as compile_diagnostic_dfa_with_budget,
    };

    use vyre_reference::composition_witness::bracket_match_witness as reference_bracket_pairs;

    fn reference_sort_regions(mut regions: Vec<RegionTriple>) -> Vec<RegionTriple> {
        regions.sort();
        regions
    }

    fn sort_regions_witness_in_place(regions: &mut [RegionTriple]) {
        regions.sort();
    }

    fn reference_dedup_regions(regions: Vec<RegionTriple>) -> Vec<RegionTriple> {
        let input: Vec<(u32, u32, u32)> = regions.iter().map(|r| (r.pid, r.start, r.end)).collect();
        let deduped = vyre_reference::composition_witness::dedup_regions_witness(input);
        deduped
            .into_iter()
            .map(|(pid, start, end)| RegionTriple::new(pid, start, end))
            .collect()
    }

    fn reference_dedup_regions_inplace(regions: &mut Vec<RegionTriple>) {
        let deduped = reference_dedup_regions(std::mem::take(regions));
        *regions = deduped;
    }

    struct MatchingDispatcher;

    impl SemanticExecutor for MatchingDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let program = &request.logical().graph().nodes()[0].program;
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                let op_id = program
                    .entry
                    .iter()
                    .find_map(|node| match node {
                        vyre_foundation::ir::Node::Region { generator, .. } => {
                            Some(generator.as_str())
                        }
                        _ => None,
                    })
                    .expect("Fix: matching primitive should expose a region generator");
                match op_id {
                    crate::pattern::BRACKET_MATCH_OP_ID => {
                        // Two input-consuming buffers: kinds ReadOnly(0), stack plain-ReadWrite(1).
                        // `match_pairs` output(2) is backend-allocated (no input slot).
                        assert_eq!(
                inputs.len(),
                2,
                "Fix: bracket_pairs_via must pass exactly the two input-consuming buffers (kinds, stack); match_pairs is backend-allocated."
            );
                        let kinds = crate::dispatch_buffers::read_u32s(&inputs[0]);
                        let depth_words = inputs[1].len() / std::mem::size_of::<u32>();

                        // Model the real backend: return ALL writable buffers in binding order
                        // [stack(1, InputOutput), match_pairs(2, output)]. The consumer must decode the
                        // pairs from outputs[1], not outputs[0] (the stack scratch).
                        Ok(vec![
                            inputs[1].clone(),
                            u32_slice_to_le_bytes(&reference_bracket_pairs(
                                &kinds,
                                depth_words as u32,
                            )),
                        ])
                    }
                    "vyre-libs::matching::region::region_sort" => {
                        // Six input-consuming buffers: pids/starts/ends ReadOnly(0-2) + the three
                        // plain-ReadWrite outputs pids_out/starts_out/ends_out(3-5, zero-filled).
                        assert_eq!(
                inputs.len(),
                6,
                "Fix: sort_regions_via must pass all six input-consuming buffers (3 RO + 3 plain-RW outputs)."
            );
                        let regions = join_regions(
                            &crate::dispatch_buffers::read_u32s(&inputs[0]),
                            &crate::dispatch_buffers::read_u32s(&inputs[1]),
                            &crate::dispatch_buffers::read_u32s(&inputs[2]),
                        );

                        let sorted = reference_sort_regions(regions);
                        let (pids, starts, ends) = split_regions(&sorted);
                        Ok(vec![
                            u32_slice_to_le_bytes(&pids),
                            u32_slice_to_le_bytes(&starts),
                            u32_slice_to_le_bytes(&ends),
                        ])
                    }
                    "vyre-libs::matching::region::dedup_regions_flag" => {
                        // Three input-consuming buffers: pids/starts/ends ReadOnly(0-2). `survivors` is
                        // WriteOnly(3) (backend-allocated, no input slot).
                        assert_eq!(
                inputs.len(),
                3,
                "Fix: dedup_region_survivor_flags_via must pass exactly the three RO buffers; survivors is backend-allocated."
            );
                        let regions = join_regions(
                            &crate::dispatch_buffers::read_u32s(&inputs[0]),
                            &crate::dispatch_buffers::read_u32s(&inputs[1]),
                            &crate::dispatch_buffers::read_u32s(&inputs[2]),
                        );

                        let flags = survivor_flags(&regions);
                        Ok(vec![u32_slice_to_le_bytes(&flags)])
                    }
                    other => panic!("unexpected matching primitive op id {other}"),
                }
            })()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    fn join_regions(pids: &[u32], starts: &[u32], ends: &[u32]) -> Vec<RegionTriple> {
        pids.iter()
            .zip(starts.iter())
            .zip(ends.iter())
            .map(|((pid, start), end)| RegionTriple::new(*pid, *start, *end))
            .collect()
    }

    fn survivor_flags(sorted_regions: &[RegionTriple]) -> Vec<u32> {
        let mut flags = Vec::with_capacity(sorted_regions.len());
        for (index, current) in sorted_regions.iter().enumerate() {
            let has_prev_overlap = sorted_regions[..index]
                .iter()
                .any(|prior| prior.pid == current.pid && prior.end >= current.start);
            flags.push(u32::from(!has_prev_overlap));
        }
        flags
    }

    #[test]
    fn dfa_compile_wrappers_use_primitive_compiler() {
        let patterns: &[&[u8]] = &[b"error", b"warning"];
        let default = compile_diagnostic_dfa(patterns);
        let budgeted = compile_diagnostic_dfa_with_budget(patterns, 1 << 20).unwrap();
        assert_eq!(default.state_count, budgeted.state_count);
        assert_eq!(default.max_pattern_len, 7);
    }

    #[test]
    fn bracket_pairs_dispatch_through_primitive() {
        let fixture = nested_diagnostic_brace_fixture();
        assert_eq!(
            bracket_pairs_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &fixture,
                8,
            )
            .unwrap(),
            reference_bracket_pairs(&fixture, 8)
        );
    }

    #[test]
    fn bracket_pairs_uncapped_large_stream_dispatches_all_parallel_workgroups() {
        let mut kinds = vec![BRACKET_KIND_OTHER; 513];
        kinds[0] = BRACKET_KIND_OPEN;
        kinds[255] = BRACKET_KIND_OPEN;
        kinds[256] = BRACKET_KIND_CLOSE;
        kinds[512] = BRACKET_KIND_CLOSE;

        assert_eq!(
            bracket_pairs_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &kinds,
                kinds.len() as u32,
            )
            .unwrap(),
            reference_bracket_pairs(&kinds, kinds.len() as u32)
        );
    }

    #[test]
    fn bracket_pairs_depth_capped_stream_keeps_single_workgroup_fallback() {
        let mut kinds = vec![BRACKET_KIND_OTHER; 513];
        kinds[0] = BRACKET_KIND_OPEN;
        kinds[64] = BRACKET_KIND_OPEN;
        kinds[65] = BRACKET_KIND_CLOSE;

        assert_eq!(
            bracket_pairs_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &kinds,
                64,
            )
            .unwrap(),
            reference_bracket_pairs(&kinds, 64)
        );
    }

    #[test]
    fn bracket_pairs_generated_dispatch_grids_cover_4096_large_streams() {
        for case in 0..4096u32 {
            let len = 257 + (case.wrapping_mul(37) % 768) as usize;
            let max_depth = if case % 2 == 0 {
                len as u32
            } else {
                1 + case.wrapping_mul(19) % 192
            };
            let mut state = 0x8BAD_F00Du32 ^ case.wrapping_mul(0x9E37_79B9);
            let mut kinds = Vec::with_capacity(len);
            for index in 0..len {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let kind = match (state.wrapping_add(index as u32)) % 7 {
                    0 | 1 => BRACKET_KIND_OPEN,
                    2 | 3 => BRACKET_KIND_CLOSE,
                    _ => BRACKET_KIND_OTHER,
                };
                kinds.push(kind);
            }

            assert_eq!(
                bracket_pairs_via(
                    &MatchingDispatcher,
                    &crate::test_parity_oracles::policy(),
                    &kinds,
                    max_depth,
                )
                .unwrap(),
                reference_bracket_pairs(&kinds, max_depth),
                "case {case}: diagnostic bracket dispatch must match primitive CPU oracle"
            );
        }
    }

    #[test]
    fn dedup_survivor_flags_nested_cluster_uses_prior_merged_span() {
        let sorted = vec![
            RegionTriple::new(7, 0, 10),
            RegionTriple::new(7, 2, 3),
            RegionTriple::new(7, 9, 12),
            RegionTriple::new(7, 20, 25),
        ];

        assert_eq!(
            dedup_region_survivor_flags_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &sorted,
            )
            .unwrap(),
            vec![1, 0, 0, 1]
        );
    }

    #[test]
    fn dedup_survivor_flags_large_stream_dispatches_region_grid() {
        let sorted = (0..513u32)
            .map(|index| RegionTriple::new(index / 171, index * 3, index * 3 + 1))
            .collect::<Vec<_>>();

        assert_eq!(
            dedup_region_survivor_flags_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &sorted,
            )
            .unwrap(),
            vec![1; sorted.len()]
        );
    }

    #[test]
    fn dedup_survivor_flags_generated_regions_cover_4096_large_streams() {
        for case in 0..4096u32 {
            let count = 257 + (case.wrapping_mul(29) % 768) as usize;
            let mut state = 0xD1CE_C0DEu32 ^ case.wrapping_mul(0x85EB_CA6B);
            let mut regions = Vec::with_capacity(count);
            for index in 0..count {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let pid = state % 7;
                state = state.rotate_left(3).wrapping_add(index as u32);
                let start = state % 4096;
                state = state.rotate_left(9) ^ case;
                let width = state % 64;
                regions.push(RegionTriple::new(pid, start, start.saturating_add(width)));
            }

            let mut sorted = regions;
            sort_regions_witness_in_place(&mut sorted);
            let flags = dedup_region_survivor_flags_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &sorted,
            )
            .unwrap();
            let actual_cluster_starts = sorted
                .iter()
                .zip(flags.iter())
                .filter_map(|(region, flag)| (*flag != 0).then_some((region.pid, region.start)))
                .collect::<Vec<_>>();
            let expected_cluster_starts = reference_dedup_regions(sorted.clone())
                .into_iter()
                .map(|region| (region.pid, region.start))
                .collect::<Vec<_>>();

            assert_eq!(
                actual_cluster_starts, expected_cluster_starts,
                "case {case}: survivor flags must mark the same cluster starts as CPU dedup"
            );
        }
    }

    #[test]
    fn region_reference_wrappers_match_primitives_exactly() {
        let regions = vec![
            RegionTriple::new(0, 7, 10),
            RegionTriple::new(0, 5, 8),
            RegionTriple::new(1, 5, 8),
        ];
        let mut in_place = regions.clone();
        reference_dedup_regions_inplace(&mut in_place);
        assert_eq!(in_place, reference_dedup_regions(regions));
    }

    #[test]
    fn region_sort_dispatches_primitive_shape() {
        let regions = vec![
            RegionTriple::new(2, 0, 1),
            RegionTriple::new(0, 5, 10),
            RegionTriple::new(0, 5, 8),
        ];
        assert_eq!(
            sort_regions_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &regions,
            )
            .unwrap(),
            reference_sort_regions(regions)
        );
    }

    #[test]
    fn region_sort_reuses_caller_owned_split_and_decode_capacity() {
        let large = (0..128)
            .map(|idx| RegionTriple::new(idx % 7, 128 - idx, 128 - idx + 3))
            .collect::<Vec<_>>();
        let small = vec![RegionTriple::new(1, 2, 3), RegionTriple::new(0, 1, 4)];
        let mut scratch = MatchingDiagnosticCompactionGpuScratch::default();
        let mut out = Vec::new();

        sort_regions_via_with_scratch_into(
            &MatchingDispatcher,
            &crate::test_parity_oracles::policy(),
            &large,
            &mut scratch,
            &mut out,
        )
        .expect("Fix: large diagnostic region sort should dispatch");
        let pids_capacity = scratch.pids.capacity();
        let decoded_capacity = scratch.decoded_regions.capacity();

        sort_regions_via_with_scratch_into(
            &MatchingDispatcher,
            &crate::test_parity_oracles::policy(),
            &small,
            &mut scratch,
            &mut out,
        )
        .expect("Fix: small diagnostic region sort should reuse scratch");

        assert_eq!(scratch.pids.capacity(), pids_capacity);
        assert_eq!(scratch.decoded_regions.capacity(), decoded_capacity);
        assert_eq!(out, reference_sort_regions(small));
    }

    #[test]
    fn dedup_flags_dispatches_primitive_shape() {
        let sorted = vec![
            RegionTriple::new(0, 5, 8),
            RegionTriple::new(0, 7, 10),
            RegionTriple::new(1, 7, 10),
        ];
        assert_eq!(
            dedup_region_survivor_flags_via(
                &MatchingDispatcher,
                &crate::test_parity_oracles::policy(),
                &sorted,
            )
            .unwrap(),
            vec![1, 0, 1]
        );
    }

    #[test]
    fn dedup_flags_reuses_caller_owned_split_capacity() {
        let large = (0..63)
            .map(|idx| RegionTriple::new(idx % 11, idx, idx + 2))
            .collect::<Vec<_>>();
        let small = vec![
            RegionTriple::new(0, 0, 2),
            RegionTriple::new(0, 1, 3),
            RegionTriple::new(1, 1, 3),
        ];
        let mut scratch = MatchingDiagnosticCompactionGpuScratch::default();
        let mut flags = Vec::new();

        dedup_region_survivor_flags_via_with_scratch_into(
            &MatchingDispatcher,
            &crate::test_parity_oracles::policy(),
            &large,
            &mut scratch,
            &mut flags,
        )
        .expect("Fix: large diagnostic dedup should dispatch");
        let pids_capacity = scratch.pids.capacity();

        dedup_region_survivor_flags_via_with_scratch_into(
            &MatchingDispatcher,
            &crate::test_parity_oracles::policy(),
            &small,
            &mut scratch,
            &mut flags,
        )
        .expect("Fix: small diagnostic dedup should reuse scratch");

        assert_eq!(scratch.pids.capacity(), pids_capacity);
        assert_eq!(flags, vec![1, 0, 1]);
    }

    #[test]
    fn empty_region_sort_error_is_actionable() {
        let err = sort_regions_via(
            &MatchingDispatcher,
            &crate::test_parity_oracles::policy(),
            &[],
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("Fix: sort_regions_via requires at least one region"));
    }
}
