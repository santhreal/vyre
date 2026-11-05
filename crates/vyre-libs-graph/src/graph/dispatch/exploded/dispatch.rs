//! Host-side dispatch of the exploded IFDS CSR build, including readback
//! validation and row canonicalisation.

use super::{CachedIfdsCsrProgram, IfdsCsrGpuScratch};
use crate::graph::exploded::{
    canonicalize_csr_within_rows_in_place as primitive_canonicalize_csr_within_rows_in_place,
    plan_ifds_csr_dispatch, validate_ifds_csr_readback, IfdsCsrDispatchPlan,
    IfdsCsrRuleInputFingerprint, IfdsCsrStaticInputKey, IFDS_CSR_COL_IDX_BUFFER,
    IFDS_CSR_COL_LEN_BUFFER, IFDS_CSR_GEN_BLOCK_BUFFER, IFDS_CSR_GEN_FACT_BUFFER,
    IFDS_CSR_GEN_PROC_BUFFER, IFDS_CSR_INTER_DST_BLOCK_BUFFER, IFDS_CSR_INTER_DST_PROC_BUFFER,
    IFDS_CSR_INTER_SRC_BLOCK_BUFFER, IFDS_CSR_INTER_SRC_PROC_BUFFER,
    IFDS_CSR_INTRA_DST_BLOCK_BUFFER, IFDS_CSR_INTRA_PROC_BUFFER, IFDS_CSR_INTRA_SRC_BLOCK_BUFFER,
    IFDS_CSR_KILLED_BUFFER, IFDS_CSR_KILL_BLOCK_BUFFER, IFDS_CSR_KILL_FACT_BUFFER,
    IFDS_CSR_KILL_PROC_BUFFER, IFDS_CSR_ROW_CURSOR_BUFFER, IFDS_CSR_ROW_PTR_BUFFER,
};

use crate::graph::dispatch::dispatch_bridge::{
    dispatch_u32_outputs_from_prepared_into, refresh_keyed_dispatch_inputs, DispatchInput,
    U32Readback,
};
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

/// GPU dispatch wrapper around the `build_cpu_reference` CPU oracle.
///
/// Returns the supergraph CSR in canonical (within-row sorted) form
/// so callers comparing against the reference oracle don't need to
/// re-canonicalise the output.
///
/// # Errors
///
/// Propagates dispatch failures and rejects dimensions or readback
/// shapes that cannot represent an exploded CSR safely.
#[allow(clippy::too_many_arguments)]
pub fn build_ifds_csr_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    flow_kill: &[(u32, u32, u32)],
) -> Result<(Vec<u32>, Vec<u32>), SemanticExecutionError> {
    let mut row_ptr = Vec::new();
    let mut col_idx = Vec::new();
    build_ifds_csr_via_into(
        dispatcher,
        policy,
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
        &mut row_ptr,
        &mut col_idx,
    )?;
    Ok((row_ptr, col_idx))
}

/// GPU dispatch wrapper around the `build_cpu_reference` CPU oracle into caller-owned CSR buffers.
#[allow(clippy::too_many_arguments)]
pub fn build_ifds_csr_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    flow_kill: &[(u32, u32, u32)],
    row_ptr_out: &mut Vec<u32>,
    col_idx_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = IfdsCsrGpuScratch::default();
    build_ifds_csr_via_with_scratch_into(
        dispatcher,
        policy,
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
        &mut scratch,
        row_ptr_out,
        col_idx_out,
    )
}

/// GPU dispatch wrapper around the `build_cpu_reference` CPU oracle into caller-owned
/// dispatch scratch and CSR buffers.
#[allow(clippy::too_many_arguments)]
pub fn build_ifds_csr_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    flow_kill: &[(u32, u32, u32)],
    scratch: &mut IfdsCsrGpuScratch,
    row_ptr_out: &mut Vec<u32>,
    col_idx_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let plan = plan_ifds_csr_dispatch(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
    )
    .map_err(SemanticExecutionError::InvalidRequest)?;
    if plan.layout.empty {
        row_ptr_out.clear();
        row_ptr_out.push(0);
        col_idx_out.clear();
        return Ok(());
    }
    let rule_fingerprint =
        IfdsCsrRuleInputFingerprint::from_rules(intra_edges, inter_edges, flow_gen, flow_kill);
    if scratch.rule_fingerprint != Some(rule_fingerprint) {
        scratch
            .rule_columns
            .prepare(intra_edges, inter_edges, flow_gen, flow_kill)
            .map_err(|error| {
                SemanticExecutionError::Backend(format!(
                    "Fix: exploded IFDS wrapper could not marshal primitive rule columns: {error}"
                ))
            })?;
        scratch.rule_fingerprint = Some(rule_fingerprint);
    }
    let rule_columns = &scratch.rule_columns;
    let static_input_key = plan.static_input_key(rule_fingerprint);
    refresh_ifds_csr_inputs(
        &mut scratch.inputs,
        &mut scratch.static_input_key,
        static_input_key,
        &plan,
        &[
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.intra_proc,
                plan.intra_field_words,
                IFDS_CSR_INTRA_PROC_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.intra_src_block,
                plan.intra_field_words,
                IFDS_CSR_INTRA_SRC_BLOCK_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.intra_dst_block,
                plan.intra_field_words,
                IFDS_CSR_INTRA_DST_BLOCK_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.inter_src_proc,
                plan.inter_field_words,
                IFDS_CSR_INTER_SRC_PROC_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.inter_src_block,
                plan.inter_field_words,
                IFDS_CSR_INTER_SRC_BLOCK_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.inter_dst_proc,
                plan.inter_field_words,
                IFDS_CSR_INTER_DST_PROC_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.inter_dst_block,
                plan.inter_field_words,
                IFDS_CSR_INTER_DST_BLOCK_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.gen_proc,
                plan.gen_field_words,
                IFDS_CSR_GEN_PROC_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.gen_block,
                plan.gen_field_words,
                IFDS_CSR_GEN_BLOCK_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.gen_fact,
                plan.gen_field_words,
                IFDS_CSR_GEN_FACT_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.kill_proc,
                plan.kill_field_words,
                IFDS_CSR_KILL_PROC_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.kill_block,
                plan.kill_field_words,
                IFDS_CSR_KILL_BLOCK_BUFFER,
            ),
            DispatchInput::u32_slice_or_zero_words(
                &rule_columns.kill_fact,
                plan.kill_field_words,
                IFDS_CSR_KILL_FACT_BUFFER,
            ),
            DispatchInput::zero_u32_words(plan.killed_words, IFDS_CSR_KILLED_BUFFER),
            DispatchInput::zero_u32_words(plan.row_ptr_words, IFDS_CSR_ROW_PTR_BUFFER),
            DispatchInput::zero_u32_words(plan.row_cursor_words, IFDS_CSR_ROW_CURSOR_BUFFER),
            DispatchInput::zero_u32_words(plan.col_idx_words, IFDS_CSR_COL_IDX_BUFFER),
            DispatchInput::zero_u32_words(plan.col_len_words, IFDS_CSR_COL_LEN_BUFFER),
        ],
    )?;
    let cached = scratch
        .program_cache
        .get_or_insert_with(plan.program_cache_key(), || CachedIfdsCsrProgram {
            program: plan.program(),
        });
    dispatch_ifds_csr_outputs_from_prepared_into(
        dispatcher,
        policy,
        &cached.program,
        &scratch.inputs,
        plan.row_ptr_words,
        row_ptr_out,
        plan.row_cursor_words,
        &mut scratch.row_cursor,
        plan.col_idx_words,
        col_idx_out,
        plan.col_len_words,
        &mut scratch.col_len_words,
    )?;
    let col_len = validate_ifds_csr_readback(
        &plan.layout,
        row_ptr_out,
        col_idx_out,
        scratch.col_len_words[0],
    )
    .map_err(SemanticExecutionError::Backend)?;
    col_idx_out.truncate(col_len);
    canonicalize_csr_within_rows_in_place(row_ptr_out, col_idx_out)?;
    Ok(())
}

/// Read the four CSR buffers the IFDS build declares as its result.
///
/// Each readback names a declared Program buffer, so the kill bitmap the
/// program also writes needs no attention here: it is written storage no caller
/// reads, and a position in the result list is not a name.
#[allow(clippy::too_many_arguments)]
fn dispatch_ifds_csr_outputs_from_prepared_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    program: &vyre_foundation::ir::Program,
    scratch_inputs: &[Vec<u8>],
    row_ptr_expected_words: usize,
    row_ptr_out: &mut Vec<u32>,
    row_cursor_expected_words: usize,
    row_cursor_out: &mut Vec<u32>,
    col_idx_expected_words: usize,
    col_idx_out: &mut Vec<u32>,
    col_len_expected_words: usize,
    col_len_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    dispatch_u32_outputs_from_prepared_into(
        dispatcher,
        policy,
        program.clone(),
        scratch_inputs,
        &mut [
            U32Readback {
                buffer: "row_ptr",
                words: row_ptr_expected_words,
                out: row_ptr_out,
            },
            U32Readback {
                buffer: "row_cursor",
                words: row_cursor_expected_words,
                out: row_cursor_out,
            },
            U32Readback {
                buffer: "col_idx",
                words: col_idx_expected_words,
                out: col_idx_out,
            },
            U32Readback {
                buffer: "col_len",
                words: col_len_expected_words,
                out: col_len_out,
            },
        ],
    )
}

fn canonicalize_csr_within_rows_in_place(
    row_ptr: &[u32],
    col_idx: &mut [u32],
) -> Result<(), SemanticExecutionError> {
    primitive_canonicalize_csr_within_rows_in_place(row_ptr, col_idx)
        .map_err(SemanticExecutionError::Backend)
}

fn refresh_ifds_csr_inputs(
    inputs: &mut Vec<Vec<u8>>,
    static_input_key: &mut Option<IfdsCsrStaticInputKey>,
    next_static_input_key: IfdsCsrStaticInputKey,
    plan: &IfdsCsrDispatchPlan,
    all_inputs: &[DispatchInput<'_>],
) -> Result<(), SemanticExecutionError> {
    refresh_keyed_dispatch_inputs(
        inputs,
        static_input_key,
        next_static_input_key,
        all_inputs,
        &[
            (
                13,
                DispatchInput::zero_u32_words(plan.killed_words, IFDS_CSR_KILLED_BUFFER),
            ),
            (
                14,
                DispatchInput::zero_u32_words(plan.row_ptr_words, IFDS_CSR_ROW_PTR_BUFFER),
            ),
            (
                15,
                DispatchInput::zero_u32_words(plan.row_cursor_words, IFDS_CSR_ROW_CURSOR_BUFFER),
            ),
            (
                16,
                DispatchInput::zero_u32_words(plan.col_idx_words, IFDS_CSR_COL_IDX_BUFFER),
            ),
            (
                17,
                DispatchInput::zero_u32_words(plan.col_len_words, IFDS_CSR_COL_LEN_BUFFER),
            ),
        ],
    )
}
