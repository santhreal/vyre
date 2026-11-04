//! Shape-checked CPU oracles for the do-calculus rules.
//!
//! The reference witnesses in `vyre_reference::composition_witness` are
//! infallible and assume well-formed operands. Every do-calculus test wants the
//! fallible form, so each test module grew its own copy of the same two length
//! checks around the same witness call, with the rule name spliced into the
//! message. Three copies lived in one file and a fourth in its sibling.
//!
//! One owner for the checks means a test cannot accidentally assert against a
//! looser oracle than its neighbour.

use vyre_reference::composition_witness::{
    do_intervention_delete_incoming_witness_into, do_rule2_reverse_incoming_witness_into,
    do_rule3_subgraph_witness_into,
};

/// Reject operands that are not an `n × n` matrix and an `n` lane mask.
///
/// `rule` names the caller in the message, which is what the tests assert on.
fn square_matrix_and_mask(
    rule: &str,
    mask_name: &str,
    adjacency: &[u32],
    mask: &[u32],
    n: u32,
) -> Result<(), String> {
    let n_usize = n as usize;
    if adjacency.len() != n_usize * n_usize {
        return Err(format!(
            "Fix: {rule} requires adjacency.len() == n*n, got {} vs {}.",
            adjacency.len(),
            n_usize * n_usize
        ));
    }
    if mask.len() != n_usize {
        return Err(format!(
            "Fix: {rule} requires {mask_name}.len() == n, got {} vs {}.",
            mask.len(),
            n_usize
        ));
    }
    Ok(())
}

/// Post-do adjacency, written into `out`, or a typed error that leaves `out` alone.
pub(crate) fn try_do_intervention_delete_incoming_cpu_into(
    adjacency: &[u32],
    intervention_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    square_matrix_and_mask(
        "do_intervention",
        "intervention_mask",
        adjacency,
        intervention_mask,
        n,
    )?;
    do_intervention_delete_incoming_witness_into(adjacency, intervention_mask, n, out);
    Ok(())
}

/// Rule 2 reversal, written into `out`, or a typed error that leaves `out` alone.
pub(crate) fn try_do_rule2_reverse_incoming_cpu_into(
    adjacency: &[u32],
    treatment_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    square_matrix_and_mask("rule2", "treatment_mask", adjacency, treatment_mask, n)?;
    do_rule2_reverse_incoming_witness_into(adjacency, treatment_mask, n, out);
    Ok(())
}

/// Rule 3 subgraph, written into `reduced` and `kept`, or a typed error that
/// leaves both alone.
pub(crate) fn try_do_rule3_subgraph_cpu_into(
    adjacency: &[u32],
    keep_mask: &[u32],
    n: u32,
    reduced: &mut Vec<u32>,
    kept: &mut Vec<u32>,
) -> Result<(), String> {
    square_matrix_and_mask("rule3", "keep_mask", adjacency, keep_mask, n)?;
    do_rule3_subgraph_witness_into(adjacency, keep_mask, n, reduced, kept);
    Ok(())
}

/// Rule 3 subgraph returning freshly allocated buffers.
pub(crate) fn try_do_rule3_subgraph_cpu(
    adjacency: &[u32],
    keep_mask: &[u32],
    n: u32,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut reduced = Vec::new();
    let mut kept = Vec::new();
    try_do_rule3_subgraph_cpu_into(adjacency, keep_mask, n, &mut reduced, &mut kept)?;
    Ok((reduced, kept))
}
