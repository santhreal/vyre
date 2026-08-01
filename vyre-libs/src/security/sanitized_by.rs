//! `sanitized_by`  -  Tier-3 sanitizer-gated forward taint step.
//!
//! Semantics:
//!
//! ```text
//!   frontier_clean = frontier_in \ sanitizers_in       (set difference)
//!   frontier_out   = csr_forward_traverse(frontier_clean, FLOWS_TO_MASK)
//! ```
//!
//! Two stages, fused into one Program:
//!
//! 1. `frontier_clean = frontier_in & !sanitizers_in` via the new
//!    `bitset_and_not` primitive  -  one Region instead of two
//!    (`bitset_not` + `bitset_and`)  -  fewer scratch buffers, fewer
//!    dispatch-time bind-point allocations.
//! 2. `frontier_out = csr_forward_traverse(frontier_clean, …)`
//!    along genuine dataflow edges only (`FLOWS_TO_MASK`).
//!
//! Pre-fix this composed three primitives via `fuse_programs(...)`
//! and threaded an `__sanitized_by_allow__*` scratch buffer; the new
//! `bitset_and_not` collapses the first two stages into a single
//! Region with no scratch, eliminating one buffer + one dispatch
//! per call.

use vyre::ir::DataType;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_primitives::bitset::and_not::bitset_and_not;
use vyre_primitives::graph::csr_forward_traverse::{bitset_words, csr_forward_traverse};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::predicate::edge_kind;

const OP_ID: &str = "vyre-libs::security::sanitized_by";

/// Build one sanitizer-guarded forward-traversal step.
///
/// `sanitizers_in` names the bitset buffer holding the sanitizer
/// nodeset. The emitted Program AND-NOTs the sanitizers against the
/// current frontier before traversing dataflow edges.
///
/// Reduced from three primitives (`bitset_not` + `bitset_and` +
/// `csr_forward_traverse`) to two by composing the first stage as
/// the new `bitset_and_not` (`frontier_clean = frontier_in & !sanitizers_in`
/// in one Region). One fewer scratch buffer, one fewer dispatch.
#[must_use]
pub fn sanitized_by(
    shape: ProgramGraphShape,
    frontier_in: &str,
    sanitizers_in: &str,
    frontier_out: &str,
) -> vyre::ir::Program {
    crate::security::assert_security_inputs(
        OP_ID,
        shape.node_count,
        &[
            ("frontier_in", frontier_in),
            ("sanitizers_in", sanitizers_in),
            ("frontier_out", frontier_out),
        ],
    );
    let words = bitset_words(shape.node_count);
    let clean_buffer = format!("__sanitized_by_clean__{frontier_in}");
    let clean = bitset_and_not(frontier_in, sanitizers_in, &clean_buffer, words);
    let traverse = csr_forward_traverse(
        shape,
        &clean_buffer,
        frontier_out,
        crate::security::flows_to::FLOWS_TO_MASK,
    );
    let fused = fuse_programs(&[clean, traverse]).unwrap_or_else(|error| {
        crate::builder::invalid_output_program(
            OP_ID,
            frontier_out,
            DataType::U32,
            format!("Fix: sanitized_by composition failed: {error}"),
        )
    });
    crate::region::tag_program(OP_ID, fused)
}

inventory::submit! {
    crate::harness::OpEntry {
        id: OP_ID,
        build: || sanitized_by(ProgramGraphShape::new(4, 3), "fin", "san", "fout"),
        test_inputs: Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            // Linear 0→1→2→3 with node 1 marked sanitizer.
            vec![vec![
                to_bytes(&[0b0001]),              // 0: fin = {0}
                to_bytes(&[0b0010]),              // 1: san = {1}
                to_bytes(&[0b0000]),              // 2: internal clean scratch
                to_bytes(&[0, 0, 0, 0]),          // 3: pg_nodes
                to_bytes(&[0, 1, 2, 3, 3]),       // 4: pg_edge_offsets
                to_bytes(&[1, 2, 3]),             // 5: pg_edge_targets
                to_bytes(&[
                    edge_kind::ASSIGNMENT,
                    edge_kind::ASSIGNMENT,
                    edge_kind::ASSIGNMENT,
                ]),                               // 6: pg_edge_kind_mask
                to_bytes(&[0, 1, 0, 0]),          // 7: pg_node_tags: node 1 is sanitizer
                to_bytes(&[0b0001]),              // 8: fout accumulator seed = {0}
            ]]
        }),
        expected_output: Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            // One forward step from {0}: fout accumulator = {0,1}.
            vec![vec![
                to_bytes(&[0b0001]),              // clean_buf = fin & !san
                to_bytes(&[0b0011]),              // fout
            ]]
        }),
        category: Some("security"),
    }
}

inventory::submit! {
    // AUDIT_2026-04-24 F-SB-01: raised from 64 to 4096 so taint
    // sanitization on deep call chains doesn't truncate silently;
    // same reasoning as flows_to / taint_flow.
    crate::harness::ConvergenceContract {
        op_id: OP_ID,
        max_iterations: 4096,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_primitives::predicate::edge_kind;

    #[test]
    fn sanitized_by_declares_sanitizer_buffer() {
        let p = sanitized_by(ProgramGraphShape::new(4, 3), "fin", "san", "fout");
        let names: Vec<&str> = p.buffers().iter().map(|b| b.name()).collect();
        assert!(names.contains(&"fin"), "frontier_in must be declared");
        assert!(names.contains(&"san"), "sanitizers_in must be declared");
        assert!(names.contains(&"fout"), "frontier_out must be declared");
    }

    #[test]
    fn sanitized_by_uses_dataflow_mask_not_universal() {
        // The traversal stage must not regress to 0xFFFF_FFFF.
        // We verify indirectly: the composed Program must not have
        // the universal mask literal. Since the mask is embedded in
        // the inner csr_forward_traverse, we check the FLOWS_TO_MASK
        // constant surface.
        use crate::security::flows_to::FLOWS_TO_MASK;
        assert_eq!(FLOWS_TO_MASK & edge_kind::CONTROL, 0);
        assert_eq!(FLOWS_TO_MASK & edge_kind::DOMINANCE, 0);
    }

    #[test]
    fn sanitized_by_program_uses_non_degenerate_shape() {
        let shape = ProgramGraphShape::new(64, 128);
        let p = sanitized_by(shape, "fin", "san", "fout");
        let fin_buf = p
            .buffers()
            .iter()
            .find(|b| b.name() == "fin")
            .expect("Fix: fin buffer");
        assert!(
            fin_buf.count >= 2,
            "bitset_words(64) = 2; count {} suggests degenerate shape",
            fin_buf.count
        );
    }

    #[test]
    fn sanitized_by_marks_sanitizer_when_taint_arrives_at_it() {
        // Linear 0->1->2->3, fin = {0}, san = {1}, fout seed = {0}.
        // After one forward step, the sanitizer node 1 IS marked in fout
        // (so audit/forensics consumers can answer "did taint reach this
        // sanitizer?"). Propagation FROM the sanitizer is blocked  -  the
        // separate test `sanitized_by_blocks_propagation_from_sanitizer_node`
        // proves that. The two tests together pin down the canonical
        // taint-with-sanitizer semantics: mark on arrival, cut on
        // departure. Matches CodeQL/Semgrep/Joern.
        let p = sanitized_by(ProgramGraphShape::new(4, 3), "fin", "san", "fout");
        let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
        let inputs = vec![
            to_bytes(&[0b0001]),
            to_bytes(&[0b0010]),
            to_bytes(&[0b0000]),
            to_bytes(&[0, 0, 0, 0]),
            to_bytes(&[0, 1, 2, 3, 3]),
            to_bytes(&[1, 2, 3]),
            to_bytes(&[
                edge_kind::ASSIGNMENT,
                edge_kind::ASSIGNMENT,
                edge_kind::ASSIGNMENT,
            ]),
            to_bytes(&[0, 1, 0, 0]),
            to_bytes(&[0b0001]),
        ];
        let values: Vec<vyre_reference::value::Value> = inputs
            .into_iter()
            .map(vyre_reference::value::Value::from)
            .collect();
        let outputs = vyre_reference::reference_eval(&p, &values).unwrap();
        let fout_word = u32::from_le_bytes(outputs[1].to_bytes()[0..4].try_into().unwrap());
        assert_eq!(
            fout_word, 0b0011,
            "sanitized_by must mark the sanitizer when taint arrives at it; \
             observability of 'taint hit this sanitizer' is the entire point  -  \
             without it, downstream SARIF/audit consumers cannot distinguish \
             'sanitized at node 1' from 'never reached node 1'."
        );
    }

    #[test]
    fn sanitized_by_blocks_propagation_from_sanitizer_node() {
        let p = sanitized_by(ProgramGraphShape::new(3, 2), "fin", "san", "fout");
        let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
        let inputs = vec![
            to_bytes(&[0b0010]),     // fin = {1}
            to_bytes(&[0b0010]),     // san = {1}
            to_bytes(&[0b0000]),     // clean scratch
            to_bytes(&[0, 0, 0]),    // pg_nodes
            to_bytes(&[0, 1, 2, 2]), // pg_edge_offsets: 0→{1}, 1→{2}, 2→{}
            to_bytes(&[1, 2]),       // pg_edge_targets
            to_bytes(&[edge_kind::ASSIGNMENT, edge_kind::ASSIGNMENT]),
            to_bytes(&[0, 1, 0]), // pg_node_tags
            to_bytes(&[0b0010]),  // fout seed = {1}
        ];
        let values: Vec<vyre_reference::value::Value> = inputs
            .into_iter()
            .map(vyre_reference::value::Value::from)
            .collect();
        let outputs = vyre_reference::reference_eval(&p, &values).unwrap();
        let fout_word = u32::from_le_bytes(outputs[1].to_bytes()[0..4].try_into().unwrap());
        assert_eq!(
            fout_word, 0b0010,
            "sanitized_by must NOT propagate from sanitizer node 1; fout should remain {{1}}"
        );
    }

    #[test]
    #[should_panic(expected = "node_count must be positive")]
    fn sanitized_by_zero_node_count_should_panic() {
        let _ = sanitized_by(ProgramGraphShape::new(0, 0), "fin", "san", "fout");
    }

    #[test]
    #[should_panic(expected = "empty buffer name")]
    fn sanitized_by_empty_buffer_name_should_panic() {
        let _ = sanitized_by(ProgramGraphShape::new(4, 3), "", "san", "fout");
    }
}
