//! `sanitized_by`  -  Tier-3 sanitizer-gated forward taint step.
//!
//! The operation traverses dataflow edges from source nodes selected by
//! `frontier_in & !sanitizers_in`. Reached sanitizer nodes remain observable in
//! `frontier_out`; only later propagation from those nodes is blocked.
//!
//! Source filtering is part of the CSR traversal stage. No intermediate
//! clean-frontier buffer or cross-dispatch synchronization is required.

use crate::graph::csr_forward_traverse::csr_forward_traverse_excluding;
use crate::graph::program_graph::ProgramGraphShape;
use vyre_foundation::composition::tag_program;

pub(crate) const OP_ID: &str = "vyre-libs::security::sanitized_by";

/// Build one sanitizer-guarded forward-traversal step.
///
/// `sanitizers_in` selects nodes that may be reached and recorded but may not
/// act as traversal sources.
#[must_use]
pub fn sanitized_by(
    shape: ProgramGraphShape,
    frontier_in: &str,
    sanitizers_in: &str,
    frontier_out: &str,
) -> vyre_foundation::ir::Program {
    crate::security::assert_security_inputs(
        OP_ID,
        shape.node_count,
        &[
            ("frontier_in", frontier_in),
            ("sanitizers_in", sanitizers_in),
            ("frontier_out", frontier_out),
        ],
    );
    let traverse = csr_forward_traverse_excluding(
        shape,
        frontier_in,
        sanitizers_in,
        frontier_out,
        crate::security::flows_to::FLOWS_TO_MASK,
    );
    tag_program(OP_ID, traverse)
}

pub(crate) const EXPECTED_SANITIZED_BY_OUTPUT_BYTES: [u8; 4] = [0x03, 0x00, 0x00, 0x00];

pub(crate) fn sanitized_by_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 0: pg_nodes
        vec![0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0], // 1: pg_edge_offsets
        vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],             // 2: pg_edge_targets
        vec![1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],             // 3: pg_edge_kind_mask (ASSIGNMENT=1)
        vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 4: pg_node_tags
        vec![1, 0, 0, 0],                                     // 5: fin = {0}
        vec![2, 0, 0, 0],                                     // 6: san = {1}
        vec![1, 0, 0, 0],                                     // 7: fout accumulator seed = {0}
    ]]
}

#[cfg(test)]
pub(crate) fn sanitized_by_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    vec![vec![EXPECTED_SANITIZED_BY_OUTPUT_BYTES.to_vec()]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate::edge_kind;

    #[test]
    fn test_sanitized_by_expected_bytes_identity() {
        let constructed = crate::fixture_bytes::u32_bytes(&[3]);
        assert_eq!(constructed, EXPECTED_SANITIZED_BY_OUTPUT_BYTES);
    }

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
    fn registration_fixture_matches_exact_byte_constant() {
        assert_eq!(EXPECTED_SANITIZED_BY_OUTPUT_BYTES, [0x03, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn sanitized_by_program_uses_non_degenerate_shape() {
        let shape = ProgramGraphShape::new(64, 128);
        let p = sanitized_by(shape, "fin", "san", "fout");
        crate::security::flow_composition::assert_non_degenerate_bitset_shape(&p, "fin", 2);
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
            to_bytes(&[0b0010]),
            to_bytes(&[0b0001]),
        ];
        let values: Vec<vyre_reference::value::Value> = inputs
            .into_iter()
            .map(vyre_reference::value::Value::from)
            .collect();
        let outputs = vyre_reference::reference_eval(&p, &values).unwrap();
        let fout_word = u32::from_le_bytes(outputs[0].to_bytes()[0..4].try_into().unwrap());
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
            to_bytes(&[0, 0, 0]),    // pg_nodes
            to_bytes(&[0, 1, 2, 2]), // pg_edge_offsets: 0→{1}, 1→{2}, 2→{}
            to_bytes(&[1, 2]),       // pg_edge_targets
            to_bytes(&[edge_kind::ASSIGNMENT, edge_kind::ASSIGNMENT]),
            to_bytes(&[0, 1, 0]), // pg_node_tags
            to_bytes(&[0b0010]),  // fin = {1}
            to_bytes(&[0b0010]),  // san = {1}
            to_bytes(&[0b0010]),  // fout seed = {1}
        ];
        let values: Vec<vyre_reference::value::Value> = inputs
            .into_iter()
            .map(vyre_reference::value::Value::from)
            .collect();
        let outputs = vyre_reference::reference_eval(&p, &values).unwrap();
        let fout_word = u32::from_le_bytes(outputs[0].to_bytes()[0..4].try_into().unwrap());
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
