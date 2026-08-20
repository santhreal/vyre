//! `resolve_family`  -  `node_tags[v] & family_mask != 0` → NodeSet bit v.
//!
//! One invocation per node. Reads the per-node tag bitmap, ANDs it
//! against the compile-time family mask, atomically-ORs the result
//! bit into `nodeset_out[v / 32]`.

use vyre_foundation::ir::Program;

use crate::label::nodeset_filter::{nodeset_filter_program, NodeSetFilter};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::label::resolve_family";

/// Build a Program: for each node `v`, if
/// `node_tags[v] & family_mask != 0`, set bit `v` in `nodeset_out`.
#[must_use]
pub fn resolve_family(
    node_tags: &str,
    nodeset_out: &str,
    node_count: u32,
    family_mask: u32,
) -> Program {
    nodeset_filter_program(
        OP_ID,
        node_tags,
        nodeset_out,
        node_count,
        NodeSetFilter::Intersects(family_mask),
    )
}

const EXPECTED_RESOLVE_FAMILY_OUTPUT_BYTES: [u8; 4] = [6, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || resolve_family("tags", "nodeset", 4, 0b0010),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            // node_tags: 0x01, 0x02, 0x06, 0x04  -  family mask 0x02
            // hits nodes 1 and 2 (0x02 and 0x06 both have bit 1).
            vec![vec![to_bytes(&[0x01, 0x02, 0x06, 0x04]), to_bytes(&[0])]]
        }),
        Some(|| {
            vec![vec![EXPECTED_RESOLVE_FAMILY_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["idempotent"])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_resolve_family(node_tags: &[u32], family_mask: u32) -> Vec<u32> {
        let mut out = vec![0_u32; node_tags.len().div_ceil(32)];
        reference_resolve_family_into(node_tags, family_mask, &mut out);
        out
    }

    fn reference_resolve_family_into(node_tags: &[u32], family_mask: u32, out: &mut Vec<u32>) {
        let words = node_tags.len().div_ceil(32);
        out.clear();
        out.resize(words, 0);
        for (node, &tag) in node_tags.iter().enumerate() {
            if (tag & family_mask) != 0 {
                out[node / 32] |= 1_u32 << (node % 32);
            }
        }
    }

    #[test]
    fn single_family_bit() {
        assert_eq!(
            reference_resolve_family(&[0x01, 0x02, 0x06, 0x04], 0x02),
            vec![0b0110]
        );
    }

    #[test]
    fn empty_family_yields_empty_nodeset() {
        assert_eq!(reference_resolve_family(&[0x01, 0x02], 0x00), vec![0]);
    }

    #[test]
    fn reference_into_reuses_nodeset_buffer() {
        let mut out = Vec::with_capacity(4);
        let ptr = out.as_ptr();
        reference_resolve_family_into(&[0x01, 0x02, 0x06, 0x04], 0x02, &mut out);
        assert_eq!(out, vec![0b0110]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn registration_fixture_matches_exact_byte_constant() {
        assert_eq!(EXPECTED_RESOLVE_FAMILY_OUTPUT_BYTES, [6, 0, 0, 0]);
        let cpu_ref = reference_resolve_family(&[0x01, 0x02, 0x06, 0x04], 0x02);
        assert_eq!(cpu_ref, vec![0b0110]);
    }

    /// Identity travels in the region generator, not in `entry_op_id`.
    /// `Program::wrapped` clears that field on every program it builds, and
    /// only the handful of builders that call `with_entry_op_id` carry one, so
    /// the assertion this replaces could not pass for any program this builder
    /// produces. What a composition audit reads is the region tree: this op's
    /// own generator wrapping the shared `nodeset_filter` skeleton, attributed
    /// back to this op. A future rewrite that inlines the skeleton instead of
    /// composing it drops the child and turns this red.
    #[test]
    fn resolve_family_wraps_the_shared_nodeset_filter_under_its_own_id() {
        use vyre_foundation::ir::{Ident, Node};

        fn regions(nodes: &[Node], out: &mut Vec<(String, Option<Ident>)>) {
            for node in nodes {
                if let Node::Region {
                    generator,
                    source_region,
                    body,
                } = node
                {
                    out.push((generator.to_string(), source_region.clone()));
                    regions(body, out);
                }
            }
        }

        let program = resolve_family("tags", "out", 4, 0x02);
        let mut found = Vec::new();
        regions(&program.entry, &mut found);

        assert!(
            found.iter().any(|(generator, source)| generator
                == crate::label::nodeset_filter::OP_ID
                && source.as_deref() == Some(OP_ID)),
            "the shared nodeset_filter skeleton must appear as a child region attributed to \
             {OP_ID}, so a composition audit sees the reuse: {found:?}"
        );
        assert!(
            found.iter().any(|(generator, _)| generator == OP_ID),
            "the program must name its own op as a region generator: {found:?}"
        );
    }
}
