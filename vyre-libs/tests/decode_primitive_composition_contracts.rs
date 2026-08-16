//! Composition contracts for standalone decode programs built from Tier-2.5 primitives.

use vyre::ir::{Node, Program};
use vyre_foundation::visit::walk_nodes;
use vyre_libs::decode::{base64_decode, inflate_stored_block};
use vyre_libs::decode::base64::BASE64_DECODE_OP_ID;
use vyre_libs::decode::inflate::INFLATE_STORED_OP_ID;

fn region_count(program: &Program, expected: &str) -> usize {
    let mut count = 0usize;
    walk_nodes(program, |node| {
        if matches!(node, Node::Region { generator, .. } if generator.as_str() == expected) {
            count += 1;
        }
    });
    count
}

/// Standalone base64 decode must preserve the primitive region and four-buffer ABI.
#[test]
fn base64_decode_delegates_to_canonical_primitive() {
    let program = base64_decode("encoded", "decoded", 8);

    assert_eq!(program.workgroup_size(), [64, 1, 1]);
    assert_eq!(program.buffers().len(), 4);
    assert_eq!(program.output_buffer_indices(), vec![2, 3]);
    assert_eq!(program.buffers()[0].count(), 8);
    assert_eq!(program.buffers()[2].count(), 6);
    assert_eq!(region_count(&program, BASE64_DECODE_OP_ID), 1);
}

/// Standalone stored-block inflate must preserve the primitive region and length sidecar ABI.
#[test]
fn inflate_stored_delegates_to_canonical_primitive() {
    let program = inflate_stored_block("encoded", "decoded", 10);

    assert_eq!(program.workgroup_size(), [64, 1, 1]);
    assert_eq!(program.buffers().len(), 3);
    assert_eq!(program.output_buffer_indices(), vec![1, 2]);
    assert_eq!(program.buffers()[0].count(), 10);
    assert_eq!(program.buffers()[1].count(), 10);
    assert_eq!(program.buffers()[2].count(), 1);
    assert_eq!(region_count(&program, INFLATE_STORED_OP_ID), 1);
}
