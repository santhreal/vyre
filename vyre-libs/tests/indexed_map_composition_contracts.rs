//! Composition contracts for operations migrated to the canonical indexed-map skeleton.

use vyre::ir::Node;
use vyre_libs::math::square;
use vyre_libs::nn::activation::parallel_residual_block;

const INDEXED_MAP_OP_ID: &str = "vyre-libs::substrate::indexed_map";

fn count_indexed_map_regions(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Node::Region {
                generator, body, ..
            } => {
                usize::from(generator.as_str() == INDEXED_MAP_OP_ID)
                    + count_indexed_map_regions(body)
            }
            _ => 0,
        })
        .sum()
}

/// Elementwise square must use the shared indexed-map child while retaining its two-buffer ABI.
#[test]
fn square_uses_canonical_indexed_map_region() {
    let program = square("input", "output", 17);

    assert_eq!(program.workgroup_size(), [64, 1, 1]);
    assert_eq!(program.buffers().len(), 2);
    assert_eq!(program.buffers()[0].count(), 17);
    assert_eq!(program.buffers()[1].count(), 17);
    assert_eq!(count_indexed_map_regions(program.entry()), 1);
}

/// Residual addition must use the shared indexed-map child without changing operand order or output role.
#[test]
fn parallel_residual_uses_canonical_indexed_map_region() {
    let program = parallel_residual_block("x", "attn", "mlp", "output", 17)
        .expect("positive residual width must build");

    assert_eq!(program.workgroup_size(), [64, 1, 1]);
    assert_eq!(program.buffers().len(), 4);
    assert_eq!(program.output_buffer_indices(), vec![3]);
    assert_eq!(count_indexed_map_regions(program.entry()), 1);
}

/// Zero-width residual construction must remain a loud validation error after the builder migration.
#[test]
fn parallel_residual_zero_width_remains_rejected() {
    assert_eq!(
        parallel_residual_block("x", "attn", "mlp", "output", 0),
        Err("Fix: n=0".to_string()),
    );
}
