//! Fixtures the registry-derived and adversarial gates in this crate share.
//!
//! The dominator-tree oracle helper and the packer, in one home so no suite
//! writes its own.

use vyre_foundation::ir::Program;
use vyre_reference::value::Value;

/// Little-endian u32 packing, the same shipped packer every other suite uses.
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

pub(crate) fn reference_eval_idoms(
    program: &Program,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    pred_offsets: &[u32],
    pred_targets: &[u32],
) -> Vec<u32> {
    let to_bytes = |w: &[u32]| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();

    let values: Vec<Value> = vec![
        Value::from(to_bytes(edge_offsets)),
        Value::from(to_bytes(edge_targets)),
        Value::from(to_bytes(pred_offsets)),
        Value::from(to_bytes(pred_targets)),
        Value::from(to_bytes(&vec![0u32; node_count as usize])),
        Value::from(to_bytes(&vec![0u32; node_count as usize])),
    ];

    let outputs = vyre_reference::reference_eval(program, &values)
        .expect("dominator-tree reference program must evaluate");
    let bytes = outputs[0].to_bytes();
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("u32 output chunk has four bytes")))
        .collect()
}
