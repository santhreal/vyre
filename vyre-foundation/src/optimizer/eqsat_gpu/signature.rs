//! The row signature the device kernel and the host packer must agree on.
//!
//! A refreshed row and a freshly packed row have to hash identically or the
//! device cannot tell which rows changed. One function owns the mix so the two
//! producers cannot drift.

use super::snapshot::SnapshotRow;

/// Structural row signature for packed GPU e-graph columns.
///
/// Matches the device-side row-signature refresh kernel and the initial image
/// packing, so a refreshed row and a freshly packed row hash identically.
#[must_use]
pub fn gpu_egraph_row_signature(language_op_id: u32, children_len: u32, children: &[u32]) -> u32 {
    let mut hash = mix_egraph_signature(0xA24B_AED4, language_op_id);
    hash = mix_egraph_signature(hash, children_len);
    for &child in children {
        hash = mix_egraph_signature(hash, child);
    }
    hash
}

pub(super) fn egraph_row_signature(row: &SnapshotRow, children: &[u32]) -> u32 {
    gpu_egraph_row_signature(row.language_op_id, row.children_len, children)
}

fn mix_egraph_signature(hash: u32, value: u32) -> u32 {
    let mixed = hash
        ^ value
            .wrapping_add(0x9E37_79B9)
            .wrapping_add(hash << 6)
            .wrapping_add(hash >> 2);
    mixed.rotate_left(13).wrapping_mul(0x85EB_CA6B)
}
