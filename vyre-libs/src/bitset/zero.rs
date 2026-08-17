//! `bitset_zero` - per-word device clear (`target[w] = 0`).
//!
//! Resident graph pipelines use this to clear scratch/output bitsets on device
//! instead of uploading zero-filled host buffers every iteration.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::bitset::zero";

/// Build a Program: `target[w] = 0` for `w` in `0..words`.
#[must_use]
pub fn bitset_zero(target: &str, words: u32) -> Program {
    ElementwiseComposer::new(OP_ID, words)
        .with_workgroup_size([256, 1, 1])
        .add_output_storage(target, BufferAccess::WriteOnly, DataType::U32, words)
        .build_pointwise(target, |_| Expr::u32(0))
}


inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || bitset_zero("target", 3),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[1, 0xDEAD_BEEF, u32::MAX])]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0, 0, 0])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_ref(target: &mut [u32]) {
        target.fill(0);
    }

    #[test]
    fn cpu_ref_clears_all_words() {
        let mut words = vec![1u32, 0xDEAD_BEEF, u32::MAX];
        cpu_ref(&mut words);
        assert_eq!(words, vec![0, 0, 0]);
    }

    #[test]
    fn emitted_program_has_one_rw_target_buffer() {
        let program = bitset_zero("target", 17);
        assert_eq!(program.workgroup_size, [256, 1, 1]);
        assert_eq!(program.buffers.len(), 1);
        assert_eq!(program.buffers[0].count, 17);
    }
}
