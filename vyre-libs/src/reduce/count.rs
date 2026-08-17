//! `reduce_count`  -  population count over a packed bitset, written
//! as a single u32 into `out[0]`.

use vyre_foundation::ir::Program;
use crate::builder::reduction::ReductionComposer;
use super::atomic_scalar::AtomicReduceKind;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::reduce::count";

/// Build a Program: `out[0] = sum_{w} popcount(bitset[w])`.
#[must_use]
pub fn reduce_count(bitset: &str, out: &str, words: u32) -> Program {
    ReductionComposer::atomic_scalar_reduction(OP_ID, bitset, out, words, AtomicReduceKind::PopcountSum)
}

/// CPU reference.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(bitset: &[u32]) -> u32 {
    bitset.iter().map(|w| w.count_ones()).sum()
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || reduce_count("bitset", "out", 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0b1111, 0xFFFF_FFFF]), to_bytes(&[0])]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[36])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_bit_count() {
        assert_eq!(cpu_ref(&[0b1111, 0xFFFF_FFFF]), 36);
    }

    #[test]
    fn program_uses_parallel_grid_stride() {
        let program = reduce_count("bitset", "out", 513);
        assert_eq!(
            program.workgroup_size(),
            [crate::reduce::atomic_scalar::WORKGROUP_SIZE, 1, 1]
        );
    }
}
