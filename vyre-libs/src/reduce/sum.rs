//! `reduce_sum`  -  wrapping unsigned sum over a u32 ValueSet.

use vyre_foundation::ir::Program;

use super::atomic_scalar::AtomicReduceKind;
use crate::builder::reduction::ReductionComposer;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::reduce::sum";

/// Build a Program: `out[0] = (Σ values_i) mod 2^32`.
#[must_use]
pub fn reduce_sum(values: &str, out: &str, count: u32) -> Program {
    ReductionComposer::atomic_scalar_reduction(OP_ID, values, out, count, AtomicReduceKind::Sum)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || reduce_sum("values", "out", 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[1, 2, 3, 4]), to_bytes(&[0])]]
        }),
        Some(|| vec![vec![vec![0x0a, 0x00, 0x00, 0x00]]]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_sum(values: &[u32]) -> u32 {
        values.iter().copied().fold(0u32, |a, b| a.wrapping_add(b))
    }

    #[test]
    fn sums_values() {
        assert_eq!(reference_sum(&[1, 2, 3, 4]), 10);
    }

    #[test]
    fn wraps_on_overflow() {
        assert_eq!(reference_sum(&[u32::MAX, 1]), 0);
    }
    #[test]
    fn program_uses_parallel_grid_stride() {
        let program = reduce_sum("values", "out", 513);
        assert_eq!(
            program.workgroup_size(),
            [crate::reduce::atomic_scalar::WORKGROUP_SIZE, 1, 1]
        );
        assert!(
            !format!("{:?}", program.entry()).contains("grid_size_pending"),
            "reduce_sum program must not carry unresolved grid-size markers"
        );
    }
}
