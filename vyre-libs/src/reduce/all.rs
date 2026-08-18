//! Boolean all-nonzero reduction over a u32 ValueSet.

use vyre_foundation::ir::Program;

use super::atomic_scalar::AtomicReduceKind;
use crate::builder::reduction::ReductionComposer;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::reduce::all";

/// Build a Program: `out[0] = (forall i: values[i] != 0) ? 1 : 0`.
#[must_use]
pub fn reduce_all(values: &str, out: &str, count: u32) -> Program {
    ReductionComposer::atomic_scalar_reduction(
        OP_ID,
        values,
        out,
        count,
        AtomicReduceKind::AllNonZero,
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || reduce_all("values", "out", 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[1, 0, 1, 1]),
                to_bytes(&[0]),
            ]]
        }),
        Some(|| vec![vec![vec![0x00, 0x00, 0x00, 0x00]]]),
    )
    .with_laws(AtomicReduceKind::AllNonZero.laws())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn true_case_reduces_to_one() {
        assert_eq!(AtomicReduceKind::AllNonZero.reference_reduce(&[1, 7, 9]), 1);
    }

    #[test]
    fn false_case_reduces_to_zero() {
        assert_eq!(AtomicReduceKind::AllNonZero.reference_reduce(&[1, 0, 9]), 0);
    }

    #[test]
    fn empty_values_reduce_to_zero() {
        assert_eq!(AtomicReduceKind::AllNonZero.reference_reduce(&[]), 0);
    }

    #[test]
    fn program_uses_parallel_grid_stride() {
        let program = reduce_all("values", "out", 513);
        assert_eq!(
            program.workgroup_size(),
            [crate::reduce::atomic_scalar::WORKGROUP_SIZE, 1, 1]
        );
    }
}
