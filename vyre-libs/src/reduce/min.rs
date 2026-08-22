//! Unsigned minimum reduction over a u32 ValueSet.

use vyre_foundation::ir::Program;

use super::atomic_scalar::AtomicReduceKind;
use crate::builder::reduction::ReductionComposer;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::reduce::min";

/// Build an atomic grid-stride u32 minimum reduction Program.
#[must_use]
pub fn reduce_min(values: &str, out: &str, count: u32) -> Program {
    ReductionComposer::atomic_scalar_reduction(OP_ID, values, out, count, AtomicReduceKind::Min)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || reduce_min("values", "out", 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[9, 3, 7, 5]),
                to_bytes(&[0]),
            ]]
        }),
        Some(|| vec![vec![vec![0x03, 0x00, 0x00, 0x00]]]),
    )
    .with_laws(AtomicReduceKind::Min.laws())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_matches_reference() {
        assert_eq!(AtomicReduceKind::Min.reference_reduce(&[9, 3, 7, 5]), 3);
    }

    #[test]
    fn empty_returns_identity() {
        assert_eq!(
            AtomicReduceKind::Min.reference_reduce(&[]),
            AtomicReduceKind::Min.identity()
        );
    }

    #[test]
    fn singleton_returns_value() {
        assert_eq!(AtomicReduceKind::Min.reference_reduce(&[3]), 3);
    }

    #[test]
    fn program_uses_parallel_grid_stride() {
        let program = reduce_min("values", "out", 513);
        assert_eq!(
            program.workgroup_size(),
            [crate::reduce::atomic_scalar::WORKGROUP_SIZE, 1, 1]
        );
    }
}
