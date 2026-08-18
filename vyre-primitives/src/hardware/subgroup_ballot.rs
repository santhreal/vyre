//! Cat-C `subgroup_ballot`  -  popcount of per-lane bool into u32 bitmask.
//!
//! Maps to the target-native subgroup ballot intrinsic via a concrete
//! driver emitter arm.

use vyre_foundation::ir::{Expr, Program};

use crate::hardware::packed_u32_input_with_output;
/// Canonical op id shared by semantics, fixtures, and driver registration.
pub const OP_ID: &str = "vyre-primitives::hardware::subgroup_ballot";

/// Build a Program that collects the per-lane boolean predicate into a u32
/// bitmask broadcast to every lane.
///
/// Every lane of the subgroup contributes one predicate bit, so the ballot's operand
/// is evaluated at EVERY lane index, not only at the indices a store guard admits.
/// Writing `load(cond_input, idx)` as the operand therefore reads `cond_input` at the
/// index of every lane in the subgroup, past the end of the buffer for a dispatch
/// wider than the buffer, no matter what branch the collective sits in. The operand is
/// a control-flow-guarded per-lane local instead, and the collective itself sits in
/// uniform control flow, where a subgroup collective's participating-lane set is
/// well defined rather than dependent on the active mask of a divergent branch.
#[must_use]
pub fn subgroup_ballot(cond_input: &str, out: &str, n: u32) -> Program {
    crate::hardware::subgroup_unary_u32_program(OP_ID, cond_input, out, n, |pred| Expr::SubgroupBallot {
        cond: Box::new(Expr::eq(pred, Expr::u32(1))),
    })
}

fn test_inputs() -> Vec<Vec<Vec<u8>>> {
    packed_u32_input_with_output(&[0, 1, 0, 1])
}

const EXPECTED_SUBGROUP_BALLOT_OUTPUT_BYTES: [u8; 16] = [
    0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00,
];

submit_hardware_intrinsic! {
    id: OP_ID,
    signature: crate::hardware::catalog::U32_UNARY_SIGNATURE,
    builder: || subgroup_ballot("cond", "out", 4),
    inputs: test_inputs,
    expected: || vec![vec![EXPECTED_SUBGROUP_BALLOT_OUTPUT_BYTES.to_vec()]],
    effects: vyre_foundation::operation::OperationEffects::READ_WRITE_SYNCHRONIZES,
    capabilities: vyre_foundation::program_caps::RequiredCapabilities::NONE.with_subgroup_ops(),
    inputs_count: 1,
    outputs_count: 1,
    semantic: crate::hardware::catalog::HardwareSemantic::SubgroupBallotU32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::assert_unary_u32_case;

    fn test_cpu_ref(cond: &[u32]) -> Vec<u32> {
        let sim = vyre_reference::subgroup::SubgroupSimulator::default();
        let n = cond.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (subgroup_start, subgroup_end) = sim.subgroup_bounds(n, i);
            let mask = cond[subgroup_start..subgroup_end]
                .iter()
                .map(|&c| c == 1)
                .collect::<Vec<_>>();
            out.push(sim.ballot_slice(&mask));
        }
        out
    }

    fn assert_case(cond: &[u32]) {
        let expected = test_cpu_ref(cond);
        assert_unary_u32_case(subgroup_ballot, cond, &expected);
    }

    #[test]
    fn one_element_true() {
        assert_case(&[1]);
    }
    #[test]
    fn one_element_false() {
        assert_case(&[0]);
    }
    #[test]
    fn mixed() {
        assert_case(&[0, 1, 0, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn registration_fixture_matches_exact_byte_constant() {
        assert_eq!(
            EXPECTED_SUBGROUP_BALLOT_OUTPUT_BYTES.to_vec(),
            crate::wire::pack_u32_slice(&test_cpu_ref(&[0, 1, 0, 1]))
        );
    }
}
