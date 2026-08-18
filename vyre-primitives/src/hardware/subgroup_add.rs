//! Cat-C `subgroup_add`  -  per-lane sum reduction broadcast to every lane.
//! Maps to hardware `subgroupAdd()`.

use vyre_foundation::ir::{Expr, Program};

use crate::hardware::packed_u32_input_with_output;
/// Canonical op id shared by semantics, fixtures, and driver registration.
pub const OP_ID: &str = "vyre-primitives::hardware::subgroup_add";

/// Build a Program whose per-lane output is the sum of all active subgroup
/// lanes.
///
/// The reduction is `Expr::subgroup_add`, which every backend emitter lowers to
/// its own subgroup add: this op is admitted to this crate because it needs that
/// arm and a reference-interpreter arm, and it earns that admission only by
/// using them. It previously summed thirty-two memory neighbours in a serial
/// loop, which made every lane redo its subgroup's whole sum out of storage, and
/// registered `HardwareSemantic::SubgroupAddU32` for work no subgroup
/// instruction performed.
///
/// The lane value is a control-flow-guarded local and the collective sits in
/// uniform control flow, for the reason `subgroup_shuffle` states: a collective
/// under a divergent branch has no well defined participating-lane set, and an
/// operand read at every lane index reads past the end of a buffer narrower than
/// the dispatch. A lane outside `values` contributes zero, so the sum is over
/// the in-range lanes of that subgroup.
#[must_use]
pub fn subgroup_add(values: &str, out: &str, n: u32) -> Program {
    crate::hardware::subgroup_unary_u32_program(OP_ID, values, out, n, Expr::subgroup_add)
}

fn test_inputs() -> Vec<Vec<Vec<u8>>> {
    packed_u32_input_with_output(&[1, 2, 3, 4])
}

const EXPECTED_SUBGROUP_ADD_OUTPUT_BYTES: [u8; 16] = [
    0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00,
];

submit_hardware_intrinsic! {
    id: OP_ID,
    signature: crate::hardware::catalog::U32_UNARY_SIGNATURE,
    builder: || subgroup_add("values", "out", 4),
    inputs: test_inputs,
    expected: || vec![vec![EXPECTED_SUBGROUP_ADD_OUTPUT_BYTES.to_vec()]],
    effects: vyre_foundation::operation::OperationEffects::READ_WRITE_SYNCHRONIZES,
    capabilities: vyre_foundation::program_caps::RequiredCapabilities::NONE.with_subgroup_ops(),
    inputs_count: 1,
    outputs_count: 1,
    semantic: crate::hardware::catalog::HardwareSemantic::SubgroupAddU32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{assert_unary_u32_case, lcg_u32};

    fn test_cpu_ref(values: &[u32]) -> Vec<u32> {
        let sim = vyre_reference::subgroup::SubgroupSimulator::default();
        let mut out = Vec::with_capacity(values.len());
        for chunk in values.chunks(sim.width()) {
            let sum = sim.add(chunk);
            for _ in 0..chunk.len() {
                out.push(sum);
            }
        }
        out
    }

    fn assert_case(values: &[u32]) {
        let expected = test_cpu_ref(values);
        assert_unary_u32_case(subgroup_add, values, &expected);
    }

    /// This boundary test proves a one-lane subgroup broadcasts its only value unchanged.
    #[test]
    fn one_element() {
        assert_case(&[42]);
    }
    /// This overflow-edge test preserves wrapping subgroup addition for the maximum u32 value.
    #[test]
    fn max_value() {
        assert_case(&[u32::MAX]);
    }
    /// This deterministic multi-subgroup test locks out cross-subgroup accumulation.
    #[test]
    fn random_sixty_four() {
        assert_case(&lcg_u32(0xC100_0033, 64));
    }

    #[test]
    fn registration_fixture_matches_exact_byte_constant() {
        assert_eq!(
            EXPECTED_SUBGROUP_ADD_OUTPUT_BYTES.to_vec(),
            crate::wire::pack_u32_slice(&test_cpu_ref(&[1, 2, 3, 4]))
        );
    }
}
