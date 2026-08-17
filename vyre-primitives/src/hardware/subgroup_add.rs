//! Cat-C `subgroup_add`  -  per-lane sum reduction broadcast to every lane.
//! Maps to hardware `subgroupAdd()`.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::hardware::{packed_u32_input_with_output, MAP_WORKGROUP};
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
    let body = vec![wrap_anonymous_region(
        OP_ID,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::let_bind("lane_value", Expr::u32(0)),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(values)),
                vec![Node::assign(
                    "lane_value",
                    Expr::load(values, Expr::var("idx")),
                )],
            ),
            Node::let_bind("sum", Expr::subgroup_add(Expr::var("lane_value"))),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![Node::store(out, Expr::var("idx"), Expr::var("sum"))],
            ),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(out, 1, DataType::U32).with_count(n),
        ],
        MAP_WORKGROUP,
        body,
    )
}

fn test_inputs() -> Vec<Vec<Vec<u8>>> {
    packed_u32_input_with_output(&[1, 2, 3, 4])
}

const EXPECTED_SUBGROUP_ADD_OUTPUT_BYTES: [u8; 16] = [
    0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::intrinsic(
        OP_ID,
        crate::hardware::catalog::U32_UNARY_SIGNATURE,
        Some(|| subgroup_add("values", "out", 4)),
        Some(test_inputs),
        Some(|| vec![vec![EXPECTED_SUBGROUP_ADD_OUTPUT_BYTES.to_vec()]]),
    )
    .with_explicit_effects(vyre_foundation::operation::OperationEffects::READ_WRITE_SYNCHRONIZES)
    .with_explicit_capabilities(
        vyre_foundation::program_caps::RequiredCapabilities::NONE.with_subgroup_ops(),
    )
}

inventory::submit! {
    crate::hardware::catalog::IntrinsicFacet {
        operation_id: OP_ID,
        shape: crate::hardware::catalog::OpShape::new(
            1,
            1,
            4,
            crate::hardware::catalog::HardwareSemantic::SubgroupAddU32,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{lcg_u32, run_program};
    use crate::wire::pack_u32_slice as pack_u32;

    fn test_cpu_ref(values: &[u32]) -> Vec<u8> {
        let sim = vyre_reference::subgroup::SubgroupSimulator::default();
        let mut out = Vec::with_capacity(values.len() * 4);
        for chunk in values.chunks(sim.width()) {
            let sum = sim.add(chunk);
            for _ in 0..chunk.len() {
                out.extend_from_slice(&sum.to_le_bytes());
            }
        }
        out
    }

    fn assert_case(values: &[u32]) {
        let n = values.len() as u32;
        let program = subgroup_add("values", "out", n.max(1));
        let outputs = run_program(
            &program,
            vec![pack_u32(values), vec![0u8; (n.max(1) * 4) as usize]],
        );
        assert_eq!(outputs, vec![test_cpu_ref(values)]);
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
            test_cpu_ref(&[1, 2, 3, 4])
        );
    }
}
