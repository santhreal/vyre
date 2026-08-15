//! Cat-C `subgroup_add`  -  per-lane sum reduction broadcast to every lane.
//! Maps to hardware `subgroupAdd()`.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::hardware::{packed_u32_input_with_output, MAP_WORKGROUP};
/// Canonical op id shared by semantics, fixtures, and driver registration.
pub const OP_ID: &str = "vyre-primitives::hardware::subgroup_add";

/// Build a Program whose per-lane output is the sum of all active subgroup
/// lanes.
#[must_use]
pub fn subgroup_add(values: &str, out: &str, n: u32) -> Program {
    let body = vec![crate::hardware::region::wrap_anonymous(
        OP_ID,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![
                    Node::let_bind(
                        "group_base",
                        Expr::mul(Expr::div(Expr::var("idx"), Expr::u32(32)), Expr::u32(32)),
                    ),
                    Node::let_bind("sum", Expr::u32(0)),
                    Node::loop_for(
                        "lane",
                        Expr::u32(0),
                        Expr::u32(32),
                        vec![
                            Node::let_bind(
                                "peer",
                                Expr::add(Expr::var("group_base"), Expr::var("lane")),
                            ),
                            Node::if_then(
                                Expr::lt(Expr::var("peer"), Expr::buf_len(values)),
                                vec![Node::assign(
                                    "sum",
                                    Expr::add(
                                        Expr::var("sum"),
                                        Expr::load(values, Expr::var("peer")),
                                    ),
                                )],
                            ),
                        ],
                    ),
                    Node::store(out, Expr::var("idx"), Expr::var("sum")),
                ],
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

fn cpu_ref(values: &[u32]) -> Vec<u8> {
    const SUBGROUP_WIDTH: usize = 32;
    let mut out = Vec::with_capacity(values.len() * 4);
    for chunk in values.chunks(SUBGROUP_WIDTH) {
        let sum = chunk.iter().copied().fold(0u32, u32::wrapping_add);
        for _ in 0..chunk.len() {
            out.extend_from_slice(&sum.to_le_bytes());
        }
    }
    out
}

fn test_inputs() -> Vec<Vec<Vec<u8>>> {
    packed_u32_input_with_output(&[1, 2, 3, 4])
}

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    let values = vec![1u32, 2, 3, 4];
    vec![vec![cpu_ref(&values)]]
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        id: OP_ID,
        semantic_version: 1,
        signature: Some(crate::hardware::catalog::U32_UNARY_SIGNATURE),
        tier: vyre_foundation::operation::OperationTier::Intrinsic,
        category: Some("hardware"),
        build: Some(|| subgroup_add("values", "out", 4)),
        test_inputs: Some(test_inputs),
        expected_output: Some(expected_output),
        laws: &[],
        tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
    }
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

    fn assert_case(values: &[u32]) {
        let n = values.len() as u32;
        let program = subgroup_add("values", "out", n.max(1));
        let outputs = run_program(
            &program,
            vec![pack_u32(values), vec![0u8; (n.max(1) * 4) as usize]],
        );
        assert_eq!(outputs, vec![cpu_ref(values)]);
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
}
