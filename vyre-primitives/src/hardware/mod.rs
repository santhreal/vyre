//! Category C hardware intrinsics.
//!
//! Every op here needs a dedicated backend emitter arm and a dedicated
//! reference-interpreter arm; none of them can be written as a composition of
//! existing `Expr`/`Node` variants. A backend that cannot lower one reports
//! `UnsupportedByBackend` instead of falling back to a slow CPU path.
//!
//! Surface: `subgroup_add` / `subgroup_ballot` / `subgroup_shuffle` (wave-level
//! collectives), `workgroup_barrier` / `storage_barrier` (memory fences),
//! `bit_reverse_u32` / `popcount_u32` (bit instructions), `fma_f32`
//! (single-round fused multiply-add), `inverse_sqrt_f32`.
//!
//! An op that composes over existing IR belongs in `vyre-libs`, not here. An
//! op is admitted here only when it needs its own emitter arm and its own
//! reference-interpreter arm.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{OperationRegistry, SemanticOperation};

/// Coarse category every hardware intrinsic registration carries.
pub const HARDWARE_CATEGORY: &str = "hardware";

macro_rules! define_unary_u32_hardware_intrinsic {
    (
        $function:ident,
        $op_id:literal,
        $expr:path,
        $cpu_map:expr,
        $fixture:expr,
        $seed:expr,
        $one_case:expr,
        $max_case:expr
    ) => {
        use vyre_foundation::ir::Program;

        /// Canonical op id.
        pub const OP_ID: &str = $op_id;

        /// Build the canonical u32 unary hardware intrinsic program.
        #[must_use]
        pub fn $function(input: &str, out: &str, n: u32) -> Program {
            crate::hardware::unary_u32_program(OP_ID, input, out, n, $expr)
        }

        fn cpu_ref(input: &[u32]) -> Vec<u8> {
            let map_lane = $cpu_map;
            let output: Vec<u32> = input.iter().copied().map(map_lane).collect();
            crate::hardware::pack_u32(&output)
        }

        fn fixture_input() -> Vec<u32> {
            $fixture.to_vec()
        }

        fn test_inputs() -> Vec<Vec<Vec<u8>>> {
            let input = fixture_input();
            let len = input.len() * 4;
            vec![vec![crate::hardware::pack_u32(&input), vec![0u8; len]]]
        }

        fn expected_output() -> Vec<Vec<Vec<u8>>> {
            let input = fixture_input();
            vec![vec![cpu_ref(&input)]]
        }

        inventory::submit! {
            vyre_foundation::operation::OperationRegistration {
                id: OP_ID,
                semantic_version: 1,
                signature: Some(crate::hardware::catalog::U32_UNARY_SIGNATURE),
                tier: vyre_foundation::operation::OperationTier::Intrinsic,
                category: Some("hardware"),
                build: Some(|| $function("input", "out", 4)),
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
                    crate::hardware::catalog::HardwareSemantic::UnaryU32Map,
                ),
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::hardware::{lcg_u32, pack_u32, run_program};

            fn assert_case(input: &[u32]) {
                let n = input.len() as u32;
                let program = $function("input", "out", n.max(1));
                let outputs = run_program(
                    &program,
                    vec![pack_u32(input), vec![0u8; (n.max(1) * 4) as usize]],
                );
                assert_eq!(outputs, vec![cpu_ref(input)]);
            }

            #[test]
            fn one_element() {
                assert_case($one_case);
            }

            #[test]
            fn max_value() {
                assert_case($max_case);
            }

            #[test]
            fn random_sixty_four() {
                let input = lcg_u32($seed, 64);
                assert_case(&input);
            }
        }
    };
}

macro_rules! define_barrier_u32_hardware_intrinsic {
    (
        $function:ident,
        $op_id:literal,
        $fixture:expr,
        $seed:expr,
        $one_case:expr
    ) => {
        use vyre_foundation::ir::Program;

        /// Canonical op id.
        pub const OP_ID: &str = $op_id;

        /// Build a Program that emits this memory barrier after an identity u32 store.
        #[must_use]
        pub fn $function(input: &str, out: &str, n: u32) -> Program {
            crate::hardware::barrier_identity_u32_program(OP_ID, input, out, n)
        }

        fn cpu_ref(input: &[u32]) -> Vec<u8> {
            crate::hardware::pack_u32(input)
        }

        fn fixture_input() -> Vec<u32> {
            $fixture.to_vec()
        }

        fn test_inputs() -> Vec<Vec<Vec<u8>>> {
            let input = fixture_input();
            let len = input.len() * 4;
            vec![vec![crate::hardware::pack_u32(&input), vec![0u8; len]]]
        }

        fn expected_output() -> Vec<Vec<Vec<u8>>> {
            let input = fixture_input();
            vec![vec![cpu_ref(&input)]]
        }

        inventory::submit! {
            vyre_foundation::operation::OperationRegistration {
                id: OP_ID,
                semantic_version: 1,
                signature: Some(crate::hardware::catalog::U32_UNARY_SIGNATURE),
                tier: vyre_foundation::operation::OperationTier::Intrinsic,
                category: Some("hardware"),
                build: Some(|| $function("input", "out", 4)),
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
                    crate::hardware::catalog::HardwareSemantic::BarrierIdentityU32,
                ),
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::hardware::{lcg_u32, pack_u32, run_program};

            fn assert_case(input: &[u32]) {
                let n = input.len() as u32;
                let program = $function("input", "out", n.max(1));
                let outputs = run_program(
                    &program,
                    vec![pack_u32(input), vec![0u8; (n.max(1) * 4) as usize]],
                );
                assert_eq!(outputs, vec![cpu_ref(input)]);
            }

            #[test]
            fn one_element() {
                assert_case($one_case);
            }

            #[test]
            fn random_sixty_four() {
                let input = lcg_u32($seed, 64);
                assert_case(&input);
            }
        }
    };
}

/// Canonical Category C catalog: signatures, conformance geometry, registry view.
pub mod catalog;
/// Region-chain wrap helper every Category C builder applies to its body.
pub mod region;

/// `bit_reverse_u32`  -  reverses every bit in each u32 lane via hardware `reverseBits`.
pub mod bit_reverse_u32;
/// `fma_f32`  -  IEEE-754 fused multiply-add (byte-identical to `f32::mul_add`).
pub mod fma_f32;
/// `inverse_sqrt_f32`  -  hardware `inverseSqrt()` approximation.
pub mod inverse_sqrt_f32;
/// `popcount_u32`  -  hardware `countOneBits` on each u32 lane.
pub mod popcount_u32;
/// `storage_barrier`  -  cross-workgroup storage-buffer memory fence.
pub mod storage_barrier;
/// `subgroup_add`  -  wave-level reduction over the subgroup.
pub mod subgroup_add;
/// `subgroup_ballot`  -  wave-level predicate ballot bitmask.
pub mod subgroup_ballot;
/// `subgroup_shuffle`  -  wave-level lane-to-lane value shuffle.
pub mod subgroup_shuffle;
/// `workgroup_barrier`  -  intra-workgroup shared-memory fence.
pub mod workgroup_barrier;

/// Iterate every registered Category C hardware intrinsic.
///
/// Selects on the registration category rather than the id prefix, so the
/// namespace contract (`vyre-primitives::hardware::*`) stays an assertable
/// property instead of the filter that makes it true.
pub fn all_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.category == Some(HARDWARE_CATEGORY))
}

pub(crate) const MAP_WORKGROUP: [u32; 3] = [64, 1, 1];

pub(crate) fn unary_u32_program<F>(
    op_id: &'static str,
    input: &str,
    out: &str,
    n: u32,
    expr: F,
) -> Program
where
    F: Fn(Expr) -> Expr,
{
    let body = vec![crate::hardware::region::wrap_anonymous(
        op_id,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![Node::store(
                    out,
                    Expr::var("idx"),
                    expr(Expr::load(input, Expr::var("idx"))),
                )],
            ),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(out, 1, DataType::U32).with_count(n),
        ],
        MAP_WORKGROUP,
        body,
    )
}

pub(crate) fn barrier_identity_u32_program(
    op_id: &'static str,
    input: &str,
    out: &str,
    n: u32,
) -> Program {
    let body = vec![crate::hardware::region::wrap_anonymous(
        op_id,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![Node::store(
                    out,
                    Expr::var("idx"),
                    Expr::load(input, Expr::var("idx")),
                )],
            ),
            Node::barrier(),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(out, 1, DataType::U32).with_count(n),
        ],
        MAP_WORKGROUP,
        body,
    )
}

pub(crate) fn ternary_f32_program(
    op_id: &'static str,
    a: &str,
    b: &str,
    c: &str,
    out: &str,
    n: u32,
) -> Program {
    let body = vec![crate::hardware::region::wrap_anonymous(
        op_id,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![Node::store(
                    out,
                    Expr::var("idx"),
                    Expr::Fma {
                        a: Box::new(Expr::load(a, Expr::var("idx"))),
                        b: Box::new(Expr::load(b, Expr::var("idx"))),
                        c: Box::new(Expr::load(c, Expr::var("idx"))),
                    },
                )],
            ),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::storage(c, 2, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::output(out, 3, DataType::F32).with_count(n),
        ],
        MAP_WORKGROUP,
        body,
    )
}

pub(crate) fn pack_u32(words: &[u32]) -> Vec<u8> {
    crate::wire::pack_u32_slice(words)
}

pub(crate) fn packed_u32_input_with_output(words: &[u32]) -> Vec<Vec<Vec<u8>>> {
    vec![vec![pack_u32(words), vec![0u8; words.len() * 4]]]
}

pub(crate) fn pack_f32(values: &[f32]) -> Vec<u8> {
    crate::wire::pack_f32_slice(values)
}

#[cfg(test)]
pub(crate) fn run_program(program: &Program, inputs: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    use vyre_reference::value::Value;
    let values: Vec<Value> = inputs.into_iter().map(|b| Value::Bytes(b.into())).collect();
    vyre_reference::reference_eval(program, &values)
        .expect("Fix: intrinsic must execute; restore this invariant before continuing.")
        .into_iter()
        .map(|v| v.to_bytes())
        .collect()
}

#[cfg(test)]
pub(crate) fn lcg_u32(seed: u32, len: usize) -> Vec<u32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn lcg_f32(seed: u32, len: usize) -> Vec<f32> {
    lcg_u32(seed, len)
        .into_iter()
        .map(|w| f32::from_bits((w >> 9) | 0x3F00_0000) - 1.0)
        .collect()
}
