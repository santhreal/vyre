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

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{OperationRegistry, SemanticOperation};

/// Coarse category every hardware intrinsic registration carries.
pub const HARDWARE_CATEGORY: &str = "hardware";

/// Submit a hardware intrinsic operation to the inventory registry.
#[macro_export]
macro_rules! submit_intrinsic_operation {
    (
        id: $op_id:expr,
        signature: $sig:expr,
        build: $build:expr,
        inputs: $inputs:expr,
        expected: $expected:expr
    ) => {
        inventory::submit! {
            vyre_foundation::operation::OperationRegistration {
                id: $op_id,
                semantic_version: 1,
                signature: $sig,
                tier: vyre_foundation::operation::OperationTier::Intrinsic,
                category: Some("hardware"),
                build: Some($build),
                test_inputs: Some($inputs),
                expected_output: Some($expected),
                laws: &[],
                numeric: vyre_foundation::numeric::NumericContract::EXACT,
                geometry_requirements: vyre_foundation::GeometryRequirements::agnostic(),
            }
        }
    };
}

macro_rules! submit_hardware_intrinsic {
    (
        id: $op_id:expr,
        signature: $sig:expr,
        builder: $builder:expr,
        inputs: $inputs:expr,
        expected: $expected:expr,
        effects: $effects:expr,
        capabilities: $caps:expr,
        inputs_count: $in_cnt:expr,
        outputs_count: $out_cnt:expr,
        semantic: $semantic:expr
    ) => {
        inventory::submit! {
            vyre_foundation::operation::OperationRegistration::intrinsic_unconstrained(
                $op_id,
                $sig,
                Some($builder),
                Some($inputs),
                Some($expected),
            )
            .with_geometry_requirements(
                vyre_foundation::GeometryRequirements::agnostic().with_element_policy(
                    vyre_foundation::ElementPolicy::Scalar,
                ),
            )
            .with_explicit_effects($effects)
            .with_explicit_capabilities($caps)
        }

        inventory::submit! {
            crate::hardware::catalog::IntrinsicFacet {
                operation_id: $op_id,
                shape: crate::hardware::catalog::OpShape::new(
                    $in_cnt,
                    $out_cnt,
                    4,
                    $semantic,
                ),
            }
        }
    };
}

macro_rules! define_unary_u32_hardware_intrinsic {
    (
        $function:ident,
        $op_id:literal,
        $expr:path,
        $cpu_map:expr,
        $fixture:expr,
        $expected:expr,
        $expected_bytes:expr,
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

        fn test_inputs() -> Vec<Vec<Vec<u8>>> {
            crate::hardware::packed_u32_input($fixture)
        }

        const EXPECTED_REGISTRATION_BYTES: &[u8] = $expected_bytes;

        submit_hardware_intrinsic! {
            id: OP_ID,
            signature: crate::hardware::catalog::U32_UNARY_SIGNATURE,
            builder: || $function("input", "out", 4),
            inputs: test_inputs,
            expected: || vec![vec![EXPECTED_REGISTRATION_BYTES.to_vec()]],
            effects: vyre_foundation::operation::OperationEffects::READ_WRITE,
            capabilities: vyre_foundation::program_caps::RequiredCapabilities::NONE,
            inputs_count: 1,
            outputs_count: 1,
            semantic: crate::hardware::catalog::HardwareSemantic::UnaryU32Map
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::hardware::{assert_unary_u32_case, lcg_u32};

            fn test_cpu_ref(input: &[u32]) -> Vec<u32> {
                let map_lane = $cpu_map;
                input.iter().copied().map(map_lane).collect()
            }

            fn assert_case(input: &[u32]) {
                let expected = test_cpu_ref(input);
                assert_unary_u32_case($function, input, &expected);
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

            #[test]
            fn registration_fixture_matches_exact_byte_constant() {
                assert_eq!(
                    EXPECTED_REGISTRATION_BYTES,
                    crate::wire::pack_u32_slice(&test_cpu_ref($fixture)).as_slice()
                );
            }
        }
    };
}

macro_rules! define_barrier_u32_hardware_intrinsic {
    (
        $function:ident,
        $op_id:literal,
        $fixture:expr,
        $fixture_bytes:expr,
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

        fn test_inputs() -> Vec<Vec<Vec<u8>>> {
            crate::hardware::packed_u32_input($fixture)
        }

        const EXPECTED_REGISTRATION_BYTES: &[u8] = $fixture_bytes;

        submit_hardware_intrinsic! {
            id: OP_ID,
            signature: crate::hardware::catalog::U32_UNARY_SIGNATURE,
            builder: || $function("input", "out", 4),
            inputs: test_inputs,
            expected: || vec![vec![EXPECTED_REGISTRATION_BYTES.to_vec()]],
            effects: vyre_foundation::operation::OperationEffects::READ_WRITE_SYNCHRONIZES,
            capabilities: vyre_foundation::program_caps::RequiredCapabilities::NONE,
            inputs_count: 1,
            outputs_count: 1,
            semantic: crate::hardware::catalog::HardwareSemantic::BarrierIdentityU32
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::hardware::{assert_unary_u32_case, lcg_u32};

            fn assert_case(input: &[u32]) {
                assert_unary_u32_case($function, input, input);
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

            #[test]
            fn registration_fixture_matches_exact_byte_constant() {
                assert_eq!(
                    EXPECTED_REGISTRATION_BYTES,
                    crate::wire::pack_u32_slice($fixture).as_slice()
                );
            }
        }
    };
}

/// Region-chain wrapping helper every Category C builder applies to its body.

/// `bit_reverse_u32`  -  reverses every bit in each u32 lane via hardware `reverseBits`.
pub mod bit_reverse_u32;
/// Canonical Category C catalog: signatures, conformance geometry, registry view.
pub mod catalog;
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

pub(crate) fn unary_program(
    op_id: &'static str,
    input: &str,
    out: &str,
    n: u32,
    ty: DataType,
    body: Vec<Node>,
) -> Program {
    let wrapped_body = vec![wrap_anonymous_region(op_id, body)];
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, ty.clone()).with_count(n),
            BufferDecl::output(out, 1, ty).with_count(n),
        ],
        MAP_WORKGROUP,
        wrapped_body,
    )
}

pub(crate) fn binary_u32_program(
    op_id: &'static str,
    a: &str,
    b: &str,
    out: &str,
    n: u32,
    body: Vec<Node>,
) -> Program {
    let wrapped_body = vec![wrap_anonymous_region(op_id, body)];
    Program::wrapped(
        vec![
            BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(out, 2, DataType::U32).with_count(n),
        ],
        MAP_WORKGROUP,
        wrapped_body,
    )
}

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
    unary_program(
        op_id,
        input,
        out,
        n,
        DataType::U32,
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
    )
}

pub(crate) fn barrier_identity_u32_program(
    op_id: &'static str,
    input: &str,
    out: &str,
    n: u32,
) -> Program {
    unary_program(
        op_id,
        input,
        out,
        n,
        DataType::U32,
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
    )
}

pub(crate) fn subgroup_unary_u32_program<F>(
    op_id: &'static str,
    input: &str,
    out: &str,
    n: u32,
    collective: F,
) -> Program
where
    F: FnOnce(Expr) -> Expr,
{
    unary_program(
        op_id,
        input,
        out,
        n,
        DataType::U32,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::let_bind("lane_value", Expr::u32(0)),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(input)),
                vec![Node::assign(
                    "lane_value",
                    Expr::load(input, Expr::var("idx")),
                )],
            ),
            Node::let_bind("result", collective(Expr::var("lane_value"))),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![Node::store(out, Expr::var("idx"), Expr::var("result"))],
            ),
        ],
    )
}

#[cfg(test)]
pub(crate) fn inverse_sqrt_f32_ref(x: f32) -> f32 {
    let safe = if x.is_finite() && x > f32::MIN_POSITIVE {
        x
    } else {
        f32::MIN_POSITIVE
    };
    1.0 / safe.sqrt()
}

pub(crate) fn ternary_f32_program(
    op_id: &'static str,
    a: &str,
    b: &str,
    c: &str,
    out: &str,
    n: u32,
) -> Program {
    let body = vec![wrap_anonymous_region(
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

/// Registration fixture: one case, one packed input buffer.
///
/// A backend-allocated output takes no host slot, so a fixture that carried a
/// zeroed placeholder for one described a call the artifact ABI rejects.
pub(crate) fn packed_u32_input(words: &[u32]) -> Vec<Vec<Vec<u8>>> {
    vec![vec![crate::wire::pack_u32_slice(words)]]
}

pub(crate) fn pack_f32(values: &[f32]) -> Vec<u8> {
    crate::wire::pack_f32_slice(values)
}
#[cfg(test)]
pub(crate) fn assert_unary_u32_case<F>(build: F, input: &[u32], expected: &[u32])
where
    F: Fn(&str, &str, u32) -> Program,
{
    let n = input.len() as u32;
    let program = build("input", "out", n.max(1));
    let outputs = run_program(&program, vec![crate::wire::pack_u32_slice(input)]);
    assert_eq!(outputs, vec![crate::wire::pack_u32_slice(expected)]);
}

#[cfg(test)]
pub(crate) fn run_program(program: &Program, inputs: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let values = vyre_reference::reference_inputs(program, inputs);
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
