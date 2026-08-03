//! Canonicalize as a dispatched compute kernel.
//!
//! V1 scope: the load-bearing rewrite  -  for every commutative `BinOp`
//! whose left operand is a literal and right operand is not, swap them
//! so literals end up on the right. Other canonicalize rules (the
//! non-literal sort tie-break and the `x == x` self-equality fold)
//! are CPU-side today; they migrate as separate kernels in V2.
//!
//! The kernel reads the ExprArena's kinds + arg arrays, marks each
//! BinOp ExprId with a `swap_mask[i] = 1` if it needs the operand
//! swap. The decoder walks the IR in lockstep with the encoder and
//! applies the swap when reconstructing each BinOp. No host-reference escape.

use vyre_foundation::ir::{Expr, Node, Program};

use super::dispatcher::{DispatchError, OptimizerDispatcher};
use super::encode::EncodeError;
use super::expr_arena::{encode_expr_arena, expr_kind, ExprArenaEncoding};

#[derive(Debug, Default)]
struct CanonicalizeKernelScratch {
    inputs: Vec<Vec<u8>>,
}

/// Errors surfaced by `gpu_canonicalize`.
#[derive(Debug)]
pub enum CanonicalizeError {
    /// Expression-arena encoding failed.
    Encode(EncodeError),
    /// Backend dispatch or output decoding failed.
    Dispatch(DispatchError),
}

impl std::fmt::Display for CanonicalizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_canonicalize encode error: {err:?}"),
            Self::Dispatch(err) => write!(f, "gpu_canonicalize dispatch error: {err}"),
        }
    }
}

impl std::error::Error for CanonicalizeError {}

/// Run literal-on-right canonicalization on `program`.
pub fn gpu_canonicalize(
    program: Program,
    dispatcher: &dyn OptimizerDispatcher,
) -> Result<Program, CanonicalizeError> {
    let arena = encode_expr_arena(&program).map_err(CanonicalizeError::Encode)?;
    if arena.expr_count == 0 {
        return Ok(program);
    }
    let mut scratch = CanonicalizeKernelScratch::default();
    let mut swap_mask = Vec::with_capacity(arena.expr_count as usize);
    run_canonicalize_kernel_with_scratch_into(&arena, dispatcher, &mut scratch, &mut swap_mask)
        .map_err(CanonicalizeError::Dispatch)?;
    Ok(rewrite_program_with_swap_mask(program, &swap_mask))
}

#[cfg(test)]
fn run_canonicalize_kernel_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn OptimizerDispatcher,
    swap_mask: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = CanonicalizeKernelScratch::default();
    run_canonicalize_kernel_with_scratch_into(&arena, dispatcher, &mut scratch, swap_mask)
}

fn run_canonicalize_kernel_with_scratch_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn OptimizerDispatcher,
    scratch: &mut CanonicalizeKernelScratch,
    swap_mask: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    super::run_encoded_analysis_kernel(
        arena,
        dispatcher,
        &mut scratch.inputs,
        swap_mask,
        build_canonicalize_program,
        "canonicalize",
        "swap_mask",
    )
}

/// Build the canonicalize analysis Program. Reads arena cols, writes
/// `swap_mask[i] = 1` for any BIN_OP whose left operand is a literal
/// and right operand is not. Each GPU thread handles one Expr id via
/// `gid_x()`; the orchestrator dispatches `ceil(expr_count / 256)`
/// workgroups to cover the input.
pub fn build_canonicalize_program(expr_count: u32) -> Program {
    super::build_encoded_analysis_program(expr_count, "swap_mask", per_expr_body())
}

fn per_expr_body() -> Vec<Node> {
    vec![
        Node::let_bind("kind", Expr::load("arena_kinds", Expr::var("i"))),
        // Only BIN_OPs are subject to swap.
        Node::if_then(
            Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::BIN_OP)),
            bin_op_body(),
        ),
    ]
}

fn bin_op_body() -> Vec<Node> {
    // Commutative op tags: Add(0x01), Mul(0x03), BitAnd(0x06),
    // BitOr(0x07), BitXor(0x08), Eq(0x0B), Ne(0x0C), And(0x12),
    // Or(0x13), AbsDiff(0x14), Min(0x15), Max(0x16). `Min`/`Max`
    // and `AbsDiff` are mathematically commutative  -  including them
    // here means literal-on-right canonicalization fires for them
    // too, lining up `(Min 5 x)` and `(Min x 5)` for CSE.
    vec![
        Node::let_bind("op", Expr::load("arena_arg0", Expr::var("i"))),
        Node::let_bind(
            "is_commutative",
            Expr::or(
                Expr::or(
                    Expr::or(
                        Expr::or(
                            Expr::eq(Expr::var("op"), Expr::u32(0x01)),
                            Expr::eq(Expr::var("op"), Expr::u32(0x03)),
                        ),
                        Expr::or(
                            Expr::eq(Expr::var("op"), Expr::u32(0x06)),
                            Expr::eq(Expr::var("op"), Expr::u32(0x07)),
                        ),
                    ),
                    Expr::or(
                        Expr::or(
                            Expr::eq(Expr::var("op"), Expr::u32(0x08)),
                            Expr::eq(Expr::var("op"), Expr::u32(0x0B)),
                        ),
                        Expr::or(
                            Expr::eq(Expr::var("op"), Expr::u32(0x0C)),
                            Expr::eq(Expr::var("op"), Expr::u32(0x12)),
                        ),
                    ),
                ),
                Expr::or(
                    Expr::or(
                        Expr::or(
                            Expr::eq(Expr::var("op"), Expr::u32(0x13)),
                            Expr::eq(Expr::var("op"), Expr::u32(0x14)),
                        ),
                        Expr::or(
                            Expr::eq(Expr::var("op"), Expr::u32(0x15)),
                            Expr::eq(Expr::var("op"), Expr::u32(0x16)),
                        ),
                    ),
                    Expr::or(
                        Expr::or(
                            // SaturatingAdd
                            Expr::eq(Expr::var("op"), Expr::u32(0x17)),
                            // SaturatingMul
                            Expr::eq(Expr::var("op"), Expr::u32(0x19)),
                        ),
                        // WrappingAdd
                        Expr::eq(Expr::var("op"), Expr::u32(0x20)),
                    ),
                ),
            ),
        ),
        Node::if_then(
            Expr::var("is_commutative"),
            vec![
                Node::let_bind("l", Expr::load("arena_arg1", Expr::var("i"))),
                Node::let_bind("r", Expr::load("arena_arg2", Expr::var("i"))),
                Node::let_bind("l_kind", Expr::load("arena_kinds", Expr::var("l"))),
                Node::let_bind("r_kind", Expr::load("arena_kinds", Expr::var("r"))),
                // Literal kinds are 0x01..=0x04 (LIT_U32..LIT_BOOL).
                // l_is_lit := (l_kind >= LIT_U32) && (l_kind <= LIT_BOOL)
                // For simplicity check kind >= 1 && kind <= 4.
                Node::let_bind(
                    "l_is_lit",
                    Expr::and(
                        Expr::ge(Expr::var("l_kind"), Expr::u32(expr_kind::LIT_U32)),
                        Expr::le(Expr::var("l_kind"), Expr::u32(expr_kind::LIT_BOOL)),
                    ),
                ),
                Node::let_bind(
                    "r_is_lit",
                    Expr::and(
                        Expr::ge(Expr::var("r_kind"), Expr::u32(expr_kind::LIT_U32)),
                        Expr::le(Expr::var("r_kind"), Expr::u32(expr_kind::LIT_BOOL)),
                    ),
                ),
                // Swap iff l is literal AND r is not.
                Node::if_then(
                    Expr::and(
                        Expr::var("l_is_lit"),
                        Expr::eq(Expr::var("r_is_lit"), Expr::bool(false)),
                    ),
                    vec![Node::store("swap_mask", Expr::var("i"), Expr::u32(1))],
                ),
                // Also swap when neither operand is literal but the
                // left arena id is strictly greater than the right.
                // Establishes a deterministic operand ordering for
                // commutative ops, which lets CSE recognise
                // `(Add a b)` and `(Add b a)` as equivalent without
                // depending on lexical authoring order.
                Node::if_then(
                    Expr::and(
                        Expr::and(
                            Expr::eq(Expr::var("l_is_lit"), Expr::bool(false)),
                            Expr::eq(Expr::var("r_is_lit"), Expr::bool(false)),
                        ),
                        Expr::gt(Expr::var("l"), Expr::var("r")),
                    ),
                    vec![Node::store("swap_mask", Expr::var("i"), Expr::u32(1))],
                ),
            ],
        ),
    ]
}

fn rewrite_program_with_swap_mask(program: Program, swap_mask: &[u32]) -> Program {
    super::rewrite_walk::rewrite_program_with_expr_rewriter(program, |expr, counter| {
        rewrite_expr(expr, swap_mask, counter)
    })
}

fn rewrite_expr(expr: &Expr, swap_mask: &[u32], counter: &mut u32) -> Expr {
    super::rewrite_walk::rewrite_simple_expr_postorder(expr, counter, &mut |rewritten, id| {
        match rewritten {
            Expr::BinOp { op, left, right }
                if swap_mask.get(id as usize).copied().unwrap_or(0) == 1 =>
            {
                Expr::BinOp {
                    op,
                    left: right,
                    right: left,
                }
            }
            other => other,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;

    struct CanonicalizeDispatcher {
        outputs: Vec<Vec<u8>>,
    }

    impl OptimizerDispatcher for CanonicalizeDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_eq!(grid_override, Some([1, 1, 1]));
            if inputs.len() != 5 {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: canonicalize test dispatcher expected 5 inputs, got {}.",
                    inputs.len()
                )));
            }
            Ok(self.outputs.clone())
        }
    }

    fn one_expr_arena() -> ExprArenaEncoding {
        ExprArenaEncoding {
            expr_count: 1,
            kinds: vec![expr_kind::LIT_U32],
            arg0: vec![0],
            arg1: vec![0],
            arg2: vec![0],
            ..ExprArenaEncoding::default()
        }
    }

    #[test]
    fn kernel_into_decodes_exact_swap_mask_into_reused_buffer() {
        let dispatcher = CanonicalizeDispatcher {
            outputs: vec![u32_slice_to_le_bytes(&[1])],
        };
        let arena = one_expr_arena();
        let mut swap_mask = Vec::with_capacity(4);
        let ptr = swap_mask.as_ptr();
        run_canonicalize_kernel_into(&arena, &dispatcher, &mut swap_mask)
            .expect("Fix: dispatch succeeds");
        assert_eq!(swap_mask, vec![1]);
        assert_eq!(swap_mask.as_ptr(), ptr);
    }

    #[test]
    fn kernel_with_scratch_reuses_dispatch_and_output_storage() {
        let dispatcher = CanonicalizeDispatcher {
            outputs: vec![u32_slice_to_le_bytes(&[1])],
        };
        let arena = one_expr_arena();
        let mut scratch = CanonicalizeKernelScratch::default();
        let mut swap_mask = Vec::with_capacity(1);

        run_canonicalize_kernel_with_scratch_into(
            &arena,
            &dispatcher,
            &mut scratch,
            &mut swap_mask,
        )
        .expect("Fix: dispatch succeeds");

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let swap_capacity = swap_mask.capacity();

        run_canonicalize_kernel_with_scratch_into(
            &arena,
            &dispatcher,
            &mut scratch,
            &mut swap_mask,
        )
        .expect("Fix: dispatch succeeds");

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(swap_mask.capacity(), swap_capacity);
        assert_eq!(swap_mask, vec![1]);
    }

    #[test]
    fn kernel_rejects_extra_outputs() {
        let dispatcher = CanonicalizeDispatcher {
            outputs: vec![u32_slice_to_le_bytes(&[1]), u32_slice_to_le_bytes(&[0])],
        };
        let mut swap_mask = Vec::new();
        let err = run_canonicalize_kernel_into(&one_expr_arena(), &dispatcher, &mut swap_mask)
            .expect_err("extra outputs must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn kernel_rejects_trailing_swap_mask_bytes() {
        let dispatcher = CanonicalizeDispatcher {
            outputs: vec![vec![1, 0, 0, 0, 2]],
        };
        let mut swap_mask = Vec::new();
        let err = run_canonicalize_kernel_into(&one_expr_arena(), &dispatcher, &mut swap_mask)
            .expect_err("trailing bytes must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }
}
