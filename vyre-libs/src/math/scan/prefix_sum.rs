//! Prefix-sum scan  -  inclusive scan over a u32 buffer.
//!
//! Category A composition and the one size contract over the Tier-2.5 scan
//! primitives. `scan_prefix_sum` is where an element count picks an algorithm:
//! at or under [`MAX_SINGLE_BLOCK_SCAN`] the compact workgroup scan, above it
//! the multi-block chain. The primitives own the two bodies and neither of them
//! chooses.

use crate::math::prefix_scan::{prefix_scan, ScanKind, MAX_SINGLE_BLOCK_SCAN};
use crate::plumbing::program::attribution::attribute_child_nodes;
use crate::reduce::multi_block_prefix_scan::multi_block_prefix_scan_sum_u32;
use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::Program;

const OP_ID: &str = "vyre-libs::math::scan_prefix_sum";

/// The single-block scan body, as a phase boundary inside one operation.
///
/// It carries the `anonymous::` prefix over the builder's own id because that
/// id registers no canonical operation, and a child region naming an
/// unregistered id claims a building block that does not exist.
const SINGLE_BLOCK_CHILD: &str = "anonymous::vyre-libs::math::prefix_scan_inclusive_sum";

/// Build a Program that computes the inclusive prefix sum of `input`
/// into `output`, both sized `n`.
///
/// **Overflow semantics** (V7-CORR-018): all accumulator additions
/// use `u32::wrapping_add`. For inputs whose cumulative sum exceeds
/// `u32::MAX`, the output wraps modulo 2^32.
#[must_use]
pub fn scan_prefix_sum(input: &str, output: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((output, vyre_foundation::ir::DataType::U32)),
            "Fix: scan_prefix_sum requires n > 0.".to_string(),
        );
    }
    if n <= MAX_SINGLE_BLOCK_SCAN {
        compose_scan_primitive(
            SINGLE_BLOCK_CHILD,
            prefix_scan(input, output, n, ScanKind::InclusiveSum),
        )
    } else {
        compose_scan_primitive(
            crate::reduce::multi_block_prefix_scan::OP_ID_INCLUSIVE_SUM,
            multi_block_prefix_scan_sum_u32(input, output, n),
        )
    }
}

/// Declare the scan primitive this composition selected as its child.
///
/// The primitive builds its own region; this replaces that region with the
/// same body under the same generator, attributed to this composition, so the
/// selection is an edge to a registered building block rather than a relabel
/// of the body.
fn compose_scan_primitive(child_id: &'static str, program: Program) -> Program {
    let child = attribute_child_nodes(OP_ID, child_id, &program);
    program.with_rewritten_wrapped_entry(vec![wrap_anonymous_region(OP_ID, child)])
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || scan_prefix_sum("input", "output", 4),
        Some(|| vec![vec![
            vyre_primitives::wire::pack_u32_slice(&[1u32, 2, 3, 4]),
        ]]),
        Some(|| vec![vec![
            // Only ReadWrite buffer: prefix sum [1, 3, 6, 10]
            vyre_primitives::wire::pack_u32_slice(&[1u32, 3, 6, 10]),
        ]]),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::{bytes_to_u32 as decode_u32_words, u32_bytes};
    use vyre_foundation::ir::{BufferAccess, Expr, Node};
    use vyre_foundation::visit::any_descendant;
    use vyre_reference::value::Value;

    /// Run `scan_prefix_sum` through the reference interpreter and return the
    /// `output` buffer. `reference_eval` takes one Value per non-workgroup buffer
    /// in binding order (outputs seeded with a zero slot) and returns the
    /// ReadWrite buffers in binding order. The large multi-block path fuses in
    /// scratch buffers (`partials`, `block_totals`, ...), so this feeds a zero
    /// slot for each and locates `output` among the returned writable buffers
    /// rather than hard-coding index 0.
    fn run_scan(n: u32, input: &[u32]) -> Vec<u32> {
        let program = scan_prefix_sum("input", "output", n);
        let mut inputs = Vec::new();
        let mut output_result_index = None;
        let mut writable_seen = 0usize;
        for buf in program.buffers() {
            if buf.access() == BufferAccess::Workgroup {
                continue;
            }
            let bytes = if buf.name() == "input" {
                u32_bytes(input)
            } else {
                vec![0u8; (buf.count() as usize).saturating_mul(4)]
            };
            inputs.push(Value::from(bytes));
            if buf.access() == BufferAccess::ReadWrite {
                if buf.name() == "output" {
                    output_result_index = Some(writable_seen);
                }
                writable_seen += 1;
            }
        }
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: prefix sum must execute");
        let idx = output_result_index.expect("output buffer must be present and writable");
        decode_u32_words(&outputs[idx].to_bytes())
    }

    #[test]
    fn prefix_sum_single_element() {
        let input = [42u32];
        let actual = run_scan(1, &input);
        assert_eq!(actual, vec![42u32]);
    }

    #[test]
    fn prefix_sum_empty_n_zero_should_trap() {
        let program = scan_prefix_sum("input", "output", 0);
        let error = vyre_reference::reference_eval(
            &program,
            &[Value::from(vec![0u8; 0]), Value::from(vec![0u8; 0])],
        )
        .expect_err("n=0 prefix_sum must trap instead of returning empty");
        let msg = error.to_string();
        assert!(
            msg.contains("trap") || msg.contains("Fix:"),
            "n=0 prefix_sum error must be actionable: {msg}"
        );
    }

    #[test]
    fn prefix_sum_boundary_small_path() {
        let input: Vec<u32> = (1..=1024).collect();
        let actual = run_scan(1024, &input);
        let expected: Vec<u32> = input
            .iter()
            .scan(0u32, |acc, &x| {
                *acc = acc.wrapping_add(x);
                Some(*acc)
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn prefix_sum_boundary_large_path_is_parallel_multi_block() {
        let program = scan_prefix_sum("input", "output", 1025);
        assert_top_region_generator(&program, OP_ID);
        // The width is owned by `multi_block_prefix_scan`, which declares the portable invocation
        // floor. Restating a number here would pin the test to a width the scan no longer uses.
        assert_eq!(
            program.workgroup_size(),
            [vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS, 1, 1]
        );
        assert!(
            !contains_loop(&program),
            "large scan_prefix_sum must not route through a serial per-element loop"
        );
        assert!(
            !contains_invocation_zero_gate(&program),
            "large scan_prefix_sum must not gate useful work behind InvocationId.x == 0"
        );
        assert!(program
            .buffers()
            .iter()
            .any(|buffer| buffer.name() == "output" && buffer.is_output()));
    }

    #[test]
    fn prefix_sum_large_path_parallel_shape_sweep() {
        for n in 1025..=4097 {
            let program = scan_prefix_sum("input", "output", n);
            assert_top_region_generator(&program, OP_ID);
            assert_eq!(
                program.workgroup_size(),
                [vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS, 1, 1],
                "n={n}"
            );
            assert!(
                !contains_loop(&program),
                "n={n}: large scan_prefix_sum must not emit a serial loop"
            );
            assert!(
                !contains_invocation_zero_gate(&program),
                "n={n}: large scan_prefix_sum must not gate useful work behind InvocationId.x == 0"
            );
            assert!(
                program
                    .buffers()
                    .iter()
                    .any(|buffer| buffer.name() == "output"
                        && buffer.is_output()
                        && buffer.count() == n),
                "n={n}: final output buffer must be declared with the requested element count"
            );
        }
    }

    #[test]
    fn prefix_sum_overflow_wraps() {
        let input = [u32::MAX, 1u32, 1u32];
        let actual = run_scan(3, &input);
        assert_eq!(actual[0], u32::MAX);
        assert_eq!(actual[1], 0u32, "u32::MAX + 1 must wrap to 0");
        assert_eq!(actual[2], 1u32, "0 + 1 must be 1");
    }

    /// Inclusive scan with `wrapping_add`, the documented overflow semantics.
    fn wrapping_scan_oracle(input: &[u32]) -> Vec<u32> {
        input
            .iter()
            .scan(0u32, |acc, &x| {
                *acc = acc.wrapping_add(x);
                Some(*acc)
            })
            .collect()
    }

    #[test]
    fn prefix_sum_large_path_matches_scan_oracle_across_block_boundaries() {
        // The n>1024 route goes through `multi_block_prefix_scan_sum_u32`, a
        // DIFFERENT algorithm than the compact one-block scan. The other
        // large-path tests assert only STRUCTURE (shape, no serial loop, no
        // invocation-zero gate); none check the VALUE. A broken cross-block
        // carry (dropped/duplicated block prefix, off-by-one block seam) would
        // pass all of them. This runs the real IR through `reference_eval` and
        // compares to the wrapping-scan oracle across exact block boundaries
        // (multiples of 1024) and off-boundaries.
        for n in [1025u32, 1536, 2048, 3072, 4096, 4097] {
            // Non-constant pattern so a mis-combined block carry changes the
            // result (a constant input hides carry bugs behind a uniform sum).
            let input: Vec<u32> = (0..n).map(|i| (i % 251) + 1).collect();
            let actual = run_scan(n, &input);
            let expected = wrapping_scan_oracle(&input);
            assert_eq!(
                actual.len(),
                n as usize,
                "n={n}: large scan must emit n outputs"
            );
            assert_eq!(
                actual, expected,
                "n={n}: large multi-block prefix sum diverged from the wrapping-scan oracle"
            );
        }
    }

    #[test]
    fn prefix_sum_large_path_wraps_across_block_seams() {
        // Overflow must wrap modulo 2^32 even when the running sum overflows
        // partway through the multi-block combine, not just within one block.
        let n = 2048u32;
        let mut input = vec![1u32; n as usize];
        input[900] = u32::MAX; // forces a wrap inside the first block's carry-out
        let actual = run_scan(n, &input);
        let expected = wrapping_scan_oracle(&input);
        assert_eq!(
            actual, expected,
            "large-path prefix sum must wrap modulo 2^32 across block boundaries"
        );
    }

    fn assert_top_region_generator(program: &Program, expected: &str) {
        match program.entry() {
            [Node::Region { generator, .. }] => assert_eq!(generator.as_str(), expected),
            other => panic!("expected single top-level Region, got {other:?}"),
        }
    }

    fn contains_loop(program: &Program) -> bool {
        program
            .entry()
            .iter()
            .any(|node| any_descendant(node, &mut |n| matches!(n, Node::Loop { .. })))
    }

    /// True when any `If` anywhere in `program` is gated on invocation zero.
    ///
    /// Descent comes from `visit::any_descendant`, the one owner of
    /// which node variants nest. The two hand-written matches this replaces both
    /// ended in `_ => false`, so a fifth body-bearing variant would have hidden
    /// the gate and the assertion would have reported the wrong shape.
    fn contains_invocation_zero_gate(program: &Program) -> bool {
        program.entry().iter().any(|node| {
            any_descendant(
                node,
                &mut |n| matches!(n, Node::If { cond, .. } if expr_is_invocation_zero(cond)),
            )
        })
    }

    fn expr_is_invocation_zero(expr: &Expr) -> bool {
        match expr {
            Expr::BinOp { op, left, right } if *op == vyre_foundation::ir::BinOp::Eq => {
                matches!(
                    (&**left, &**right),
                    (Expr::InvocationId { axis: 0 }, Expr::LitU32(0))
                        | (Expr::LitU32(0), Expr::InvocationId { axis: 0 })
                )
            }
            Expr::BinOp { left, right, .. } => {
                expr_is_invocation_zero(left) || expr_is_invocation_zero(right)
            }
            Expr::UnOp { operand, .. } => expr_is_invocation_zero(operand),
            Expr::Load { index, .. } => expr_is_invocation_zero(index),
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => {
                expr_is_invocation_zero(cond)
                    || expr_is_invocation_zero(true_val)
                    || expr_is_invocation_zero(false_val)
            }
            Expr::Atomic {
                index,
                expected,
                value,
                ..
            } => {
                expr_is_invocation_zero(index)
                    || expected
                        .as_ref()
                        .is_some_and(|expr| expr_is_invocation_zero(expr))
                    || expr_is_invocation_zero(value)
            }
            Expr::Cast { value, .. } => expr_is_invocation_zero(value),
            Expr::Call { args, .. } => args.iter().any(expr_is_invocation_zero),
            Expr::Fma { a, b, c } => {
                expr_is_invocation_zero(a)
                    || expr_is_invocation_zero(b)
                    || expr_is_invocation_zero(c)
            }
            _ => false,
        }
    }
}
