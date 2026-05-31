//! Generated truth and structure checks for the GPU-native text line index.

#![cfg(all(feature = "text", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_foundation::ir::{BufferAccess, Expr, Node, Program};
use vyre_primitives::reduce::multi_block_prefix_scan::BLOCK_LANES;
use vyre_primitives::text::line_index::{line_index, reference_line_index};

fn independent_prefix_flag_line_index(source: &[u8]) -> Vec<u32> {
    let mut acc = 0u32;
    let mut out = Vec::with_capacity(source.len());
    for (index, &byte) in source.iter().enumerate() {
        let flag = u32::from(
            byte == b'\n'
                || (byte == b'\r' && index + 1 < source.len() && source[index + 1] != b'\n'),
        );
        acc = acc.wrapping_add(flag);
        out.push(acc.wrapping_sub(flag));
    }
    out
}

fn byte_strategy() -> impl Strategy<Value = u8> {
    prop_oneof![
        4 => Just(b'\n'),
        4 => Just(b'\r'),
        1 => Just(0u8),
        1 => Just(0xFFu8),
        8 => any::<u8>(),
    ]
}

fn contains_loop(program: &Program) -> bool {
    program.entry().iter().any(node_contains_loop)
}

fn node_contains_loop(node: &Node) -> bool {
    match node {
        Node::Loop { .. } => true,
        Node::Block(children) => children.iter().any(node_contains_loop),
        Node::If {
            then, otherwise, ..
        } => then.iter().any(node_contains_loop) || otherwise.iter().any(node_contains_loop),
        Node::Region { body, .. } => body.iter().any(node_contains_loop),
        _ => false,
    }
}

fn contains_lane_zero_serial_gate(program: &Program) -> bool {
    program
        .entry()
        .iter()
        .any(node_contains_lane_zero_serial_gate)
}

fn node_contains_lane_zero_serial_gate(node: &Node) -> bool {
    match node {
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            expr_is_invocation_zero(cond)
                || then.iter().any(node_contains_lane_zero_serial_gate)
                || otherwise.iter().any(node_contains_lane_zero_serial_gate)
        }
        Node::Block(children) => children.iter().any(node_contains_lane_zero_serial_gate),
        Node::Loop { body, .. } => body.iter().any(node_contains_lane_zero_serial_gate),
        Node::Region { body, .. } => body.iter().any(node_contains_lane_zero_serial_gate),
        _ => false,
    }
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
            expr_is_invocation_zero(a) || expr_is_invocation_zero(b) || expr_is_invocation_zero(c)
        }
        _ => false,
    }
}

fn output_buffer_names(program: &Program) -> Vec<&str> {
    program
        .buffers()
        .iter()
        .filter(|buffer| {
            buffer.is_output()
                || buffer.is_pipeline_live_out()
                || matches!(
                    buffer.access(),
                    BufferAccess::ReadWrite | BufferAccess::WriteOnly
                )
        })
        .map(|buffer| buffer.name())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn reference_matches_independent_prefix_flags(
        source in proptest::collection::vec(byte_strategy(), 0..=256),
    ) {
        prop_assert_eq!(
            reference_line_index(&source),
            independent_prefix_flag_line_index(&source)
        );
    }

    #[test]
    fn builder_does_not_regress_to_lane_zero_serial_loop(
        n in 1u32..=(BLOCK_LANES * 4),
    ) {
        let program = line_index("source", "lines", n);

        prop_assert_eq!(program.workgroup_size(), [BLOCK_LANES, 1, 1]);
        prop_assert!(
            !contains_loop(&program),
            "line_index must not contain a serial byte loop for n={n}"
        );
        prop_assert!(
            !contains_lane_zero_serial_gate(&program),
            "line_index must not gate all useful work behind InvocationId.x == 0 for n={n}"
        );
        let has_source = program.buffers().iter().any(|buffer| {
            buffer.name() == "source"
                && buffer.access() == BufferAccess::ReadOnly
                && buffer.count() == n
        });
        let has_flags = program.buffers().iter().any(|buffer| {
            buffer.name() == "__lines_line_break_flags"
                && buffer.count() == n
                && buffer.is_pipeline_live_out()
                && !buffer.is_output()
        });
        let has_lines = program
            .buffers()
            .iter()
            .any(|buffer| buffer.name() == "lines" && buffer.count() == n && buffer.is_output());
        prop_assert!(has_source, "line_index source input missing for n={n}");
        prop_assert!(has_flags, "line_index break flags missing for n={n}");
        prop_assert!(has_lines, "line_index final output missing for n={n}");
        prop_assert_eq!(
            program
                .buffers()
                .iter()
                .filter(|buffer| buffer.is_output())
                .count(),
            1
        );
        prop_assert!(
            output_buffer_names(&program).contains(&"lines"),
            "line_index final output must remain visible for n={n}"
        );
    }
}
