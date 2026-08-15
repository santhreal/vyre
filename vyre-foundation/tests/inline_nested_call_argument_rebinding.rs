//! A callee that passes one of its own parameters to another op.
//!
//! Inlining rewrites a callee body into the caller's namespace under one
//! policy: rename callee locals, substitute a scalar argument for the parameter
//! it was bound to, and retarget a buffer-reference argument at the caller's
//! buffer. That policy used to be restated in three places, and the copy that
//! ran over a nested call's arguments had half of it: it renamed locals and did
//! nothing else. So `call outer(BufferRef(data))`, where `outer`'s body is
//! `call inner(load(param, 0))`, inlined to a program that reads `param` - a
//! buffer only the callee declares.
//!
//! That miscompiles rather than failing: the leaked name lowers to a binding
//! the caller never declared, and the kernel reads whichever slot that binding
//! lands on. Validation does not catch it either, because the inlined program
//! keeps the caller's buffer table, so `param` is simply an undeclared read.
//!
//! Both argument kinds are covered, because they fail differently. A
//! buffer-reference argument leaks the parameter's name; a scalar argument
//! leaks it AND loses the substituted value.

use std::ops::ControlFlow;

use vyre_foundation::ir::{
    inline_calls_with_resolver, BufferAccess, BufferDecl, DataType, Expr, Node, Program,
};
use vyre_foundation::transform::visit::{expr_buffer_ref, try_for_each_expr, ExprBufferRef};

/// `inner(v) = v * 2`, taking its argument through a read-only parameter.
fn inner_callee() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("v", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::mul(
                Expr::Load {
                    buffer: "v".into(),
                    index: Box::new(Expr::u32(0)),
                },
                Expr::u32(2),
            ),
        )],
    )
}

/// `outer(param) = inner(param[0])`: the nested call's argument reads the
/// callee's own parameter, which is the position the partial policy walked.
fn outer_callee() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("param", 0, BufferAccess::ReadOnly, DataType::U32).with_count(8),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::Call {
                op_id: "inner".into(),
                args: vec![Expr::Load {
                    buffer: "param".into(),
                    index: Box::new(Expr::u32(0)),
                }],
            },
        )],
    )
}

fn resolver(op_id: &str) -> Option<Program> {
    match op_id {
        "inner" => Some(inner_callee()),
        "outer" => Some(outer_callee()),
        _ => None,
    }
}

/// Caller declaring `data` and `idx`, calling `outer` with one argument.
fn caller(arg: Expr) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("data", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::storage("idx", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Call {
                op_id: "outer".into(),
                args: vec![arg],
            },
        )],
    )
}

/// Every buffer name any expression in `program` reads or writes.
///
/// Descent is `try_for_each_expr` and the buffer question is
/// `expr_buffer_ref`, both exhaustive owners, so a buffer reached through a
/// position this test did not think of still counts. A hand-rolled walk here
/// would report "no leak" for the positions it forgot, which is the same
/// failure this test exists to catch.
fn buffers_named(program: &Program) -> Vec<String> {
    let mut names = Vec::new();
    let flow: ControlFlow<()> = try_for_each_expr(program.entry(), |expr| {
        match expr_buffer_ref(expr) {
            ExprBufferRef::Read(buffer) | ExprBufferRef::ReadWrite(buffer) => {
                names.push(buffer.to_string());
            }
            ExprBufferRef::None | ExprBufferRef::Unknown => {}
        }
        ControlFlow::Continue(())
    });
    assert!(flow.is_continue(), "the collector never stops early");
    names
}

/// The buffer names a program is allowed to mention: its own declarations.
fn declared(program: &Program) -> Vec<String> {
    program
        .buffers()
        .iter()
        .map(|buffer| buffer.name().to_string())
        .collect()
}

/// A buffer-reference argument must reach the nested call too, so the nested
/// callee reads the CALLER's buffer and the parameter name disappears.
#[test]
fn a_buffer_argument_is_retargeted_inside_a_nested_call() {
    let program = caller(Expr::BufferRef {
        buffer: "data".into(),
    });
    let inlined = inline_calls_with_resolver(&program, resolver).expect("inline");

    let declared = declared(&inlined);
    for buffer in buffers_named(&inlined) {
        assert!(
            declared.contains(&buffer),
            "inlined program reads `{buffer}`, which it does not declare; declared: {declared:?}"
        );
    }
    assert!(
        buffers_named(&inlined).iter().any(|name| name == "data"),
        "the caller's buffer must be what the nested callee reads"
    );
}

/// A scalar argument must be substituted inside the nested call. Leaving the
/// parameter behind loses the value as well as leaking the name.
#[test]
fn a_scalar_argument_is_substituted_inside_a_nested_call() {
    let program = caller(Expr::Load {
        buffer: "idx".into(),
        index: Box::new(Expr::u32(0)),
    });
    let inlined = inline_calls_with_resolver(&program, resolver).expect("inline");

    let declared = declared(&inlined);
    let named = buffers_named(&inlined);
    for buffer in &named {
        assert!(
            declared.contains(buffer),
            "inlined program reads `{buffer}`, which it does not declare; declared: {declared:?}"
        );
    }
    assert!(
        named.iter().any(|name| name == "idx"),
        "the scalar argument's own read must survive into the nested callee, got {named:?}"
    );
}

/// The refusal a new buffer-carrying expression position gets, stated as
/// behavior: a callee parameter reached through a position the policy cannot
/// retarget is refused, not emitted against a buffer the caller lacks.
///
/// `Expr::BufferRef` naming a parameter that was bound to a scalar is that
/// case today: there is no caller buffer to point at, so the read cannot be
/// retargeted and the parameter name must not survive either.
#[test]
fn a_parameter_reached_through_a_buffer_ref_is_not_leaked() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("idx", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Call {
                op_id: "buffer_ref_callee".into(),
                args: vec![Expr::Load {
                    buffer: "idx".into(),
                    index: Box::new(Expr::u32(0)),
                }],
            },
        )],
    );
    fn resolver(op_id: &str) -> Option<Program> {
        (op_id == "buffer_ref_callee").then(|| {
            Program::wrapped(
                vec![
                    BufferDecl::storage("p", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(1),
                    BufferDecl::output("result", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "result",
                    Expr::u32(0),
                    Expr::BufLen {
                        buffer: "p".into(),
                    },
                )],
            )
        })
    }
    let inlined = inline_calls_with_resolver(&program, resolver).expect("inline");
    let declared = declared(&inlined);
    for buffer in buffers_named(&inlined) {
        assert!(
            declared.contains(&buffer),
            "inlined program reads `{buffer}`, which it does not declare; declared: {declared:?}"
        );
    }
}
