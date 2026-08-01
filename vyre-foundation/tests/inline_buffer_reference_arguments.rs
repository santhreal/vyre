//! Passing a whole buffer to a composite op.
//!
//! A composite op that indexes a table cannot receive that table as a scalar:
//! the value at one index is not the table. Before `Expr::BufferRef` existed
//! the only way to name a buffer at a call site was `Expr::Var`, which the
//! validator treats as a scope-bound variable, so every such program was
//! rejected with "reference to undeclared variable". These tests pin the
//! binding rules that replaced that dead end:
//!
//! - a buffer argument RETARGETS accesses inside the callee at the caller's
//!   buffer and keeps the callee's index expression,
//! - a scalar argument still SUBSTITUTES its value, unchanged,
//! - `BufLen` follows the same split, the caller's real length for a buffer
//!   argument and 1 for a scalar,
//! - the callee's own parameter buffer never appears in the inlined result.
//!
//! The last point is the one that miscompiles silently: a leaked callee
//! buffer name lowers to a binding the caller never declared, and the kernel
//! reads whatever slot that binding happens to land on.

use vyre_foundation::ir::{
    inline_calls_with_resolver, AtomicOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program,
};
use vyre_foundation::memory_model::MemoryOrdering;

/// Callee reading `table[i]` where `table` is its input buffer and `i` its
/// index parameter. `i` is a second read-only buffer so both binding kinds
/// appear in one callee.
fn table_lookup_callee() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("table", 0, BufferAccess::ReadOnly, DataType::U32).with_count(16),
            BufferDecl::storage("i", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("result", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::Load {
                buffer: "table".into(),
                index: Box::new(Expr::add(
                    Expr::Load {
                        buffer: "i".into(),
                        index: Box::new(Expr::u32(0)),
                    },
                    Expr::u32(3),
                )),
            },
        )],
    )
}

fn resolver(op_id: &str) -> Option<Program> {
    match op_id {
        "table_lookup" => Some(table_lookup_callee()),
        "table_len" => Some(buflen_callee()),
        "table_bump" => Some(atomic_callee()),
        _ => None,
    }
}

/// Callee whose result is the length of its input buffer.
fn buflen_callee() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("table", 0, BufferAccess::ReadOnly, DataType::U32).with_count(16),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::BufLen {
                buffer: "table".into(),
            },
        )],
    )
}

/// Callee that atomically adds one to `table[0]`.
fn atomic_callee() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("table", 0, BufferAccess::ReadOnly, DataType::U32).with_count(16),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::Atomic {
                op: AtomicOp::Add,
                buffer: "table".into(),
                index: Box::new(Expr::u32(0)),
                expected: None,
                value: Box::new(Expr::u32(1)),
                ordering: MemoryOrdering::Relaxed,
            },
        )],
    )
}

/// Caller with a real `data` table plus an `idx` scalar source, calling
/// `op_id` with the given arguments.
fn caller_calling(op_id: &str, args: Vec<Expr>) -> Program {
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
                op_id: op_id.into(),
                args,
            },
        )],
    )
}

/// Collect every `Expr::Load` in a program as `(buffer, index debug string)`.
///
/// The walker descends into `Node::Region`, which the inliner wraps expanded
/// callees in. A walker that stops at `Region` reports zero loads for every
/// inlined program and makes these assertions vacuous.
fn loads(program: &Program) -> Vec<(String, String)> {
    let mut found = Vec::new();
    walk_nodes(program.entry(), &mut |expr| {
        if let Expr::Load { buffer, index } = expr {
            found.push((buffer.to_string(), format!("{index:?}")));
        }
    });
    found
}

/// Collect the buffer named by every `Expr::Atomic` in a program.
fn atomic_buffers(program: &Program) -> Vec<String> {
    let mut found = Vec::new();
    walk_nodes(program.entry(), &mut |expr| {
        if let Expr::Atomic { buffer, .. } = expr {
            found.push(buffer.to_string());
        }
    });
    found
}

/// Collect the buffer named by every `Expr::BufLen` in a program.
fn buflens(program: &Program) -> Vec<String> {
    let mut found = Vec::new();
    walk_nodes(program.entry(), &mut |expr| {
        if let Expr::BufLen { buffer } = expr {
            found.push(buffer.to_string());
        }
    });
    found
}

fn walk_nodes(nodes: &[Node], visit: &mut impl FnMut(&Expr)) {
    for node in nodes {
        match node {
            Node::Let { value, .. } | Node::Assign { value, .. } => walk_expr(value, visit),
            Node::Store { index, value, .. } => {
                walk_expr(index, visit);
                walk_expr(value, visit);
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                walk_expr(cond, visit);
                walk_nodes(then, visit);
                walk_nodes(otherwise, visit);
            }
            Node::Loop { from, to, body, .. } => {
                walk_expr(from, visit);
                walk_expr(to, visit);
                walk_nodes(body, visit);
            }
            Node::Block(inner) => walk_nodes(inner, visit),
            Node::Region { body, .. } => walk_nodes(body, visit),
            _ => {}
        }
    }
}

fn walk_expr(expr: &Expr, visit: &mut impl FnMut(&Expr)) {
    visit(expr);
    match expr {
        Expr::Load { index, .. } => walk_expr(index, visit),
        Expr::BinOp { left, right, .. } => {
            walk_expr(left, visit);
            walk_expr(right, visit);
        }
        Expr::UnOp { operand, .. } => walk_expr(operand, visit),
        Expr::Cast { value, .. } => walk_expr(value, visit),
        Expr::Fma { a, b, c } => {
            walk_expr(a, visit);
            walk_expr(b, visit);
            walk_expr(c, visit);
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            walk_expr(cond, visit);
            walk_expr(true_val, visit);
            walk_expr(false_val, visit);
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            walk_expr(index, visit);
            if let Some(expected) = expected {
                walk_expr(expected, visit);
            }
            walk_expr(value, visit);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                walk_expr(arg, visit);
            }
        }
        _ => {}
    }
}

/// The whole point of the variant: the callee's `table[i + 3]` becomes a read
/// of the CALLER's `data` at the same offset, not a substituted scalar and
/// not a read of a buffer the caller never declared.
#[test]
fn buffer_argument_retargets_the_load_and_preserves_the_index() {
    let caller = caller_calling(
        "table_lookup",
        vec![
            Expr::BufferRef {
                buffer: "data".into(),
            },
            Expr::Load {
                buffer: "idx".into(),
                index: Box::new(Expr::u32(0)),
            },
        ],
    );
    let inlined = inline_calls_with_resolver(&caller, resolver).expect("inline");

    let loads = loads(&inlined);
    let data_loads: Vec<&(String, String)> =
        loads.iter().filter(|(buf, _)| buf == "data").collect();
    assert_eq!(
        data_loads.len(),
        1,
        "expected exactly one retargeted load of `data`, got {loads:?}"
    );
    // The callee's index arithmetic survives: `i + 3` where `i` was itself
    // substituted with the caller's `idx[0]` load.
    let index = &data_loads[0].1;
    assert!(
        index.contains("LitU32(3)"),
        "the callee's `+ 3` offset must survive retargeting, got index {index}"
    );
    assert!(
        index.contains("idx"),
        "the scalar argument must be substituted into the index, got index {index}"
    );
}

/// A leaked callee buffer name lowers to a binding the caller never declared.
/// Neither the buffer parameter nor the callee's output buffer may appear.
#[test]
fn the_callees_parameter_buffer_never_leaks_into_the_caller() {
    let caller = caller_calling(
        "table_lookup",
        vec![
            Expr::BufferRef {
                buffer: "data".into(),
            },
            Expr::Load {
                buffer: "idx".into(),
                index: Box::new(Expr::u32(0)),
            },
        ],
    );
    let inlined = inline_calls_with_resolver(&caller, resolver).expect("inline");

    for (buffer, _) in loads(&inlined) {
        assert!(
            buffer == "data" || buffer == "idx",
            "inlined program reads callee-local buffer `{buffer}`; only the caller's own buffers may survive"
        );
    }
    let names: Vec<String> = inlined
        .buffers()
        .iter()
        .map(|b| b.name().to_string())
        .collect();
    assert_eq!(
        names,
        vec!["data".to_string(), "idx".to_string(), "out".to_string()],
        "inlining must not add the callee's buffers to the caller"
    );
}

/// A scalar argument keeps the pre-`BufferRef` behaviour exactly: the read of
/// the parameter is replaced by the argument value itself, with no load of
/// any buffer named by the callee.
#[test]
fn scalar_argument_still_substitutes_the_value_rather_than_retargeting() {
    let caller = caller_calling(
        "table_lookup",
        vec![
            Expr::BufferRef {
                buffer: "data".into(),
            },
            Expr::u32(7),
        ],
    );
    let inlined = inline_calls_with_resolver(&caller, resolver).expect("inline");

    let loads = loads(&inlined);
    assert_eq!(
        loads.len(),
        1,
        "the scalar argument must not produce a load, got {loads:?}"
    );
    assert_eq!(loads[0].0, "data");
    assert!(
        loads[0].1.contains("LitU32(7)"),
        "the literal argument must appear in the index, got {}",
        loads[0].1
    );
}

/// `BufLen` of a buffer argument is the CALLER buffer's length. Collapsing it
/// to 1, which is correct for a scalar argument, would make every callee that
/// bounds a loop by its input length run exactly one iteration.
#[test]
fn buflen_of_a_buffer_argument_becomes_the_caller_buffers_length() {
    let caller = caller_calling(
        "table_len",
        vec![Expr::BufferRef {
            buffer: "data".into(),
        }],
    );
    let inlined = inline_calls_with_resolver(&caller, resolver).expect("inline");

    assert_eq!(
        buflens(&inlined),
        vec!["data".to_string()],
        "BufLen must retarget at the caller's buffer, not collapse to a literal"
    );
}

/// The scalar half of the same rule: a scalar argument really is one value,
/// so its length is 1 and no `BufLen` node survives.
#[test]
fn buflen_of_a_scalar_argument_stays_one() {
    let caller = caller_calling("table_len", vec![Expr::u32(7)]);
    let inlined = inline_calls_with_resolver(&caller, resolver).expect("inline");

    assert!(
        buflens(&inlined).is_empty(),
        "a scalar argument has length 1 and must fold, got {:?}",
        buflens(&inlined)
    );
    let dump = format!("{:?}", inlined.entry());
    assert!(
        dump.contains("LitU32(1)"),
        "the folded length literal 1 must appear in the inlined program"
    );
}

/// Rebinding must be TOTAL: every expression that names a buffer follows the
/// argument, not just `Load`. A parameter buffer is declared read-only, so an
/// atomic on one is not something a well-formed callee does, but a missing arm
/// here would leave that access pointed at a name the caller never declared
/// while the loads beside it correctly moved. Cover the arm so no buffer-naming
/// expression can be forgotten.
#[test]
fn atomic_on_a_buffer_argument_retargets_to_the_caller_buffer() {
    let caller = caller_calling(
        "table_bump",
        vec![Expr::BufferRef {
            buffer: "data".into(),
        }],
    );
    let inlined = inline_calls_with_resolver(&caller, resolver).expect("inline");

    assert_eq!(
        atomic_buffers(&inlined),
        vec!["data".to_string()],
        "the atomic must retarget at the caller's buffer"
    );
}

/// A buffer reference is not a value. Anywhere except a call argument it has
/// no type and nothing can consume it, so the validator must reject it rather
/// than let it reach a backend.
#[test]
fn a_buffer_reference_outside_a_call_argument_is_rejected() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("data", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::BufferRef {
                buffer: "data".into(),
            },
        )],
    );
    let report = vyre_foundation::validate::validate(&program);
    assert!(
        report.iter().any(|e| e.to_string().contains("V051")),
        "storing a buffer reference must raise V051, got {:?}",
        report
    );
}
