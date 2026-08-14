use super::*;
use crate::ir::BinOp;
use crate::transform::visit::{any_subexpr, for_each_node};

/// Test: d(x*x)/dx = 2*x for a simple square program.
#[test]
fn grad_simple_square() {
    // Forward: out[i] = x[i] * x[i]
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::output("out", 1, DataType::F32).with_count(4),
        ],
        [64, 1, 1],
        vec![Node::Store {
            buffer: "out".into(),
            index: Expr::InvocationId { axis: 0 },
            value: Expr::mul(
                Expr::Load {
                    buffer: "x".into(),
                    index: Box::new(Expr::InvocationId { axis: 0 }),
                },
                Expr::Load {
                    buffer: "x".into(),
                    index: Box::new(Expr::InvocationId { axis: 0 }),
                },
            ),
        }],
    );

    let result = grad(&program, &["out"], &["x"]);
    assert!(result.is_ok(), "grad should succeed: {:?}", result.err());
    let backward = result.unwrap();

    // The backward program should declare grad_x and grad_out buffers.
    let buf_names: Vec<&str> = backward.buffers().iter().map(|b| b.name()).collect();
    assert!(
        buf_names.contains(&"grad_out"),
        "should have grad_out buffer"
    );
    assert!(buf_names.contains(&"grad_x"), "should have grad_x buffer");
}

/// Test: non-differentiable op returns error.
#[test]
fn grad_bitwise_errors() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Store {
            buffer: "out".into(),
            index: Expr::u32(0),
            value: Expr::BinOp {
                op: BinOp::BitAnd,
                left: Box::new(Expr::Load {
                    buffer: "x".into(),
                    index: Box::new(Expr::u32(0)),
                }),
                right: Box::new(Expr::u32(0xFF)),
            },
        }],
    );

    let result = grad(&program, &["out"], &["x"]);
    assert!(result.is_err());
    match result.unwrap_err() {
        AutodiffError::NotDifferentiable { op, .. } => {
            assert!(op.contains("BitAnd"));
        }
        e => panic!("expected NotDifferentiable, got: {e}"),
    }
}

/// Test: missing buffer name returns error.
#[test]
fn grad_missing_buffer() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![],
    );

    let result = grad(&program, &["nonexistent"], &[]);
    assert!(matches!(result, Err(AutodiffError::BufferNotFound { .. })));
}

/// Test: exp derivative  -  d(exp(x))/dx = exp(x).
#[test]
fn grad_exp() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::output("out", 1, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Store {
            buffer: "out".into(),
            index: Expr::u32(0),
            value: Expr::UnOp {
                op: crate::ir::UnOp::Exp,
                operand: Box::new(Expr::Load {
                    buffer: "x".into(),
                    index: Box::new(Expr::u32(0)),
                }),
            },
        }],
    );

    let backward = grad(&program, &["out"], &["x"]).expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - exp should be differentiable");
    assert!(
        backward.buffers().iter().any(|b| b.name() == "x"),
        "exp backward program must declare an x adjoint buffer"
    );
}

#[test]
fn generated_backward_program_zeroes_gradient_buffers_before_accumulation() {
    for count in [1u32, 2, 3, 8, 31, 32, 127, 1024] {
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(count),
                BufferDecl::storage("w", 1, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(count),
                BufferDecl::output("out", 2, DataType::F32).with_count(count),
            ],
            [64, 1, 1],
            vec![
                Node::let_bind(
                    "xw",
                    Expr::mul(
                        Expr::load("x", Expr::InvocationId { axis: 0 }),
                        Expr::load("w", Expr::InvocationId { axis: 0 }),
                    ),
                ),
                Node::Store {
                    buffer: "out".into(),
                    index: Expr::InvocationId { axis: 0 },
                    value: Expr::add(
                        Expr::var("xw"),
                        Expr::load("x", Expr::InvocationId { axis: 0 }),
                    ),
                },
            ],
        );

        let backward = grad(&program, &["out"], &["x", "w"])
            .expect("Fix: generated differentiable affine-product program must autodiff");
        let flattened = flatten_autodiff_test_nodes(backward.entry());
        let seed_index = flattened
            .iter()
            .position(|node| {
                matches!(
                    node,
                    Node::Store { buffer, value, .. }
                        if buffer.as_str() == "grad_out"
                            && matches!(value, Expr::LitF32(v) if *v == 1.0)
                )
            })
            .expect("Fix: backward program must seed grad_out after clearing gradients");
        let zeroed = flattened[..seed_index]
            .iter()
            .filter_map(|node| match node {
                Node::Store {
                    buffer,
                    value: Expr::LitF32(v),
                    ..
                } if *v == 0.0 => Some(buffer.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
                zeroed,
                vec!["grad_out", "grad_x", "grad_w"],
                "Fix: count={count} backward program must clear every gradient buffer before seeding or accumulating"
            );
    }
}

/// Every node in `nodes` and in every nested body, flattened.
///
/// Descent comes from `transform::visit::for_each_node`, the one owner of which
/// node variants nest. The hand-written match this replaces ended in `_ => {}`,
/// so a node inside a fifth body-bearing variant never reached the list and
/// every assertion driven off it read a shorter program than the one produced.
fn flatten_autodiff_test_nodes(nodes: &[Node]) -> Vec<&Node> {
    let mut out = Vec::new();
    for_each_node(nodes, |node| out.push(node));
    out
}

/// Regression test for VF-LOWER-002: the backward loop body must reference the
/// REVERSED induction variable, not the original forward variable.
///
/// Forward:  for i in 0..N { out[i] = x[i] * w[i] }
/// Backward: for i in 0..N { accumulate into grad_x[N-1-i] and grad_w[N-1-i] }
///
/// Before the fix, adj_body used the raw `i` variable so every backward
/// iteration wrote to the *same* forward index, identical to a non-reversed
/// pass, which is wrong for any asymmetric operation.
///
/// After the fix, every Store inside the backward loop body must index its
/// buffer with an expression of the form `(to - 1) - (i - from)` so that
/// iteration 0 addresses element N-1, iteration 1 addresses N-2, etc.
#[test]
fn backward_loop_body_uses_reversed_index_not_forward_var() {
    // Forward: for i in 0..4 { out[i] = x[i] * w[i] }
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::storage("w", 1, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::output("out", 2, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::Loop {
            var: "i".into(),
            from: Expr::u32(0),
            to: Expr::u32(4),
            body: vec![Node::Store {
                buffer: "out".into(),
                index: Expr::var("i"),
                value: Expr::mul(
                    Expr::load("x", Expr::var("i")),
                    Expr::load("w", Expr::var("i")),
                ),
            }],
        }],
    );

    let backward =
        grad(&program, &["out"], &["x", "w"]).expect("forward loop program must be differentiable");

    // Find the backward Loop node.
    let all_nodes = flatten_autodiff_test_nodes(backward.entry());
    let loop_node = all_nodes.iter().find(|n| matches!(n, Node::Loop { .. }));
    let loop_node = loop_node
        .expect("Fix: backward program for a forward loop must contain a backward Loop node");
    let Node::Loop {
        var: bwd_var,
        from: bwd_from,
        to: bwd_to,
        body: bwd_body,
    } = loop_node
    else {
        unreachable!()
    };

    // Bounds must be preserved (same iteration count).
    assert_eq!(*bwd_from, Expr::LitU32(0), "backward loop must start at 0");
    assert_eq!(*bwd_to, Expr::LitU32(4), "backward loop must end at 4");

    // Every Store inside the backward loop body must NOT use the bare loop
    // variable as its index.  The index must contain a subtraction expression
    // that encodes the reversal `(to-1) - (var - from)`.
    //
    // We verify this by checking that no Store index is simply `Var("i")`:
    // if the raw var appears as a top-level Store index the substitution was
    // not applied and the backward pass would write in forward order.
    fn contains_bare_loop_var(expr: &Expr, var_name: &str) -> bool {
        // Children come from `transform::visit::any_subexpr`, the owner of which
        // expression variants carry operands. The hand-written match this
        // replaces listed three and ended in `_ => false`, so a reversal
        // expression built with `Select`, `Fma` or a cast read as containing no
        // reference to the induction variable at all.
        any_subexpr(
            expr,
            &mut |sub| matches!(sub, Expr::Var(name) if name.as_str() == var_name),
        )
    }

    /// The index of every `Store` in `nodes`, at any nesting depth.
    ///
    /// Descent comes from `transform::visit::for_each_node`, the one owner of
    /// which node variants nest. The hand-written match this replaces ended in
    /// `_ => {}` and had no `Region` arm at all, so a store inside a region was
    /// already invisible and the assertion below passed on a backward body it
    /// had not inspected.
    fn store_indices_in_nodes<'a>(nodes: &'a [Node], out: &mut Vec<&'a Expr>) {
        for_each_node(nodes, |node| {
            if let Node::Store { index, .. } = node {
                out.push(index);
            }
        });
    }

    let mut store_indices: Vec<&Expr> = Vec::new();
    store_indices_in_nodes(bwd_body, &mut store_indices);

    // There must be at least one store in the backward body.
    assert!(
        !store_indices.is_empty(),
        "Fix: backward loop body must contain at least one Store"
    );

    // No store index should be the bare induction variable `bwd_var`.
    for index in &store_indices {
        assert!(
            !matches!(index, Expr::Var(name) if name.as_str() == bwd_var.as_str()),
            "Fix: backward loop body Store index must not be the bare induction variable \
             `{bwd_var}`: that would write in forward order. \
             The index must be the reversed expression `(to-1) - (var - from)`. \
             Got index: {index:?}"
        );
    }

    // Confirm the reversal expression IS present: at least one index must
    // contain a Sub chain that references `bwd_var`.
    let has_reversed_index = store_indices.iter().any(|idx| {
        // reversed = Sub(Sub(to-1), Sub(var, from))  or any variant thereof
        contains_bare_loop_var(idx, bwd_var.as_str())
    });
    assert!(
        has_reversed_index,
        "Fix: backward loop body must reference the induction variable inside a \
         reversal expression (e.g. (to-1)-(var-from)), but no Store index \
         contained any reference to `{bwd_var}`. Indices: {store_indices:?}"
    );
}
