//! Soundness-invariant lock: loop_unroll duplicates the loop body once per
//! trip. When the body DECLARES A LOCAL (`let x`), each unrolled copy must get
//! its OWN scope -- otherwise the duplicated `let x` bindings become duplicate
//! siblings in the enclosing sequence ("duplicate sibling let binding", V032).
//! loop_unroll isolates each iteration in a `Node::Block` (the
//! `isolate_iteration_scope` path); this test locks that wrapping by proving the
//! unrolled program (a) still validates and (b) preserves observable output. A
//! regression that flat-splices a Let-bearing body instead of Block-wrapping it
//! turns reference_eval(unrolled) into an Err. This is the scope-extension dual
//! (read_only_load_hoist / loop_licm) in its CLONING form: duplicating a binding
//! into one scope collides exactly as hoisting one into a shared scope does.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_unroll::LoopUnroll;
use vyre_reference::value::Value;

/// `loop i in 0..3 { let x = i + 100; store(out, 0, x); }` -- the body declares
/// `x`, so unrolling produces three copies of `let x`. Each must live in its own
/// iteration scope. out[0] ends as the last iteration's value (2 + 100 = 102).
fn program_with_local_in_unrollable_body() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(3),
            vec![
                Node::let_bind("x", Expr::add(Expr::var("i"), Expr::u32(100))),
                Node::store("out", Expr::u32(0), Expr::var("x")),
            ],
        )],
    )
}

#[test]
fn loop_unroll_isolates_per_iteration_locals() {
    let program = program_with_local_in_unrollable_body();
    let inputs: [Value; 0] = []; // `out` is the only buffer and it is an output.

    let original = vyre_reference::reference_eval(&program, &inputs)
        .expect("original loop is well-scoped and must run");
    assert_eq!(
        original,
        vec![Value::from(102u32.to_le_bytes().to_vec())],
        "out[0] == last iteration's x = 2 + 100"
    );

    let result = LoopUnroll::transform(program);
    assert!(
        result.changed,
        "the small Let-bearing loop must actually unroll (else this test locks nothing)"
    );

    // The unrolled program must STILL validate and run. If loop_unroll
    // flat-spliced the three `let x` copies into the enclosing sequence instead
    // of giving each its own Block scope, they would be duplicate sibling
    // bindings the validator rejects -- reference_eval errors and this panics.
    let after = vyre_reference::reference_eval(&result.program, &inputs).expect(
        "loop_unroll must give each duplicated `let x` its own iteration scope; \
         flat-splicing them produces a duplicate sibling binding (V032)",
    );
    assert_eq!(
        after, original,
        "unrolled program must preserve observable output"
    );
}

/// The same body, wrapped in a `Node::Region`. A region body is walked without
/// a fresh scope frame, so this pins that unrolling a region-bodied loop still
/// validates and still produces the same output. It is coverage, not a
/// regression lock: the pass produced a valid program for this shape before
/// `body_declares_locals` learned about regions as well.
fn program_with_local_inside_a_region_body() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(3),
            vec![Node::Region {
                generator: "vyre.test.unroll_region_scope".into(),
                source_region: None,
                body: std::sync::Arc::new(vec![
                    Node::let_bind("x", Expr::add(Expr::var("i"), Expr::u32(100))),
                    Node::store("out", Expr::u32(0), Expr::var("x")),
                ]),
            }],
        )],
    )
}

#[test]
fn loop_unroll_isolates_locals_declared_inside_a_region_body() {
    let program = program_with_local_inside_a_region_body();
    let inputs: [Value; 0] = [];

    let original = vyre_reference::reference_eval(&program, &inputs)
        .expect("original loop is well-scoped and must run");
    assert_eq!(
        original,
        vec![Value::from(102u32.to_le_bytes().to_vec())],
        "out[0] == last iteration's x = 2 + 100"
    );

    let result = LoopUnroll::transform(program);
    assert!(
        result.changed,
        "the small Let-bearing loop must actually unroll (else this test locks nothing)"
    );

    let after = vyre_reference::reference_eval(&result.program, &inputs).expect(
        "a `let` inside a region body is a sibling binding, so each unrolled \
         iteration needs its own scope (V032)",
    );
    assert_eq!(
        after, original,
        "unrolled program must preserve observable output"
    );
}
