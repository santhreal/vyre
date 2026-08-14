//! Canonicalization on the shipped release corpus: idempotent, and a semantic
//! no-op.
//!
//! `Program::canonicalized` is the cache key every optimizer fact cache and
//! every compiled-artifact lookup is derived from, and it now reports what it
//! changed instead of rebuilding the tree unconditionally. Two properties make
//! that rewrite safe, and neither is visible from the fixtures alone:
//!
//! * idempotence, because a program whose canonical form is not a fixed point
//!   gets a different key on every pass boundary, so nothing ever hits cache;
//! * semantic transparency, because canonicalization reorders commutative
//!   operands and drops `Block` wrappers, and either one applied to the wrong
//!   shape is a miscompile that the fingerprint would then certify as identical.
//!
//! Both are asserted over the shipped release corpus, which is the same
//! generator the release optimization gate scores, so the shapes here are the
//! shapes production canonicalizes.

use std::collections::BTreeMap;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::corpus::generate_release_corpus;
use vyre_reference::execution::{is_reference_input, is_reference_output};
use vyre_reference::value::Value;

/// Deterministic corpus sample. The corpus cycles its eight families with
/// period eight, and 37 is coprime with eight, so a stride of 37 reaches every
/// family across a spread of seeds without paying for 4096 reference runs.
fn sampled_cases() -> Vec<(String, Program)> {
    let cases = generate_release_corpus();
    let sample: Vec<(String, Program)> = cases
        .into_iter()
        .step_by(37)
        .map(|case| (case.id, case.program))
        .collect();
    assert!(
        sample.len() > 64,
        "corpus sample collapsed to {} cases; the release corpus generator shrank and this test stopped covering anything",
        sample.len()
    );
    sample
}

/// One deterministic array per non-output buffer, in that program's own buffer
/// order.
///
/// Built per program rather than once: canonicalization sorts the buffer table,
/// and the interpreter's input ABI is positional, so feeding both programs one
/// fixed vector would compare different buffer bindings rather than different
/// semantics. Contents are derived from the buffer name and the element index so
/// a dropped or reordered operand shows as a different output instead of
/// cancelling out against uniform data.
fn inputs_for(program: &Program) -> Vec<Value> {
    program
        .buffers()
        .iter()
        .filter(|decl| is_reference_input(decl))
        .map(|decl| {
            let seed = decl.name().bytes().fold(1u32, |acc, byte| {
                acc.wrapping_mul(31).wrapping_add(u32::from(byte))
            });
            Value::Array(
                (0..decl.count().max(1))
                    .map(|index| {
                        Value::U32(seed.wrapping_add(index).wrapping_mul(2_654_435_761) >> 8)
                    })
                    .collect(),
            )
        })
        .collect()
}

/// Returned buffer values keyed by buffer name, so a reordered buffer table is
/// compared by identity instead of by position.
fn named_outputs(program: &Program, values: &[Value]) -> BTreeMap<String, Value> {
    program
        .buffers()
        .iter()
        .filter(|decl| is_reference_output(decl))
        .map(|decl| decl.name().to_owned())
        .zip(values.iter().cloned())
        .collect()
}

#[test]
fn canonicalization_is_idempotent_on_the_release_corpus() {
    for (id, program) in sampled_cases() {
        let once = program.canonicalized();
        let twice = once.canonicalized();
        assert_eq!(
            once.fingerprint(),
            twice.fingerprint(),
            "case `{id}` is not canonical after one application, so its cache key changes on every pass boundary. Fix: make the canonical walk reach a fixed point in one pass."
        );
        let once_bytes = once
            .canonical_wire_bytes()
            .expect("canonical corpus program must encode");
        let twice_bytes = twice
            .canonical_wire_bytes()
            .expect("twice-canonical corpus program must encode");
        assert_eq!(
            once_bytes, twice_bytes,
            "case `{id}` encodes to different canonical bytes on the second application"
        );
        assert_eq!(
            program
                .canonical_wire_bytes()
                .expect("corpus program must encode"),
            once_bytes,
            "case `{id}`: canonical bytes of a program must equal the canonical bytes of its canonical form"
        );
    }
}

#[test]
fn canonicalization_preserves_reference_semantics_on_the_release_corpus() {
    for (id, program) in sampled_cases() {
        let before_values = vyre_reference::reference_eval(&program, &inputs_for(&program))
            .unwrap_or_else(|error| {
                panic!("corpus case `{id}` must run on the reference interpreter: {error:?}")
            });
        let canonical = program.canonicalized();
        let after_values = vyre_reference::reference_eval(&canonical, &inputs_for(&canonical))
            .unwrap_or_else(|error| {
                panic!("canonicalized corpus case `{id}` must still run on the reference interpreter: {error:?}")
            });
        assert_eq!(
            named_outputs(&program, &before_values),
            named_outputs(&canonical, &after_values),
            "case `{id}`: canonicalization changed the program's observable output"
        );
    }
}

/// Shapes the release corpus does not generate but canonicalization rewrites.
///
/// The corpus is straight-line arithmetic, so it never puts a `Block` in front
/// of the walk, and the whole scope question goes untested by it: flattening a
/// `Block` that owns a binding hoists that binding into the enclosing scope,
/// which either collides with a later binding of the same name or changes which
/// value a later read sees. Each fixture below carries a binding block, a
/// transparent block, or a commutative pair inside a body position, and is
/// runnable so the difference shows up as output rather than as shape.
fn scope_sensitive_fixtures() -> Vec<(&'static str, Program)> {
    let buffers = || {
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ]
    };
    let wrapped = |body: Vec<Node>| Program::wrapped(buffers(), [1, 1, 1], body);
    vec![
        (
            "binding-block-then-rebind",
            wrapped(vec![
                Node::block(vec![
                    Node::let_bind(
                        "shadowed",
                        Expr::add(Expr::load("in", Expr::u32(0)), Expr::u32(1)),
                    ),
                    Node::store("out", Expr::u32(0), Expr::var("shadowed")),
                ]),
                Node::let_bind("shadowed", Expr::u32(7)),
                Node::store("out", Expr::u32(1), Expr::var("shadowed")),
            ]),
        ),
        (
            "binding-block-inside-loop",
            wrapped(vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(4),
                vec![
                    Node::block(vec![
                        Node::let_bind("local", Expr::add(Expr::var("i"), Expr::u32(2))),
                        Node::store("out", Expr::var("i"), Expr::var("local")),
                    ]),
                    Node::block(vec![
                        Node::let_bind("local", Expr::u32(9)),
                        Node::store("out", Expr::var("i"), Expr::var("local")),
                    ]),
                ],
            )]),
        ),
        (
            "commutative-pairs-in-branches",
            wrapped(vec![Node::if_then_else(
                Expr::lt(Expr::load("in", Expr::u32(0)), Expr::u32(1_000_000)),
                vec![
                    Node::block(vec![Node::store(
                        "out",
                        Expr::u32(2),
                        Expr::add(Expr::u32(3), Expr::load("in", Expr::u32(1))),
                    )]),
                    Node::store(
                        "out",
                        Expr::u32(3),
                        Expr::mul(Expr::load("in", Expr::u32(2)), Expr::u32(5)),
                    ),
                ],
                vec![Node::store("out", Expr::u32(2), Expr::u32(0))],
            )]),
        ),
    ]
}

#[test]
fn canonicalization_preserves_reference_semantics_on_scope_sensitive_shapes() {
    for (name, program) in scope_sensitive_fixtures() {
        let before_values = vyre_reference::reference_eval(&program, &inputs_for(&program))
            .unwrap_or_else(|error| {
                panic!("fixture `{name}` must run on the reference interpreter: {error:?}")
            });
        let canonical = program.canonicalized();
        let after_values = vyre_reference::reference_eval(&canonical, &inputs_for(&canonical))
            .unwrap_or_else(|error| {
                panic!("canonicalized fixture `{name}` must still run on the reference interpreter, but canonicalization made it invalid: {error:?}")
            });
        assert_eq!(
            named_outputs(&program, &before_values),
            named_outputs(&canonical, &after_values),
            "fixture `{name}`: canonicalization changed the program's observable output"
        );
    }
}
