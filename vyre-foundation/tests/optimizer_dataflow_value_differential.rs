//! Oracle-differential hunt over MULTI-STATEMENT dataflow transforms at varying
//! input (CSE, copy-propagation, dead-store elimination, store-to-load
//! forwarding, branch-value hoisting).
//!
//! `optimizer_value_dependent_reference_parity` covers single-store scalar
//! expressions; `optimizer_idempotence_proptest` covers multi-statement Let
//! chains but pins every runtime value to `gid_x == 0`. Neither drives the
//! cross-statement dataflow passes, which forward a stored value to a later
//! load, dedup a repeated subexpression, drop an overwritten store, or hoist a
//! branch-invariant value, at a VARYING runtime input. Those passes are
//! exactly where a value bug is most damaging (they move/erase computation), and
//! the store-to-load-forward reassignment guard in particular is only meaningful
//! when the forwarded value actually depends on the input. This loads a runtime
//! `u32` and asserts `reference_eval(base) == reference_eval(optimize(base))`
//! across adversarial input values for each shape.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::pre_lowering as optimize;
use vyre_reference::value::Value;

fn in_i() -> Expr {
    Expr::load("in", Expr::u32(0))
}

const PROBES: &[u32] = &[
    0,
    1,
    2,
    3,
    7,
    8,
    255,
    0x5555_5555,
    0xAAAA_AAAA,
    0x8000_0000,
    0x7FFF_FFFF,
    0xFFFF_FFFF,
];

/// `out` (output, binding 0) + `in` (ReadOnly, binding 1). Input maps to `in`.
fn out_in(body: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("in", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        body,
    )
}

/// `out` (output) + `in` (ReadOnly) + `scratch` (ReadWrite, binding 2). Inputs
/// map to the non-output buffers in declaration order: `[in, scratch]`.
fn out_in_scratch(body: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("in", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("scratch", 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        body,
    )
}

/// Straight-line / branch shapes that read only `in`.
fn single_input_programs() -> Vec<(&'static str, Program)> {
    // CSE: let a = in*3+7; let b = in*3+7; out[0] = a + b   (== 2*(in*3+7))
    let cse = out_in(vec![
        Node::let_bind(
            "a",
            Expr::add(Expr::mul(in_i(), Expr::u32(3)), Expr::u32(7)),
        ),
        Node::let_bind(
            "b",
            Expr::add(Expr::mul(in_i(), Expr::u32(3)), Expr::u32(7)),
        ),
        Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::var("a"), Expr::var("b")),
        ),
    ]);

    // Copy-prop: let a = in; let b = a; let c = b; out[0] = c << 2
    let copy_prop = out_in(vec![
        Node::let_bind("a", in_i()),
        Node::let_bind("b", Expr::var("a")),
        Node::let_bind("c", Expr::var("b")),
        Node::store("out", Expr::u32(0), Expr::shl(Expr::var("c"), Expr::u32(2))),
    ]);

    // Dead store: out[0] = 0xDEADBEEF; out[0] = in + 1   (first store is dead)
    let dead_store = out_in(vec![
        Node::store("out", Expr::u32(0), Expr::u32(0xDEAD_BEEF)),
        Node::store("out", Expr::u32(0), Expr::add(in_i(), Expr::u32(1))),
    ]);

    // Redundant load / load CSE: let a = in; let b = in; out[0] = a*7 + b*3 (== in*10)
    let redundant_load = out_in(vec![
        Node::let_bind("a", in_i()),
        Node::let_bind("b", in_i()),
        Node::store(
            "out",
            Expr::u32(0),
            Expr::add(
                Expr::mul(Expr::var("a"), Expr::u32(7)),
                Expr::mul(Expr::var("b"), Expr::u32(3)),
            ),
        ),
    ]);

    // Branch-value hoist: both arms store the SAME value, so out == in+5 for
    // every input regardless of which arm the condition selects.
    let branch_hoist = out_in(vec![Node::if_then_else(
        Expr::bitand(in_i(), Expr::u32(1)),
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(in_i(), Expr::u32(5)),
        )],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(in_i(), Expr::u32(5)),
        )],
    )]);

    // Branch with input-dependent arms (not hoistable to one value, but
    // branch-collapse / value-hoist must preserve per-input semantics).
    let branch_distinct = out_in(vec![Node::if_then_else(
        Expr::bitand(in_i(), Expr::u32(1)),
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::mul(in_i(), Expr::u32(2)),
        )],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(in_i(), Expr::u32(100)),
        )],
    )]);

    vec![
        ("cse", cse),
        ("copy_prop", copy_prop),
        ("dead_store", dead_store),
        ("redundant_load", redundant_load),
        ("branch_hoist", branch_hoist),
        ("branch_distinct", branch_distinct),
    ]
}

/// Store-to-load-forward shapes that also use a `scratch` ReadWrite buffer.
fn scratch_programs() -> Vec<(&'static str, Program)> {
    // Valid forward: store(scratch, in^magic); x = load(scratch); out = x + 1.
    // out == (in ^ magic) + 1.
    let forward_valid = out_in_scratch(vec![
        Node::let_bind("v", Expr::bitxor(in_i(), Expr::u32(0x9E37_79B9))),
        Node::store("scratch", Expr::u32(0), Expr::var("v")),
        Node::let_bind("x", Expr::load("scratch", Expr::u32(0))),
        Node::store("out", Expr::u32(0), Expr::add(Expr::var("x"), Expr::u32(1))),
    ]);

    // Reassignment guard (dynamic): t = in; store(scratch, t); t := in ^ ~0;
    // x = load(scratch); out = x. The load observes the STORED `in`, so out
    // must equal `in`: NOT the reassigned `t` (in ^ 0xFFFFFFFF). Forwarding `t`
    // would write the reassigned value: a miscompile for any input with set
    // low bits.
    let forward_reassign_guard = out_in_scratch(vec![
        Node::let_bind("t", in_i()),
        Node::store("scratch", Expr::u32(0), Expr::var("t")),
        Node::assign("t", Expr::bitxor(in_i(), Expr::u32(0xFFFF_FFFF))),
        Node::let_bind("x", Expr::load("scratch", Expr::u32(0))),
        Node::store("out", Expr::u32(0), Expr::var("x")),
    ]);

    vec![
        ("forward_valid", forward_valid),
        ("forward_reassign_guard", forward_reassign_guard),
    ]
}

/// Sanity: the oracle computes real values. CSE program at in=10 == 2*(10*3+7) == 74.
#[test]
fn cse_oracle_computes_the_real_value() {
    let (_, cse) = single_input_programs()
        .into_iter()
        .find(|(name, _)| *name == "cse")
        .expect("cse program present");
    let out = vyre_reference::reference_eval(&cse, &[Value::U32(10)])
        .expect("cse program must run on the reference oracle");
    assert_eq!(
        out[0],
        Value::from(74u32.to_le_bytes().to_vec()),
        "out == 2*(10*3+7) == 74"
    );
}

#[test]
fn full_optimize_preserves_single_input_dataflow_value() {
    for (name, program) in single_input_programs() {
        let optimized = optimize::optimize(program.clone());
        for &v in PROBES {
            let inputs = [Value::U32(v)];
            let base = vyre_reference::reference_eval(&program, &inputs)
                .unwrap_or_else(|e| panic!("base `{name}` must run on the oracle: {e}"));
            let opt = vyre_reference::reference_eval(&optimized, &inputs)
                .unwrap_or_else(|e| panic!("optimized `{name}` must run on the oracle: {e}"));
            assert_eq!(
                base, opt,
                "optimize::optimize changed the observable result of `{name}` at in[0]={v:#010x}"
            );
        }
    }
}

#[test]
fn full_optimize_preserves_store_forward_value() {
    for (name, program) in scratch_programs() {
        let optimized = optimize::optimize(program.clone());
        for &v in PROBES {
            // [in, scratch] in declaration order; scratch is overwritten before
            // read, so its initial value is irrelevant.
            let inputs = [Value::U32(v), Value::U32(0)];
            let base = vyre_reference::reference_eval(&program, &inputs)
                .unwrap_or_else(|e| panic!("base `{name}` must run on the oracle: {e}"));
            let opt = vyre_reference::reference_eval(&optimized, &inputs)
                .unwrap_or_else(|e| panic!("optimized `{name}` must run on the oracle: {e}"));
            assert_eq!(
                base, opt,
                "optimize::optimize changed the observable result of `{name}` at in[0]={v:#010x}"
            );
        }
    }
}
