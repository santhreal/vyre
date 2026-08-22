//! The five canonical bundles the pins are taken over, plus the backend
//! witness variant of the region-chain bundle.

use std::sync::Arc;

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_conform_spec::ConformanceCase;
use vyre_primitives::wire::pack_u32_slice as bytes_u32;

use super::test_operation::TEST_IDENTITY_U32_OP;

/// One canonical bundle: the program and the corpus it is certified over.
pub(crate) type BundleBuilderFn = fn() -> (Program, Vec<ConformanceCase>);

// ---------------------------------------------------------------------------
// Bundle 1  -  trivial const
// ---------------------------------------------------------------------------
pub(crate) fn bundle_trivial_const() -> (Program, Vec<ConformanceCase>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let corpus = vec![ConformanceCase {
        name: "tc1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 2  -  1-op add
// ---------------------------------------------------------------------------
pub(crate) fn bundle_one_op_add() -> (Program, Vec<ConformanceCase>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(1), Expr::u32(2)),
        )],
    );
    let corpus = vec![ConformanceCase {
        name: "add1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 3  -  loop-add
// ---------------------------------------------------------------------------
pub(crate) fn bundle_loop_add() -> (Program, Vec<ConformanceCase>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::let_bind("acc", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::assign(
                    "acc",
                    Expr::add(Expr::var("acc"), Expr::var("i")),
                )],
            ),
            Node::store("out", Expr::u32(0), Expr::var("acc")),
        ],
    );
    let corpus = vec![ConformanceCase {
        name: "loop1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 4  -  composed nested regions
// ---------------------------------------------------------------------------
pub(crate) fn bundle_composed_nested() -> (Program, Vec<ConformanceCase>) {
    let inner = vec![Node::store("out", Expr::u32(0), Expr::u32(7))];
    let outer = vec![Node::Region {
        generator: "inner".into(),
        source_region: None,
        body: Arc::new(inner),
    }];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Region {
            generator: "outer".into(),
            source_region: None,
            body: Arc::new(outer),
        }],
    );
    let corpus = vec![ConformanceCase {
        name: "nest1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 5  -  Region-chain with executable dialect op
//
// Contains a Node::Region (intrinsic-like generator) and an Expr::call to a
// operation registry; the bundle certificate hashes remain stable.
// ---------------------------------------------------------------------------
pub(crate) fn bundle_region_chain_intrinsic_dialect() -> (Program, Vec<ConformanceCase>) {
    let body = vec![
        Node::let_bind("acc", Expr::u32(0)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(3),
            vec![Node::assign(
                "acc",
                Expr::add(Expr::var("acc"), Expr::var("i")),
            )],
        ),
        Node::let_bind(
            "dial",
            Expr::call(TEST_IDENTITY_U32_OP, vec![Expr::var("acc")]),
        ),
        Node::store("out", Expr::u32(0), Expr::var("dial")),
    ];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Region {
            generator: "vyre.intrinsics.math.accum".into(),
            source_region: None,
            body: Arc::new(body),
        }],
    );
    let corpus = vec![ConformanceCase {
        name: "rd1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

pub(crate) fn bundle_region_chain_backend_witness() -> (Program, Vec<ConformanceCase>) {
    let body = vec![
        Node::let_bind("acc", Expr::u32(0)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(3),
            vec![Node::assign(
                "acc",
                Expr::add(Expr::var("acc"), Expr::var("i")),
            )],
        ),
        Node::let_bind("dial", Expr::add(Expr::var("acc"), Expr::u32(1))),
        Node::store("out", Expr::u32(0), Expr::var("acc")),
    ];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Region {
            generator: "vyre.intrinsics.math.accum".into(),
            source_region: None,
            body: Arc::new(body),
        }],
    );
    let corpus = vec![ConformanceCase {
        name: "rd-backend".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}
