//! Tests for V139: async transfer offset and size expressions must be workgroup-uniform.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::validate::validate;

fn codes(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> Vec<String> {
    let program = Program::wrapped(buffers, [16, 1, 1], entry);
    validate(&program)
        .into_iter()
        .map(|error| error.code().to_string())
        .collect()
}

fn reports_v139(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> bool {
    codes(buffers, entry).iter().any(|code| code == "V139")
}

fn standard_buffers() -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
        BufferDecl::storage("dst", 1, BufferAccess::ReadWrite, DataType::U32).with_count(64),
        BufferDecl::storage("rw_buf", 2, BufferAccess::ReadWrite, DataType::U32).with_count(64),
        BufferDecl::storage("ro_buf", 3, BufferAccess::ReadOnly, DataType::U32).with_count(64),
        BufferDecl::uniform("uni_buf", 4, DataType::U32).with_count(64),
    ]
}

#[test]
fn async_load_with_invocation_id_offset_fails_v139() {
    let entry = vec![
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::InvocationId { axis: 0 }),
            size: Box::new(Expr::u32(16)),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), entry),
        "AsyncLoad with InvocationId offset must fail V139"
    );
}

#[test]
fn async_load_with_invocation_id_size_fails_v139() {
    let entry = vec![
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::u32(0)),
            size: Box::new(Expr::InvocationId { axis: 0 }),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), entry),
        "AsyncLoad with InvocationId size must fail V139"
    );
}

#[test]
fn async_store_with_invocation_id_offset_fails_v139() {
    let entry = vec![
        Node::AsyncStore {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::InvocationId { axis: 0 }),
            size: Box::new(Expr::u32(16)),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), entry),
        "AsyncStore with InvocationId offset must fail V139"
    );
}

#[test]
fn async_store_with_invocation_id_size_fails_v139() {
    let entry = vec![
        Node::AsyncStore {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::u32(0)),
            size: Box::new(Expr::InvocationId { axis: 0 }),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), entry),
        "AsyncStore with InvocationId size must fail V139"
    );
}

#[test]
fn async_load_with_divergent_let_binding_fails_v139() {
    let entry = vec![
        Node::let_bind("lane_offset", Expr::InvocationId { axis: 0 }),
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::var("lane_offset")),
            size: Box::new(Expr::u32(16)),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), entry),
        "AsyncLoad with divergent local var offset must fail V139"
    );
}

#[test]
fn async_load_with_read_write_buffer_load_fails_v139() {
    let entry = vec![
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::load("rw_buf", Expr::u32(0))),
            size: Box::new(Expr::u32(16)),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), entry),
        "AsyncLoad with load from ReadWrite buffer must fail V139"
    );
}

#[test]
fn async_load_with_divergent_index_into_read_only_buffer_fails_v139() {
    let entry = vec![
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::load("ro_buf", Expr::InvocationId { axis: 0 })),
            size: Box::new(Expr::u32(16)),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), entry),
        "AsyncLoad with divergent index into ReadOnly buffer must fail V139"
    );
}

#[test]
fn async_load_with_literal_and_workgroup_id_succeeds() {
    let entry = vec![
        Node::let_bind(
            "tile_offset",
            Expr::mul(Expr::WorkgroupId { axis: 0 }, Expr::u32(1024)),
        ),
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::var("tile_offset")),
            size: Box::new(Expr::u32(1024)),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        !reports_v139(standard_buffers(), entry),
        "AsyncLoad with workgroup-uniform offset must pass V139"
    );
}

#[test]
fn async_load_with_read_only_and_uniform_buffer_loads_succeeds() {
    let entry = vec![
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::load("ro_buf", Expr::u32(0))),
            size: Box::new(Expr::load("uni_buf", Expr::WorkgroupId { axis: 0 })),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        !reports_v139(standard_buffers(), entry),
        "AsyncLoad with uniform buffer loads must pass V139"
    );
}

/// WHY: the uniformity proof for a load belongs to the operand, not to a name.
/// `async_load_with_read_only_and_uniform_buffer_loads_succeeds` above pins the
/// direct spelling; this one pins the other half of the asymmetry. `Node::Let`
/// records uniformity through the ordinary analysis, which has no buffer table
/// in scope and so refuses every load, and hoisting a proven-uniform read-only
/// load into a binding for readability silently loses the proof. The rule was
/// untested in that direction, and a registered operation shipped with the
/// hoisted spelling because nothing named it.
///
/// Closes: a read-only load used as a transfer offset through a `Let` binding.
///
/// Does not catch: the same hoist through a buffer store and reload, which is a
/// different expression and is refused for the ordinary reason.
#[test]
fn a_uniform_load_proves_the_operand_it_is_written_in_not_a_binding() {
    let bound = vec![
        Node::let_bind("hoisted_offset", Expr::load("ro_buf", Expr::u32(0))),
        Node::AsyncLoad {
            source: Ident::from("src"),
            destination: Ident::from("dst"),
            offset: Box::new(Expr::var("hoisted_offset")),
            size: Box::new(Expr::u32(16)),
            tag: Ident::from("transfer"),
        },
        Node::AsyncWait {
            tag: Ident::from("transfer"),
        },
    ];
    assert!(
        reports_v139(standard_buffers(), bound),
        "a binding does not carry the load proof, so the transfer offset must name the load itself"
    );
}
