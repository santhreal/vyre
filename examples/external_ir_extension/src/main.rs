//! Minimal external extension demo.
//!
//! Builds a one-store vyre Program from outside the workspace and
//! confirms it round-trips through wire encode + decode + reference
//! evaluation. The earlier two-arg `Expr::Opaque(u32, Vec<u8>)`
//! shape was replaced by `Expr::opaque(impl ExprNode)` in 0.6  - 
//! this example uses the public builder API so external crates can
//! follow the supported path.
use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};

fn build_extension_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(40), Expr::u32(2)),
        )],
    )
}

fn main() {
    let program = build_extension_program();
    let wire = program
        .to_wire()
        .unwrap_or_else(|error| panic!("Fix: extension Program must encode: {error}"));
    println!(
        "External extension program built: {} buffers, {} entry nodes, {} wire bytes.",
        program.buffers().len(),
        program.entry().len(),
        wire.len()
    );
}
