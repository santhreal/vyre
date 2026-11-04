//! Mapping a logical index space onto the lanes of one workgroup.

use vyre_foundation::ir::{Expr, Node};

/// Chunks a workgroup of `tile` lanes needs to cover `count` indices.
#[must_use]
pub const fn chunks(count: u32, tile: u32) -> u32 {
    if tile == 0 {
        return 1;
    }
    let chunks = count.div_ceil(tile);
    if chunks == 0 {
        1
    } else {
        chunks
    }
}

/// A strided walk that offers each index of `0..count` to `body` once.
#[must_use]
pub fn for_each_index(count: u32, tile: u32, var: &str, body: Vec<Node>) -> Node {
    let chunk = format!("{var}_chunk");
    Node::loop_for(
        chunk.clone(),
        Expr::u32(0),
        Expr::u32(chunks(count, tile)),
        vec![
            Node::let_bind(
                var,
                Expr::add(
                    Expr::var("local"),
                    Expr::mul(Expr::var(chunk), Expr::u32(tile)),
                ),
            ),
            Node::if_then(Expr::lt(Expr::var(var), Expr::u32(count)), body),
        ],
    )
}
