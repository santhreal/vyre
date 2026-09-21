// Wire-format round-trip property tests.

#[path = "wire_roundtrip_proptest_support__arb_node.rs"]
mod wire_roundtrip_proptest_support_arb_node;

use crate::ir_arbitrary::*;
use proptest::prelude::*;
use smallvec::smallvec;
use std::sync::Arc;
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{AtomicOp, BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_spec::ExtensionDataTypeId;
use vyre_spec::TypeId;

pub(crate) use crate::opaque_echo_extension::{EchoExpr, EchoNode};

/// The wire suite's opaque leaf carries a payload, so the round-trip has
/// something to preserve. Everything else comes from `ir_arbitrary`.
fn arb_expr() -> BoxedStrategy<Expr> {
    arb_expr_with(
        arb_opaque_bytes()
            .prop_map(|payload| Expr::Opaque(Arc::new(EchoExpr { payload })))
            .boxed(),
    )
}
