// Wire-format round-trip property tests.
// (allow(dead_code) moved to parent wire_roundtrip_proptest.rs)

#[path = "ir_arbitrary.rs"]
mod ir_arbitrary;
#[path = "wire_roundtrip_proptest_support__arb_node.rs"]
mod wire_roundtrip_proptest_support_arb_node;

use ir_arbitrary::*;
use proptest::prelude::*;
use smallvec::smallvec;
use std::sync::Arc;
use vyre_foundation::ir::{AtomicOp, BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_foundation::MemoryOrdering;
use vyre_spec::extension::ExtensionDataTypeId;
use vyre_spec::TypeId;

#[path = "../support/opaque_echo_extension.rs"]
mod opaque_echo_extension;

pub(crate) use opaque_echo_extension::{EchoExpr, EchoNode};

/// The wire suite's opaque leaf carries a payload, so the round-trip has
/// something to preserve. Everything else comes from `ir_arbitrary`.
fn arb_expr() -> BoxedStrategy<Expr> {
    arb_expr_with(
        arb_opaque_bytes()
            .prop_map(|payload| Expr::Opaque(Arc::new(EchoExpr { payload })))
            .boxed(),
    )
}
