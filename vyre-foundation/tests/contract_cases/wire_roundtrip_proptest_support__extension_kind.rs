// Wire-format round-trip property tests.
// (allow(dead_code) moved to parent wire_roundtrip_proptest.rs)

#[path = "ir_arbitrary.rs"]
mod ir_arbitrary;
#[path = "wire_roundtrip_proptest_support__arb_node.rs"]
mod wire_roundtrip_proptest_support_arb_node;

use ir_arbitrary::*;
use proptest::collection::vec as prop_vec;
use proptest::prelude::*;
use smallvec::smallvec;
use std::sync::Arc;
use vyre_foundation::extension::{OpaqueExprResolver, OpaqueNodeResolver};
use vyre_foundation::ir::{
    AtomicOp, BinOp, BufferDecl, DataType, Expr, ExprNode, Node, NodeExtension, Program, UnOp,
};
use vyre_foundation::MemoryOrdering;
use vyre_spec::data_type::TypeId;
use vyre_spec::extension::ExtensionDataTypeId;

const EXPR_OPAQUE_KIND: &str = "test.wire.expr";
const NODE_OPAQUE_KIND: &str = "test.wire.node";

#[derive(Debug)]
struct TestOpaqueExpr {
    payload: Vec<u8>,
}

impl ExprNode for TestOpaqueExpr {
    fn extension_kind(&self) -> &'static str {
        EXPR_OPAQUE_KIND
    }

    fn debug_identity(&self) -> &str {
        "wire-expr"
    }

    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }

    fn cse_safe(&self) -> bool {
        true
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        *blake3::hash(&self.payload).as_bytes()
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn wire_payload(&self) -> Vec<u8> {
        self.payload.clone()
    }
}

fn deserialize_test_opaque_expr(bytes: &[u8]) -> Result<Arc<dyn ExprNode>, String> {
    Ok(Arc::new(TestOpaqueExpr {
        payload: bytes.to_vec(),
    }))
}

inventory::submit! {
    OpaqueExprResolver {
        kind: EXPR_OPAQUE_KIND,
        deserialize: deserialize_test_opaque_expr,
    }
}

#[derive(Debug)]
struct TestOpaqueNode {
    payload: Vec<u8>,
}

impl NodeExtension for TestOpaqueNode {
    fn extension_kind(&self) -> &'static str {
        NODE_OPAQUE_KIND
    }

    fn debug_identity(&self) -> &str {
        "wire-node"
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        *blake3::hash(&self.payload).as_bytes()
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn wire_payload(&self) -> Vec<u8> {
        self.payload.clone()
    }
}

fn deserialize_test_opaque_node(bytes: &[u8]) -> Result<Arc<dyn NodeExtension>, String> {
    Ok(Arc::new(TestOpaqueNode {
        payload: bytes.to_vec(),
    }))
}

inventory::submit! {
    OpaqueNodeResolver {
        kind: NODE_OPAQUE_KIND,
        deserialize: deserialize_test_opaque_node,
    }
}

/// The wire suite's opaque leaf carries a payload, so the round-trip has
/// something to preserve. Everything else comes from `ir_arbitrary`.
fn arb_expr() -> BoxedStrategy<Expr> {
    arb_expr_with(
        arb_opaque_bytes()
            .prop_map(|payload| Expr::Opaque(Arc::new(TestOpaqueExpr { payload })))
            .boxed(),
    )
}
