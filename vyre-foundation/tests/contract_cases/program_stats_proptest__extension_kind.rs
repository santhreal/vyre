// ProgramStats cache invariants  -  50 random programs verify every field.
// (allow(dead_code) moved to parent program_stats_proptest.rs)

#[path = "ir_arbitrary.rs"]
mod ir_arbitrary;
#[path = "program_stats_proptest__arb_node.rs"]
mod program_stats_proptest_arb_node;

use ir_arbitrary::*;
use proptest::collection::vec as prop_vec;
use proptest::prelude::*;
use std::sync::Arc;
use vyre_foundation::ir::ProgramStats;
use vyre_foundation::ir::{DataType, Expr, ExprNode, Node, NodeExtension, Program};

// ─── capability constants (mirroring src/ir_inner/model/program/stats.rs) ───
const CAP_SUBGROUP_OPS: u32 = 1 << 0;
const CAP_F16: u32 = 1 << 1;
const CAP_BF16: u32 = 1 << 2;
const CAP_F64: u32 = 1 << 3;
const CAP_ASYNC_DISPATCH: u32 = 1 << 4;
const CAP_INDIRECT_DISPATCH: u32 = 1 << 5;
const CAP_TENSOR_OPS: u32 = 1 << 6;
const CAP_TRAP: u32 = 1 << 7;

// ─── simple opaque test types (no wire-roundtrip needed here) ───
#[derive(Debug)]
struct EchoExpr;

impl ExprNode for EchoExpr {
    fn extension_kind(&self) -> &'static str {
        "test.stats.expr"
    }
    fn debug_identity(&self) -> &str {
        "test-expr"
    }
    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }
    fn cse_safe(&self) -> bool {
        true
    }
    fn stable_fingerprint(&self) -> [u8; 32] {
        [0; 32]
    }
    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct EchoNode;

impl NodeExtension for EchoNode {
    fn extension_kind(&self) -> &'static str {
        "test.stats.node"
    }
    fn debug_identity(&self) -> &str {
        "test-node"
    }
    fn stable_fingerprint(&self) -> [u8; 32] {
        [0; 32]
    }
    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ─── proptest strategies ───

/// The stats suite needs only that an opaque leaf exists so `opaque_count`
/// moves, so its opaque expression carries no payload. Every other generator
/// comes from `ir_arbitrary`.
fn arb_expr() -> BoxedStrategy<Expr> {
    arb_expr_with(Just(Expr::Opaque(Arc::new(EchoExpr))).boxed())
}

fn arb_node_with_depth(depth: u32) -> BoxedStrategy<Node> {
    let leaf = arb_statement_leaf(arb_expr);

    if depth == 0 {
        return leaf;
    }

    let deeper = arb_node_with_depth(depth - 1);

    leaf.prop_recursive(3, 64, 3, move |inner| {
        // The shared control flow carries weight 3 so each of `If`, `Loop` and
        // `Block` keeps the one-eleventh share it had when all eleven arms were
        // written out here.
        prop_oneof![
            3 => arb_control_flow(arb_expr, inner),
            1 => (arb_ident(), prop_vec(deeper.clone(), 0..=3),).prop_map(|(generator, body)| {
                Node::Region {
                    generator: generator.into(),
                    source_region: None,
                    body: Arc::new(body),
                }
            }),
            // Async nodes (affects CAP_ASYNC_DISPATCH)
            1 => (arb_ident(), arb_ident(), arb_expr(), arb_expr(), arb_tag(),).prop_map(
                |(source, destination, offset, size, tag)| Node::AsyncLoad {
                    source: source.into(),
                    destination: destination.into(),
                    offset: Box::new(offset),
                    size: Box::new(size),
                    tag: tag.into(),
                }
            ),
            1 => (arb_ident(), arb_ident(), arb_expr(), arb_expr(), arb_tag(),).prop_map(
                |(source, destination, offset, size, tag)| Node::AsyncStore {
                    source: source.into(),
                    destination: destination.into(),
                    offset: Box::new(offset),
                    size: Box::new(size),
                    tag: tag.into(),
                }
            ),
            1 => arb_tag().prop_map(|tag| Node::AsyncWait { tag: tag.into() }),
            // Indirect dispatch (affects CAP_INDIRECT_DISPATCH)
            1 => (arb_ident(), any::<u64>()).prop_map(|(count_buffer, count_offset)| {
                Node::IndirectDispatch {
                    count_buffer: count_buffer.into(),
                    count_offset,
                }
            }),
            // Trap (affects CAP_TRAP)
            1 => (arb_expr(), arb_tag()).prop_map(|(address, tag)| Node::Trap {
                address: Box::new(address),
                tag: tag.into(),
            }),
            // Resume (no stats effect, completeness)
            1 => arb_tag().prop_map(|tag| Node::Resume { tag: tag.into() }),
            // Opaque node (affects opaque_count)
            1 => Just(Node::Opaque(Arc::new(EchoNode))),
        ]
    })
    .boxed()
}
