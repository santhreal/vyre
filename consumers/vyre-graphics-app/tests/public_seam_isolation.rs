//! Public seam isolation and domain-neutrality contract test.
//!
//! Proves that the graphics consumer application receives no compiler-internal
//! special case, passes, or intrinsic bypasses, and compiles strictly through
//! the public facade.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre::ir::{Expr, GraphValueId, Node, ValueLifetime};
use vyre::visit::child_bodies;
use vyre_libs::graph_compositions::{
    build_interactive_graphics_pipeline, InteractiveGraphicsPipelineParams,
};
#[test]
fn test_consumer_uses_only_public_seam_and_standard_ir() {
    let params = InteractiveGraphicsPipelineParams {
        width: 16,
        height: 16,
        box_count: 4,
        segment_count: 2,
        stroke_radius: 1,
        stroke_color: 0xFF00_00FF,
        glyph_count: 2,
        atlas_w: 8,
        atlas_h: 8,
        clip_rect: (0, 0, 16, 16),
        patch_w: 4,
        patch_h: 4,
        patch_dest: (0, 0),
    };

    let graph = build_interactive_graphics_pipeline(params)
        .expect("graphics pipeline must construct via public builder");

    // Verify all graph nodes contain only standard IR node variants
    let mut observed_node_kinds = BTreeSet::new();
    let mut observed_expr_kinds = BTreeSet::new();

    for node in graph.nodes() {
        assert!(
            !node.name.is_empty(),
            "node name must be non-empty neutral identifier"
        );
        for inner_node in node.program.entry() {
            record_node_kinds(
                inner_node,
                &mut observed_node_kinds,
                &mut observed_expr_kinds,
            );
        }
    }

    // Assert that no domain-specific or graphics-specific opcode exists in IR
    let forbidden_patterns = [
        "gpu_draw",
        "raster_intrinsic",
        "texture_sample_hw",
        "scissor_hw",
    ];
    for kind in &observed_node_kinds {
        for forbidden in forbidden_patterns {
            assert!(
                !kind.contains(forbidden),
                "IR must remain domain-neutral without graphics intrinsics: {kind}"
            );
        }
    }

    // Compile through the frozen public compiler entry point
    let mut facts = ExternalFacts::new(Digest([76; 32]), std::collections::BTreeMap::new());
    for (v_id, v) in graph.values().iter().enumerate() {
        if v.contract.lifetime == ValueLifetime::Constant {
            facts
                .constant_identities
                .insert(GraphValueId(v_id as u32), Digest([76; 32]));
        }
    }
    let request = CompileRequest::new(
        graph,
        facts,
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 100_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("public compile request must validate");

    let artifact = compile(&request).expect("graph must compile through public compiler");
    assert!(!artifact.to_bytes().unwrap().is_empty());
}

#[test]
fn test_reference_driver_is_registered_in_dev_dependencies() {
    let profile = vyre_driver_reference::target_profile().expect("reference target profile");
    assert_eq!(profile.identity(), "reference-graph");
}

/// Record the kind of `node` and of every node nested under it.
///
/// Children come from `child_bodies`, the single exhaustive owner of which
/// `Node` variants nest, so a new nesting variant is reached here without an
/// arm of its own.
fn record_node_kinds(
    node: &Node,
    node_kinds: &mut BTreeSet<String>,
    expr_kinds: &mut BTreeSet<String>,
) {
    match node {
        Node::Let { value, .. } => {
            node_kinds.insert("Let".into());
            record_expr_kinds(value, expr_kinds);
        }
        Node::Store { index, value, .. } => {
            node_kinds.insert("Store".into());
            record_expr_kinds(index, expr_kinds);
            record_expr_kinds(value, expr_kinds);
        }
        Node::If { cond, .. } => {
            node_kinds.insert("If".into());
            record_expr_kinds(cond, expr_kinds);
        }
        Node::Loop { from, to, .. } => {
            node_kinds.insert("Loop".into());
            record_expr_kinds(from, expr_kinds);
            record_expr_kinds(to, expr_kinds);
        }
        Node::Region { generator, .. } => {
            node_kinds.insert(format!("Region:{generator}"));
        }
        _ => {
            node_kinds.insert("Other".into());
        }
    }
    for child in child_bodies(node).into_iter().flatten() {
        record_node_kinds(child, node_kinds, expr_kinds);
    }
}

fn record_expr_kinds(expr: &Expr, expr_kinds: &mut BTreeSet<String>) {
    match expr {
        Expr::Var(..) => {
            expr_kinds.insert("Var".into());
        }
        Expr::LitU32(..) | Expr::LitI32(..) | Expr::LitF32(..) | Expr::LitBool(..) => {
            expr_kinds.insert("Lit".into());
        }
        Expr::BinOp { op, left, right } => {
            expr_kinds.insert(format!("BinOp:{op:?}"));
            record_expr_kinds(left, expr_kinds);
            record_expr_kinds(right, expr_kinds);
        }
        Expr::UnOp { op, operand } => {
            expr_kinds.insert(format!("UnOp:{op:?}"));
            record_expr_kinds(operand, expr_kinds);
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_kinds.insert("Select".into());
            record_expr_kinds(cond, expr_kinds);
            record_expr_kinds(true_val, expr_kinds);
            record_expr_kinds(false_val, expr_kinds);
        }
        Expr::Load { index, .. } => {
            expr_kinds.insert("Load".into());
            record_expr_kinds(index, expr_kinds);
        }
        _ => {
            expr_kinds.insert("OtherExpr".into());
        }
    }
}
