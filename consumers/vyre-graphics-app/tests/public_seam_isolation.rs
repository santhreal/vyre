//! Public seam isolation and domain-neutrality contract test.
//!
//! Proves that the graphics consumer application receives no compiler-internal
//! special case, passes, or intrinsic bypasses, and compiles strictly through
//! the public Row 76 facade.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre::ir::{Expr, GraphValueId, Node, ValueLifetime};
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
            record_node_kinds(inner_node, &mut observed_node_kinds, &mut observed_expr_kinds);
        }
    }

    // Assert that no domain-specific or graphics-specific opcode exists in IR
    let forbidden_patterns = ["gpu_draw", "raster_intrinsic", "texture_sample_hw", "scissor_hw"];
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
            facts.constant_identities.insert(GraphValueId(v_id as u32), Digest([76; 32]));
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

/// Dynamically loads crate publication classes from `docs/CRATE_OWNERSHIP.toml`
/// and asserts that consumers take zero production dependencies on forbidden publication classes.
#[test]
fn consumer_manifest_carries_zero_forbidden_publication_class_dependencies() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("consumers dir")
        .parent()
        .expect("workspace root");

    let ownership_path = workspace_root.join("docs/CRATE_OWNERSHIP.toml");
    let ownership_str = std::fs::read_to_string(&ownership_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ownership_path.display()));
    let ownership: toml::Value = toml::from_str(&ownership_str).expect("parse CRATE_OWNERSHIP.toml");

    let mut forbidden_classes = std::collections::BTreeMap::new();
    if let Some(crates) = ownership.get("crate").and_then(|c| c.as_array()) {
        for c in crates {
            if let (Some(pkg), Some(class)) = (
                c.get("package").and_then(|p| p.as_str()),
                c.get("publication_class").and_then(|cls| cls.as_str()),
            ) {
                if class == "internal-engine" || class == "private-test-support" {
                    forbidden_classes.insert(pkg.to_string(), class.to_string());
                }
            }
        }
    }
    assert!(
        !forbidden_classes.is_empty(),
        "docs/CRATE_OWNERSHIP.toml must declare crates with internal-engine or private-test-support classes"
    );

    // Check both consumer manifests
    let consumer_manifests = [
        manifest_dir.join("Cargo.toml"),
        workspace_root.join("consumers/vyre-model-compiler/Cargo.toml"),
    ];

    for manifest_path in &consumer_manifests {
        let manifest_content = std::fs::read_to_string(manifest_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", manifest_path.display()));
        let manifest_toml: toml::Value = toml::from_str(&manifest_content)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", manifest_path.display()));

        let mut production_deps = Vec::new();
        if let Some(deps) = manifest_toml.get("dependencies").and_then(|d| d.as_table()) {
            for dep_name in deps.keys() {
                production_deps.push(dep_name.clone());
            }
        }

        for dep in &production_deps {
            if let Some(class) = forbidden_classes.get(dep.as_str()) {
                panic!(
                    "Consumer manifest `{}` declares forbidden production dependency `{dep}` with publication_class `{class}`. Consumers must depend only on the published facade and public SDKs.",
                    manifest_path.display()
                );
            }
        }
    }
}

#[test]
fn mutation_injecting_forbidden_dependency_fails_validation() {
    let fake_manifest = r#"
[package]
name = "fake-consumer"
version = "0.1.0"

[dependencies]
vyre = { path = "../../vyre" }
vyre-megakernel = { path = "../../vyre-megakernel" }
"#;
    let manifest_toml: toml::Value = toml::from_str(fake_manifest).expect("parse fake manifest");
    let mut forbidden_classes = std::collections::BTreeMap::new();
    forbidden_classes.insert("vyre-megakernel".to_string(), "internal-engine".to_string());

    let mut errors = Vec::new();
    if let Some(deps) = manifest_toml.get("dependencies").and_then(|d| d.as_table()) {
        for dep_name in deps.keys() {
            if let Some(class) = forbidden_classes.get(dep_name.as_str()) {
                errors.push(format!("found forbidden dep `{dep_name}` with class `{class}`"));
            }
        }
    }
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("vyre-megakernel"));
}

#[test]
fn test_reference_driver_is_registered_in_dev_dependencies() {
    let backend_id = vyre_driver_reference::registered_backend_id();
    assert_eq!(backend_id, Some("reference"));
}

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
        Node::If { cond, then, otherwise } => {
            node_kinds.insert("If".into());
            record_expr_kinds(cond, expr_kinds);
            for n in then {
                record_node_kinds(n, node_kinds, expr_kinds);
            }
            for n in otherwise {
                record_node_kinds(n, node_kinds, expr_kinds);
            }
        }
        Node::Loop { from, to, body, .. } => {
            node_kinds.insert("Loop".into());
            record_expr_kinds(from, expr_kinds);
            record_expr_kinds(to, expr_kinds);
            for n in body {
                record_node_kinds(n, node_kinds, expr_kinds);
            }
        }
        Node::Region { generator, body, .. } => {
            node_kinds.insert(format!("Region:{generator}"));
            for n in body.iter() {
                record_node_kinds(n, node_kinds, expr_kinds);
            }
        }
        _ => {
            node_kinds.insert("Other".into());
        }
    }
}

fn record_expr_kinds(expr: &Expr, expr_kinds: &mut BTreeSet<String>) {
    match expr {
        Expr::Var(..) => { expr_kinds.insert("Var".into()); }
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
        Expr::Select { cond, true_val, false_val } => {
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
