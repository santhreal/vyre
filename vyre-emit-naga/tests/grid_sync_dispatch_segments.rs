//! Proves the dispatch-segment cut a whole-grid fence names.
//!
//! WHY: a whole-grid fence used to be a flat refusal in this emitter, so every
//! fused program carrying one failed to emit for Metal and wgpu. No shading
//! language has a whole-grid barrier, but the fence is still a schedule fact:
//! it is a launch boundary, and the descriptor lowers to one compute entry
//! point per segment submitted in order. The defect class this closes is a
//! fence that silently vanishes or silently degrades: a fence dropped without a
//! cut leaves a kernel with no cross-workgroup synchronization while still
//! emitting, which is a wrong answer instead of a refusal.
//!
//! What this does not catch: whether a dispatch layer actually submits the
//! segments in order. That is the driver's contract and the conformance
//! harness's job.

use std::collections::BTreeSet;
use std::path::Path;

use naga::{Literal, Statement};
use vyre_emit_naga::{emit, grid_segment_entry_points, EmitError, GRID_SEGMENT_ENTRY_PREFIX};
use vyre_foundation::ir::{DataType, MemoryOrdering};
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, for_loop, global_rw, if_then, lit, store_global,
};
use vyre_lower::{
    dispatch_segments, DispatchSplitError, KernelDescriptor, KernelOpKind, LiteralValue,
    NestedBodyControl,
};
use vyre_test_support::monorepo::vyre_workspace_root;

use crate::naga_probe::count_statements;

fn fence() -> vyre_lower::KernelOp {
    effect(
        KernelOpKind::Barrier {
            ordering: MemoryOrdering::GridSync,
        },
        [],
    )
}

/// Three stores separated by two dispatch-level fences.
///
/// Every store addresses index result 0, which only the first segment defines,
/// so a later segment is correct only if the defining literal is recomputed in
/// it. A launch boundary ends every register; forwarding one is not an option.
fn three_segment_descriptor() -> KernelDescriptor {
    descriptor("grid_sync_three_segments")
        .slot(global_rw(0, DataType::U32, "out"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(11),
                    LiteralValue::U32(22),
                    LiteralValue::U32(33),
                ])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(store_global(0, 0, 1))
                .op(fence())
                .op(lit(2, 2))
                .op(store_global(0, 0, 2))
                .op(fence())
                .op(lit(3, 3))
                .op(store_global(0, 0, 3)),
        )
        .build()
}

fn store_values(function: &naga::Function) -> Vec<Literal> {
    let mut values = Vec::new();
    collect_store_values(&function.body, function, &mut values);
    values
}

fn collect_store_values(block: &naga::Block, function: &naga::Function, out: &mut Vec<Literal>) {
    for statement in block.iter() {
        match statement {
            Statement::Store { value, .. } => {
                if let naga::Expression::Literal(literal) = function.expressions[*value] {
                    out.push(literal);
                }
            }
            Statement::Block(inner) => collect_store_values(inner, function, out),
            Statement::If { accept, reject, .. } => {
                collect_store_values(accept, function, out);
                collect_store_values(reject, function, out);
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_store_values(body, function, out);
                collect_store_values(continuing, function, out);
            }
            _ => {}
        }
    }
}

#[test]
fn a_dispatch_level_fence_emits_one_compute_entry_point_per_segment() {
    let desc = three_segment_descriptor();
    let module = emit(&desc).expect("Fix: a dispatch-level whole-grid fence must emit");

    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|ep| ep.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["main", "main_grid_segment_1", "main_grid_segment_2"],
        "Fix: two dispatch-level fences cut the descriptor into three segments, emitted in submission order."
    );
    assert_eq!(
        grid_segment_entry_points(&desc).expect("Fix: segment names must resolve"),
        names,
        "Fix: the entry-point names a caller reads must be the names the module carries."
    );
    for ep in &module.entry_points {
        assert_eq!(
            ep.stage,
            naga::ShaderStage::Compute,
            "Fix: every dispatch segment is a compute entry point."
        );
        assert_eq!(
            ep.workgroup_size, desc.dispatch.workgroup_size,
            "Fix: a segment runs the descriptor's launch geometry, not a narrowed one."
        );
        assert!(
            ep.name == "main" || ep.name.starts_with(GRID_SEGMENT_ENTRY_PREFIX),
            "Fix: a segment name after the first must carry the published prefix so a dispatch layer can resolve it."
        );
    }
}

#[test]
fn each_segment_stores_its_own_value_and_no_barrier_survives_the_cut() {
    let module = emit(&three_segment_descriptor()).expect("Fix: fenced descriptor must emit");
    let expected = [
        Literal::U32(11),
        Literal::U32(22),
        Literal::U32(33),
    ];
    for (index, ep) in module.entry_points.iter().enumerate() {
        assert_eq!(
            store_values(&ep.function),
            vec![expected[index]],
            "Fix: segment {index} must store exactly the value its own ops name."
        );
        assert_eq!(
            count_statements(&ep.function.body, &|statement| matches!(
                statement,
                Statement::Barrier(_)
            )),
            0,
            "Fix: a whole-grid fence becomes a launch boundary, never a barrier instruction inside a segment."
        );
    }
}

#[test]
fn a_fence_free_descriptor_still_emits_exactly_one_entry_point_named_main() {
    let desc = descriptor("no_fence")
        .slot(global_rw(0, DataType::U32, "out"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(store_global(0, 0, 1)),
        )
        .build();
    let module = emit(&desc).expect("Fix: a fence-free descriptor must emit");
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|ep| ep.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["main"],
        "Fix: a descriptor with no whole-grid fence emits the single entry point it emitted before the cut existed."
    );
}

/// A fence placed in a child body of an op imposing `control` on its bodies,
/// paired with the construct name the cut must refuse it under, or `None` when
/// the placement is cuttable.
///
/// The match has no catch-all arm: a new [`NestedBodyControl`] variant fails to
/// compile here until someone states a fence placement for it and whether a
/// launch boundary expresses it.
fn fence_under(control: NestedBodyControl) -> (KernelDescriptor, Option<&'static str>) {
    let fenced_child = body()
        .literals([LiteralValue::U32(0), LiteralValue::U32(5)])
        .op(lit(0, 0))
        .op(lit(1, 1))
        .op(store_global(0, 0, 1))
        .op(fence())
        .op(store_global(0, 0, 0));
    match control {
        NestedBodyControl::Unconditional => (
            descriptor("fence_under_region")
                .slot(global_rw(0, DataType::U32, "out"))
                .dispatch(64, 1, 1)
                .body(
                    body()
                        .op(effect(
                            KernelOpKind::Region {
                                generator: "test::region".into(),
                            },
                            [0],
                        ))
                        .child(fenced_child),
                )
                .build(),
            None,
        ),
        NestedBodyControl::Conditional => (
            descriptor("fence_under_branch")
                .slot(global_rw(0, DataType::U32, "out"))
                .dispatch(64, 1, 1)
                .body(
                    body()
                        .literals([LiteralValue::Bool(true)])
                        .op(lit(0, 9))
                        .op(if_then(9, 0))
                        .child(fenced_child),
                )
                .build(),
            Some("structured branch arm"),
        ),
        NestedBodyControl::Repeated => (
            descriptor("fence_under_loop")
                .slot(global_rw(0, DataType::U32, "out"))
                .dispatch(64, 1, 1)
                .body(
                    body()
                        .literals([LiteralValue::U32(0), LiteralValue::U32(4)])
                        .op(lit(0, 8))
                        .op(lit(1, 9))
                        .op(for_loop("i", 8, 9, 0))
                        .child(fenced_child),
                )
                .build(),
            Some("structured loop body"),
        ),
    }
}

/// Every [`NestedBodyControl`] variant, checked against the enum in source.
fn every_nested_body_control() -> Vec<NestedBodyControl> {
    let tested = vec![
        NestedBodyControl::Unconditional,
        NestedBodyControl::Conditional,
        NestedBodyControl::Repeated,
    ];
    let named: BTreeSet<String> = tested
        .iter()
        .map(|control| format!("{control:?}"))
        .collect();
    let declared = parse_enum_variants(
        &vyre_workspace_root().join("vyre-lower/src/op_facts.rs"),
        "NestedBodyControl",
    );
    assert_eq!(
        named, declared,
        "Fix: every NestedBodyControl variant declared in vyre-lower must have a fence placement here."
    );
    tested
}

fn parse_enum_variants(source_path: &Path, enum_name: &str) -> BTreeSet<String> {
    let content = std::fs::read_to_string(source_path)
        .unwrap_or_else(|error| panic!("Fix: {source_path:?} must be readable: {error}"));
    let header = format!("pub enum {enum_name} {{");
    let mut variants = BTreeSet::new();
    let mut in_enum = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if !in_enum {
            in_enum = trimmed == header;
            continue;
        }
        if trimmed == "}" {
            break;
        }
        if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        let ident = trimmed.split(&['{', '(', ',', ' '][..]).next().unwrap_or("");
        if ident.starts_with(char::is_uppercase) {
            variants.insert(ident.to_owned());
        }
    }
    assert!(
        !variants.is_empty(),
        "Fix: {source_path:?} must declare `{header}` so this test reads the real variant space."
    );
    variants
}

#[test]
fn a_fence_is_cut_only_where_a_launch_boundary_expresses_it() {
    for control in every_nested_body_control() {
        let (desc, refusal) = fence_under(control);
        let result = dispatch_segments(&desc);
        match refusal {
            None => {
                let segments =
                    result.unwrap_or_else(|error| panic!("Fix: {control:?} body is transparent, so its fence is a dispatch-level fence: {error}"));
                assert_eq!(
                    segments.len(),
                    2,
                    "Fix: a fence promoted out of a {control:?} body cuts the descriptor in two."
                );
                let names: Vec<String> = emit(&desc)
                    .expect("Fix: a promoted fence must emit")
                    .entry_points
                    .into_iter()
                    .map(|ep| ep.name)
                    .collect();
                assert_eq!(
                    names,
                    vec!["main".to_owned(), format!("{GRID_SEGMENT_ENTRY_PREFIX}1")],
                    "Fix: a promoted fence emits one entry point per segment."
                );
            }
            Some(construct) => {
                assert_eq!(
                    result.expect_err("Fix: a fence a launch boundary cannot express must be refused, not degraded to a workgroup barrier"),
                    DispatchSplitError::FenceUnderNestedControl { construct },
                    "Fix: the refusal must name the construct the fence could not be promoted out of."
                );
                let error = emit(&desc)
                    .expect_err("Fix: emission must refuse a fence no launch boundary expresses");
                let message = error.to_string();
                assert!(
                    message.contains(construct),
                    "Fix: emission refusal must name `{construct}`, got: {message}"
                );
                assert!(
                    matches!(error, EmitError::InvalidDescriptor(_)),
                    "Fix: an unrepresentable fence placement is a descriptor defect, not a Naga construction failure."
                );
            }
        }
    }
}
