//! Verifies metadata and provenance preservation for canonical program tagging.

use vyre_foundation::composition::{mark_self_exclusive_region, tag_program};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

fn primitive_program(entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [64, 1, 1],
        entry,
    )
    .with_entry_op_id("vyre-primitives::test::primitive")
}

/// This test prevents Cat-A tagging from rebuilding a Program and silently dropping its certified id, launch geometry, or buffer contract.
#[test]
fn tag_program_preserves_program_metadata_and_child_generator() {
    let primitive = primitive_program(vec![Node::Region {
        generator: "vyre-primitives::test::primitive".into(),
        source_region: None,
        body: vec![Node::store("out", Expr::u32(0), Expr::u32(7))].into(),
    }]);

    let tagged = tag_program("vyre-libs::test::consumer", primitive);

    assert_eq!(
        tagged.entry_op_id(),
        Some("vyre-primitives::test::primitive")
    );
    assert_eq!(tagged.workgroup_size(), [64, 1, 1]);
    assert_eq!(tagged.buffers().len(), 1);
    assert_eq!(tagged.buffers()[0].name(), "out");
    let [Node::Region {
        generator,
        source_region: None,
        body,
    }] = tagged.entry()
    else {
        panic!("expected one Cat-A parent region");
    };
    assert_eq!(generator.as_ref(), "vyre-libs::test::consumer");
    let [Node::Region {
        generator,
        source_region: Some(parent),
        ..
    }] = body.as_slice()
    else {
        panic!("expected one reparented primitive child region");
    };
    assert_eq!(generator.as_ref(), "vyre-primitives::test::primitive");
    assert_eq!(parent.as_str(), "vyre-libs::test::consumer");
}

/// This test prevents tagging from losing the self-composition exclusion that protects stateful bodies during fusion.
#[test]
fn tag_program_marks_self_exclusive_parent_without_clearing_flag() {
    let primitive = primitive_program(vec![Node::Return]).with_non_composable_with_self(true);

    let tagged = tag_program("vyre-libs::test::stateful", primitive);

    assert!(tagged.is_non_composable_with_self());
    let [Node::Region { generator, .. }] = tagged.entry() else {
        panic!("expected one tagged parent region");
    };
    assert_eq!(
        generator.as_ref(),
        mark_self_exclusive_region("vyre-libs::test::stateful")
    );
}

/// This test locks out the former disagreement between tagging implementations when a runnable root region must become an inline child.
#[test]
fn tag_program_reparents_generated_root_region_to_inline_child() {
    let primitive = primitive_program(vec![Node::store("out", Expr::u32(0), Expr::u32(11))]);

    let tagged = tag_program("vyre-libs::test::inline_parent", primitive);

    let [Node::Region { body, .. }] = tagged.entry() else {
        panic!("expected one tagged parent region");
    };
    let [Node::Region {
        generator,
        source_region: Some(parent),
        ..
    }] = body.as_slice()
    else {
        panic!("expected one inline child region");
    };
    assert_eq!(generator.as_ref(), "inline::vyre-libs::test::inline_parent");
    assert_eq!(parent.as_str(), "vyre-libs::test::inline_parent");
}

/// This test keeps composition from allocating one copy of the parent name per
/// reparented child. The name is interned, so every child cites the same
/// allocation; a wrapper that owns its own `String` reads identical under
/// equality while costing an allocation per child and per pass that rebuilds
/// the region, which equality assertions cannot see.
#[test]
fn every_reparented_child_shares_one_parent_name_allocation() {
    let primitive = primitive_program(
        (0..3u32)
            .map(|slot| Node::Region {
                generator: "vyre-primitives::test::primitive".into(),
                source_region: None,
                body: vec![Node::store("out", Expr::u32(slot), Expr::u32(slot + 1))].into(),
            })
            .collect(),
    );

    let tagged = tag_program("vyre-libs::test::shared_parent", primitive);

    let [Node::Region { body, .. }] = tagged.entry() else {
        panic!("expected one tagged parent region");
    };
    let parents: Vec<_> = body
        .iter()
        .map(|node| {
            let Node::Region {
                source_region: Some(parent),
                ..
            } = node
            else {
                panic!("expected every child to cite the composing parent");
            };
            parent.shared_text()
        })
        .collect();
    assert_eq!(parents.len(), 3);
    for parent in &parents[1..] {
        assert!(
            std::sync::Arc::ptr_eq(&parents[0], parent),
            "Fix: reparenting must share the interned parent name, not copy it"
        );
    }
}
