//! Test: descriptor diff tests.
use vyre_debug::descriptor_diff::diff_descriptors;

#[path = "support/mod.rs"]
mod support;
use support::minimal_program;

#[test]
fn diff_descriptors_identical_returns_empty_diff() {
    let p = minimal_program();
    let desc1 = vyre_lower::lower_verified(&p)
        .map(|lowered| lowered.descriptor)
        .unwrap();
    let desc2 = vyre_lower::lower_verified(&p)
        .map(|lowered| lowered.descriptor)
        .unwrap();
    let diff = diff_descriptors(&desc1, &desc2);
    assert!(diff.bindings_dropped.is_empty());
    assert!(diff.bindings_added.is_empty());
    assert!(diff.op_count_delta.is_empty());
    assert!(!diff.root_shape_changed);
}

#[test]
fn diff_descriptors_after_descriptor_dce_removes_ops() {
    let p = minimal_program();
    let mut desc_before = vyre_lower::lower_verified(&p)
        .map(|lowered| lowered.descriptor)
        .unwrap();
    // Add a dead op manually
    desc_before.body.ops.push(vyre_lower::KernelOp {
        result: Some(999),
        kind: vyre_lower::KernelOpKind::Literal,
        operands: vec![0],
    });
    let mut desc_after = desc_before.clone();
    desc_after.body.ops.pop(); // Remove the op to create a difference

    let diff = diff_descriptors(&desc_before, &desc_after);
    // op_count_delta should have a negative entry for the root path []
    let delta = diff.op_count_delta.get(&vec![]).copied().unwrap_or(0);
    assert!(delta < 0, "Expected negative delta, got {}", delta);
}
