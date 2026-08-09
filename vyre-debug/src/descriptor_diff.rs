use std::collections::{BTreeMap, HashSet};
use vyre_foundation::ir::Program;
use vyre_lower::KernelDescriptor;

/// Structural differences between two lowered kernel descriptors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DescriptorDiff {
    /// Binding slots present only in the earlier descriptor.
    pub bindings_dropped: Vec<u32>,
    /// Binding slots present only in the later descriptor.
    pub bindings_added: Vec<u32>,
    #[serde(
        serialize_with = "crate::path_map_serde::serialize_i64",
        deserialize_with = "crate::path_map_serde::deserialize_i64"
    )]
    /// Operation-count change for each child-body path.
    pub op_count_delta: BTreeMap<Vec<usize>, i64>,
    /// Whether root operation kinds or count changed.
    pub root_shape_changed: bool,
}

/// Compare binding and operation structure between two descriptors.
pub fn diff_descriptors(before: &KernelDescriptor, after: &KernelDescriptor) -> DescriptorDiff {
    let before_bindings: HashSet<u32> = before.bindings.slots.iter().map(|s| s.slot).collect();
    let after_bindings: HashSet<u32> = after.bindings.slots.iter().map(|s| s.slot).collect();

    let mut dropped: Vec<u32> = before_bindings
        .difference(&after_bindings)
        .copied()
        .collect();
    dropped.sort_unstable();
    let mut added: Vec<u32> = after_bindings
        .difference(&before_bindings)
        .copied()
        .collect();
    added.sort_unstable();

    let mut before_counts = BTreeMap::new();
    let mut after_counts = BTreeMap::new();

    fn walk_counts(
        body: &vyre_lower::KernelBody,
        path: Vec<usize>,
        map: &mut BTreeMap<Vec<usize>, usize>,
    ) {
        map.insert(path.clone(), body.ops.len());
        for (i, child) in body.child_bodies.iter().enumerate() {
            let mut p = path.clone();
            p.push(i);
            walk_counts(child, p, map);
        }
    }

    walk_counts(&before.body, vec![], &mut before_counts);
    walk_counts(&after.body, vec![], &mut after_counts);

    let mut op_count_delta = BTreeMap::new();
    for (path, before_count) in &before_counts {
        let after_count = after_counts.get(path).copied().unwrap_or(0);
        if *before_count != after_count {
            op_count_delta.insert(path.clone(), (after_count as i64) - (*before_count as i64));
        }
    }
    for (path, after_count) in &after_counts {
        if !before_counts.contains_key(path) {
            op_count_delta.insert(path.clone(), *after_count as i64);
        }
    }

    let mut root_shape_changed = false;
    if before.body.ops.len() != after.body.ops.len() {
        root_shape_changed = true;
    } else {
        for (a, b) in before.body.ops.iter().zip(after.body.ops.iter()) {
            if std::mem::discriminant(&a.kind) != std::mem::discriminant(&b.kind) {
                root_shape_changed = true;
                break;
            }
        }
    }

    DescriptorDiff {
        bindings_dropped: dropped,
        bindings_added: added,
        op_count_delta,
        root_shape_changed,
    }
}

/// Verification history produced while applying rewrites one at a time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RewriteBisectResult {
    /// First rewrite after which descriptor verification failed.
    pub first_failing_rewrite: Option<String>,
    /// Verification errors reported at the first failure.
    pub verify_errors: Vec<String>,
    /// Descriptor difference recorded after each applied rewrite.
    pub rewrite_history: Vec<(String, DescriptorDiff)>,
}

/// Apply diagnostic descriptor rewrites after verified lowering and report the
/// first rewrite that violates descriptor invariants.
pub fn bisect_rewrites(program: &Program) -> Result<RewriteBisectResult, String> {
    let mut current = vyre_lower::lower_verified(program)
        .map(|lowered| lowered.descriptor)
        .map_err(|error| error.to_string())?;

    if let Err(errs) = vyre_lower::verify::verify(&current) {
        return Ok(RewriteBisectResult {
            first_failing_rewrite: Some("initial".to_string()),
            verify_errors: errs.iter().map(|e| format!("{:?}", e)).collect(),
            rewrite_history: vec![],
        });
    }

    let mut first_failing_rewrite = None;
    let mut verify_errors = Vec::new();
    let mut rewrite_history = Vec::new();

    let steps: Vec<(&str, fn(&KernelDescriptor) -> KernelDescriptor)> = vec![
        ("strength_reduce", vyre_lower::rewrites::strength_reduce),
        (
            "shared_mem_promote",
            vyre_lower::rewrites::shared_mem_promote,
        ),
        ("bank_conflict_pad", vyre_lower::rewrites::bank_conflict_pad),
        (
            "const_buffer_promote",
            vyre_lower::rewrites::const_buffer_promote,
        ),
        (
            "descriptor_const_fold",
            vyre_lower::rewrites::descriptor_const_fold,
        ),
        ("identity_elim", vyre_lower::rewrites::identity_elim),
        ("branch_collapse", vyre_lower::rewrites::branch_collapse),
        ("loop_unroll", vyre_lower::rewrites::loop_unroll),
        ("licm", vyre_lower::rewrites::licm),
        ("load_forwarding", vyre_lower::rewrites::load_forwarding),
        ("descriptor_dce#1", vyre_lower::rewrites::descriptor_dce),
        ("dead_store", vyre_lower::rewrites::dead_store),
        ("descriptor_dce#2", vyre_lower::rewrites::descriptor_dce),
        ("canonicalize", vyre_lower::rewrites::canonicalize),
        ("descriptor_cse", vyre_lower::rewrites::descriptor_cse),
        (
            "drop_unused_bindings",
            vyre_lower::rewrites::drop_unused_bindings,
        ),
        (
            "drop_unused_literals",
            vyre_lower::rewrites::drop_unused_literals,
        ),
        (
            "drop_unused_child_bodies",
            vyre_lower::rewrites::drop_unused_child_bodies,
        ),
    ];

    for (name, func) in steps {
        let next = func(&current);
        let diff = diff_descriptors(&current, &next);
        rewrite_history.push((name.to_string(), diff));

        if let Err(errs) = vyre_lower::verify::verify(&next) {
            first_failing_rewrite = Some(name.to_string());
            verify_errors = errs.iter().map(|e| format!("{:?}", e)).collect();
            break;
        }

        current = next;
    }

    Ok(RewriteBisectResult {
        first_failing_rewrite,
        verify_errors,
        rewrite_history,
    })
}
