//! The GPU-resident e-graph mirror, its packing, and the merge back.
//!
//! The mirror is only useful if it says the same thing as the e-graph it was
//! built from, if the packed image indexes the columns it claims to, and if an
//! equivalence a device reports lands through the same merge the CPU uses.
//! These cases hold all three, including the refusals: an out-of-range e-class
//! id is counted and not applied, and a malformed snapshot is reported rather
//! than packed.
//!
//! Everything read here is public. The one figure that is not, the 32-bit
//! column limit, is proved beside its own function.

use rustc_hash::FxHashMap;
use std::hash::Hash;

use vyre_foundation::optimizer::eqsat::{EChildren, EClassId, EGraph, ENodeLang};
use vyre_foundation::optimizer::eqsat_gpu::{
    apply_equivalences, apply_equivalences_to_egraph, bridge_equivalence_batch_with_report,
    ApplyEquivalencesReport, Equivalence, GpuEGraphDeviceImageError, GpuEGraphSnapshot,
    OpIdRegistry,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum TinyLang {
    Lit(u32),
    Add(EClassId, EClassId),
}

impl ENodeLang for TinyLang {
    fn children(&self) -> EChildren {
        match self {
            Self::Lit(_) => EChildren::new(),
            Self::Add(left, right) => [*left, *right].into_iter().collect(),
        }
    }

    fn with_children(&self, children: &[EClassId]) -> Self {
        match self {
            Self::Lit(value) => Self::Lit(*value),
            Self::Add(_, _) => Self::Add(children[0], children[1]),
        }
    }
}

fn tiny_op_name(node: &TinyLang) -> &'static str {
    match node {
        TinyLang::Lit(_) => "lit",
        TinyLang::Add(_, _) => "add",
    }
}

fn tiny_cost(node: &TinyLang) -> u64 {
    match node {
        TinyLang::Lit(_) => 1,
        TinyLang::Add(_, _) => 4,
    }
}

/// Empty snapshot: zero rows, zero children, registry empty.
#[test]
fn empty_snapshot() {
    let snap = GpuEGraphSnapshot::default();
    assert!(snap.is_empty());
    assert_eq!(snap.node_count(), 0);
    assert_eq!(snap.child_count(), 0);
    assert!(snap.op_ids.is_empty());
}

/// Build a 3-node snapshot via the iterator builder; assert
/// row layout + children column line up.
#[test]
fn build_three_node_snapshot() {
    let snap = GpuEGraphSnapshot::build([
        (0u32, "lit_u32", &[][..]),
        (1u32, "lit_u32", &[][..]),
        (2u32, "binop_add", &[0u32, 1u32][..]),
    ]);
    assert_eq!(snap.node_count(), 3);
    assert_eq!(snap.child_count(), 2);
    let empty: &[u32] = &[];
    assert_eq!(snap.children_of(0), Some(empty));
    assert_eq!(snap.children_of(1), Some(empty));
    assert_eq!(snap.children_of(2), Some(&[0, 1][..]));
    assert_eq!(snap.children_of(99), None);
}

/// `OpIdRegistry::intern` returns the same id for repeated
/// names.
#[test]
fn op_id_intern_dedups() {
    let mut reg = OpIdRegistry::default();
    let a = reg.intern("foo");
    let b = reg.intern("bar");
    let c = reg.intern("foo");
    assert_eq!(a, c);
    assert_ne!(a, b);
    assert_eq!(reg.len(), 2);
    assert_eq!(reg.name_of(a), Some("foo"));
    assert_eq!(reg.name_of(b), Some("bar"));
    assert_eq!(reg.name_of(99), None);
}

/// `rows_by_eclass` groups multi-row e-classes.
#[test]
fn rows_by_eclass_groups_correctly() {
    let snap = GpuEGraphSnapshot::build([
        (0u32, "lit_u32", &[][..]),
        (0u32, "var", &[][..]),
        (1u32, "binop_add", &[0u32][..]),
    ]);
    let groups = snap.rows_by_eclass();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups.get(&0).unwrap().len(), 2);
    assert_eq!(groups.get(&1).unwrap().len(), 1);
}

#[test]
fn generated_snapshot_integrity_accepts_pack_boundaries_and_forward_children() {
    for node_count in [1_usize, 2, 7, 8, 9, 16, 17, 31, 32, 33, 65, 128] {
        let mut rows = Vec::with_capacity(node_count);
        let mut child_storage = Vec::new();
        for row in 0..node_count {
            let start = child_storage.len();
            if row > 0 {
                child_storage.push((row - 1) as u32);
            }
            if row > 1 && row % 3 == 0 {
                child_storage.push((row / 2) as u32);
            }
            rows.push((
                row as u32,
                if row % 2 == 0 { "lit" } else { "add" },
                start,
                child_storage.len() - start,
            ));
        }
        let build_rows = rows
            .iter()
            .map(|&(class, op, start, len)| (class, op, &child_storage[start..start + len]))
            .collect::<Vec<_>>();
        let snapshot = GpuEGraphSnapshot::build(build_rows);

        snapshot
            .validate_integrity()
            .unwrap_or_else(|error| panic!("node_count={node_count}: {error}"));
    }
}

#[test]
fn snapshot_integrity_rejects_unknown_op_id() {
    let mut snapshot = GpuEGraphSnapshot::build([(0u32, "lit", &[][..])]);
    snapshot.rows[0].language_op_id = 99;

    let error = snapshot
        .validate_integrity()
        .expect_err("Fix: malformed GPU snapshot op ids must be rejected before upload.");

    assert_eq!(error.context(), "unknown language_op_id");
    assert_eq!(error.row(), 0);
    assert_eq!(error.value(), 99);
}

#[test]
fn snapshot_integrity_rejects_out_of_bounds_child_range() {
    let mut snapshot = GpuEGraphSnapshot::build([(0u32, "lit", &[][..])]);
    snapshot.rows[0].children_offset = 1;
    snapshot.rows[0].children_len = 1;

    let error = snapshot
        .validate_integrity()
        .expect_err("Fix: malformed GPU snapshot child ranges must be rejected before upload.");

    assert_eq!(error.context(), "children range end");
    assert_eq!(error.row(), 0);
}

#[test]
fn snapshot_integrity_rejects_dangling_child_eclass() {
    let snapshot =
        GpuEGraphSnapshot::build([(0u32, "lit", &[][..]), (1u32, "add", &[0u32, 99u32][..])]);

    let error = snapshot
        .validate_integrity()
        .expect_err("Fix: malformed GPU snapshot child eclasses must be rejected before upload.");

    assert_eq!(error.context(), "dangling child eclass");
    assert_eq!(error.row(), 1);
    assert_eq!(error.value(), 99);
}

#[test]
fn device_image_packs_single_upload_slab_with_sorted_group_index() {
    let snapshot = GpuEGraphSnapshot::build([
        (2u32, "lit", &[][..]),
        (1u32, "lit", &[][..]),
        (2u32, "add", &[1u32, 2u32][..]),
    ]);

    let image = snapshot
        .try_pack_device_image()
        .expect("Fix: valid GPU e-graph snapshot must pack into a device image");
    let layout = image.layout();

    assert_eq!(layout.row_count(), 3);
    assert_eq!(layout.child_count(), 2);
    assert_eq!(layout.eclass_group_count(), 2);
    assert_eq!(image.row_eclass_ids(), &[2, 1, 2]);
    assert_eq!(image.row_language_op_ids(), &[0, 0, 1]);
    assert_eq!(image.row_children_offsets(), &[0, 0, 0]);
    assert_eq!(image.row_children_lens(), &[0, 0, 2]);
    assert_eq!(image.row_signatures().len(), 3);
    assert_ne!(image.row_signatures()[0], image.row_signatures()[2]);
    assert_eq!(image.children(), &[1, 2]);
    assert_eq!(image.group_eclass_ids(), &[1, 2]);
    assert_eq!(image.group_offsets(), &[0, 1, 3]);
    assert_eq!(image.group_rows(), &[1, 0, 2]);
    assert_eq!(
        image.words().len(),
        layout.row_eclass_ids().len()
            + layout.row_language_op_ids().len()
            + layout.row_children_offsets().len()
            + layout.row_children_lens().len()
            + layout.row_signatures().len()
            + layout.children().len()
            + layout.group_eclass_ids().len()
            + layout.group_offsets().len()
            + layout.group_rows().len()
    );
}

#[test]
fn generated_device_image_pack_accepts_empty_and_power_boundaries() {
    for node_count in [0_usize, 1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129] {
        let mut rows = Vec::with_capacity(node_count);
        let mut child_storage = Vec::new();
        for row in 0..node_count {
            let start = child_storage.len();
            if row > 0 {
                child_storage.push((row - 1) as u32);
            }
            rows.push((
                row as u32,
                if row & 1 == 0 { "lit" } else { "neg" },
                start,
                child_storage.len() - start,
            ));
        }
        let build_rows = rows
            .iter()
            .map(|&(class, op, start, len)| (class, op, &child_storage[start..start + len]))
            .collect::<Vec<_>>();
        let snapshot = GpuEGraphSnapshot::build(build_rows);

        let image = snapshot
            .try_pack_device_image()
            .unwrap_or_else(|error| panic!("node_count={node_count}: {error}"));

        assert_eq!(image.layout().row_count(), node_count);
        assert_eq!(image.row_eclass_ids().len(), node_count);
        assert_eq!(image.row_language_op_ids().len(), node_count);
        assert_eq!(image.row_signatures().len(), node_count);
        assert_eq!(image.group_rows().len(), node_count);
        assert_eq!(image.group_offsets().len(), node_count + 1);
    }
}

#[test]
fn row_signatures_group_structural_duplicates_without_eclass_identity() {
    let snapshot = GpuEGraphSnapshot::build([
        (1u32, "lit", &[][..]),
        (2u32, "lit", &[][..]),
        (10u32, "add", &[1u32, 2u32][..]),
        (11u32, "add", &[1u32, 2u32][..]),
        (12u32, "add", &[2u32, 1u32][..]),
        (13u32, "mul", &[1u32, 2u32][..]),
    ]);

    let image = snapshot
        .try_pack_device_image()
        .expect("Fix: valid duplicate-signature snapshot must pack");

    assert_eq!(image.row_signatures()[2], image.row_signatures()[3]);
    assert_ne!(image.row_signatures()[2], image.row_signatures()[4]);
    assert_ne!(image.row_signatures()[2], image.row_signatures()[5]);
}

#[test]
fn device_image_rejects_malformed_snapshot_before_pack() {
    let mut snapshot = GpuEGraphSnapshot::build([(0u32, "lit", &[][..])]);
    snapshot.rows[0].language_op_id = 42;

    let error = snapshot
        .try_pack_device_image()
        .expect_err("Fix: device image packing must reject malformed snapshots");

    match error {
        GpuEGraphDeviceImageError::Integrity(error) => {
            assert_eq!(error.context(), "unknown language_op_id");
            assert_eq!(error.row(), 0);
            assert_eq!(error.value(), 42);
        }
        GpuEGraphDeviceImageError::Layout(error) => {
            panic!("expected integrity error, got layout error: {error}")
        }
    }
}

/// Snapshot directly from the CPU EGraph canonicalizes children and
/// assigns stable operation ids.
#[test]
fn snapshot_from_egraph_uses_canonical_children() {
    let mut egraph = EGraph::new();
    let a = egraph.add(TinyLang::Lit(1));
    let b = egraph.add(TinyLang::Lit(2));
    let add = egraph.add(TinyLang::Add(a, b));
    assert_eq!(add.0, 2);

    let snap = GpuEGraphSnapshot::from_egraph_with(&egraph, |node| match node {
        TinyLang::Lit(_) => "lit",
        TinyLang::Add(_, _) => "add",
    });

    assert_eq!(snap.node_count(), 3);
    assert_eq!(snap.child_count(), 2);
    assert_eq!(snap.op_ids.name_of(0), Some("lit"));
    assert_eq!(snap.op_ids.name_of(1), Some("add"));
    assert_eq!(snap.children_of(2), Some(&[0, 1][..]));
}

/// `apply_equivalences` calls the merger for each equivalence
/// and counts state-changing merges.
#[test]
fn apply_equivalences_counts_state_changes() {
    let equivalences = vec![
        Equivalence { left: 0, right: 1 },
        Equivalence { left: 1, right: 0 }, // no-op (already merged)
        Equivalence { left: 2, right: 3 },
    ];
    let mut canonical: FxHashMap<u32, u32> = FxHashMap::default();
    let applied = apply_equivalences(&equivalences, |a, b| {
        let canon_a = *canonical.get(&a).unwrap_or(&a);
        let canon_b = *canonical.get(&b).unwrap_or(&b);
        if canon_a == canon_b {
            false
        } else {
            let (lo, hi) = if canon_a < canon_b {
                (canon_a, canon_b)
            } else {
                (canon_b, canon_a)
            };
            canonical.insert(hi, lo);
            canonical.insert(a, lo);
            canonical.insert(b, lo);
            true
        }
    });
    assert_eq!(applied, 2);
}

/// Empty equivalence batch is a no-op.
#[test]
fn apply_equivalences_empty_batch() {
    let applied = apply_equivalences(&[], |_, _| true);
    assert_eq!(applied, 0);
}

/// EGraph merge bridge ignores invalid ids and rebuilds after valid
/// merges.
#[test]
fn apply_equivalences_to_egraph_merges_valid_ids() {
    let mut egraph = EGraph::new();
    let a = egraph.add(TinyLang::Lit(1));
    let b = egraph.add(TinyLang::Lit(2));
    let c = egraph.add(TinyLang::Lit(3));
    let report = apply_equivalences_to_egraph(
        &mut egraph,
        &[
            Equivalence {
                left: a.0,
                right: b.0,
            },
            Equivalence {
                left: c.0,
                right: 99,
            },
        ],
    );
    assert_eq!(
        report,
        ApplyEquivalencesReport {
            requested: 2,
            valid: 1,
            merged: 1,
            rebuild_unions: 0,
        }
    );
    assert_eq!(egraph.find(a), egraph.find(b));
    assert_ne!(egraph.find(a), egraph.find(c));
}

#[test]
fn gpu_egraph_bridge_reports_compact_columns_apply_and_extraction_parity() {
    let mut egraph = EGraph::new();
    let one = egraph.add(TinyLang::Lit(1));
    let two = egraph.add(TinyLang::Lit(2));
    let add = egraph.add(TinyLang::Add(one, two));
    let folded = egraph.add(TinyLang::Lit(3));

    let report = bridge_equivalence_batch_with_report(
        &mut egraph,
        add,
        tiny_op_name,
        &[Equivalence {
            left: add.0,
            right: folded.0,
        }],
        tiny_cost,
    )
    .expect("Fix: valid GPU e-graph bridge probe must produce a parity report");

    assert_eq!(report.snapshot_rows, 4);
    assert_eq!(report.snapshot_children, 2);
    assert!(report.device_words > report.snapshot_rows);
    assert_eq!(report.device_eclass_groups, 4);
    assert_eq!(report.equivalences_requested, 1);
    assert_eq!(report.equivalences_valid, 1);
    assert_eq!(report.equivalences_merged, 1);
    assert_eq!(report.cpu_equivalences_valid, 1);
    assert_eq!(report.cpu_equivalences_merged, 1);
    assert_eq!(report.cpu_extraction_cost, Some(1));
    assert_eq!(report.gpu_extraction_cost, Some(1));
    assert!(report.snapshot_ns > 0);
    assert!(report.pack_ns > 0);
    assert!(report.cpu_apply_ns > 0);
    assert!(report.gpu_apply_ns > 0);
    assert!(report.cpu_extraction_ns > 0);
    assert!(report.gpu_extraction_ns > 0);
    assert!(report.recall_parity);
    assert!(report.class_id_deterministic);
}
