//! Integration test crate for the containing Vyre package.

use super::*;

/// Twenty distinct pass names, enough that the scheduler's lookup tables are
/// exercised by index rather than by a single-entry map.
const STRESS_PASS_NAMES: [&str; 20] = [
    "stress_pass_00",
    "stress_pass_01",
    "stress_pass_02",
    "stress_pass_03",
    "stress_pass_04",
    "stress_pass_05",
    "stress_pass_06",
    "stress_pass_07",
    "stress_pass_08",
    "stress_pass_09",
    "stress_pass_10",
    "stress_pass_11",
    "stress_pass_12",
    "stress_pass_13",
    "stress_pass_14",
    "stress_pass_15",
    "stress_pass_16",
    "stress_pass_17",
    "stress_pass_18",
    "stress_pass_19",
];

#[test]
fn scheduler_lookup_tables_use_static_str_keys() {
    // Structural assertion: pass_index must be FxHashMap<&'static str, usize>.
    fn assert_static_str_map(_: &FxHashMap<&'static str, usize>) {}

    let scheduler = PassScheduler::try_default().expect("Fix: built-in passes must be valid");
    assert_static_str_map(&scheduler.pass_index);

    // N=20 passes: build scheduler, topo-sort runs inside with_passes, then
    // exercise the lookup loop via the metrics runner and direct query methods.
    //
    // The names are literals rather than leaked boxes. A leak is what a
    // `&'static str` key costs, and the free that paid it back reconstituted a
    // `Box` from a shared reference while the scheduler still held that key:
    // a retag the aliasing model rejects over memory a live table points at.
    let names: Vec<&'static str> = STRESS_PASS_NAMES.to_vec();
    let passes: Vec<_> = names
        .iter()
        .map(|&name| {
            ProgramPassKind::new(TestPass {
                metadata: PassMetadata::new(name, &[], &[]),
                changes: false,
            })
        })
        .collect();
    let scheduler20 = PassScheduler::with_passes(passes);
    assert_static_str_map(&scheduler20.pass_index);

    // Lookup phase: run_with_metrics iterates execution_order and checks the
    // indexed dirty flags for every pass.
    let report = scheduler20
        .run_with_metrics(trivial_program())
        .expect("Fix: stress scheduler must run");
    assert_eq!(
        report.passes.len(),
        names.len(),
        "all clean test passes should be considered exactly once before convergence"
    );

    // Direct pass_index lookups via public query API.
    for &name in &names {
        assert!(
            !scheduler20.reaches(name, name),
            "a pass must not reach itself"
        );
        assert!(
            scheduler20.pass_index.contains_key(name),
            "pass_index must contain {name}"
        );
    }
}

#[test]
fn scheduler_preserves_program_identity_when_pass_skips() {
    let program = trivial_program();
    let original_entry = Arc::clone(program.entry_arc());

    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(SkipPass)]);
    let result = scheduler
        .run(program)
        .expect("Fix: scheduler must converge when all passes SKIP");

    assert!(
        Arc::ptr_eq(&original_entry, result.entry_arc()),
        "scheduler must preserve entry Arc identity when a pass returns SKIP; \
         reconcile_runnable_top_level must not allocate a fresh Vec or Arc"
    );
}
/// WHY: An optimizer pass that runs on an already-normalized program must return
/// the input `Program` with its `entry` `Arc` (and buffer table `Arc`) intact
/// rather than rebuilding a fresh `Vec<Node>` + `Program` and discarding cached
/// `OnceLock` memos (`fingerprint`, `stats`, `hash`). Under `PassScheduler`,
/// `snapshot = program.clone()` holds a second strong reference during every
/// pass invocation, so calling `Program::map_entry` on a no-op forces a full
/// `Vec<Node>` deep clone. Asserting `Arc::ptr_eq` on both the engine functions
/// and `ProgramPass` transforms catches any regression back to unconditional
/// `map_entry` / `with_rewritten_entry` reconstruction.
#[test]
fn unchanged_passes_preserve_entry_and_buffer_arc_identity() {
    use crate::ir::{BufferAccess, CollectiveOp, CommGroup};
    use crate::optimizer::passes::algebraic::{canonicalize::Canonicalize, canonicalize_engine};
    use crate::optimizer::passes::cleanup::{
        region_inline::RegionInlinePass, region_inline_engine,
    };
    use crate::optimizer::passes::fusion_cse::{
        cse::{self, CsePass},
        dce::{self, DcePass},
    };
    use crate::optimizer::passes::memory::decode_scan_fuse::{self, DecodeScanFuse};
    use crate::optimizer::AdapterCaps;

    // Canonical, live Let + Store + oversized Region (> 64 nodes so
    // RegionInlinePass::analyze returns RUN while transform leaves it intact) +
    // ReadWrite buffer referenced by AllReduce (so DecodeScanFuse::run scans
    // cross-workgroup uses and leaves buffers untouched).
    let mut large_region_body = Vec::with_capacity(66);
    for idx in 0..65u32 {
        large_region_body.push(Node::store(
            "out",
            Expr::u32(idx),
            Expr::add(Expr::gid_x(), Expr::u32(idx + 1)),
        ));
    }
    let program = Program::new_raw(
        vec![
            BufferDecl::read_write("handoff", 0, DataType::U32).with_count(16),
            BufferDecl::written("out", 1, BufferAccess::WriteOnly, DataType::U32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("live_sum", Expr::add(Expr::gid_x(), Expr::u32(1))),
            Node::store("out", Expr::u32(65), Expr::var("live_sum")),
            Node::AllReduce {
                buffer: "handoff".into(),
                op: CollectiveOp::Sum,
                group: CommGroup::WORLD,
            },
            Node::Region {
                generator: "oversized".into(),
                source_region: None,
                body: Arc::new(large_region_body),
            },
        ],
    );
    let entry_arc = Arc::clone(program.entry_arc());

    let canon_borrowed = canonicalize_engine::run_borrowed(&program);
    assert!(Arc::ptr_eq(&entry_arc, canon_borrowed.entry_arc()));
    let canon_res = Canonicalize::transform(program.clone());
    assert!(!canon_res.changed);
    assert!(Arc::ptr_eq(&entry_arc, canon_res.program.entry_arc()));

    let inline_res = RegionInlinePass::transform(program.clone());
    assert!(!inline_res.changed);
    assert!(Arc::ptr_eq(&entry_arc, inline_res.program.entry_arc()));
    let inline_out = region_inline_engine::run(program.clone());
    assert!(Arc::ptr_eq(&entry_arc, inline_out.entry_arc()));

    let cse_res = CsePass::transform(program.clone());
    assert!(!cse_res.changed);
    assert!(Arc::ptr_eq(&entry_arc, cse_res.program.entry_arc()));
    let cse_out = cse::cse(program.clone());
    assert!(Arc::ptr_eq(&entry_arc, cse_out.entry_arc()));

    let dce_res = DcePass::transform(program.clone());
    assert!(!dce_res.changed);
    assert!(Arc::ptr_eq(&entry_arc, dce_res.program.entry_arc()));
    let dce_out = dce::dce(program.clone());
    assert!(Arc::ptr_eq(&entry_arc, dce_out.entry_arc()));

    let caps = AdapterCaps::conservative();
    let dsf_res = DecodeScanFuse::transform_for_adapter(program.clone(), &caps);
    assert!(!dsf_res.changed);
    assert!(Arc::ptr_eq(&entry_arc, dsf_res.program.entry_arc()));
    let dsf_out = decode_scan_fuse::run(program, &caps);
    assert_eq!(
        dsf_out.buffers().as_ptr(),
        canon_res.program.buffers().as_ptr()
    );
}
