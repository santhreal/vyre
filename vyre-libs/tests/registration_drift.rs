//! Canonical operation coverage gate.
//!
//! Every library-owned semantic operation must carry deterministic fixture
//! coverage or an explicit reason that its execution belongs to another
//! product boundary.

use std::collections::HashSet;

use vyre_foundation::operation::{OperationRegistry, OperationTier};

/// Ids declared in the registry whose executable coverage lives in a
/// subsystem-specific test instead of the fixture harness.
/// Every entry must carry a concrete reason.
const EXEMPT_OP_IDS: &[(&str, &str)] = &[
    (
        "core.indirect_dispatch",
        "Runtime-only op  -  exercised end-to-end by runtime dispatch tests, not a fixture harness.",
    ),
    (
        "io.dma_from_nvme",
        "IO op  -  requires NVMe block device; covered by runtime IO tests, not the fixture harness.",
    ),
    (
        "io.write_back_to_nvme",
        "IO op  -  requires NVMe block device; covered by runtime IO tests, not the fixture harness.",
    ),
    (
        "mem.unmap",
        "Memory lifecycle op  -  covered by runtime memory tests, not the fixture harness.",
    ),
    (
        "mem.zerocopy_map",
        "Memory lifecycle op  -  covered by runtime memory tests, not the fixture harness.",
    ),
    // The five per-row phases of typedef annotation. Each is a composite CALLEE:
    // it takes the VAST node table and the source haystack as buffer-reference
    // arguments and the row index as a scalar, so it has no standalone dispatch
    // shape  -  its own buffer declarations are sized for one row purely so the
    // registry can validate it, and inlining retargets every access onto the
    // caller's buffers. A fixture harness dispatches an op on its own, which for
    // these would measure the placeholder shape rather than any real work.
    //
    // They are executed, with real expected values, through their caller:
    // `vyre-libs::parsing::c11_annotate_typedef_names` submits the canonical
    // operation whose expected output comes from
    // `reference_c11_annotate_typedef_names`. Calls inline before lowering, so
    // that fixture runs all five phases. See
    // vyre-libs/src/parsing/c/parse/vast/typedef_ann/row_phases.rs.
    (
        "vyre-libs::parsing::c11_typedef_scope_open_for_row",
        "Composite callee of c11_annotate_typedef_names  -  no standalone dispatch shape; executed through that op's harness fixture.",
    ),
    (
        "vyre-libs::parsing::c11_typedef_visible_name_for_row",
        "Composite callee of c11_annotate_typedef_names  -  no standalone dispatch shape; executed through that op's harness fixture.",
    ),
    (
        "vyre-libs::parsing::c11_typedef_visible_name_for_row_packed_haystack",
        "Composite callee of c11_annotate_typedef_names  -  no standalone dispatch shape; executed through that op's harness fixture.",
    ),
    (
        "vyre-libs::parsing::c11_typedef_decl_kind_for_row",
        "Composite callee of c11_annotate_typedef_names  -  no standalone dispatch shape; executed through that op's harness fixture.",
    ),
    (
        "vyre-libs::parsing::c11_typedef_decl_kind_for_row_packed_haystack",
        "Composite callee of c11_annotate_typedef_names  -  no standalone dispatch shape; executed through that op's harness fixture.",
    ),
];

#[test]
fn every_library_operation_has_fixture_coverage() {
    let registry = OperationRegistry::global();
    let exemptions: std::collections::HashMap<&str, &str> = EXEMPT_OP_IDS.iter().copied().collect();

    let tested: HashSet<&'static str> = vyre_libs::operation_catalog::fixture_entries()
        .map(|entry| entry.id)
        .collect();

    let mut drift: Vec<String> = Vec::new();
    for operation in registry.iter().filter(|operation| {
        matches!(
            operation.tier,
            OperationTier::Library | OperationTier::Runtime
        )
    }) {
        if tested.contains(operation.id) {
            continue;
        }
        if exemptions.contains_key(operation.id) {
            continue;
        }
        drift.push(operation.id.to_string());
    }

    if !drift.is_empty() {
        drift.sort();
        let mut rendered =
            String::from("registration drift: semantic operation has no fixture coverage.\n");
        for id in &drift {
            rendered.push_str(&format!("  - {id}\n"));
        }
        rendered.push_str(
            "Fix: add deterministic fixtures to the canonical OperationRegistration or record the owner-specific execution boundary in EXEMPT_OP_IDS.\n",
        );
        panic!("{rendered}");
    }
}
