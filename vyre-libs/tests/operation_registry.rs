//! Canonical semantic operation registry coverage for library compositions.

use std::collections::BTreeMap;

use vyre_foundation::ir::ProgramGraph;
use vyre_foundation::logical::{LogicalProgramGraph, LOGICAL_ALGORITHM_VERSION};
use vyre_foundation::operation::{OperationRegistry, OperationTier};

#[test]
fn library_fixtures_are_canonical_semantic_registrations() {
    let entries: Vec<_> = vyre_libs::operation_catalog::all_entries().collect();
    let registered_library_ids: Vec<_> = OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Library)
        .map(|entry| entry.id)
        .collect();
    let catalog_ids: Vec<_> = entries.iter().map(|entry| entry.id).collect();
    assert_eq!(
        catalog_ids, registered_library_ids,
        "Fix: the library operation view must include every canonical library registration exactly once"
    );
    assert!(
        !entries.is_empty(),
        "Fix: linked library features must register at least one operation"
    );

    let mut domain_failures = Vec::new();
    for entry in entries {
        assert_eq!(entry.tier, OperationTier::Library, "{}", entry.id);
        assert_eq!(
            OperationRegistry::global()
                .get(entry.id)
                .expect("Fix: fixture view must resolve through the canonical registry")
                .id,
            entry.id
        );
        let program = entry
            .program()
            .expect("Fix: library compositions must provide a neutral Program builder");
        assert_eq!(program.entry_op_id(), Some(entry.id), "{}", entry.id);
        let graph = match ProgramGraph::from_program(entry.id, program) {
            Ok(graph) => graph,
            Err(error) => {
                domain_failures.push(format!("{}: invalid graph: {error}", entry.id));
                continue;
            }
        };
        let logical = match LogicalProgramGraph::validate(&graph, &BTreeMap::new()) {
            Ok(logical) => logical,
            Err(error) => {
                domain_failures.push(format!("{}: {error}", entry.id));
                continue;
            }
        };
        assert_eq!(LOGICAL_ALGORITHM_VERSION, 2);
        assert_eq!(logical.regions().len(), 1, "{}", entry.id);
        let domain = &logical.regions()[0];
        assert_eq!(
            domain.extents.len(),
            domain.index_map.axes.len(),
            "{}",
            entry.id
        );
        assert_eq!(
            domain.extents.len(),
            domain.layout.strides.len(),
            "{}",
            entry.id
        );
        assert!(domain.layout.contiguous, "{}", entry.id);
        assert!(domain.max_points > 0, "{}", entry.id);
        assert!(domain.aliases.inputs_disjoint, "{}", entry.id);
        assert!(domain.aliases.outputs_disjoint, "{}", entry.id);
    }
    assert!(
        domain_failures.is_empty(),
        "Fix: every library composition must close a positive logical domain before search:\n{}",
        domain_failures.join("\n")
    );

    let tolerances = [
        ("vyre-libs::nn::softmax", 1),
        ("vyre-libs::nn::attention", 4),
        ("vyre-libs::nn::gqa_attention", 4),
        ("vyre-libs::nn::layer_norm", 1),
        ("vyre-libs::nn::silu", 1),
        ("vyre-libs::nn::logit_softcap", 2),
        ("vyre-libs::nn::rms_norm", 2),
        ("vyre-libs::nn::rms_norm_linear", 2),
        ("vyre-libs::math::fft::fft_convolve_circular_complex", 4),
        ("vyre-libs::math::linalg::matmul_strassen_2x2", 32),
        ("vyre-libs::optim::newton_schulz_5step", 64),
        ("vyre-libs::optim::ema_apply", 1),
        ("vyre-libs::optim::muoneq_r", 8),
    ];
    for (id, expected) in tolerances {
        assert_eq!(
            OperationRegistry::global()
                .get(id)
                .expect("Fix: tolerance owner must be registered")
                .tolerance(),
            expected,
            "{id}"
        );
    }
    assert!(OperationRegistry::global()
        .get("unknown-operation")
        .is_none());
}

/// Every linked registration is usable through the registry alone.
///
/// WHY: these three assertions lived in `vyre-test-support`, whose own binary
/// links nothing that registers, so they ran against an empty registry and the
/// two tests that called them could not pass. They belong wherever the
/// registrations are, and the roster is the registry itself rather than a
/// hardcoded floor, so a dialect added tomorrow is checked without an edit.
#[test]
fn every_registration_carries_a_version_and_a_way_to_build_itself() {
    let mut checked = 0usize;
    let mut constraint_failures = Vec::new();
    for entry in OperationRegistry::global().iter() {
        assert!(!entry.id.is_empty(), "Fix: a registration has an empty id.");
        assert!(
            entry.semantic_version > 0,
            "Fix: operation `{}` registers semantic_version 0; version a registration from 1.",
            entry.id
        );
        assert!(
            entry.build.is_some() || entry.signature.is_some(),
            "Fix: operation `{}` registers neither a builder nor a signature, so nothing can be done with it through the registry.",
            entry.id
        );
        match entry.schedule_constraints() {
            Ok(constraints) => {
                for (scope, width) in [
                    ("workgroup", constraints.cooperative_width),
                    ("subgroup", constraints.subgroup_width),
                ] {
                    if matches!(
                        width,
                        vyre_foundation::CooperativeWidth::AtLeast(0)
                            | vyre_foundation::CooperativeWidth::Exactly(0)
                    ) {
                        constraint_failures
                            .push(format!("{}: {scope} width records zero", entry.id));
                    }
                }
                if constraints.requires_cooperative_launch
                    && constraints.memory_ordering
                        != Some(vyre_foundation::ir::MemoryOrdering::GridSync)
                {
                    constraint_failures.push(format!(
                        "{}: cooperative launch lacks grid-sync ordering",
                        entry.id
                    ));
                }
            }
            Err(error) => constraint_failures.push(format!("{}: {error}", entry.id)),
        }
        checked += 1;
    }
    assert!(
        constraint_failures.is_empty(),
        "Fix: every operation must expose one compatible neutral schedule constraint decision:\n{}",
        constraint_failures.join("\n")
    );
    assert!(
        checked > 0,
        "Fix: the registry is empty in a binary that links vyre-libs, so inventory submissions are not reaching the link."
    );
}
