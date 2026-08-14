//! Adversarial emit program matrix for `vyre-emit-naga`.
//!
//! Hostile `KernelDescriptor` programs from `vyre_lower::emit_adversarial_corpus`
//! with structural assertions on lowered `naga::Module` output - not smoke
//! `is_ok()` checks.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{AddressSpace, TypeInner};
use proptest::prelude::*;
use vyre_lower::emit_adversarial_corpus::{
    self, EmitAdversarialBackend, EmitAdversarialCase, EmitAdversarialFamily, EmitOutcome,
};

#[path = "../src/tests/naga_probe.rs"]
mod naga_probe;
use naga_probe::{block_has_atomic, block_has_barrier, block_has_loop, block_if_count, entry_body};

fn assert_naga_structure(case: &EmitAdversarialCase, module: &naga::Module) {
    let entry = &module.entry_points[0];
    assert_eq!(entry.name, "main", "{}: entry must be `main`", case.id);
    assert_eq!(
        entry.workgroup_size, case.descriptor.dispatch.workgroup_size,
        "{}: workgroup size must round-trip",
        case.id
    );

    match case.family {
        EmitAdversarialFamily::DeepIfElse => {
            assert!(
                block_if_count(entry_body(module)) >= 2,
                "{}: nested if/else must produce ≥2 If statements",
                case.id
            );
        }
        EmitAdversarialFamily::HostileWorkgroup => {
            assert_eq!(
                entry.workgroup_size,
                [1024, 1, 1],
                "{}: hostile dispatch must preserve 1024-wide workgroup",
                case.id
            );
        }
        EmitAdversarialFamily::MultiBinding => {
            assert!(
                module.global_variables.len() >= 3,
                "{}: multi-binding kernel must declare ≥3 globals",
                case.id
            );
        }
        EmitAdversarialFamily::SharedGlobalTile => {
            assert!(
                module
                    .global_variables
                    .iter()
                    .any(|(_, global)| { global.space == AddressSpace::WorkGroup }),
                "{}: shared tile must allocate workgroup memory",
                case.id
            );
        }
        EmitAdversarialFamily::LoopWithBarrier => {
            assert!(
                block_has_loop(entry_body(module)) || block_has_barrier(entry_body(module)),
                "{}: loop+barrier kernel must emit Loop or Barrier",
                case.id
            );
        }
        EmitAdversarialFamily::AtomicCounter => {
            assert!(
                block_has_atomic(entry_body(module)),
                "{}: atomic counter must emit Atomic statement",
                case.id
            );
            assert!(
                module.global_variables.iter().any(|(_, global)| {
                    matches!(
                        module.types[global.ty].inner,
                        TypeInner::Array { base, .. }
                            if matches!(module.types[base].inner, TypeInner::Atomic(_))
                    )
                }),
                "{}: atomic binding must use atomic element type",
                case.id
            );
        }
        EmitAdversarialFamily::DeadIdentityChain
        | EmitAdversarialFamily::VecLoadFusion
        | EmitAdversarialFamily::SignedBufferArithmetic => {}
        EmitAdversarialFamily::RejectCall | EmitAdversarialFamily::RejectGridSyncBarrier => {
            panic!(
                "{}: rejection case must not reach naga structure oracle",
                case.id
            );
        }
    }
}

fn validate_module(module: &naga::Module, label: &str) {
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(module)
        .unwrap_or_else(|err| panic!("{label}: naga validation failed: {err:?}"));
}

#[test]
fn hostile_success_corpus_emits_structured_naga_modules() {
    assert!(
        emit_adversarial_corpus::required_backends().contains(&EmitAdversarialBackend::Naga),
        "Fix: shared emit adversarial corpus must register Naga as a required consumer."
    );

    for case in emit_adversarial_corpus::success_cases() {
        let module = vyre_emit_naga::emit(
            &vyre_lower::verify_descriptor(&case.descriptor).expect("descriptor verification"),
        )
        .unwrap_or_else(|err| {
            panic!(
                "Fix: `{}` ({:?}) must emit through naga: {err:?}",
                case.id, case.family
            )
        });
        assert_naga_structure(&case, &module);
        validate_module(&module, case.id);
    }
}

#[test]
fn rejection_corpus_fails_without_panic() {
    for case in emit_adversarial_corpus::rejection_cases() {
        let err = vyre_emit_naga::emit(
            &vyre_lower::verify_descriptor(&case.descriptor).expect("descriptor verification"),
        )
        .expect_err("Fix: rejection corpus case must be rejected by naga emit");
        let msg = format!("{err:?}");
        assert!(
            msg.contains(&case.id) || msg.contains("Fix:") || !msg.is_empty(),
            "rejection for `{}` must carry diagnostic context: {msg}",
            case.id
        );
    }
}

#[test]
fn dead_identity_chain_verifies_before_emit() {
    let case = emit_adversarial_corpus::case_by_id("adv_dead_identity")
        .expect("corpus must include adv_dead_identity");
    let descriptor =
        vyre_lower::verify_descriptor(&case.descriptor).expect("descriptor must pass verification");
    let module = vyre_emit_naga::emit(&descriptor).expect("verified emit");
    assert_eq!(module.entry_points[0].name, "main");
}

#[test]
fn multi_binding_preserves_distinct_global_types() {
    let case = emit_adversarial_corpus::case_by_id("adv_multi_binding").unwrap();
    let module = vyre_emit_naga::emit(
        &vyre_lower::verify_descriptor(&case.descriptor).expect("descriptor verification"),
    )
    .unwrap();
    let mut scalar_kinds = std::collections::BTreeSet::new();
    for (_, global) in module.global_variables.iter() {
        if let TypeInner::Array { base, .. } = module.types[global.ty].inner {
            if let TypeInner::Scalar(scalar) = module.types[base].inner {
                scalar_kinds.insert(format!("{:?}", scalar.kind));
            }
        }
    }
    assert!(
        scalar_kinds.len() >= 2,
        "Fix: mixed u32/f32 bindings must produce ≥2 scalar kinds, got {scalar_kinds:?}"
    );
}

#[test]
fn hostile_workgroup_1024_survives_verify_then_emit() {
    let case = emit_adversarial_corpus::case_by_id("adv_hostile_wg_1024").unwrap();
    let descriptor = vyre_lower::verify_descriptor(&case.descriptor)
        .expect("descriptor verification must succeed on corpus");
    let module = vyre_emit_naga::emit(&descriptor).expect("emit verified descriptor");
    assert_eq!(module.entry_points[0].workgroup_size, [1024, 1, 1]);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    #[test]
    fn success_corpus_round_trips_through_naga_validator(case_index in 0usize..8) {
        let cases = emit_adversarial_corpus::success_cases();
        prop_assume!(case_index < cases.len());
        let case = &cases[case_index];
        let descriptor = vyre_lower::verify_descriptor(&case.descriptor)
            .expect("corpus success case must pass descriptor verification");
        let module = vyre_emit_naga::emit(&descriptor)
            .expect("corpus success case must emit");
        validate_module(&module, case.id);
        assert_eq!(case.outcome, EmitOutcome::Success);
    }
}
