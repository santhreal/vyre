//! Choke-point contract for shared target-payload admission.
//!
//! WHY: every concrete driver used to open its payload with its own copy of
//! these checks, and the copies drifted. Two backends demanded an entry point
//! named `main` and two did not; one spelled shared rejections in different
//! text. A payload one backend refused, another ran. `vyre_driver::materialize`
//! is now the single admission decision, so this pins that decision against
//! real compiled payloads, one case per rejection branch.
//!
//! This exercises the neutral choke point through a backend that can produce a
//! genuine payload; it does not test anything SPIR-V specific. The companion
//! structural contract proves no backend has grown a private copy again.
//!
//! Not covered here: native module loading, which needs a device.

use vyre_driver::materialize::{self, MaterializerTarget};
use vyre_driver::BackendError;
use vyre_megakernel::{
    Artifact, TargetEntryPoint, TargetModuleBundle, TargetPayload, TargetPayloadFormat,
    TargetProfile,
};

mod target_artifacts;
use target_artifacts::{foreign_artifact, spirv};

/// A real artifact and the real payload a target compiler produced for it.
fn compiled() -> (Artifact, TargetPayload) {
    spirv().compiled()
}

fn target<'a>(payload: &'a TargetPayload) -> MaterializerTarget<'a> {
    MaterializerTarget {
        backend_id: vyre_driver_spirv::SPIRV_BACKEND_ID,
        format: payload.format(),
        profile: payload.profile(),
    }
}

/// Rebuild a payload around a perturbed bundle and entry list.
fn repack(
    artifact: &Artifact,
    payload: &TargetPayload,
    bundle: &TargetModuleBundle,
    entries: Vec<TargetEntryPoint>,
) -> TargetPayload {
    TargetPayload::new(
        artifact,
        payload.format().clone(),
        payload.profile().clone(),
        entries,
        bundle.to_bytes().expect("bundle must encode"),
    )
    .expect("perturbed payload must still seal")
}

fn bundle_of(payload: &TargetPayload) -> TargetModuleBundle {
    TargetModuleBundle::from_bytes(payload.bytes()).expect("bundle must decode")
}

fn expect_invalid_program(error: BackendError, expected: &str) {
    match error {
        BackendError::InvalidProgram { fix } => assert!(
            fix.contains(expected),
            "rejection must name the cause; wanted `{expected}`, got `{fix}`"
        ),
        other => panic!("expected InvalidProgram, got {other:?}"),
    }
}

/// WHY: the accepting path must hand back one admitted module per selected
/// fusion group, already carrying the decoded Program and the payload's grid.
#[test]
fn admission_pairs_every_selected_group_with_its_program_and_grid() {
    let (artifact, payload) = compiled();
    let admitted =
        materialize::admit(&artifact, &payload, target(&payload)).expect("payload must admit");

    assert_eq!(
        admitted.len(),
        artifact.fusion().len(),
        "one admitted module per compiler-selected fusion group"
    );
    for (module, entry) in admitted.iter().zip(payload.entries()) {
        assert_eq!(module.image.entry_point, entry.name);
        let launch = module
            .config
            .launch
            .expect("admission must freeze the recorded launch");
        assert_eq!(
            launch.grid(),
            entry.grid_size,
            "admission must carry the payload entry grid, not a default"
        );
        assert_eq!(launch.workgroup(), entry.workgroup_size);
        assert!(
            !module.image.bytes.is_empty(),
            "admitted module must retain its target-native bytes"
        );
    }
}

/// WHY: an admitted module is submitted as recorded. A tuner override on an
/// admitted config would be a second launch authority, and whichever one a
/// backend resolved to, the other is a shape nothing compiled against. Every
/// admitted module of a multi-entry payload is checked, not the first.
#[test]
fn admission_states_the_frozen_launch_and_no_tuner_override() {
    let (artifact, payload) = spirv().compiled_two_stage();
    let admitted =
        materialize::admit(&artifact, &payload, target(&payload)).expect("payload must admit");

    assert!(admitted.len() > 1, "the fixture must admit several modules");
    for module in &admitted {
        let config = &module.config;
        assert!(
            config.launch.is_some(),
            "every admitted module submits the recorded launch"
        );
        assert_eq!(config.workgroup_override, None);
        assert_eq!(config.grid_override, None);
        assert_eq!(config.dispatch_elements, None);
        assert_eq!(config.dispatch_grid, None);
        config
            .validate_launch_authority("spirv-admission-test")
            .expect("an admitted config must state exactly one launch authority");
    }
}

/// WHY: a payload sealed against a different artifact must never open, or a
/// backend would execute a kernel compiled for another program.
#[test]
fn admission_rejects_a_payload_sealed_for_another_artifact() {
    let (_, payload) = compiled();
    let other = foreign_artifact();
    let error = materialize::admit(&other, &payload, target(&payload))
        .expect_err("foreign artifact must be rejected");
    expect_invalid_program(error, "not authenticated");
}

/// WHY: format is the one mismatch that is a capability answer, not corruption,
/// so it must stay UnsupportedFeature and name the acquiring backend.
#[test]
fn admission_reports_a_foreign_payload_format_as_unsupported() {
    let (artifact, payload) = compiled();
    let foreign = TargetPayloadFormat::new("not-spv", 1).expect("format must build");
    let mismatched = MaterializerTarget {
        format: &foreign,
        ..target(&payload)
    };
    match materialize::admit(&artifact, &payload, mismatched) {
        Err(BackendError::UnsupportedFeature { name, backend }) => {
            assert!(
                name.contains(payload.format().identity()),
                "rejection must name the offered format, got `{name}`"
            );
            assert_eq!(backend, vyre_driver_spirv::SPIRV_BACKEND_ID);
        }
        other => panic!("expected UnsupportedFeature, got {other:?}"),
    }
}

/// WHY: a payload built for another device profile is not runnable here even
/// though it is authentic for the artifact.
#[test]
fn admission_rejects_a_payload_built_for_another_profile() {
    let (artifact, payload) = compiled();
    let foreign = TargetProfile::new("foreign-profile", 1, [64, 1, 1], 64, 0, 32)
        .expect("profile must build");
    let mismatched = MaterializerTarget {
        profile: &foreign,
        ..target(&payload)
    };
    let error = materialize::admit(&artifact, &payload, mismatched)
        .expect_err("foreign profile must be rejected");
    expect_invalid_program(error, "profile does not match");
}

/// WHY: this is the check two of four backends were missing. A module whose
/// entry point is not `main` was executable on half the fleet.
#[test]
fn admission_rejects_a_module_whose_entry_point_is_not_main() {
    let (artifact, payload) = compiled();
    let mut bundle = bundle_of(&payload);
    bundle.modules[0].entry_point = "not_main".to_string();
    let mut entries = payload.entries().to_vec();
    entries[0].name = "not_main".to_string();
    let perturbed = repack(&artifact, &payload, &bundle, entries);

    let error = materialize::admit(&artifact, &perturbed, target(&perturbed))
        .expect_err("non-main entry point must be rejected");
    expect_invalid_program(error, "entry point must be `main`");
}

/// WHY: entry metadata and the emitted module must name the same entry, or the
/// dispatch grid is read from an entry that describes a different kernel.
#[test]
fn admission_rejects_entry_metadata_that_names_another_entry() {
    let (artifact, payload) = compiled();
    let bundle = bundle_of(&payload);
    let mut entries = payload.entries().to_vec();
    entries[0].name = "main_shadow".to_string();
    let perturbed = repack(&artifact, &payload, &bundle, entries);

    let error = materialize::admit(&artifact, &perturbed, target(&perturbed))
        .expect_err("entry name disagreement must be rejected");
    expect_invalid_program(error, "must name the emitted target entry point");
}

/// WHY: the module count is how admission knows the payload implements the whole
/// selected plan and not a subset of it.
#[test]
fn admission_rejects_a_module_count_that_disagrees_with_the_plan() {
    let (artifact, payload) = compiled();
    let bundle = TargetModuleBundle::new(Vec::new());
    let perturbed = repack(&artifact, &payload, &bundle, payload.entries().to_vec());

    let error = materialize::admit(&artifact, &perturbed, target(&perturbed))
        .expect_err("a bundle that implements no selected group must be rejected");
    expect_invalid_program(error, "module count must equal");
}

/// WHY: a corrupt Program wire must fail admission, never reach a backend that
/// would decode garbage into a dispatch. The bundle schema catches this during
/// decode, so it surfaces as an attributed decode failure rather than as the
/// defensive per-module wire check further down `admit`.
#[test]
fn admission_rejects_a_malformed_program_wire() {
    let (artifact, payload) = compiled();
    let mut bundle = bundle_of(&payload);
    bundle.modules[0].program = vec![0xff, 0xff, 0xff, 0xff];
    let perturbed = repack(&artifact, &payload, &bundle, payload.entries().to_vec());

    match materialize::admit(&artifact, &perturbed, target(&perturbed)) {
        Err(BackendError::KernelCompileFailed {
            backend,
            compiler_message,
        }) => {
            assert_eq!(backend, vyre_driver_spirv::SPIRV_BACKEND_ID);
            assert!(
                compiler_message.contains("selected Program is malformed"),
                "rejection must name the malformed wire, got `{compiler_message}`"
            );
        }
        other => panic!("expected KernelCompileFailed, got {other:?}"),
    }
}

/// WHY: bundle corruption is a decode failure, and it must be attributed to the
/// backend that offered the payload rather than reported as a bad program.
#[test]
fn admission_attributes_bundle_corruption_to_the_acquiring_backend() {
    let (artifact, payload) = compiled();
    let mut bytes = payload.bytes().to_vec();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    let corrupt = TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        bytes,
    )
    .expect("corrupt bundle must still seal into a payload");

    match materialize::admit(&artifact, &corrupt, target(&corrupt)) {
        Err(BackendError::KernelCompileFailed {
            backend,
            compiler_message,
        }) => {
            assert_eq!(backend, vyre_driver_spirv::SPIRV_BACKEND_ID);
            assert!(
                compiler_message.contains("rebuild the target payload"),
                "decode failure must state the corrective action, got `{compiler_message}`"
            );
        }
        other => panic!("expected KernelCompileFailed, got {other:?}"),
    }
}

/// WHY: closes the class "two parallel per-group lists in one payload are paired
/// by position, and nothing states the orders agree". `admit` zipped the bundle's
/// modules, the artifact's fusion records and the payload's entries. The bundle
/// canonically sorts its modules by `(stage, group)`, the fusion records are in
/// the artifact's own plan order, and the entries are in the order the target
/// compiler emitted them, so all three pairings rested on three orders happening
/// to agree. The one check that could have caught an entry paired with the wrong
/// module compared entry names, and every entry a compiler emits is named `main`,
/// so a reordered entry list admitted and each module ran with another module's
/// resource bindings.
///
/// The fix is resolution rather than a refusal: order carries no meaning once each
/// module names the record and entry it belongs to, so a reordered list must still
/// pair correctly. Launch geometry is no longer the observable, because a
/// submission reads it out of the artifact record; the resource bindings are, and
/// they are still the entry's own.
///
/// Does not catch: an entry whose bindings are wrong for the group it is correctly
/// paired with. Node identity proves which group an entry describes, not that the
/// description is right; that is the target compiler's contract. It also leaves
/// two of admission's resolution failures unproven, because a sealed payload
/// cannot carry them: `TargetPayload::new` runs `validate_entries`, which already
/// refuses a duplicate entry node and an entry node absent from the artifact.
#[test]
fn admission_pairs_an_entry_with_its_own_module_whatever_the_entry_order() {
    let (artifact, payload) = spirv().compiled_two_stage();
    let mut entries = payload.entries().to_vec();
    assert_eq!(
        entries.len(),
        2,
        "the two-stage fixture must offer exactly two entries to permute"
    );
    assert_ne!(
        entries[0].resource_bindings, entries[1].resource_bindings,
        "the fixture entries must differ, or a mispaired entry is unobservable"
    );
    let expected_bindings = entries
        .iter()
        .map(|entry| (entry.node, entry.resource_bindings.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let recorded_grid = artifact
        .geometry()
        .iter()
        .map(|record| (record.node, record.grid))
        .collect::<std::collections::BTreeMap<_, _>>();
    entries.reverse();
    let reordered = repack(&artifact, &payload, &bundle_of(&payload), entries);

    let admitted = materialize::admit(&artifact, &reordered, target(&reordered))
        .expect("a reordered entry list still names which group each entry describes");

    assert_eq!(admitted.len(), artifact.fusion().len());
    for module in &admitted {
        let node = *module
            .image
            .nodes
            .first()
            .expect("an admitted module carries its group's member nodes");
        assert_eq!(
            Some(&module.resource_bindings),
            expected_bindings.get(&node),
            "module for node {node:?} must carry the bindings of the entry that names that node"
        );
        let launch = module
            .config
            .launch
            .expect("admission must freeze the recorded launch");
        assert_eq!(
            Some(launch.grid()),
            recorded_grid.get(&node).copied(),
            "module for node {node:?} must submit the grid the artifact recorded for that node"
        );
    }
}

/// WHY: submission order is the compiler's. `admit` used to walk the bundle's
/// own module list, so the order a target compiler serialized its modules in
/// decided which entry point ran first, and a bundle sorted any other way would
/// run a consumer before its producer. Two independent rules now hold that
/// order: admission walks the selected plan, whose recorded order the artifact
/// refuses unless it follows the dependency DAG, and a bundle whose modules are
/// not in canonical stage/group order never decodes.
#[test]
fn admission_submits_in_the_recorded_plan_order() {
    let (artifact, payload) = spirv().compiled_two_stage();
    let planned = artifact
        .fusion()
        .iter()
        .map(|record| record.id)
        .collect::<Vec<_>>();
    assert!(
        planned.len() > 1,
        "the fixture must select several fusion groups"
    );

    let admitted =
        materialize::admit(&artifact, &payload, target(&payload)).expect("payload must admit");
    assert_eq!(
        admitted
            .iter()
            .map(|module| module.image.group)
            .collect::<Vec<_>>(),
        planned,
        "admission must hand back the modules in the recorded plan order"
    );

    let mut reversed = bundle_of(&payload);
    reversed.modules.reverse();
    let repacked = TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        reversed.to_bytes().expect("bundle must encode"),
    )
    .expect("a payload seals over whatever bytes it carries");
    let error = materialize::admit(&artifact, &repacked, target(&repacked))
        .expect_err("a bundle out of canonical order must be refused");
    match error {
        BackendError::KernelCompileFailed {
            compiler_message, ..
        } => assert!(
            compiler_message.contains("canonical stage/group order"),
            "the refusal must name the order rule; got {compiler_message}"
        ),
        other => panic!("expected KernelCompileFailed, got {other:?}"),
    }
}

/// WHY: an entry that restates launch geometry is a second authority, and the one
/// a target compiler emits is the one a backend would have submitted. A payload
/// stating any shape other than the recorded one never seals, so no consumer has
/// to decide which of the two to believe.
#[test]
fn a_payload_entry_that_restates_the_recorded_geometry_never_seals() {
    let (artifact, payload) = spirv().compiled_two_stage();
    let recorded = artifact
        .geometry()
        .first()
        .expect("the fixture artifact records its launches");
    let bundle = bundle_of(&payload);

    for (field, mutate) in [
        (
            "grid_size",
            (|entry: &mut TargetEntryPoint| {
                entry.grid_size = [entry.grid_size[0] + 1, 1, 1];
            }) as fn(&mut TargetEntryPoint),
        ),
        ("workgroup_size", |entry: &mut TargetEntryPoint| {
            entry.workgroup_size = [entry.workgroup_size[0] + 1, 1, 1];
        }),
        ("dynamic_shared_bytes", |entry: &mut TargetEntryPoint| {
            entry.dynamic_shared_bytes += 256;
        }),
    ] {
        let mut entries = payload.entries().to_vec();
        mutate(&mut entries[0]);
        let error = TargetPayload::new(
            &artifact,
            payload.format().clone(),
            payload.profile().clone(),
            entries,
            bundle.to_bytes().expect("bundle must encode"),
        )
        .expect_err("Fix: an entry restating the recorded geometry must be refused at seal.");
        let text = format!("{error:?}");
        assert!(
            text.contains(field),
            "the refusal must name the restated field; got {text}"
        );
    }

    // The unmutated entry list, which agrees with the record, still seals.
    TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        bundle.to_bytes().expect("bundle must encode"),
    )
    .expect("Fix: a payload agreeing with the record must seal.");
    assert_eq!(recorded.node, artifact.geometry()[0].node);
}
