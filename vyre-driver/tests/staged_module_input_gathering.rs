//! Resolving one module's staged target bindings out of the execution state.
//!
//! # Why this suite exists
//!
//! A fused artifact carries a value from one module to the next through a
//! buffer the reading module's `Program` declares as consuming no host input.
//! Staging a launch from the `Program`'s input order therefore skips it, and
//! the reading module runs over an allocation nothing filled. That returns a
//! wrong answer rather than a rejection, which is how the fused operations in
//! `vyre-libs::security` read their stage-one bitset as zero on every case.
//!
//! `vyre_megakernel::staged_input_slots` states which bindings a launch fills
//! and in what order; `gather_artifact_inputs` resolves each one to the value
//! bound to it, and `BindingPlan::validate_named_inputs` checks the bytes
//! against the declaration each name resolves to. The rejection paths here are
//! reachable by no passing operation, so conformance execution proves none of
//! them.
//!
//! What this does not catch: a backend that never calls either function and
//! keeps the `Program`-order default. Conformance execution over the linked
//! backends is what judges that, because a skipped carrier is only visible in
//! the answer.

use std::collections::{BTreeMap, BTreeSet};

use vyre_driver::materialize::{gather_artifact_inputs, InstanceCore, NEUTRAL_MESSAGES};
use vyre_driver::{BindingPlan, DeviceIdentity};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};
use vyre_megakernel::{ArtifactInputSlot, ArtifactValueId, Digest};

/// Canonical identity of the value the seed input binds.
const SEED: ArtifactValueId = ArtifactValueId(7);
/// Canonical identity of the value one module writes and the next reads.
const CARRIED: ArtifactValueId = ArtifactValueId(8);

/// Bytes per element of every fixture buffer.
const ELEMENT_BYTES: usize = 4;
/// Elements the carried value holds.
const CARRIED_ELEMENTS: usize = 4;

/// One staged binding.
fn input_slot(name: &str, slot: u32, elements: Option<usize>) -> ArtifactInputSlot {
    ArtifactInputSlot {
        name: name.to_string(),
        group: 0,
        slot,
        expected_max: elements.map(|count| count * ELEMENT_BYTES),
        launch_zeros: elements.map(|count| vec![0; count * ELEMENT_BYTES].into_boxed_slice()),
    }
}

/// The staged bindings of the module that reads the carried value.
fn reader_slots() -> Vec<ArtifactInputSlot> {
    vec![
        input_slot("seed", 0, Some(2)),
        input_slot("carried", 1, Some(CARRIED_ELEMENTS)),
    ]
}

/// An instance whose module 0 writes `carried` and whose module 1 reads it.
fn fused_core() -> InstanceCore {
    InstanceCore {
        artifact: Digest([1; 32]),
        payload: Digest([2; 32]),
        device: DeviceIdentity {
            backend: "fixture",
            device: "fixture-0".to_string(),
            generation: 1,
        },
        values: BTreeMap::from([
            ("seed".to_string(), SEED),
            ("carried".to_string(), CARRIED),
        ]),
        outputs: BTreeSet::new(),
        retained: BTreeSet::from([CARRIED]),
        messages: NEUTRAL_MESSAGES,
        module_inputs: vec![vec![SEED, CARRIED], vec![SEED, CARRIED]],
        module_outputs: vec![vec![CARRIED], Vec::new()],
        module_named_resources: vec![
            BTreeMap::from([
                ("seed".to_string(), SEED),
                ("carried".to_string(), CARRIED),
            ]),
            BTreeMap::from([
                ("seed".to_string(), SEED),
                ("carried".to_string(), CARRIED),
            ]),
        ],
        module_buffer_slots: vec![
            BTreeMap::from([
                ("seed".to_string(), (0, 0)),
                ("carried".to_string(), (0, 1)),
            ]),
            BTreeMap::from([
                ("seed".to_string(), (0, 0)),
                ("carried".to_string(), (0, 1)),
            ]),
        ],
        module_resources: vec![
            BTreeMap::from([((0, 0), SEED), ((0, 1), CARRIED)]),
            BTreeMap::from([((0, 0), SEED), ((0, 1), CARRIED)]),
        ],
        retained_predecessors: BTreeMap::new(),
    }
}

/// Two elements of seed bytes.
fn seed_bytes() -> Vec<u8> {
    [3_u32, 5].into_iter().flat_map(u32::to_le_bytes).collect()
}

/// The bytes module 0 leaves behind for module 1.
fn carried_bytes() -> Vec<u8> {
    [11_u32, 22, 33, 44]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// WHY: the carried value is what a fused artifact passes between modules. A
/// gather that skips it hands the reading module an unwritten allocation, and
/// the operation answers with the zeros it read.
#[test]
fn the_reading_module_stages_the_value_the_previous_module_produced() {
    let core = fused_core();
    let slots = reader_slots();
    let state = BTreeMap::from([(SEED, seed_bytes()), (CARRIED, carried_bytes())]);

    let inputs = gather_artifact_inputs(&core, 1, &slots, &state)
        .expect("Fix: the reading module must stage every binding its descriptor declares");

    assert_eq!(inputs, vec![&seed_bytes()[..], &carried_bytes()[..]]);
}

/// WHY: a module that writes the slot it also reads runs before anything has
/// bound the value. Rejecting it would refuse a correct artifact, and leaving
/// the slot out would launch the module one input short.
#[test]
fn a_slot_this_module_produces_launches_over_the_allocation_it_reserved() {
    let core = fused_core();
    let slots = reader_slots();
    let state = BTreeMap::from([(SEED, seed_bytes())]);

    let inputs = gather_artifact_inputs(&core, 0, &slots, &state)
        .expect("Fix: a module that produces its own staged slot must launch");

    assert_eq!(inputs[1], vec![0; CARRIED_ELEMENTS * ELEMENT_BYTES]);
}

/// WHY: an unbound value this module does not produce is a dispatch over
/// undefined memory. Reading zeros there is a wrong answer, so the gather must
/// refuse and name the binding.
#[test]
fn an_unbound_value_no_module_produced_is_refused() {
    let core = fused_core();
    let state = BTreeMap::from([(SEED, seed_bytes())]);

    let error = gather_artifact_inputs(&core, 1, &reader_slots(), &state)
        .expect_err("Fix: an unbound staged value must be refused");

    let message = error.to_string();
    assert!(
        message.contains("`carried`") && message.contains("unbound"),
        "Fix: the rejection must name the unbound binding: {message}"
    );
}

/// WHY: a value longer than the declaration it fills overruns the allocation
/// the launch made for it. The rejection names both the canonical value and the
/// binding, because the artifact is where the mismatch is repaired.
#[test]
fn a_value_longer_than_its_static_declaration_is_refused() {
    let core = fused_core();
    let oversize = [1_u32; CARRIED_ELEMENTS + 1]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let state = BTreeMap::from([(SEED, seed_bytes()), (CARRIED, oversize)]);

    let error = gather_artifact_inputs(&core, 1, &reader_slots(), &state)
        .expect_err("Fix: an oversize staged value must be refused");

    let message = error.to_string();
    assert!(
        message.contains("`carried`") && message.contains("(`carried`)"),
        "Fix: the rejection must name the canonical value and the binding: {message}"
    );
    assert!(
        message.contains("20 byte(s)") && message.contains("16 byte(s)"),
        "Fix: the rejection must state the supplied and declared byte counts: {message}"
    );
}

/// WHY: a runtime-sized declaration has no ceiling, so a bound value of any
/// length satisfies it. Treating its absent ceiling as zero refuses every
/// runtime-sized stage.
#[test]
fn a_runtime_sized_slot_accepts_a_value_of_any_length() {
    let core = fused_core();
    let state = BTreeMap::from([(SEED, seed_bytes()), (CARRIED, carried_bytes())]);
    let slots = vec![input_slot("seed", 0, Some(2)), input_slot("carried", 1, None)];

    let inputs = gather_artifact_inputs(&core, 1, &slots, &state)
        .expect("Fix: a runtime-sized staged slot must accept its bound value");

    assert_eq!(inputs[1], &carried_bytes()[..]);
}

/// The program both fixture modules are lowered from.
fn fused_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("seed", 0, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::output("carried", 1, DataType::U32).with_count(CARRIED_ELEMENTS as u32),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ],
        [4, 1, 1],
        Vec::new(),
    )
}

/// WHY: staged inputs are neither in nor the same length as the plan's own
/// input order, so they are checked against the binding each name resolves to.
/// Checking them positionally compares the carried value against the seed
/// declaration and rejects a correct launch.
#[test]
fn staged_inputs_are_checked_against_the_binding_each_name_resolves_to() {
    let plan = BindingPlan::build(&fused_program()).expect("Fix: the fixture plan must build");

    plan.validate_named_inputs(
        &["seed", "carried"],
        &[&seed_bytes()[..], &carried_bytes()[..]],
    )
    .expect("Fix: staged inputs matching their own declarations must be admitted");
}

/// WHY: a name and its bytes are one staged slot. A launch given a different
/// number of each binds one buffer's bytes to another buffer.
#[test]
fn a_name_count_that_disagrees_with_the_input_count_is_refused() {
    let plan = BindingPlan::build(&fused_program()).expect("Fix: the fixture plan must build");

    let error = plan
        .validate_named_inputs(&["seed", "carried"], &[&seed_bytes()[..]])
        .expect_err("Fix: a staged name with no bytes must be refused");

    let message = error.to_string();
    assert!(
        message.contains("2 named binding(s)") && message.contains("1 input buffer(s)"),
        "Fix: the rejection must state both counts: {message}"
    );
}

/// WHY: a name that resolves to no binding means the payload's resource
/// bindings were built from a different Program than the module was lowered
/// from. Skipping it would stage bytes into whatever buffer followed.
#[test]
fn a_staged_name_that_resolves_to_no_binding_is_refused() {
    let plan = BindingPlan::build(&fused_program()).expect("Fix: the fixture plan must build");

    let error = plan
        .validate_named_inputs(&["absent"], &[&seed_bytes()[..]])
        .expect_err("Fix: a staged name outside the plan must be refused");

    let message = error.to_string();
    assert!(
        message.contains("`absent`"),
        "Fix: the rejection must name the binding that resolves to nothing: {message}"
    );
}

/// WHY: a staged value shorter than the declaration it fills leaves the tail of
/// the launch allocation undefined, which is the same wrong answer the skipped
/// carrier produced.
#[test]
fn a_staged_value_shorter_than_its_declaration_is_refused() {
    let plan = BindingPlan::build(&fused_program()).expect("Fix: the fixture plan must build");

    let error = plan
        .validate_named_inputs(&["carried"], &[&carried_bytes()[..ELEMENT_BYTES]])
        .expect_err("Fix: a short staged value must be refused");

    let message = error.to_string();
    assert!(
        message.contains("`carried`") && message.contains("16 bytes"),
        "Fix: the rejection must name the binding and its declared size: {message}"
    );
}

/// WHY: bytes that do not divide into whole elements cannot be a value of the
/// declared type, whatever their total length.
#[test]
fn a_staged_value_misaligned_to_its_element_size_is_refused() {
    let plan = BindingPlan::build(&fused_program()).expect("Fix: the fixture plan must build");
    let misaligned = carried_bytes()[..CARRIED_ELEMENTS * ELEMENT_BYTES - 1].to_vec();

    let error = plan
        .validate_named_inputs(&["carried"], &[&misaligned[..]])
        .expect_err("Fix: a misaligned staged value must be refused");

    let message = error.to_string();
    assert!(
        message.contains("`carried`") && message.contains("element size"),
        "Fix: the rejection must name the binding and its element size: {message}"
    );
}
