//! `witness_plan` contracts over the public `vyre_conform` surface.

use vyre::ir::BufferAccess;
use vyre::ir::Program;
use vyre_conform::witness_plan::*;
use vyre::ir::{BufferDecl, DataType, Node};

#[test]
fn witness_input_plan_accepts_logical_fixture_order_after_output_buffer() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("input", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(2),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: logical input planning must succeed when an output is declared first.");
    let case = vec![vec![1, 0, 0, 0, 2, 0, 0, 0]];
    let mut backend_inputs = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect("Fix: logical fixture bytes must route even when outputs precede inputs.");

    assert_eq!(
        backend_inputs,
        vec![case[0].as_slice()],
        "Fix: the plan must use logical fixture order, not raw Program::buffers indices."
    );
    assert_eq!(
        plan.buffer_indices().collect::<Vec<_>>(),
        vec![1],
        "Fix: buffer_indices must report the Program::buffers position behind each planned \
         slice, so a caller that rewrites one input reads the right declaration."
    );
}

#[test]
fn owned_expansion_matches_the_borrowed_stream_byte_for_byte() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: static read-write zero-fill planning must succeed.");
    let case = vec![7u32.to_le_bytes().to_vec()];
    let mut borrowed = Vec::new();
    let mut owned = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut borrowed)
        .expect("Fix: borrowed expansion must succeed for a zero-fillable read-write buffer.");
    plan_witness_inputs_owned_into(&case, &plan, &mut owned)
        .expect("Fix: owned expansion must succeed wherever the borrowed one does.");

    assert_eq!(
        owned.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        borrowed,
        "Fix: the owned expansion must copy the planned stream, not reorder or resynthesize it."
    );
}

#[test]
fn owned_expansion_reports_the_same_rejection_as_the_borrowed_one() {
    let program = Program::wrapped(
        vec![BufferDecl::storage(
            "scratch",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: dynamic read-write buffers may be fixture-backed per case.");
    let mut owned = Vec::new();

    let error = plan_witness_inputs_owned_into(&[], &plan, &mut owned)
        .expect_err("Fix: the owned expansion must not zero-fill a runtime-sized buffer.");

    assert!(
        error.contains("runtime-sized read-write buffer"),
        "Fix: the owned expansion must surface the borrowed expansion's diagnosis, got: {error}"
    );
}

#[test]
fn witness_input_plan_accepts_fixture_backed_runtime_sized_read_input() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: runtime-sized read-only buffers must be fixture-backed, not rejected.");
    let case = vec![vec![0xA5; 12]];
    let mut backend_inputs = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect("Fix: concrete fixture bytes must satisfy a runtime-sized input buffer.");

    assert_eq!(
        backend_inputs,
        vec![case[0].as_slice()],
        "Fix: dynamic fixture-backed inputs must be passed through byte-exactly."
    );
}

#[test]
fn witness_input_plan_uses_zeroed_static_read_write_inputs() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: static read-write zero-fill planning must succeed.");
    let case = vec![1u32.to_le_bytes().to_vec()];
    let mut backend_inputs = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect("Fix: static read-write buffers may be omitted and zero-filled.");

    assert_eq!(
        backend_inputs,
        vec![case[0].as_slice(), &[0, 0, 0, 0][..]],
        "Fix: backend dispatch input stream must append zero-filled static read-write buffers."
    );
}

#[test]
fn witness_input_plan_rejects_omitted_runtime_sized_read_write_input() {
    let program = Program::wrapped(
        vec![BufferDecl::storage(
            "scratch",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: dynamic read-write buffers may be fixture-backed per case.");
    let mut backend_inputs = Vec::new();

    let error = plan_witness_inputs_into(&[], &plan, &mut backend_inputs)
        .expect_err("Fix: omitted dynamic read-write input must not be silently zeroed.");

    assert!(
        error.contains("runtime-sized read-write buffer"),
        "Fix: error must explain that dynamic read-write buffers need concrete fixture bytes, got: {error}"
    );
}
