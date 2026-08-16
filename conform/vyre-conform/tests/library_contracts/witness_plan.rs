//! `witness_plan` contracts over the public `vyre_conform` surface.

use vyre::ir::BufferAccess;
use vyre::ir::Program;
use vyre::ir::{BufferDecl, DataType, Node};
use vyre_conform::witness_plan::*;

#[test]
fn witness_input_plan_accepts_logical_fixture_order_after_output_buffer() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("input", 1, BufferAccess::ReadOnly, DataType::U32).with_count(2),
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
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
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
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
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

#[test]
fn witness_input_plan_rejects_undersized_static_input_byte_length() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan =
        WitnessInputPlan::for_program(&program).expect("Fix: static input planning must succeed.");
    // Expected 8 bytes (2 x u32), but fixture provides 4 bytes
    let case = vec![vec![1, 0, 0, 0]];
    let mut backend_inputs = Vec::new();

    let error = plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect_err("Fix: undersized static input fixture must be rejected.");

    assert!(
        error.contains("expected 8 bytes from its static buffer declaration but received 4 bytes"),
        "Fix: error must diagnose static byte length mismatch, got: {error}"
    );
}

#[test]
fn witness_input_plan_rejects_oversized_static_input_byte_length() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan =
        WitnessInputPlan::for_program(&program).expect("Fix: static input planning must succeed.");
    // Expected 4 bytes (1 x u32), but fixture provides 8 bytes
    let case = vec![vec![1, 0, 0, 0, 2, 0, 0, 0]];
    let mut backend_inputs = Vec::new();

    let error = plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect_err("Fix: oversized static input fixture must be rejected.");

    assert!(
        error.contains("expected 4 bytes from its static buffer declaration but received 8 bytes"),
        "Fix: error must diagnose static byte length mismatch, got: {error}"
    );
}

#[test]
fn witness_input_plan_skips_backend_allocated_write_only_and_workgroup_buffers() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("w_out", 1, BufferAccess::WriteOnly, DataType::U32).with_count(1),
            BufferDecl::workgroup("wg", 16, DataType::U32),
            BufferDecl::storage("s_out", 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1)
                .with_pipeline_live_out(true),
            BufferDecl::output("out", 4, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: witness input planning must succeed for mixed buffer declarations.");
    assert_eq!(
        plan.source_count(),
        1,
        "Fix: only read-only host input buffer should be in the witness input plan."
    );
    assert_eq!(
        plan.buffer_indices().collect::<Vec<_>>(),
        vec![0],
        "Fix: stream order should only reference the read-only buffer."
    );
    let case = vec![1u32.to_le_bytes().to_vec()];
    let mut backend_inputs = Vec::new();
    plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect("Fix: fixture matching only input buffer must succeed.");
    assert_eq!(
        backend_inputs,
        vec![case[0].as_slice()],
        "Fix: planned inputs must route only the read-only host input."
    );
}

#[test]
fn registry_derived_fixture_mutation_fails_validation_before_gpu_execution() {
    let registry = vyre_registry_link::operation::live_operation_registry();
    let mut ops_tested = 0usize;
    let mut mutations_tested = 0usize;

    for entry in registry.iter() {
        let Some(build) = entry.build else {
            continue;
        };
        let Some(test_inputs) = entry.test_inputs else {
            continue;
        };
        let program = build();
        let plan = WitnessInputPlan::for_program(&program).unwrap_or_else(|err| {
            panic!("{}: failed to build witness input plan: {err}", entry.id)
        });
        let cases = test_inputs();
        let mut backend_inputs = Vec::new();

        for (case_idx, case) in cases.iter().enumerate() {
            // 1. Valid case must succeed
            plan_witness_inputs_into(case, &plan, &mut backend_inputs).unwrap_or_else(|err| {
                panic!(
                    "{}: valid fixture case {case_idx} must satisfy witness input plan: {err}",
                    entry.id
                )
            });

            // 2. Every explicit, statically sized planned input rejects byte-size mutations.
            let source_buffer_indices = plan.buffer_indices().collect::<Vec<_>>();
            let full_buffer_layout = case.len() > plan.source_count();
            for (source_index, buffer_index) in source_buffer_indices.into_iter().enumerate() {
                let fixture_index = if full_buffer_layout {
                    buffer_index
                } else {
                    source_index
                };
                let Some(buf) = case.get(fixture_index) else {
                    continue;
                };
                let expected = program.buffers()[buffer_index]
                    .static_byte_len()
                    .expect("Fix: fixture mutation must read the canonical static buffer shape");
                let Some(expected) = expected else {
                    continue;
                };
                assert_eq!(
                    buf.len(),
                    expected,
                    "Fix: {} case {case_idx} planned input at fixture index {fixture_index} must match its static declaration before mutation",
                    entry.id
                );

                if !buf.is_empty() {
                    let mut truncated_case = case.clone();
                    truncated_case[fixture_index].pop();
                    let mut mutated_inputs = Vec::new();
                    let trunc_result =
                        plan_witness_inputs_into(&truncated_case, &plan, &mut mutated_inputs);
                    assert!(
                        trunc_result.is_err(),
                        "Fix: {}: truncated planned fixture buffer at index {fixture_index} must fail witness planning before GPU execution.",
                        entry.id
                    );
                    mutations_tested += 1;
                }

                let mut padded_case = case.clone();
                padded_case[fixture_index].push(0x5A);
                let mut mutated_inputs = Vec::new();
                let pad_result = plan_witness_inputs_into(&padded_case, &plan, &mut mutated_inputs);
                assert!(
                    pad_result.is_err(),
                    "Fix: {}: padded planned fixture buffer at index {fixture_index} must fail witness planning before GPU execution.",
                    entry.id
                );
                mutations_tested += 1;
            }
        }
        ops_tested += 1;
    }

    assert!(
        ops_tested >= 100,
        "Fix: registry-derived mutation coverage must test live ops, got {ops_tested}."
    );
    assert!(
        mutations_tested > 0,
        "Fix: registry-derived mutation coverage found no explicit static planned inputs."
    );
}

#[test]
fn witness_input_plan_accepts_multiple_logical_inputs_after_output_buffer_with_equal_lengths() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("in_a", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("in_b", 2, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: witness input planning must succeed for multiple inputs after output.");
    assert_eq!(plan.source_count(), 2);

    let case = vec![10u32.to_le_bytes().to_vec(), 20u32.to_le_bytes().to_vec()];
    let mut backend_inputs = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut backend_inputs).expect(
        "Fix: logical fixture bytes must route accurately when multiple inputs share byte length.",
    );

    assert_eq!(
        backend_inputs,
        vec![case[0].as_slice(), case[1].as_slice()],
        "Fix: each planned input must receive its own distinct logical fixture bytes."
    );
}

#[test]
fn witness_input_plan_does_not_reorder_logical_inputs_to_match_lengths() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("wide", 1, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::storage("narrow", 2, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan =
        WitnessInputPlan::for_program(&program).expect("Fix: witness input planning must succeed.");
    let case = vec![
        10u32.to_le_bytes().to_vec(),
        [20u32.to_le_bytes(), 30u32.to_le_bytes()].concat(),
    ];
    let mut backend_inputs = Vec::new();

    let error = plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect_err("Fix: wrong-sized logical fixtures must not be reordered by byte length.");

    assert!(
        error.contains("fixture index `0` / program index `1` expected 8 bytes"),
        "Fix: the first logical fixture must stay bound to the first planned input, got: {error}"
    );
}

#[test]
fn witness_input_plan_accepts_full_program_buffer_layout_fixture() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("in_a", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("in_b", 2, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan =
        WitnessInputPlan::for_program(&program).expect("Fix: witness input planning must succeed.");

    // Full Program::buffers order fixture: dummy out, in_a, in_b
    let case = vec![
        vec![0, 0, 0, 0],
        10u32.to_le_bytes().to_vec(),
        20u32.to_le_bytes().to_vec(),
    ];
    let mut backend_inputs = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect("Fix: full buffer layout fixture must route to input buffers by buffer_index.");

    assert_eq!(
        backend_inputs,
        vec![case[1].as_slice(), case[2].as_slice()],
        "Fix: full buffer layout fixture must route non-output buffers into the input stream."
    );
}

#[test]
fn witness_input_plan_accepts_explicit_static_read_write_fixture_bytes() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: static read-write planning must succeed.");

    let case = vec![10u32.to_le_bytes().to_vec(), 99u32.to_le_bytes().to_vec()];
    let mut backend_inputs = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect("Fix: explicit static read-write fixture bytes must be accepted.");

    assert_eq!(
        backend_inputs,
        vec![case[0].as_slice(), case[1].as_slice()],
        "Fix: explicit static read-write bytes must override zero-fill default."
    );
}

#[test]
fn witness_input_plan_rejects_mismatched_explicit_static_read_write_byte_length() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: static read-write planning must succeed.");

    // scratch expects 8 bytes, but fixture provides 4 bytes
    let case = vec![10u32.to_le_bytes().to_vec(), 99u32.to_le_bytes().to_vec()];
    let mut backend_inputs = Vec::new();

    let error = plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect_err("Fix: explicit static read-write fixture with wrong length must be rejected.");

    assert!(
        error.contains("expected 8 bytes from its static buffer declaration but received 4 bytes"),
        "Fix: error must diagnose static read-write byte length mismatch, got: {error}"
    );
}

#[test]
fn witness_input_plan_accepts_explicit_dynamic_read_write_fixture_bytes() {
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
        .expect("Fix: dynamic read-write planning must succeed.");

    let case = vec![vec![1, 2, 3, 4, 5, 6, 7, 8]];
    let mut backend_inputs = Vec::new();

    plan_witness_inputs_into(&case, &plan, &mut backend_inputs)
        .expect("Fix: dynamic read-write buffer with explicit fixture bytes must succeed.");

    assert_eq!(
        backend_inputs,
        vec![case[0].as_slice()],
        "Fix: dynamic read-write fixture bytes must pass through byte-exactly."
    );
}
