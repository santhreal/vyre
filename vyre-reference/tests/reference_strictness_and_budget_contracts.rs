//! Contracts for the strict, budgeted, independent reference oracle.
//!
//! WHY: proves that the oracle
//! 1. interprets IR directly without invoking production optimizer, schedule,
//!    lowering, or emitter transforms;
//! 2. returns a structured failure class rather than a default value, at the
//!    site of the fault;
//! 3. terminates under work budget exhaustion and reports the bound;
//! 4. cannot issue an expected output or a certificate from permissive mode;
//! 5. evaluates single-rank collectives directly without external lowering.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CommGroup, DataType, Expr, Ident, Node, Program,
};
use vyre_reference::value::Value;
use vyre_reference::{
    DeterministicSchedulePolicy, ReferenceBudget, ReferenceErrorClass, ReferenceRequest,
    WorkloadEnvelope, REFERENCE_ORACLE_VERSION, REFERENCE_REQUEST_SCHEMA_VERSION,
};
use vyre_test_support::pass_programs::{indexed_input_copy_program, single_input_copy_program};

#[test]
fn oracle_path_invokes_no_production_transforms() {
    // 1. Derive the optimizer pass catalog dynamically from foundation.
    let catalog = vyre_foundation::optimizer::pass_catalog::optimization_catalog()
        .expect("optimizer catalog must be discoverable");
    assert!(
        !catalog.is_empty(),
        "Fix: optimizer catalog must contain registered production passes"
    );

    // Verify catalog contains known optimization passes.
    let pass_names: Vec<&str> = catalog.iter().map(|entry| entry.name).collect();
    assert!(
        pass_names.contains(&"const_fold.unary.logical_not_involution"),
        "catalog must include registered rules"
    );
    // 2. Build a program with dead lets, constants, and loops that production passes would transform.
    let program = Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [4, 1, 1],
        vec![
            // Dead let that DCE would eliminate:
            Node::let_bind("unused_dead_value", Expr::add(Expr::u32(10), Expr::u32(20))),
            // Real computation:
            Node::let_bind("idx", Expr::gid_x()),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::add(Expr::load("in", Expr::var("idx")), Expr::u32(1)),
            ),
        ],
    );

    // Fingerprint before reference execution.
    let fingerprint_before = program.fingerprint();

    // 3. Execute through the strict ReferenceRequest oracle.
    let inputs = vec![vyre_reference::value::Value::from(
        vec![
            10u32.to_le_bytes(),
            20u32.to_le_bytes(),
            30u32.to_le_bytes(),
            40u32.to_le_bytes(),
        ]
        .concat(),
    )];
    let request = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard());
    let result = request
        .execute()
        .expect("reference oracle must execute directly");

    // Assert the program AST was not modified by any production pass.
    assert_eq!(
        program.fingerprint(),
        fingerprint_before,
        "reference execution must not mutate or transform input Program"
    );

    // Assert exact correct output was computed directly.
    let out_bytes = result.outputs[0].to_bytes();
    let expected = vec![
        11u32.to_le_bytes(),
        21u32.to_le_bytes(),
        31u32.to_le_bytes(),
        41u32.to_le_bytes(),
    ]
    .concat();
    assert_eq!(out_bytes, expected);

    // Assert certificate is issued with oracle version.
    assert_eq!(result.certificate.oracle_version, REFERENCE_ORACLE_VERSION);
    assert_eq!(
        result.certificate.schema_version,
        REFERENCE_REQUEST_SCHEMA_VERSION
    );
}

/// The classes a host fault raises, which no IR program can request.
///
/// `Poison` needs a thread to panic while holding a buffer lock, and the oracle
/// fails closed with a process-level abort rather than a `ReferenceError` when
/// that happens, which `oob.rs` proves in place. `Nontermination` is reported
/// through `BudgetExhaustion` because the work ceiling is what observes a
/// non-terminating program. Both are named here so a reader sees the decision
/// instead of an absence.
const HOST_FAULT_CLASSES: [ReferenceErrorClass; 2] = [
    ReferenceErrorClass::Poison,
    ReferenceErrorClass::Nontermination,
];

/// WHY: strict mode is the mode whose outputs a device is graded against. An
/// out-of-bounds load absorbed as a zero, a store dropped, or an atomic
/// answered with `old = 0` produces an output the program never computed, and
/// the oracle then certifies it. Each case below drives one class through
/// `ReferenceRequest::execute` and requires the structured class at the site.
#[test]
fn strict_execution_refuses_every_fault_class_it_can_raise() {
    for (class, program, inputs) in strictness_cases() {
        let request = ReferenceRequest::new(&program, &inputs, ReferenceBudget::bounded(65_536));
        let error = match request.execute() {
            Ok(result) => panic!(
                "Fix: the {} case must fail, got {} outputs",
                class.name(),
                result.outputs.len()
            ),
            Err(error) => error,
        };
        assert_eq!(
            error.error_class(),
            class,
            "Fix: the {} case must return its own failure class, got: {error}",
            class.name()
        );
    }
}

/// WHY: the class space is derived, so a new class must be placed rather than
/// silently uncovered. This requires every member of
/// `ReferenceErrorClass::ALL` to be either driven by a case above or recorded
/// in `HOST_FAULT_CLASSES`.
#[test]
fn every_failure_class_is_either_driven_or_recorded_as_a_host_fault() {
    let driven: Vec<ReferenceErrorClass> = strictness_cases()
        .into_iter()
        .map(|(class, _, _)| class)
        .collect();
    for class in ReferenceErrorClass::ALL {
        assert!(
            driven.contains(&class) || HOST_FAULT_CLASSES.contains(&class),
            "Fix: failure class `{}` is neither driven by a strictness case nor \
             recorded in HOST_FAULT_CLASSES; add a case or record the decision.",
            class.name()
        );
    }
}

/// One program per failure class a strict execution can raise, with the inputs
/// it is submitted with.
fn strictness_cases() -> Vec<(ReferenceErrorClass, Program, Vec<Value>)> {
    vec![
        (
            ReferenceErrorClass::MissingValue,
            single_input_copy_program(),
            Vec::new(),
        ),
        (
            ReferenceErrorClass::TypeMismatch,
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::bitand(Expr::f32(1.5), Expr::u32(3)),
                )],
            ),
            Vec::new(),
        ),
        (
            ReferenceErrorClass::Overflow,
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [u32::MAX, u32::MAX, u32::MAX],
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            ),
            Vec::new(),
        ),
        (
            ReferenceErrorClass::OutOfBoundsAccess,
            indexed_input_copy_program(64),
            vec![Value::from(7u32.to_le_bytes().to_vec())],
        ),
        (
            ReferenceErrorClass::IncompleteDispatchSemantics,
            Program::wrapped(
                vec![
                    BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(1),
                    BufferDecl::storage("dst", 1, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(1),
                ],
                [1, 1, 1],
                vec![Node::AllGather {
                    input: Ident::from("src"),
                    output: Ident::from("dst"),
                    group: CommGroup(7),
                }],
            ),
            vec![
                Value::from(1u32.to_le_bytes().to_vec()),
                Value::from(0u32.to_le_bytes().to_vec()),
            ],
        ),
        (
            ReferenceErrorClass::BudgetExhaustion,
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::Loop {
                    var: Ident::from("i"),
                    from: Expr::u32(0),
                    to: Expr::u32(1_000_000),
                    body: vec![Node::store("out", Expr::u32(0), Expr::var("i"))],
                }],
            ),
            Vec::new(),
        ),
    ]
}

#[test]
fn budget_exhaustion_terminates_and_reports_bound() {
    // Construct a program with a loop that exceeds the work ceiling.
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(1_000_000),
            body: vec![Node::store("out", Expr::u32(0), Expr::var("i"))],
        }],
    );

    // Tight budget of 25 steps.
    let tight_budget = ReferenceBudget::bounded(25);
    let request = ReferenceRequest::new(&program, &[], tight_budget);
    let error = request
        .execute()
        .expect_err("execution must fail on budget exhaustion");

    assert_eq!(
        error.error_class(),
        ReferenceErrorClass::BudgetExhaustion,
        "failure must be classified as BudgetExhaustion"
    );
    let ceiling_source = error
        .step_ceiling_source()
        .expect("must carry StepCeilingExceeded payload");
    assert_eq!(
        ceiling_source.ceiling, 25,
        "reported ceiling must match bound"
    );
}

#[test]
fn permissive_mode_cannot_issue_expected_output_or_certificate() {
    let program = single_input_copy_program();

    let inputs = vec![vyre_reference::value::Value::from(
        DISTINCTIVE_OUTPUT.to_le_bytes().to_vec(),
    )];
    let permissive_request = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard());

    // A permissive run reports what it absorbed and nothing a device could be
    // graded against. Permissive is the method rather than a field, so a strict
    // caller cannot reach a permissive result at all, and the report type it
    // does return carries no output value and no certificate: the bytes a
    // strict run would have produced are not in the type.
    let report = permissive_request
        .execute_permissive()
        .expect("permissive execution succeeds");
    let recorded = format!("{report:?}");
    for byte in DISTINCTIVE_OUTPUT.to_le_bytes() {
        assert!(
            byte == 0 || !recorded.contains(&byte.to_string()),
            "Fix: a permissive report must not carry output bytes; found {byte} in {recorded}"
        );
    }
    assert_eq!(
        report.output_digest.len(),
        64,
        "Fix: a permissive report must summarize its bytes as a digest, got: {}",
        report.output_digest
    );
    assert_eq!(report.oob_report.total(), 0);
}

/// A value no digest, tally, or step count can produce by coincidence.
const DISTINCTIVE_OUTPUT: u32 = 0xDEAD_BEEF;

#[test]
fn single_rank_collectives_interpreted_directly_without_lowering() {
    // Test AllGather direct interpretation
    let allgather_prog = Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::storage("dst", 1, BufferAccess::ReadWrite, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![Node::AllGather {
            input: Ident::from("src"),
            output: Ident::from("dst"),
            group: CommGroup::WORLD,
        }],
    );

    let src_bytes = vec![10u32.to_le_bytes(), 20u32.to_le_bytes()].concat();
    let dst_zeros = vec![0u32.to_le_bytes(), 0u32.to_le_bytes()].concat();

    let allgather_inputs = vec![
        vyre_reference::value::Value::from(src_bytes.clone()),
        vyre_reference::value::Value::from(dst_zeros),
    ];
    let request = ReferenceRequest::new(
        &allgather_prog,
        &allgather_inputs,
        ReferenceBudget::standard(),
    );

    let result = request
        .execute()
        .expect("AllGather must execute directly in reference oracle");
    assert_eq!(result.outputs[0].to_bytes(), src_bytes);

    // Test AllReduce direct interpretation (WORLD)
    let allreduce_prog = Program::wrapped(
        vec![BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(2)],
        [1, 1, 1],
        vec![Node::AllReduce {
            buffer: Ident::from("buf"),
            op: vyre_spec::CollectiveOp::Sum,
            group: CommGroup::WORLD,
        }],
    );
    let allreduce_inputs = vec![vyre_reference::value::Value::from(src_bytes.clone())];
    let request_allreduce = ReferenceRequest::new(
        &allreduce_prog,
        &allreduce_inputs,
        ReferenceBudget::standard(),
    );
    let result_allreduce = request_allreduce
        .execute()
        .expect("AllReduce must execute directly");
    assert_eq!(result_allreduce.outputs[0].to_bytes(), src_bytes);

    // Non-WORLD collective must return IncompleteDispatchSemantics error
    let non_world_prog = Program::wrapped(
        vec![BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(2)],
        [1, 1, 1],
        vec![Node::AllReduce {
            buffer: Ident::from("buf"),
            op: vyre_spec::CollectiveOp::Sum,
            group: CommGroup(42),
        }],
    );
    let non_world_inputs = vec![vyre_reference::value::Value::from(src_bytes)];
    let request_non_world = ReferenceRequest::new(
        &non_world_prog,
        &non_world_inputs,
        ReferenceBudget::standard(),
    );
    let non_world_err = request_non_world
        .execute()
        .expect_err("non-WORLD collective must fail on single-rank oracle");
    assert_eq!(
        non_world_err.error_class(),
        ReferenceErrorClass::IncompleteDispatchSemantics
    );
}

#[test]
fn reference_request_carries_mandatory_budget_and_envelope() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [2, 2, 1],
        vec![Node::store("out", Expr::u32(0), Expr::gid_x())],
    );

    let budget = ReferenceBudget::new(10_000, 1024 * 1024, 64);
    let envelope = WorkloadEnvelope::for_program(&program).with_grid([2, 1, 1]);
    let request = ReferenceRequest::new(&program, &[], budget)
        .with_workload_envelope(envelope)
        .with_schedule_policy(DeterministicSchedulePolicy::LaneReversed);

    assert_eq!(request.budget.work_ceiling, 10_000);
    assert_eq!(request.budget.max_memory_bytes, 1024 * 1024);
    assert_eq!(request.budget.max_recursion_depth, 64);
    assert_eq!(
        request.schedule_policy,
        DeterministicSchedulePolicy::LaneReversed
    );

    let result = request.execute().expect("request must execute");
    assert_eq!(result.certificate.oracle_version, REFERENCE_ORACLE_VERSION);
    assert_eq!(
        result.certificate.schedule_policy,
        DeterministicSchedulePolicy::LaneReversed
    );
}
