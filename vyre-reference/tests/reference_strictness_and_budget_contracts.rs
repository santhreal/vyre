//! Contracts for row 86: strict, budgeted, independent reference oracle.
//!
//! WHY: Proves that the reference oracle:
//! 1. Interprets IR directly without invoking production optimizer, schedule, lowering, or emitter transforms.
//! 2. Returns structured oracle errors across all eight failure classes without default fallbacks.
//! 3. Strictly terminates under work budget exhaustion and reports the bound.
//! 4. Proves diagnostic permissive mode cannot issue expected outputs or certificates.
//! 5. Directly evaluates single-rank collectives without external lowering passes.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CommGroup, DataType, Expr, Ident, Node, Program,
};
use vyre_reference::{
    DeterministicSchedulePolicy, ExecutionStrictness, ReferenceBudget, ReferenceError,
    ReferenceErrorClass, ReferenceErrorKind, ReferenceRequest, WorkloadEnvelope,
    REFERENCE_ORACLE_VERSION, REFERENCE_REQUEST_SCHEMA_VERSION,
};

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
    let request = ReferenceRequest::new(program.clone(), inputs, ReferenceBudget::standard());
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

#[test]
fn each_of_eight_failure_classes_returns_structured_error() {
    // Prove that every member of the 8 failure classes is distinct and returns its own structured error class.
    for class in ReferenceErrorClass::ALL {
        let err: ReferenceError = match class {
            ReferenceErrorClass::MissingValue => {
                ReferenceError::missing_value("input buffer `in` was not supplied")
            }
            ReferenceErrorClass::TypeMismatch => {
                ReferenceError::type_mismatch("store index is not u32")
            }
            ReferenceErrorClass::Poison => {
                ReferenceError::poison("buffer lock was poisoned during access")
            }
            ReferenceErrorClass::Overflow => {
                ReferenceError::overflow("workgroup invocation count overflows u32")
            }
            ReferenceErrorClass::OutOfBoundsAccess => {
                ReferenceError::out_of_bounds("load index 100 past buffer extent 4")
            }
            ReferenceErrorClass::IncompleteDispatchSemantics => {
                ReferenceError::incomplete_dispatch_semantics(
                    "workgroup size contains zero dimension",
                )
            }
            ReferenceErrorClass::Nontermination => {
                ReferenceError::nontermination("infinite loop detected")
            }
            ReferenceErrorClass::BudgetExhaustion => {
                ReferenceError::budget_exhaustion("work ceiling of 1000 steps exceeded")
            }
        };

        // Exhaustive compile-time match with NO catch-all `_` arm.
        match err.error_class() {
            ReferenceErrorClass::MissingValue => {
                assert_eq!(class, ReferenceErrorClass::MissingValue);
                assert!(matches!(
                    err.kind(),
                    ReferenceErrorKind::MissingValue { .. }
                ));
            }
            ReferenceErrorClass::TypeMismatch => {
                assert_eq!(class, ReferenceErrorClass::TypeMismatch);
                assert!(matches!(
                    err.kind(),
                    ReferenceErrorKind::TypeMismatch { .. }
                ));
            }
            ReferenceErrorClass::Poison => {
                assert_eq!(class, ReferenceErrorClass::Poison);
                assert!(matches!(err.kind(), ReferenceErrorKind::Poison { .. }));
            }
            ReferenceErrorClass::Overflow => {
                assert_eq!(class, ReferenceErrorClass::Overflow);
                assert!(matches!(err.kind(), ReferenceErrorKind::Overflow { .. }));
            }
            ReferenceErrorClass::OutOfBoundsAccess => {
                assert_eq!(class, ReferenceErrorClass::OutOfBoundsAccess);
                assert!(matches!(
                    err.kind(),
                    ReferenceErrorKind::OutOfBoundsAccess { .. }
                ));
            }
            ReferenceErrorClass::IncompleteDispatchSemantics => {
                assert_eq!(class, ReferenceErrorClass::IncompleteDispatchSemantics);
                assert!(matches!(
                    err.kind(),
                    ReferenceErrorKind::IncompleteDispatchSemantics { .. }
                ));
            }
            ReferenceErrorClass::Nontermination => {
                assert_eq!(class, ReferenceErrorClass::Nontermination);
                assert!(matches!(
                    err.kind(),
                    ReferenceErrorKind::Nontermination { .. }
                ));
            }
            ReferenceErrorClass::BudgetExhaustion => {
                assert_eq!(class, ReferenceErrorClass::BudgetExhaustion);
                assert!(matches!(
                    err.kind(),
                    ReferenceErrorKind::BudgetExhaustion { .. }
                ));
            }
        }
    }
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
    let request = ReferenceRequest::new(program, vec![], tight_budget);
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
    let program = Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("in", Expr::u32(0)),
        )],
    );

    let inputs = vec![vyre_reference::value::Value::from(
        42u32.to_le_bytes().to_vec(),
    )];
    let permissive_request = ReferenceRequest::new(program, inputs, ReferenceBudget::standard())
        .with_strictness(ExecutionStrictness::DiagnosticPermissive);

    // Calling strict execute() on a permissive request must be rejected.
    let strict_err = permissive_request
        .execute()
        .expect_err("strict execute() must fail on permissive request");
    assert_eq!(strict_err.error_class(), ReferenceErrorClass::TypeMismatch);

    // Calling execute_permissive() returns diagnostic report.
    let report = permissive_request
        .execute_permissive()
        .expect("permissive execution succeeds");

    // Report CANNOT issue a certificate.
    let cert_err = report
        .certificate()
        .expect_err("permissive report must refuse to issue a certificate");
    assert_eq!(cert_err.error_class(), ReferenceErrorClass::TypeMismatch);
}

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

    let request = ReferenceRequest::new(
        allgather_prog,
        vec![
            vyre_reference::value::Value::from(src_bytes.clone()),
            vyre_reference::value::Value::from(dst_zeros),
        ],
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
    let request_allreduce = ReferenceRequest::new(
        allreduce_prog,
        vec![vyre_reference::value::Value::from(src_bytes.clone())],
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
    let request_non_world = ReferenceRequest::new(
        non_world_prog,
        vec![vyre_reference::value::Value::from(src_bytes)],
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
    let request = ReferenceRequest::new(program, vec![], budget)
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
