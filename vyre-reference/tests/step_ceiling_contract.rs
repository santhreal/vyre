//! WHY: the reference interpreter bounded the memory a program may ask for and
//! not the work it may do, so a program with a data-derived trip count ran until
//! it finished. The parity oracle had no termination contract, and the stress
//! sweep bounded it from outside on a thread it abandoned, which is a caller-side
//! workaround rather than a contract.
//!
//! Closes: the interpreter's termination contract. Every public entry point
//! reaches one arming site, so a refusal names the program and the ceiling it
//! exceeded and a caller selects on `step_ceiling_source` instead of matching a
//! message. The charge sites are the statement driver, the loop iteration
//! boundary, and the scheduler round, so a loop with an empty body and a
//! barrier-release cycle that advances no statement are both bounded.
//!
//! Does not catch: whether `MAX_REFERENCE_STEPS` is high enough for the corpus.
//! That is measured against every registered fixture in
//! `vyre-libs/tests/reference_step_ceiling_corpus.rs`, which is where the
//! registry is linked. It also does not measure the dense contraction the
//! admission below exists for: `vyre-bench`'s micro-case reference proof
//! evaluates that program at its shipped size.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::step_budget::MAX_REFERENCE_STEPS;
use vyre_reference::value::Value;

/// `for i in 0..load(trip, 0) { out[0] = i }`: the trip count is data, so no
/// declared extent bounds it.
fn data_derived_trip_count() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("trip", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::loop_(
            "i",
            Expr::u32(0),
            Expr::load("trip", Expr::u32(0)),
            vec![Node::store("out", Expr::u32(0), Expr::var("i"))],
        )],
    )
}

/// The same trip count over an empty body, which executes no statement.
fn data_derived_trip_count_empty_body() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("trip", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::loop_(
                "i",
                Expr::u32(0),
                Expr::load("trip", Expr::u32(0)),
                Vec::new(),
            ),
            Node::store("out", Expr::u32(0), Expr::u32(1)),
        ],
    )
}

fn trip_inputs(trip: u32) -> Vec<Value> {
    vec![Value::from(trip.to_le_bytes().to_vec())]
}

#[test]
fn a_data_derived_trip_count_is_refused_by_name() {
    let program = data_derived_trip_count();
    let start = std::time::Instant::now();
    let error = vyre_reference::ReferenceRequest::new(
        &program,
        &trip_inputs(u32::MAX),
        vyre_reference::ReferenceBudget::with_work_ceiling(4_096),
    )
    .outputs_and_steps()
    .expect_err("Fix: a trip count no declared extent bounds must reach the work ceiling");
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "Fix: the interpreter must refuse an unbounded trip count promptly; elapsed {:?}",
        start.elapsed()
    );
    let source = error.step_ceiling_source().expect(
        "Fix: a work-ceiling refusal must carry the ceiling it exceeded, not only a message",
    );
    assert_eq!(
        source.ceiling, 4_096,
        "Fix: the refusal must state the ceiling the run was given"
    );
    assert!(
        !source.program.is_empty(),
        "Fix: the refusal must name the program that reached the ceiling"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains(&source.program) && rendered.contains("4096"),
        "Fix: the rendered message must name the program and the ceiling, got {rendered}"
    );
}

#[test]
fn an_anonymous_program_is_named_by_its_fingerprint() {
    let program = data_derived_trip_count();
    assert!(
        program.entry_op_id.is_none(),
        "Fix: this fixture exists to exercise the unnamed-program path"
    );
    let error = vyre_reference::ReferenceRequest::new(
        &program,
        &trip_inputs(u32::MAX),
        vyre_reference::ReferenceBudget::with_work_ceiling(512),
    )
    .outputs_and_steps()
    .expect_err("Fix: the run must reach the ceiling");
    let named = &error
        .step_ceiling_source()
        .expect("Fix: a work-ceiling refusal carries its source")
        .program;
    assert!(
        named.starts_with("0x") && named.len() == 18,
        "Fix: an unnamed program is refused under a fingerprint prefix, got {named}"
    );
}

#[test]
fn a_named_program_is_refused_under_its_entry_op_id() {
    let mut program = data_derived_trip_count();
    program.entry_op_id = Some("hostile::trip_count_op".to_string());
    let start = std::time::Instant::now();
    let error = vyre_reference::ReferenceRequest::new(
        &program,
        &trip_inputs(u32::MAX),
        vyre_reference::ReferenceBudget::with_work_ceiling(4_096),
    )
    .outputs_and_steps()
    .expect_err("Fix: a hostile trip count no declared extent bounds must reach the work ceiling");
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "Fix: the interpreter must refuse an unbounded trip count promptly; elapsed {:?}",
        start.elapsed()
    );
    let source = error.step_ceiling_source().expect(
        "Fix: a work-ceiling refusal must carry the ceiling it exceeded, not only a message",
    );
    assert_eq!(source.ceiling, 4_096);
    assert_eq!(source.program, "hostile::trip_count_op");
    let rendered = error.to_string();
    assert!(
        rendered.contains("hostile::trip_count_op") && rendered.contains("4096"),
        "Fix: the rendered message must name the program and the ceiling, got {rendered}"
    );
}

#[test]
fn an_empty_loop_body_is_bounded_too() {
    let error = vyre_reference::ReferenceRequest::new(
        &data_derived_trip_count_empty_body(),
        &trip_inputs(u32::MAX),
        vyre_reference::ReferenceBudget::with_work_ceiling(1_024),
    )
    .outputs_and_steps()
    .expect_err("Fix: a loop whose body executes no statement must still charge its iterations");
    assert_eq!(
        error
            .step_ceiling_source()
            .expect("Fix: a work-ceiling refusal carries its source")
            .ceiling,
        1_024
    );
}

#[test]
fn a_bounded_trip_count_evaluates_and_reports_its_steps() {
    let (outputs, steps) = vyre_reference::ReferenceRequest::new(
        &data_derived_trip_count(),
        &trip_inputs(64),
        vyre_reference::ReferenceBudget::with_work_ceiling(4_096),
    )
    .outputs_and_steps()
    .expect("Fix: a trip count the ceiling admits must evaluate");
    assert_eq!(
        outputs[0].to_bytes(),
        63u32.to_le_bytes().to_vec(),
        "Fix: bounding the work must not change what the program computes"
    );
    assert!(
        steps >= 64,
        "Fix: 64 iterations charge at least one step each, got {steps}"
    );
    assert!(
        steps < 4_096,
        "Fix: a run the ceiling admits must report fewer steps than the ceiling, got {steps}"
    );
}

#[test]
fn the_default_entry_point_charges_against_the_shipped_ceiling() {
    let (_outputs, steps) =
        vyre_reference::ReferenceRequest::standard(&data_derived_trip_count(), &trip_inputs(8))
            .outputs_and_steps()
            .expect("Fix: a small program must evaluate under the shipped ceiling");
    assert!(steps > 0, "Fix: an evaluated program charges steps");
    assert!(
        steps < MAX_REFERENCE_STEPS,
        "Fix: the shipped ceiling must admit an eight-iteration loop"
    );
}

/// `for i in 0..trips { out[0] = i }` over one invocation, with the trip count
/// declared as a literal.
fn declared_trip_count(trips: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::loop_(
            "i",
            Expr::u32(0),
            Expr::u32(trips),
            vec![Node::store("out", Expr::u32(0), Expr::var("i"))],
        )],
    )
}

#[test]
fn a_declared_extent_above_the_ceiling_is_admitted_at_the_work_it_declares() {
    let trips = 4_096u32;
    let ceiling = 1_024u64;
    let (outputs, steps) =
        vyre_reference::ReferenceRequest::new(&declared_trip_count(trips), &[], vyre_reference::ReferenceBudget::with_work_ceiling(ceiling)).outputs_and_steps().expect(
            "Fix: a program whose trip count is a declared literal must be admitted at the work it declares, not refused against a ceiling sized for a smaller program",
        );
    assert_eq!(
        outputs[0].to_bytes(),
        (trips - 1).to_le_bytes().to_vec(),
        "Fix: admitting the declared work must not change what the program computes"
    );
    assert!(
        steps > ceiling,
        "Fix: this case only proves admission while the run charges more than the ceiling it was armed with, got {steps} steps against {ceiling}"
    );
}

#[test]
fn admission_does_not_lift_the_ceiling_for_a_data_derived_trip_count() {
    let error = vyre_reference::ReferenceRequest::new(&data_derived_trip_count(), &trip_inputs(u32::MAX), vyre_reference::ReferenceBudget::with_work_ceiling(1_024)).outputs_and_steps()
        .expect_err(
            "Fix: admitting declared work must not admit a trip count nothing declares; the refusal is what bounds the oracle",
        );
    assert_eq!(
        error
            .step_ceiling_source()
            .expect("Fix: a work-ceiling refusal carries its source")
            .ceiling,
        1_024,
        "Fix: the refusal must state the ceiling the run was armed with, unraised"
    );
}

#[test]
fn a_declared_extent_inside_a_data_derived_loop_declares_nothing() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("trip", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::loop_(
            "outer",
            Expr::u32(0),
            Expr::load("trip", Expr::u32(0)),
            vec![Node::loop_(
                "inner",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("out", Expr::u32(0), Expr::var("inner"))],
            )],
        )],
    );
    let error = vyre_reference::ReferenceRequest::new(&program, &trip_inputs(u32::MAX), vyre_reference::ReferenceBudget::with_work_ceiling(2_048)).outputs_and_steps()
        .expect_err(
            "Fix: one undeclared trip count leaves the body's work undeclared, whatever its inner loops declare",
        );
    assert_eq!(
        error
            .step_ceiling_source()
            .expect("Fix: a work-ceiling refusal carries its source")
            .ceiling,
        2_048
    );
}
#[test]
fn a_single_invocation_with_an_unbounded_inner_loop_is_refused_by_the_statement_driver() {
    let program = data_derived_trip_count();
    assert_eq!(program.workgroup_size(), [1, 1, 1]);
    let start = std::time::Instant::now();
    let error = vyre_reference::ReferenceRequest::new(&program, &trip_inputs(u32::MAX), vyre_reference::ReferenceBudget::with_work_ceiling(512)).outputs_and_steps()
        .expect_err("Fix: a single invocation with an unbounded inner loop must be refused by the statement driver");
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "Fix: statement driver step ceiling must terminate promptly; elapsed {:?}",
        start.elapsed()
    );
    let source = error
        .step_ceiling_source()
        .expect("Fix: refusal carries step ceiling source");
    assert_eq!(source.ceiling, 512);
}

#[test]
fn an_unbounded_invocation_loop_is_refused_by_the_invocation_driver() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("trip", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(100),
        ],
        [1, 1, 1],
        vec![Node::loop_(
            "i",
            Expr::u32(0),
            Expr::load("trip", Expr::u32(0)),
            vec![Node::store("out", Expr::gid_x(), Expr::u32(1))],
        )],
    );
    let start = std::time::Instant::now();
    let error = vyre_reference::ReferenceRequest::new(
        &program,
        &trip_inputs(1),
        vyre_reference::ReferenceBudget::with_work_ceiling(10),
    )
    .outputs_and_steps()
    .expect_err("Fix: multi-invocation program exceeding step ceiling must be refused");
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "Fix: invocation driver ceiling must terminate promptly; elapsed {:?}",
        start.elapsed()
    );
    let source = error
        .step_ceiling_source()
        .expect("Fix: refusal carries step ceiling source");
    assert_eq!(source.ceiling, 10);
}

#[test]
fn the_heaviest_legitimate_corpus_work_completes_under_the_ceiling_with_margin() {
    assert!(
        vyre_reference::step_budget::MEASURED_HEAVIEST_CORPUS_STEPS > 0,
        "Fix: measured heaviest corpus steps must be non-zero"
    );
    assert_eq!(
        vyre_reference::step_budget::MAX_REFERENCE_STEPS,
        vyre_reference::step_budget::MEASURED_HEAVIEST_CORPUS_STEPS
            * vyre_reference::step_budget::STEP_CEILING_HEADROOM,
        "Fix: step ceiling must maintain the measured headroom multiple"
    );
    assert!(
        vyre_reference::step_budget::STEP_CEILING_HEADROOM >= 16,
        "Fix: headroom multiple must be generous enough to admit legitimate workloads with margin"
    );
    let trips = 10_000u32;
    let program = declared_trip_count(trips);
    let (outputs, steps) = vyre_reference::ReferenceRequest::standard(&program, &[])
        .outputs_and_steps()
        .expect("Fix: legitimate program must complete under shipped ceiling");
    assert_eq!(outputs[0].to_bytes(), (trips - 1).to_le_bytes().to_vec());
    assert!(
        steps < MAX_REFERENCE_STEPS,
        "Fix: legitimate workload must complete under ceiling with margin, charged {steps} against ceiling {MAX_REFERENCE_STEPS}"
    );
}
