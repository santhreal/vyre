//! Verification contracts for Row 86: typed ReferenceRequest, mandatory budget, and OOB load refusal.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_reference::{
    reference_eval, OutOfBoundsOp, ReferenceBudget, ReferenceRequest, ReferenceRequestError,
    ScheduleExplorationPolicy, Value, WorkloadEnvelope,
};

fn u32_bytes(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn simple_add_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(4),
            BufferDecl::read("b", 1, DataType::U32).with_count(4),
            BufferDecl::output("out", 2, DataType::U32).with_count(4),
        ],
        [4, 1, 1],
        vec![
            Node::Let {
                name: Ident::from("idx"),
                value: Expr::InvocationId { axis: 0 },
            },
            Node::Let {
                name: Ident::from("val_a"),
                value: Expr::load("a", Expr::var("idx")),
            },
            Node::Let {
                name: Ident::from("val_b"),
                value: Expr::load("b", Expr::var("idx")),
            },
            Node::Let {
                name: Ident::from("sum"),
                value: Expr::add(Expr::var("val_a"), Expr::var("val_b")),
            },
            Node::Store {
                buffer: Ident::from("out"),
                index: Expr::var("idx"),
                value: Expr::var("sum"),
            },
        ],
    )
}

fn oob_load_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("small_buf", 0, DataType::U32).with_count(2),
            BufferDecl::output("out", 1, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![
            Node::Let {
                name: Ident::from("val"),
                value: Expr::load("small_buf", Expr::u32(5)),
            },
            Node::Store {
                buffer: Ident::from("out"),
                index: Expr::u32(0),
                value: Expr::var("val"),
            },
        ],
    )
}


#[test]
fn a_request_without_a_budget_does_not_construct() {
    let program = simple_add_program();
    let inputs = vec![
        Value::from(u32_bytes(&[10, 20, 30, 40])),
        Value::from(u32_bytes(&[1, 2, 3, 4])),
    ];

    // Builder without .budget() must fail construction
    let builder = ReferenceRequest::builder(&program, &inputs);
    let err = builder
        .build()
        .expect_err("ReferenceRequest::builder without budget must fail to construct");
    assert_eq!(err, ReferenceRequestError::MissingBudget);
    assert!(err.to_string().contains("mandatory ReferenceBudget"));

    // Builder with explicit budget must succeed
    let request = ReferenceRequest::builder(&program, &inputs)
        .budget(ReferenceBudget::standard())
        .build()
        .expect("ReferenceRequest with budget must construct successfully");
    assert_eq!(request.budget, ReferenceBudget::standard());
}

#[test]
fn an_out_of_bounds_load_is_refused_naming_buffer_index_and_extent() {
    let program = oob_load_program();
    let inputs = vec![Value::from(u32_bytes(&[100, 200]))]; // 2 elements
    let request = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard());

    let err = reference_eval(&request)
        .expect_err("out-of-bounds load at index 5 with extent 2 must be refused");
    let oob = err
        .out_of_bounds_source()
        .expect("ReferenceError must contain structured OutOfBoundsAccess");

    assert_eq!(oob.buffer, "small_buf");
    assert_eq!(oob.index, 5);
    assert_eq!(oob.extent, 2);
    assert_eq!(oob.operation, OutOfBoundsOp::Load);

    let message = err.to_string();
    assert!(
        message.contains("small_buf"),
        "error message must name the buffer: {message}"
    );
    assert!(
        message.contains("5"),
        "error message must name the index: {message}"
    );
    assert!(
        message.contains("2"),
        "error message must name the extent: {message}"
    );
}

#[test]
fn the_same_graph_and_abi_produce_the_same_answer_through_the_new_entry_point() {
    let program = simple_add_program();
    let inputs = vec![
        Value::from(u32_bytes(&[10, 20, 30, 40])),
        Value::from(u32_bytes(&[1, 2, 3, 4])),
    ];
    let expected = vec![Value::from(u32_bytes(&[11, 22, 33, 44]))];

    // Standard execution through ReferenceRequest
    let req = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard());
    let response = reference_eval(&req).expect("reference evaluation must succeed");
    assert_eq!(response.outputs(), &expected);
    assert!(response.steps_charged > 0);

    // Schedule exploration: LaneReversed
    let req_reversed = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard())
        .with_schedule(ScheduleExplorationPolicy::LaneReversed);
    let resp_reversed = req_reversed.execute().expect("reversed schedule must succeed");
    assert_eq!(resp_reversed.outputs(), &expected);

    // Schedule exploration: LaneRotated
    let req_rotated = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard())
        .with_schedule(ScheduleExplorationPolicy::LaneRotated(2));
    let resp_rotated = req_rotated.execute().expect("rotated schedule must succeed");
    assert_eq!(resp_rotated.outputs(), &expected);

    // Workload envelope with explicit grid
    let req_grid = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard())
        .with_envelope(WorkloadEnvelope::new().with_grid([4, 1, 1]));
    let resp_grid = req_grid.execute().expect("grid dispatch must succeed");
    assert_eq!(resp_grid.outputs(), &expected);
}
