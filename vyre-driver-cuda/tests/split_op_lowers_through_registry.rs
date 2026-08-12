//! A registered op whose body calls another registered op must lower.
//!
//! The canonical pre-emit pipeline must resolve composition calls through the
//! sole semantic operation registry. These tests prove a split operation
//! survives verified lowering and emits target code without a driver-owned
//! definition or provider installation.

use vyre_driver::DispatchConfig;
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{OperationRegistration, OperationTier};

const CALLEE_OP_ID: &str = "test::split_lowering::row_pair_sum";
const CALLER_OP_ID: &str = "test::split_lowering::caller";

/// Callee phase: a scalar computation over the value it is handed.
fn row_pair_sum() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("row", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [64, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::add(
                Expr::mul(Expr::load("row", Expr::u32(0)), Expr::u32(3)),
                Expr::u32(7),
            ),
        )],
    )
    .with_entry_op_id(CALLEE_OP_ID)
}

/// Caller: the shape an over-budget op takes after a split. It reads the
/// invocation id itself and hoists it to the call site as an argument,
/// because a callee body may not read a per-invocation built-in.
fn caller() -> Program {
    let row = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("nodes", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(64),
        ],
        [64, 1, 1],
        vec![Node::if_then(
            Expr::lt(row.clone(), Expr::u32(32)),
            vec![Node::store(
                "out",
                row.clone(),
                Expr::call(CALLEE_OP_ID, vec![row]),
            )],
        )],
    )
    .with_entry_op_id(CALLER_OP_ID)
}

const CALLEE_SIG: Signature = Signature {
    inputs: &[TypedParam {
        name: "row",
        ty: "u32",
    }],
    outputs: &[TypedParam {
        name: "result",
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

inventory::submit! {
    OperationRegistration::new(
        CALLEE_OP_ID,
        OperationTier::External,
        Some(row_pair_sum),
        None,
        None,
    )
    .with_signature(CALLEE_SIG)
    .with_category("test")
}

/// The call resolves and disappears. Before the fix this failed with
/// `InlineUnknownOp` naming the callee.
#[test]
fn a_call_to_a_registered_op_resolves_in_the_pre_emit_pipeline() {
    let prepared = vyre_lower::lower_verified(&caller())
        .unwrap_or_else(|error| panic!("split op must survive verified lowering: {error}"))
        .program;
    let dump = format!("{:?}", prepared.entry());
    assert!(
        !dump.contains("Call {"),
        "every call must be inlined out: {dump}"
    );
}

/// The callee's own input buffer must not survive into the caller: the
/// argument replaces every read of it.
#[test]
fn the_callees_input_buffer_does_not_leak_into_the_caller() {
    let prepared = vyre_lower::lower_verified(&caller())
        .expect("verified lowering")
        .program;
    let dump = format!("{:?}", prepared.entry());
    assert!(
        !dump.contains("\"row\""),
        "no read may still name the callee's own buffer: {dump}"
    );
}

/// End to end: a split op emits real PTX, and the store address is
/// register-computed because the invocation id reached the store intact.
#[test]
fn a_split_op_emits_ptx() {
    let ptx = vyre_driver_cuda::codegen::program_to_ptx(&caller(), &DispatchConfig::default())
        .unwrap_or_else(|error| panic!("split op must emit PTX: {error}"));
    let stores: Vec<&str> = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("st.global"))
        .collect();
    assert_eq!(stores.len(), 1, "one store, got {stores:?}");
    assert!(
        !stores[0].contains('+'),
        "the store address must be register-computed, not a constant offset: {}",
        stores[0]
    );
}
