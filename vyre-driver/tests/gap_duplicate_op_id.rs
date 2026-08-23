//! Semantic operation duplicate-ID admission contract.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{OperationRegistration, OperationRegistry, OperationTier};

const DUPLICATE_ID: &str = "external_fixture::duplicate_operation::identity";

fn duplicate_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("output", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("output", Expr::u32(0), Expr::u32(7))],
    )
}

inventory::submit! {
    OperationRegistration::new_unconstrained(
        DUPLICATE_ID,
        OperationTier::External,
        Some(duplicate_program),
        None,
        None,
    )
}

inventory::submit! {
    OperationRegistration::new_unconstrained(
        DUPLICATE_ID,
        OperationTier::External,
        Some(duplicate_program),
        None,
        None,
    )
}

/// WHY: one semantic identity must never depend on linked-inventory order.
#[test]
fn duplicate_operation_id_is_rejected_before_lookup() {
    let panic = std::panic::catch_unwind(|| OperationRegistry::global().get(DUPLICATE_ID))
        .expect_err("duplicate semantic operation IDs must reject registry construction");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("registry panic must provide an actionable message");

    assert!(
        message.contains(&format!(
            "duplicate operation registration `{DUPLICATE_ID}`; keep exactly one semantic owner"
        )),
        "duplicate operation rejection must identify the stable ID and corrective action: {message}"
    );
}
