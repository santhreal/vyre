//! Atomic LRU update: safely update access timestamps/priority in a shared buffer.
//!
//! Category-B composition over `AtomicOp::Max`.

use crate::region::wrap_anonymous;
use vyre_foundation::ir::{AtomicOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::memory_model::MemoryOrdering;

/// Build a Program that atomically updates an LRU slot.
#[must_use]
pub fn atomic_lru_update_u32(buffer: &str, index: Expr, timestamp: Expr) -> Program {
    let body = vec![
        Node::let_bind("idx", index),
        Node::let_bind("ts", timestamp),
        Node::let_bind(
            "_prev",
            Expr::Atomic {
                op: AtomicOp::Max,
                buffer: buffer.into(),
                index: Box::new(Expr::var("idx")),
                expected: None,
                value: Box::new(Expr::var("ts")),
                ordering: MemoryOrdering::SeqCst,
            },
        ),
    ];

    Program::wrapped(
        vec![BufferDecl::storage(buffer, 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::math::atomic::lru_update_u32",
            body,
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::math::atomic::lru_update_u32",
        || atomic_lru_update_u32("buffer", Expr::u32(0), Expr::u32(12345)),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![vec![
                to_bytes(&[0u32]), // buffer (single slot, initial value 0)
            ]]
        }),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            // Single lane writes timestamp 12345 into slot 0.
            vec![vec![to_bytes(&[12345u32])]]
        }),
    )
    .with_category("math")
}
