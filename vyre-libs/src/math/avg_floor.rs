//! Floor of the average of two u32 lanes, computed without overflowing.
//!
//! `(a & b) + ((a ^ b) >> 1)` is exact for the whole u32 range, where
//! `(a + b) / 2` wraps.

use vyre_foundation::ir::{Expr, Program};

const OP_ID: &str = "vyre-libs::math::avg_floor";

/// Computes average floor.
#[must_use]
pub fn avg_floor(a: &str, b: &str, out: &str, size: u32) -> Program {
    crate::builder::elementwise::u32_elementwise_binary(OP_ID, a, b, out, size, |lx, rx| {
        Expr::add(
            Expr::bitand(lx.clone(), rx.clone()),
            Expr::shr(Expr::bitxor(lx, rx), Expr::u32(1)),
        )
    })
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || avg_floor("a", "b", "out", 4),
        Some(|| {
            let a = [10u32, u32::MAX, 7, 100];
            let b = [20u32, u32::MAX, 12, 0];
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![vec![to_bytes(&a), to_bytes(&b)]]
        }),
        Some(|| {
            // (10,20)->15, (MAX,MAX)->MAX, (7,12)->9, (100,0)->50.
            vec![vec![vec![
                0x0f, 0x00, 0x00, 0x00, // 15
                0xff, 0xff, 0xff, 0xff, // u32::MAX
                0x09, 0x00, 0x00, 0x00, // 9
                0x32, 0x00, 0x00, 0x00, // 50
            ]]]
        }),
    )
    .with_category("math")
}
