//! Test: memory vector.
use super::*;
use vyre_lower::descriptor_builder::{effect, lit, op};

fn dynamic_reassociated_vector_load_kernel(seed: u32) -> KernelDescriptor {
    let stride = seed.wrapping_mul(13).wrapping_add(1) << 2;
    two_slot_u32_kernel(
        "dynamic_reassociated_vec_load",
        vec![
            op(KernelOpKind::LocalInvocationId, [0], 0),
            lit(0, 1),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 2),
            lit(1, 3),
            lit(2, 4),
            lit(3, 5),
            op(KernelOpKind::LoadGlobal, [0, 2], 6),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 3], 7),
            op(KernelOpKind::LoadGlobal, [0, 7], 8),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 4], 9),
            op(KernelOpKind::LoadGlobal, [0, 9], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 5], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 12),
            lit(4, 13),
            effect(KernelOpKind::StoreGlobal, [1, 13, 12]),
        ],
        vec![
            LiteralValue::U32(stride),
            LiteralValue::U32(1),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
            LiteralValue::U32(0),
        ],
    )
}

fn dynamic_reassociated_vector_store_kernel(seed: u32) -> KernelDescriptor {
    let stride = seed.wrapping_mul(17).wrapping_add(2) << 2;
    let value_base = 0x1000_0000_u32.wrapping_add(seed.rotate_left(seed % 31));
    two_slot_u32_kernel(
        "dynamic_reassociated_vec_store",
        vec![
            op(KernelOpKind::LocalInvocationId, [0], 0),
            lit(0, 1),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 2),
            lit(1, 3),
            lit(2, 4),
            lit(3, 5),
            lit(4, 6),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 6], 7),
            lit(5, 8),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 8], 9),
            lit(6, 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            lit(7, 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            effect(KernelOpKind::StoreGlobal, [1, 2, 7]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 3], 14),
            effect(KernelOpKind::StoreGlobal, [1, 14, 9]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 4], 15),
            effect(KernelOpKind::StoreGlobal, [1, 15, 11]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 5], 16),
            effect(KernelOpKind::StoreGlobal, [1, 16, 13]),
        ],
        vec![
            LiteralValue::U32(stride),
            LiteralValue::U32(1),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
            LiteralValue::U32(value_base),
            LiteralValue::U32(value_base.wrapping_add(1)),
            LiteralValue::U32(value_base.wrapping_add(2)),
            LiteralValue::U32(value_base.wrapping_add(3)),
        ],
    )
}

fn dynamic_misaligned_gather_to_vector_store_kernel() -> KernelDescriptor {
    two_slot_u32_kernel(
        "dynamic_misaligned_gather_to_vec_store",
        vec![
            op(KernelOpKind::LocalInvocationId, [0], 0),
            lit(0, 1),
            lit(1, 2),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 3),
            lit(2, 4),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 4], 5),
            op(KernelOpKind::LoadGlobal, [0, 3], 6),
            op(KernelOpKind::BinOpKind(BinOp::Add), [3, 2], 7),
            op(KernelOpKind::LoadGlobal, [0, 7], 8),
            lit(3, 9),
            op(KernelOpKind::BinOpKind(BinOp::Add), [3, 9], 10),
            op(KernelOpKind::LoadGlobal, [0, 10], 11),
            lit(4, 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [3, 12], 13),
            op(KernelOpKind::LoadGlobal, [0, 13], 14),
            effect(KernelOpKind::StoreGlobal, [1, 5, 6]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [5, 2], 15),
            effect(KernelOpKind::StoreGlobal, [1, 15, 8]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [5, 9], 16),
            effect(KernelOpKind::StoreGlobal, [1, 16, 11]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [5, 12], 17),
            effect(KernelOpKind::StoreGlobal, [1, 17, 14]),
        ],
        vec![
            LiteralValue::U32(5),
            LiteralValue::U32(1),
            LiteralValue::U32(4),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
        ],
    )
}

mod load_cache_hoist_contracts;
mod load_vector_contracts;
mod store_pruning_contracts;
mod store_vector_contracts;
