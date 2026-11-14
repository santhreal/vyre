//! Every capacity a neutral rewrite needs is a fact the caller states.
//!
//! Constant capacity differs between targets. A default stored in this crate
//! would promote a binding on a device that cannot hold it, while looking like
//! a valid transform. The cases below pin that the stated value decides the
//! verdict and that an unstated fact yields no transform rather than one
//! computed from an assumed limit.

use std::num::NonZeroU32;

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::analyses::AnalysisFacts;
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, fixed_global_ro, global_rw, lit, op,
};
use vyre_lower::rewrites::apply_lowering_rewrites;
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue, MemoryClass};

fn nonzero(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("a stated capacity is nonzero")
}

/// A fixed 64-element read-only global read twice, the shape the
/// constant-buffer rule admits.
fn twice_read_lookup_table() -> KernelDescriptor {
    descriptor("lookup")
        .slot(fixed_global_ro(0, DataType::U32, 64, "lut"))
        .slot(global_rw(1, DataType::U32, "dest"))
        .dispatch(64, 1, 1)
        .body(body().literals([LiteralValue::U32(0)]).ops([
            lit(0, 0),
            op(KernelOpKind::LoadGlobal, [0, 0], 1),
            op(KernelOpKind::LoadGlobal, [0, 0], 2),
            op(KernelOpKind::BinOpKind(BinOp::Add), [1, 2], 3),
            effect(KernelOpKind::StoreGlobal, [1, 0, 3]),
        ]))
        .build()
}

#[test]
fn constant_promotion_waits_for_a_stated_constant_capacity() {
    let kernel = twice_read_lookup_table();

    let unstated = apply_lowering_rewrites(&kernel, &AnalysisFacts::none());
    let stated = apply_lowering_rewrites(
        &kernel,
        &AnalysisFacts::none().with_constant_buffer_bytes(nonzero(64 * 1024)),
    );

    assert_eq!(
        unstated.bindings.slots[0].memory_class,
        MemoryClass::Global,
        "no reported constant capacity means no promotion"
    );
    assert_eq!(stated.bindings.slots[0].memory_class, MemoryClass::Constant);
}

#[test]
fn the_stated_constant_capacity_decides_eligibility() {
    let kernel = twice_read_lookup_table();

    // The binding is 64 elements of u32: 256 bytes.
    let too_small = apply_lowering_rewrites(
        &kernel,
        &AnalysisFacts::none().with_constant_buffer_bytes(nonzero(128)),
    );
    let large_enough = apply_lowering_rewrites(
        &kernel,
        &AnalysisFacts::none().with_constant_buffer_bytes(nonzero(256)),
    );

    assert_eq!(
        too_small.bindings.slots[0].memory_class,
        MemoryClass::Global,
        "a 256-byte binding does not fit a stated 128-byte capacity"
    );
    assert_eq!(
        large_enough.bindings.slots[0].memory_class,
        MemoryClass::Constant
    );
}
