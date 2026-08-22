//! Regression tests for VYRE-PTX-001, VYRE-PTX-002, VYRE-PTX-003.
//!
//! Each test asserts the exact PTX instruction suffix/mnemonic that the
//! fix introduced, confirming the pre-fix behaviour is gone.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::descriptor_builder::{
    binop, body, descriptor, global_rw, lit, load_global, op, store_global,
};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

/// Verify then emit, reporting `what` if either step rejects the descriptor.
fn emit(desc: &KernelDescriptor, what: &str) -> String {
    let verified = vyre_lower::verify_descriptor(desc).expect("descriptor verification");
    vyre_emit_ptx::emit(&verified).unwrap_or_else(|e| panic!("{what} must emit without error: {e}"))
}

fn thread_id(result: u32) -> vyre_lower::KernelOp {
    op(KernelOpKind::LocalInvocationId, [0], result)
}

fn cast(target: DataType, source: u32, result: u32) -> vyre_lower::KernelOp {
    op(KernelOpKind::Cast { target }, [source], result)
}

/// Build a minimal descriptor that emits a BinOp on I32 operands via
/// LocalInvocationId (so the op survives constant folding) and stores the
/// result.
fn i32_binop_descriptor(kind: BinOp) -> KernelDescriptor {
    descriptor("i32_binop")
        .slot(global_rw(0, DataType::I32, "out"))
        .dispatch(64, 1, 1)
        .body(
            body()
                // result 0 keeps the op live, result 1 makes the BinOp
                // operands I32, result 2 is the U32 shift amount PTX wants.
                .op(thread_id(0))
                .op(cast(DataType::I32, 0, 1))
                .op(lit(0, 2))
                .op(lit(1, 3))
                .op(binop(kind, 1, 2, 4))
                .op(store_global(0, 3, 4))
                .literals([LiteralValue::U32(1), LiteralValue::U32(0)]),
        )
        .build()
}

/// VYRE-PTX-001: `Shr` on I32 must emit `shr.s32` (arithmetic), not
/// `shr.u32` (logical).
///
/// Before the fix, `ptx_binop_suffix(BinOp::Shr, PtxType::I32)` returned
/// `"u32"` for all types, so the emitted instruction was `shr.u32` even when
/// the operand register was %s<N> (signed class). That is a silent miscompile:
/// `(-4) >> 1` via `shr.u32` produces 0x7FFFFFFE instead of -2 (0xFFFFFFFE).
#[test]
fn shr_on_i32_emits_s32_suffix_not_u32() {
    let ptx = emit(&i32_binop_descriptor(BinOp::Shr), "I32 Shr descriptor");

    assert!(
        ptx.contains("shr.s32"),
        "Shr on I32 operands must emit `shr.s32` (arithmetic shift); \
         found `shr.u32` instead, that is a logical shift which gives wrong results \
         for negative values. PTX emitted:\n{ptx}"
    );
    // Guard against regression: u32-suffixed shr must NOT appear for this I32
    // descriptor (there is no u32 operand in this kernel).
    assert!(
        !ptx.contains("shr.u32"),
        "shr.u32 must not appear in an I32-operand Shr kernel; \
         regression: the unsigned logical shift is back. PTX emitted:\n{ptx}"
    );
}

/// Complementary guard: `Shr` on U32 must still emit `shr.u32`: the fix
/// must not accidentally break unsigned shifts.
#[test]
fn shr_on_u32_still_emits_u32_suffix() {
    let desc = descriptor("u32_shr")
        .slot(global_rw(0, DataType::U32, "out"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .op(thread_id(0))
                .op(lit(0, 1))
                .op(lit(1, 2))
                .op(binop(BinOp::Shr, 0, 1, 3))
                .op(store_global(0, 2, 3))
                .literals([LiteralValue::U32(1), LiteralValue::U32(0)]),
        )
        .build();
    let ptx = emit(&desc, "U32 Shr descriptor");
    assert!(
        ptx.contains("shr.u32"),
        "Shr on U32 operands must still emit `shr.u32`; \
         the signed-shift fix must not break unsigned shifts. PTX emitted:\n{ptx}"
    );
    assert!(
        !ptx.contains("shr.s32"),
        "shr.s32 must not appear in a U32-only Shr kernel. PTX emitted:\n{ptx}"
    );
}

/// VYRE-PTX-002: the overflow guard in `ensure_buffer_length_reg` must emit
/// `trap;`: not a plausible-address load, when the slot byte offset would
/// overflow u32.  We can only exercise the overflow branch by triggering
/// `ensure_buffer_length_reg` with a slot that was registered during
/// `preload_bindings` (the normal path), so this test instead validates that
/// the checked-arithmetic path in `preload_bindings` produces the correct
/// `[%rd0 + 8]` offset for slot 1 (byte_offset = 1*4+4 = 8), confirming the
/// non-overflow path is correct and the overflow path is the only way `trap;`
/// can appear.
#[test]
fn ensure_buffer_length_reg_emits_correct_offset_for_slot_1() {
    let desc = descriptor("two_slot_bounds")
        .slots([
            global_rw(0, DataType::U32, "a"),
            global_rw(1, DataType::U32, "b"),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .op(lit(0, 0))
                .op(lit(0, 1))
                .op(load_global(0, 0, 2))
                .op(load_global(1, 1, 3))
                .op(store_global(0, 0, 3))
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let ptx = emit(&desc, "two-slot descriptor");

    // slot 0: byte_offset = 0*4+4 = 4  → `[%rd0 + 4]`
    // slot 1: byte_offset = 1*4+4 = 8  → `[%rd0 + 8]`
    assert!(
        ptx.contains("[%rd0 + 4]"),
        "slot 0 length load must use offset 4; PTX emitted:\n{ptx}"
    );
    assert!(
        ptx.contains("[%rd0 + 8]"),
        "slot 1 length load must use offset 8; PTX emitted:\n{ptx}"
    );
    // No trap should appear for a valid descriptor.
    assert!(
        !ptx.contains("trap;"),
        "valid two-slot descriptor must not emit trap; PTX emitted:\n{ptx}"
    );
}

/// VYRE-PTX-003: storing a U64 value to an F16 binding must emit
/// `cvt.rn.f32.u64` (single-step, no precision loss) and must NOT emit the
/// old two-step truncating path `cvt.u32.u64`.
#[test]
fn f16_store_of_u64_value_uses_direct_cvt_rn_f32_u64() {
    // Load a U64 value then store it to an F16 binding, exercising the
    // ensure_f32_store_operand(U64) path. Cast(U64) from a LocalInvocationId
    // (U32) is what produces the U64 register.
    let desc = descriptor("u64_to_f16_store")
        .slot(global_rw(0, DataType::F16, "out_f16"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .op(thread_id(0))
                .op(cast(DataType::U64, 0, 1))
                .op(lit(0, 2))
                .op(store_global(0, 2, 1))
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let ptx = emit(&desc, "U64 to F16 store descriptor");

    // The fixed path: single-instruction conversion preserving all 64 bits.
    assert!(
        ptx.contains("cvt.rn.f32.u64"),
        "storing a U64 value to an F16 binding must emit `cvt.rn.f32.u64`; \
         the old two-step path `cvt.u32.u64` silently truncated the high 32 bits. \
         PTX emitted:\n{ptx}"
    );
    // The broken path: must be gone.
    assert!(
        !ptx.contains("cvt.u32.u64"),
        "`cvt.u32.u64` must not appear; it truncates the high 32 bits of the U64 \
         value before conversion. PTX emitted:\n{ptx}"
    );
}
