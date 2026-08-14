//! Cat-C `popcount_u32`  -  count set bits in each u32 lane via the hardware
//! `countOneBits` instruction.

define_unary_u32_hardware_intrinsic!(
    popcount_u32,
    "vyre-primitives::hardware::popcount_u32",
    vyre_foundation::ir::Expr::popcount,
    |value: u32| value.count_ones(),
    &[0u32, 1, 0xFFFF_FFFF, 0x1234_5678],
    0xC0FF_EE11,
    &[1],
    &[u32::MAX]
);
