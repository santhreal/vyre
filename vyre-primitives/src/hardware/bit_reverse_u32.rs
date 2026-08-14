//! Cat-C `bit_reverse_u32`  -  reverse the bit order within each u32 lane
//! via the hardware `reverseBits` instruction.

define_unary_u32_hardware_intrinsic!(
    bit_reverse_u32,
    "vyre-primitives::hardware::bit_reverse_u32",
    vyre_foundation::ir::Expr::reverse_bits,
    |value: u32| value.reverse_bits(),
    &[0u32, 1, 0x8000_0000, 0x1234_5678],
    0x1EA0_7733,
    &[1],
    &[u32::MAX]
);
