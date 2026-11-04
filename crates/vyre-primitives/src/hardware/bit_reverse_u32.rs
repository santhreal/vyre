//! Cat-C `bit_reverse_u32`  -  reverse the bit order within each u32 lane
//! via the hardware `reverseBits` instruction.

define_unary_u32_hardware_intrinsic!(
    bit_reverse_u32,
    "vyre-primitives::hardware::bit_reverse_u32",
    vyre_foundation::ir::Expr::reverse_bits,
    |value: u32| value.reverse_bits(),
    &[0u32, 1, 0x8000_0000, 0x1234_5678],
    &[0u32, 0x8000_0000, 1, 0x1E6A_2C48],
    &[
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x48, 0x2C, 0x6A,
        0x1E,
    ],
    0x1EA0_7733,
    &[1],
    &[u32::MAX]
);
