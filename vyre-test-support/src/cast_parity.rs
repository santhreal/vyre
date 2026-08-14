//! Probe corpora and pinned oracles for the integer cast-parity gates.
//!
//! Every backend that lowers an integer cast owes the same two answers: the
//! widening cast fills the high word from the SOURCE signedness, and the
//! narrowing cast truncates then re-extends. Those answers are Rust `as`
//! semantics, so the probe words and the pinned result vectors are one decision,
//! not one per backend, and a backend that carried its own copy could have its
//! pin edited to match a miscompile without any other target noticing.
//!
//! What stays per backend is the reference arm and the dispatch: each cast
//! parity target recomputes `expected` from the corpus with Rust `as`, asserts
//! it against the pin here, and only then compares the device result. The
//! lowerings being compared are unrelated (wgpu synthesizes a `vec2<u32>` high
//! word through a shift and a multiply; PTX emits a native `cvt`), so neither
//! target may stand in for the other's reference.

/// Signed 32-bit probe words for a widening cast: the sign boundary, both
/// extremes, and the quarter-range patterns that separate a sign-replicate from
/// an arithmetic shift.
pub const SIGNED_WIDENING_INPUTS: [i32; 10] = [
    -7,
    7,
    -1,
    0,
    1,
    i32::MIN,
    i32::MAX,
    -128,
    0x4000_0000,
    -0x4000_0000,
];

/// `i32 as i64 as u64` over [`SIGNED_WIDENING_INPUTS`]: every negative source
/// carries a `0xFFFF_FFFF` high word.
pub const I32_TO_I64_EXPECTED: [u64; 10] = [
    0xFFFF_FFFF_FFFF_FFF9,
    0x0000_0000_0000_0007,
    0xFFFF_FFFF_FFFF_FFFF,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0xFFFF_FFFF_8000_0000,
    0x0000_0000_7FFF_FFFF,
    0xFFFF_FFFF_FFFF_FF80,
    0x0000_0000_4000_0000,
    0xFFFF_FFFF_C000_0000,
];

/// Unsigned 32-bit probe words for a widening cast into an unsigned target.
pub const UNSIGNED_WIDENING_INPUTS: [u32; 7] =
    [0xFFFF_FFFF, 0x8000_0000, 7, 0, 1, 0x7FFF_FFFF, 0xDEAD_BEEF];

/// Unsigned 32-bit probe words for a widening cast into a SIGNED 64-bit target,
/// where the zero-extend must still be selected off the source.
pub const UNSIGNED_TO_SIGNED_WIDENING_INPUTS: [u32; 5] =
    [0xFFFF_FFFF, 0x8000_0000, 7, 0, 0x7FFF_FFFF];

/// Probe words for a narrowing cast: 300 (low byte 44), 0x12345 (low half
/// 0x2345), 200 (`i8` -56), 0xFFFF (`i16` -1 / `u16` max), 0x8000 (`i16` MIN),
/// 0xFFFFFFFF (all ones), then 0, 127, 128, 255 around the byte boundary.
pub const NARROWING_INPUTS: [u32; 10] = [
    300,
    0x0001_2345,
    200,
    0x0000_FFFF,
    0x0000_8000,
    0xFFFF_FFFF,
    0,
    127,
    128,
    255,
];

/// `u32 as u8` over [`NARROWING_INPUTS`], zero-extended back to 32 bits.
pub const U32_TO_U8_EXPECTED: [u32; 10] = [44, 0x45, 200, 0xFF, 0, 0xFF, 0, 127, 128, 255];

/// `u32 as u16` over [`NARROWING_INPUTS`], zero-extended back to 32 bits.
pub const U32_TO_U16_EXPECTED: [u32; 10] =
    [300, 0x2345, 200, 0xFFFF, 0x8000, 0xFFFF, 0, 127, 128, 255];

/// `u32 as u8 as i8` over [`NARROWING_INPUTS`], sign-extended back to 32 bits.
pub const U32_TO_I8_EXPECTED: [i32; 10] = [44, 69, -56, -1, 0, -1, 0, 127, -128, -1];

/// `u32 as u16 as i16` over [`NARROWING_INPUTS`], sign-extended back to 32 bits.
pub const U32_TO_I16_EXPECTED: [i32; 10] = [300, 0x2345, 200, -1, -32768, -1, 0, 127, 128, 255];

/// The signed widening corpus as the `u32` bit patterns a 32-bit source buffer
/// carries.
#[must_use]
pub fn signed_widening_words() -> Vec<u32> {
    SIGNED_WIDENING_INPUTS.iter().map(|&v| v as u32).collect()
}
