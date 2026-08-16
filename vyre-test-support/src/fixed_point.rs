//! 16.16 fixed-point oracles for parity suites.
//!
//! Every `_via` end-to-end parity test compares a GPU IR program against a host
//! oracle written in the same arithmetic the program emits. The arithmetic is
//! the contract: bits `[16..48]` of a signed 64-bit product, a two's-complement
//! word, a deterministic generator whose sequence the assertion depends on.
//! Restating any of it per file is how six copies of the multiply came to keep
//! an unsigned form after the kernel it mirrors was corrected to signed.
//!
//! Nothing here touches the IR, a dispatcher, or a wire format, so a suite that
//! needs the oracle does not take a dependency on the crate whose kernels it is
//! checking.

/// The 16.16 fixed-point unit, `1.0`.
pub const FIXED_ONE: u32 = 1 << 16;

/// Encode `v` as a two's-complement 16.16 word, rounded to nearest.
#[must_use]
pub fn to_fixed(v: f64) -> u32 {
    (v * f64::from(FIXED_ONE)).round() as i64 as u32
}

/// Decode a two's-complement 16.16 word to the signed value it encodes.
#[must_use]
pub fn from_fixed(v: u32) -> f64 {
    f64::from(v as i32) / f64::from(FIXED_ONE)
}

/// Signed 16.16 multiply: bits `[16..48]` of the signed 64-bit product.
///
/// Operands are two's-complement `i32` carried in a `u32`. A weighted-Jacobi
/// residual, a sheaf coupling and a gradient are all routinely negative, so the
/// multiply must be signed; the unsigned form silently corrupts negative
/// operands. For non-negative operands it is bit-identical to the unsigned form.
#[must_use]
pub fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// Multiply a square 16.16 matrix by a 16.16 vector with wrapping accumulation.
#[must_use]
pub fn fixed_matvec(matrix: &[u32], vector: &[u32], n: usize) -> Vec<u32> {
    (0..n)
        .map(|row| {
            let mut acc = 0u32;
            for column in 0..n {
                acc = acc.wrapping_add(fixed_mul(matrix[row * n + column], vector[column]));
            }
            acc
        })
        .collect()
}

/// Signed division by a known-positive divisor, truncating toward zero.
///
/// Mirrors the fixed weighted-Jacobi `delta` divide, whose numerator is negative
/// whenever the residual is negative.
#[must_use]
pub fn fixed_sdiv_by_positive(numerator: u32, denominator: u32) -> u32 {
    ((numerator as i32) / (denominator as i32)) as u32
}

/// Advance the deterministic xorshift32 generator used by parity sweeps.
pub fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A signed 16.16 sample whose magnitude is drawn from `magnitude_mask`.
fn signed_fixed_with_mask(state: &mut u32, magnitude_mask: u32) -> u32 {
    let magnitude = (xorshift32(state) & magnitude_mask) as i32;
    if xorshift32(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Generate a signed 16.16 sample in approximately `[-8, 8)`.
pub fn signed_fixed_19(state: &mut u32) -> u32 {
    signed_fixed_with_mask(state, 0x0007_FFFF)
}

/// Generate a signed 16.16 sample in approximately `[-4, 4)`.
pub fn signed_fixed_18(state: &mut u32) -> u32 {
    signed_fixed_with_mask(state, 0x0003_FFFF)
}

/// Generate a signed 16.16 sample in approximately `[-2, 2)`.
pub fn signed_fixed_17(state: &mut u32) -> u32 {
    signed_fixed_with_mask(state, 0x0001_FFFF)
}
