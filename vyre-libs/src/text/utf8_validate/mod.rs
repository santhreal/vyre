//! Tier 2.5 UTF-8 validator  -  single-pass byte classification with
//! structural sequence checks.
//!
//! Each invocation reads one source byte (`source[i]`, low 8 bits)
//! and writes one of four classification codes into `classes[i]`:
//!
//! - [`UTF8_ASCII`]  -  byte 0x00..0x7F, single-byte sequence
//! - [`UTF8_LEAD_2`]  -  byte 0xC0..0xDF, lead of a 2-byte sequence
//! - [`UTF8_LEAD_3`]  -  byte 0xE0..0xEF, lead of a 3-byte sequence
//! - [`UTF8_LEAD_4`]  -  byte 0xF0..0xF7, lead of a 4-byte sequence
//! - [`UTF8_CONT`]    -  byte 0x80..0xBF, continuation byte
//! - [`UTF8_INVALID`]  -  byte 0xC0/0xC1 (overlong) or ≥ 0xF8 (out of range)
//!
//! Malformed lead/continuation structure is reported as
//! [`UTF8_INVALID`] at the offending byte. Valid bytes retain the
//! shape code parser dialects need for downstream tokenization.

mod program;
#[cfg(any(test, feature = "cpu-parity", feature = "text"))]
mod reference;
mod sequence_rules;

#[cfg(test)]
mod tests;

pub use program::{utf8_validate, utf8_validate_u8};
#[cfg(any(test, feature = "cpu-parity", feature = "text"))]
pub use reference::reference_utf8_validate;

/// Stable op id for the registered Tier 3 wrapper.
pub(crate) const OP_ID: &str = "vyre-primitives::text::utf8_validate";
/// Byte-lane workgroup used by the UTF-8 classifier.
pub const UTF8_VALIDATE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
/// Dispatch grid for one UTF-8 validation pass over `n` bytes.
#[must_use]
pub const fn utf8_validate_dispatch_grid(n: u32) -> [u32; 3] {
    let blocks = n.div_ceil(UTF8_VALIDATE_WORKGROUP_SIZE[0]);
    if blocks == 0 {
        [1, 1, 1]
    } else {
        [blocks, 1, 1]
    }
}

/// 0x00..0x7F  -  single-byte ASCII.
pub const UTF8_ASCII: u32 = 0;
/// 0xC2..0xDF  -  lead of a valid 2-byte sequence.
pub const UTF8_LEAD_2: u32 = 1;
/// 0xE0..0xEF  -  lead of a 3-byte sequence.
pub const UTF8_LEAD_3: u32 = 2;
/// 0xF0..0xF7  -  lead of a 4-byte sequence.
pub const UTF8_LEAD_4: u32 = 3;
/// 0x80..0xBF  -  continuation byte.
pub const UTF8_CONT: u32 = 4;
/// 0xC0, 0xC1 (overlong) or 0xF8..0xFF (out of range)  -  invalid lead.
pub const UTF8_INVALID: u32 = 5;

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || utf8_validate("source", "classes", 8),
        Some(|| {
            vec![vec![
                vec![0xC3, 0x00, 0x00, 0x00, 0xA9, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x9F, 0x00, 0x00, 0x00, 0x98, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00],
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ]]
        }),
        Some(|| {
            vec![vec![
                vec![0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00],
            ]]
        }),
    )
}
