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

use vyre_foundation::ir::DataType;

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

/// How a narrowing case's stored 32-bit word is read back.
///
/// Each case pins its result in the signedness a reader checks by inspection:
/// `u32 as u16` is a truncation to `0x2345`, `u32 as i16` is a sign extension to
/// `-1`. Keeping one pin per case in its own signedness means neither arm holds
/// a re-encoded copy of the other's numbers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PinnedNarrowing {
    /// Zero-extended back to 32 bits.
    Unsigned(&'static [u32; 10]),
    /// Sign-extended back to 32 bits.
    Signed(&'static [i32; 10]),
}

/// One `u32 -> narrow -> wide` case of the narrowing-cast matrix.
///
/// The four cases are the same program with two types substituted, so a target
/// binds its dispatch once and walks this table rather than repeating a test
/// body per case. Nothing here names a target, a dialect or a driver.
pub struct NarrowingCase {
    /// Narrow integer the cast truncates to.
    pub narrow: DataType,
    /// Non-narrowing integer that carries the narrowed value into a 32-bit
    /// store slot, so the word read back is what the cast produced rather than
    /// what a byte-element store would have masked it to.
    pub wide: DataType,
    /// The Rust `as` chain this case must reproduce, applied to one probe word
    /// and returned as the 32-bit pattern a target stores.
    pub reference: fn(u32) -> u32,
    /// Pinned result of `reference` over [`NARROWING_INPUTS`].
    pub pinned: PinnedNarrowing,
    /// The cast as it is written in Rust, for the failure message.
    pub label: &'static str,
}

/// Every narrowing cast a 32-bit source can reach, with its oracle.
pub const NARROWING_CASES: &[NarrowingCase] = &[
    NarrowingCase {
        narrow: DataType::U8,
        wide: DataType::U32,
        reference: |value| u32::from(value as u8),
        pinned: PinnedNarrowing::Unsigned(&U32_TO_U8_EXPECTED),
        label: "as u8",
    },
    NarrowingCase {
        narrow: DataType::U16,
        wide: DataType::U32,
        reference: |value| u32::from(value as u16),
        pinned: PinnedNarrowing::Unsigned(&U32_TO_U16_EXPECTED),
        label: "as u16",
    },
    NarrowingCase {
        narrow: DataType::I8,
        wide: DataType::I32,
        reference: |value| i32::from(value as u8 as i8) as u32,
        pinned: PinnedNarrowing::Signed(&U32_TO_I8_EXPECTED),
        label: "as i8",
    },
    NarrowingCase {
        narrow: DataType::I16,
        wide: DataType::I32,
        reference: |value| i32::from(value as u16 as i16) as u32,
        pinned: PinnedNarrowing::Signed(&U32_TO_I16_EXPECTED),
        label: "as i16",
    },
];

impl NarrowingCase {
    /// This case's Rust `as` result over [`NARROWING_INPUTS`], as stored words.
    #[must_use]
    pub fn reference_words(&self) -> Vec<u32> {
        NARROWING_INPUTS
            .iter()
            .map(|&value| (self.reference)(value))
            .collect()
    }

    /// Assert the recomputed reference still matches this case's pin.
    ///
    /// # Panics
    ///
    /// Panics when the two disagree, which means either the pin was edited or
    /// the Rust `as` chain beside it was, and a target comparing against the
    /// drifted oracle would report agreement with the wrong answer.
    pub fn assert_pin_holds(&self) {
        let reference = self.reference_words();
        match self.pinned {
            PinnedNarrowing::Unsigned(pinned) => assert_eq!(
                reference,
                pinned.as_slice(),
                "reference `u32 {}` drifted from its pin",
                self.label
            ),
            PinnedNarrowing::Signed(pinned) => {
                let signed: Vec<i32> = reference.iter().map(|&word| word as i32).collect();
                assert_eq!(
                    signed,
                    pinned.as_slice(),
                    "reference `u32 {}` drifted from its pin",
                    self.label
                );
            }
        }
    }

    /// Assert a target's stored words reproduce this case exactly.
    ///
    /// `target` names the arm in the failure message. The pin is checked first,
    /// so a drifted oracle is reported as a drifted oracle rather than as a
    /// device divergence.
    ///
    /// # Panics
    ///
    /// Panics when the pin drifted or when `device_words` differ from the Rust
    /// `as` result for any probe input.
    pub fn assert_target_words(&self, target: &str, device_words: &[u32]) {
        self.assert_pin_holds();
        let reference = self.reference_words();
        match self.pinned {
            PinnedNarrowing::Unsigned(_) => assert_eq!(
                device_words,
                reference.as_slice(),
                "{target} `u32 {}` diverged from Rust.\n  inputs:   {:?}\n  expected: {:?}\n  device:   {:?}",
                self.label,
                NARROWING_INPUTS,
                reference,
                device_words
            ),
            PinnedNarrowing::Signed(_) => {
                let device: Vec<i32> = device_words.iter().map(|&word| word as i32).collect();
                let expected: Vec<i32> = reference.iter().map(|&word| word as i32).collect();
                assert_eq!(
                    device,
                    expected,
                    "{target} `u32 {}` diverged from Rust.\n  inputs:   {:?}\n  expected: {:?}\n  device:   {:?}",
                    self.label,
                    NARROWING_INPUTS,
                    expected,
                    device
                );
            }
        }
    }
}

/// The signed widening corpus as the `u32` bit patterns a 32-bit source buffer
/// carries.
#[must_use]
pub fn signed_widening_words() -> Vec<u32> {
    SIGNED_WIDENING_INPUTS.iter().map(|&v| v as u32).collect()
}
