//! Where one packed field sits, and what it decodes to.
//!
//! Every consumer of a packed buffer answers the same three questions: which
//! container word holds element `n`, how far the field sits above that word's
//! least significant bit, and what a code means once it is masked out. The
//! answers follow from the storage width, the container width and the packing
//! order the contract already states, so they are derived here once rather than
//! restated by each builder. A builder that recomputes them can disagree with
//! the producer that wrote the bytes, and nothing in the program says so.
//!
//! `vyre-reference` decodes the same layouts independently. That is deliberate:
//! it is the parity oracle, and an oracle that shares the implementation it
//! checks proves nothing.

use serde::{Deserialize, Serialize};

use crate::ir::{DataType, Expr};

use super::{PackingOrder, QuantizedContract};

/// Where one packed field sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct PackedField {
    /// Container words from the start of the buffer.
    pub word: u64,
    /// Bits the field sits above the container's least significant bit.
    pub shift_bits: u32,
    /// Mask selecting the field once it is shifted down.
    pub mask: u64,
}

/// The format a packed field is decoded into.
///
/// The variants are the compute formats a packed integer code is read as. A
/// codebook format decodes through a table rather than a shift, so it is not
/// one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum FieldTarget {
    /// A signed 32-bit integer.
    SignedInt32,
    /// A binary32 float.
    Float32,
}

impl FieldTarget {
    /// The data type a field decoded into this target holds.
    #[must_use]
    pub const fn data_type(self) -> DataType {
        match self {
            Self::SignedInt32 => DataType::I32,
            Self::Float32 => DataType::F32,
        }
    }

    /// `value` as a literal of this target's type.
    #[must_use]
    fn literal(self, value: u32) -> Expr {
        match self {
            Self::SignedInt32 => Expr::i32(i32::try_from(value).unwrap_or(i32::MAX)),
            Self::Float32 => Expr::f32(value as f32),
        }
    }
}

impl QuantizedContract {
    /// Packed fields one container word holds.
    #[must_use]
    pub fn fields_per_container(&self) -> u32 {
        let storage_bits = self.storage.bit_width();
        if storage_bits == 0 {
            return 1;
        }
        (self.container.bit_width() / storage_bits).max(1)
    }

    /// Container words `elements` packed fields occupy, including a partly
    /// filled last word.
    #[must_use]
    pub fn container_words(&self, elements: u64) -> u64 {
        elements.div_ceil(u64::from(self.fields_per_container()))
    }

    /// The mask that selects one field once it is shifted down.
    #[must_use]
    pub fn field_mask(&self) -> u64 {
        let storage_bits = self.storage.bit_width();
        if storage_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << storage_bits) - 1
        }
    }

    /// Where element `index` sits.
    #[must_use]
    pub fn field(&self, index: u64) -> PackedField {
        let fields = u64::from(self.fields_per_container());
        let storage_bits = self.storage.bit_width();
        let position = index % fields;
        let field = match self.packing {
            PackingOrder::LowFieldFirst => position,
            PackingOrder::HighFieldFirst => fields - 1 - position,
        };
        PackedField {
            word: index / fields,
            shift_bits: u32::try_from(field).unwrap_or(0) * storage_bits,
            mask: self.field_mask(),
        }
    }

    /// The container word element `index` sits in.
    #[must_use]
    pub fn field_word(&self, index: Expr) -> Expr {
        Expr::div(index, Expr::u32(self.fields_per_container()))
    }

    /// The bits element `index` sits above its container word's least
    /// significant bit.
    #[must_use]
    pub fn field_shift(&self, index: Expr) -> Expr {
        let fields = self.fields_per_container();
        let storage_bits = self.storage.bit_width();
        let position = Expr::rem(index, Expr::u32(fields));
        match self.packing {
            PackingOrder::LowFieldFirst => Expr::mul(position, Expr::u32(storage_bits)),
            PackingOrder::HighFieldFirst => Expr::sub(
                Expr::u32((fields - 1) * storage_bits),
                Expr::mul(position, Expr::u32(storage_bits)),
            ),
        }
    }

    /// The raw code sitting `shift` bits above `word`.
    #[must_use]
    pub fn extract_field(&self, word: Expr, shift: Expr) -> Expr {
        Expr::bitand(
            Expr::shr(word, shift),
            Expr::u32(u32::try_from(self.field_mask()).unwrap_or(u32::MAX)),
        )
    }

    /// The raw code for element `index` of the buffer bound to `buffer`.
    #[must_use]
    pub fn load_field(&self, buffer: &str, index: Expr) -> Expr {
        let word = Expr::load(buffer, self.field_word(index.clone()));
        self.extract_field(word, self.field_shift(index))
    }

    /// The raw code for element `index` of the row starting at `word_base`.
    ///
    /// A row-major buffer passes the container index its row starts at, which
    /// is the only thing that changes between one row and the next.
    #[must_use]
    pub fn load_row_field(&self, buffer: &str, word_base: Expr, index: Expr) -> Expr {
        let word = Expr::load(buffer, Expr::add(word_base, self.field_word(index.clone())));
        self.extract_field(word, self.field_shift(index))
    }

    /// `code` decoded into `target`.
    ///
    /// An unsigned grid reads the code as it stands. A signed grid stores the
    /// negative half above the positive one, so a code with its top bit set is
    /// the code less one full span.
    #[must_use]
    pub fn decode_field(&self, code: Expr, target: FieldTarget) -> Expr {
        let storage_bits = self.storage.bit_width();
        if !self.signed || storage_bits == 0 || storage_bits >= 32 {
            return Expr::cast(target.data_type(), code);
        }
        let sign_bit = 1u32 << (storage_bits - 1);
        let span = 1u32 << storage_bits;
        Expr::select(
            Expr::eq(
                Expr::bitand(code.clone(), Expr::u32(sign_bit)),
                Expr::u32(0),
            ),
            Expr::cast(target.data_type(), code.clone()),
            Expr::sub(Expr::cast(target.data_type(), code), target.literal(span)),
        )
    }
}
