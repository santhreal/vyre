//! Typed matrix fragment behavior: element widths, per-operand tile extents,
//! derived operand words and specification validation.
//!
//! The descriptor used to state one tile shape and one operand arity, so the
//! fragment layout of a single native form was written into the neutral IR and
//! every other form was unstatable. Extents, lane distribution and staging
//! storage are declared here instead, and the word counts a target reads are
//! computed from them.

use super::{
    FragmentOperand, FragmentValue, MatrixMmaElement, MatrixMmaSpec, MatrixSpecError,
    MatrixTileShape, TensorAccessMap,
};

/// Bits in one 32-bit operand word.
const WORD_BITS: u32 = 32;

impl MatrixMmaElement {
    /// Storage width of one element, in bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::F16 | Self::BF16 => 16,
            Self::TF32 | Self::F32 => 32,
        }
    }
}

impl MatrixTileShape {
    /// Extents of one operand's tile as `[rows, columns]`.
    #[must_use]
    pub const fn extents(self, operand: FragmentOperand) -> [u16; 2] {
        match operand {
            FragmentOperand::Left => [self.m, self.k],
            FragmentOperand::Right => [self.k, self.n],
            FragmentOperand::Accumulator => [self.m, self.n],
        }
    }

    /// Elements one operand's tile holds.
    #[must_use]
    pub const fn elements(self, operand: FragmentOperand) -> u32 {
        let [rows, columns] = self.extents(operand);
        rows as u32 * columns as u32
    }

    /// Whether every extent is stated.
    #[must_use]
    pub const fn is_stated(self) -> bool {
        self.m != 0 && self.n != 0 && self.k != 0
    }
}

impl TensorAccessMap {
    /// Element stride between rows, resolving a packed declaration against the
    /// tile's own column extent.
    #[must_use]
    pub const fn effective_row_stride(&self, columns: u16) -> u32 {
        if self.row_stride == 0 {
            columns as u32
        } else {
            self.row_stride
        }
    }
}

impl FragmentValue {
    /// A register-resident fragment: no staging storage is declared.
    #[must_use]
    pub const fn in_registers(
        element: MatrixMmaElement,
        layout: super::MatrixMmaLayout,
        lanes: u16,
    ) -> Self {
        Self {
            element,
            layout,
            lanes,
            access: None,
        }
    }

    /// Whether this fragment is held in registers rather than staged through
    /// addressable storage.
    #[must_use]
    pub const fn is_register_resident(&self) -> bool {
        self.access.is_none()
    }

    /// Words of 32 bits each invocation contributes for a tile of `elements`.
    ///
    /// # Errors
    ///
    /// Fails when the tile does not distribute evenly across the fragment's
    /// lanes, or when the per-invocation share does not fill whole words.
    pub fn words_per_lane(
        &self,
        operand: FragmentOperand,
        elements: u32,
    ) -> Result<u32, MatrixSpecError> {
        if self.lanes == 0 {
            return Err(MatrixSpecError::ZeroLanes);
        }
        let lanes = u32::from(self.lanes);
        if elements % lanes != 0 {
            return Err(MatrixSpecError::UnevenDistribution {
                operand,
                elements,
                lanes: self.lanes,
            });
        }
        let bits_per_lane = (elements / lanes) * self.element.bits();
        if bits_per_lane == 0 || bits_per_lane % WORD_BITS != 0 {
            return Err(MatrixSpecError::PartialWord {
                operand,
                bits_per_lane,
            });
        }
        Ok(bits_per_lane / WORD_BITS)
    }
}

impl MatrixMmaSpec {
    /// The fragment carrying one operand.
    #[must_use]
    pub const fn fragment(&self, operand: FragmentOperand) -> &FragmentValue {
        match operand {
            FragmentOperand::Left => &self.left,
            FragmentOperand::Right => &self.right,
            FragmentOperand::Accumulator => &self.accumulator,
        }
    }

    /// Words each invocation contributes per operand, in operand order.
    ///
    /// # Errors
    ///
    /// Fails on the first operand whose declared fragment cannot be carried:
    /// an unstated extent, a tile that does not distribute across its lanes, a
    /// per-invocation share that does not fill whole words, or a staging map
    /// narrower than the tile it stages.
    pub fn operand_words(&self) -> Result<[u32; 3], MatrixSpecError> {
        if !self.tile.is_stated() {
            return Err(MatrixSpecError::ZeroExtent);
        }
        let mut words = [0u32; 3];
        for (slot, operand) in [
            FragmentOperand::Left,
            FragmentOperand::Right,
            FragmentOperand::Accumulator,
        ]
        .into_iter()
        .enumerate()
        {
            let fragment = self.fragment(operand);
            let [_, columns] = self.tile.extents(operand);
            if let Some(access) = &fragment.access {
                if access.alignment == 0 {
                    return Err(MatrixSpecError::ZeroAlignment { operand });
                }
                let stride = access.effective_row_stride(columns);
                if stride < u32::from(columns) {
                    return Err(MatrixSpecError::ShortRowStride {
                        operand,
                        stride,
                        columns,
                    });
                }
            }
            words[slot] = fragment.words_per_lane(operand, self.tile.elements(operand))?;
        }
        Ok(words)
    }

    /// Operand ids this op reads.
    ///
    /// # Errors
    ///
    /// Fails for the same reasons as `operand_words`.
    pub fn operand_count(&self) -> Result<u32, MatrixSpecError> {
        let [left, right, accumulator] = self.operand_words()?;
        Ok(left + right + accumulator)
    }

    /// Result ids this op defines, which is the accumulator fragment's words.
    ///
    /// # Errors
    ///
    /// Fails for the same reasons as `operand_words`.
    pub fn result_count(&self) -> Result<u32, MatrixSpecError> {
        Ok(self.operand_words()?[2])
    }

    /// Whether every declared fact can be carried to a target.
    ///
    /// # Errors
    ///
    /// Fails for the same reasons as `operand_words`.
    pub fn validate(&self) -> Result<(), MatrixSpecError> {
        self.operand_words().map(|_| ())
    }
}

impl std::fmt::Display for FragmentOperand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Left => "left",
            Self::Right => "right",
            Self::Accumulator => "accumulator",
        })
    }
}

impl std::fmt::Display for MatrixSpecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroExtent => formatter.write_str("a tile extent is zero"),
            Self::ZeroLanes => formatter.write_str("a fragment states zero lanes"),
            Self::UnevenDistribution {
                operand,
                elements,
                lanes,
            } => write!(
                formatter,
                "the {operand} tile holds {elements} elements, which does not distribute across {lanes} lanes"
            ),
            Self::PartialWord {
                operand,
                bits_per_lane,
            } => write!(
                formatter,
                "each lane holds {bits_per_lane} bits of the {operand} tile, which is not whole 32-bit words"
            ),
            Self::ShortRowStride {
                operand,
                stride,
                columns,
            } => write!(
                formatter,
                "the {operand} tile stages with a row stride of {stride} elements but occupies {columns} columns"
            ),
            Self::ZeroAlignment { operand } => {
                write!(formatter, "the {operand} tile states zero base alignment")
            }
        }
    }
}

impl std::error::Error for MatrixSpecError {}
