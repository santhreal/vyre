//! Sequential mathematical witnesses for LR parsing and syntax analysis.

use std::fmt;
use vyre_spec::c11_expr_token::TOK_EOF;

/// Action entry unpacked in LR parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LrAction {
    /// No valid transition; syntax error.
    Error,
    /// Shift to next state and advance token stream.
    Shift(u32),
    /// Reduce by production id without advancing token stream.
    Reduce(u32),
    /// Accept the input stream.
    Accept,
}

impl LrAction {
    /// Unpack an action word from the packed 32-bit representation.
    #[inline]
    #[must_use]
    pub fn unpack(packed: u32) -> Self {
        match packed >> 30 {
            0 => Self::Error,
            1 => Self::Shift(packed & 0x3FFF_FFFF),
            2 => Self::Reduce(packed & 0x3FFF_FFFF),
            3 => Self::Accept,
            _ => Self::Error,
        }
    }
}

/// A single grammar production: `lhs nonterminal -> rhs_len symbols`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LrProduction {
    /// Nonterminal index on the left-hand side.
    pub lhs: u32,
    /// Number of symbols to pop from the parser stack on reduce.
    pub rhs_len: u32,
}

/// Errors emitted by the sequential reference LR parser witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseLrWitnessError {
    /// The current state has no action for the lookahead token.
    UnexpectedToken {
        /// LR state at the error point.
        state: u32,
        /// Lookahead token id that had no action.
        token: u32,
        /// Token stream position.
        pos: usize,
    },
    /// The production id returned by the action table does not exist.
    InvalidProduction {
        /// Invalid production id.
        prod_id: u32,
    },
    /// Tried to pop states from an empty stack.
    StackUnderflow,
    /// The goto table has no entry for `(state, nonterminal)`.
    NoGoto {
        /// LR state after reduction.
        state: u32,
        /// Nonterminal id for the missing goto entry.
        nonterminal: u32,
    },
}

impl fmt::Display for ParseLrWitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedToken { state, token, pos } => {
                write!(
                    f,
                    "LR unexpected token: state={state} token={token} pos={pos}. \
                     Fix: validate token stream against grammar or extend action table."
                )
            }
            Self::InvalidProduction { prod_id } => {
                write!(
                    f,
                    "LR invalid production id {prod_id} in action table. \
                     Fix: rebuild tables so every reduce action references a valid production."
                )
            }
            Self::StackUnderflow => {
                write!(
                    f,
                    "LR stack underflow on reduce. \
                     Fix: verify push/pop balance in grammar and that goto table matches."
                )
            }
            Self::NoGoto { state, nonterminal } => {
                write!(
                    f,
                    "LR missing goto: state={state} nt={nonterminal}. \
                     Fix: regenerate goto table from closure sets."
                )
            }
        }
    }
}

impl std::error::Error for ParseLrWitnessError {}

/// Parse a token stream sequentially against precomputed LR tables.
///
/// Returns the sequence of production ids that were reduced.
///
/// # Errors
///
/// Returns [`ParseLrWitnessError`] on syntax errors or internal table mismatches.
pub fn parse_lr_witness(
    action_table: &[u32],
    goto_table: &[u32],
    productions: &[LrProduction],
    num_tokens: u32,
    num_nonterminals: u32,
    tokens: &[u32],
) -> Result<Vec<u32>, ParseLrWitnessError> {
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    stack.push(0);
    let mut pos = 0usize;
    let mut reductions: Vec<u32> = Vec::with_capacity(tokens.len());

    loop {
        let state = *stack.last().ok_or(ParseLrWitnessError::StackUnderflow)?;
        let token = if pos < tokens.len() {
            tokens[pos]
        } else {
            TOK_EOF
        };
        if token >= num_tokens {
            return Err(ParseLrWitnessError::UnexpectedToken { state, token, pos });
        }

        let action_idx = (state as usize) * (num_tokens as usize) + (token as usize);
        let packed_action = action_table.get(action_idx).copied().unwrap_or(0);
        let action = LrAction::unpack(packed_action);

        match action {
            LrAction::Accept => return Ok(reductions),
            LrAction::Shift(next_state) => {
                stack.push(next_state);
                pos += 1;
            }
            LrAction::Reduce(prod_id) => {
                let prod = productions
                    .get(prod_id as usize)
                    .ok_or(ParseLrWitnessError::InvalidProduction { prod_id })?;
                if prod_id == 0 {
                    return Ok(reductions);
                }
                if stack.len() <= prod.rhs_len as usize {
                    return Err(ParseLrWitnessError::StackUnderflow);
                }
                for _ in 0..prod.rhs_len {
                    stack.pop();
                }
                let new_state = *stack.last().ok_or(ParseLrWitnessError::StackUnderflow)?;
                if prod.lhs >= num_nonterminals {
                    return Err(ParseLrWitnessError::NoGoto {
                        state: new_state,
                        nonterminal: prod.lhs,
                    });
                }
                let goto_idx =
                    (new_state as usize) * (num_nonterminals as usize) + (prod.lhs as usize);
                let goto_state = goto_table.get(goto_idx).copied().unwrap_or(u32::MAX);
                if goto_state == u32::MAX {
                    return Err(ParseLrWitnessError::NoGoto {
                        state: new_state,
                        nonterminal: prod.lhs,
                    });
                }
                stack.push(goto_state);
                reductions.push(prod_id);
            }
            LrAction::Error => {
                return Err(ParseLrWitnessError::UnexpectedToken { state, token, pos });
            }
        }
    }
}

/// Sequential mathematical witness for C/C++ line-splice classification.
///
/// Returns one `u32 ∈ {0, 1}` per input byte: 0 for spliced-out bytes
/// (`\` followed by `\n`, `\r`, or `\r\n`), 1 for kept bytes.
#[must_use]
pub fn line_splice_classify_witness(source: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    line_splice_classify_witness_into(source, &mut out);
    out
}

/// Sequential mathematical witness for line-splice classification into caller storage.
pub fn line_splice_classify_witness_into(source: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(source.len());
    for i in 0..source.len() {
        let b_m2 = i.checked_sub(2).map(|j| source[j]).unwrap_or(0);
        let b_m1 = i.checked_sub(1).map(|j| source[j]).unwrap_or(0);
        let b_0 = source[i];
        let b_p1 = source.get(i + 1).copied().unwrap_or(0);
        let case1 = b_0 == b'\\' && b_p1 == b'\n';
        let case2 = b_0 == b'\\' && b_p1 == b'\r';
        let case3 = b_m1 == b'\\' && b_0 == b'\n';
        let case4 = b_m1 == b'\\' && b_0 == b'\r';
        let case5 = b_m2 == b'\\' && b_m1 == b'\r' && b_0 == b'\n';
        let dropped = case1 || case2 || case3 || case4 || case5;
        out.push(u32::from(!dropped));
    }
}

/// Returns true if `byte` is one of ASCII whitespace {SP, TAB, LF, CR}.
#[must_use]
#[inline]
pub const fn is_structural_whitespace_witness(byte: u8) -> bool {
    matches!(byte, 0x20 | 0x09 | 0x0A | 0x0D)
}

/// Sequential mathematical witness for whitespace word classification.
///
/// Maps packed 4-byte little-endian `u32` words into 4-bit whitespace bitmasks.
#[must_use]
pub fn whitespace_classify_word_witness(words_in: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    whitespace_classify_word_witness_into(words_in, &mut out);
    out
}

/// Sequential mathematical witness for whitespace word classification into caller storage.
pub fn whitespace_classify_word_witness_into(words_in: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(words_in.len());
    for word in words_in {
        let bytes = [
            (*word & 0xFF) as u8,
            ((*word >> 8) & 0xFF) as u8,
            ((*word >> 16) & 0xFF) as u8,
            ((*word >> 24) & 0xFF) as u8,
        ];
        let mut mask = 0u32;
        for (lane, byte) in bytes.iter().enumerate() {
            if is_structural_whitespace_witness(*byte) {
                mask |= 1u32 << lane;
            }
        }
        out.push(mask);
    }
}
