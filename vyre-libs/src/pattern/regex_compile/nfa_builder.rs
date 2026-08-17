//! Thompson construction state: the builder, its byte-set alphabet, and the
//! state-major table emission the accelerator consumes.

use super::construct_budget::STATE_CAP;
use super::{RegexCompileError, LANES};

// ---- Thompson NFA construction over byte transitions ----

#[derive(Debug)]
pub(super) struct NfaBuilder {
    state_count: usize,
    /// Flat byte transitions. Emission consumes the stream directly,
    /// so construction does not need one allocation per NFA state.
    transitions: Vec<ByteTransition>,
    /// Flat epsilon (free) transitions.
    epsilons: Vec<(u32, u32)>,
}

#[derive(Debug, Clone)]
struct ByteTransition {
    src: u32,
    set: ByteSet,
    dst: u32,
}

#[derive(Debug, Clone)]
pub(super) struct ByteSet {
    pub(super) bits: [u64; 4], // 256 bits → 4 u64s
}

impl ByteSet {
    pub(super) fn new() -> Self {
        Self { bits: [0; 4] }
    }
    pub(super) fn insert(&mut self, b: u8) {
        self.bits[(b / 64) as usize] |= 1u64 << (b % 64);
    }
    pub(super) fn from_byte(b: u8) -> Self {
        let mut s = Self::new();
        s.insert(b);
        s
    }
    pub(super) fn from_range(lo: u8, hi: u8) -> Self {
        let mut s = Self::new();
        for b in lo..=hi {
            s.insert(b);
        }
        s
    }
    pub(super) fn for_each_set_byte(&self, mut f: impl FnMut(u8)) {
        for (word_idx, &word) in self.bits.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                f((word_idx * 64 + bit) as u8);
                bits &= bits - 1;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Fragment {
    pub(super) start: u32,
    pub(super) end: u32,
    /// Sum of byte-steps along the longest path. Used as the
    /// `pattern_len` reported in `NfaPlan::accept_states`.
    pub(super) match_len: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PatternAnchors {
    pub(super) start: bool,
    pub(super) end: bool,
}

impl NfaBuilder {
    pub(super) fn new() -> Self {
        Self {
            state_count: 0,
            transitions: Vec::new(),
            epsilons: Vec::new(),
        }
    }

    pub(super) fn state_count(&self) -> usize {
        self.state_count
    }

    pub(super) fn fresh_state(&mut self) -> Result<u32, RegexCompileError> {
        if self.state_count >= STATE_CAP {
            return Err(RegexCompileError::TooManyStates {
                states: self.state_count.saturating_add(1),
                cap: STATE_CAP,
            });
        }
        let state =
            u32::try_from(self.state_count).map_err(|_| RegexCompileError::TooManyStates {
                states: self.state_count,
                cap: STATE_CAP,
            })?;
        self.state_count =
            self.state_count
                .checked_add(1)
                .ok_or(RegexCompileError::TooManyStates {
                    states: usize::MAX,
                    cap: STATE_CAP,
                })?;
        Ok(state)
    }

    pub(super) fn add_byte_transition(&mut self, src: u32, set: ByteSet, dst: u32) {
        self.transitions.push(ByteTransition { src, set, dst });
    }

    pub(super) fn add_epsilon(&mut self, src: u32, dst: u32) {
        self.epsilons.push((src, dst));
    }

    /// State-major emission, matching the contract of
    /// `nfa::build_transition_table` + `build_epsilon_table`.
    pub(super) fn emit_state_major_tables(
        &self,
    ) -> Result<(Vec<u32>, Vec<u32>), RegexCompileError> {
        let n = self.state_count();
        let mut transitions = zeroed_u32_table(
            table_word_count(n, 256, "transition")?,
            "transition table word",
        )?;
        let mut epsilons =
            zeroed_u32_table(table_word_count(n, 1, "epsilon")?, "epsilon table word")?;

        for edge in &self.transitions {
            let src = edge.src as usize;
            let dst_lane = (edge.dst / 32) as usize;
            let dst_bit = 1u32 << (edge.dst % 32);
            edge.set.for_each_set_byte(|byte| {
                let idx = src * 256 * LANES + (byte as usize) * LANES + dst_lane;
                transitions[idx] |= dst_bit;
            });
        }
        for &(src, dst) in &self.epsilons {
            let dst_lane = (dst / 32) as usize;
            let dst_bit = 1u32 << (dst % 32);
            let idx = src as usize * LANES + dst_lane;
            epsilons[idx] |= dst_bit;
        }
        Ok((transitions, epsilons))
    }
}

fn table_word_count(
    states: usize,
    byte_columns: usize,
    table: &'static str,
) -> Result<usize, RegexCompileError> {
    states
        .checked_mul(byte_columns)
        .and_then(|words| words.checked_mul(LANES))
        .ok_or(RegexCompileError::TableWordCountOverflow { table })
}

fn zeroed_u32_table(words: usize, field: &'static str) -> Result<Vec<u32>, RegexCompileError> {
    let mut table = Vec::new();
    reserve_vec(&mut table, words, field)?;
    table.resize(words, 0);
    Ok(table)
}

pub(super) fn reserve_vec<T>(
    vec: &mut Vec<T>,
    requested: usize,
    field: &'static str,
) -> Result<(), RegexCompileError> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(vec, requested).map_err(|source| {
        RegexCompileError::StorageReserveFailed {
            field,
            requested,
            message: source.to_string(),
        }
    })
}
