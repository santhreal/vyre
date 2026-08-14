//! The macro-name hash table the preprocessor expansion ops read.
//!
//! `opt_dynamic_macro_expansion` looks a macro name up by hashing its token id
//! into a fixed slot count, so every test that drives expansion builds the same
//! three parallel buffers. This module owns that layout; each test root owns the
//! way it feeds the buffers to the reference oracle or a backend, because the
//! input shaping differs (empty streams and capacity boundaries are contracts of
//! their own).

#![allow(deprecated)]

/// Slot value meaning "no macro defined here".
pub(crate) const EMPTY_SLOT: u32 = u32::MAX;
/// Slots in the macro table, matching the op's fixed capacity.
pub(crate) const TABLE_SLOTS: usize = 4096;

/// Slot index for a macro name token, the same mapping the op computes.
pub(crate) fn hash_token(tok: u32) -> usize {
    (tok.wrapping_mul(2_654_435_769) & (TABLE_SLOTS as u32 - 1)) as usize
}

/// The three parallel buffers a macro table is passed to the expander as.
pub(crate) struct MacroFixture {
    pub(crate) keys: Vec<u32>,
    pub(crate) vals: Vec<u32>,
    pub(crate) sizes: Vec<u32>,
}

impl MacroFixture {
    /// A table with no macro defined in any slot.
    pub(crate) fn empty() -> Self {
        Self {
            keys: vec![EMPTY_SLOT; TABLE_SLOTS],
            vals: vec![0; TABLE_SLOTS],
            sizes: vec![0; TABLE_SLOTS],
        }
    }

    /// Define `token` as expanding to `replacement`, stored at
    /// `replacement_offset` in the value buffer.
    pub(crate) fn insert(&mut self, token: u32, replacement_offset: usize, replacement: &[u32]) {
        let slot = hash_token(token);
        self.keys[slot] = token;
        self.vals[slot] = replacement_offset as u32;
        self.sizes[replacement_offset] = replacement.len() as u32;
        for (idx, value) in replacement.iter().enumerate() {
            self.vals[replacement_offset + idx] = *value;
        }
    }
}
