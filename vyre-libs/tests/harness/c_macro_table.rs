//! GPU-resident macro table fixture shared by the named-macro expansion contracts.
//!
//! The table geometry, the FNV-1a name hash, the multiplicative slot index and the probe order are
//! the megakernel's own layout, so a test that builds its own copy of them stops proving anything
//! about the pass the moment one of them changes. Every contract that installs a macro and runs
//! `opt_named_macro_expansion` builds its input here.

use vyre::ir::Expr;
use vyre_libs::parsing::c::preprocess::expansion::{
    opt_named_macro_expansion, C_MACRO_KIND_OBJECT_LIKE, C_MACRO_REPLACEMENT_LITERAL,
};
use vyre_primitives::wire::pack_u32_slice as u32_bytes;
use vyre_reference::value::Value;

/// Hash slot sentinel for an unoccupied macro entry.
pub(crate) const EMPTY_SLOT: u32 = u32::MAX;
/// Macro hash table capacity, a power of two so the probe wraps by mask.
pub(crate) const TABLE_SLOTS: usize = 4096;
/// Probe mask for [`TABLE_SLOTS`].
pub(crate) const TABLE_MASK: u32 = 4095;
/// Byte capacity of the macro-name pool.
pub(crate) const NAME_POOL_BYTES: usize = 16_384;
const FNV1A32_OFFSET: u32 = 0x811c_9dc5;
const FNV1A32_PRIME: u32 = 0x0100_0193;

/// Expand source bytes to one word per byte, the layout the pass reads its haystack in.
pub(crate) fn source_words(source: &[u8]) -> Vec<u32> {
    source.iter().map(|byte| u32::from(*byte)).collect()
}

/// FNV-1a 32 over macro-name bytes.
pub(crate) fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut hash = FNV1A32_OFFSET;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV1A32_PRIME);
    }
    hash
}

/// First probe slot for a macro-name hash.
pub(crate) fn macro_slot(hash: u32) -> usize {
    (hash.wrapping_mul(2_654_435_769) & TABLE_MASK) as usize
}

/// The macro table, name pool and replacement arena as the pass consumes them.
#[derive(Clone)]
pub(crate) struct NamedMacroFixture {
    pub(crate) name_hashes: Vec<u32>,
    pub(crate) name_starts: Vec<u32>,
    pub(crate) name_lens: Vec<u32>,
    pub(crate) name_words: Vec<u32>,
    pub(crate) vals: Vec<u32>,
    pub(crate) sizes: Vec<u32>,
    pub(crate) kinds: Vec<u32>,
    pub(crate) param_counts: Vec<u32>,
    pub(crate) replacement_params: Vec<u32>,
    next_name_offset: usize,
}

impl NamedMacroFixture {
    /// An empty table: every slot carries [`EMPTY_SLOT`] and every macro is object-like.
    pub(crate) fn empty() -> Self {
        Self {
            name_hashes: vec![EMPTY_SLOT; TABLE_SLOTS],
            name_starts: vec![0; TABLE_SLOTS],
            name_lens: vec![0; TABLE_SLOTS],
            name_words: vec![0; NAME_POOL_BYTES],
            vals: vec![0; TABLE_SLOTS],
            sizes: vec![0; TABLE_SLOTS],
            kinds: vec![C_MACRO_KIND_OBJECT_LIKE; TABLE_SLOTS],
            param_counts: vec![0; TABLE_SLOTS],
            replacement_params: vec![C_MACRO_REPLACEMENT_LITERAL; TABLE_SLOTS],
            next_name_offset: 0,
        }
    }

    /// Copy `name` into the pool and point `slot` at it. Public because a bounds contract stages a
    /// name range by hand before running the pass.
    pub(crate) fn install_name(&mut self, slot: usize, name: &[u8]) {
        let start = self.next_name_offset;
        let end = start + name.len();
        // The pass supports names past the legacy 16k pool, and one contract covers exactly that,
        // so the pool grows rather than capping the fixture at its initial size.
        if end > self.name_words.len() {
            self.name_words.resize(end, 0);
        }
        for (idx, byte) in name.iter().enumerate() {
            self.name_words[start + idx] = u32::from(*byte);
        }
        self.name_starts[slot] = start as u32;
        self.name_lens[slot] = name.len() as u32;
        self.next_name_offset = end;
    }

    /// Install `name` at its hashed slot, linear-probing past occupied slots.
    pub(crate) fn insert(
        &mut self,
        name: &[u8],
        replacement_offset: usize,
        kind: u32,
        param_count: u32,
        replacement: &[(u32, u32)],
    ) {
        let name_hash = fnv1a32(name);
        let mut slot = macro_slot(name_hash);
        while self.name_hashes[slot] != EMPTY_SLOT {
            slot = (slot + 1) & (TABLE_SLOTS - 1);
        }
        self.name_hashes[slot] = name_hash;
        self.install_name(slot, name);
        self.vals[slot] = replacement_offset as u32;
        self.kinds[slot] = kind;
        self.param_counts[slot] = param_count;
        self.write_replacement(replacement_offset, replacement);
    }

    /// Install `name` at an exact slot under an exact hash, so a test can stage a collision the
    /// hash function would not otherwise produce.
    pub(crate) fn insert_at_slot_with_hash(
        &mut self,
        slot: usize,
        hash: u32,
        name: &[u8],
        replacement_offset: usize,
        kind: u32,
        replacement: &[(u32, u32)],
    ) {
        assert_eq!(self.name_hashes[slot], EMPTY_SLOT);
        self.name_hashes[slot] = hash;
        self.install_name(slot, name);
        self.vals[slot] = replacement_offset as u32;
        self.kinds[slot] = kind;
        self.param_counts[slot] = 0;
        self.write_replacement(replacement_offset, replacement);
    }

    fn write_replacement(&mut self, replacement_offset: usize, replacement: &[(u32, u32)]) {
        self.sizes[replacement_offset] = replacement.len() as u32;
        for (idx, (tok, param)) in replacement.iter().enumerate() {
            self.vals[replacement_offset + idx] = *tok;
            self.replacement_params[replacement_offset + idx] = *param;
        }
    }
}

/// An extracted token stream over a source buffer.
pub(crate) struct TokenStream<'a> {
    pub(crate) source: &'a [u8],
    pub(crate) types: Vec<u32>,
    pub(crate) starts: Vec<u32>,
    pub(crate) lens: Vec<u32>,
}

/// Run the named-macro expansion pass over `stream` against `fixture`.
pub(crate) fn run_named_macro_expansion(
    stream: &TokenStream<'_>,
    fixture: &NamedMacroFixture,
    max_out_tokens: u32,
) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    let program = opt_named_macro_expansion(
        "in_tok_types",
        "in_tok_starts",
        "in_tok_lens",
        "source_words",
        "macro_name_hashes",
        "macro_name_starts",
        "macro_name_lens",
        "macro_name_words",
        "macro_vals",
        "macro_sizes",
        "macro_kinds",
        "macro_param_counts",
        "macro_replacement_params",
        "out_tok_types",
        "out_tok_counts",
        Expr::u32(stream.types.len() as u32),
        Expr::u32(stream.source.len() as u32),
        max_out_tokens,
    );
    let values = [
        Value::from(u32_bytes(&stream.types)),
        Value::from(u32_bytes(&stream.starts)),
        Value::from(u32_bytes(&stream.lens)),
        Value::from(u32_bytes(&source_words(stream.source))),
        Value::from(u32_bytes(&fixture.name_hashes)),
        Value::from(u32_bytes(&fixture.name_starts)),
        Value::from(u32_bytes(&fixture.name_lens)),
        Value::from(u32_bytes(&fixture.name_words)),
        Value::from(u32_bytes(&fixture.vals)),
        Value::from(u32_bytes(&fixture.sizes)),
        Value::from(u32_bytes(&fixture.kinds)),
        Value::from(u32_bytes(&fixture.param_counts)),
        Value::from(u32_bytes(&fixture.replacement_params)),
        Value::from(vec![0u8; max_out_tokens.max(1) as usize * 4]),
        Value::from(vec![0u8; 4]),
    ];
    vyre_reference::reference_eval(&program, &values)
}
