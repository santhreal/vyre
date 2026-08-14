//! The token-keyed macro table the dynamic-expansion op takes, and the run
//! that drives it through the reference interpreter.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same three parallel buffers and pass them in the same
//! order, so the table and the run have one owner here rather than a copy per
//! suite.

use vyre::ir::Expr;
use vyre_libs::parsing::c::preprocess::expansion::opt_dynamic_macro_expansion;
use vyre_primitives::wire::pack_u32_slice;
use vyre_reference::value::Value;

/// Slot value meaning no macro is defined at that slot.
pub(crate) const EMPTY_SLOT: u32 = u32::MAX;
/// Slots in the macro table, a power of two so the probe wraps by mask.
pub(crate) const TABLE_SLOTS: usize = 4096;
/// Probe mask for [`TABLE_SLOTS`].
pub(crate) const TABLE_MASK: u32 = TABLE_SLOTS as u32 - 1;

/// The slot a macro name token hashes to, the same mapping the op computes.
#[must_use]
pub(crate) fn hash_token(tok: u32) -> usize {
    (tok.wrapping_mul(2_654_435_769) & TABLE_MASK) as usize
}

/// The three parallel buffers a macro table is passed to the expander as.
#[derive(Clone)]
pub(crate) struct MacroFixture {
    pub(crate) keys: Vec<u32>,
    pub(crate) vals: Vec<u32>,
    pub(crate) sizes: Vec<u32>,
}

impl MacroFixture {
    /// A table with no macro defined: every key carries [`EMPTY_SLOT`].
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            keys: vec![EMPTY_SLOT; TABLE_SLOTS],
            vals: vec![0; TABLE_SLOTS],
            sizes: vec![0; TABLE_SLOTS],
        }
    }

    /// Define `token` as expanding to `replacement`, stored at
    /// `replacement_offset` in the value arena.
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

/// The dynamic-expansion program the macro suites drive, on the reference
/// interpreter or on a backend.
#[must_use]
pub(crate) fn dynamic_macro_expansion_program(
    input_len: usize,
    max_out_tokens: u32,
) -> vyre_foundation::ir::Program {
    opt_dynamic_macro_expansion(
        "in_tok_types",
        "macro_keys",
        "macro_vals",
        "macro_sizes",
        "out_tok_types",
        "out_tok_counts",
        Expr::u32(input_len as u32),
        max_out_tokens,
    )
}

/// Run the dynamic macro expansion over `input` against `fixture`.
///
/// An empty input and a zero output capacity still get one word of buffer,
/// because a zero-length binding is not a buffer the interpreter accepts and
/// the empty stream is a case the contracts assert on.
pub(crate) fn run_dynamic_macro_expansion(
    input: &[u32],
    fixture: &MacroFixture,
    max_out_tokens: u32,
) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    let program = dynamic_macro_expansion_program(input.len(), max_out_tokens);
    let input_bytes = if input.is_empty() {
        vec![0u8; 4]
    } else {
        pack_u32_slice(input)
    };
    let values = [
        Value::from(input_bytes),
        Value::from(pack_u32_slice(&fixture.keys)),
        Value::from(pack_u32_slice(&fixture.vals)),
        Value::from(pack_u32_slice(&fixture.sizes)),
        Value::from(vec![0u8; max_out_tokens.max(1) as usize * 4]),
        Value::from(vec![0u8; 4]),
    ];
    vyre_reference::reference_eval(&program, &values)
}
