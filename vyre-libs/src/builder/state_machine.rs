//! Canonical table-walking state machine / DFA composer.
//!
//! Unifies table lookup, 2D table addressing, and state transition loops
//! (`state' = table[state * stride + symbol]`) across pattern matching,
//! lexing, decoding, character classification, and LR parsing.
//!
//! # Transition Semantics
//!
//! A table-driven automaton represents its transition function as a flat
//! row-major buffer:
//!
//! ```text
//! transitions[state * stride + symbol] = next_state
//! ```
//!
//! For byte-oriented DFAs (Aho-Corasick, regex DFAs, lexers), `stride` is
//! typically 256 (`DEFAULT_BYTE_ALPHABET_SIZE`). For LR parsers, `stride` is
//! the number of grammar terminals or nonterminals.
//!
//! [`TableStateMachineComposer`](crate::builder::state_machine::TableStateMachineComposer) encapsulates:
//! 1. Transition index and load expressions (`transition_index`, `transition_expr`).
//! 2. State update nodes (`advance_node`, `advance_step_nodes`).
//! 3. Sequential loops and bounded suffix replay walks (`walk_loop`, `linear_scan_body`).
//! 4. Tiled decode-to-scan bodies with ping-pong buffering (`tiled_decode_scan_body`).
//! 5. Host-side flat index calculators (`flat_index`, `flat_byte_index`).

use vyre_foundation::ir::{DataType, Expr, Node};

/// Default alphabet size for byte-driven DFAs (0..=255).
pub const DEFAULT_BYTE_ALPHABET_SIZE: u32 = 256;

/// Default state variable name in generated IR.
pub const DEFAULT_STATE_VAR: &str = "state";

/// Shared composer for table-walking state machines and 2D table lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableStateMachineComposer<'a> {
    /// Buffer name carrying the transition table.
    pub transitions: &'a str,
    /// Row stride for `state * stride + symbol` lookup.
    pub stride: u32,
    /// Variable name holding the current state in IR.
    pub state_var: &'a str,
    /// Alphabet size / symbol domain bound.
    pub alphabet_size: u32,
}

impl<'a> TableStateMachineComposer<'a> {
    /// Create a new byte-oriented DFA composer with default stride 256 and state var `"state"`.
    #[must_use]
    pub const fn new(transitions: &'a str) -> Self {
        Self {
            transitions,
            stride: DEFAULT_BYTE_ALPHABET_SIZE,
            state_var: DEFAULT_STATE_VAR,
            alphabet_size: DEFAULT_BYTE_ALPHABET_SIZE,
        }
    }

    /// Create a composer with an explicit transition row stride.
    #[must_use]
    pub const fn with_stride(transitions: &'a str, stride: u32) -> Self {
        Self {
            transitions,
            stride,
            state_var: DEFAULT_STATE_VAR,
            alphabet_size: stride,
        }
    }

    /// Create a composer with an explicit alphabet size.
    #[must_use]
    pub const fn with_alphabet_size(transitions: &'a str, alphabet_size: u32) -> Self {
        Self {
            transitions,
            stride: alphabet_size,
            state_var: DEFAULT_STATE_VAR,
            alphabet_size,
        }
    }

    /// Override the state variable name.
    #[must_use]
    pub const fn state_var(mut self, var: &'a str) -> Self {
        self.state_var = var;
        self
    }

    /// Override the row stride.
    #[must_use]
    pub const fn stride(mut self, stride: u32) -> Self {
        self.stride = stride;
        self
    }

    /// Override the alphabet size.
    #[must_use]
    pub const fn alphabet_size(mut self, alphabet_size: u32) -> Self {
        self.alphabet_size = alphabet_size;
        self
    }

    /// Compute the flat 1D index `state * stride + symbol` as an [`Expr`].
    #[must_use]
    pub fn transition_index(&self, state: Expr, symbol: Expr) -> Expr {
        Expr::add(Expr::mul(state, Expr::u32(self.stride)), symbol)
    }

    /// Emit an [`Expr`] loading the next state `table[state * stride + symbol]`.
    #[must_use]
    pub fn transition_expr(&self, state: Expr, symbol: Expr) -> Expr {
        Expr::load(self.transitions, self.transition_index(state, symbol))
    }

    /// Emit a bounds-safe transition load where `symbol` is masked with `& 0xFF`.
    #[must_use]
    pub fn safe_byte_transition_expr(&self, state: Expr, byte: Expr) -> Expr {
        let masked = Expr::bitand(byte, Expr::u32(0xFF));
        self.transition_expr(state, masked)
    }

    /// Emit a state assignment node: `state_var = table[state_var * stride + symbol]`.
    #[must_use]
    pub fn advance_node(&self, symbol: Expr) -> Node {
        Node::assign(
            self.state_var,
            self.transition_expr(Expr::var(self.state_var), symbol),
        )
    }

    /// Emit a state assignment node targeting an explicitly named state variable.
    #[must_use]
    pub fn advance_node_named(&self, state_var: &str, symbol: Expr) -> Node {
        Node::assign(
            state_var,
            self.transition_expr(Expr::var(state_var), symbol),
        )
    }

    /// Advance state from an element loaded from `input` at index `idx`.
    #[must_use]
    pub fn advance_step_nodes(&self, input: &str, idx: Expr) -> Vec<Node> {
        vec![self.advance_node(Expr::load(input, idx))]
    }

    /// Emit a loop walking transitions from `start` to `end`.
    #[must_use]
    pub fn walk_loop(&self, loop_var: &str, start: Expr, end: Expr, step_nodes: Vec<Node>) -> Node {
        Node::loop_for(loop_var, start, end, step_nodes)
    }

    /// Emit a loop loading symbols from `input` and advancing state for each index in `start..end`.
    #[must_use]
    pub fn walk_input_slice(&self, loop_var: &str, input: &str, start: Expr, end: Expr) -> Node {
        Node::loop_for(
            loop_var,
            start,
            end,
            vec![self.advance_node(Expr::load(input, Expr::var(loop_var)))],
        )
    }

    /// Build a single-invocation linear scan body over `0..valid_len`.
    ///
    /// The scanner walks `input`, updates `self.state_var`, and writes
    /// `accept[state]` into `matches[step]`.
    #[must_use]
    pub fn linear_scan_body(
        &self,
        input: &str,
        accept: &str,
        matches: &str,
        valid_len: Expr,
    ) -> Vec<Node> {
        vec![Node::if_then(
            Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
            vec![
                Node::let_bind(self.state_var, Expr::u32(0)),
                Node::loop_for(
                    "decode_scan_step",
                    Expr::u32(0),
                    valid_len,
                    vec![
                        Node::let_bind("byte", Expr::load(input, Expr::var("decode_scan_step"))),
                        self.advance_node(Expr::var("byte")),
                        Node::store(
                            matches,
                            Expr::var("decode_scan_step"),
                            Expr::load(accept, Expr::var(self.state_var)),
                        ),
                    ],
                ),
            ],
        )]
    }

    /// Build a per-invocation bounded suffix scan body.
    ///
    /// Each invocation walks the suffix window ending at byte `i`
    /// (`max(0, i + 1 - max_pattern_len)..=i`) and writes `accept[state]`
    /// into `matches[i]`.
    #[must_use]
    pub fn bounded_suffix_scan_body(
        &self,
        haystack: &str,
        accept: &str,
        matches: &str,
        max_pattern_len: u32,
    ) -> Vec<Node> {
        let max_pattern_len = max_pattern_len.max(1);
        let i = Expr::var("i");
        let end = Expr::add(i.clone(), Expr::u32(1));
        let start = Expr::select(
            Expr::lt(i.clone(), Expr::u32(max_pattern_len - 1)),
            Expr::u32(0),
            Expr::sub(end.clone(), Expr::u32(max_pattern_len)),
        );
        vec![
            Node::let_bind("i", Expr::LogicalIndex { axis: 0 }),
            Node::if_then(
                Expr::lt(i.clone(), Expr::buf_len(haystack)),
                vec![
                    Node::let_bind(self.state_var, Expr::u32(0)),
                    Node::let_bind("scan_start", start),
                    Node::loop_for(
                        "step",
                        Expr::var("scan_start"),
                        end,
                        vec![self.advance_node(Expr::load(haystack, Expr::var("step")))],
                    ),
                    Node::store(matches, i, Expr::load(accept, Expr::var(self.state_var))),
                ],
            ),
        ]
    }

    /// Build a single-invocation tiled decode-to-scan transition body with ping-pong buffering.
    #[must_use]
    pub fn tiled_decode_scan_body<ByteAt, StoreDecoded>(
        &self,
        accept: &str,
        matches: &str,
        valid_len: Expr,
        tile_width: u32,
        mut byte_at: ByteAt,
        mut store_decoded: StoreDecoded,
    ) -> Vec<Node>
    where
        ByteAt: FnMut(Expr) -> Expr,
        StoreDecoded: FnMut(Expr, Expr) -> Option<Node>,
    {
        let tile_width = tile_width.max(1).next_power_of_two();
        let tile_count = tiled_scan_tile_count_expr(valid_len.clone(), tile_width);
        vec![Node::if_then(
            Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
            vec![
                Node::let_bind(self.state_var, Expr::u32(0)),
                Node::let_bind("decode_scan_ping", Expr::u32(0)),
                Node::let_bind("decode_scan_pong", Expr::u32(0)),
                Node::loop_for(
                    "decode_scan_tile_index",
                    Expr::u32(0),
                    tile_count,
                    vec![
                        Node::let_bind(
                            "decode_scan_tile_base",
                            Expr::mul(Expr::var("decode_scan_tile_index"), Expr::u32(tile_width)),
                        ),
                        Node::loop_for(
                            "decode_scan_tile_lane",
                            Expr::u32(0),
                            Expr::u32(tile_width),
                            self.tiled_lane_body(
                                accept,
                                matches,
                                valid_len,
                                &mut byte_at,
                                &mut store_decoded,
                            ),
                        ),
                    ],
                ),
            ],
        )]
    }

    fn tiled_lane_body<ByteAt, StoreDecoded>(
        &self,
        accept: &str,
        matches: &str,
        valid_len: Expr,
        byte_at: &mut ByteAt,
        store_decoded: &mut StoreDecoded,
    ) -> Vec<Node>
    where
        ByteAt: FnMut(Expr) -> Expr,
        StoreDecoded: FnMut(Expr, Expr) -> Option<Node>,
    {
        let index = Expr::add(
            Expr::var("decode_scan_tile_base"),
            Expr::var("decode_scan_tile_lane"),
        );
        let slot_is_ping = Expr::eq(
            Expr::bitand(Expr::var("decode_scan_tile_lane"), Expr::u32(1)),
            Expr::u32(0),
        );
        let decoded = byte_at(index.clone());
        let mut body = vec![Node::let_bind("decode_scan_byte", decoded)];
        if let Some(store) = store_decoded(index.clone(), Expr::var("decode_scan_byte")) {
            body.push(store);
        }
        body.extend([
            Node::if_then_else(
                slot_is_ping,
                vec![Node::assign(
                    "decode_scan_ping",
                    Expr::var("decode_scan_byte"),
                )],
                vec![Node::assign(
                    "decode_scan_pong",
                    Expr::var("decode_scan_byte"),
                )],
            ),
            self.advance_node(Expr::var("decode_scan_byte")),
            Node::store(
                matches,
                index.clone(),
                Expr::load(accept, Expr::var(self.state_var)),
            ),
        ]);
        vec![Node::if_then(Expr::lt(index, valid_len), body)]
    }
}

impl TableStateMachineComposer<'_> {
    // -----------------------------------------------------------------------
    // Static / Free Utility Functions
    // -----------------------------------------------------------------------

    /// Compute host-side flat 1D index: `row * stride + col`.
    #[cfg(test)]
    #[must_use]
    #[inline]
    pub const fn flat_index(row: u32, stride: u32, col: u32) -> usize {
        (row as usize) * (stride as usize) + (col as usize)
    }

    /// Compute host-side flat 1D byte-DFA transition index: `state * 256 + byte`.
    #[must_use]
    #[inline]
    pub const fn flat_byte_index(state: u32, byte: u8) -> usize {
        (state as usize) * 256 + (byte as usize)
    }

    /// Look up `byte` in a fixed 256-entry table with `& 0xFF` masking.
    #[must_use]
    pub fn byte_table_lookup(table: &str, byte: Expr) -> Expr {
        vyre_primitives::ir_safe::byte_table_lookup(table, byte)
    }

    /// Widen source byte to U32 and look it up in a 256-entry table with `& 0xFF` mask.
    #[must_use]
    pub fn source_byte_table_lookup(table: &str, source: &str, index: Expr) -> Expr {
        vyre_primitives::ir_safe::source_byte_table_lookup(table, source, index)
    }

    /// General 2D table lookup: `Expr::load(table, row * stride + col)`.
    #[must_use]
    pub fn table_lookup_2d(table: &str, row: Expr, stride: u32, col: Expr) -> Expr {
        Expr::load(table, Expr::add(Expr::mul(row, Expr::u32(stride)), col))
    }

    /// Load a byte from `buffer` at `index`, masked with `& 0xFF`.
    #[must_use]
    pub fn masked_byte_load(buffer: &str, index: Expr) -> Expr {
        Expr::bitand(
            Expr::cast(DataType::U32, Expr::load(buffer, index)),
            Expr::u32(0xFF),
        )
    }

    /// Convenience helper for 256-alphabet byte transition expression.
    #[must_use]
    pub fn byte_transition_expr(transitions: &str, state: Expr, byte: Expr) -> Expr {
        TableStateMachineComposer::new(transitions).transition_expr(state, byte)
    }

    /// Convenience helper for 256-alphabet byte state advance node.
    #[must_use]
    pub fn byte_advance_state_node(state_var: &str, transitions: &str, byte: Expr) -> Node {
        TableStateMachineComposer::new(transitions)
            .state_var(state_var)
            .advance_node(byte)
    }
}

fn tiled_scan_tile_count_expr(valid_len: Expr, tile_width: u32) -> Expr {
    let tile_width = tile_width.max(1).next_power_of_two();
    Expr::select(
        Expr::eq(valid_len.clone(), Expr::u32(0)),
        Expr::u32(0),
        Expr::add(
            Expr::div(Expr::sub(valid_len, Expr::u32(1)), Expr::u32(tile_width)),
            Expr::u32(1),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_indices_match_layout_contract() {
        assert_eq!(TableStateMachineComposer::flat_byte_index(0, b'a'), 97);
        assert_eq!(
            TableStateMachineComposer::flat_byte_index(1, b'b'),
            256 + 98
        );
        assert_eq!(TableStateMachineComposer::flat_index(3, 10, 5), 35);
    }

    #[test]
    fn transition_expr_matches_stride() {
        let composer = TableStateMachineComposer::new("trans");
        let expr = composer.transition_expr(Expr::var("state"), Expr::var("byte"));
        let rendered = format!("{expr:?}");
        assert!(rendered.contains("trans"));
        assert!(rendered.contains("256"));
    }

    #[test]
    fn advance_node_assigns_state_var() {
        let composer = TableStateMachineComposer::new("trans").state_var("curr_state");
        let node = composer.advance_node(Expr::var("symbol"));
        match node {
            Node::Assign { name, value: _ } => {
                assert_eq!(name.as_ref(), "curr_state");
            }
            other => panic!("expected Assign node, got {other:?}"),
        }
    }

    #[test]
    fn custom_stride_composer() {
        let composer = TableStateMachineComposer::with_stride("action_table", 42);
        assert_eq!(composer.stride, 42);
        let index_expr = composer.transition_index(Expr::var("s"), Expr::var("tok"));
        let rendered = format!("{index_expr:?}");
        assert!(rendered.contains("42"));
    }

    #[test]
    fn linear_scan_body_construction() {
        let composer = TableStateMachineComposer::new("transitions");
        let body = composer.linear_scan_body("input", "accept", "matches", Expr::u32(100));
        assert_eq!(body.len(), 1);
    }

    #[test]
    fn tiled_decode_scan_body_construction() {
        let composer = TableStateMachineComposer::new("transitions");
        let body = composer.tiled_decode_scan_body(
            "accept",
            "matches",
            Expr::u32(64),
            16,
            |idx| Expr::load("input", idx),
            |idx, val| Some(Node::store("decoded", idx, val)),
        );
        assert_eq!(body.len(), 1);
    }

    #[test]
    fn bounded_suffix_scan_body_construction() {
        let composer = TableStateMachineComposer::new("transitions");
        let body = composer.bounded_suffix_scan_body("haystack", "accept", "matches", 16);
        assert_eq!(body.len(), 2);
    }

    #[test]
    fn table_lookup_2d_and_byte_lookup() {
        let lookup2d =
            TableStateMachineComposer::table_lookup_2d("table", Expr::var("r"), 16, Expr::var("c"));
        let rendered2d = format!("{lookup2d:?}");
        assert!(rendered2d.contains("table"));
        assert!(rendered2d.contains("16"));

        let byte_lookup = TableStateMachineComposer::byte_table_lookup("lut", Expr::var("b"));
        let rendered_byte = format!("{byte_lookup:?}");
        assert!(rendered_byte.contains("lut"));
        assert!(rendered_byte.contains("255"));
    }
}
