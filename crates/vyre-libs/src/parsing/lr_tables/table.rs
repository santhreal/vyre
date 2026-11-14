//! Static LR tables: productions and the action and goto slices a parse reads.

/// A single grammar production: `lhs nonterminal -> rhs_len symbols`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Production {
    /// Nonterminal index on the left-hand side.
    pub lhs: u32,
    /// Number of symbols to pop from the parser stack on reduce.
    pub rhs_len: u32,
}

/// Precomputed LR tables backed by static slices.
///
/// Clone is `O(1)` because the payload is only scalar metadata plus slice
/// pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LrTables {
    /// Number of parser states.
    pub num_states: u32,
    /// Number of terminal symbols, including EOF.
    pub num_tokens: u32,
    /// Number of nonterminal symbols.
    pub num_nonterminals: u32,
    /// Flat action table: `action[state * num_tokens + token]`.
    pub action: &'static [u32],
    /// Flat goto table: `goto[state * num_nonterminals + nt]`.
    /// `u32::MAX` means "no goto".
    pub goto: &'static [u32],
    /// Production rules indexed by production id.
    pub productions: &'static [Production],
}
