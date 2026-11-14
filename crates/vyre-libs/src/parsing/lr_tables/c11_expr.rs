//! Precomputed LR tables for the C11 expression grammar.
//!
//! `vyre_spec::c11_expr_token` owns the terminal numbering; those ids are the
//! column order of [`ACTION_TABLE`] and the token values a caller feeds the
//! GPU parser. Name that module to read or write a token id.

use super::action::{pack_accept, pack_error, pack_reduce, pack_shift};
use super::{LrTables, Production};

/// Expression nonterminal id.
pub const NT_E: u32 = 0;
/// Term nonterminal id.
pub const NT_T: u32 = 1;
/// Factor nonterminal id.
pub const NT_F: u32 = 2;

const ERR: u32 = pack_error();
const ACC: u32 = pack_accept();

const fn sh(state: u32) -> u32 {
    pack_shift(state)
}

const fn rd(prod: u32) -> u32 {
    pack_reduce(prod)
}

const fn prod(lhs: u32, rhs_len: u32) -> Production {
    Production { lhs, rhs_len }
}

/// Precomputed action table: 17 states x 9 tokens.
#[rustfmt::skip]
pub static ACTION_TABLE: &[u32] = &[
    sh(4), sh(5), ERR,   ERR,   ERR,   ERR,    sh(6),  ERR,   ERR,
    ERR,   ERR,   sh(7), sh(8), ERR,   ERR,    ERR,    ERR,   ACC,
    ERR,   ERR,   rd(3), rd(3), sh(9), sh(10), ERR,    rd(3), rd(3),
    ERR,   ERR,   rd(6), rd(6), rd(6), rd(6),  ERR,    rd(6), rd(6),
    ERR,   ERR,   rd(8), rd(8), rd(8), rd(8),  ERR,    rd(8), rd(8),
    ERR,   ERR,   rd(9), rd(9), rd(9), rd(9),  ERR,    rd(9), rd(9),
    sh(4), sh(5), ERR,   ERR,   ERR,   ERR,    sh(6),  ERR,   ERR,
    sh(4), sh(5), ERR,   ERR,   ERR,   ERR,    sh(6),  ERR,   ERR,
    sh(4), sh(5), ERR,   ERR,   ERR,   ERR,    sh(6),  ERR,   ERR,
    sh(4), sh(5), ERR,   ERR,   ERR,   ERR,    sh(6),  ERR,   ERR,
    sh(4), sh(5), ERR,   ERR,   ERR,   ERR,    sh(6),  ERR,   ERR,
    ERR,   ERR,   sh(7), sh(8), ERR,   ERR,    ERR,    sh(16), ERR,
    ERR,   ERR,   rd(1), rd(1), sh(9), sh(10), ERR,    rd(1), rd(1),
    ERR,   ERR,   rd(2), rd(2), sh(9), sh(10), ERR,    rd(2), rd(2),
    ERR,   ERR,   rd(4), rd(4), rd(4), rd(4),  ERR,    rd(4), rd(4),
    ERR,   ERR,   rd(5), rd(5), rd(5), rd(5),  ERR,    rd(5), rd(5),
    ERR,   ERR,   rd(7), rd(7), rd(7), rd(7),  ERR,    rd(7), rd(7),
];

const NO: u32 = u32::MAX;

/// Precomputed goto table: 17 states x 3 nonterminals.
#[rustfmt::skip]
pub static GOTO_TABLE: &[u32] = &[
    1,  2,  3,
    NO, NO, NO,
    NO, NO, NO,
    NO, NO, NO,
    NO, NO, NO,
    NO, NO, NO,
    11, 2,  3,
    NO, 12, 3,
    NO, 13, 3,
    NO, NO, 14,
    NO, NO, 15,
    NO, NO, NO,
    NO, NO, NO,
    NO, NO, NO,
    NO, NO, NO,
    NO, NO, NO,
    NO, NO, NO,
];

/// Production rules for the C expression grammar.
pub static PRODUCTIONS: &[Production] = &[
    prod(NT_E, 1),
    prod(NT_E, 3),
    prod(NT_E, 3),
    prod(NT_E, 1),
    prod(NT_T, 3),
    prod(NT_T, 3),
    prod(NT_T, 1),
    prod(NT_F, 3),
    prod(NT_F, 1),
    prod(NT_F, 1),
];

/// Static precomputed tables for the C expression grammar.
pub static C11_EXPR: LrTables = LrTables {
    num_states: 17,
    num_tokens: 9,
    num_nonterminals: 3,
    action: ACTION_TABLE,
    goto: GOTO_TABLE,
    productions: PRODUCTIONS,
};
