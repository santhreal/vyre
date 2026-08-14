//! GNU extended-asm token fixtures: operand lists, clobbers, and asm goto labels.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::{build_fixture, Fixture, FixtureToken};
use vyre_libs::parsing::c::lex::tokens::*;

/// ```c
/// asm volatile ("mov %2, %0\n\tadd %1, %0"
///   : "=r" (out0), "=r" (out1)
///   : "r" (in0), "r" (in1));
/// ```
pub(crate) fn fixture_asm_multiple_output_input_operands() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_GNU_ASM),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"mov %2, %0\\n\\tadd %1, %0\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("out0", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("out1", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("in0", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("in1", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// asm ("" : : : "memory", "cc");
/// ```
pub(crate) fn fixture_asm_memory_and_cc_clobbers() -> Fixture {
    c_tokens("asm ( \"\" : : : \"memory\" , \"cc\" ) ;")
}

/// ```c
/// asm goto ("jmp %l0\n\tjmp %l1"
///   :
///   :
///   :
///   : fail, ok);
/// ```
pub(crate) fn fixture_asm_goto_multiple_labels() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_GNU_ASM),
        FixtureToken::new("goto", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"jmp %l0\\n\\tjmp %l1\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("fail", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("ok", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// asm ("mov %[src], %[dst]"
///   : [dst] "=&r" (out)
///   : [src] "r" (in));
/// ```
pub(crate) fn fixture_asm_symbolic_names_and_earlyclobber() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_GNU_ASM),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"mov %[src], %[dst]\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("[dst]", TOK_IDENTIFIER),
        FixtureToken::new("\"=&r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("out", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("[src]", TOK_IDENTIFIER),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("in", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// __asm__ __volatile__ ("rdtsc" : "=A" (ticks));
/// ```
pub(crate) fn fixture_asm_extended_output_only() -> Fixture {
    c_tokens("__asm__ __volatile__ ( \"rdtsc\" : \"=A\" ( ticks ) ) ;")
}

/// ```c
/// asm goto ("" :::: label1, label2, label3);
/// ```
pub(crate) fn fixture_asm_goto_three_labels() -> Fixture {
    c_tokens("asm goto ( \"\" : : : : label1 , label2 , label3 ) ;")
}
