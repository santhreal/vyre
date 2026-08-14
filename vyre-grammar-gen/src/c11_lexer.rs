//! C11 max-munch lexer patterns for the host oracle.
//!
//! The token ids [`C11_PATTERNS`] emits are `vyre_spec::c11_token`, re-exported
//! here so `vyre_grammar_gen::c11_lexer::TOK_*` keeps resolving. The numbering
//! is the wire contract between the blobs this crate emits and the GPU lexer
//! and parser that decode them, so it has one declaration site in the
//! foundation-layer spec crate that both sides depend down onto. A value
//! carried in two places drifts silently: the generated tables keep validating
//! and the parser reads every affected token as something else.

use crate::dfa::{DfaBuilder, DfaTable};
use regex_automata::MatchKind;
pub use vyre_spec::c11_token::*;

/// `(token_id, regex source)` in **priority** order: earlier wins on tie length
/// in [`crate::lex_c11_max_munch`].
pub const C11_PATTERNS: &[(u32, &str)] = &[
    (TOK_AUTO, r"auto"),
    (TOK_BREAK, r"break"),
    (TOK_CASE, r"case"),
    (TOK_CHAR_KW, r"char"),
    (TOK_CONST, r"const"),
    (TOK_CONTINUE, r"continue"),
    (TOK_DEFAULT, r"default"),
    (TOK_DO, r"do"),
    (TOK_DOUBLE, r"double"),
    (TOK_ELSE, r"else"),
    (TOK_ENUM, r"enum"),
    (TOK_EXTERN, r"extern"),
    (TOK_FLOAT_KW, r"float"),
    (TOK_FOR, r"for"),
    (TOK_GOTO, r"goto"),
    (TOK_IF, r"if"),
    (TOK_INLINE, r"inline"),
    (TOK_INT, r"int"),
    (TOK_LONG, r"long"),
    (TOK_REGISTER, r"register"),
    (TOK_RESTRICT, r"restrict"),
    (TOK_RETURN, r"return"),
    (TOK_SHORT, r"short"),
    (TOK_SIGNED, r"signed"),
    (TOK_SIZEOF, r"sizeof"),
    (TOK_STATIC, r"static"),
    (TOK_STRUCT, r"struct"),
    (TOK_SWITCH, r"switch"),
    (TOK_TYPEDEF, r"typedef"),
    (TOK_UNION, r"union"),
    (TOK_UNSIGNED, r"unsigned"),
    (TOK_VOID, r"void"),
    (TOK_VOLATILE, r"volatile"),
    (TOK_WHILE, r"while"),
    (TOK_ALIGNAS, r"_Alignas"),
    (TOK_ALIGNOF, r"_Alignof"),
    (TOK_ATOMIC, r"_Atomic"),
    (TOK_BOOL, r"_Bool"),
    (TOK_COMPLEX, r"_Complex"),
    (TOK_GENERIC, r"_Generic"),
    (TOK_IMAGINARY, r"_Imaginary"),
    (TOK_NORETURN, r"_Noreturn"),
    (TOK_STATIC_ASSERT, r"_Static_assert"),
    (TOK_THREAD_LOCAL, r"_Thread_local"),
    (TOK_GNU_ASM, r"asm"),
    (TOK_GNU_ASM, r"__asm"),
    (TOK_GNU_ASM, r"__asm__"),
    (TOK_GNU_ATTRIBUTE, r"__attribute"),
    (TOK_GNU_ATTRIBUTE, r"__attribute__"),
    (TOK_GNU_TYPEOF, r"typeof"),
    (TOK_GNU_TYPEOF, r"__typeof"),
    (TOK_GNU_TYPEOF, r"__typeof__"),
    (TOK_GNU_EXTENSION, r"__extension__"),
    (TOK_ALIGNOF, r"__alignof"),
    (TOK_ALIGNOF, r"__alignof__"),
    (TOK_INLINE, r"__inline"),
    (TOK_INLINE, r"__inline__"),
    (TOK_COMPLEX, r"__complex__"),
    (TOK_GNU_REAL, r"__real__"),
    (TOK_GNU_IMAG, r"__imag__"),
    (TOK_VOLATILE, r"__volatile__"),
    (TOK_BUILTIN_CONSTANT_P, r"__builtin_constant_p"),
    (TOK_BUILTIN_CHOOSE_EXPR, r"__builtin_choose_expr"),
    (
        TOK_BUILTIN_TYPES_COMPATIBLE_P,
        r"__builtin_types_compatible_p",
    ),
    (TOK_IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_]*"),
    (TOK_INTEGER, r"0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*"),
    (TOK_STRING, r#""([^"\\]|\\.)*""#),
    (TOK_WHITESPACE, r"[ \t\n\r\v\f]+"),
    (TOK_COMMENT, r"//[^\n]*"),
    (TOK_COMMENT, r"/\*([^*]|\*[^/])*\*/"),
    (TOK_PREPROC, r"#[^\n]*"),
    (TOK_HASH, r"#"),
    (TOK_ARROW, r"->"),
    (TOK_INC, r"\+\+"),
    (TOK_DEC, r"--"),
    (TOK_PLUS_EQ, r"\+="),
    (TOK_MINUS_EQ, r"-="),
    (TOK_STAR_EQ, r"\*="),
    (TOK_SLASH_EQ, r"/="),
    (TOK_LSHIFT_EQ, r"<<="),
    (TOK_RSHIFT_EQ, r">>="),
    (TOK_PERCENT_EQ, r"%="),
    (TOK_AMP_EQ, r"&="),
    (TOK_PIPE_EQ, r"\|="),
    (TOK_CARET_EQ, r"\^="),
    (TOK_HASHHASH, r"##"),
    (TOK_EQ, r"=="),
    (TOK_NE, r"!="),
    (TOK_LE, r"<="),
    (TOK_GE, r">="),
    (TOK_AND, r"&&"),
    (TOK_OR, r"\|\|"),
    (TOK_LSHIFT, r"<<"),
    (TOK_RSHIFT, r">>"),
    (TOK_ELLIPSIS, r"\.\.\."),
    (TOK_LPAREN, r"\("),
    (TOK_RPAREN, r"\)"),
    (TOK_LBRACE, r"\{"),
    (TOK_RBRACE, r"\}"),
    (TOK_LBRACKET, r"\["),
    (TOK_RBRACKET, r"\]"),
    (TOK_SEMICOLON, r";"),
    (TOK_COMMA, r","),
    (TOK_DOT, r"\."),
    (TOK_PLUS, r"\+"),
    (TOK_MINUS, r"-"),
    (TOK_STAR, r"\*"),
    (TOK_SLASH, r"/"),
    (TOK_PERCENT, r"%"),
    (TOK_AMP, r"&"),
    (TOK_PIPE, r"\|"),
    (TOK_CARET, r"\^"),
    (TOK_TILDE, r"~"),
    (TOK_BANG, r"!"),
    (TOK_ASSIGN, r"="),
    (TOK_LT, r"<"),
    (TOK_GT, r">"),
    (TOK_QUESTION, r"\?"),
    (TOK_COLON, r":"),
];

fn add_c11_patterns(b: &mut DfaBuilder) {
    for &(id, p) in C11_PATTERNS {
        b.add_pattern(id, p);
    }
}

/// DFA for GPU / `SGGC` blobs: [`MatchKind::All`], wire-stable with existing paths.
///
/// # Panics
///
/// Panics if any pattern in [`C11_PATTERNS`] fails to compile. All patterns are
/// compile-time constants; a failure here is a programmer error (broken pattern
/// in the source), not a runtime condition. Silent recovery would produce a
/// zero-recall DFA that rejects all input on the GPU.
// INTENTIONAL: C11_PATTERNS is a compile-time constant; any build failure is a
// programmer error that must abort loudly, not silently degrade to an all-Error DFA.
#[allow(clippy::expect_used)]
pub fn build_c11_lexer_dfa() -> DfaTable {
    let mut b = DfaBuilder::new(0, 0);
    add_c11_patterns(&mut b);
    b.build().expect(
        "Fix: C11 lexer DFA build failed. A pattern in C11_PATTERNS is invalid. \
         All patterns are compile-time constants, fix the broken pattern.",
    )
}

/// **Host** DFA: [`MatchKind::LeftmostFirst`], for DFA table experiments (not
/// the regex-based [`crate::lex_c11_max_munch`]).
///
/// # Panics
///
/// Panics if any pattern in [`C11_PATTERNS`] fails to compile. All patterns are
/// compile-time constants; a failure here is a programmer error, not a runtime
/// condition. Silent recovery would produce a zero-recall DFA.
// INTENTIONAL: same programmer-error contract as build_c11_lexer_dfa.
#[must_use]
#[allow(clippy::expect_used)]
pub fn build_c11_lexer_dfa_for_host() -> DfaTable {
    let mut b = DfaBuilder::new(0, 0);
    add_c11_patterns(&mut b);
    b.build_with_match_kind(MatchKind::LeftmostFirst).expect(
        "Fix: C11 host DFA build failed. A pattern in C11_PATTERNS is invalid. \
         All patterns are compile-time constants, fix the broken pattern.",
    )
}
