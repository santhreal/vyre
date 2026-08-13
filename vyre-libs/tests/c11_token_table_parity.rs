//! `vyre-grammar-gen` keeps a hand-maintained copy of the C11 token ids that
//! `vyre_libs::parsing::c::lex::tokens` owns, and the copy cannot be collapsed:
//! `vyre-grammar-gen` is a leaf generator with no vyre dependencies, and
//! `vyre-libs` dev-depends on it as the host lexer oracle, so making it consume
//! the vyre-libs table would invert the layering and close a dependency cycle
//! over the whole compiler stack. See the `tokens` module doc.
//!
//! What is enforceable is that the copy cannot drift. Every id that
//! `C11_PATTERNS` can emit is checked against the vyre-libs constant of the
//! same name, and the covered-id set is derived from `C11_PATTERNS` at run
//! time, so a new generator pattern carrying an unrecorded id turns this suite
//! red instead of silently disagreeing with the GPU lexer that the oracle
//! tests diff against.

#![cfg(feature = "c-parser")]

use std::collections::BTreeSet;

/// Pair each token id the generator declares with the vyre-libs constant that
/// owns it. `stringify!` keeps the two names identical by construction: a
/// renamed constant on either side stops compiling.
macro_rules! shared_token_ids {
    ($($name:ident),* $(,)?) => {
        const SHARED: &[(&str, u32, u32)] = &[
            $((
                stringify!($name),
                c_grammar_gen::c11_lexer::$name,
                vyre_libs::parsing::c::lex::tokens::$name,
            )),*
        ];
    };
}

shared_token_ids![
    TOK_IDENTIFIER,
    TOK_INTEGER,
    TOK_STRING,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_SEMICOLON,
    TOK_COMMA,
    TOK_DOT,
    TOK_ARROW,
    TOK_PLUS,
    TOK_MINUS,
    TOK_STAR,
    TOK_SLASH,
    TOK_PERCENT,
    TOK_AMP,
    TOK_PIPE,
    TOK_CARET,
    TOK_TILDE,
    TOK_BANG,
    TOK_ASSIGN,
    TOK_LT,
    TOK_GT,
    TOK_HASH,
    TOK_QUESTION,
    TOK_COLON,
    TOK_EQ,
    TOK_NE,
    TOK_LE,
    TOK_GE,
    TOK_AND,
    TOK_OR,
    TOK_LSHIFT,
    TOK_RSHIFT,
    TOK_INC,
    TOK_DEC,
    TOK_PLUS_EQ,
    TOK_MINUS_EQ,
    TOK_STAR_EQ,
    TOK_SLASH_EQ,
    TOK_ELLIPSIS,
    TOK_PERCENT_EQ,
    TOK_AMP_EQ,
    TOK_PIPE_EQ,
    TOK_CARET_EQ,
    TOK_LSHIFT_EQ,
    TOK_RSHIFT_EQ,
    TOK_HASHHASH,
    TOK_IF,
    TOK_ELSE,
    TOK_FOR,
    TOK_WHILE,
    TOK_RETURN,
    TOK_STRUCT,
    TOK_TYPEDEF,
    TOK_INT,
    TOK_CHAR_KW,
    TOK_VOID,
    TOK_DO,
    TOK_SWITCH,
    TOK_CASE,
    TOK_DEFAULT,
    TOK_BREAK,
    TOK_CONTINUE,
    TOK_GOTO,
    TOK_SIZEOF,
    TOK_AUTO,
    TOK_CONST,
    TOK_DOUBLE,
    TOK_ENUM,
    TOK_EXTERN,
    TOK_FLOAT_KW,
    TOK_INLINE,
    TOK_LONG,
    TOK_REGISTER,
    TOK_RESTRICT,
    TOK_SHORT,
    TOK_SIGNED,
    TOK_STATIC,
    TOK_UNION,
    TOK_UNSIGNED,
    TOK_VOLATILE,
    TOK_ALIGNAS,
    TOK_ALIGNOF,
    TOK_ATOMIC,
    TOK_BOOL,
    TOK_COMPLEX,
    TOK_GENERIC,
    TOK_IMAGINARY,
    TOK_NORETURN,
    TOK_STATIC_ASSERT,
    TOK_THREAD_LOCAL,
    TOK_GNU_ASM,
    TOK_GNU_ATTRIBUTE,
    TOK_GNU_TYPEOF,
    TOK_GNU_EXTENSION,
    TOK_GNU_REAL,
    TOK_GNU_IMAG,
    TOK_BUILTIN_CONSTANT_P,
    TOK_BUILTIN_CHOOSE_EXPR,
    TOK_BUILTIN_TYPES_COMPATIBLE_P,
    TOK_COMMENT,
    TOK_WHITESPACE,
    TOK_PREPROC,
];

#[test]
fn generator_token_ids_match_the_owning_vyre_libs_table() {
    let drifted: Vec<&(&str, u32, u32)> = SHARED
        .iter()
        .filter(|(_, generator, owner)| generator != owner)
        .collect();
    assert!(
        drifted.is_empty(),
        "the generator's token id copy drifted from the owning table: {drifted:?}"
    );
}

#[test]
fn every_emittable_generator_token_id_is_pinned() {
    let emittable: BTreeSet<u32> = c_grammar_gen::C11_PATTERNS
        .iter()
        .map(|&(id, _)| id)
        .collect();
    let pinned: BTreeSet<u32> = SHARED.iter().map(|&(_, generator, _)| generator).collect();
    let unpinned: Vec<u32> = emittable.difference(&pinned).copied().collect();
    assert!(
        unpinned.is_empty(),
        "C11_PATTERNS emits token ids with no counterpart pinned against the owning table: \
         {unpinned:?}; add the constant to shared_token_ids! or to the owning table"
    );
}
