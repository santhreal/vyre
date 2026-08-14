//! Compact spellings for the C-frontend token fixtures.
//!
//! A fixture is a token stream, and a token stream written one
//! `FixtureToken::new(..)` or one `TOK_*` per line is 20 to 60 lines per case.
//! Two unrelated cases that share eight consecutive tokens then read as copies
//! of each other: `dup-scan` measured 555 duplicated lines inside
//! `fixtures/` alone, all of it incidental collisions between distinct
//! constructs spelled the same way.
//!
//! These three spellings write the same streams on one line per construct:
//!
//!   * [`c_rows`] for a raw `(tok_types, tok_starts, tok_lens)` triple, spelled
//!     as kind names with an optional `:len` (`"PREPROC:7 LPAREN INT:3"`).
//!   * [`c_tokens`] for a lexeme-driven [`Fixture`], spelled as the C source
//!     itself (`"int x = 1 ;"`); the kind follows from the lexeme, and
//!     `KIND@lexeme` overrides it where it cannot.
//!   * [`c_atoms`] for the packed-haystack [`Atom`] layout, where a keyword or
//!     punctuator lexeme is a bare token and any other word is an identifier;
//!     `#name` forces an identifier and `KIND@` a bare token of that kind.
//!
//! Every kind name is the `TOK_` constant without its prefix, so a spelling and
//! the token table cannot drift apart without a panic naming the bad word.

use vyre_libs::parsing::c::lex::keyword::C_KEYWORDS;
use vyre_libs::parsing::c::lex::tokens::*;

use super::rows::starts_for_lens;
use super::scope_fixture::{ident, tok, Atom};
use super::token_fixture::{build_fixture, Fixture, FixtureToken};

/// Every C token kind, keyed by its `TOK_` constant name without the prefix.
const KINDS: &[(&str, u32)] = &[
    ("EOF", TOK_EOF),
    ("IDENTIFIER", TOK_IDENTIFIER),
    ("INTEGER", TOK_INTEGER),
    ("FLOAT", TOK_FLOAT),
    ("STRING", TOK_STRING),
    ("CHAR", TOK_CHAR),
    ("LPAREN", TOK_LPAREN),
    ("RPAREN", TOK_RPAREN),
    ("LBRACE", TOK_LBRACE),
    ("RBRACE", TOK_RBRACE),
    ("LBRACKET", TOK_LBRACKET),
    ("RBRACKET", TOK_RBRACKET),
    ("SEMICOLON", TOK_SEMICOLON),
    ("COMMA", TOK_COMMA),
    ("DOT", TOK_DOT),
    ("ARROW", TOK_ARROW),
    ("PLUS", TOK_PLUS),
    ("MINUS", TOK_MINUS),
    ("STAR", TOK_STAR),
    ("SLASH", TOK_SLASH),
    ("PERCENT", TOK_PERCENT),
    ("AMP", TOK_AMP),
    ("PIPE", TOK_PIPE),
    ("CARET", TOK_CARET),
    ("TILDE", TOK_TILDE),
    ("BANG", TOK_BANG),
    ("ASSIGN", TOK_ASSIGN),
    ("LT", TOK_LT),
    ("GT", TOK_GT),
    ("HASH", TOK_HASH),
    ("QUESTION", TOK_QUESTION),
    ("COLON", TOK_COLON),
    ("EQ", TOK_EQ),
    ("NE", TOK_NE),
    ("LE", TOK_LE),
    ("GE", TOK_GE),
    ("AND", TOK_AND),
    ("OR", TOK_OR),
    ("LSHIFT", TOK_LSHIFT),
    ("RSHIFT", TOK_RSHIFT),
    ("INC", TOK_INC),
    ("DEC", TOK_DEC),
    ("PLUS_EQ", TOK_PLUS_EQ),
    ("MINUS_EQ", TOK_MINUS_EQ),
    ("STAR_EQ", TOK_STAR_EQ),
    ("SLASH_EQ", TOK_SLASH_EQ),
    ("ELLIPSIS", TOK_ELLIPSIS),
    ("PERCENT_EQ", TOK_PERCENT_EQ),
    ("AMP_EQ", TOK_AMP_EQ),
    ("PIPE_EQ", TOK_PIPE_EQ),
    ("CARET_EQ", TOK_CARET_EQ),
    ("LSHIFT_EQ", TOK_LSHIFT_EQ),
    ("RSHIFT_EQ", TOK_RSHIFT_EQ),
    ("HASHHASH", TOK_HASHHASH),
    ("IF", TOK_IF),
    ("ELSE", TOK_ELSE),
    ("FOR", TOK_FOR),
    ("WHILE", TOK_WHILE),
    ("RETURN", TOK_RETURN),
    ("STRUCT", TOK_STRUCT),
    ("TYPEDEF", TOK_TYPEDEF),
    ("INT", TOK_INT),
    ("CHAR_KW", TOK_CHAR_KW),
    ("VOID", TOK_VOID),
    ("DO", TOK_DO),
    ("SWITCH", TOK_SWITCH),
    ("CASE", TOK_CASE),
    ("DEFAULT", TOK_DEFAULT),
    ("BREAK", TOK_BREAK),
    ("CONTINUE", TOK_CONTINUE),
    ("GOTO", TOK_GOTO),
    ("SIZEOF", TOK_SIZEOF),
    ("AUTO", TOK_AUTO),
    ("CONST", TOK_CONST),
    ("DOUBLE", TOK_DOUBLE),
    ("ENUM", TOK_ENUM),
    ("EXTERN", TOK_EXTERN),
    ("FLOAT_KW", TOK_FLOAT_KW),
    ("INLINE", TOK_INLINE),
    ("LONG", TOK_LONG),
    ("REGISTER", TOK_REGISTER),
    ("RESTRICT", TOK_RESTRICT),
    ("SHORT", TOK_SHORT),
    ("SIGNED", TOK_SIGNED),
    ("STATIC", TOK_STATIC),
    ("UNION", TOK_UNION),
    ("UNSIGNED", TOK_UNSIGNED),
    ("VOLATILE", TOK_VOLATILE),
    ("ALIGNAS", TOK_ALIGNAS),
    ("ALIGNOF", TOK_ALIGNOF),
    ("ATOMIC", TOK_ATOMIC),
    ("BOOL", TOK_BOOL),
    ("COMPLEX", TOK_COMPLEX),
    ("GENERIC", TOK_GENERIC),
    ("IMAGINARY", TOK_IMAGINARY),
    ("NORETURN", TOK_NORETURN),
    ("STATIC_ASSERT", TOK_STATIC_ASSERT),
    ("THREAD_LOCAL", TOK_THREAD_LOCAL),
    ("GNU_ASM", TOK_GNU_ASM),
    ("GNU_ATTRIBUTE", TOK_GNU_ATTRIBUTE),
    ("GNU_TYPEOF", TOK_GNU_TYPEOF),
    ("GNU_EXTENSION", TOK_GNU_EXTENSION),
    ("GNU_REAL", TOK_GNU_REAL),
    ("GNU_IMAG", TOK_GNU_IMAG),
    ("BUILTIN_CONSTANT_P", TOK_BUILTIN_CONSTANT_P),
    ("BUILTIN_CHOOSE_EXPR", TOK_BUILTIN_CHOOSE_EXPR),
    ("BUILTIN_TYPES_COMPATIBLE_P", TOK_BUILTIN_TYPES_COMPATIBLE_P),
    ("GNU_AUTO_TYPE", TOK_GNU_AUTO_TYPE),
    ("GNU_TYPEOF_UNQUAL", TOK_GNU_TYPEOF_UNQUAL),
    ("GNU_INT128", TOK_GNU_INT128),
    ("GNU_BUILTIN_VA_LIST", TOK_GNU_BUILTIN_VA_LIST),
    ("GNU_ADDRESS_SPACE", TOK_GNU_ADDRESS_SPACE),
    ("GNU_LABEL", TOK_GNU_LABEL),
    ("BITINT_KW", TOK_BITINT_KW),
    ("FLOAT16_KW", TOK_FLOAT16_KW),
    ("FLOAT32_KW", TOK_FLOAT32_KW),
    ("FLOAT64_KW", TOK_FLOAT64_KW),
    ("FLOAT128_KW", TOK_FLOAT128_KW),
    ("GNU_FLOAT128_KW", TOK_GNU_FLOAT128_KW),
    ("GNU_BF16_KW", TOK_GNU_BF16_KW),
    ("GNU_FP16_KW", TOK_GNU_FP16_KW),
    ("DECIMAL32_KW", TOK_DECIMAL32_KW),
    ("DECIMAL64_KW", TOK_DECIMAL64_KW),
    ("DECIMAL128_KW", TOK_DECIMAL128_KW),
    ("FORCEINLINE_KW", TOK_FORCEINLINE_KW),
    ("NULLABILITY_KW", TOK_NULLABILITY_KW),
    ("COMMENT", TOK_COMMENT),
    ("WHITESPACE", TOK_WHITESPACE),
    ("PREPROC", TOK_PREPROC),
    ("ERR_UNTERMINATED_STRING", TOK_ERR_UNTERMINATED_STRING),
    ("ERR_UNTERMINATED_CHAR", TOK_ERR_UNTERMINATED_CHAR),
    ("ERR_UNTERMINATED_COMMENT", TOK_ERR_UNTERMINATED_COMMENT),
    ("ERR_INVALID_ESCAPE", TOK_ERR_INVALID_ESCAPE),
    ("PP_NULL", TOK_PP_NULL),
    ("PP_DEFINE", TOK_PP_DEFINE),
    ("PP_UNDEF", TOK_PP_UNDEF),
    ("PP_INCLUDE", TOK_PP_INCLUDE),
    ("PP_IF", TOK_PP_IF),
    ("PP_IFDEF", TOK_PP_IFDEF),
    ("PP_IFNDEF", TOK_PP_IFNDEF),
    ("PP_ELIF", TOK_PP_ELIF),
    ("PP_ELSE", TOK_PP_ELSE),
    ("PP_ENDIF", TOK_PP_ENDIF),
    ("PP_PRAGMA", TOK_PP_PRAGMA),
    ("PP_LINE", TOK_PP_LINE),
    ("PP_ERROR", TOK_PP_ERROR),
    ("PP_INCLUDE_NEXT", TOK_PP_INCLUDE_NEXT),
    ("PP_WARNING", TOK_PP_WARNING),
    ("PP_IDENT", TOK_PP_IDENT),
    ("PP_SCCS", TOK_PP_SCCS),
    ("PP_EFFECT_INCLUDE", TOK_PP_EFFECT_INCLUDE),
    ("PP_EFFECT_INCLUDE_NEXT", TOK_PP_EFFECT_INCLUDE_NEXT),
    ("PP_EFFECT_PRAGMA", TOK_PP_EFFECT_PRAGMA),
    ("PP_EFFECT_PRAGMA_ONCE", TOK_PP_EFFECT_PRAGMA_ONCE),
    (
        "PP_EFFECT_PRAGMA_DIAGNOSTIC_PUSH",
        TOK_PP_EFFECT_PRAGMA_DIAGNOSTIC_PUSH,
    ),
    (
        "PP_EFFECT_PRAGMA_DIAGNOSTIC_POP",
        TOK_PP_EFFECT_PRAGMA_DIAGNOSTIC_POP,
    ),
    (
        "PP_EFFECT_PRAGMA_DIAGNOSTIC_IGNORED",
        TOK_PP_EFFECT_PRAGMA_DIAGNOSTIC_IGNORED,
    ),
    (
        "PP_EFFECT_PRAGMA_DIAGNOSTIC_WARNING",
        TOK_PP_EFFECT_PRAGMA_DIAGNOSTIC_WARNING,
    ),
    (
        "PP_EFFECT_PRAGMA_DIAGNOSTIC_ERROR",
        TOK_PP_EFFECT_PRAGMA_DIAGNOSTIC_ERROR,
    ),
    ("PP_EFFECT_ERROR_DIAGNOSTIC", TOK_PP_EFFECT_ERROR_DIAGNOSTIC),
    (
        "PP_EFFECT_WARNING_DIAGNOSTIC",
        TOK_PP_EFFECT_WARNING_DIAGNOSTIC,
    ),
    ("PP_EFFECT_IDENT", TOK_PP_EFFECT_IDENT),
    ("PP_EFFECT_SCCS", TOK_PP_EFFECT_SCCS),
    ("PP_EFFECT_LINE", TOK_PP_EFFECT_LINE),
    ("PP_EMBED", TOK_PP_EMBED),
    ("PP_ELIFDEF", TOK_PP_ELIFDEF),
    ("PP_ELIFNDEF", TOK_PP_ELIFNDEF),
    ("PP_IMPORT", TOK_PP_IMPORT),
];

/// Punctuator lexemes, for the spellings that infer a kind from the lexeme.
const PUNCTUATORS: &[(&str, u32)] = &[
    ("(", TOK_LPAREN),
    (")", TOK_RPAREN),
    ("{", TOK_LBRACE),
    ("}", TOK_RBRACE),
    ("[", TOK_LBRACKET),
    ("]", TOK_RBRACKET),
    (";", TOK_SEMICOLON),
    (",", TOK_COMMA),
    (".", TOK_DOT),
    ("->", TOK_ARROW),
    ("+", TOK_PLUS),
    ("-", TOK_MINUS),
    ("*", TOK_STAR),
    ("/", TOK_SLASH),
    ("%", TOK_PERCENT),
    ("&", TOK_AMP),
    ("|", TOK_PIPE),
    ("^", TOK_CARET),
    ("~", TOK_TILDE),
    ("!", TOK_BANG),
    ("=", TOK_ASSIGN),
    ("<", TOK_LT),
    (">", TOK_GT),
    ("#", TOK_HASH),
    ("?", TOK_QUESTION),
    (":", TOK_COLON),
    ("==", TOK_EQ),
    ("!=", TOK_NE),
    ("<=", TOK_LE),
    (">=", TOK_GE),
    ("&&", TOK_AND),
    ("||", TOK_OR),
    ("<<", TOK_LSHIFT),
    (">>", TOK_RSHIFT),
    ("++", TOK_INC),
    ("--", TOK_DEC),
    ("+=", TOK_PLUS_EQ),
    ("-=", TOK_MINUS_EQ),
    ("*=", TOK_STAR_EQ),
    ("/=", TOK_SLASH_EQ),
    ("...", TOK_ELLIPSIS),
    ("%=", TOK_PERCENT_EQ),
    ("&=", TOK_AMP_EQ),
    ("|=", TOK_PIPE_EQ),
    ("^=", TOK_CARET_EQ),
    ("<<=", TOK_LSHIFT_EQ),
    (">>=", TOK_RSHIFT_EQ),
    ("##", TOK_HASHHASH),
];

/// The kind a spelling names, or a panic naming the word that is not a kind.
#[must_use]
pub(crate) fn kind_of(name: &str) -> u32 {
    KINDS
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, kind)| *kind)
        .unwrap_or_else(|| {
            panic!(
                "C fixture spelling names {name:?}, which is not a C token kind. \
                 Fix: use a TOK_ constant name without the prefix, such as \"LPAREN\"."
            )
        })
}

/// The raw kind of a lexeme before keyword promotion.
///
/// Keywords deliberately stay `TOK_IDENTIFIER`: [`build_fixture`] runs the same
/// promotion the lexer runs, so `"int"` becomes `TOK_INT` there and a fixture
/// cannot disagree with the keyword table.
#[must_use]
pub(crate) fn raw_kind_of_lexeme(lexeme: &str) -> u32 {
    if let Some((_, kind)) = PUNCTUATORS
        .iter()
        .find(|(candidate, _)| *candidate == lexeme)
    {
        return *kind;
    }
    let first = lexeme.as_bytes()[0];
    match first {
        b'"' => TOK_STRING,
        b'\'' => TOK_CHAR,
        b'0'..=b'9' => {
            if lexeme.contains('.') {
                TOK_FLOAT
            } else {
                TOK_INTEGER
            }
        }
        _ => TOK_IDENTIFIER,
    }
}

/// Raw token rows from a kind-name spelling: `"PREPROC:7 GNU_ATTRIBUTE:13 LPAREN"`.
///
/// A bare name is one source byte wide; `NAME:len` gives the span width the
/// case needs. Starts follow from the widths, so a span assertion reads the
/// same layout it did when the widths were a second parallel `vec!`.
#[must_use]
pub(crate) fn c_rows(spelling: &str) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut tok_types = Vec::new();
    let mut tok_lens = Vec::new();
    for word in spelling.split_whitespace() {
        let (name, len) = match word.split_once(':') {
            Some((name, len)) => (
                name,
                len.parse::<u32>().unwrap_or_else(|_| {
                    panic!(
                        "C fixture spelling {word:?} has a token width that is not a number. \
                         Fix: write NAME:LEN with a decimal length."
                    )
                }),
            ),
            None => (word, 1),
        };
        tok_types.push(kind_of(name));
        tok_lens.push(len);
    }
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// Token kinds only, from the same kind-name spelling as [`c_rows`].
#[must_use]
pub(crate) fn c_kinds(spelling: &str) -> Vec<u32> {
    c_rows(spelling).0
}

/// A lexeme-driven fixture from C source text: `c_tokens("int x = 1 ;")`.
///
/// Tokens are whitespace separated because that is exactly how
/// [`build_fixture`] joins them into the fixture source, so the spelling and
/// the source it produces are the same bytes. `KIND@lexeme` spells a token
/// whose kind the lexeme cannot imply, such as `PREPROC@#define`.
#[must_use]
pub(crate) fn c_tokens(spelling: &'static str) -> Fixture {
    let tokens: Vec<FixtureToken> = spelling
        .split_whitespace()
        .map(|word| match word.split_once('@') {
            Some((name, lexeme)) => FixtureToken::new(lexeme, kind_of(name)),
            None => FixtureToken::new(word, raw_kind_of_lexeme(word)),
        })
        .collect();
    build_fixture(&tokens)
}

/// Scope atoms from a lexeme spelling: `c_atoms("typedef int T ; #T x ;")`.
///
/// A keyword or punctuator lexeme is a bare token, which is what the packed
/// haystack layout needs: only identifiers contribute bytes. `#name` forces an
/// identifier for a word the keyword table also claims, and `KIND@` a bare
/// token of that kind for one no lexeme names.
#[must_use]
pub(crate) fn c_atoms(spelling: &str) -> Vec<Atom> {
    spelling
        .split_whitespace()
        .map(|word| {
            if let Some(name) = word.strip_prefix('#') {
                return ident(name);
            }
            if let Some(name) = word.strip_suffix('@') {
                return tok(kind_of(name));
            }
            if let Some((_, kind)) = PUNCTUATORS.iter().find(|(candidate, _)| *candidate == word) {
                return tok(*kind);
            }
            if let Some((_, kind)) = C_KEYWORDS.iter().find(|(candidate, _)| *candidate == word) {
                return tok(*kind);
            }
            ident(word)
        })
        .collect()
}
