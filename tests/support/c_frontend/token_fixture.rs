//! Lexeme-driven token fixtures for the C frontend.
//!
//! A fixture is a list of `(lexeme, raw token kind)` pairs. Building it here
//! runs the same keyword promotion the real lexer runs, so a test that writes
//! `("int", TOK_IDENTIFIER)` still gets `TOK_INT`, and every test file agrees on
//! how source offsets are laid out.

use vyre_libs::parsing::c::lex::keyword::reference_c_keyword_types;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds,
};

#[derive(Clone, Copy)]
pub(crate) struct FixtureToken {
    pub(crate) lexeme: &'static str,
    pub(crate) raw_kind: u32,
}

impl FixtureToken {
    pub(crate) const fn new(lexeme: &'static str, raw_kind: u32) -> Self {
        Self { lexeme, raw_kind }
    }
}

pub(crate) struct Fixture {
    pub(crate) source: String,
    pub(crate) raw_kinds: Vec<u32>,
    pub(crate) tok_types: Vec<u32>,
    pub(crate) tok_starts: Vec<u32>,
    pub(crate) tok_lens: Vec<u32>,
}

pub(crate) fn build_fixture(tokens: &[FixtureToken]) -> Fixture {
    let mut source = String::new();
    let mut raw_kinds = Vec::with_capacity(tokens.len());
    let mut tok_starts = Vec::with_capacity(tokens.len());
    let mut tok_lens = Vec::with_capacity(tokens.len());

    for token in tokens {
        if !source.is_empty() && !source.ends_with('\n') {
            source.push(' ');
        }
        tok_starts.push(source.len() as u32);
        source.push_str(token.lexeme);
        tok_lens.push(token.lexeme.len() as u32);
        raw_kinds.push(token.raw_kind);
    }

    let promoted = reference_c_keyword_types(&raw_kinds, &tok_starts, &tok_lens, source.as_bytes());

    Fixture {
        source,
        raw_kinds: promoted.clone(),
        tok_types: promoted,
        tok_starts,
        tok_lens,
    }
}

/// Typed VAST for a fixture: build, annotate typedef names, classify.
pub(crate) fn classify(fix: &Fixture) -> Vec<u8> {
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    reference_c11_classify_vast_node_kinds(&annotated)
}

/// `c_fixture![("int", TOK_IDENTIFIER), ("x", TOK_IDENTIFIER)]`.
///
/// The expansion is rooted at `$crate::c_frontend`, the name every consumer
/// gives this module, so the macro works from either crate without the caller
/// importing the fixture builder.
#[allow(unused_macros)]
macro_rules! c_fixture {
    ($(($lexeme:literal, $kind:expr $(,)?)),+ $(,)?) => {
        $crate::c_frontend::token_fixture::build_fixture(&[
            $(
                $crate::c_frontend::token_fixture::FixtureToken::new($lexeme, $kind),
            )+
        ])
    };
}

#[allow(unused_imports)]
pub(crate) use c_fixture;
