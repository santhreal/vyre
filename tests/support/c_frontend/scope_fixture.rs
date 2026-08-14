//! Atom-driven token fixtures and CPU reference passes for C scope semantics.
//!
//! Scope, typedef-visibility, and namespace contracts need a token stream whose
//! identifier bytes are packed into a haystack with no separators, which is a
//! different layout from [`super::token_fixture`]'s lexeme-joined source. This
//! module owns that layout and the four reference passes every scope test runs
//! over it: scope tree, raw VAST, typedef annotation, and classification.

use vyre_libs::parsing::c::lex::tokens::TOK_IDENTIFIER;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds,
};
use vyre_libs::parsing::c::sema::reference_scope_tree;

use super::rows::{bytes, word_at};

/// The VAST row fields a scope test reads, owned by [`super::rows`]. Re-exported
/// so a scope test's `use scope_fixture::*` still names them; a given test only
/// reads some of them.
#[allow(unused_imports)]
pub(crate) use super::rows::{
    flags_at, kind_at, FLAGS_FIELD, ORDINARY_FLAG_DECL, TYPEDEF_FLAG_DECL, TYPEDEF_FLAG_VISIBLE,
};

/// `u32` fields per scope-tree row.
pub(crate) const SCOPE_TREE_STRIDE_U32: usize = 4;

/// One token in a scope fixture: a bare kind, or an identifier with a lexeme.
#[derive(Clone)]
pub(crate) enum Atom {
    Tok(u32),
    Ident(&'static str),
}

/// A token stream whose identifier bytes are packed back to back.
pub(crate) struct ScopeFixture {
    pub(crate) tok_types: Vec<u32>,
    pub(crate) tok_starts: Vec<u32>,
    pub(crate) tok_lens: Vec<u32>,
    pub(crate) haystack: Vec<u8>,
}

pub(crate) fn tok(t: u32) -> Atom {
    Atom::Tok(t)
}

pub(crate) fn ident(name: &'static str) -> Atom {
    Atom::Ident(name)
}

/// Build a scope fixture. `_name` labels the fixture at the call site.
pub(crate) fn fixture(_name: &'static str, atoms: &[Atom]) -> ScopeFixture {
    let mut tok_types = Vec::with_capacity(atoms.len());
    let mut tok_starts = Vec::with_capacity(atoms.len());
    let mut tok_lens = Vec::with_capacity(atoms.len());
    let mut haystack = Vec::new();
    let mut cursor = 0u32;
    for atom in atoms {
        match atom {
            Atom::Tok(t) => {
                tok_types.push(*t);
                tok_starts.push(0);
                tok_lens.push(0);
            }
            Atom::Ident(name) => {
                tok_types.push(TOK_IDENTIFIER);
                tok_starts.push(cursor);
                tok_lens.push(name.len() as u32);
                haystack.extend_from_slice(name.as_bytes());
                cursor += name.len() as u32;
            }
        }
    }
    ScopeFixture {
        tok_types,
        tok_starts,
        tok_lens,
        haystack,
    }
}

pub(crate) fn emit_u32_bytes(words: &[u32]) -> Vec<u8> {
    bytes(words)
}

pub(crate) fn scope_tree_word_at(buf: &[u8], token_idx: usize, field: usize) -> u32 {
    word_at(buf, token_idx * SCOPE_TREE_STRIDE_U32 + field)
}

pub(crate) fn scope_tree_for(fix: &ScopeFixture) -> Vec<u8> {
    let haystack_u32: Vec<u32> = fix.haystack.iter().copied().map(u32::from).collect();
    let words = reference_scope_tree(
        &fix.tok_types,
        &fix.tok_starts,
        &fix.tok_lens,
        &haystack_u32,
    );
    emit_u32_bytes(&words)
}

pub(crate) fn raw_vast(fix: &ScopeFixture) -> Vec<u8> {
    reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens)
}

pub(crate) fn annotate_cpu(fix: &ScopeFixture) -> Vec<u8> {
    reference_c11_annotate_typedef_names(&raw_vast(fix), &fix.haystack)
}

pub(crate) fn classify_cpu_annotated(fix: &ScopeFixture) -> Vec<u8> {
    reference_c11_classify_vast_node_kinds(&annotate_cpu(fix))
}

pub(crate) fn pg_lower_cpu(vast: &[u8]) -> Vec<u8> {
    reference_ast_to_pg_nodes(vast)
}
