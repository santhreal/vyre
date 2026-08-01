//! Shared C-token assembly and Program Graph row assertions for integration tests.

use vyre_libs::parsing::c::lex::keyword::reference_c_keyword_types;
use vyre_libs::parsing::c::lex::tokens::{TOK_COMMENT, TOK_WHITESPACE};

const VAST_STRIDE_U32: usize = 10;
const PG_STRIDE_U32: usize = 6;

/// Token rows assembled from explicit lexeme fixtures.
pub(crate) struct Assembled {
    pub(crate) source: String,
    #[allow(dead_code)]
    pub(crate) raw_kinds: Vec<u32>,
    pub(crate) tok_types: Vec<u32>,
    pub(crate) tok_starts: Vec<u32>,
    pub(crate) tok_lens: Vec<u32>,
}

/// Assemble non-whitespace fixture rows and classify C keywords.
pub(crate) fn assemble(lexemes: &[(&str, u32)]) -> Assembled {
    let mut source = String::new();
    let mut tok_starts = Vec::new();
    let mut tok_lens = Vec::new();
    let mut raw_kinds = Vec::new();

    for (lexeme, kind) in lexemes {
        if *kind == TOK_WHITESPACE || *kind == TOK_COMMENT {
            source.push_str(lexeme);
            continue;
        }
        if !source.is_empty() && !source.ends_with('\n') {
            source.push(' ');
        }
        tok_starts.push(source.len() as u32);
        source.push_str(lexeme);
        tok_lens.push(lexeme.len() as u32);
        raw_kinds.push(*kind);
    }

    let tok_types =
        reference_c_keyword_types(&raw_kinds, &tok_starts, &tok_lens, source.as_bytes());
    Assembled {
        source,
        raw_kinds,
        tok_types,
        tok_starts,
        tok_lens,
    }
}

/// Find the first token row whose source span equals `needle`.
pub(crate) fn find_row_for_lexeme(assembled: &Assembled, needle: &str) -> usize {
    assembled
        .tok_starts
        .iter()
        .zip(&assembled.tok_lens)
        .position(|(start, len)| {
            let start = *start as usize;
            let end = start.saturating_add(*len as usize);
            assembled.source.as_bytes().get(start..end) == Some(needle.as_bytes())
        })
        .unwrap_or_else(|| panic!("lexeme {needle:?} not found in fixture"))
}

/// Read the kind field from one typed VAST row.
pub(crate) fn row_typed_kind(typed: &[u8], row: usize) -> u32 {
    word_at(typed, row * VAST_STRIDE_U32)
}

/// Assert a typed VAST row and its lowered Program Graph source span.
pub(crate) fn assert_pg_row(assembled: &Assembled, pg: &[u8], typed: &[u8], idx: usize, kind: u32) {
    assert_eq!(
        word_at(typed, idx * VAST_STRIDE_U32),
        kind,
        "typed kind at {idx}"
    );
    assert_eq!(word_at(pg, idx * PG_STRIDE_U32), kind, "PG kind at {idx}");
    assert_eq!(
        word_at(pg, idx * PG_STRIDE_U32 + 1),
        assembled.tok_starts[idx],
        "PG span_start at {idx}"
    );
    assert_eq!(
        word_at(pg, idx * PG_STRIDE_U32 + 2),
        assembled.tok_starts[idx] + assembled.tok_lens[idx],
        "PG span_end at {idx}"
    );
}

fn word_at(buf: &[u8], word: usize) -> u32 {
    let offset = word * 4;
    u32::from_le_bytes(
        buf[offset..offset + 4]
            .try_into()
            .expect("complete u32 word"),
    )
}
