//! Row accessors and row-shape assertions for the C frontend's packed buffers.
//!
//! VAST rows, property-graph rows, and semantic-graph rows are all `u32`
//! records in a byte buffer, so every C frontend test needs the same handful of
//! field readers and the same "PG row preserves the VAST row" assertion. One
//! owner here keeps the field offsets from drifting per test file.

use super::token_fixture::Fixture;

/// `u32` fields per VAST row.
pub(crate) const VAST_STRIDE_U32: usize = 10;
/// `u32` fields per property-graph row.
pub(crate) const PG_STRIDE_U32: usize = 6;
/// Bytes per VAST row.
pub(crate) const VAST_STRIDE_BYTES: usize = VAST_STRIDE_U32 * core::mem::size_of::<u32>();
/// Flags field index within a VAST row.
pub(crate) const FLAGS_FIELD: usize = 7;
/// The name is a typedef visible at this point.
pub(crate) const TYPEDEF_FLAG_VISIBLE: u32 = 1;
/// The row declares a typedef name.
pub(crate) const TYPEDEF_FLAG_DECL: u32 = 1 << 1;
/// The row declares an ordinary (non-typedef) identifier.
pub(crate) const ORDINARY_FLAG_DECL: u32 = 1 << 2;
/// "No such row" marker used by every C frontend row buffer.
pub(crate) const SENTINEL: u32 = u32::MAX;

pub(crate) fn bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

pub(crate) fn haystack_words(source: &[u8]) -> Vec<u8> {
    vyre_primitives::wire::pack_bytes_as_u32_slice(source)
}

pub(crate) fn word_at(buf: &[u8], word: usize) -> u32 {
    let offset = word * 4;
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

pub(crate) fn kind_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32)
}

pub(crate) fn flags_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + FLAGS_FIELD)
}

/// Assert the VAST row at `idx` classified as `kind`.
///
/// [`super::expression_pipeline::assert_kind`] is the same check against a
/// caller-supplied stride, for the expression-shape and property-graph rows.
pub(crate) fn assert_kind(rows: &[u8], idx: usize, kind: u32) {
    assert_eq!(word_at(rows, idx * VAST_STRIDE_U32), kind, "kind[{idx}]");
}

pub(crate) fn pg_word_at(buf: &[u8], idx: usize, field: usize) -> u32 {
    word_at(buf, idx * PG_STRIDE_U32 + field)
}

pub(crate) fn node_count_from_vast(buf: &[u8]) -> u32 {
    u32::try_from(buf.len() / VAST_STRIDE_BYTES).unwrap_or_default()
}

pub(crate) fn row_indices(rows: &[u8], kind: u32) -> Vec<usize> {
    row_indices_by_stride(rows, VAST_STRIDE_U32, kind)
}

pub(crate) fn row_indices_by_stride(rows: &[u8], stride_words: usize, kind: u32) -> Vec<usize> {
    rows.chunks_exact(stride_words * core::mem::size_of::<u32>())
        .enumerate()
        .filter_map(|(idx, row)| {
            let row_kind = u32::from_le_bytes(row[0..4].try_into().unwrap());
            (row_kind == kind).then_some(idx)
        })
        .collect()
}

pub(crate) fn lexeme_indices(fix: &Fixture, lexeme: &str) -> Vec<usize> {
    fix.tok_starts
        .iter()
        .zip(&fix.tok_lens)
        .enumerate()
        .filter_map(|(idx, (start, len))| {
            let start = *start as usize;
            let end = start.saturating_add(*len as usize);
            (fix.source.as_bytes().get(start..end) == Some(lexeme.as_bytes())).then_some(idx)
        })
        .collect()
}

pub(crate) fn token_indices_containing(fix: &Fixture, needle: &str) -> Vec<usize> {
    fix.tok_starts
        .iter()
        .zip(&fix.tok_lens)
        .enumerate()
        .filter_map(|(idx, (start, len))| {
            let start = *start as usize;
            let end = start.saturating_add(*len as usize);
            let token = fix.source.as_bytes().get(start..end)?;
            token
                .windows(needle.len())
                .any(|window| window == needle.as_bytes())
                .then_some(idx)
        })
        .collect()
}

/// Token starts for unit-separated lexemes of the given lengths.
pub(crate) fn starts_for_lens(lens: &[u32]) -> Vec<u32> {
    let mut cursor = 0u32;
    lens.iter()
        .map(|len| {
            let start = cursor;
            cursor = cursor.saturating_add(*len).saturating_add(1);
            start
        })
        .collect()
}

/// Assert the PG row at `idx` reproduces the VAST row's kind, span, and links.
pub(crate) fn assert_pg_preserves_row(
    typed_vast: &[u8],
    pg: &[u8],
    tok_starts: &[u32],
    tok_lens: &[u32],
    idx: usize,
    expected_kind: u32,
) {
    assert_eq!(
        pg_word_at(pg, idx, 0),
        expected_kind,
        "PG kind mismatch at row {idx}"
    );
    assert_eq!(
        pg_word_at(pg, idx, 1),
        tok_starts[idx],
        "PG span_start mismatch at row {idx}"
    );
    assert_eq!(
        pg_word_at(pg, idx, 2),
        tok_starts[idx] + tok_lens[idx],
        "PG span_end mismatch at row {idx}"
    );
    assert_eq!(
        pg_word_at(pg, idx, 3),
        word_at(typed_vast, idx * VAST_STRIDE_U32 + 1),
        "PG parent mismatch at row {idx}"
    );
    assert_eq!(
        pg_word_at(pg, idx, 4),
        word_at(typed_vast, idx * VAST_STRIDE_U32 + 2),
        "PG first_child mismatch at row {idx}"
    );
    assert_eq!(
        pg_word_at(pg, idx, 5),
        word_at(typed_vast, idx * VAST_STRIDE_U32 + 3),
        "PG next_sibling mismatch at row {idx}"
    );
}

/// [`assert_pg_preserves_row`] against the spans a [`Fixture`] already carries.
pub(crate) fn assert_pg_preserves_fixture_row(
    typed_vast: &[u8],
    pg: &[u8],
    fix: &Fixture,
    idx: usize,
    expected_kind: u32,
) {
    assert_pg_preserves_row(
        typed_vast,
        pg,
        &fix.tok_starts,
        &fix.tok_lens,
        idx,
        expected_kind,
    );
}

/// Compare two packed `u32` buffers, reporting the first differing row.
pub(crate) fn assert_words_eq(actual: &[u8], expected: &[u8], context: &str) {
    if actual == expected {
        return;
    }
    let limit = (actual.len() / 4).min(expected.len() / 4);
    for w in 0..limit {
        let a = word_at(actual, w);
        let e = word_at(expected, w);
        if a != e {
            let row = w / VAST_STRIDE_U32;
            let actual_row: Vec<u32> = (0..VAST_STRIDE_U32)
                .map(|field| word_at(actual, row * VAST_STRIDE_U32 + field))
                .collect();
            let expected_row: Vec<u32> = (0..VAST_STRIDE_U32)
                .map(|field| word_at(expected, row * VAST_STRIDE_U32 + field))
                .collect();
            let nearby_start = row.saturating_sub(3);
            let nearby_end = (row + 4).min(limit / VAST_STRIDE_U32);
            let nearby_actual: Vec<Vec<u32>> = (nearby_start..nearby_end)
                .map(|nearby_row| {
                    (0..VAST_STRIDE_U32)
                        .map(|field| word_at(actual, nearby_row * VAST_STRIDE_U32 + field))
                        .collect()
                })
                .collect();
            panic!(
                "{context}: word {w} differs (row={row}, field={}): actual={a}, expected={e}; actual_row={actual_row:?}; expected_row={expected_row:?}; nearby_actual_start={nearby_start}; nearby_actual={nearby_actual:?}",
                w % VAST_STRIDE_U32
            );
        }
    }
    panic!(
        "{context}: byte lengths differ: actual={}, expected={}",
        actual.len(),
        expected.len()
    );
}
