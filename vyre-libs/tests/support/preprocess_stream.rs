//! The directive token stream the C-preprocessor kernel roundtrips feed.
//!
//! Every `gpu_*` preprocessor kernel is checked against
//! `reference_c_preprocessor_directive_metadata` over the same stream shape:
//! one `TOK_PREPROC` token per directive row, one sentinel `0` token per other
//! byte. Six suites each carried their own copy of that walk and the copies had
//! drifted apart, so it has one owner here.

use vyre_libs::parsing::c::lex::tokens::TOK_PREPROC;

/// What the builder does with a byte that ends a line.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LineEnds {
    /// `\n` and `\r` are consumed without producing a token.
    Skipped,
    /// Line terminators produce a sentinel token like any other non-directive
    /// byte, so a token index outside a directive row is its source offset.
    Tokenized,
}

pub(crate) fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The stream with line terminators skipped.
pub(crate) fn build_token_stream(source: &[u8]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    build_token_stream_with(source, LineEnds::Skipped)
}

/// A directive row runs from the start of its line to the nearest line
/// terminator that a backslash does not splice, and its leading run of
/// horizontal whitespace is C's: space, tab, vertical tab and form feed.
pub(crate) fn build_token_stream_with(
    source: &[u8],
    line_ends: LineEnds,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut tok_types = Vec::new();
    let mut tok_starts = Vec::new();
    let mut tok_lens = Vec::new();
    let mut i = 0usize;
    let mut at_line_start = true;
    while i < source.len() {
        if at_line_start {
            let mut j = i;
            while j < source.len() && matches!(source[j], b' ' | b'\t' | 0x0B | 0x0C) {
                j += 1;
            }
            if j < source.len() && source[j] == b'#' {
                let row_start = i;
                let row_end = directive_row_end(source, row_start, j);
                tok_types.push(TOK_PREPROC);
                tok_starts.push(row_start as u32);
                tok_lens.push((row_end - row_start) as u32);
                i = row_end;
                at_line_start = true;
                continue;
            }
        }
        let ends_line = matches!(source[i], b'\n' | b'\r');
        if ends_line && line_ends == LineEnds::Skipped {
            at_line_start = true;
            i += 1;
            continue;
        }
        at_line_start = ends_line;
        tok_types.push(0);
        tok_starts.push(i as u32);
        tok_lens.push(1);
        i += 1;
    }
    (tok_types, tok_starts, tok_lens)
}

/// Walk from `scan_from` to the first line terminator that is not spliced by a
/// preceding backslash, consuming a `\r\n` pair as one terminator.
fn directive_row_end(source: &[u8], row_start: usize, scan_from: usize) -> usize {
    let mut row_end = scan_from;
    while row_end < source.len() {
        let spliced = row_end > row_start && source[row_end - 1] == b'\\';
        match source[row_end] {
            b'\n' if spliced => row_end += 1,
            b'\n' => return row_end,
            b'\r' if spliced => {
                row_end += 1;
                if row_end < source.len() && source[row_end] == b'\n' {
                    row_end += 1;
                }
            }
            b'\r' => return row_end,
            _ => row_end += 1,
        }
    }
    row_end
}

/// Pack macro names into the `(names_packed, offsets)` layout the kernels take.
pub(crate) fn pack_defined_macros(names: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
    let mut packed = Vec::new();
    let mut offsets = Vec::with_capacity(names.len() + 1);
    offsets.push(0u32);
    for name in names {
        packed.extend_from_slice(name);
        offsets.push(packed.len() as u32);
    }
    (packed, offsets)
}
