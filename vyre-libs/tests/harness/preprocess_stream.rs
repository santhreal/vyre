//! The directive token stream the C-preprocessor kernel roundtrips feed.
//!
//! Every `gpu_*` preprocessor kernel is checked against
//! `reference_c_preprocessor_directive_metadata` over the same stream shape:
//! one `TOK_PREPROC` token per directive row, one sentinel `0` token per other
//! byte. Six suites each carried their own copy of that walk and the copies had
//! drifted apart, so it has one owner here.

use vyre_libs::parsing::c::lex::tokens::TOK_PREPROC;
use vyre_libs::parsing::c::preprocess::gpu_directive_metadata::gpu_directive_metadata;
use vyre_libs::parsing::c::preprocess::reference_c_preprocessor_directive_metadata;
use vyre_primitives::wire::pack_u32_slice;
use vyre_reference::value::Value;

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

/// The stage-1 result every GPU preprocessor roundtrip feeds into its own
/// stage-2 kernel.
///
/// Each kernel declares the source and the token columns as packed U32 words,
/// so the byte buffers are padded to a whole number of words and the row count
/// is padded to at least one row.
pub(crate) struct DirectiveStage {
    /// Rows the kernel produced, before padding.
    pub(crate) n: usize,
    /// Rows the buffers are padded to.
    pub(crate) n_padded: usize,
    pub(crate) tok_starts_bytes: Vec<u8>,
    pub(crate) tok_lens_bytes: Vec<u8>,
    pub(crate) source_bytes: Vec<u8>,
    pub(crate) directive_kinds_bytes: Vec<u8>,
}

impl DirectiveStage {
    /// A zeroed output column, one word per padded row.
    pub(crate) fn zero_column(&self) -> Vec<u8> {
        vec![0u8; self.n_padded * 4]
    }
}

/// Runs `gpu_directive_metadata` over `source` and returns its padded buffers.
pub(crate) fn run_directive_metadata_stage(source: &[u8]) -> DirectiveStage {
    let (tok_types, tok_starts, tok_lens) = build_token_stream(source);
    run_directive_metadata_stage_from_parts(source, &tok_types, &tok_starts, &tok_lens)
}

/// The same stage over a token stream the caller built itself.
pub(crate) fn run_directive_metadata_stage_from_parts(
    source: &[u8],
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> DirectiveStage {
    let n = tok_types.len();
    let n_padded = n.max(1);
    let source_padded = (source.len().div_ceil(4) * 4).max(4);

    let mut tok_types_bytes = pack_u32_slice(tok_types);
    tok_types_bytes.resize(n_padded * 4, 0);
    let mut tok_starts_bytes = pack_u32_slice(tok_starts);
    tok_starts_bytes.resize(n_padded * 4, 0);
    let mut tok_lens_bytes = pack_u32_slice(tok_lens);
    tok_lens_bytes.resize(n_padded * 4, 0);
    let mut source_bytes = source.to_vec();
    source_bytes.resize(source_padded, 0);

    let program = gpu_directive_metadata(n as u32, source.len() as u32);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(tok_types_bytes),
            Value::from(tok_starts_bytes.clone()),
            Value::from(tok_lens_bytes.clone()),
            Value::from(source_bytes.clone()),
            Value::from(vec![0u8; n_padded * 4]),
            Value::from(vec![0u8; n_padded * 4]),
        ],
    )
    .expect("17a kernel eval");
    let mut directive_kinds_bytes = outputs[0].to_bytes().to_vec();
    directive_kinds_bytes.resize(n_padded * 4, 0);

    DirectiveStage {
        n,
        n_padded,
        tok_starts_bytes,
        tok_lens_bytes,
        source_bytes,
        directive_kinds_bytes,
    }
}

/// `(names_packed_padded, offsets_bytes)` for the defined-macro kernel inputs.
pub(crate) fn padded_defined_macros(names: &[&[u8]]) -> (Vec<u8>, Vec<u8>) {
    let (mut names_padded, offsets) = pack_defined_macros(names);
    let pad_len = (names_padded.len().div_ceil(4) * 4).max(4);
    names_padded.resize(pad_len, 0);
    (names_padded, pack_u32_slice(&offsets))
}

/// An output column decoded and cut back to the unpadded row count.
pub(crate) fn column_words(bytes: &[u8], n: usize) -> Vec<u32> {
    let mut words = unpack_u32(bytes);
    words.truncate(n);
    words
}

/// The CPU oracle directive kinds and values for a source.
pub(crate) fn cpu_kinds_and_values(
    source: &[u8],
    defined_macros: &[&[u8]],
) -> (Vec<u32>, Vec<u32>) {
    let (tok_types, tok_starts, tok_lens) = build_token_stream(source);
    reference_c_preprocessor_directive_metadata(
        &tok_types,
        &tok_starts,
        &tok_lens,
        source,
        defined_macros,
    )
    .expect("Reference oracle eval")
}
