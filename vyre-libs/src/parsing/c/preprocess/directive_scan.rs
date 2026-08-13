//! One owner for the CPU-side preprocessor directive-row scan.
//!
//! Three reference entry points consume `TOK_PREPROC` rows: directive metadata
//! in [`super`], side-effect metadata in [`super::effects`], and include
//! loading in [`super::source`]. Each one used to carry its own copy of the
//! token-stream walk and of the per-row splice/classify/payload step, so a
//! bounds rule or an error message could be tightened in one and not the other
//! two. Both stages live here once.

#[cfg(any(test, feature = "cpu-parity"))]
use crate::parsing::c::lex::tokens::TOK_PREPROC;
#[cfg(any(test, feature = "cpu-parity"))]
use crate::parsing::c::preprocess::c_logical_directive_len;
use crate::parsing::c::preprocess::{
    c_directive_payload, c_translation_phase_line_splice, classify_phase2_preprocessor_directive,
    CLineSplicedSource, CPreprocessorDirective, CPreprocessorError,
};

/// One `TOK_PREPROC` row resolved against the source buffer.
#[cfg(any(test, feature = "cpu-parity"))]
pub(super) struct DirectiveRow<'a> {
    /// Index of the row's token in the token streams.
    pub index: usize,
    /// Physical offset of the row's first byte in the source buffer.
    pub start: usize,
    /// Physical row bytes, phase-2 splices still present.
    pub bytes: &'a [u8],
}

/// Walk every `TOK_PREPROC` row in a compact token stream.
///
/// Validates stream lengths, host-index range, and that each token span both
/// fits the source buffer and covers the full phase-2 spliced row, then hands
/// the resolved row to `visit`.
///
/// # Errors
///
/// Returns a diagnostic when the token streams disagree in length, an index
/// does not fit host `usize`, a span overflows the address space, a span falls
/// outside the source buffer, or a span truncates a spliced directive row.
#[cfg(any(test, feature = "cpu-parity"))]
pub(super) fn for_each_directive_row<F>(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
    source: &[u8],
    mut visit: F,
) -> Result<(), CPreprocessorError>
where
    F: FnMut(DirectiveRow<'_>) -> Result<(), CPreprocessorError>,
{
    if tok_types.len() != tok_starts.len() || tok_types.len() != tok_lens.len() {
        return Err(CPreprocessorError {
            offset: tok_types.len().min(tok_starts.len()).min(tok_lens.len()),
            message: "Fix: token type/start/length streams must have identical lengths",
        });
    }

    for (index, ((tok_type, start), len)) in
        tok_types.iter().zip(tok_starts).zip(tok_lens).enumerate()
    {
        if *tok_type != TOK_PREPROC {
            continue;
        }
        let start = usize::try_from(*start).map_err(|_| CPreprocessorError {
            offset: index,
            message: "Fix: token start does not fit host usize",
        })?;
        let len = usize::try_from(*len).map_err(|_| CPreprocessorError {
            offset: index,
            message: "Fix: token length does not fit host usize",
        })?;
        let token_end = start.checked_add(len).ok_or(CPreprocessorError {
            offset: start,
            message: "Fix: token span overflows source address space",
        })?;
        let logical_len = c_logical_directive_len(source, start);
        if logical_len > len {
            return Err(CPreprocessorError {
                offset: start + len,
                message: "Fix: TOK_PREPROC span must include the full phase-2 spliced directive row",
            });
        }
        if token_end > source.len() {
            return Err(CPreprocessorError {
                offset: start,
                message: "Fix: preprocessor token span must be inside the source buffer",
            });
        }
        let logical_end = start.checked_add(logical_len).ok_or(CPreprocessorError {
            offset: start,
            message: "Fix: directive logical span overflows source address space",
        })?;
        let bytes = source.get(start..logical_end).ok_or(CPreprocessorError {
            offset: start,
            message: "Fix: preprocessor token span must be inside the source buffer",
        })?;
        visit(DirectiveRow { index, start, bytes })?;
    }
    Ok(())
}

/// A directive row after phase-2 splicing and classification.
///
/// Holds the splice map alongside the classification so every derived offset
/// and every diagnostic can be mapped back to a physical source offset through
/// [`ScannedDirective::source_offset`].
pub(super) struct ScannedDirective {
    /// Phase-2 bytes of the row and the map back to physical offsets.
    pub spliced: CLineSplicedSource,
    /// Directive kind and logical spans, in phase-2 coordinates.
    pub directive: CPreprocessorDirective,
    /// Physical offset of the row's first byte in the source buffer.
    directive_offset: usize,
}

impl ScannedDirective {
    /// Splice and classify one physical directive row.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic, with its offset already mapped to physical source
    /// coordinates, when the row is not a directive row or uses an unsupported
    /// directive spelling.
    pub(super) fn classify(
        row: &[u8],
        directive_offset: usize,
    ) -> Result<Self, CPreprocessorError> {
        let spliced = c_translation_phase_line_splice(row);
        let directive =
            classify_phase2_preprocessor_directive(&spliced.bytes).map_err(|mut err| {
                err.offset = directive_offset + spliced.original_offset(err.offset);
                err
            })?;
        Ok(Self {
            spliced,
            directive,
            directive_offset,
        })
    }

    /// Physical source offset of a phase-2 offset inside this row.
    pub(super) fn source_offset(&self, phase2_offset: usize) -> usize {
        self.directive_offset + self.spliced.original_offset(phase2_offset)
    }

    /// Remap a diagnostic raised at a phase-2 offset inside this row.
    pub(super) fn remap(&self, mut err: CPreprocessorError) -> CPreprocessorError {
        err.offset = self.source_offset(err.offset);
        err
    }

    /// The directive's payload bytes, in phase-2 coordinates.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic, offset already physical, when the payload span is
    /// not inside the spliced row.
    pub(super) fn payload(&self) -> Result<&[u8], CPreprocessorError> {
        c_directive_payload(&self.spliced.bytes, self.directive).map_err(|err| self.remap(err))
    }
}

/// Skip a run of horizontal whitespace, stopping at anything else.
///
/// Unlike [`super::skip_ws_and_comments`] this does not treat a comment as
/// whitespace: it runs on directive payloads, which the comment-strip stage has
/// already replaced with spaces.
pub(super) fn skip_horizontal_ws(bytes: &[u8], mut index: usize) -> usize {
    while matches!(bytes.get(index), Some(b' ' | b'\t' | b'\x0b' | b'\x0c')) {
        index += 1;
    }
    index
}
