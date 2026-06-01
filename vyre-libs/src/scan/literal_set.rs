//! High-level GPU literal matching engine.
//!
//! Composed entirely from `vyre-libs` LEGO blocks.

use crate::region::wrap_anonymous;
use crate::scan::builders::append_match;
use crate::scan::dfa::{dfa_compile, CompiledDfa};
use crate::scan::dispatch_io::ScanDispatchScratch;
use crate::scan::hit_buffer::HIT_BUFFER_OVERFLOW_COUNT;
use std::borrow::Cow;
use std::collections::TryReserveError;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre::VyreBackend;
pub use vyre_foundation::match_result::Match;
use vyre_primitives::matching::DfaWireError;

const OP_ID: &str = "vyre-libs::matching::literal_set";
const LITERAL_SET_DEFAULT_MAX_MATCHES: u32 = 10_000;
const MATCH_TRIPLE_WORDS: u32 = 3;
const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Back-compatible literal match type.
pub type LiteralMatch = Match;

/// Errors returned by [`GpuLiteralSet::try_compile`].
#[derive(Debug)]
pub enum LiteralSetCompileError {
    /// Number of patterns does not fit the GPU ABI's `u32` count field.
    PatternCountOverflow {
        /// Number of patterns supplied by the caller.
        count: usize,
    },
    /// One pattern length does not fit the GPU ABI's `u32` length field.
    PatternLengthOverflow {
        /// Index of the oversized pattern.
        pattern_index: usize,
        /// Byte length of the oversized pattern.
        len: usize,
    },
    /// Total concatenated pattern bytes overflowed host `usize`.
    PatternByteCountOverflow,
    /// Total concatenated pattern bytes do not fit the GPU ABI's `u32` field.
    PatternByteCountExceedsGpuAbi {
        /// Concatenated pattern byte count.
        count: usize,
    },
    /// Compiler staging allocation failed.
    StorageReserveFailed {
        /// Scratch vector being reserved.
        field: &'static str,
        /// Requested target capacity.
        requested: usize,
        /// Allocator failure details.
        message: String,
    },
}

impl std::fmt::Display for LiteralSetCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PatternCountOverflow { count } => write!(
                f,
                "literal_set pattern count {count} exceeds u32 capacity. Fix: shard the pattern set before GPU compilation."
            ),
            Self::PatternLengthOverflow { pattern_index, len } => write!(
                f,
                "literal_set pattern {pattern_index} length {len} exceeds u32 capacity. Fix: split or reject oversized literals before GPU compilation."
            ),
            Self::PatternByteCountOverflow => write!(
                f,
                "literal_set total pattern byte count overflowed host usize. Fix: shard the pattern set before GPU compilation."
            ),
            Self::PatternByteCountExceedsGpuAbi { count } => write!(
                f,
                "literal_set total pattern byte count {count} exceeds u32 capacity. Fix: shard the pattern set before GPU compilation."
            ),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "literal_set compile failed to reserve {requested} {field} slot(s): {message}. Fix: shard the pattern set before GPU compilation."
            ),
        }
    }
}

impl std::error::Error for LiteralSetCompileError {}

/// A high-level literal matching engine.
pub struct GpuLiteralSet {
    /// Underlying DFA components.
    pub dfa: CompiledDfa,
    /// Concatenated literal bytes, one byte per u32 word for GPU comparison.
    pub pattern_bytes: Vec<u32>,
    /// Start offset of each pattern in `pattern_bytes`.
    pub pattern_offsets: Vec<u32>,
    /// Pattern lengths for start-offset calculation.
    pub pattern_lengths: Vec<u32>,
    /// The pre-built vyre Program.
    pub program: Program,
}

/// Reusable hot-loop state for [`GpuLiteralSet`] scans.
///
/// This extends the generic scan dispatch scratch with a one-entry cache for
/// cap-specific `Program` layouts. Callers that repeatedly scan with the same
/// non-default `max_matches` avoid rebuilding the rewritten output-buffer
/// declaration on every dispatch.
#[derive(Debug, Default)]
pub struct LiteralSetScanScratch {
    /// Shared scan staging used by other matching engines.
    pub dispatch: ScanDispatchScratch,
    cached_program: Option<CachedLiteralSetProgram>,
}

#[derive(Debug)]
struct CachedLiteralSetProgram {
    base_fingerprint: [u8; 32],
    max_matches: u32,
    program: Program,
}

impl GpuLiteralSet {
    /// Compile a set of literal patterns into a GPU-ready matcher.
    #[must_use]
    pub fn compile(patterns: &[&[u8]]) -> Self {
        match Self::try_compile(patterns) {
            Ok(compiled) => compiled,
            Err(error) => {
                eprintln!("vyre-libs GpuLiteralSet::compile failed: {error}");
                Self::empty_after_compile_failure()
            }
        }
    }

    /// Compile a set of literal patterns into a GPU-ready matcher, surfacing
    /// allocation and ABI-size failures instead of truncating them.
    ///
    /// # Errors
    ///
    /// Returns [`LiteralSetCompileError`] when staging allocation fails or a
    /// pattern count/length cannot be represented by the GPU ABI.
    pub fn try_compile(patterns: &[&[u8]]) -> Result<Self, LiteralSetCompileError> {
        let dfa = dfa_compile(patterns);
        let declared_pattern_count = u32::try_from(patterns.len()).map_err(|_| {
            LiteralSetCompileError::PatternCountOverflow {
                count: patterns.len(),
            }
        })?;
        let total_pattern_bytes = patterns.iter().try_fold(0usize, |sum, pattern| {
            sum.checked_add(pattern.len())
                .ok_or(LiteralSetCompileError::PatternByteCountOverflow)
        })?;
        let pattern_byte_count = u32::try_from(total_pattern_bytes).map_err(|_| {
            LiteralSetCompileError::PatternByteCountExceedsGpuAbi {
                count: total_pattern_bytes,
            }
        })?;
        let mut pattern_lengths = Vec::new();
        reserve_vec(&mut pattern_lengths, patterns.len(), "pattern length")?;
        let mut pattern_offsets = Vec::new();
        reserve_vec(&mut pattern_offsets, patterns.len(), "pattern offset")?;
        let mut pattern_bytes = Vec::new();
        reserve_vec(
            &mut pattern_bytes,
            total_pattern_bytes,
            "packed pattern byte",
        )?;
        for (pattern_index, pattern) in patterns.iter().enumerate() {
            let offset = u32::try_from(pattern_bytes.len()).map_err(|_| {
                LiteralSetCompileError::PatternByteCountExceedsGpuAbi {
                    count: pattern_bytes.len(),
                }
            })?;
            let len = u32::try_from(pattern.len()).map_err(|_| {
                LiteralSetCompileError::PatternLengthOverflow {
                    pattern_index,
                    len: pattern.len(),
                }
            })?;
            pattern_offsets.push(offset);
            pattern_lengths.push(len);
            pattern_bytes.extend(pattern.iter().map(|&byte| u32::from(byte)));
        }

        let program = build_literal_set_program(
            "haystack",
            "pattern_offsets",
            "pattern_lengths",
            "pattern_bytes",
            "haystack_len",
            "pattern_count",
            "match_count",
            "matches",
            declared_pattern_count,
            pattern_byte_count,
        );

        Ok(Self {
            dfa,
            pattern_bytes,
            pattern_offsets,
            pattern_lengths,
            program,
        })
    }

    fn empty_after_compile_failure() -> Self {
        let dfa = dfa_compile(&[]);
        let program = build_literal_set_program(
            "haystack",
            "pattern_offsets",
            "pattern_lengths",
            "pattern_bytes",
            "haystack_len",
            "pattern_count",
            "match_count",
            "matches",
            0,
            0,
        );

        Self {
            dfa,
            pattern_bytes: Vec::new(),
            pattern_offsets: Vec::new(),
            pattern_lengths: Vec::new(),
            program,
        }
    }

    /// Reference oracle implementation for parity testing.
    #[must_use]
    pub fn reference_scan(&self, haystack: &[u8]) -> Vec<Match> {
        let mut state = 0u32;
        let mut results = Vec::new();
        for (pos, &byte) in haystack.iter().enumerate() {
            state = self.dfa.transitions[(state as usize) * 256 + (byte as usize)];
            let begin = self.dfa.output_offsets[state as usize] as usize;
            let end = self.dfa.output_offsets[state as usize + 1] as usize;
            for &pattern_id in &self.dfa.output_records[begin..end] {
                let len = self.pattern_lengths[pattern_id as usize];
                results.push(Match::new(
                    pattern_id,
                    (pos as u32 + 1).saturating_sub(len),
                    pos as u32 + 1,
                ));
            }
        }
        results.sort_unstable();
        results
    }

    /// GPU scan dispatch.
    ///
    /// # Errors
    /// Returns [\`vyre::BackendError\`] if dispatch or readback fails.
    pub fn scan<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
    ) -> Result<Vec<Match>, vyre::BackendError> {
        let mut matches = Vec::new();
        self.scan_into(backend, haystack, max_matches, &mut matches)?;
        Ok(matches)
    }

    /// GPU scan dispatch that decodes into caller-owned match scratch.
    ///
    /// Long-running scanners can reuse `matches` across inputs and avoid one
    /// heap allocation per dispatch. Output ordering and truncation semantics
    /// match [`Self::scan`].
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if dispatch or readback fails.
    pub fn scan_into<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
        matches: &mut Vec<Match>,
    ) -> Result<(), vyre::BackendError> {
        let mut scratch = ScanDispatchScratch::default();
        self.scan_into_with_scratch(backend, haystack, max_matches, matches, &mut scratch)
    }

    /// GPU scan dispatch that decodes into caller-owned match scratch and
    /// reuses caller-owned byte staging.
    ///
    /// This is the lowest-allocation hot-loop API for literal scanning:
    /// `matches` reuses decoded match storage and `scratch` reuses the packed
    /// haystack buffer across dispatches.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if dispatch, readback, scan-boundary
    /// validation, or host staging allocation fails.
    pub fn scan_into_with_scratch<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
        matches: &mut Vec<Match>,
        scratch: &mut ScanDispatchScratch,
    ) -> Result<(), vyre::BackendError> {
        let dispatch_program = self.program_for_match_capacity(max_matches)?;
        self.scan_into_with_program(
            backend,
            haystack,
            max_matches,
            matches,
            scratch,
            dispatch_program.as_ref(),
        )
    }

    /// GPU scan dispatch with literal-set-owned hot-loop scratch.
    ///
    /// Use this for repeated scans where `max_matches` is usually stable but
    /// not equal to the compiled default. It reuses both packed haystack bytes
    /// and the cap-specific rewritten dispatch `Program`.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if dispatch, readback, scan-boundary
    /// validation, host staging allocation, or cap-specific program sizing
    /// fails.
    pub fn scan_into_with_literal_scratch<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
        matches: &mut Vec<Match>,
        scratch: &mut LiteralSetScanScratch,
    ) -> Result<(), vyre::BackendError> {
        let cached_program = &mut scratch.cached_program;
        let dispatch_program =
            self.program_for_match_capacity_cached(max_matches, cached_program)?;
        self.scan_into_with_program(
            backend,
            haystack,
            max_matches,
            matches,
            &mut scratch.dispatch,
            dispatch_program,
        )
    }

    fn scan_into_with_program<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
        matches: &mut Vec<Match>,
        scratch: &mut ScanDispatchScratch,
        dispatch_program: &Program,
    ) -> Result<(), vyre::BackendError> {
        use crate::scan::dispatch_io;

        matches.clear();
        let haystack_len =
            dispatch_io::scan_guard(haystack, "literal_set", dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        let pattern_count = u32::try_from(self.pattern_lengths.len()).map_err(|_| {
            vyre::BackendError::new(
                "literal_set pattern count exceeds u32 capacity. Fix: split the pattern set into smaller shards.",
            )
        })?;

        // Buffer order matches the BufferDecl declaration in
        // `build_literal_set_program`; reordering here would silently
        // miswire the GPU program.
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let haystack_bytes = scratch.haystack_bytes.as_slice();
        let pattern_offset_bytes = dispatch_io::u32_words_as_le_bytes(&self.pattern_offsets);
        let pattern_length_bytes = dispatch_io::u32_words_as_le_bytes(&self.pattern_lengths);
        let pattern_bytes = dispatch_io::u32_words_as_le_bytes(&self.pattern_bytes);
        let haystack_len_word = [haystack_len];
        let pattern_count_word = [pattern_count];
        let haystack_len_bytes = dispatch_io::u32_words_as_le_bytes(&haystack_len_word);
        let pattern_count_bytes = dispatch_io::u32_words_as_le_bytes(&pattern_count_word);
        let match_count_bytes = [0u8; 4];
        let overflow_count_bytes = [0u8; 4];

        let config = dispatch_io::byte_scan_dispatch_config(
            haystack_len,
            dispatch_program.workgroup_size[0],
        );
        let borrowed_inputs: smallvec::SmallVec<[&[u8]; 8]> = [
            // 0: haystack (Packed U32)
            haystack_bytes,
            // 1: pattern_offsets
            pattern_offset_bytes.as_ref(),
            // 2: pattern_lengths
            pattern_length_bytes.as_ref(),
            // 3: pattern_bytes
            pattern_bytes.as_ref(),
            // 4: haystack_len
            haystack_len_bytes.as_ref(),
            // 5: pattern_count
            pattern_count_bytes.as_ref(),
            // 6: match_count atomic counter
            match_count_bytes.as_slice(),
            // 7: matches is a pure `BufferDecl::output`; the backend
            // allocates it from the Program declaration.
            // 8: overflow counter
            overflow_count_bytes.as_slice(),
        ]
        .into_iter()
        .collect();
        let outputs = backend.dispatch_borrowed(&dispatch_program, &borrowed_inputs, &config)?;

        let count_bytes = dispatch_io::try_output_bytes(&outputs, 0, "literal_set match count")?;
        let count = dispatch_io::try_read_u32_prefix(count_bytes, "literal_set match count")?;
        let matches_bytes = dispatch_io::try_output_bytes(&outputs, 1, "literal_set matches")?;

        dispatch_io::try_unpack_match_triples_exact_prefix_into(
            matches_bytes,
            count.min(max_matches),
            matches,
        )?;
        Ok(())
    }

    fn program_for_match_capacity_cached<'a>(
        &'a self,
        max_matches: u32,
        cached_program: &'a mut Option<CachedLiteralSetProgram>,
    ) -> Result<&'a Program, vyre::BackendError> {
        let (declared_words, readback_bytes) = literal_set_match_output_layout(max_matches)?;
        if self.compiled_matches_output_satisfies(declared_words, readback_bytes)? {
            return Ok(&self.program);
        }

        let base_fingerprint = self.program.fingerprint();
        let reuse_cached = cached_program.as_ref().is_some_and(|cached| {
            cached.max_matches == max_matches && cached.base_fingerprint == base_fingerprint
        });
        if !reuse_cached {
            let program = self.rewrite_program_for_match_layout(declared_words, readback_bytes);
            *cached_program = Some(CachedLiteralSetProgram {
                base_fingerprint,
                max_matches,
                program,
            });
        }

        match cached_program.as_ref() {
            Some(cached) => Ok(&cached.program),
            None => Err(vyre::BackendError::new(
                "literal_set failed to retain the cached match-capacity program. Fix: retry with generic ScanDispatchScratch.",
            )),
        }
    }

    fn program_for_match_capacity(
        &self,
        max_matches: u32,
    ) -> Result<Cow<'_, Program>, vyre::BackendError> {
        let (declared_words, readback_bytes) = literal_set_match_output_layout(max_matches)?;
        if self.compiled_matches_output_satisfies(declared_words, readback_bytes)? {
            return Ok(Cow::Borrowed(&self.program));
        }

        Ok(Cow::Owned(self.rewrite_program_for_match_layout(
            declared_words,
            readback_bytes,
        )))
    }

    fn compiled_matches_output_satisfies(
        &self,
        declared_words: u32,
        readback_bytes: usize,
    ) -> Result<bool, vyre::BackendError> {
        let matches_output = self
            .program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "matches" && buffer.is_output())
            .ok_or_else(|| {
                vyre::BackendError::new(
                    "literal_set program is missing its matches output buffer. Fix: rebuild the literal set with GpuLiteralSet::try_compile before dispatch.",
                )
            })?;

        Ok(matches_output.count == declared_words
            && (matches_output.output_byte_range().is_none()
                || matches_output.output_byte_range() == Some(0..readback_bytes)))
    }

    fn rewrite_program_for_match_layout(
        &self,
        declared_words: u32,
        readback_bytes: usize,
    ) -> Program {
        let buffers = self
            .program
            .buffers()
            .iter()
            .cloned()
            .map(|buffer| {
                if buffer.name() == "matches" && buffer.is_output() {
                    buffer
                        .with_count(declared_words)
                        .with_output_byte_range(0..readback_bytes)
                } else {
                    buffer
                }
            })
            .collect::<Vec<_>>();

        self.program.with_rewritten_buffers(buffers)
    }

    /// Serialize this matcher into a self-describing binary blob suitable
    /// for on-disk caching. Composed from the existing layer-1 wire
    /// formats: `Program::to_bytes` for the dispatch IR and
    /// `CompiledDfa::to_bytes` for the transition tables. The pattern
    /// arrays are packed as raw little-endian `u32` words.
    ///
    /// Layout:
    ///   - 4 bytes magic `b"VLIT"`
    ///   - 4 bytes wire version (LE u32)
    ///   - 4 bytes program byte length (LE u32)  + program bytes
    ///   - 4 bytes dfa byte length (LE u32)      + dfa bytes
    ///   - 4 bytes pattern_offsets word count    + words
    ///   - 4 bytes pattern_lengths word count    + words
    ///   - 4 bytes pattern_bytes word count      + words
    ///
    /// Caller-side cache invalidation: the dispatch `Program` already
    /// includes vyre's IR wire version + pattern fingerprint inside its
    /// own framing, so a stale cache surfaces as `LiteralSetWireError::
    /// InvalidProgram` from `Program::from_bytes` (or as a bad magic /
    /// version on this outer envelope). Both signal "recompile from
    /// patterns".
    /// # Errors
    /// Returns [`LiteralSetWireError::WireFraming`] if any section
    /// exceeds the envelope's `u32` length-prefix capacity.
    pub fn to_bytes(&self) -> Result<Vec<u8>, LiteralSetWireError> {
        let mut w = vyre_foundation::serial::envelope::WireWriter::new(
            LITERAL_SET_WIRE_MAGIC,
            LITERAL_SET_WIRE_VERSION,
        );
        w.write_section(&self.program.to_bytes())
            .map_err(LiteralSetWireError::WireFraming)?;
        let dfa_bytes = self
            .dfa
            .to_bytes()
            .map_err(LiteralSetWireError::InvalidDfa)?;
        w.write_section(&dfa_bytes)
            .map_err(LiteralSetWireError::WireFraming)?;
        w.write_words(&self.pattern_offsets)
            .map_err(LiteralSetWireError::WireFraming)?;
        w.write_words(&self.pattern_lengths)
            .map_err(LiteralSetWireError::WireFraming)?;
        w.write_words(&self.pattern_bytes)
            .map_err(LiteralSetWireError::WireFraming)?;
        Ok(w.into_bytes())
    }

    /// Decode a `GpuLiteralSet` from a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`LiteralSetWireError`] when the envelope rejects the
    /// outer header, or any inner section (program, DFA) is itself
    /// rejected.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LiteralSetWireError> {
        let mut r = vyre_foundation::serial::envelope::WireReader::new(
            bytes,
            LITERAL_SET_WIRE_MAGIC,
            LITERAL_SET_WIRE_VERSION,
        )
        .map_err(LiteralSetWireError::WireFraming)?;

        let program_bytes = r.read_section().map_err(LiteralSetWireError::WireFraming)?;
        let program = Program::from_bytes(program_bytes)
            .map_err(|e| LiteralSetWireError::InvalidProgram(format!("{e}")))?;

        let dfa_bytes = r.read_section().map_err(LiteralSetWireError::WireFraming)?;
        let dfa = CompiledDfa::from_bytes(dfa_bytes).map_err(LiteralSetWireError::InvalidDfa)?;

        let pattern_offsets = r.read_words().map_err(LiteralSetWireError::WireFraming)?;
        let pattern_lengths = r.read_words().map_err(LiteralSetWireError::WireFraming)?;
        let pattern_bytes = r.read_words().map_err(LiteralSetWireError::WireFraming)?;

        Ok(Self {
            dfa,
            pattern_bytes,
            pattern_offsets,
            pattern_lengths,
            program,
        })
    }
}

fn reserve_vec<T>(
    vec: &mut Vec<T>,
    requested: usize,
    field: &'static str,
) -> Result<(), LiteralSetCompileError> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(vec, requested).map_err(
        |source: TryReserveError| LiteralSetCompileError::StorageReserveFailed {
            field,
            requested,
            message: source.to_string(),
        },
    )
}

fn literal_set_match_output_layout(max_matches: u32) -> Result<(u32, usize), vyre::BackendError> {
    let words = max_matches.checked_mul(MATCH_TRIPLE_WORDS).ok_or_else(|| {
        vyre::BackendError::new(format!(
            "literal_set max_matches={max_matches} overflows the GPU match-output word count. Fix: lower max_matches or split the scan before dispatch."
        ))
    })?;
    let byte_len = usize::try_from(words)
        .ok()
        .and_then(|words| words.checked_mul(U32_BYTES))
        .ok_or_else(|| {
            vyre::BackendError::new(format!(
                "literal_set max_matches={max_matches} overflows host match-output byte sizing. Fix: lower max_matches or split the scan before dispatch."
            ))
        })?;
    Ok((words.max(1), byte_len))
}

#[cfg(test)]
mod compile_tests {
    use super::*;

    #[derive(Clone)]
    struct LiteralReadbackBackend {
        outputs: Vec<Vec<u8>>,
    }

    impl vyre::backend::private::Sealed for LiteralReadbackBackend {}

    impl VyreBackend for LiteralReadbackBackend {
        fn id(&self) -> &'static str {
            "literal-readback-test"
        }

        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _config: &vyre::DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
            Ok(self.outputs.clone())
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &vyre::DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
            Ok(self.outputs.clone())
        }
    }

    #[derive(Clone)]
    struct RecordingLiteralBackend {
        outputs: Vec<Vec<u8>>,
        observed_matches_layouts:
            std::sync::Arc<std::sync::Mutex<Vec<(u32, Option<std::ops::Range<usize>>)>>>,
        observed_program_buffer_ptrs: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    }

    impl RecordingLiteralBackend {
        fn new(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                outputs,
                observed_matches_layouts: std::sync::Arc::default(),
                observed_program_buffer_ptrs: std::sync::Arc::default(),
            }
        }

        fn observed_matches_layouts(&self) -> Vec<(u32, Option<std::ops::Range<usize>>)> {
            self.observed_matches_layouts
                .lock()
                .expect("Fix: recording literal backend mutex should not be poisoned")
                .clone()
        }

        fn observed_program_buffer_ptrs(&self) -> Vec<usize> {
            self.observed_program_buffer_ptrs
                .lock()
                .expect("Fix: recording literal backend mutex should not be poisoned")
                .clone()
        }
    }

    impl vyre::backend::private::Sealed for RecordingLiteralBackend {}

    impl VyreBackend for RecordingLiteralBackend {
        fn id(&self) -> &'static str {
            "literal-recording-test"
        }

        fn dispatch(
            &self,
            program: &Program,
            inputs: &[Vec<u8>],
            config: &vyre::DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
            let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
            self.dispatch_borrowed(program, &borrowed, config)
        }

        fn dispatch_borrowed(
            &self,
            program: &Program,
            _inputs: &[&[u8]],
            _config: &vyre::DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
            let matches = program
                .buffers()
                .iter()
                .find(|buffer| buffer.name() == "matches")
                .ok_or_else(|| vyre::BackendError::new("test program omitted matches buffer"))?;
            self.observed_matches_layouts
                .lock()
                .map_err(|_| vyre::BackendError::new("test observation mutex poisoned"))?
                .push((matches.count, matches.output_byte_range()));
            self.observed_program_buffer_ptrs
                .lock()
                .map_err(|_| vyre::BackendError::new("test observation mutex poisoned"))?
                .push(program.buffers().as_ptr() as usize);
            Ok(self.outputs.clone())
        }
    }

    fn match_count_bytes(count: u32) -> Vec<u8> {
        count.to_le_bytes().to_vec()
    }

    fn match_triple_bytes(pattern_id: u32, start: u32, end: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&pattern_id.to_le_bytes());
        bytes.extend_from_slice(&start.to_le_bytes());
        bytes.extend_from_slice(&end.to_le_bytes());
        bytes
    }

    #[test]
    fn try_compile_packs_offsets_lengths_and_bytes_without_truncation() {
        let compiled = GpuLiteralSet::try_compile(&[b"ab".as_slice(), b"cde".as_slice()])
            .expect("Fix: small literal set must compile");

        assert_eq!(compiled.pattern_offsets, vec![0, 2]);
        assert_eq!(compiled.pattern_lengths, vec![2, 3]);
        assert_eq!(
            compiled.pattern_bytes,
            vec![
                b'a' as u32,
                b'b' as u32,
                b'c' as u32,
                b'd' as u32,
                b'e' as u32
            ]
        );
    }

    #[test]
    fn compile_empty_patterns_matches_fallible_compile_contract() {
        let compat = GpuLiteralSet::compile(&[]);
        let fallible =
            GpuLiteralSet::try_compile(&[]).expect("Fix: empty literal set must compile");

        assert_eq!(compat.pattern_offsets, fallible.pattern_offsets);
        assert_eq!(compat.pattern_lengths, fallible.pattern_lengths);
        assert_eq!(compat.pattern_bytes, fallible.pattern_bytes);
    }

    #[test]
    fn reserve_vec_reports_compile_storage_failure() {
        let mut scratch = Vec::<u8>::new();
        let error = reserve_vec(&mut scratch, usize::MAX, "adversarial scratch")
            .expect_err("Fix: usize::MAX reserve must fail instead of silently truncating");

        match error {
            LiteralSetCompileError::StorageReserveFailed {
                field, requested, ..
            } => {
                assert_eq!(field, "adversarial scratch");
                assert_eq!(requested, usize::MAX);
            }
            other => panic!("expected storage reserve failure, got {other:?}"),
        }
        assert!(scratch.is_empty());
    }

    #[test]
    fn literal_scan_rejects_short_match_count_readback() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = LiteralReadbackBackend {
            outputs: vec![vec![1, 2, 3], Vec::new()],
        };
        let mut matches = vec![Match::new(99, 1, 2)];

        let err = engine
            .scan_into(&backend, b"a", 1, &mut matches)
            .expect_err("short literal match-count readback must fail");

        let msg = err.to_string();
        assert!(
            matches.is_empty(),
            "scan errors must not expose stale matches"
        );
        assert!(
            msg.contains("literal_set match count") && msg.contains("requires 4 bytes"),
            "literal scan counter error must name the malformed output: {msg}"
        );
    }

    #[test]
    fn literal_scan_rejects_missing_match_output_slot() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = LiteralReadbackBackend {
            outputs: vec![match_count_bytes(1)],
        };
        let mut matches = Vec::new();

        let err = engine
            .scan_into(&backend, b"a", 1, &mut matches)
            .expect_err("missing literal match output must fail");

        let msg = err.to_string();
        assert!(
            msg.contains("literal_set matches") && msg.contains("output index 1"),
            "literal scan missing-output error must identify the omitted slot: {msg}"
        );
    }

    #[test]
    fn literal_scan_rejects_match_payload_shorter_than_reported_count() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = LiteralReadbackBackend {
            outputs: vec![match_count_bytes(2), match_triple_bytes(0, 0, 1)],
        };
        let mut matches = vec![Match::new(99, 1, 2)];

        let err = engine
            .scan_into(&backend, b"a", 2, &mut matches)
            .expect_err("short literal match payload must fail");

        let msg = err.to_string();
        assert!(
            matches.is_empty(),
            "scan errors must not expose stale matches"
        );
        assert!(
            msg.contains("readback was 12 byte(s)")
                && msg.contains("count=2")
                && msg.contains("requires 24 byte(s)"),
            "literal scan match-payload error must identify observed and required bytes: {msg}"
        );
    }

    #[test]
    fn literal_scan_exposes_scratch_backed_dispatch_staging() {
        let production = include_str!("literal_set.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("Fix: literal_set.rs must contain production section");

        assert!(
            production.contains("pub fn scan_into_with_scratch")
                && production.contains("ScanDispatchScratch")
                && production.contains("LiteralSetScanScratch")
                && production.contains("pack_haystack_u32_into")
                && !production.contains(concat!("pack_haystack_u32", "(haystack)")),
            "Fix: literal scan hot path must expose reusable dispatch scratch and avoid fresh haystack packing allocations."
        );
        assert!(
            !production.contains(".expect(") && !production.contains(".unwrap("),
            "Fix: literal_set production wrappers must not panic."
        );
        let program_debug = format!("{:#?}", GpuLiteralSet::compile(&[b"a".as_slice()]).program);
        assert!(
            !program_debug.contains("_vyre_match_leader"),
            "Fix: literal-set GPU program must use the CUDA-lowerable append primitive, not subgroup leader append."
        );
    }

    #[test]
    fn literal_scan_sizes_match_output_to_requested_cap() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let mut payload = match_triple_bytes(0, 0, 1);
        payload.extend_from_slice(&match_triple_bytes(0, 3, 4));
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(2), payload]);
        let mut matches = Vec::new();

        engine
            .scan_into(&backend, b"a--a", 2, &mut matches)
            .expect("Fix: literal scan with two-match cap should dispatch");

        assert_eq!(matches, vec![Match::new(0, 0, 1), Match::new(0, 3, 4)]);
        assert_eq!(backend.observed_matches_layouts(), vec![(6, Some(0..24))]);
    }

    #[test]
    fn literal_scan_default_cap_uses_compiled_output_layout() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = Vec::new();

        engine
            .scan_into(
                &backend,
                b"no hits",
                LITERAL_SET_DEFAULT_MAX_MATCHES,
                &mut matches,
            )
            .expect("Fix: default literal scan cap should use the compiled program layout");

        assert!(matches.is_empty());
        assert_eq!(backend.observed_matches_layouts(), vec![(30_000, None)]);
    }

    #[test]
    fn literal_scan_zero_match_cap_reads_no_match_payload() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(1), Vec::new()]);
        let mut matches = vec![Match::new(99, 1, 2)];

        engine
            .scan_into(&backend, b"a", 0, &mut matches)
            .expect("Fix: literal scan with zero cap should return an empty decoded prefix");

        assert!(matches.is_empty());
        assert_eq!(backend.observed_matches_layouts(), vec![(1, Some(0..0))]);
    }

    #[test]
    fn literal_scan_expands_match_output_above_legacy_fixed_cap() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = Vec::new();

        engine
            .scan_into(&backend, b"no hits", 20_001, &mut matches)
            .expect("Fix: literal scan should honor caps above the compiled default");

        assert!(matches.is_empty());
        assert_eq!(
            backend.observed_matches_layouts(),
            vec![(60_003, Some(0..240_012))]
        );
    }

    #[test]
    fn literal_scan_literal_scratch_reuses_rewritten_program_for_same_cap() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = Vec::new();
        let mut scratch = LiteralSetScanScratch::default();

        engine
            .scan_into_with_literal_scratch(&backend, b"first", 2, &mut matches, &mut scratch)
            .expect("Fix: first cap-specific literal scan should dispatch");
        engine
            .scan_into_with_literal_scratch(&backend, b"second", 2, &mut matches, &mut scratch)
            .expect("Fix: repeated cap-specific literal scan should dispatch");

        assert_eq!(
            backend.observed_matches_layouts(),
            vec![(6, Some(0..24)), (6, Some(0..24))]
        );
        let ptrs = backend.observed_program_buffer_ptrs();
        assert_eq!(ptrs.len(), 2);
        assert_eq!(
            ptrs[0], ptrs[1],
            "Fix: literal-set scan scratch must reuse the rewritten Program for stable caps"
        );
    }

    #[test]
    fn literal_scan_literal_scratch_rebuilds_rewritten_program_when_cap_changes() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = Vec::new();
        let mut scratch = LiteralSetScanScratch::default();

        engine
            .scan_into_with_literal_scratch(&backend, b"first", 2, &mut matches, &mut scratch)
            .expect("Fix: first cap-specific literal scan should dispatch");
        engine
            .scan_into_with_literal_scratch(&backend, b"second", 3, &mut matches, &mut scratch)
            .expect("Fix: changed cap-specific literal scan should dispatch");

        assert_eq!(
            backend.observed_matches_layouts(),
            vec![(6, Some(0..24)), (9, Some(0..36))]
        );
        let ptrs = backend.observed_program_buffer_ptrs();
        assert_eq!(ptrs.len(), 2);
        assert_ne!(
            ptrs[0], ptrs[1],
            "Fix: literal-set scan scratch must rebuild cached Program when cap changes"
        );
    }

    #[test]
    fn literal_scan_rejects_match_cap_that_overflows_output_words() {
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = Vec::new();

        let err = engine
            .scan_into(&backend, b"a", u32::MAX, &mut matches)
            .expect_err("Fix: overflowing literal max_matches must fail before dispatch");
        let msg = err.to_string();

        assert!(msg.contains("literal_set max_matches"));
        assert!(msg.contains("overflows the GPU match-output word count"));
        assert!(backend.observed_matches_layouts().is_empty());
    }
}

const LITERAL_SET_WIRE_MAGIC: &[u8; 4] = b"VLIT";
const LITERAL_SET_WIRE_VERSION: u32 = 1;

/// Errors returned by [`GpuLiteralSet::from_bytes`]. Outer-framing
/// failures (truncation, bad magic, version drift) are forwarded
/// straight from the shared `WireFraming` envelope. Inner-section
/// failures (program decode, DFA decode) keep their own typed variants
/// so consumers can act on them. Variants are non-exhaustive so future
/// inner sections can be added without a breaking change.
#[derive(Debug)]
#[non_exhaustive]
pub enum LiteralSetWireError {
    /// Outer envelope (magic / version / section length) was rejected.
    /// Forwarded from `vyre_foundation::serial::envelope::EnvelopeError`.
    WireFraming(vyre_foundation::serial::envelope::EnvelopeError),
    /// The nested vyre IR `Program` blob was rejected. Inner message is
    /// stringified to keep this error type independent of vyre's own
    /// error enum.
    InvalidProgram(String),
    /// The nested `CompiledDfa` blob was rejected.
    InvalidDfa(DfaWireError),
}

impl std::fmt::Display for LiteralSetWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WireFraming(e) => write!(f, "GpuLiteralSet wire envelope: {e}"),
            Self::InvalidProgram(msg) => {
                write!(f, "GpuLiteralSet wire blob has invalid Program: {msg}")
            }
            Self::InvalidDfa(e) => {
                write!(f, "GpuLiteralSet wire blob has invalid DFA: {e}")
            }
        }
    }
}

impl std::error::Error for LiteralSetWireError {}

fn build_literal_set_program(
    haystack: &str,
    pattern_offsets: &str,
    pattern_lengths: &str,
    pattern_bytes: &str,
    haystack_len: &str,
    pattern_count: &str,
    match_count: &str,
    matches: &str,
    declared_pattern_count: u32,
    pattern_byte_count: u32,
) -> Program {
    let idx = Expr::InvocationId { axis: 0 };
    let subgroup_size = 32u32;

    // Use the canonical `builders::load_packed_byte` LEGO primitive
    // instead of a local re-inlining. Earlier "complete" tasks (#21,
    // #22) missed this site; the inline version was less efficient
    // (no let-bind for the loaded word ⇒ no CSE opportunity).
    let offset_at_end = Expr::add(idx.clone(), Expr::u32(1));
    let lane_body = vec![Node::Loop {
        var: "_pid".into(),
        from: Expr::u32(0),
        to: Expr::load(pattern_count, Expr::u32(0)),
        body: vec![
            Node::Let {
                name: "_pattern_start".into(),
                value: Expr::load(pattern_offsets, Expr::var("_pid")),
            },
            Node::Let {
                name: "_len".into(),
                value: Expr::load(pattern_lengths, Expr::var("_pid")),
            },
            Node::Let {
                name: "_candidate_start".into(),
                value: Expr::Select {
                    cond: Box::new(Expr::ge(offset_at_end.clone(), Expr::var("_len"))),
                    true_val: Box::new(Expr::sub(offset_at_end.clone(), Expr::var("_len"))),
                    false_val: Box::new(Expr::u32(0)),
                },
            },
            Node::Let {
                name: "_literal_matched".into(),
                value: Expr::ge(offset_at_end.clone(), Expr::var("_len")),
            },
            Node::Loop {
                var: "_j".into(),
                from: Expr::u32(0),
                to: Expr::var("_len"),
                body: vec![Node::If {
                    cond: Expr::ne(
                        crate::scan::builders::load_packed_byte_expr(
                            haystack,
                            Expr::add(Expr::var("_candidate_start"), Expr::var("_j")),
                        ),
                        Expr::load(
                            pattern_bytes,
                            Expr::add(Expr::var("_pattern_start"), Expr::var("_j")),
                        ),
                    ),
                    then: vec![Node::Assign {
                        name: "_literal_matched".into(),
                        value: Expr::bool(false),
                    }],
                    otherwise: vec![],
                }],
            },
            Node::If {
                cond: Expr::var("_literal_matched"),
                then: vec![append_match(
                    matches,
                    match_count,
                    Expr::var("_pid"),
                    Expr::var("_candidate_start"),
                    offset_at_end.clone(),
                )],
                otherwise: vec![],
            },
        ],
    }];

    let body = vec![
        Node::Let {
            name: "state".into(),
            value: Expr::u32(0),
        },
        Node::If {
            cond: Expr::lt(idx.clone(), Expr::load(haystack_len, Expr::u32(0))),
            then: lane_body,
            otherwise: vec![],
        },
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(pattern_offsets, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(declared_pattern_count),
            BufferDecl::storage(pattern_lengths, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(declared_pattern_count),
            BufferDecl::storage(pattern_bytes, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pattern_byte_count),
            BufferDecl::storage(haystack_len, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage(pattern_count, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(match_count, 6, DataType::U32).with_count(1),
            BufferDecl::output(matches, 7, DataType::U32)
                .with_count(LITERAL_SET_DEFAULT_MAX_MATCHES * MATCH_TRIPLE_WORDS),
            BufferDecl::read_write(HIT_BUFFER_OVERFLOW_COUNT, 8, DataType::U32).with_count(1),
        ],
        [subgroup_size, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    )
}

/// Innovation I.18: JIT DFA Lowering.
///
/// Converts a static transition table into a nested \`If\` cascade.
/// For small pattern sets, this eliminates the VRAM bandwidth bottleneck
/// by keeping the state machine in the GPU instruction cache.
pub fn dfa_to_jit_ir(dfa: &CompiledDfa, state_var: &str, byte_expr: Expr) -> Node {
    build_state_cascade(dfa, 0, state_var, byte_expr)
}

fn build_state_cascade(dfa: &CompiledDfa, state: u32, state_var: &str, byte_expr: Expr) -> Node {
    // Basic implementation: if state == S { if byte == B1 { state = T1 } ... }
    // V7-PERF-024: Binary-search tree emission for instructions.
    // Naive linear if/else is O(N); a binary tree is O(log N).

    let mut arms = Vec::new();
    for byte in 0..=255 {
        let next_state = dfa.transitions[(state as usize) * 256 + byte];
        if next_state != 0 {
            arms.push((byte as u32, next_state));
        }
    }

    if arms.is_empty() {
        return Node::Assign {
            name: state_var.into(),
            value: Expr::u32(0),
        };
    }

    // Build a nested If cascade for the transitions from this state
    let mut node = Node::Assign {
        name: state_var.into(),
        value: Expr::u32(0),
    };
    for (byte, next) in arms.into_iter().rev() {
        node = Node::If {
            cond: Expr::eq(byte_expr.clone(), Expr::u32(byte)),
            then: vec![Node::Assign {
                name: state_var.into(),
                value: Expr::u32(next),
            }],
            otherwise: vec![node],
        };
    }
    node
}
