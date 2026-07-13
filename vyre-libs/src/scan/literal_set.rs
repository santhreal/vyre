//! High-level GPU literal matching engine.
//!
//! Composed entirely from `vyre-libs` LEGO blocks.

use crate::scan::classic_ac::{
    ascii_case_variants, build_ac_bounded_count_suffix3_prefilter_program,
    classic_ac_candidate_suffix3_bloom_words_ci, presence_bitmap_words, presence_by_region_words,
    try_build_ac_bounded_ranges_suffix3_prefilter_program_ext,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_program, CLASSIC_AC_SUFFIX2_MASK_WORDS,
};
use crate::scan::dfa::{dfa_compile, dfa_compile_case_insensitive, CompiledDfa};
use crate::scan::dispatch_io::ScanDispatchScratch;
use std::borrow::Cow;
use std::collections::TryReserveError;
use vyre::backend::PendingDispatch;
use vyre::ir::Program;
use vyre::{DispatchConfig, VyreBackend};
use vyre_driver::Resource;
pub use vyre_foundation::match_result::Match;
use vyre_primitives::matching::DfaWireError;

const LITERAL_SET_DEFAULT_MAX_MATCHES: u32 = 10_000;
const MATCH_TRIPLE_WORDS: u32 = 3;
const U32_BYTES: usize = std::mem::size_of::<u32>();
const U32_COUNTER_BYTES: usize = std::mem::size_of::<u32>();
const LITERAL_SET_INPUT_COUNT: usize = 10;
const LITERAL_SET_COUNT_INPUT_COUNT: usize = 8;

/// Resident-resource index containing the mutable literal-set match counter.
pub const LITERAL_SET_MATCH_COUNT_RESOURCE_INDEX: usize = 6;

/// Resident-resource index containing literal-set match triples.
pub const LITERAL_SET_MATCHES_RESOURCE_INDEX: usize = 10;

/// Resident-resource index containing the mutable literal-set match counter.
pub const LITERAL_SET_RESET_RESOURCE_INDICES: [usize; 1] = [LITERAL_SET_MATCH_COUNT_RESOURCE_INDEX];

/// Resident-resource binding order for a prepared literal-set scan.
pub const LITERAL_SET_SCAN_RESOURCE_INDICES: [usize; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/// Resident-resource index containing the mutable literal-set count result.
pub const LITERAL_SET_COUNT_RESOURCE_INDEX: usize = 7;

/// Resident-resource index containing the mutable literal-set count result.
pub const LITERAL_SET_COUNT_RESET_RESOURCE_INDICES: [usize; 1] = [LITERAL_SET_COUNT_RESOURCE_INDEX];

/// Resident-resource binding order for a prepared literal-set count scan.
pub const LITERAL_SET_COUNT_SCAN_RESOURCE_INDICES: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

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
    /// Dispatch program construction failed for the compiled DFA.
    DispatchProgramBuildFailed {
        /// Actionable builder diagnostic.
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
            Self::DispatchProgramBuildFailed { message } => write!(
                f,
                "literal_set DFA dispatch program build failed: {message}"
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
    /// ASCII case-insensitive matching. When set, the `dfa` transition table is
    /// folded (`b'A'` behaves like `b'a'`) and the candidate prefilter masks are
    /// built to admit BOTH cases of each pattern byte. It is a compile-time
    /// property that must survive wire round-trips (the masks are rebuilt lazily
    /// from the raw `pattern_bytes`, which are identical for a case-sensitive and
    /// a case-insensitive set (so the flag, not the bytes, distinguishes them)).
    pub case_insensitive: bool,
}

/// Reusable hot-loop state for [`GpuLiteralSet`] scans.
///
/// This extends the generic scan dispatch scratch with a one-entry cache for
/// cap-specific `Program` layouts plus suffix-prefilter tables. Callers that
/// repeatedly scan with the same non-default `max_matches` avoid rebuilding the
/// rewritten output-buffer declaration and candidate masks on every dispatch.
#[derive(Debug, Default)]
pub struct LiteralSetScanScratch {
    /// Shared scan staging used by other matching engines.
    pub dispatch: ScanDispatchScratch,
    cached_program: Option<CachedLiteralSetProgram>,
    cached_count_program: Option<CachedLiteralSetCountProgram>,
    cached_prefilter: Option<LiteralSetPrefilterTables>,
}

/// Backend-neutral prepared literal-set scan payload.
///
/// This owns the exact byte buffers consumed by the GPU program. Callers with
/// resident-resource support can upload `inputs` once, append a zeroed output
/// resource sized from `matches_output_bytes`, reset
/// [`LITERAL_SET_RESET_RESOURCE_INDICES`], and dispatch
/// [`LITERAL_SET_SCAN_RESOURCE_INDICES`] without rebuilding the literal tables.
#[derive(Clone, Debug)]
pub struct LiteralSetPreparedScan {
    /// Cap-specific dispatch program for this scan.
    pub program: Program,
    /// Input buffers in program binding order, excluding the `matches` output.
    pub inputs: Vec<Vec<u8>>,
    /// Standard byte-scan dispatch geometry for `haystack_len`.
    pub dispatch_config: DispatchConfig,
    /// Validated haystack byte length.
    pub haystack_len: u32,
    /// Caller-provided output cap.
    pub max_matches: u32,
    /// Full resident output allocation size for the `matches` resource.
    pub matches_output_bytes: usize,
    /// Total bytes in `inputs`.
    pub encoded_input_bytes: u64,
}

impl LiteralSetPreparedScan {
    /// Byte length required to read the match counter.
    #[must_use]
    pub const fn match_count_readback_bytes(&self) -> usize {
        U32_COUNTER_BYTES
    }

    /// Byte length required to read up to `match_count` match triples.
    ///
    /// The returned range is clamped to `max_matches`, matching the decoder
    /// used by [`GpuLiteralSet::scan`].
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] when byte-size arithmetic overflows.
    pub fn match_triples_readback_bytes(
        &self,
        match_count: u32,
    ) -> Result<usize, vyre::BackendError> {
        literal_set_match_triple_bytes(match_count.min(self.max_matches))
    }

    /// Decode scan outputs into caller-owned match storage.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] when output buffers are missing,
    /// malformed, or too short for the reported match count.
    pub fn decode_outputs_into(
        &self,
        outputs: &[Vec<u8>],
        matches: &mut Vec<Match>,
    ) -> Result<(), vyre::BackendError> {
        decode_literal_set_outputs_into(outputs, self.max_matches, matches)
    }
}

/// Backend-neutral prepared literal-set count payload.
///
/// This is the count/presence sibling of [`LiteralSetPreparedScan`]: it keeps
/// the DFA and suffix-prefilter inputs in program binding order but returns
/// only the mutable `match_count` resource.
#[derive(Clone, Debug)]
pub struct LiteralSetPreparedCount {
    /// Count-only suffix-prefiltered dispatch program.
    pub program: Program,
    /// Input buffers in program binding order.
    pub inputs: Vec<Vec<u8>>,
    /// Standard byte-scan dispatch geometry for `haystack_len`.
    pub dispatch_config: DispatchConfig,
    /// Validated haystack byte length.
    pub haystack_len: u32,
    /// Total bytes in `inputs`.
    pub encoded_input_bytes: u64,
}

impl LiteralSetPreparedCount {
    /// Byte length required to read the count result.
    #[must_use]
    pub const fn count_readback_bytes(&self) -> usize {
        U32_COUNTER_BYTES
    }

    /// Decode the count output from either borrowed or resident dispatch.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] when the output slot is missing or too
    /// short for one `u32` counter.
    pub fn decode_outputs(&self, outputs: &[Vec<u8>]) -> Result<u32, vyre::BackendError> {
        decode_literal_set_count_outputs(outputs)
    }
}

/// Resident-resource index of the per-region presence read-write buffer in a
/// prepared region-presence dispatch: the one resource a resident runtime resets
/// (zeroes) before each scan and reads back after.
pub const LITERAL_SET_PRESENCE_BY_REGION_OUTPUT_RESOURCE_INDEX: usize = 6;

/// Backend-neutral prepared RESIDENT region-presence dispatch payload.
///
/// The presence sibling of [`LiteralSetPreparedScan`] / [`LiteralSetPreparedCount`].
/// Owns the exact byte buffers consumed by the region-presence program. A
/// resident runtime uploads the immutable DFA / suffix-prefilter tables ONCE,
/// resets [`LITERAL_SET_PRESENCE_BY_REGION_OUTPUT_RESOURCE_INDEX`], dispatches,
/// and reads back the per-region presence bitmap, re-uploading only the haystack
/// across the files of a corpus. A direct caller can dispatch `inputs` through a
/// borrowed-input backend and decode binding 0 via [`Self::decode_presence`].
#[derive(Clone, Debug)]
pub struct LiteralSetPreparedPresenceByRegion {
    /// Region-presence dispatch program.
    pub program: Program,
    /// Input buffers in program binding order (binding 6 is the zeroed per-region
    /// presence read-write resource = the whole output).
    pub inputs: Vec<Vec<u8>>,
    /// Standard byte-scan dispatch geometry for `haystack_len`.
    pub dispatch_config: DispatchConfig,
    /// Validated haystack byte length.
    pub haystack_len: u32,
    /// Number of coalesced regions (rows in the presence bitmap).
    pub region_count: u32,
    /// Total `u32` words in the per-region presence bitmap
    /// (`region_count * presence_bitmap_words(pattern_count)`).
    pub total_words: usize,
    /// Byte size of the binding-6 presence resource (`total_words * 4`): the
    /// reset + readback length for a resident dispatch.
    pub presence_output_bytes: usize,
    /// Total bytes in `inputs`.
    pub encoded_input_bytes: u64,
}

impl LiteralSetPreparedPresenceByRegion {
    /// Decode the per-region presence bitmap from a dispatch's output buffers:
    /// `region_count * presence_bitmap_words(pattern_count)` packed `u32` words,
    /// IDENTICAL to [`GpuLiteralSet::scan_presence_by_region`]'s return.
    /// `outputs[0]` is the presence-resource readback.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] when the output slot is missing or too short.
    pub fn decode_presence(&self, outputs: &[Vec<u8>]) -> Result<Vec<u32>, vyre::BackendError> {
        let presence_bytes = crate::scan::dispatch_io::try_output_bytes(
            outputs,
            0,
            "literal_set prepared presence_by_region",
        )?;
        Ok(decode_presence_words(presence_bytes, self.total_words))
    }
}

/// The seven corpus-invariant region-presence table byte buffers, at their program
/// binding positions. ONE source of truth for which region-presence tables are
/// immutable across a corpus: shared by [`GpuLiteralSet::build_presence_by_region_dispatch`]
/// (the borrowed / async / prepared paths) and
/// [`GpuLiteralSet::resident_presence_tables`] (the resident pipeline), so all four
/// paths encode byte-identical tables. Every buffer is built through the fail-closed
/// `copy_u32_words_as_le_bytes`.
struct PresenceImmutableTableBytes {
    /// Lane-major DFA transition table (binding 1).
    transitions: Vec<u8>,
    /// DFA output-offset table (binding 2).
    output_offsets: Vec<u8>,
    /// DFA output-record table (binding 3).
    output_records: Vec<u8>,
    /// Per-pattern length table (binding 4).
    pattern_lengths: Vec<u8>,
    /// Suffix prefilter end mask (binding 7).
    candidate_end_mask: Vec<u8>,
    /// Suffix prefilter 2-gram mask (binding 8).
    candidate_suffix2_mask: Vec<u8>,
    /// Suffix prefilter 3-gram bloom (binding 9).
    candidate_suffix3_bloom: Vec<u8>,
}

/// IMMUTABLE region-presence tables (corpus-invariant) plus a `max_regions`-sized
/// program, produced by [`GpuLiteralSet::resident_presence_tables`] and consumed
/// by [`ResidentPresencePipeline`](crate::scan::resident_presence::ResidentPresencePipeline).
///
/// Every field here is a function of the compiled matcher alone, none depends on
/// the haystack or region layout, so a resident session uploads them once and
/// re-dispatches across a whole corpus.
pub(crate) struct ResidentPresenceTables {
    /// Region-presence program sized for up to `max_regions` coalesced files.
    pub(crate) program: Program,
    /// Lane-major DFA transition table bytes (binding 1).
    pub(crate) transitions: Vec<u8>,
    /// DFA output-offset table bytes (binding 2).
    pub(crate) output_offsets: Vec<u8>,
    /// DFA output-record table bytes (binding 3).
    pub(crate) output_records: Vec<u8>,
    /// Per-pattern length table bytes (binding 4).
    pub(crate) pattern_lengths: Vec<u8>,
    /// Suffix prefilter end-mask bytes (binding 7).
    pub(crate) candidate_end_mask: Vec<u8>,
    /// Suffix prefilter 2-gram mask bytes (binding 8).
    pub(crate) candidate_suffix2_mask: Vec<u8>,
    /// Suffix prefilter 3-gram bloom bytes (binding 9).
    pub(crate) candidate_suffix3_bloom: Vec<u8>,
    /// Pattern count (bit width of each per-region presence row).
    pub(crate) pattern_count: u32,
    /// Presence bitmap `u32` words per region (`presence_bitmap_words(pattern_count)`).
    pub(crate) presence_words: u32,
    /// Program workgroup X extent, for the per-scan byte-scan dispatch geometry.
    pub(crate) workgroup_x: u32,
}

#[derive(Debug)]
struct CachedLiteralSetProgram {
    base_fingerprint: [u8; 32],
    max_matches: u32,
    program: Program,
}

#[derive(Debug)]
struct CachedLiteralSetCountProgram {
    pattern_fingerprint: u64,
    program: Program,
}

#[derive(Debug)]
struct LiteralSetPrefilterTables {
    pattern_fingerprint: u64,
    candidate_end_mask: [u32; 8],
    candidate_suffix2_mask: [u32; CLASSIC_AC_SUFFIX2_MASK_WORDS],
    candidate_suffix3_bloom: Vec<u32>,
}

/// Borrowed little-endian byte views of the DFA + prefilter `u32` tables, prepared
/// once and then referenced (in binding order) by a GPU dispatch's `borrowed_inputs`
/// array.
///
/// Four scan entry points: [`GpuLiteralSet::scan_presence`],
/// [`GpuLiteralSet::scan_presence_by_region_with_scratch`],
/// [`GpuLiteralSet::scan_presence_and_positions_by_region`] (via its scratch entry),
/// and `scan_into_with_program`: declared the IDENTICAL seven-view block inline.
/// This is the single source of that byte prep so the view set cannot drift between
/// the four kernels (a divergence would silently miswire one dispatch's bindings).
/// On little-endian hosts every view is a zero-copy borrow
/// ([`dispatch_io::u32_words_as_le_bytes`] → `bytemuck::cast_slice`); on big-endian
/// it owns a byte-swapped copy. The fields' lifetimes are tied to the source `dfa`,
/// `pattern_lengths`, and prefilter tables, so this struct must outlive the
/// `borrowed_inputs` array that references it.
///
/// `count_with_program` deliberately keeps its own NARROWER prep: the count-only
/// kernel never reads `output_records`/`pattern_lengths`, so building them there
/// would be misleading (it would imply the count kernel consumes them).
struct DfaPrefilterByteViews<'a> {
    transitions: Cow<'a, [u8]>,
    output_offsets: Cow<'a, [u8]>,
    output_records: Cow<'a, [u8]>,
    pattern_lengths: Cow<'a, [u8]>,
    candidate_end_mask: Cow<'a, [u8]>,
    candidate_suffix2_mask: Cow<'a, [u8]>,
    candidate_suffix3_bloom: Cow<'a, [u8]>,
}

impl<'a> DfaPrefilterByteViews<'a> {
    fn new(
        dfa: &'a CompiledDfa,
        pattern_lengths: &'a [u32],
        prefilter: &'a LiteralSetPrefilterTables,
    ) -> Self {
        use crate::scan::dispatch_io::u32_words_as_le_bytes;
        Self {
            transitions: u32_words_as_le_bytes(&dfa.transitions),
            output_offsets: u32_words_as_le_bytes(&dfa.output_offsets),
            output_records: u32_words_as_le_bytes(&dfa.output_records),
            pattern_lengths: u32_words_as_le_bytes(pattern_lengths),
            candidate_end_mask: u32_words_as_le_bytes(&prefilter.candidate_end_mask),
            candidate_suffix2_mask: u32_words_as_le_bytes(&prefilter.candidate_suffix2_mask),
            candidate_suffix3_bloom: u32_words_as_le_bytes(&prefilter.candidate_suffix3_bloom),
        }
    }
}

/// Result of [`GpuLiteralSet::scan_all_timed`]: the backend-owned timing plus a
/// flag stating WHICH dispatch that timing describes.
///
/// `scan_all` auto-resizes (up to two dispatches). `resized == false` means a
/// single dispatch produced the returned matches and `timed` is that dispatch's
/// timing; `resized == true` means the first pass saturated and `timed` is the
/// resize RE-dispatch's timing (the one whose output was decoded). The flag makes
/// the attribution explicit rather than silently reporting one launch's time as
/// though no resize occurred (Law 10).
#[derive(Debug)]
pub struct ScanAllTimed {
    /// Timing of the dispatch whose output produced the returned matches.
    pub timed: vyre_driver::TimedDispatchResult,
    /// `true` iff an auto-resize occurred and `timed` is the resize re-dispatch.
    pub resized: bool,
}

/// Resident literal-set POSITION session: uploads the immutable DFA + suffix
/// prefilter tables into backend resources ONCE, then re-dispatches the
/// `(pattern_id, start, end)` match scan across a corpus re-uploading only the
/// per-file haystack (and resetting the 4-byte match counter), the position-scan
/// sibling of [`ResidentPresencePipeline`], eliminating the multi-MiB per-scan
/// table re-upload the borrowed [`GpuLiteralSet::scan_into`] path repeats on every
/// file.
///
/// Construct with [`GpuLiteralSet::prepare_resident_scan`]. All eleven bindings are
/// resident (the CUDA resident dispatch rejects a borrowed-resource mix), including
/// the `matches` output resource, the resident dispatch resolves it as an output
/// and reads it back. The `matches` buffer is fixed-size (`max_matches` triples);
/// a scan that overflows fails CLOSED (never a silent truncated decode, Law 10).
pub struct ResidentLiteralScan {
    /// Match program sized for `max_matches` triples.
    program: Program,
    /// Resident haystack buffer, sized to `haystack_capacity` padded bytes.
    haystack: Resource,
    /// Resident DFA transition table (immutable, uploaded once).
    transitions: Resource,
    /// Resident DFA output-offset table (immutable, uploaded once).
    output_offsets: Resource,
    /// Resident DFA output-record table (immutable, uploaded once).
    output_records: Resource,
    /// Resident per-pattern length table (immutable, uploaded once).
    pattern_lengths: Resource,
    /// Resident haystack-length control buffer (1 u32; re-uploaded per scan).
    haystack_len_buf: Resource,
    /// Resident atomic match counter (1 u32; reset to 0 per scan).
    match_count_buf: Resource,
    /// Resident suffix prefilter end mask (immutable, uploaded once).
    candidate_end_mask: Resource,
    /// Resident suffix prefilter 2-gram mask (immutable, uploaded once).
    candidate_suffix2_mask: Resource,
    /// Resident suffix prefilter 3-gram bloom (immutable, uploaded once).
    candidate_suffix3_bloom: Resource,
    /// Resident match-output buffer (`max_matches × 3` u32; the read-back triples).
    matches_buf: Resource,
    /// Padded byte capacity of the resident haystack buffer.
    haystack_capacity: usize,
    /// Match cap this session's `matches` buffer was sized for.
    max_matches: u32,
    /// Program workgroup X extent, for the per-scan byte-scan dispatch geometry.
    workgroup_x: u32,
}

// SAFETY mirror of the `ResidentPresencePipeline` contract: `Resource` handles are
// plain ids and `Program` is `Send + Sync`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ResidentLiteralScan>;
};

/// Allocate a resident buffer sized to `bytes` and upload them once. The
/// position-session analogue of `resident_presence`'s `allocate_and_upload`.
fn allocate_and_upload_resident(
    backend: &dyn VyreBackend,
    bytes: &[u8],
) -> Result<Resource, vyre::BackendError> {
    let resource = backend.allocate_resident(bytes.len())?;
    backend.upload_resident(&resource, bytes)?;
    Ok(resource)
}

impl GpuLiteralSet {
    /// Prepare a RESIDENT position-scan session: upload the immutable DFA +
    /// suffix-prefilter tables into backend resources ONCE, sized for a haystack up
    /// to `haystack_capacity_bytes` and a fixed `max_matches` triple cap. Each
    /// [`ResidentLiteralScan::scan_into`] then re-uploads only the haystack and
    /// resets the 4-byte counter, the position-scan sibling of
    /// [`Self::prepare_resident_presence`].
    ///
    /// The `matches` output is resident and fixed-size; a batch with more than
    /// `max_matches` matches fails CLOSED at scan time (never a silent partial).
    /// Callers that must not fail on overflow use the borrowed auto-resizing
    /// [`Self::scan_all`] instead.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if the backend cannot allocate/upload resident
    /// resources, or if the program/table sizing overflows the GPU ABI.
    pub fn prepare_resident_scan(
        &self,
        backend: &dyn VyreBackend,
        haystack_capacity_bytes: usize,
        max_matches: u32,
    ) -> Result<ResidentLiteralScan, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let program = self.program_for_match_capacity(max_matches)?.into_owned();
        let prefilter_tables = self.build_prefilter_tables()?;
        let tables = self.presence_immutable_table_bytes(&prefilter_tables)?;
        let (_declared_words, matches_output_bytes) = literal_set_match_output_layout(max_matches)?;

        let haystack_capacity = dispatch_io::haystack_padded_u32_byte_len(haystack_capacity_bytes)?;
        let haystack = backend.allocate_resident(haystack_capacity)?;

        // The seven immutable tables: allocate + upload ONCE.
        let transitions = allocate_and_upload_resident(backend, &tables.transitions)?;
        let output_offsets = allocate_and_upload_resident(backend, &tables.output_offsets)?;
        let output_records = allocate_and_upload_resident(backend, &tables.output_records)?;
        let pattern_lengths = allocate_and_upload_resident(backend, &tables.pattern_lengths)?;
        let candidate_end_mask = allocate_and_upload_resident(backend, &tables.candidate_end_mask)?;
        let candidate_suffix2_mask =
            allocate_and_upload_resident(backend, &tables.candidate_suffix2_mask)?;
        let candidate_suffix3_bloom =
            allocate_and_upload_resident(backend, &tables.candidate_suffix3_bloom)?;

        // Per-scan control + output buffers (all resident (no borrowed mix)).
        let haystack_len_buf = backend.allocate_resident(U32_BYTES)?;
        let match_count_buf = backend.allocate_resident(U32_BYTES)?;
        let matches_buf = backend.allocate_resident(matches_output_bytes)?;

        Ok(ResidentLiteralScan {
            workgroup_x: program.workgroup_size[0],
            program,
            haystack,
            transitions,
            output_offsets,
            output_records,
            pattern_lengths,
            haystack_len_buf,
            match_count_buf,
            candidate_end_mask,
            candidate_suffix2_mask,
            candidate_suffix3_bloom,
            matches_buf,
            haystack_capacity,
            max_matches,
        })
    }
}

impl ResidentLiteralScan {
    /// Scan `haystack` against the resident session, decoding the `(pattern_id,
    /// start, end)` triples into caller-owned `matches` (cleared first). IDENTICAL
    /// output to [`GpuLiteralSet::scan_into`] with the same `max_matches`. `scratch`
    /// reuses the packed-haystack staging across scans.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure, if `haystack`
    /// exceeds the resident capacity, or if the match count exceeds `max_matches`
    /// (fail closed (never a silent truncated decode)).
    pub fn scan_into(
        &self,
        backend: &dyn VyreBackend,
        haystack: &[u8],
        matches: &mut Vec<Match>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), vyre::BackendError> {
        self.scan_into_timed(backend, haystack, matches, scratch)
            .map(|_timed| ())
    }

    /// [`Self::scan_into`] returning the backend-owned dispatch timing
    /// ([`vyre_driver::TimedDispatchResult`]) so a consumer can attribute the
    /// resident scan's GPU-kernel time separately from host staging/readback 
    /// matching [`ResidentPresencePipeline::scan_into_timed`].
    ///
    /// # Errors
    /// See [`Self::scan_into`].
    pub fn scan_into_timed(
        &self,
        backend: &dyn VyreBackend,
        haystack: &[u8],
        matches: &mut Vec<Match>,
        scratch: &mut Vec<u8>,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre::BackendError> {
        use crate::scan::dispatch_io;

        matches.clear();
        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "ResidentLiteralScan::scan",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;

        // Stage the haystack into the resident buffer (real bytes only; the kernel
        // bounds its cursor with haystack_len so the stale tail is never read).
        dispatch_io::pack_haystack_u32_into(haystack, scratch)?;
        if scratch.len() > self.haystack_capacity {
            return Err(vyre::BackendError::new(format!(
                "ResidentLiteralScan haystack is {} packed byte(s) but the resident buffer holds {}. Fix: raise haystack_capacity_bytes in prepare_resident_scan or shard the haystack.",
                scratch.len(),
                self.haystack_capacity
            )));
        }
        backend.upload_resident_at(&self.haystack, 0, scratch)?;

        // Reset only the atomic match counter (binding 6). Triples are written from
        // slot 0 upward and only `count` are read back, so stale triples beyond the
        // new count are never observed (a 4-byte reset, not a full buffer clear).
        backend.upload_resident_at(&self.match_count_buf, 0, &0u32.to_le_bytes())?;
        backend.upload_resident_at(&self.haystack_len_buf, 0, &haystack_len.to_le_bytes())?;

        // Bind in program (BufferDecl) order, every binding resident. This is the
        // literal MATCH program's 11-binding order (0..=10); binding 6 (match_count,
        // read_write) and binding 10 (matches, output) are the two read-back buffers.
        let resources = [
            self.haystack.clone(),                // 0: haystack (Packed U32)
            self.transitions.clone(),             // 1: transitions
            self.output_offsets.clone(),          // 2: output_offsets
            self.output_records.clone(),          // 3: output_records
            self.pattern_lengths.clone(),         // 4: pattern_lengths
            self.haystack_len_buf.clone(),        // 5: haystack_len
            self.match_count_buf.clone(),         // 6: match_count (read_write)
            self.candidate_end_mask.clone(),      // 7: candidate_end_mask
            self.candidate_suffix2_mask.clone(),  // 8: candidate_suffix2_mask
            self.candidate_suffix3_bloom.clone(), // 9: candidate_suffix3_bloom
            self.matches_buf.clone(),             // 10: matches (output)
        ];

        let config = dispatch_io::byte_scan_dispatch_config(haystack_len, self.workgroup_x);
        let timed = backend.dispatch_resident_timed(&self.program, &resources, &config)?;

        // Output ordering = read_write then output by binding: match_count(6) ->
        // outputs[0], matches(10) -> outputs[1], the exact shape the borrowed match
        // dispatch produces, so the ONE-PLACE sync decoder applies unchanged. The
        // capped decoder fails closed if the device count exceeds max_matches.
        decode_literal_set_outputs_into(&timed.outputs, self.max_matches, matches)?;
        Ok(timed)
    }

    /// The match cap this session's resident `matches` buffer was sized for.
    #[must_use]
    pub fn max_matches(&self) -> u32 {
        self.max_matches
    }

    /// Padded byte capacity of the resident haystack buffer.
    #[must_use]
    pub fn haystack_capacity(&self) -> usize {
        self.haystack_capacity
    }

    /// Free every resident resource this session allocated. Attempts all frees and
    /// returns the first error; the session is consumed.
    ///
    /// # Errors
    /// Returns the first [`vyre::BackendError`] from freeing a resource.
    pub fn free(self, backend: &dyn VyreBackend) -> Result<(), vyre::BackendError> {
        let mut first_err = None;
        for resource in [
            self.haystack,
            self.transitions,
            self.output_offsets,
            self.output_records,
            self.pattern_lengths,
            self.haystack_len_buf,
            self.match_count_buf,
            self.candidate_end_mask,
            self.candidate_suffix2_mask,
            self.candidate_suffix3_bloom,
            self.matches_buf,
        ] {
            if let Err(error) = backend.free_resident(resource) {
                first_err.get_or_insert(error);
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// A RESIDENT session for the FUSED per-region presence + positions scan
/// ([`GpuLiteralSet::scan_presence_and_positions_by_region`]).
///
/// Construct with [`GpuLiteralSet::prepare_resident_fused_scan`]. It is the
/// fusion of [`ResidentPresencePipeline`] (the per-region presence bitmap +
/// region controls) and [`ResidentLiteralScan`] (the positioned match output):
/// one all-resident dispatch of the fused program produces BOTH the per-region
/// presence bitmap AND the `(pattern_id, start, end)` triples, uploading the
/// immutable DFA + suffix-prefilter tables ONCE and re-staging only the haystack,
/// the region controls, and two zeroed accumulators (presence prefix + match
/// counter) per scan.
///
/// All 14 bindings (0..=13) are resident, including the two read-write
/// accumulators (presence at 6, match_count at 12) and the `matches` output at
/// 13, the CUDA resident dispatch rejects a borrowed mix. The `matches` buffer
/// is fixed-size (`max_matches` triples); a scan that overflows fails CLOSED
/// (never a silent truncated decode, Law 10).
pub struct ResidentFusedRegionScan {
    /// Fused program sized for `max_regions` files and `max_matches` triples.
    program: Program,
    haystack: Resource,
    transitions: Resource,
    output_offsets: Resource,
    output_records: Resource,
    pattern_lengths: Resource,
    haystack_len_buf: Resource,
    /// Resident per-region presence buffer (read-write; used prefix reset per scan).
    presence: Resource,
    candidate_end_mask: Resource,
    candidate_suffix2_mask: Resource,
    candidate_suffix3_bloom: Resource,
    /// Resident region-start offsets (sized for `max_regions`; padded per scan).
    region_starts_buf: Resource,
    /// Resident shard base offset control (1 u32; re-uploaded per scan).
    region_base_buf: Resource,
    /// Resident atomic match counter (1 u32; reset to 0 per scan).
    match_count_buf: Resource,
    /// Resident match-output buffer (`max_matches × 3` u32; the read-back triples).
    matches_buf: Resource,
    /// Padded byte capacity of the resident haystack buffer.
    haystack_capacity: usize,
    /// Largest coalesced-file count the presence/region buffers were sized for.
    max_regions: u32,
    /// Presence bitmap `u32` words per region.
    presence_words: u32,
    /// Match cap this session's `matches` buffer was sized for.
    max_matches: u32,
    /// Program workgroup X extent, for the per-scan byte-scan dispatch geometry.
    workgroup_x: u32,
}

// SAFETY mirror of the sibling resident pipelines: `Resource` handles are plain
// ids and `Program` is `Send + Sync`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ResidentFusedRegionScan>;
};

impl GpuLiteralSet {
    /// Prepare a RESIDENT session for the FUSED per-region presence + positions
    /// scan: upload the immutable DFA + suffix-prefilter tables ONCE, sized for a
    /// haystack up to `haystack_capacity_bytes`, up to `max_regions` coalesced
    /// files, and a fixed `max_matches` triple cap. Each
    /// [`ResidentFusedRegionScan::scan_into`] then re-uploads only the haystack,
    /// the region controls, and the two zeroed accumulators, the fused sibling of
    /// [`Self::prepare_resident_presence`] and [`Self::prepare_resident_scan`].
    ///
    /// The `matches` output is resident and fixed-size; a batch with more than
    /// `max_matches` matches fails CLOSED at scan time (never a silent partial).
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if `max_regions` is zero, if the backend
    /// cannot allocate/upload resident resources, or if the program/table sizing
    /// overflows the GPU ABI.
    pub fn prepare_resident_fused_scan(
        &self,
        backend: &dyn VyreBackend,
        haystack_capacity_bytes: usize,
        max_regions: u32,
        max_matches: u32,
    ) -> Result<ResidentFusedRegionScan, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let pattern_count = u32::try_from(self.pattern_lengths.len()).map_err(|_| {
            vyre::BackendError::new(
                "literal_set fused resident scan: pattern count exceeds u32 GPU ABI".to_string(),
            )
        })?;
        if max_regions == 0 {
            return Err(vyre::BackendError::new(
                "literal_set resident fused scan: max_regions must be >= 1 (it sizes the resident presence buffer and the kernel's region binary-search width). Fix: pass the largest coalesced-batch file count the session will scan.".to_string(),
            ));
        }
        let program = try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
            &self.dfa,
            pattern_count,
            max_regions,
            max_matches,
        )
        .map_err(vyre::BackendError::new)?;
        let prefilter_tables = self.build_prefilter_tables()?;
        let tables = self.presence_immutable_table_bytes(&prefilter_tables)?;
        let presence_words = presence_bitmap_words(pattern_count);
        let (_declared_words, matches_output_bytes) = literal_set_match_output_layout(max_matches)?;

        let haystack_capacity = dispatch_io::haystack_padded_u32_byte_len(haystack_capacity_bytes)?;
        let haystack = backend.allocate_resident(haystack_capacity)?;

        // The seven immutable tables: allocate + upload ONCE.
        let transitions = allocate_and_upload_resident(backend, &tables.transitions)?;
        let output_offsets = allocate_and_upload_resident(backend, &tables.output_offsets)?;
        let output_records = allocate_and_upload_resident(backend, &tables.output_records)?;
        let pattern_lengths = allocate_and_upload_resident(backend, &tables.pattern_lengths)?;
        let candidate_end_mask = allocate_and_upload_resident(backend, &tables.candidate_end_mask)?;
        let candidate_suffix2_mask =
            allocate_and_upload_resident(backend, &tables.candidate_suffix2_mask)?;
        let candidate_suffix3_bloom =
            allocate_and_upload_resident(backend, &tables.candidate_suffix3_bloom)?;

        // Read-write presence buffer sized for the full max_regions capacity.
        let presence_capacity_words = (max_regions as usize)
            .checked_mul(presence_words as usize)
            .ok_or_else(|| {
                vyre::BackendError::new(format!(
                    "resident fused scan capacity {max_regions} regions × {presence_words} words/region overflows host usize. Fix: lower max_regions or shard the pattern set."
                ))
            })?;
        let presence_capacity_bytes =
            presence_capacity_words.checked_mul(U32_BYTES).ok_or_else(|| {
                vyre::BackendError::new(
                    "resident fused scan presence-buffer byte capacity overflows host usize. Fix: lower max_regions or shard the pattern set.".to_string(),
                )
            })?;
        let presence = backend.allocate_resident(presence_capacity_bytes)?;

        // Per-scan control + output buffers (all resident (no borrowed mix)).
        let haystack_len_buf = backend.allocate_resident(U32_BYTES)?;
        let region_starts_capacity_bytes =
            (max_regions as usize).checked_mul(U32_BYTES).ok_or_else(|| {
                vyre::BackendError::new(
                    "resident fused scan region-starts byte capacity overflows host usize. Fix: lower max_regions.".to_string(),
                )
            })?;
        let region_starts_buf = backend.allocate_resident(region_starts_capacity_bytes)?;
        let region_base_buf = backend.allocate_resident(U32_BYTES)?;
        let match_count_buf = backend.allocate_resident(U32_BYTES)?;
        let matches_buf = backend.allocate_resident(matches_output_bytes)?;

        Ok(ResidentFusedRegionScan {
            workgroup_x: program.workgroup_size[0],
            program,
            haystack,
            transitions,
            output_offsets,
            output_records,
            pattern_lengths,
            haystack_len_buf,
            presence,
            candidate_end_mask,
            candidate_suffix2_mask,
            candidate_suffix3_bloom,
            region_starts_buf,
            region_base_buf,
            match_count_buf,
            matches_buf,
            haystack_capacity,
            max_regions,
            presence_words,
            max_matches,
        })
    }
}

impl ResidentFusedRegionScan {
    /// Scan a coalesced batch (`region_starts` ascending, beginning at 0) against
    /// the resident session, decoding the per-region presence bitmap into `out`
    /// and the `(pattern_id, start, end)` triples into `matches`: IDENTICAL output
    /// to [`GpuLiteralSet::scan_presence_and_positions_by_region`] with the same
    /// `max_matches`. `scratch` reuses the packed-haystack / reset staging across
    /// scans.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure, if
    /// `region_starts` is empty / does not begin at 0, if `region_count` exceeds
    /// `max_regions`, if `haystack` exceeds the resident capacity, or if the match
    /// count exceeds `max_matches` (fail closed (never a silent truncated decode)).
    #[allow(clippy::too_many_arguments)]
    pub fn scan_into(
        &self,
        backend: &dyn VyreBackend,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        out: &mut Vec<u32>,
        matches: &mut Vec<Match>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), vyre::BackendError> {
        self.scan_into_timed(
            backend,
            haystack,
            region_starts,
            region_base,
            out,
            matches,
            scratch,
        )
        .map(|_timed| ())
    }

    /// [`Self::scan_into`] returning the backend-owned dispatch timing
    /// ([`vyre_driver::TimedDispatchResult`]) so a consumer can attribute the fused
    /// resident scan's GPU-kernel time separately from host staging/readback.
    ///
    /// # Errors
    /// See [`Self::scan_into`].
    #[allow(clippy::too_many_arguments)]
    pub fn scan_into_timed(
        &self,
        backend: &dyn VyreBackend,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        out: &mut Vec<u32>,
        matches: &mut Vec<Match>,
        scratch: &mut Vec<u8>,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre::BackendError> {
        use crate::scan::dispatch_io;

        out.clear();
        matches.clear();

        let region_count = u32::try_from(region_starts.len()).map_err(|_| {
            vyre::BackendError::new(
                "resident fused scan: region count exceeds u32 GPU ABI".to_string(),
            )
        })?;
        if region_count == 0 {
            return Err(vyre::BackendError::new(
                "resident fused scan: region_starts must be non-empty. Fix: pass one start offset per coalesced file, beginning with 0.".to_string(),
            ));
        }
        if region_starts[0] != 0 {
            return Err(vyre::BackendError::new(
                "resident fused scan: region_starts[0] must be 0 (the kernel binary-search lower bound). Fix: the first coalesced file must start at offset 0.".to_string(),
            ));
        }
        if region_count > self.max_regions {
            return Err(vyre::BackendError::new(format!(
                "resident fused scan batch has {region_count} regions but the session was prepared for at most {}. Fix: raise max_regions in prepare_resident_fused_scan, or dispatch this batch through the per-batch-sized borrowed GpuLiteralSet::scan_presence_and_positions_by_region.",
                self.max_regions
            )));
        }

        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "ResidentFusedRegionScan::scan",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;

        // (1) Stage the haystack (real bytes only; the kernel bounds its cursor
        // with haystack_len so the stale tail is never read).
        dispatch_io::pack_haystack_u32_into(haystack, scratch)?;
        if scratch.len() > self.haystack_capacity {
            return Err(vyre::BackendError::new(format!(
                "ResidentFusedRegionScan haystack is {} packed byte(s) but the resident buffer holds {}. Fix: raise haystack_capacity_bytes in prepare_resident_fused_scan or shard the haystack.",
                scratch.len(),
                self.haystack_capacity
            )));
        }
        backend.upload_resident_at(&self.haystack, 0, scratch)?;

        // (2) Zero the USED prefix of the resident presence buffer (binding 6 is
        // OR-accumulated, so it must arrive zeroed). Rows beyond region_count are
        // never written and never read. `scratch` reuse is safe (synchronous
        // upload copy).
        let used_words = (region_count as usize)
            .checked_mul(self.presence_words as usize)
            .ok_or_else(|| {
                vyre::BackendError::new(
                    "resident fused scan used-word count overflows host usize. Fix: lower the region count or shard the pattern set.".to_string(),
                )
            })?;
        let reset_bytes = used_words.checked_mul(U32_BYTES).ok_or_else(|| {
            vyre::BackendError::new(
                "resident fused scan presence-reset byte count overflows host usize. Fix: lower the region count or shard the pattern set.".to_string(),
            )
        })?;
        scratch.clear();
        scratch.resize(reset_bytes, 0);
        backend.upload_resident_at(&self.presence, 0, scratch)?;

        // (3) Per-scan control buffers (all resident). haystack_len, region_base,
        // and the 4-byte match_count reset are each one u32.
        backend.upload_resident_at(&self.haystack_len_buf, 0, &haystack_len.to_le_bytes())?;
        backend.upload_resident_at(&self.region_base_buf, 0, &region_base.to_le_bytes())?;
        backend.upload_resident_at(&self.match_count_buf, 0, &0u32.to_le_bytes())?;

        // region_starts padded to the fixed max_regions width with u32::MAX (a
        // sentinel strictly greater than any candidate position, so the region
        // binary search never maps a hit to a padding row). `scratch` reuse safe.
        scratch.clear();
        let region_starts_words = self.max_regions as usize;
        scratch.reserve(region_starts_words.saturating_mul(U32_BYTES));
        for &start in region_starts {
            scratch.extend_from_slice(&start.to_le_bytes());
        }
        for _ in (region_count as usize)..region_starts_words {
            scratch.extend_from_slice(&u32::MAX.to_le_bytes());
        }
        backend.upload_resident_at(&self.region_starts_buf, 0, scratch)?;

        // (4) Bind in program (BufferDecl) order, every binding resident. This is
        // the fused program's 14-binding order (0..=13); the read-back buffers are
        // presence(6, read_write), match_count(12, read_write) and matches(13,
        // output).
        let resources = [
            self.haystack.clone(),                // 0: haystack (Packed U32)
            self.transitions.clone(),             // 1: transitions
            self.output_offsets.clone(),          // 2: output_offsets
            self.output_records.clone(),          // 3: output_records
            self.pattern_lengths.clone(),         // 4: pattern_lengths
            self.haystack_len_buf.clone(),        // 5: haystack_len
            self.presence.clone(),                // 6: presence (read_write)
            self.candidate_end_mask.clone(),      // 7: candidate_end_mask
            self.candidate_suffix2_mask.clone(),  // 8: candidate_suffix2_mask
            self.candidate_suffix3_bloom.clone(), // 9: candidate_suffix3_bloom
            self.region_starts_buf.clone(),       // 10: region_starts (padded)
            self.region_base_buf.clone(),         // 11: region_base
            self.match_count_buf.clone(),         // 12: match_count (read_write)
            self.matches_buf.clone(),             // 13: matches (output)
        ];

        let config = dispatch_io::byte_scan_dispatch_config(haystack_len, self.workgroup_x);
        let timed = backend.dispatch_resident_timed(&self.program, &resources, &config)?;

        // Output ordering = read_write then output by binding: presence(6) ->
        // outputs[0], match_count(12) -> outputs[1], matches(13) -> outputs[2] 
        // the exact shape the borrowed fused dispatch produces.
        let presence_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            0,
            "ResidentFusedRegionScan presence buffer",
        )?;
        decode_presence_words_into(presence_bytes, used_words, out);
        if out.len() != used_words {
            let returned = out.len();
            out.clear();
            return Err(vyre::BackendError::new(format!(
                "ResidentFusedRegionScan presence readback returned {returned} u32 word(s) but the {region_count}-region scan needs {used_words}. Fix: ensure the backend reads back the full binding-6 presence resource."
            )));
        }

        let count_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            1,
            "ResidentFusedRegionScan match count",
        )?;
        let count =
            dispatch_io::try_read_u32_prefix(count_bytes, "ResidentFusedRegionScan match count")?;
        let matches_bytes =
            dispatch_io::try_output_bytes(&timed.outputs, 2, "ResidentFusedRegionScan matches")?;
        // Capped decode: fail closed if the device count exceeds max_matches (the
        // fixed resident matches buffer cannot hold a truncated decode. Law 10).
        dispatch_io::try_unpack_match_triples_capped_into(
            matches_bytes,
            count,
            self.max_matches,
            "ResidentFusedRegionScan matches",
            matches,
        )?;
        Ok(timed)
    }

    /// Largest coalesced-file count this session's presence buffer was sized for.
    #[must_use]
    pub fn max_regions(&self) -> u32 {
        self.max_regions
    }

    /// The match cap this session's resident `matches` buffer was sized for.
    #[must_use]
    pub fn max_matches(&self) -> u32 {
        self.max_matches
    }

    /// Padded byte capacity of the resident haystack buffer.
    #[must_use]
    pub fn haystack_capacity(&self) -> usize {
        self.haystack_capacity
    }

    /// Free every resident resource this session allocated. Attempts all frees and
    /// returns the first error; the session is consumed.
    ///
    /// # Errors
    /// Returns the first [`vyre::BackendError`] from freeing a resource.
    pub fn free(self, backend: &dyn VyreBackend) -> Result<(), vyre::BackendError> {
        let mut first_err = None;
        for resource in [
            self.haystack,
            self.transitions,
            self.output_offsets,
            self.output_records,
            self.pattern_lengths,
            self.haystack_len_buf,
            self.presence,
            self.candidate_end_mask,
            self.candidate_suffix2_mask,
            self.candidate_suffix3_bloom,
            self.region_starts_buf,
            self.region_base_buf,
            self.match_count_buf,
            self.matches_buf,
        ] {
            if let Err(error) = backend.free_resident(resource) {
                first_err.get_or_insert(error);
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// In-flight handle for [`GpuLiteralSet::scan_presence_by_region_async`].
///
/// Returned the moment the GPU dispatch is submitted, so the caller can run
/// host-side work concurrently with the device scan, then decode the per-region
/// presence bitmap with [`Self::await_words`]. The owned input buffers the scan
/// was submitted with are retained here (never read again on the host) so their
/// backing memory stays valid for the whole dispatch on backends whose async
/// upload reads host memory after submit returns.
pub struct PendingPresenceByRegion {
    pending: Box<dyn PendingDispatch>,
    total_words: usize,
    // Owned dispatch inputs kept alive until `await_words`. Retained purely so
    // the device-side async upload's backing memory remains valid for the whole
    // dispatch; never read again on the host.
    _inputs: Vec<Vec<u8>>,
}

impl PendingPresenceByRegion {
    /// Non-blocking readiness probe. `true` means [`Self::await_words`] will not
    /// block the caller thread. Backends that cannot probe without cost report
    /// `true` unconditionally (the caller then blocks inside `await_words`). See
    /// [`vyre::backend::PendingDispatch::is_ready`].
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.pending.is_ready()
    }

    /// Block until the GPU scan completes and decode the per-region presence
    /// bitmap: `region_count × presence_bitmap_words(pattern_count)` packed `u32`
    /// words, IDENTICAL to [`GpuLiteralSet::scan_presence_by_region`]'s return
    /// (bit `p` of region `r`'s row is set iff pattern `p`'s literal occurs in
    /// region `r`). Calling this when [`Self::is_ready`] is `true` does not block.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure.
    pub fn await_words(self) -> Result<Vec<u32>, vyre::BackendError> {
        let outputs = self.pending.await_result()?;
        let presence_bytes = crate::scan::dispatch_io::try_output_bytes(
            &outputs,
            0,
            "literal_set presence_by_region async",
        )?;
        Ok(decode_presence_words(presence_bytes, self.total_words))
    }
}

/// In-flight handle for [`GpuLiteralSet::scan_presence_async`].
///
/// The global-presence sibling of [`PendingPresenceByRegion`]: returned the
/// moment the GPU dispatch is submitted so the caller can overlap host-side work
/// with the device scan, then decode the whole-haystack presence bitmap with
/// [`Self::await_words`]. The owned dispatch inputs are retained (never read
/// again on the host) so their backing memory stays valid for the whole dispatch
/// on backends whose async upload reads host memory after submit returns.
pub struct PendingPresence {
    pending: Box<dyn PendingDispatch>,
    presence_words: usize,
    // Owned dispatch inputs kept alive until `await_words`; see the field note on
    // [`PendingPresenceByRegion`].
    _inputs: Vec<Vec<u8>>,
}

impl PendingPresence {
    /// Non-blocking readiness probe. `true` means [`Self::await_words`] will not
    /// block the caller thread. See [`vyre::backend::PendingDispatch::is_ready`].
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.pending.is_ready()
    }

    /// Block until the GPU scan completes and decode the global presence bitmap:
    /// `presence_bitmap_words(pattern_count)` packed `u32` words, IDENTICAL to
    /// [`GpuLiteralSet::scan_presence`]'s return (bit `p` is set iff pattern `p`'s
    /// literal occurs anywhere in the haystack). Calling this when
    /// [`Self::is_ready`] is `true` does not block.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure.
    pub fn await_words(self) -> Result<Vec<u32>, vyre::BackendError> {
        let outputs = self.pending.await_result()?;
        let presence_bytes =
            crate::scan::dispatch_io::try_output_bytes(&outputs, 0, "literal_set presence async")?;
        Ok(decode_presence_words(presence_bytes, self.presence_words))
    }
}

/// In-flight handle for [`GpuLiteralSet::scan_into_async`].
///
/// The position-scan sibling of [`PendingPresence`]: returned the moment the GPU
/// match dispatch is submitted so the caller can overlap host-side work with the
/// device scan, then decode the `(pattern_id, start, end)` match triples with
/// [`Self::await_into`] / [`Self::await_matches`]. The retained prepared payload
/// both backs the async upload's owned inputs (kept valid for the whole dispatch)
/// and carries the `max_matches` cap the decode clamps to, the same fail-closed
/// truncation contract as the synchronous [`GpuLiteralSet::scan_into`].
pub struct PendingMatches {
    pending: Box<dyn PendingDispatch>,
    // Retained so (a) the owned input buffers backing the async upload stay valid
    // for the whole dispatch and (b) `decode_outputs_into` has the max_matches cap.
    prepared: LiteralSetPreparedScan,
}

impl PendingMatches {
    /// Non-blocking readiness probe. `true` means [`Self::await_into`] will not
    /// block the caller thread. See [`vyre::backend::PendingDispatch::is_ready`].
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.pending.is_ready()
    }

    /// Block until the GPU scan completes and decode the match triples into
    /// caller-owned `matches` (cleared first), IDENTICAL to
    /// [`GpuLiteralSet::scan_into`]'s output. Fails closed if the device match
    /// count exceeds the prepared `max_matches` (never a silent truncated decode,
    /// Law 10). Calling this when [`Self::is_ready`] is `true` does not block.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure or match-count
    /// overflow.
    pub fn await_into(self, matches: &mut Vec<Match>) -> Result<(), vyre::BackendError> {
        matches.clear();
        let outputs = self.pending.await_result()?;
        self.prepared.decode_outputs_into(&outputs, matches)
    }

    /// [`Self::await_into`] returning a freshly allocated match vector.
    ///
    /// # Errors
    /// See [`Self::await_into`].
    pub fn await_matches(self) -> Result<Vec<Match>, vyre::BackendError> {
        let mut matches = Vec::new();
        self.await_into(&mut matches)?;
        Ok(matches)
    }
}

/// In-flight handle for [`GpuLiteralSet::scan_presence_and_positions_by_region_async`].
///
/// The fused-scan sibling of [`PendingPresenceByRegion`] and [`PendingMatches`]:
/// one submitted dispatch yields BOTH the per-region presence bitmap (returned by
/// [`Self::await_into`]) AND the `(pattern_id, start, end)` match triples (decoded
/// into the caller's buffer). The owned inputs are retained so the async upload's
/// backing memory stays valid, and `max_matches` is carried so the decode keeps
/// the same fail-closed overflow contract as the synchronous fused scan.
pub struct PendingFusedRegion {
    pending: Box<dyn PendingDispatch>,
    total_words: usize,
    max_matches: u32,
    // Owned dispatch inputs kept alive until the await; see the field note on
    // [`PendingPresenceByRegion`].
    _inputs: Vec<Vec<u8>>,
}

impl PendingFusedRegion {
    /// Non-blocking readiness probe. See
    /// [`vyre::backend::PendingDispatch::is_ready`].
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.pending.is_ready()
    }

    /// Block until the GPU scan completes, decode the `(pattern_id, start, end)`
    /// triples into caller-owned `matches` (cleared first), and RETURN the
    /// per-region presence bitmap, both IDENTICAL to
    /// [`GpuLiteralSet::scan_presence_and_positions_by_region`]'s outputs. Fails
    /// closed if the device match count exceeds the prepared `max_matches` (never a
    /// silent truncated decode, Law 10). Calling this when [`Self::is_ready`] is
    /// `true` does not block.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure or match-count
    /// overflow.
    pub fn await_into(self, matches: &mut Vec<Match>) -> Result<Vec<u32>, vyre::BackendError> {
        use crate::scan::dispatch_io;
        matches.clear();
        let outputs = self.pending.await_result()?;
        // presence(6)->outputs[0], match_count(12)->outputs[1], matches(13)->outputs[2].
        let presence_bytes = dispatch_io::try_output_bytes(
            &outputs,
            0,
            "literal_set presence_and_positions_by_region async presence",
        )?;
        let presence = decode_presence_words(presence_bytes, self.total_words);
        let count_bytes = dispatch_io::try_output_bytes(
            &outputs,
            1,
            "literal_set presence_and_positions_by_region async match count",
        )?;
        let count = dispatch_io::try_read_u32_prefix(
            count_bytes,
            "literal_set presence_and_positions_by_region async match count",
        )?;
        let matches_bytes = dispatch_io::try_output_bytes(
            &outputs,
            2,
            "literal_set presence_and_positions_by_region async matches",
        )?;
        dispatch_io::try_unpack_match_triples_capped_into(
            matches_bytes,
            count,
            self.max_matches,
            "literal_set presence_and_positions_by_region async matches",
            matches,
        )?;
        Ok(presence)
    }
}

impl GpuLiteralSet {
    /// Compile a set of literal patterns into a GPU-ready matcher.
    ///
    /// # Panics
    ///
    /// Aborts when staging allocation fails or a pattern count/length cannot be
    /// represented by the GPU ABI. Returning an empty matcher would silently
    /// match NOTHING, reporting every input as clean (Law 10). Fail closed
    /// instead; callers that must recover use [`Self::try_compile`].
    #[must_use]
    pub fn compile(patterns: &[&[u8]]) -> Self {
        Self::compile_folded(patterns, false)
    }

    /// ASCII-CASE-INSENSITIVE counterpart of [`Self::compile`]: `A`/`a` … `Z`/`z`
    /// match interchangeably. The fold is baked into the DFA transition table and
    /// the candidate prefilter masks at compile time, so the scan matches
    /// mixed-case input with ZERO per-byte host folding and no second resident
    /// haystack copy (it replaces the consumer's `to_ascii_lowercase` pass).
    ///
    /// # Panics
    /// See [`Self::compile`].
    #[must_use]
    pub fn compile_case_insensitive(patterns: &[&[u8]]) -> Self {
        Self::compile_folded(patterns, true)
    }

    fn compile_folded(patterns: &[&[u8]], case_insensitive: bool) -> Self {
        match Self::try_compile_folded(patterns, case_insensitive) {
            Ok(compiled) => compiled,
            Err(error) => {
                panic!(
                    "vyre-libs GpuLiteralSet::compile failed: {error}. \
                     returning an empty matcher would silently match nothing and report every input as clean; \
                     use try_compile and reduce the pattern set below the GPU ABI limits."
                )
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
        Self::try_compile_folded(patterns, false)
    }

    /// ASCII-case-insensitive counterpart of [`Self::try_compile`].
    ///
    /// # Errors
    /// See [`Self::try_compile`].
    pub fn try_compile_case_insensitive(
        patterns: &[&[u8]],
    ) -> Result<Self, LiteralSetCompileError> {
        Self::try_compile_folded(patterns, true)
    }

    fn try_compile_folded(
        patterns: &[&[u8]],
        case_insensitive: bool,
    ) -> Result<Self, LiteralSetCompileError> {
        let dfa = if case_insensitive {
            dfa_compile_case_insensitive(patterns)
        } else {
            dfa_compile(patterns)
        };
        let declared_pattern_count = u32::try_from(patterns.len()).map_err(|_| {
            LiteralSetCompileError::PatternCountOverflow {
                count: patterns.len(),
            }
        })?;
        let total_pattern_bytes = patterns.iter().try_fold(0usize, |sum, pattern| {
            sum.checked_add(pattern.len())
                .ok_or(LiteralSetCompileError::PatternByteCountOverflow)
        })?;
        u32::try_from(total_pattern_bytes).map_err(|_| {
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

        let program = try_build_literal_set_program(&dfa, declared_pattern_count)
            .map_err(|message| LiteralSetCompileError::DispatchProgramBuildFailed { message })?;

        Ok(Self {
            dfa,
            pattern_bytes,
            pattern_offsets,
            pattern_lengths,
            program,
            case_insensitive,
        })
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

    /// TIMED counterpart of [`Self::scan_into`]: decodes the `(pattern_id, start,
    /// end)` triples into `matches` exactly as [`Self::scan_into`] AND returns
    /// backend-owned timing ([`vyre_driver::TimedDispatchResult`]), so a consumer
    /// or benchmark can attribute the position scan's cost between the GPU kernel
    /// (`device_ns`) and host staging/readback (`wall_ns - device_ns`), the
    /// "attribution everywhere" contract on the position path, matching the
    /// resident `scan_into_timed`.
    ///
    /// The decoded matches are identical to [`Self::scan_into`]'s (same program,
    /// same inputs, only `dispatch_borrowed_timed` vs `dispatch_borrowed`
    /// differs). This reuses the one owned-buffer prepare path
    /// ([`Self::prepare_scan_dispatch`]); the untimed hot path
    /// ([`Self::scan_into_with_scratch`]) is untouched and pays no timing cost.
    /// Like [`Self::scan`] (and unlike [`Self::scan_all`]) this fails closed if a
    /// chunk exceeds `max_matches` rather than auto-resizing.
    ///
    /// # Errors
    /// See [`Self::scan_into`]. On a backend whose `dispatch_borrowed_timed` only
    /// records host wall time, the result's `device_ns` is `None` (loud absence,
    /// not a fabricated zero).
    pub fn scan_into_timed<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
        matches: &mut Vec<Match>,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre::BackendError> {
        matches.clear();
        let prepared = self.prepare_scan_dispatch(haystack, max_matches)?;
        let borrowed: smallvec::SmallVec<[&[u8]; 8]> =
            prepared.inputs.iter().map(Vec::as_slice).collect();
        let timed = backend.dispatch_borrowed_timed(
            &prepared.program,
            &borrowed,
            &prepared.dispatch_config,
        )?;
        prepared.decode_outputs_into(&timed.outputs, matches)?;
        Ok(timed)
    }

    /// ASYNC counterpart of [`Self::scan_into`]: submit the GPU match dispatch and
    /// return a [`PendingMatches`] handle IMMEDIATELY, so the caller can OVERLAP
    /// host-side work with the in-flight GPU scan, then decode the `(pattern_id,
    /// start, end)` triples via [`PendingMatches::await_into`].
    ///
    /// This is the position-scan sibling of [`Self::scan_presence_async`]. On a
    /// backend that genuinely pipelines host/device work (wgpu, cuda) the GPU scan
    /// executes while the caller does host work; on the synchronous default in
    /// [`VyreBackend::dispatch_async`] the handle is trivially ready and this is
    /// equivalent (same triples, no overlap, no silent change of result).
    ///
    /// Reuses the one owned-buffer prepare path ([`Self::prepare_scan_dispatch`]),
    /// whose owned inputs the [`PendingMatches`] handle RETAINS until the decode,
    /// keeping the device-side upload's backing memory valid for the whole
    /// dispatch. Like [`Self::scan_into`] (and unlike [`Self::scan_all`]) it fails
    /// closed if a chunk exceeds `max_matches` rather than auto-resizing.
    ///
    /// # Errors
    /// See [`Self::scan_into`]. Errors that surface only during GPU execution come
    /// back from [`PendingMatches::await_into`], not here.
    pub fn scan_into_async<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
    ) -> Result<PendingMatches, vyre::BackendError> {
        let prepared = self.prepare_scan_dispatch(haystack, max_matches)?;
        let pending = backend.dispatch_async(
            &prepared.program,
            &prepared.inputs,
            &prepared.dispatch_config,
        )?;
        Ok(PendingMatches { pending, prepared })
    }

    /// GPU scan dispatch that returns EVERY match with no fixed cap and no
    /// consumer-side paging: it auto-resizes the match buffer to the exact device
    /// count and NEVER silently truncates.
    ///
    /// The fixed-cap [`Self::scan`] fails closed when a chunk has more matches
    /// than `max_matches`, forcing every consumer to implement a paging retry
    /// loop (e.g. keyhog's `split_positioned_window`). This method makes that
    /// loop dead code: it dispatches once at an initial capacity; the match
    /// kernel's atomic counter reports the TRUE total even past the cap, so on
    /// saturation it resizes the output to exactly that count and re-dispatches
    /// ONCE. Common case (matches fit the initial capacity) is a single
    /// dispatch; a saturated chunk costs exactly two. The result is complete or
    /// it is a structured error (never a silent partial (Law 10)).
    ///
    /// Memory scales with the true match count, by contract. Callers that must
    /// bound host memory instead of recall use the capped [`Self::scan`].
    ///
    /// ## Why there is no `scan_all_async`
    /// This method is intentionally SYNC-ONLY. Its completeness guarantee is a
    /// count-then-maybe-resize protocol (dispatch → read the true count → resize
    /// and re-dispatch on saturation), and the second dispatch's size is not known
    /// until the first completes, so a fire-and-forget async handle cannot be
    /// well-typed without threading the engine + backend back through the await
    /// (a worse API than the two clean primitives that already compose to the same
    /// effect). An async caller that wants host/device overlap composes the
    /// existing primitives: submit [`Self::scan_into_async`] at a fixed cap (full
    /// overlap, common case), and on its fail-closed overflow error fall back to a
    /// synchronous `scan_all` for the rare saturated chunk, or use the resident
    /// [`Self::prepare_resident_scan`] path for a hot corpus loop.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if dispatch or readback fails, or if the
    /// true match count exceeds the u32 GPU match-output ABI (fail closed with
    /// the exact count and the fix, never a truncated decode).
    pub fn scan_all<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
    ) -> Result<Vec<Match>, vyre::BackendError> {
        let mut matches = Vec::new();
        self.scan_all_into(backend, haystack, &mut matches)?;
        Ok(matches)
    }

    /// [`Self::scan_all`] decoding into caller-owned match scratch.
    ///
    /// # Errors
    /// See [`Self::scan_all`].
    pub fn scan_all_into<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        matches: &mut Vec<Match>,
    ) -> Result<(), vyre::BackendError> {
        let mut scratch = ScanDispatchScratch::default();
        self.scan_all_into_with_scratch(backend, haystack, matches, &mut scratch)
    }

    /// [`Self::scan_all`] reusing caller-owned byte staging across dispatches.
    ///
    /// The haystack is packed and the prefilter tables are built ONCE; only the
    /// output-buffer capacity changes on an auto-resize retry, so the resize
    /// re-dispatches without re-packing the corpus.
    ///
    /// # Errors
    /// See [`Self::scan_all`].
    pub fn scan_all_into_with_scratch<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        matches: &mut Vec<Match>,
        scratch: &mut ScanDispatchScratch,
    ) -> Result<(), vyre::BackendError> {
        use crate::scan::dispatch_io;

        matches.clear();
        let haystack_len =
            dispatch_io::scan_guard(haystack, "literal_set", dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let haystack_bytes = scratch.haystack_bytes.as_slice();
        let prefilter_tables = self.build_prefilter_tables()?;
        let views = DfaPrefilterByteViews::new(&self.dfa, &self.pattern_lengths, &prefilter_tables);

        // Start at the default capacity; the atomic counter reports the true
        // total even when the output saturates, so one resize to that exact
        // count captures everything. Bounded to two dispatches, the recount is
        // exact and deterministic, so a third would indicate a nondeterministic
        // backend and is rejected loudly rather than looped forever.
        let mut capacity = LITERAL_SET_DEFAULT_MAX_MATCHES;
        for attempt in 0..2 {
            let program = self.program_for_match_capacity(capacity)?;
            let outputs = self.dispatch_literal_scan_outputs(
                backend,
                haystack_bytes,
                haystack_len,
                &views,
                program.as_ref(),
            )?;
            let count = decode_literal_set_count_outputs(&outputs)?;
            if count <= capacity {
                // count <= capacity: every triple was written; decode exactly.
                return decode_literal_set_outputs_into(&outputs, capacity, matches);
            }
            if attempt == 1 {
                return Err(vyre::BackendError::new(format!(
                    "literal_set scan_all recount instability: resized to exact device count {capacity} yet the re-dispatch reported {count}. Fix: this indicates a nondeterministic backend match counter; the scan cannot be completed without silent truncation."
                )));
            }
            // Saturation: `count` is the exact true total. Resize to it and
            // re-dispatch once. `program_for_match_capacity` fails closed if
            // `count` overflows the u32 match-output ABI (its own structured
            // error) (one place for that bound).
            capacity = count;
        }
        // Unreachable: the loop returns on both branches within two attempts.
        Err(vyre::BackendError::new(
            "literal_set scan_all exhausted its bounded auto-resize attempts without a decode. Fix: report this as a vyre bug, the count/capacity invariant was violated.",
        ))
    }

    /// TIMED counterpart of [`Self::scan_all`]: the complete-or-error auto-resize
    /// scan, returning backend-owned timing
    /// ([`vyre_driver::TimedDispatchResult`]) alongside the full match set.
    ///
    /// ## Which dispatch the timing attributes
    /// `scan_all` may dispatch TWICE, a first pass at the default capacity, then,
    /// if the output saturated, a second pass resized to the exact device count.
    /// The returned timing is the timing of the dispatch that PRODUCED THE RETURNED
    /// MATCHES: the single pass when the first fit, or the resize re-dispatch when
    /// a resize happened (`resized` in the result tells the caller which). This is
    /// the honest choice, the timing describes the decode the caller receives, not
    /// a hidden earlier pass, and it is loudly reported rather than silently summed
    /// (Law 10: a summed wall-time would misattribute two GPU launches as one).
    ///
    /// The returned matches are byte-for-byte identical to [`Self::scan_all`]'s
    /// (same programs, same inputs; only `dispatch_borrowed_timed` vs
    /// `dispatch_borrowed` differs). The untimed hot path is untouched.
    ///
    /// # Errors
    /// See [`Self::scan_all`]. On a backend whose `dispatch_borrowed_timed` only
    /// records host wall time, `device_ns` is `None` (loud absence, not a
    /// fabricated zero).
    pub fn scan_all_timed<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        matches: &mut Vec<Match>,
    ) -> Result<ScanAllTimed, vyre::BackendError> {
        use crate::scan::dispatch_io;

        matches.clear();
        let haystack_len =
            dispatch_io::scan_guard(haystack, "literal_set", dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        let mut haystack_bytes = Vec::new();
        dispatch_io::pack_haystack_u32_into(haystack, &mut haystack_bytes)?;
        let prefilter_tables = self.build_prefilter_tables()?;
        let views = DfaPrefilterByteViews::new(&self.dfa, &self.pattern_lengths, &prefilter_tables);

        let mut capacity = LITERAL_SET_DEFAULT_MAX_MATCHES;
        for attempt in 0..2 {
            let program = self.program_for_match_capacity(capacity)?;
            let timed = self.dispatch_literal_scan_outputs_timed(
                backend,
                &haystack_bytes,
                haystack_len,
                &views,
                program.as_ref(),
            )?;
            let count = decode_literal_set_count_outputs(&timed.outputs)?;
            if count <= capacity {
                // count <= capacity: every triple was written; decode exactly.
                decode_literal_set_outputs_into(&timed.outputs, capacity, matches)?;
                return Ok(ScanAllTimed {
                    timed,
                    resized: attempt == 1,
                });
            }
            if attempt == 1 {
                return Err(vyre::BackendError::new(format!(
                    "literal_set scan_all_timed recount instability: resized to exact device count {capacity} yet the re-dispatch reported {count}. Fix: this indicates a nondeterministic backend match counter; the scan cannot be completed without silent truncation."
                )));
            }
            capacity = count;
        }
        Err(vyre::BackendError::new(
            "literal_set scan_all_timed exhausted its bounded auto-resize attempts without a decode. Fix: report this as a vyre bug, the count/capacity invariant was violated.",
        ))
    }

    /// GPU count-only dispatch.
    ///
    /// Use this when the caller needs match cardinality or presence without
    /// materializing every `(pattern_id, start, end)` triple. It dispatches the
    /// suffix-prefiltered bounded DFA count kernel and reads one `u32`.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if dispatch, readback, scan-boundary
    /// validation, or host staging allocation fails.
    pub fn count<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
    ) -> Result<u32, vyre::BackendError> {
        let mut scratch = LiteralSetScanScratch::default();
        self.count_with_literal_scratch(backend, haystack, &mut scratch)
    }

    /// GPU scan dispatch that decodes into caller-owned match scratch and
    /// reuses caller-owned byte staging.
    ///
    /// `matches` reuses decoded match storage and `scratch` reuses the packed
    /// haystack buffer across dispatches. For stable literal-set hot loops, use
    /// [`Self::prepare_literal_scratch`] with
    /// [`Self::scan_into_with_literal_scratch`] to also reuse the derived
    /// suffix-prefilter tables and cap-specific program layout.
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
        let prefilter_tables = self.build_prefilter_tables()?;
        self.scan_into_with_program(
            backend,
            haystack,
            max_matches,
            matches,
            scratch,
            dispatch_program.as_ref(),
            &prefilter_tables,
        )
    }

    /// Prepare literal-set-owned hot-loop scratch for repeated dispatches.
    ///
    /// This builds the cap-specific `Program` layout and suffix-prefilter
    /// tables outside the timed scan path. It is useful for callers that know
    /// their match-capacity budget before scanning a stream of similarly shaped
    /// inputs.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if match-capacity sizing or
    /// suffix-prefilter staging fails.
    pub fn prepare_literal_scratch(
        &self,
        max_matches: u32,
        scratch: &mut LiteralSetScanScratch,
    ) -> Result<(), vyre::BackendError> {
        self.program_for_match_capacity_cached(max_matches, &mut scratch.cached_program)?;
        self.prefilter_tables_cached(&mut scratch.cached_prefilter)?;
        Ok(())
    }

    /// Prepare count-only hot-loop scratch for repeated dispatches.
    ///
    /// This builds the count dispatch `Program` and suffix-prefilter tables
    /// outside the timed count path without preparing match-list output state.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if suffix-prefilter staging fails.
    pub fn prepare_count_scratch(
        &self,
        scratch: &mut LiteralSetScanScratch,
    ) -> Result<(), vyre::BackendError> {
        self.count_program_cached(&mut scratch.cached_count_program)?;
        self.prefilter_tables_cached(&mut scratch.cached_prefilter)?;
        Ok(())
    }

    /// Prepare a backend-neutral dispatch payload for this literal set.
    ///
    /// The returned plan owns packed haystack bytes, DFA tables, suffix
    /// prefilter tables, the zeroed match counter, and the cap-specific
    /// `Program`. Direct callers can dispatch `inputs` through a normal
    /// borrowed-input backend. Runtimes with resident resources can upload the
    /// same `inputs` once and reuse the immutable resources across repeated
    /// scans of the same haystack.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if scan-boundary validation,
    /// cap-specific program sizing, suffix-prefilter staging, or input-buffer
    /// allocation fails.
    pub fn prepare_scan_dispatch(
        &self,
        haystack: &[u8],
        max_matches: u32,
    ) -> Result<LiteralSetPreparedScan, vyre::BackendError> {
        let dispatch_program = self.program_for_match_capacity(max_matches)?;
        let prefilter_tables = self.build_prefilter_tables()?;
        self.prepare_scan_dispatch_with_program(
            haystack,
            max_matches,
            dispatch_program.as_ref(),
            &prefilter_tables,
        )
    }

    /// Prepare a backend-neutral count-only dispatch payload.
    ///
    /// Runtimes with resident resources can upload the returned `inputs` once,
    /// reset [`LITERAL_SET_COUNT_RESET_RESOURCE_INDICES`], dispatch
    /// [`LITERAL_SET_COUNT_SCAN_RESOURCE_INDICES`], and read back
    /// [`LITERAL_SET_COUNT_RESOURCE_INDEX`] for one `u32` result.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if scan-boundary validation,
    /// suffix-prefilter staging, or input-buffer allocation fails.
    pub fn prepare_count_dispatch(
        &self,
        haystack: &[u8],
    ) -> Result<LiteralSetPreparedCount, vyre::BackendError> {
        let count_program = self.count_program();
        let prefilter_tables = self.build_prefilter_tables()?;
        self.prepare_count_dispatch_with_program(haystack, &count_program, &prefilter_tables)
    }

    /// GPU scan dispatch with literal-set-owned hot-loop scratch.
    ///
    /// Use this for repeated scans where `max_matches` is usually stable but
    /// not equal to the compiled default. It reuses packed haystack bytes,
    /// suffix-prefilter tables, and the cap-specific rewritten dispatch
    /// `Program`.
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
        let prefilter_tables = self.prefilter_tables_cached(&mut scratch.cached_prefilter)?;
        self.scan_into_with_program(
            backend,
            haystack,
            max_matches,
            matches,
            &mut scratch.dispatch,
            dispatch_program,
            prefilter_tables,
        )
    }

    /// GPU count-only dispatch with literal-set-owned hot-loop scratch.
    ///
    /// Reuses packed haystack bytes, suffix-prefilter tables, and the count
    /// dispatch `Program` across repeated scans.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] if dispatch, readback, scan-boundary
    /// validation, or host staging allocation fails.
    pub fn count_with_literal_scratch<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        scratch: &mut LiteralSetScanScratch,
    ) -> Result<u32, vyre::BackendError> {
        let count_program = self.count_program_cached(&mut scratch.cached_count_program)?;
        let prefilter_tables = self.prefilter_tables_cached(&mut scratch.cached_prefilter)?;
        self.count_with_program(
            backend,
            haystack,
            &mut scratch.dispatch,
            count_program,
            prefilter_tables,
        )
    }

    /// GPU PRESENCE scan: return a per-pattern presence bitmap as packed `u32`
    /// words (bit `p`: word `p >> 5`, bit `p & 31`: set iff pattern `p`'s literal
    /// occurs in `haystack`). This is the compact-output counterpart of
    /// [`Self::scan`] for prefilter consumers that need only WHICH patterns fired,
    /// not where. The kernel performs one idempotent `atomic_or` per hit into a
    /// `ceil(patterns/32)`-word bitmap instead of appending an `(id,start,end)`
    /// triple through an atomic counter, so match-DENSE inputs stay near the scan
    /// throughput ceiling rather than collapsing on per-hit output serialization +
    /// large triple readback (the dominant cost measured on dense corpora).
    ///
    /// The bitmap is sound for presence: concurrent lanes setting the same bit
    /// race harmlessly (OR is idempotent), and bits in the same word are merged by
    /// `atomic_or`. Inputs 0-5 / 7-9 are byte-identical to [`Self::scan`].
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure, scan-boundary
    /// validation, or a pattern count exceeding the u32 GPU ABI.
    pub fn scan_presence<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
    ) -> Result<Vec<u32>, vyre::BackendError> {
        let mut scratch = ScanDispatchScratch::default();
        self.scan_presence_with_scratch(backend, haystack, &mut scratch)
    }

    /// [`Self::scan_presence`] with caller-owned hot-loop scratch (reuses the
    /// packed-haystack staging buffer across repeated scans).
    ///
    /// # Errors
    /// See [`Self::scan_presence`].
    pub fn scan_presence_with_scratch<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        scratch: &mut ScanDispatchScratch,
    ) -> Result<Vec<u32>, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let pattern_count = u32::try_from(self.pattern_lengths.len()).map_err(|_| {
            vyre::BackendError::new(
                "literal_set presence: pattern count exceeds u32 GPU ABI".to_string(),
            )
        })?;
        let presence_words = presence_bitmap_words(pattern_count) as usize;
        let program =
            try_build_ac_bounded_ranges_suffix3_presence_program(&self.dfa, pattern_count)
                .map_err(vyre::BackendError::new)?;
        let prefilter_tables = self.build_prefilter_tables()?;

        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "literal_set_presence",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let haystack_bytes = scratch.haystack_bytes.as_slice();
        let views = DfaPrefilterByteViews::new(&self.dfa, &self.pattern_lengths, &prefilter_tables);
        let haystack_len_word = [haystack_len];
        let haystack_len_bytes = dispatch_io::u32_words_as_le_bytes(&haystack_len_word);
        // Presence buffer (binding 6) is read-write: uploaded zeroed, dispatched,
        // and read back. It is the entire output.
        let presence_zeroed = zeroed_presence_bytes(presence_words)?;

        let config =
            dispatch_io::byte_scan_dispatch_config(haystack_len, program.workgroup_size[0]);
        let borrowed_inputs: smallvec::SmallVec<[&[u8]; 10]> = [
            haystack_bytes,                         // 0: haystack (Packed U32)
            views.transitions.as_ref(),             // 1: transitions
            views.output_offsets.as_ref(),          // 2: output_offsets
            views.output_records.as_ref(),          // 3: output_records
            views.pattern_lengths.as_ref(),         // 4: pattern_lengths
            haystack_len_bytes.as_ref(),            // 5: haystack_len
            presence_zeroed.as_slice(),             // 6: presence (read_write)
            views.candidate_end_mask.as_ref(),      // 7: candidate_end_mask
            views.candidate_suffix2_mask.as_ref(),  // 8: candidate_suffix2_mask
            views.candidate_suffix3_bloom.as_ref(), // 9: candidate_suffix3_bloom
        ]
        .into_iter()
        .collect();
        let outputs = backend.dispatch_borrowed(&program, &borrowed_inputs, &config)?;
        // `presence` is the only read-write/output buffer, so it is outputs[0].
        let presence_bytes = dispatch_io::try_output_bytes(&outputs, 0, "literal_set presence")?;
        Ok(decode_presence_words(presence_bytes, presence_words))
    }

    /// GPU REGION-PRESENCE scan: return a per-REGION presence bitmap, where region
    /// `r` is the slice `[region_starts[r], region_starts[r+1])` of `haystack` (a
    /// coalesced batch of independent files). The result is `region_starts.len() ×
    /// presence_bitmap_words(pattern_count)` packed `u32` words: bit `p` of region
    /// `r`'s row is set iff pattern `p`'s literal occurs inside region `r`.
    ///
    /// This is the dense-batch counterpart of [`Self::scan_presence`]: it keeps the
    /// idempotent-`atomic_or` output (no per-hit counter, no triple readback, so it
    /// stays near the scan-throughput ceiling on match-dense corpora) while
    /// preserving per-file attribution, which the global presence bitmap loses. The
    /// consumer (e.g. a coalesced GPU phase-1 scanner) gets the exact per-file
    /// trigger set it needs without materializing spans or reducing triples on the
    /// host.
    ///
    /// `region_starts` must be ascending with `region_starts[0] == 0`. A match never
    /// spans a region boundary (the consumer inserts separator bytes between files),
    /// so the end-position attribution the kernel performs equals start attribution.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure, scan-boundary
    /// validation, an empty or non-zero-based `region_starts`, or a pattern/region
    /// count exceeding the u32 GPU ABI.
    pub fn scan_presence_by_region<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
    ) -> Result<Vec<u32>, vyre::BackendError> {
        let mut scratch = ScanDispatchScratch::default();
        self.scan_presence_by_region_with_scratch(backend, haystack, region_starts, 0, &mut scratch)
    }

    /// [`Self::scan_presence_by_region`] with caller-owned hot-loop scratch and an
    /// explicit `region_base` shard offset.
    ///
    /// `region_base` is added to every candidate position before the region
    /// binary search, so a SHARDED caller can dispatch a slice `haystack` (with
    /// local positions) against the WHOLE batch's `region_starts` by passing the
    /// shard's global start offset. Pass `0` for a single-dispatch scan. Each
    /// shard returns the full `region_count × words` bitmap (rows it didn't touch
    /// stay zero); OR the per-shard bitmaps to assemble the batch result.
    ///
    /// # Errors
    /// See [`Self::scan_presence_by_region`].
    pub fn scan_presence_by_region_with_scratch<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        scratch: &mut ScanDispatchScratch,
    ) -> Result<Vec<u32>, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let (pattern_count, region_count) = validate_region_starts(
            region_starts,
            &self.pattern_lengths,
            "literal_set region-presence",
        )?;
        let total_words = presence_by_region_words(pattern_count, region_count) as usize;
        let program = try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
            &self.dfa,
            pattern_count,
            region_count,
        )
        .map_err(vyre::BackendError::new)?;
        let prefilter_tables = self.build_prefilter_tables()?;

        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "literal_set_presence_by_region",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let haystack_bytes = scratch.haystack_bytes.as_slice();
        let views = DfaPrefilterByteViews::new(&self.dfa, &self.pattern_lengths, &prefilter_tables);
        let haystack_len_word = [haystack_len];
        let haystack_len_bytes = dispatch_io::u32_words_as_le_bytes(&haystack_len_word);
        let region_starts_bytes = dispatch_io::u32_words_as_le_bytes(region_starts);
        let region_base_bytes = region_base.to_le_bytes();
        // Per-region presence buffer (binding 6) is read-write: uploaded zeroed,
        // dispatched, read back. It is the entire output.
        let presence_zeroed = zeroed_presence_bytes(total_words)?;

        let config =
            dispatch_io::byte_scan_dispatch_config(haystack_len, program.workgroup_size[0]);
        let borrowed_inputs: smallvec::SmallVec<[&[u8]; 12]> = [
            haystack_bytes,                         // 0: haystack (Packed U32)
            views.transitions.as_ref(),             // 1: transitions
            views.output_offsets.as_ref(),          // 2: output_offsets
            views.output_records.as_ref(),          // 3: output_records
            views.pattern_lengths.as_ref(),         // 4: pattern_lengths
            haystack_len_bytes.as_ref(),            // 5: haystack_len
            presence_zeroed.as_slice(),             // 6: per-region presence (read_write)
            views.candidate_end_mask.as_ref(),      // 7: candidate_end_mask
            views.candidate_suffix2_mask.as_ref(),  // 8: candidate_suffix2_mask
            views.candidate_suffix3_bloom.as_ref(), // 9: candidate_suffix3_bloom
            region_starts_bytes.as_ref(),           // 10: region_starts
            region_base_bytes.as_slice(),           // 11: region_base (shard offset)
        ]
        .into_iter()
        .collect();
        let outputs = backend.dispatch_borrowed(&program, &borrowed_inputs, &config)?;
        let presence_bytes =
            dispatch_io::try_output_bytes(&outputs, 0, "literal_set presence_by_region")?;
        Ok(decode_presence_words(presence_bytes, total_words))
    }

    /// TIMED counterpart of [`Self::scan_presence_by_region`]: runs the same
    /// region-presence dispatch but returns backend-owned timing
    /// ([`vyre_driver::TimedDispatchResult`]) alongside the decoded per-region
    /// presence bitmap, so a consumer or benchmark can attribute the per-scan cost
    /// between the GPU kernel (`device_ns`) and host-side staging/readback
    /// (`wall_ns - device_ns`), the "attribution everywhere" contract, on the
    /// hot literal region-presence path rather than only the resident path.
    ///
    /// The returned bitmap is byte-for-byte identical to
    /// [`Self::scan_presence_by_region`]'s (same program, same inputs); only the
    /// dispatch call differs (`dispatch_borrowed_timed` vs `dispatch_borrowed`).
    /// The returned result's `outputs` are the same raw presence bytes already
    /// decoded into the returned bitmap (mirrors the resident `scan_into_timed`
    /// contract). This reuses the one owned-buffer prepare path
    /// ([`Self::build_presence_by_region_dispatch`]); the untimed hot path is
    /// untouched and pays no timing cost.
    ///
    /// # Errors
    /// See [`Self::scan_presence_by_region`]. On a backend whose
    /// `dispatch_borrowed_timed` only records host wall time, `device_ns` is
    /// `None` (loud absence, not a fabricated zero).
    pub fn scan_presence_by_region_timed<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
    ) -> Result<(Vec<u32>, vyre_driver::TimedDispatchResult), vyre::BackendError> {
        use crate::scan::dispatch_io;

        let (program, inputs, config, total_words, _haystack_len) =
            self.build_presence_by_region_dispatch(haystack, region_starts, region_base)?;
        let borrowed: smallvec::SmallVec<[&[u8]; 12]> = inputs.iter().map(Vec::as_slice).collect();
        let timed = backend.dispatch_borrowed_timed(&program, &borrowed, &config)?;
        let presence_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            0,
            "literal_set presence_by_region timed",
        )?;
        let presence = decode_presence_words(presence_bytes, total_words);
        Ok((presence, timed))
    }

    /// ASYNC counterpart of [`Self::scan_presence_by_region_with_scratch`]: submit
    /// the GPU region-presence dispatch and return a [`PendingPresenceByRegion`]
    /// handle IMMEDIATELY, so the caller can OVERLAP host-side work (e.g. a downstream
    /// scanner's trigger-independent entropy-candidate generation) with the in-flight GPU
    /// scan, then decode the per-region bitmap via
    /// [`PendingPresenceByRegion::await_words`].
    ///
    /// On a backend that genuinely pipelines host/device work (wgpu, cuda) the GPU
    /// scan executes while the caller does host work; on a backend that cannot
    /// (the synchronous default in [`VyreBackend::dispatch_async`]) the handle is
    /// trivially ready and this is equivalent, same bitmap, no overlap, no silent
    /// change of result. See [`vyre::backend::PendingDispatch`].
    ///
    /// Unlike the synchronous entry there is NO `scratch` parameter: the async
    /// dispatch ABI is `&[Vec<u8>]`, so inputs are built into OWNED buffers that
    /// the returned handle RETAINS until `await_words`. This keeps the device-side
    /// upload's backing memory valid for the whole dispatch, required on backends
    /// (e.g. the CUDA stream h2d copy) whose async upload reads host memory after
    /// this call returns.
    ///
    /// `region_base` has the same sharded-offset meaning as in
    /// [`Self::scan_presence_by_region_with_scratch`] (pass `0` for a single-shard
    /// scan).
    ///
    /// # Errors
    /// See [`Self::scan_presence_by_region`]. Errors that surface only during GPU
    /// execution come back from [`PendingPresenceByRegion::await_words`], not here.
    pub fn scan_presence_by_region_async<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
    ) -> Result<PendingPresenceByRegion, vyre::BackendError> {
        let (program, inputs, config, total_words, _haystack_len) =
            self.build_presence_by_region_dispatch(haystack, region_starts, region_base)?;
        let pending = backend.dispatch_async(&program, &inputs, &config)?;
        Ok(PendingPresenceByRegion {
            pending,
            total_words,
            _inputs: inputs,
        })
    }

    /// Build the OWNED region-presence dispatch payload shared by
    /// [`Self::scan_presence_by_region_async`] and
    /// [`Self::prepare_presence_by_region_dispatch`]: the cap-specific program,
    /// the 12 input buffers in binding order (binding 6 is the zeroed per-region
    /// presence read-write resource = the whole output), the byte-scan dispatch
    /// config, and the total presence-bitmap `u32` word count.
    ///
    /// Inputs are OWNED (the async ABI is `&[Vec<u8>]`, and a resident runtime
    /// uploads them once), built through the fallible `copy_u32_words_as_le_bytes`
    /// so an allocation failure fails CLOSED (`BackendError`) instead of aborting
    /// on OOM (the same contract as [`Self::prepare_scan_dispatch`]).
    ///
    /// # Errors
    /// See [`Self::scan_presence_by_region`].
    /// Encode the seven corpus-invariant region-presence tables into owned
    /// little-endian byte buffers through the fail-closed `copy_u32_words_as_le_bytes`.
    /// The single source of truth for which tables are immutable across a corpus 
    /// reused by the borrowed/async/prepared builder and the resident pipeline so
    /// every path encodes byte-identical tables.
    fn presence_immutable_table_bytes(
        &self,
        prefilter: &LiteralSetPrefilterTables,
    ) -> Result<PresenceImmutableTableBytes, vyre::BackendError> {
        Ok(PresenceImmutableTableBytes {
            transitions: copy_u32_words_as_le_bytes(&self.dfa.transitions, "transition table")?,
            output_offsets: copy_u32_words_as_le_bytes(
                &self.dfa.output_offsets,
                "output offset table",
            )?,
            output_records: copy_u32_words_as_le_bytes(
                &self.dfa.output_records,
                "output record table",
            )?,
            pattern_lengths: copy_u32_words_as_le_bytes(
                &self.pattern_lengths,
                "pattern length table",
            )?,
            candidate_end_mask: copy_u32_words_as_le_bytes(
                &prefilter.candidate_end_mask,
                "candidate end mask",
            )?,
            candidate_suffix2_mask: copy_u32_words_as_le_bytes(
                &prefilter.candidate_suffix2_mask,
                "candidate suffix2 mask",
            )?,
            candidate_suffix3_bloom: copy_u32_words_as_le_bytes(
                &prefilter.candidate_suffix3_bloom,
                "candidate suffix3 bloom",
            )?,
        })
    }

    /// SINGLE owner of the presence-dispatch common staging: bindings 0..=9, which
    /// are BYTE-IDENTICAL across the global, by-region, and fused presence programs
    /// (haystack, the 7 immutable DFA/suffix-prefilter tables via
    /// [`Self::presence_immutable_table_bytes`], `haystack_len`, and the zeroed
    /// binding-6 presence read-write output). Each caller passes the presence
    /// buffer's `total_words` (global vs by-region sizing) and the FULL binding
    /// count (10 / 12 / 13) so the returned `inputs` vec is reserved once for the
    /// whole payload, then appends only its extra tail bindings (region_starts,
    /// region_base, match_count). Returns `(haystack_len, inputs)`.
    ///
    /// This is the ONE PLACE for the 0..=9 order: reordering here miswires all
    /// three presence programs at once, and a per-path copy would drift silently.
    /// Owned + fail-closed (`copy_u32_words_as_le_bytes` / `try_reserve`), so an
    /// allocation failure returns `BackendError` rather than aborting on OOM.
    ///
    /// # Errors
    /// Scan-boundary validation, host staging allocation, or table encoding.
    fn build_presence_common_inputs(
        &self,
        haystack: &[u8],
        guard_ctx: &'static str,
        total_words: usize,
        total_binding_count: usize,
    ) -> Result<(u32, Vec<Vec<u8>>), vyre::BackendError> {
        use crate::scan::dispatch_io;

        let prefilter_tables = self.build_prefilter_tables()?;
        let haystack_len =
            dispatch_io::scan_guard(haystack, guard_ctx, dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        let mut haystack_packed = Vec::new();
        dispatch_io::pack_haystack_u32_into(haystack, &mut haystack_packed)?;
        let haystack_len_word = [haystack_len];
        // Presence buffer (binding 6) is read-write: uploaded zeroed, dispatched,
        // read back. It is the entire (per-region or global) output.
        let presence_zeroed = zeroed_presence_bytes(total_words)?;

        let mut inputs: Vec<Vec<u8>> = Vec::new();
        vyre_foundation::allocation::try_reserve_vec_to_capacity(&mut inputs, total_binding_count)
            .map_err(|source| {
                vyre::BackendError::new(format!(
                    "literal_set presence ({guard_ctx}) could not reserve {total_binding_count} input buffer slot(s): {source}. Fix: shard the literal set or haystack before dispatch."
                ))
            })?;
        let tables = self.presence_immutable_table_bytes(&prefilter_tables)?;
        inputs.push(haystack_packed); // 0: haystack (Packed U32)
        inputs.push(tables.transitions); // 1
        inputs.push(tables.output_offsets); // 2
        inputs.push(tables.output_records); // 3
        inputs.push(tables.pattern_lengths); // 4
        inputs.push(copy_u32_words_as_le_bytes(
            &haystack_len_word,
            "haystack length",
        )?); // 5
        inputs.push(presence_zeroed); // 6: presence (read_write) = the output
        inputs.push(tables.candidate_end_mask); // 7
        inputs.push(tables.candidate_suffix2_mask); // 8
        inputs.push(tables.candidate_suffix3_bloom); // 9
        Ok((haystack_len, inputs))
    }

    fn build_presence_by_region_dispatch(
        &self,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
    ) -> Result<(Program, Vec<Vec<u8>>, DispatchConfig, usize, u32), vyre::BackendError> {
        use crate::scan::dispatch_io;

        let (pattern_count, region_count) = validate_region_starts(
            region_starts,
            &self.pattern_lengths,
            "literal_set region-presence",
        )?;
        let total_words = presence_by_region_words(pattern_count, region_count) as usize;
        let program = try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
            &self.dfa,
            pattern_count,
            region_count,
        )
        .map_err(vyre::BackendError::new)?;

        // Bindings 0..=9 (the ONE-PLACE common staging), reserved for all 12.
        const PRESENCE_BY_REGION_INPUT_COUNT: usize = 12;
        let (haystack_len, mut inputs) = self.build_presence_common_inputs(
            haystack,
            "literal_set_presence_by_region",
            total_words,
            PRESENCE_BY_REGION_INPUT_COUNT,
        )?;
        // Tail bindings unique to the by-region program.
        inputs.push(copy_u32_words_as_le_bytes(region_starts, "region starts")?); // 10
        inputs.push(region_base.to_le_bytes().to_vec()); // 11: region_base (4 bytes)

        let config =
            dispatch_io::byte_scan_dispatch_config(haystack_len, program.workgroup_size[0]);
        Ok((program, inputs, config, total_words, haystack_len))
    }

    /// Build the OWNED GLOBAL-presence dispatch payload (the 10 input buffers in
    /// binding order, binding 6 = the zeroed presence read-write resource = the
    /// whole output; the presence-bitmap `u32` word count) shared by
    /// [`Self::scan_presence_timed`]. This is the global-presence sibling of
    /// [`Self::build_presence_by_region_dispatch`] and reuses the same
    /// [`Self::presence_immutable_table_bytes`] table encoder, so every presence
    /// path encodes byte-identical immutable tables.
    ///
    /// Inputs are OWNED and built through the fail-closed
    /// `copy_u32_words_as_le_bytes`, so an allocation failure fails CLOSED
    /// (`BackendError`) rather than aborting on OOM. The untimed hot path
    /// ([`Self::scan_presence_with_scratch`]) keeps its zero-copy borrowed-views
    /// staging and is untouched.
    ///
    /// # Errors
    /// See [`Self::scan_presence`].
    fn build_presence_dispatch(
        &self,
        haystack: &[u8],
    ) -> Result<(Program, Vec<Vec<u8>>, DispatchConfig, usize), vyre::BackendError> {
        use crate::scan::dispatch_io;

        let pattern_count = u32::try_from(self.pattern_lengths.len()).map_err(|_| {
            vyre::BackendError::new(
                "literal_set presence: pattern count exceeds u32 GPU ABI".to_string(),
            )
        })?;
        let presence_words = presence_bitmap_words(pattern_count) as usize;
        let program =
            try_build_ac_bounded_ranges_suffix3_presence_program(&self.dfa, pattern_count)
                .map_err(vyre::BackendError::new)?;

        // The global-presence program is EXACTLY the 0..=9 common staging, no tail
        // bindings (so it reserves and fills all 10 through the shared owner).
        const PRESENCE_INPUT_COUNT: usize = 10;
        let (haystack_len, inputs) = self.build_presence_common_inputs(
            haystack,
            "literal_set_presence",
            presence_words,
            PRESENCE_INPUT_COUNT,
        )?;

        let config =
            dispatch_io::byte_scan_dispatch_config(haystack_len, program.workgroup_size[0]);
        Ok((program, inputs, config, presence_words))
    }

    /// TIMED counterpart of [`Self::scan_presence`]: runs the same global-presence
    /// dispatch but returns backend-owned timing
    /// ([`vyre_driver::TimedDispatchResult`]) alongside the decoded presence
    /// bitmap, so a consumer or benchmark can split the per-scan cost between the
    /// GPU kernel (`device_ns`) and host-side staging/readback, the "attribution
    /// everywhere" contract on the global-presence path.
    ///
    /// The returned bitmap is byte-for-byte identical to [`Self::scan_presence`]'s
    /// (same program, same inputs); only the dispatch call differs
    /// (`dispatch_borrowed_timed` vs `dispatch_borrowed`). This reuses the one
    /// owned-buffer prepare path ([`Self::build_presence_dispatch`]); the untimed
    /// hot path is untouched and pays no timing cost.
    ///
    /// # Errors
    /// See [`Self::scan_presence`]. On a backend whose `dispatch_borrowed_timed`
    /// only records host wall time, `device_ns` is `None` (loud absence, not a
    /// fabricated zero).
    pub fn scan_presence_timed<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
    ) -> Result<(Vec<u32>, vyre_driver::TimedDispatchResult), vyre::BackendError> {
        use crate::scan::dispatch_io;

        let (program, inputs, config, presence_words) = self.build_presence_dispatch(haystack)?;
        let borrowed: smallvec::SmallVec<[&[u8]; 10]> = inputs.iter().map(Vec::as_slice).collect();
        let timed = backend.dispatch_borrowed_timed(&program, &borrowed, &config)?;
        let presence_bytes =
            dispatch_io::try_output_bytes(&timed.outputs, 0, "literal_set presence timed")?;
        let presence = decode_presence_words(presence_bytes, presence_words);
        Ok((presence, timed))
    }

    /// ASYNC counterpart of [`Self::scan_presence`]: submit the global-presence
    /// GPU dispatch and return a [`PendingPresence`] handle IMMEDIATELY, so the
    /// caller can OVERLAP host-side work with the in-flight GPU scan, then decode
    /// the presence bitmap via [`PendingPresence::await_words`].
    ///
    /// This is the global-presence sibling of
    /// [`Self::scan_presence_by_region_async`]. On a backend that genuinely
    /// pipelines host/device work (wgpu, cuda) the GPU scan executes while the
    /// caller does host work; on the synchronous default in
    /// [`VyreBackend::dispatch_async`] the handle is trivially ready and this is
    /// equivalent (same bitmap, no overlap, no silent change of result).
    ///
    /// Inputs are built into OWNED buffers (through the shared
    /// [`Self::build_presence_dispatch`]) that the handle RETAINS until
    /// `await_words`, keeping the device-side upload's backing memory valid for the
    /// whole dispatch.
    ///
    /// # Errors
    /// See [`Self::scan_presence`]. Errors that surface only during GPU execution
    /// come back from [`PendingPresence::await_words`], not here.
    pub fn scan_presence_async<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
    ) -> Result<PendingPresence, vyre::BackendError> {
        let (program, inputs, config, presence_words) = self.build_presence_dispatch(haystack)?;
        let pending = backend.dispatch_async(&program, &inputs, &config)?;
        Ok(PendingPresence {
            pending,
            presence_words,
            _inputs: inputs,
        })
    }

    /// Prepare a backend-neutral RESIDENT region-presence dispatch payload: the
    /// same owned buffers [`Self::scan_presence_by_region_async`] dispatches, but
    /// returned for a resident runtime to upload ONCE (the immutable DFA /
    /// suffix-prefilter tables) and re-dispatch across many files of a corpus,
    /// re-uploading only the haystack and resetting the binding-6 presence
    /// resource ([`LITERAL_SET_PRESENCE_BY_REGION_OUTPUT_RESOURCE_INDEX`]). This
    /// is the presence sibling of [`Self::prepare_scan_dispatch`].
    ///
    /// A direct caller can also dispatch `inputs` through a normal borrowed-input
    /// backend and decode binding 0 via [`LiteralSetPreparedPresenceByRegion::decode_presence`].
    ///
    /// # Errors
    /// See [`Self::scan_presence_by_region`].
    pub fn prepare_presence_by_region_dispatch(
        &self,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
    ) -> Result<LiteralSetPreparedPresenceByRegion, vyre::BackendError> {
        let region_count = u32::try_from(region_starts.len()).map_err(|_| {
            vyre::BackendError::new(
                "literal_set region-presence: region count exceeds u32 GPU ABI".to_string(),
            )
        })?;
        let (program, inputs, dispatch_config, total_words, haystack_len) =
            self.build_presence_by_region_dispatch(haystack, region_starts, region_base)?;
        let presence_output_bytes = total_words.saturating_mul(U32_BYTES);
        let encoded_input_bytes = inputs.iter().try_fold(0_u64, |sum, input| {
            let len = u64::try_from(input.len()).map_err(|source| {
                vyre::BackendError::new(format!(
                    "literal_set prepared region-presence input byte length does not fit u64: {source}. Fix: shard the scan before dispatch."
                ))
            })?;
            sum.checked_add(len).ok_or_else(|| {
                vyre::BackendError::new(
                    "literal_set prepared region-presence input byte total overflowed u64. Fix: shard the scan before dispatch.",
                )
            })
        })?;
        Ok(LiteralSetPreparedPresenceByRegion {
            program,
            inputs,
            dispatch_config,
            haystack_len,
            region_count,
            total_words,
            presence_output_bytes,
            encoded_input_bytes,
        })
    }

    /// Extract the IMMUTABLE region-presence tables, everything that does NOT
    /// change across the files of a corpus, plus a `max_regions`-sized program,
    /// for [`ResidentPresencePipeline`](crate::scan::resident_presence::ResidentPresencePipeline)
    /// to upload into backend-resident resources ONCE.
    ///
    /// The borrowed / async / prepared presence paths re-encode and re-upload the
    /// DFA transition / output / pattern-length tables and the suffix prefilter
    /// masks on EVERY dispatch (see [`Self::build_presence_by_region_dispatch`]).
    /// Those seven buffers depend only on `self`, not on the haystack or region
    /// layout, so a resident session uploads them once and re-dispatches across a
    /// corpus, transferring only the per-file haystack and the binding-6 presence
    /// reset. This is the presence sibling of
    /// [`RulePipeline::prepare_resident`](crate::scan::mega_scan::RulePipeline::prepare_resident)'s
    /// table extraction.
    ///
    /// The returned [`program`](ResidentPresenceTables::program) is sized for
    /// `max_regions` coalesced files (binding 6's element count and the kernel's
    /// `ceil_log2(max_regions)` region binary-search width); the actual per-scan
    /// region count is read dynamically from `buf_len(region_starts)`, so the same
    /// program serves any batch with `region_count <= max_regions`.
    ///
    /// Bytes are built through the same fallible `copy_u32_words_as_le_bytes` the
    /// prepared path uses, so an allocation failure fails CLOSED (`BackendError`)
    /// instead of aborting on OOM.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] when `max_regions` is zero, the pattern
    /// count exceeds the u32 GPU ABI, the presence program cannot be built, or any
    /// table allocation fails.
    pub(crate) fn resident_presence_tables(
        &self,
        max_regions: u32,
    ) -> Result<ResidentPresenceTables, vyre::BackendError> {
        let pattern_count = u32::try_from(self.pattern_lengths.len()).map_err(|_| {
            vyre::BackendError::new(
                "literal_set region-presence: pattern count exceeds u32 GPU ABI".to_string(),
            )
        })?;
        if max_regions == 0 {
            return Err(vyre::BackendError::new(
                "literal_set resident region-presence: max_regions must be >= 1 (it sizes the resident presence buffer and the kernel's region binary-search width). Fix: pass the largest coalesced-batch file count the session will scan.".to_string(),
            ));
        }
        let program = try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
            &self.dfa,
            pattern_count,
            max_regions,
        )
        .map_err(vyre::BackendError::new)?;
        let prefilter_tables = self.build_prefilter_tables()?;
        let tables = self.presence_immutable_table_bytes(&prefilter_tables)?;
        let presence_words = presence_bitmap_words(pattern_count);
        let workgroup_x = program.workgroup_size[0];
        Ok(ResidentPresenceTables {
            program,
            transitions: tables.transitions,
            output_offsets: tables.output_offsets,
            output_records: tables.output_records,
            pattern_lengths: tables.pattern_lengths,
            candidate_end_mask: tables.candidate_end_mask,
            candidate_suffix2_mask: tables.candidate_suffix2_mask,
            candidate_suffix3_bloom: tables.candidate_suffix3_bloom,
            pattern_count,
            presence_words,
            workgroup_x,
        })
    }

    /// FUSED region-presence + match-positions GPU scan with a default dispatch
    /// scratch. See [`Self::scan_presence_and_positions_by_region_with_scratch`] for
    /// the hot-loop variant that reuses caller-owned staging.
    ///
    /// # Errors
    /// See [`Self::scan_presence_and_positions_by_region_with_scratch`].
    pub fn scan_presence_and_positions_by_region<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        max_matches: u32,
        matches: &mut Vec<Match>,
    ) -> Result<Vec<u32>, vyre::BackendError> {
        let mut scratch = ScanDispatchScratch::default();
        self.scan_presence_and_positions_by_region_with_scratch(
            backend,
            haystack,
            region_starts,
            region_base,
            max_matches,
            matches,
            &mut scratch,
        )
    }

    /// FUSED region-presence + match-positions GPU scan: ONE dispatch returns BOTH
    /// the per-region presence bitmap (the return value, identical to
    /// [`Self::scan_presence_by_region`]) AND the `(pattern_id, start, end)` match
    /// triples (decoded into `matches`, identical to [`Self::scan_into`]).
    ///
    /// This collapses the two-pass pattern a coalesced consumer uses today, a
    /// presence-by-region scan to learn WHICH patterns fired per file, then a
    /// SEPARATE position scan over the same haystack to learn WHERE, into a single
    /// suffix3-gated walk. Both outputs are recall-identical to the separate scans by
    /// construction: the same candidate gate, DFA replay, and `output_records`
    /// iteration drive both, so the presence bits equal
    /// [`Self::scan_presence_by_region`]'s and the triples equal [`Self::scan_into`]'s.
    ///
    /// `region_starts` / `region_base` behave as in
    /// [`Self::scan_presence_by_region_with_scratch`]; `max_matches` bounds the triple
    /// output as in [`Self::scan_into`]. The returned bitmap is
    /// `region_starts.len() × presence_bitmap_words(pattern_count)` words.
    ///
    /// ## PERF CAVEAT (MEASURED, RTX 5090 / wgpu / release)
    /// This is a CORRECTNESS-equivalent primitive, NOT (yet) a perf win. Although it
    /// does one haystack walk instead of two, it is ~20x SLOWER than calling
    /// [`Self::scan_presence_by_region`] + [`Self::scan_into`] separately. Cause: the
    /// suffix3 prefilter inlines the replay 3x (i==0 / i==1 / general exits); the
    /// fused replay (region binary search + atomic_or + triple append) is large, so
    /// the fused kernel is ~3x bigger → register/occupancy collapse slows the whole
    /// scan. Prefer the two separate scans until the prefilter calls the replay as a
    /// function rather than inlining it (or a CUDA-backend measurement justifies the
    /// fused path). Proof + numbers: `tests/literal_set_presence_and_positions_gpu.rs`.
    ///
    /// # Errors
    /// Returns [`vyre::BackendError`] on dispatch/readback failure, scan-boundary
    /// validation, an empty or non-zero-based `region_starts`, or a pattern/region
    /// count exceeding the u32 GPU ABI.
    #[allow(clippy::too_many_arguments)]
    pub fn scan_presence_and_positions_by_region_with_scratch<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        max_matches: u32,
        matches: &mut Vec<Match>,
        scratch: &mut ScanDispatchScratch,
    ) -> Result<Vec<u32>, vyre::BackendError> {
        use crate::scan::dispatch_io;

        matches.clear();
        let (pattern_count, region_count) = validate_region_starts(
            region_starts,
            &self.pattern_lengths,
            "literal_set region-presence+positions",
        )?;
        let total_words = presence_by_region_words(pattern_count, region_count) as usize;
        let program = try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
            &self.dfa,
            pattern_count,
            region_count,
            max_matches,
        )
        .map_err(vyre::BackendError::new)?;
        let prefilter_tables = self.build_prefilter_tables()?;

        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "literal_set_presence_and_positions_by_region",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let haystack_bytes = scratch.haystack_bytes.as_slice();
        let views = DfaPrefilterByteViews::new(&self.dfa, &self.pattern_lengths, &prefilter_tables);
        let haystack_len_word = [haystack_len];
        let haystack_len_bytes = dispatch_io::u32_words_as_le_bytes(&haystack_len_word);
        let region_starts_bytes = dispatch_io::u32_words_as_le_bytes(region_starts);
        let region_base_bytes = region_base.to_le_bytes();
        // Both read-write buffers are uploaded zeroed; the matches output (binding
        // 13) is a pure `BufferDecl::output` the backend allocates from the program.
        let presence_zeroed = zeroed_presence_bytes(total_words)?;
        let match_count_bytes = [0u8; 4];

        let config =
            dispatch_io::byte_scan_dispatch_config(haystack_len, program.workgroup_size[0]);
        let borrowed_inputs: smallvec::SmallVec<[&[u8]; 13]> = [
            haystack_bytes,                         // 0: haystack (Packed U32)
            views.transitions.as_ref(),             // 1: transitions
            views.output_offsets.as_ref(),          // 2: output_offsets
            views.output_records.as_ref(),          // 3: output_records
            views.pattern_lengths.as_ref(),         // 4: pattern_lengths
            haystack_len_bytes.as_ref(),            // 5: haystack_len
            presence_zeroed.as_slice(),             // 6: per-region presence (read_write)
            views.candidate_end_mask.as_ref(),      // 7: candidate_end_mask
            views.candidate_suffix2_mask.as_ref(),  // 8: candidate_suffix2_mask
            views.candidate_suffix3_bloom.as_ref(), // 9: candidate_suffix3_bloom
            region_starts_bytes.as_ref(),           // 10: region_starts
            region_base_bytes.as_slice(),           // 11: region_base (shard offset)
            match_count_bytes.as_slice(),           // 12: match_count (read_write)
                                                    // 13: matches output (backend-allocated)
        ]
        .into_iter()
        .collect();
        let outputs = backend.dispatch_borrowed(&program, &borrowed_inputs, &config)?;

        // Output ordering = read_write + output buffers by binding:
        // presence(6) -> outputs[0], match_count(12) -> outputs[1], matches(13) -> outputs[2].
        let presence_bytes = dispatch_io::try_output_bytes(
            &outputs,
            0,
            "literal_set presence_and_positions_by_region presence",
        )?;
        let presence_words = decode_presence_words(presence_bytes, total_words);

        let count_bytes = dispatch_io::try_output_bytes(
            &outputs,
            1,
            "literal_set presence_and_positions_by_region match count",
        )?;
        let count = dispatch_io::try_read_u32_prefix(
            count_bytes,
            "literal_set presence_and_positions_by_region match count",
        )?;
        let matches_bytes = dispatch_io::try_output_bytes(
            &outputs,
            2,
            "literal_set presence_and_positions_by_region matches",
        )?;
        // The kernel's atomic match counter overcounts past the fixed cap, so a
        // count over `max_matches` means positions were dropped: fail closed
        // instead of silently decoding the truncated prefix (Law 10). Presence
        // bits stay exact (one bit per region/pattern, never truncated); it is
        // the positions list that overflows, so the overflow is surfaced here.
        dispatch_io::try_unpack_match_triples_capped_into(
            matches_bytes,
            count,
            max_matches,
            "literal_set presence_and_positions_by_region matches",
            matches,
        )?;

        Ok(presence_words)
    }

    /// Build the OWNED fused presence+positions-by-region dispatch payload (the 13
    /// input buffers in binding order; binding 6 = zeroed per-region presence
    /// read-write, binding 12 = zeroed match_count read-write; binding 13 = the
    /// backend-allocated matches output; the presence-bitmap `u32` word count)
    /// shared by [`Self::scan_presence_and_positions_by_region_timed`]. Reuses the
    /// same [`Self::presence_immutable_table_bytes`] encoder as every other
    /// presence path, so the immutable tables are byte-identical.
    ///
    /// Inputs are OWNED and built through the fail-closed
    /// `copy_u32_words_as_le_bytes`, so an allocation failure fails CLOSED
    /// (`BackendError`) rather than aborting on OOM. The untimed hot path
    /// ([`Self::scan_presence_and_positions_by_region_with_scratch`]) keeps its
    /// zero-copy borrowed-views staging and is untouched.
    ///
    /// # Errors
    /// See [`Self::scan_presence_and_positions_by_region`].
    fn build_presence_and_positions_by_region_dispatch(
        &self,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        max_matches: u32,
    ) -> Result<(Program, Vec<Vec<u8>>, DispatchConfig, usize), vyre::BackendError> {
        use crate::scan::dispatch_io;

        let (pattern_count, region_count) = validate_region_starts(
            region_starts,
            &self.pattern_lengths,
            "literal_set region-presence+positions",
        )?;
        let total_words = presence_by_region_words(pattern_count, region_count) as usize;
        let program = try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
            &self.dfa,
            pattern_count,
            region_count,
            max_matches,
        )
        .map_err(vyre::BackendError::new)?;

        // Bindings 0..=9 (the ONE-PLACE common staging), reserved for all 13. The
        // matches output (binding 13) is a backend-allocated `BufferDecl::output`,
        // not an input, so only bindings 10..=12 are appended here.
        const PRESENCE_AND_POSITIONS_INPUT_COUNT: usize = 13;
        let (haystack_len, mut inputs) = self.build_presence_common_inputs(
            haystack,
            "literal_set_presence_and_positions_by_region",
            total_words,
            PRESENCE_AND_POSITIONS_INPUT_COUNT,
        )?;
        // Tail bindings unique to the fused program.
        inputs.push(copy_u32_words_as_le_bytes(region_starts, "region starts")?); // 10
        inputs.push(region_base.to_le_bytes().to_vec()); // 11: region_base (shard offset)
        inputs.push(vec![0u8; 4]); // 12: match_count (read_write, zeroed)

        let config =
            dispatch_io::byte_scan_dispatch_config(haystack_len, program.workgroup_size[0]);
        Ok((program, inputs, config, total_words))
    }

    /// TIMED counterpart of [`Self::scan_presence_and_positions_by_region`]: runs
    /// the same fused ONE-dispatch scan but returns backend-owned timing
    /// ([`vyre_driver::TimedDispatchResult`]) alongside BOTH decoded outputs, the
    /// per-region presence bitmap (return value's `.0`) and the `(pid, start, end)`
    /// match triples (decoded into `matches`). This is the "attribution
    /// everywhere" contract on the fused path, so a benchmark can attribute the
    /// (documented ~20x-heavier) fused kernel's cost between device and staging.
    ///
    /// Both outputs are byte-for-byte identical to the untimed
    /// [`Self::scan_presence_and_positions_by_region`] (same program, same inputs;
    /// only `dispatch_borrowed_timed` vs `dispatch_borrowed` differs). Reuses the
    /// one owned-buffer prepare path
    /// ([`Self::build_presence_and_positions_by_region_dispatch`]); the untimed hot
    /// path is untouched and pays no timing cost. The same overflow contract holds:
    /// a match count over `max_matches` fails closed (Law 10), never a silent
    /// truncated decode.
    ///
    /// # Errors
    /// See [`Self::scan_presence_and_positions_by_region`]. On a backend whose
    /// `dispatch_borrowed_timed` only records host wall time, `device_ns` is `None`
    /// (loud absence, not a fabricated zero).
    pub fn scan_presence_and_positions_by_region_timed<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        max_matches: u32,
        matches: &mut Vec<Match>,
    ) -> Result<(Vec<u32>, vyre_driver::TimedDispatchResult), vyre::BackendError> {
        use crate::scan::dispatch_io;

        matches.clear();
        let (program, inputs, config, total_words) = self
            .build_presence_and_positions_by_region_dispatch(
                haystack,
                region_starts,
                region_base,
                max_matches,
            )?;
        let borrowed: smallvec::SmallVec<[&[u8]; 13]> = inputs.iter().map(Vec::as_slice).collect();
        let timed = backend.dispatch_borrowed_timed(&program, &borrowed, &config)?;

        // Output ordering = read_write + output buffers by binding:
        // presence(6) -> outputs[0], match_count(12) -> outputs[1], matches(13) -> outputs[2].
        let presence_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            0,
            "literal_set presence_and_positions_by_region timed presence",
        )?;
        let presence = decode_presence_words(presence_bytes, total_words);

        let count_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            1,
            "literal_set presence_and_positions_by_region timed match count",
        )?;
        let count = dispatch_io::try_read_u32_prefix(
            count_bytes,
            "literal_set presence_and_positions_by_region timed match count",
        )?;
        let matches_bytes = dispatch_io::try_output_bytes(
            &timed.outputs,
            2,
            "literal_set presence_and_positions_by_region timed matches",
        )?;
        dispatch_io::try_unpack_match_triples_capped_into(
            matches_bytes,
            count,
            max_matches,
            "literal_set presence_and_positions_by_region timed matches",
            matches,
        )?;

        Ok((presence, timed))
    }

    /// ASYNC counterpart of [`Self::scan_presence_and_positions_by_region`]: submit
    /// the fused ONE-dispatch scan and return a [`PendingFusedRegion`] handle
    /// IMMEDIATELY, so the caller can OVERLAP host-side work with the in-flight GPU
    /// scan, then decode BOTH outputs (the per-region presence bitmap and the
    /// `(pid, start, end)` triples) via [`PendingFusedRegion::await_into`].
    ///
    /// On a backend that genuinely pipelines host/device work (wgpu, cuda) the GPU
    /// scan executes while the caller does host work; on the synchronous default in
    /// [`VyreBackend::dispatch_async`] the handle is trivially ready, same outputs,
    /// no overlap, no silent change of result. Reuses the one owned-buffer prepare
    /// path ([`Self::build_presence_and_positions_by_region_dispatch`]); the
    /// [`PendingFusedRegion`] retains the owned inputs until the decode. The same
    /// fail-closed overflow contract holds (count over `max_matches` errors at the
    /// await, never a silent truncated decode).
    ///
    /// # Errors
    /// See [`Self::scan_presence_and_positions_by_region`]. Errors that surface only
    /// during GPU execution come back from [`PendingFusedRegion::await_into`].
    pub fn scan_presence_and_positions_by_region_async<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        max_matches: u32,
    ) -> Result<PendingFusedRegion, vyre::BackendError> {
        let (program, inputs, config, total_words) = self
            .build_presence_and_positions_by_region_dispatch(
                haystack,
                region_starts,
                region_base,
                max_matches,
            )?;
        let pending = backend.dispatch_async(&program, &inputs, &config)?;
        Ok(PendingFusedRegion {
            pending,
            total_words,
            max_matches,
            _inputs: inputs,
        })
    }

    fn scan_into_with_program<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        max_matches: u32,
        matches: &mut Vec<Match>,
        scratch: &mut ScanDispatchScratch,
        dispatch_program: &Program,
        prefilter_tables: &LiteralSetPrefilterTables,
    ) -> Result<(), vyre::BackendError> {
        use crate::scan::dispatch_io;

        matches.clear();
        let haystack_len =
            dispatch_io::scan_guard(haystack, "literal_set", dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let haystack_bytes = scratch.haystack_bytes.as_slice();
        let views = DfaPrefilterByteViews::new(&self.dfa, &self.pattern_lengths, prefilter_tables);

        let outputs = self.dispatch_literal_scan_outputs(
            backend,
            haystack_bytes,
            haystack_len,
            &views,
            dispatch_program,
        )?;

        decode_literal_set_outputs_into(&outputs, max_matches, matches)?;
        Ok(())
    }

    /// Dispatch the suffix3-prefiltered bounded-DFA MATCH program and return the
    /// raw `[count, matches]` output buffers. The SINGLE owner of the 10-input
    /// binding order (buffer order matches the `BufferDecl` declaration in
    /// `try_build_literal_set_program`; reordering here silently miswires the GPU
    /// program), shared by the fixed-cap [`Self::scan_into_with_program`] and the
    /// auto-resizing [`Self::scan_all_into_with_scratch`] so the two paths cannot
    /// drift in their input wiring.
    fn dispatch_literal_scan_outputs<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack_bytes: &[u8],
        haystack_len: u32,
        views: &DfaPrefilterByteViews<'_>,
        dispatch_program: &Program,
    ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let haystack_len_word = [haystack_len];
        let haystack_len_bytes = dispatch_io::u32_words_as_le_bytes(&haystack_len_word);
        // Fresh zeroed atomic match counter every dispatch: on an auto-resize
        // retry the recount must start from zero, not accumulate.
        let match_count_bytes = [0u8; 4];

        let config = dispatch_io::byte_scan_dispatch_config(
            haystack_len,
            dispatch_program.workgroup_size[0],
        );
        let borrowed_inputs = Self::literal_scan_borrowed_inputs(
            haystack_bytes,
            haystack_len_bytes.as_ref(),
            match_count_bytes.as_slice(),
            views,
        );
        backend.dispatch_borrowed(dispatch_program, &borrowed_inputs, &config)
    }

    /// TIMED sibling of [`Self::dispatch_literal_scan_outputs`]: identical staging
    /// and binding order (through the shared [`Self::literal_scan_borrowed_inputs`]
    /// owner), but calls `dispatch_borrowed_timed` so the raw `[count, matches]`
    /// outputs arrive inside a [`vyre_driver::TimedDispatchResult`]. The untimed
    /// path is byte-identical and pays no timing cost.
    fn dispatch_literal_scan_outputs_timed<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack_bytes: &[u8],
        haystack_len: u32,
        views: &DfaPrefilterByteViews<'_>,
        dispatch_program: &Program,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let haystack_len_word = [haystack_len];
        let haystack_len_bytes = dispatch_io::u32_words_as_le_bytes(&haystack_len_word);
        let match_count_bytes = [0u8; 4];
        let config = dispatch_io::byte_scan_dispatch_config(
            haystack_len,
            dispatch_program.workgroup_size[0],
        );
        let borrowed_inputs = Self::literal_scan_borrowed_inputs(
            haystack_bytes,
            haystack_len_bytes.as_ref(),
            match_count_bytes.as_slice(),
            views,
        );
        backend.dispatch_borrowed_timed(dispatch_program, &borrowed_inputs, &config)
    }

    /// SINGLE owner of the 10-input binding order for the literal MATCH program
    /// (buffer order matches the `BufferDecl` declaration in
    /// `try_build_literal_set_program`; reordering here silently miswires the GPU
    /// program). Shared by the untimed [`Self::dispatch_literal_scan_outputs`] and
    /// the timed [`Self::dispatch_literal_scan_outputs_timed`] so the two dispatch
    /// paths cannot drift in their input wiring. Binding 10 (`matches`) is a pure
    /// `BufferDecl::output` the backend allocates from the Program, so it is not an
    /// input here.
    fn literal_scan_borrowed_inputs<'a>(
        haystack_bytes: &'a [u8],
        haystack_len_bytes: &'a [u8],
        match_count_bytes: &'a [u8],
        views: &'a DfaPrefilterByteViews<'a>,
    ) -> smallvec::SmallVec<[&'a [u8]; 10]> {
        [
            haystack_bytes,                         // 0: haystack (Packed U32)
            views.transitions.as_ref(),             // 1: transitions
            views.output_offsets.as_ref(),          // 2: output_offsets
            views.output_records.as_ref(),          // 3: output_records
            views.pattern_lengths.as_ref(),         // 4: pattern_lengths
            haystack_len_bytes,                     // 5: haystack_len
            match_count_bytes,                      // 6: match_count atomic counter
            views.candidate_end_mask.as_ref(),      // 7: candidate_end_mask
            views.candidate_suffix2_mask.as_ref(),  // 8: candidate_suffix2_mask
            views.candidate_suffix3_bloom.as_ref(), // 9: candidate_suffix3_bloom
        ]
        .into_iter()
        .collect()
    }

    fn count_with_program<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        scratch: &mut ScanDispatchScratch,
        count_program: &Program,
        prefilter_tables: &LiteralSetPrefilterTables,
    ) -> Result<u32, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let haystack_len =
            dispatch_io::scan_guard(haystack, "literal_set", dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let haystack_bytes = scratch.haystack_bytes.as_slice();
        let transition_bytes = dispatch_io::u32_words_as_le_bytes(&self.dfa.transitions);
        let output_offset_bytes = dispatch_io::u32_words_as_le_bytes(&self.dfa.output_offsets);
        let candidate_end_mask_bytes =
            dispatch_io::u32_words_as_le_bytes(&prefilter_tables.candidate_end_mask);
        let candidate_suffix2_mask_bytes =
            dispatch_io::u32_words_as_le_bytes(&prefilter_tables.candidate_suffix2_mask);
        let candidate_suffix3_bloom_bytes =
            dispatch_io::u32_words_as_le_bytes(&prefilter_tables.candidate_suffix3_bloom);
        let haystack_len_word = [haystack_len];
        let haystack_len_bytes = dispatch_io::u32_words_as_le_bytes(&haystack_len_word);
        let match_count_bytes = [0u8; U32_COUNTER_BYTES];
        let config =
            dispatch_io::byte_scan_dispatch_config(haystack_len, count_program.workgroup_size[0]);

        let borrowed_inputs: smallvec::SmallVec<[&[u8]; 8]> = [
            // 0: haystack (Packed U32)
            haystack_bytes,
            // 1: transitions
            transition_bytes.as_ref(),
            // 2: output_offsets
            output_offset_bytes.as_ref(),
            // 3: candidate_end_mask
            candidate_end_mask_bytes.as_ref(),
            // 4: candidate_suffix2_mask
            candidate_suffix2_mask_bytes.as_ref(),
            // 5: candidate_suffix3_bloom
            candidate_suffix3_bloom_bytes.as_ref(),
            // 6: haystack_len
            haystack_len_bytes.as_ref(),
            // 7: match_count atomic counter and readback
            match_count_bytes.as_slice(),
        ]
        .into_iter()
        .collect();
        let outputs = backend.dispatch_borrowed(count_program, &borrowed_inputs, &config)?;
        decode_literal_set_count_outputs(&outputs)
    }

    fn prepare_scan_dispatch_with_program(
        &self,
        haystack: &[u8],
        max_matches: u32,
        dispatch_program: &Program,
        prefilter_tables: &LiteralSetPrefilterTables,
    ) -> Result<LiteralSetPreparedScan, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let haystack_len =
            dispatch_io::scan_guard(haystack, "literal_set", dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        let (_, matches_output_bytes) = literal_set_match_output_layout(max_matches)?;
        let mut inputs = Vec::new();
        vyre_foundation::allocation::try_reserve_vec_to_capacity(
            &mut inputs,
            LITERAL_SET_INPUT_COUNT,
        )
        .map_err(|source| {
            vyre::BackendError::new(format!(
                "literal_set prepared scan could not reserve {LITERAL_SET_INPUT_COUNT} input buffer slot(s): {source}. Fix: shard the literal set or haystack before preparing resident dispatch."
            ))
        })?;

        let mut haystack_bytes = Vec::new();
        dispatch_io::pack_haystack_u32_into(haystack, &mut haystack_bytes)?;
        inputs.push(haystack_bytes);
        inputs.push(copy_u32_words_as_le_bytes(
            &self.dfa.transitions,
            "transition table",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &self.dfa.output_offsets,
            "output offset table",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &self.dfa.output_records,
            "output record table",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &self.pattern_lengths,
            "pattern length table",
        )?);
        inputs.push(haystack_len.to_le_bytes().to_vec());
        inputs.push(vec![0_u8; U32_COUNTER_BYTES]);
        inputs.push(copy_u32_words_as_le_bytes(
            &prefilter_tables.candidate_end_mask,
            "candidate end mask",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &prefilter_tables.candidate_suffix2_mask,
            "candidate suffix2 mask",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &prefilter_tables.candidate_suffix3_bloom,
            "candidate suffix3 bloom",
        )?);

        let encoded_input_bytes = inputs.iter().try_fold(0_u64, |sum, input| {
            let len = u64::try_from(input.len()).map_err(|source| {
                vyre::BackendError::new(format!(
                    "literal_set prepared scan input byte length does not fit u64: {source}. Fix: shard the scan before dispatch."
                ))
            })?;
            sum.checked_add(len).ok_or_else(|| {
                vyre::BackendError::new(
                    "literal_set prepared scan input byte total overflowed u64. Fix: shard the scan before dispatch.",
                )
            })
        })?;

        Ok(LiteralSetPreparedScan {
            program: dispatch_program.clone(),
            inputs,
            dispatch_config: dispatch_io::byte_scan_dispatch_config(
                haystack_len,
                dispatch_program.workgroup_size[0],
            ),
            haystack_len,
            max_matches,
            matches_output_bytes,
            encoded_input_bytes,
        })
    }

    fn prepare_count_dispatch_with_program(
        &self,
        haystack: &[u8],
        count_program: &Program,
        prefilter_tables: &LiteralSetPrefilterTables,
    ) -> Result<LiteralSetPreparedCount, vyre::BackendError> {
        use crate::scan::dispatch_io;

        let haystack_len =
            dispatch_io::scan_guard(haystack, "literal_set", dispatch_io::DEFAULT_MAX_SCAN_BYTES)?;
        let mut inputs = Vec::new();
        vyre_foundation::allocation::try_reserve_vec_to_capacity(
            &mut inputs,
            LITERAL_SET_COUNT_INPUT_COUNT,
        )
        .map_err(|source| {
            vyre::BackendError::new(format!(
                "literal_set prepared count could not reserve {LITERAL_SET_COUNT_INPUT_COUNT} input buffer slot(s): {source}. Fix: shard the literal set or haystack before preparing resident dispatch."
            ))
        })?;

        let mut haystack_bytes = Vec::new();
        dispatch_io::pack_haystack_u32_into(haystack, &mut haystack_bytes)?;
        inputs.push(haystack_bytes);
        inputs.push(copy_u32_words_as_le_bytes(
            &self.dfa.transitions,
            "transition table",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &self.dfa.output_offsets,
            "output offset table",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &prefilter_tables.candidate_end_mask,
            "candidate end mask",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &prefilter_tables.candidate_suffix2_mask,
            "candidate suffix2 mask",
        )?);
        inputs.push(copy_u32_words_as_le_bytes(
            &prefilter_tables.candidate_suffix3_bloom,
            "candidate suffix3 bloom",
        )?);
        inputs.push(haystack_len.to_le_bytes().to_vec());
        inputs.push(vec![0_u8; U32_COUNTER_BYTES]);

        let encoded_input_bytes = inputs.iter().try_fold(0_u64, |sum, input| {
            let len = u64::try_from(input.len()).map_err(|source| {
                vyre::BackendError::new(format!(
                    "literal_set prepared count input byte length does not fit u64: {source}. Fix: shard the scan before dispatch."
                ))
            })?;
            sum.checked_add(len).ok_or_else(|| {
                vyre::BackendError::new(
                    "literal_set prepared count input byte total overflowed u64. Fix: shard the scan before dispatch.",
                )
            })
        })?;

        Ok(LiteralSetPreparedCount {
            program: count_program.clone(),
            inputs,
            dispatch_config: dispatch_io::byte_scan_dispatch_config(
                haystack_len,
                count_program.workgroup_size[0],
            ),
            haystack_len,
            encoded_input_bytes,
        })
    }

    fn prefilter_tables_cached<'a>(
        &'a self,
        cached_prefilter: &'a mut Option<LiteralSetPrefilterTables>,
    ) -> Result<&'a LiteralSetPrefilterTables, vyre::BackendError> {
        let pattern_fingerprint = self.pattern_fingerprint();
        let reuse_cached = cached_prefilter
            .as_ref()
            .is_some_and(|cached| cached.pattern_fingerprint == pattern_fingerprint);
        if !reuse_cached {
            *cached_prefilter =
                Some(self.build_prefilter_tables_with_fingerprint(pattern_fingerprint)?);
        }
        cached_prefilter.as_ref().ok_or_else(|| {
            vyre::BackendError::new(
                "literal_set failed to retain cached suffix-prefilter tables. Fix: retry with generic ScanDispatchScratch.",
            )
        })
    }

    fn build_prefilter_tables(&self) -> Result<LiteralSetPrefilterTables, vyre::BackendError> {
        self.build_prefilter_tables_with_fingerprint(self.pattern_fingerprint())
    }

    fn count_program_cached<'a>(
        &'a self,
        cached_count_program: &'a mut Option<CachedLiteralSetCountProgram>,
    ) -> Result<&'a Program, vyre::BackendError> {
        let pattern_fingerprint = self.pattern_fingerprint();
        let reuse_cached = cached_count_program
            .as_ref()
            .is_some_and(|cached| cached.pattern_fingerprint == pattern_fingerprint);
        if !reuse_cached {
            *cached_count_program = Some(CachedLiteralSetCountProgram {
                pattern_fingerprint,
                program: self.count_program(),
            });
        }
        cached_count_program
            .as_ref()
            .map(|cached| &cached.program)
            .ok_or_else(|| {
                vyre::BackendError::new(
                    "literal_set failed to retain the cached count program. Fix: retry without reusable scratch.",
                )
            })
    }

    fn count_program(&self) -> Program {
        build_ac_bounded_count_suffix3_prefilter_program(&self.dfa)
    }

    fn build_prefilter_tables_with_fingerprint(
        &self,
        pattern_fingerprint: u64,
    ) -> Result<LiteralSetPrefilterTables, vyre::BackendError> {
        let pattern_vectors = self.materialize_pattern_bytes()?;
        let pattern_refs = pattern_vectors
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        // Case-insensitive matching folds the DFA transition table, but the
        // suffix prefilter is checked against the RAW haystack byte, which the
        // kernel does not fold, so the masks must admit BOTH cases of each
        // pattern byte or an uppercase candidate would be rejected before the
        // DFA replay (a silent under-fire). One flag drives all three tables.
        let ci = self.case_insensitive;
        Ok(LiteralSetPrefilterTables {
            pattern_fingerprint,
            candidate_end_mask: literal_set_candidate_end_byte_mask_words(&pattern_refs, ci),
            candidate_suffix2_mask: literal_set_candidate_suffix2_mask_words(&pattern_refs, ci),
            candidate_suffix3_bloom: classic_ac_candidate_suffix3_bloom_words_ci(&pattern_refs, ci),
        })
    }

    fn materialize_pattern_bytes(&self) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
        if self.pattern_offsets.len() != self.pattern_lengths.len() {
            return Err(vyre::BackendError::new(format!(
                "literal_set pattern metadata is malformed: {} offsets for {} lengths. Fix: rebuild the literal set with GpuLiteralSet::try_compile before dispatch.",
                self.pattern_offsets.len(),
                self.pattern_lengths.len()
            )));
        }

        let mut patterns = Vec::new();
        vyre_foundation::allocation::try_reserve_vec_to_capacity(
            &mut patterns,
            self.pattern_lengths.len(),
        )
        .map_err(|source| {
            vyre::BackendError::new(format!(
                "literal_set could not reserve {} decoded pattern slot(s): {source}. Fix: shard the pattern set before dispatch.",
                self.pattern_lengths.len()
            ))
        })?;

        for (pattern_index, (&offset, &len)) in self
            .pattern_offsets
            .iter()
            .zip(&self.pattern_lengths)
            .enumerate()
        {
            let start = usize::try_from(offset).map_err(|source| {
                vyre::BackendError::new(format!(
                    "literal_set pattern {pattern_index} offset {offset} cannot fit host usize: {source}. Fix: rebuild the literal set with GpuLiteralSet::try_compile before dispatch."
                ))
            })?;
            let len = usize::try_from(len).map_err(|source| {
                vyre::BackendError::new(format!(
                    "literal_set pattern {pattern_index} length {len} cannot fit host usize: {source}. Fix: rebuild the literal set with GpuLiteralSet::try_compile before dispatch."
                ))
            })?;
            let end = start.checked_add(len).ok_or_else(|| {
                vyre::BackendError::new(format!(
                    "literal_set pattern {pattern_index} byte range overflows host usize. Fix: rebuild the literal set with GpuLiteralSet::try_compile before dispatch."
                ))
            })?;
            let words = self.pattern_bytes.get(start..end).ok_or_else(|| {
                vyre::BackendError::new(format!(
                    "literal_set pattern {pattern_index} byte range {start}..{end} exceeds packed pattern byte table length {}. Fix: rebuild the literal set with GpuLiteralSet::try_compile before dispatch.",
                    self.pattern_bytes.len()
                ))
            })?;
            let mut pattern = Vec::new();
            vyre_foundation::allocation::try_reserve_vec_to_capacity(&mut pattern, words.len())
                .map_err(|source| {
                    vyre::BackendError::new(format!(
                        "literal_set could not reserve {} byte(s) for pattern {pattern_index}: {source}. Fix: shard the pattern set before dispatch.",
                        words.len()
                    ))
                })?;
            for (byte_index, &word) in words.iter().enumerate() {
                let byte = u8::try_from(word).map_err(|source| {
                    vyre::BackendError::new(format!(
                        "literal_set pattern {pattern_index} byte {byte_index} has non-byte word {word}: {source}. Fix: rebuild the literal set with GpuLiteralSet::try_compile before dispatch."
                    ))
                })?;
                pattern.push(byte);
            }
            patterns.push(pattern);
        }

        Ok(patterns)
    }

    fn pattern_fingerprint(&self) -> u64 {
        // Same owner as `GpuLiteralSet::cache_key` (engine.rs) over the same
        // slices, one hash impl, no drift. The case-insensitive flag is folded
        // in because a ci and a non-ci matcher share identical pattern bytes but
        // build DIFFERENT prefilter masks; without it their cached tables collide.
        let case_word = [u32::from(self.case_insensitive)];
        crate::scan::engine::fnv1a64_word_slices([
            self.pattern_offsets.as_slice(),
            self.pattern_lengths.as_slice(),
            self.pattern_bytes.as_slice(),
            &case_word,
        ])
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
    /// includes vyre's IR wire version inside its own framing, so a stale
    /// current-version cache surfaces as `LiteralSetWireError::
    /// InvalidProgram` from `Program::from_bytes` (or as a bad magic /
    /// version on this outer envelope). Legacy literal-compare and bounded-DFA
    /// blobs are migrated by decoding their DFA/pattern sections and
    /// rebuilding the current suffix-prefiltered bounded-DFA dispatch program.
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
        // v4: the case-insensitive flag. The DFA transitions and pattern bytes
        // are identical for a case-sensitive vs case-insensitive set over the
        // same folded-lowercase patterns, so this flag, not the bytes, is what
        // makes `from_bytes` rebuild the FOLDED prefilter masks. Omitting it would
        // silently rebuild case-sensitive masks and under-fire on uppercase input.
        w.write_words(&[u32::from(self.case_insensitive)])
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
        let (mut r, wire_version) =
            literal_set_wire_reader(bytes).map_err(LiteralSetWireError::WireFraming)?;

        let program_bytes = r.read_section().map_err(LiteralSetWireError::WireFraming)?;
        if wire_version == LITERAL_SET_WIRE_VERSION {
            Program::from_bytes(program_bytes)
                .map_err(|e| LiteralSetWireError::InvalidProgram(format!("{e}")))?;
        }

        let dfa_bytes = r.read_section().map_err(LiteralSetWireError::WireFraming)?;
        let dfa = CompiledDfa::from_bytes(dfa_bytes).map_err(LiteralSetWireError::InvalidDfa)?;

        let pattern_offsets = r.read_words().map_err(LiteralSetWireError::WireFraming)?;
        let pattern_lengths = r.read_words().map_err(LiteralSetWireError::WireFraming)?;
        let pattern_bytes = r.read_words().map_err(LiteralSetWireError::WireFraming)?;
        // The case-insensitive flag is a v4+ trailing section. Legacy blobs
        // (v1/v2/v3) predate it and were always case-sensitive → default false.
        let case_insensitive = if wire_version == LITERAL_SET_WIRE_VERSION {
            let flag = r.read_words().map_err(LiteralSetWireError::WireFraming)?;
            flag.first().copied().unwrap_or(0) != 0
        } else {
            false
        };
        let pattern_count =
            u32::try_from(pattern_lengths.len()).map_err(|source| {
                LiteralSetWireError::InvalidProgram(format!(
                    "literal_set decoded pattern length count {} exceeds u32 GPU buffer metadata: {source}. Fix: shard the pattern set before caching.",
                    pattern_lengths.len()
                ))
            })?;
        // Cross-section invariant: the DFA output table (decoded independently
        // of the pattern arrays) emits pattern ids that index `pattern_lengths`
        // in `reference_scan` and GPU post-process. A stale/crafted blob whose
        // DFA references an id >= pattern_lengths.len() must fail closed here,
        // not OOB-panic the pub reference oracle.
        if let Some(&max_id) = dfa.output_records.iter().max() {
            if max_id as usize >= pattern_lengths.len() {
                return Err(LiteralSetWireError::InvalidProgram(format!(
                    "literal_set decoded DFA emits pattern id {max_id} but only {} pattern length(s) were decoded. Fix: the cache is stale/corrupt; recompile the literal set.",
                    pattern_lengths.len()
                )));
            }
        }
        let program = try_build_literal_set_program(&dfa, pattern_count).map_err(|message| {
            LiteralSetWireError::InvalidProgram(format!(
                "literal_set decoded DFA cannot rebuild current dispatch Program: {message}"
            ))
        })?;

        Ok(Self {
            dfa,
            pattern_bytes,
            pattern_offsets,
            pattern_lengths,
            program,
            case_insensitive,
        })
    }
}

fn literal_set_wire_reader(
    bytes: &[u8],
) -> Result<
    (vyre_foundation::serial::envelope::WireReader<'_>, u32),
    vyre_foundation::serial::envelope::EnvelopeError,
> {
    match vyre_foundation::serial::envelope::WireReader::new(
        bytes,
        LITERAL_SET_WIRE_MAGIC,
        LITERAL_SET_WIRE_VERSION,
    ) {
        Ok(reader) => Ok((reader, LITERAL_SET_WIRE_VERSION)),
        Err(vyre_foundation::serial::envelope::EnvelopeError::VersionMismatch {
            found:
                legacy_version @ (LITERAL_SET_LEGACY_LITERAL_COMPARE_WIRE_VERSION
                | LITERAL_SET_LEGACY_BOUNDED_DFA_WIRE_VERSION
                | LITERAL_SET_LEGACY_CASE_SENSITIVE_WIRE_VERSION),
            ..
        }) => vyre_foundation::serial::envelope::WireReader::new(
            bytes,
            LITERAL_SET_WIRE_MAGIC,
            legacy_version,
        )
        .map(|reader| (reader, legacy_version)),
        Err(error) => Err(error),
    }
}

fn literal_set_candidate_end_byte_mask_words(
    patterns: &[&[u8]],
    case_insensitive: bool,
) -> [u32; 8] {
    let mut mask = [0_u32; 8];
    for pattern in patterns
        .iter()
        .copied()
        .filter(|pattern| !pattern.is_empty())
    {
        // The raw end byte may be either case under case-insensitive matching.
        let (variants, n) = ascii_case_variants(pattern[pattern.len() - 1], case_insensitive);
        for &byte in &variants[..n] {
            let byte = usize::from(byte);
            mask[byte / 32] |= 1_u32 << (byte % 32);
        }
    }
    mask
}

fn literal_set_candidate_suffix2_mask_words(
    patterns: &[&[u8]],
    case_insensitive: bool,
) -> [u32; CLASSIC_AC_SUFFIX2_MASK_WORDS] {
    let mut mask = [0_u32; CLASSIC_AC_SUFFIX2_MASK_WORDS];
    for pattern in patterns
        .iter()
        .copied()
        .filter(|pattern| !pattern.is_empty())
    {
        match pattern.len() {
            1 => {
                // Length-1: any previous byte, the single byte in either case.
                let (cv, cn) = ascii_case_variants(pattern[0], case_insensitive);
                for &current in &cv[..cn] {
                    let current = usize::from(current);
                    for previous in 0..=u8::MAX {
                        set_suffix2_candidate_bit(&mut mask, usize::from(previous), current);
                    }
                }
            }
            len => {
                // Every case combination of the raw 2-byte suffix.
                let (pv, pn) = ascii_case_variants(pattern[len - 2], case_insensitive);
                let (cv, cn) = ascii_case_variants(pattern[len - 1], case_insensitive);
                for &previous in &pv[..pn] {
                    for &current in &cv[..cn] {
                        set_suffix2_candidate_bit(
                            &mut mask,
                            usize::from(previous),
                            usize::from(current),
                        );
                    }
                }
            }
        }
    }
    mask
}

fn set_suffix2_candidate_bit(
    mask: &mut [u32; CLASSIC_AC_SUFFIX2_MASK_WORDS],
    previous: usize,
    current: usize,
) {
    let suffix = (previous << 8) | current;
    mask[suffix / 32] |= 1_u32 << (suffix % 32);
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

/// Validate the region-presence precondition shared by every region-presence
/// entry point (sync, async-build, and positions). `ctx` is the error-message
/// prefix that names the calling surface; the checks and their wording are
/// otherwise identical, so they live here in ONE place. Returns
/// `(pattern_count, region_count)` as the GPU-ABI `u32`s the callers need.
fn validate_region_starts(
    region_starts: &[u32],
    pattern_lengths: &[u32],
    ctx: &str,
) -> Result<(u32, u32), vyre::BackendError> {
    let pattern_count = u32::try_from(pattern_lengths.len()).map_err(|_| {
        vyre::BackendError::new(format!("{ctx}: pattern count exceeds u32 GPU ABI"))
    })?;
    let region_count = u32::try_from(region_starts.len())
        .map_err(|_| vyre::BackendError::new(format!("{ctx}: region count exceeds u32 GPU ABI")))?;
    if region_count == 0 {
        return Err(vyre::BackendError::new(format!(
            "{ctx}: region_starts must be non-empty. Fix: pass one start offset per coalesced file, beginning with 0."
        )));
    }
    if region_starts[0] != 0 {
        return Err(vyre::BackendError::new(format!(
            "{ctx}: region_starts[0] must be 0 (the kernel binary-search lower bound). Fix: the first coalesced file must start at offset 0."
        )));
    }
    Ok((pattern_count, region_count))
}

/// Allocate the zeroed binding-6 presence output buffer (`words * 4` bytes)
/// through the fail-closed `try_reserve` path, matching the owned/prepared
/// contract (an OOM here returns `BackendError`, never aborts the process).
/// The single owner for every presence-buffer allocation.
fn zeroed_presence_bytes(words: usize) -> Result<Vec<u8>, vyre::BackendError> {
    let byte_len = words.checked_mul(U32_BYTES).ok_or_else(|| {
        vyre::BackendError::new(
            "literal_set region-presence output byte length overflowed host usize. Fix: shard the literal set or corpus before dispatch."
                .to_string(),
        )
    })?;
    let mut bytes = Vec::new();
    vyre_foundation::allocation::try_reserve_vec_to_capacity(&mut bytes, byte_len).map_err(
        |source| {
            vyre::BackendError::new(format!(
                "literal_set region-presence could not reserve {byte_len} byte(s) for the presence output: {source}. Fix: shard the literal set or corpus before dispatch."
            ))
        },
    )?;
    bytes.resize(byte_len, 0);
    Ok(bytes)
}

fn copy_u32_words_as_le_bytes(
    words: &[u32],
    field: &'static str,
) -> Result<Vec<u8>, vyre::BackendError> {
    let byte_len = words.len().checked_mul(U32_BYTES).ok_or_else(|| {
        vyre::BackendError::new(format!(
            "literal_set prepared scan {field} byte length overflowed host usize. Fix: shard the literal set before preparing resident dispatch."
        ))
    })?;
    let mut bytes = Vec::new();
    vyre_foundation::allocation::try_reserve_vec_to_capacity(&mut bytes, byte_len).map_err(
        |source| {
            vyre::BackendError::new(format!(
                "literal_set prepared scan could not reserve {byte_len} byte(s) for {field}: {source}. Fix: shard the literal set before preparing resident dispatch."
            ))
        },
    )?;
    if cfg!(target_endian = "little") {
        bytes.extend_from_slice(bytemuck::cast_slice(words));
    } else {
        for &word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    Ok(bytes)
}

/// Decode the first `total_words` little-endian `u32` words of a presence readback
/// into `out` (cleared first). The single decoder for EVERY region-presence wire
/// result, sync, async, prepared, fused, and resident, so the bit layout has one
/// source. Trailing bytes beyond `total_words` are ignored: a resident readback may
/// return the full buffer capacity, of which only the used prefix is meaningful.
pub(crate) fn decode_presence_words_into(
    presence_bytes: &[u8],
    total_words: usize,
    out: &mut Vec<u32>,
) {
    out.clear();
    out.extend(
        presence_bytes
            .chunks_exact(4)
            .take(total_words)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
    );
}

/// [`decode_presence_words_into`] returning a freshly-allocated `Vec`.
pub(crate) fn decode_presence_words(presence_bytes: &[u8], total_words: usize) -> Vec<u32> {
    let mut out = Vec::new();
    decode_presence_words_into(presence_bytes, total_words, &mut out);
    out
}

fn decode_literal_set_outputs_into(
    outputs: &[Vec<u8>],
    max_matches: u32,
    matches: &mut Vec<Match>,
) -> Result<(), vyre::BackendError> {
    let count_bytes =
        crate::scan::dispatch_io::try_output_bytes(outputs, 0, "literal_set match count")?;
    let count =
        crate::scan::dispatch_io::try_read_u32_prefix(count_bytes, "literal_set match count")?;
    let matches_bytes =
        crate::scan::dispatch_io::try_output_bytes(outputs, 1, "literal_set matches")?;

    // The kernel's atomic match counter overcounts past the fixed cap, so a
    // count over `max_matches` means matches were dropped: fail closed instead
    // of silently decoding the truncated prefix (Law 10).
    crate::scan::dispatch_io::try_unpack_match_triples_capped_into(
        matches_bytes,
        count,
        max_matches,
        "literal_set matches",
        matches,
    )
}

fn decode_literal_set_count_outputs(outputs: &[Vec<u8>]) -> Result<u32, vyre::BackendError> {
    let count_bytes = crate::scan::dispatch_io::try_output_bytes(outputs, 0, "literal_set count")?;
    crate::scan::dispatch_io::try_read_u32_prefix(count_bytes, "literal_set count")
}

fn literal_set_match_triple_bytes(count: u32) -> Result<usize, vyre::BackendError> {
    let words = count.checked_mul(MATCH_TRIPLE_WORDS).ok_or_else(|| {
        vyre::BackendError::new(format!(
            "literal_set match count {count} overflows the GPU match-output word count. Fix: lower max_matches or split the scan before dispatch."
        ))
    })?;
    usize::try_from(words)
        .ok()
        .and_then(|words| words.checked_mul(U32_BYTES))
        .ok_or_else(|| {
            vyre::BackendError::new(format!(
                "literal_set match count {count} overflows host match-output byte sizing. Fix: lower max_matches or split the scan before dispatch."
            ))
        })
}

fn literal_set_match_output_layout(max_matches: u32) -> Result<(u32, usize), vyre::BackendError> {
    let words = max_matches.checked_mul(MATCH_TRIPLE_WORDS).ok_or_else(|| {
        vyre::BackendError::new(format!(
            "literal_set max_matches={max_matches} overflows the GPU match-output word count. Fix: lower max_matches or split the scan before dispatch."
        ))
    })?;
    let byte_len = literal_set_match_triple_bytes(max_matches)?;
    Ok((words.max(1), byte_len))
}

#[cfg(test)]
mod compile_tests {
    use super::*;

    /// ONE-PLACE lock: the candidate-end-byte and candidate-suffix2 masks have
    /// TWO independent derivations, the pattern-derived builders here
    /// (`literal_set_candidate_*`, used by the presence/prefilter path) and the
    /// DFA-derived builders (`classic_ac_candidate_*`, used by the count path).
    /// They must produce byte-identical masks for the same literal set (both
    /// answer "which 1-/2-byte suffixes can complete a match"); if they ever
    /// diverge, one path's prefilter under- or over-fires relative to the other.
    /// This differential locks them so a future edit to either cannot drift
    /// silently. (The case-insensitive folding of the pattern-derived builder is
    /// covered separately by `literal_set_case_insensitive.rs`; here we compare
    /// the case-SENSITIVE forms against the DFA the same patterns compile to.)
    #[test]
    fn candidate_masks_pattern_derived_equals_dfa_derived() {
        use crate::scan::classic_ac::{
            classic_ac_candidate_end_byte_mask_words, classic_ac_candidate_suffix2_mask_words,
        };

        // Deterministic LCG; small byte alphabet so patterns share suffixes and
        // the DFA develops real failure links (the interesting case).
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        const ALPHABET: &[u8] = b"abcx_9";

        for case in 0..600 {
            let pattern_count = 1 + (next() % 8) as usize;
            let owned: Vec<Vec<u8>> = (0..pattern_count)
                .map(|_| {
                    // Include length-1 patterns (the "any previous byte" suffix2 branch).
                    let len = 1 + (next() % 5) as usize;
                    (0..len)
                        .map(|_| ALPHABET[(next() as usize) % ALPHABET.len()])
                        .collect()
                })
                .collect();
            let patterns: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            let dfa = dfa_compile(&patterns);

            assert_eq!(
                literal_set_candidate_end_byte_mask_words(&patterns, false),
                classic_ac_candidate_end_byte_mask_words(&dfa),
                "case {case}: end-byte mask must agree between pattern- and DFA-derived builders\n\
                 patterns={:?}",
                owned
                    .iter()
                    .map(|p| String::from_utf8_lossy(p).into_owned())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                literal_set_candidate_suffix2_mask_words(&patterns, false),
                classic_ac_candidate_suffix2_mask_words(&dfa),
                "case {case}: suffix2 mask must agree between pattern- and DFA-derived builders\n\
                 patterns={:?}",
                owned
                    .iter()
                    .map(|p| String::from_utf8_lossy(p).into_owned())
                    .collect::<Vec<_>>(),
            );
        }
    }

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
        observed_input_lengths: std::sync::Arc<std::sync::Mutex<Vec<Vec<usize>>>>,
    }

    impl RecordingLiteralBackend {
        fn new(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                outputs,
                observed_matches_layouts: std::sync::Arc::default(),
                observed_program_buffer_ptrs: std::sync::Arc::default(),
                observed_input_lengths: std::sync::Arc::default(),
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

        fn observed_input_lengths(&self) -> Vec<Vec<usize>> {
            self.observed_input_lengths
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
            inputs: &[&[u8]],
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
            self.observed_input_lengths
                .lock()
                .map_err(|_| vyre::BackendError::new("test observation mutex poisoned"))?
                .push(inputs.iter().map(|input| input.len()).collect());
            Ok(self.outputs.clone())
        }
    }

    #[derive(Clone)]
    struct RecordingCountBackend {
        outputs: Vec<Vec<u8>>,
        observed_input_lengths: std::sync::Arc<std::sync::Mutex<Vec<Vec<usize>>>>,
        observed_buffer_names: std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    }

    impl RecordingCountBackend {
        fn new(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                outputs,
                observed_input_lengths: std::sync::Arc::default(),
                observed_buffer_names: std::sync::Arc::default(),
            }
        }

        fn observed_input_lengths(&self) -> Vec<Vec<usize>> {
            self.observed_input_lengths
                .lock()
                .expect("Fix: recording count backend mutex should not be poisoned")
                .clone()
        }

        fn observed_buffer_names(&self) -> Vec<Vec<String>> {
            self.observed_buffer_names
                .lock()
                .expect("Fix: recording count backend mutex should not be poisoned")
                .clone()
        }
    }

    impl vyre::backend::private::Sealed for RecordingCountBackend {}

    impl VyreBackend for RecordingCountBackend {
        fn id(&self) -> &'static str {
            "literal-count-recording-test"
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
            inputs: &[&[u8]],
            _config: &vyre::DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
            self.observed_input_lengths
                .lock()
                .map_err(|_| vyre::BackendError::new("test observation mutex poisoned"))?
                .push(inputs.iter().map(|input| input.len()).collect());
            self.observed_buffer_names
                .lock()
                .map_err(|_| vyre::BackendError::new("test observation mutex poisoned"))?
                .push(
                    program
                        .buffers()
                        .iter()
                        .map(|buffer| buffer.name().to_string())
                        .collect(),
                );
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

    fn decode_u32_words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn decode_reference_matches(outputs: &[vyre_reference::value::Value]) -> Vec<Match> {
        let count = decode_u32_words(&outputs[0].to_bytes())[0] as usize;
        decode_u32_words(&outputs[1].to_bytes())
            .into_iter()
            .take(count.saturating_mul(3))
            .collect::<Vec<_>>()
            .chunks_exact(3)
            .map(|chunk| Match::new(chunk[0], chunk[1], chunk[2]))
            .collect()
    }

    #[test]
    fn decode_outputs_fails_closed_when_count_exceeds_cap() {
        // Law 10 regression at the shared literal_set decode call site
        // (`decode_literal_set_outputs_into`, used by GpuLiteralSet::scan): the
        // kernel's atomic counter reports 7 matches into a buffer holding the cap
        // of 3 triples. The capped decode must error (naming the 4 dropped
        // matches), not silently return the truncated 3.
        let mut triples = Vec::new();
        for i in 0..3u32 {
            triples.extend_from_slice(&match_triple_bytes(0, i, i + 1));
        }
        let outputs = vec![match_count_bytes(7), triples];
        let mut matches = vec![Match::new(5, 5, 5)];
        let err = decode_literal_set_outputs_into(&outputs, 3, &mut matches)
            .expect_err("count 7 over cap 3 must fail closed, not truncate");
        let msg = err.to_string();
        assert!(
            msg.contains("literal_set matches")
                && msg.contains("exceeds the output-buffer cap 3")
                && msg.contains("drop 4 match(es)")
                && matches.is_empty(),
            "literal_set decode must surface the dropped-match overflow and expose no partial set: {msg}"
        );
    }

    #[test]
    fn decode_outputs_decodes_exact_set_within_cap() {
        // Positive twin: a count within the cap decodes the real triples verbatim
        // (the buffer physically holds 4 slots; the counter reports 2).
        let mut triples = Vec::new();
        triples.extend_from_slice(&match_triple_bytes(2, 0, 2));
        triples.extend_from_slice(&match_triple_bytes(5, 3, 6));
        triples.extend_from_slice(&match_triple_bytes(0, 0, 0));
        triples.extend_from_slice(&match_triple_bytes(0, 0, 0));
        let outputs = vec![match_count_bytes(2), triples];
        let mut matches = Vec::new();
        decode_literal_set_outputs_into(&outputs, 4, &mut matches)
            .expect("count 2 within cap 4 must decode");
        assert_eq!(matches, vec![Match::new(2, 0, 2), Match::new(5, 3, 6)]);
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
    fn literal_prefilter_masks_are_derived_from_literal_suffixes() {
        let patterns: [&[u8]; 3] = [b"a", b"bc", b"token"];
        let end_mask = literal_set_candidate_end_byte_mask_words(&patterns, false);
        let suffix2_mask = literal_set_candidate_suffix2_mask_words(&patterns, false);

        let end_contains = |byte: u8| {
            let byte = usize::from(byte);
            (end_mask[byte / 32] & (1_u32 << (byte % 32))) != 0
        };
        let suffix2_contains = |previous: u8, current: u8| {
            let suffix = (usize::from(previous) << 8) | usize::from(current);
            (suffix2_mask[suffix / 32] & (1_u32 << (suffix % 32))) != 0
        };

        assert!(end_contains(b'a'));
        assert!(end_contains(b'c'));
        assert!(end_contains(b'n'));
        assert!(!end_contains(b'z'));
        assert!(suffix2_contains(0, b'a'));
        assert!(suffix2_contains(u8::MAX, b'a'));
        assert!(suffix2_contains(b'b', b'c'));
        assert!(suffix2_contains(b'e', b'n'));
        assert!(!suffix2_contains(b'x', b'n'));
    }

    #[test]
    fn prepare_literal_scratch_populates_reusable_program_and_prefilter_tables() {
        let engine =
            GpuLiteralSet::try_compile(&[b"a".as_slice(), b"bc".as_slice(), b"token".as_slice()])
                .expect("Fix: small literal set must compile");
        let mut scratch = LiteralSetScanScratch::default();

        engine
            .prepare_literal_scratch(3, &mut scratch)
            .expect("Fix: literal hot-loop scratch preparation should build derived state");

        assert!(
            scratch.cached_program.is_some(),
            "Fix: non-default match cap should prepare a reusable rewritten Program."
        );
        let prefilter = scratch
            .cached_prefilter
            .as_ref()
            .expect("Fix: scratch preparation should cache suffix-prefilter tables.");
        assert_ne!(
            prefilter.candidate_end_mask, [0; 8],
            "Fix: suffix-prefilter preparation must materialize candidate-end bits."
        );
        assert!(
            prefilter
                .candidate_suffix2_mask
                .iter()
                .any(|&word| word != 0),
            "Fix: suffix-prefilter preparation must materialize suffix2 candidate bits."
        );
        assert!(
            prefilter
                .candidate_suffix3_bloom
                .iter()
                .any(|&word| word != 0),
            "Fix: suffix-prefilter preparation must materialize suffix3 candidate bits."
        );
        assert!(scratch.cached_count_program.is_none());
    }

    #[test]
    fn prepare_count_scratch_populates_count_program_and_prefilter_tables() {
        let engine =
            GpuLiteralSet::try_compile(&[b"a".as_slice(), b"bc".as_slice(), b"token".as_slice()])
                .expect("Fix: small literal set must compile");
        let mut scratch = LiteralSetScanScratch::default();

        engine
            .prepare_count_scratch(&mut scratch)
            .expect("Fix: literal count scratch preparation should build derived state");

        assert!(
            scratch.cached_count_program.is_some(),
            "Fix: count hot-loop scratch should prepare the count-only program."
        );
        assert!(
            scratch.cached_prefilter.is_some(),
            "Fix: count hot-loop scratch should prepare suffix-prefilter tables."
        );
        assert!(
            scratch.cached_program.is_none(),
            "Fix: count scratch preparation should not build match-list output programs."
        );
    }

    #[test]
    fn prepare_scan_dispatch_matches_borrowed_input_layout() {
        let engine =
            GpuLiteralSet::try_compile(&[b"a".as_slice(), b"bc".as_slice(), b"token".as_slice()])
                .expect("Fix: small literal set must compile");
        let plan = engine
            .prepare_scan_dispatch(b"xx token bc a", 3)
            .expect("Fix: prepared literal scan dispatch should own input buffers");
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = Vec::new();

        engine
            .scan_into(&backend, b"xx token bc a", 3, &mut matches)
            .expect("Fix: recording backend should accept literal scan");

        assert_eq!(plan.inputs.len(), LITERAL_SET_INPUT_COUNT);
        assert_eq!(
            backend.observed_input_lengths()[0],
            plan.inputs.iter().map(Vec::len).collect::<Vec<_>>(),
            "Fix: prepared dispatch buffers must stay in the same ABI order as direct scan dispatch."
        );
        assert_eq!(
            plan.dispatch_config.grid_override,
            Some([1, 1, 1]),
            "Fix: prepared dispatch must preserve byte-scan grid geometry."
        );
        assert_eq!(plan.match_count_readback_bytes(), U32_COUNTER_BYTES);
        assert_eq!(
            plan.match_triples_readback_bytes(u32::MAX)
                .expect("Fix: clamped readback sizing should not overflow"),
            plan.matches_output_bytes
        );
        assert_eq!(
            plan.encoded_input_bytes,
            plan.inputs
                .iter()
                .map(|input| input.len() as u64)
                .sum::<u64>()
        );
    }

    #[test]
    fn prepared_scan_decodes_resident_style_readback() {
        let engine = GpuLiteralSet::try_compile(&[b"a".as_slice(), b"bc".as_slice()])
            .expect("Fix: small literal set must compile");
        let plan = engine
            .prepare_scan_dispatch(b"abc", 2)
            .expect("Fix: prepared literal scan dispatch should build");
        let outputs = vec![
            match_count_bytes(2),
            [match_triple_bytes(0, 0, 1), match_triple_bytes(1, 1, 3)].concat(),
        ];
        let mut matches = Vec::new();

        plan.decode_outputs_into(&outputs, &mut matches)
            .expect("Fix: prepared scan decoder should read count plus match triples");

        assert_eq!(
            matches,
            vec![Match::new(0, 0, 1), Match::new(1, 1, 3)],
            "Fix: prepared dispatch decode must match public GpuLiteralSet scan semantics."
        );
    }

    #[test]
    fn literal_count_uses_count_only_program_and_readback() {
        let engine = GpuLiteralSet::try_compile(&[b"a".as_slice(), b"bc".as_slice()])
            .expect("Fix: small literal set must compile");
        let backend = RecordingCountBackend::new(vec![match_count_bytes(3)]);
        let mut scratch = LiteralSetScanScratch::default();

        let count = engine
            .count_with_literal_scratch(&backend, b"abcabc", &mut scratch)
            .expect("Fix: literal count dispatch should decode one count output");

        assert_eq!(count, 3);
        assert_eq!(
            backend.observed_input_lengths()[0].len(),
            LITERAL_SET_COUNT_INPUT_COUNT,
            "Fix: count-only dispatch must not upload output_records or pattern lengths."
        );
        assert_eq!(
            backend.observed_buffer_names()[0],
            vec![
                "haystack",
                "transitions",
                "output_offsets",
                "candidate_end_mask",
                "candidate_suffix2_mask",
                "candidate_suffix3_bloom",
                "haystack_len",
                "match_count"
            ],
            "Fix: literal count must dispatch the suffix3 count program ABI."
        );
        assert!(
            scratch.cached_count_program.is_some(),
            "Fix: count hot loops should reuse the count program."
        );
    }

    #[test]
    fn prepare_count_dispatch_matches_count_input_layout() {
        let engine = GpuLiteralSet::try_compile(&[b"a".as_slice(), b"bc".as_slice()])
            .expect("Fix: small literal set must compile");
        let plan = engine
            .prepare_count_dispatch(b"abcabc")
            .expect("Fix: prepared literal count dispatch should own input buffers");
        let backend = RecordingCountBackend::new(vec![match_count_bytes(3)]);

        let count = engine
            .count(&backend, b"abcabc")
            .expect("Fix: recording backend should accept literal count");

        assert_eq!(count, 3);
        assert_eq!(plan.inputs.len(), LITERAL_SET_COUNT_INPUT_COUNT);
        assert_eq!(
            backend.observed_input_lengths()[0],
            plan.inputs.iter().map(Vec::len).collect::<Vec<_>>(),
            "Fix: prepared count buffers must stay in the same ABI order as direct count dispatch."
        );
        assert_eq!(plan.dispatch_config.grid_override, Some([1, 1, 1]));
        assert_eq!(plan.count_readback_bytes(), U32_COUNTER_BYTES);
        assert_eq!(
            plan.decode_outputs(&[match_count_bytes(3)])
                .expect("Fix: prepared count decoder should read one u32"),
            3
        );
        assert_eq!(
            plan.encoded_input_bytes,
            plan.inputs
                .iter()
                .map(|input| input.len() as u64)
                .sum::<u64>()
        );
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
        // No LAZY panics (no fix hint); explicit panic!() fail-loud is allowed.
        assert!(
            !production.contains(".expect(") && !production.contains(".unwrap("),
            "Fix: literal_set production wrappers must not use bare .unwrap()/.expect() (use an explicit panic!() with a fix hint)."
        );
        // Law 10 regression guard: GpuLiteralSet::compile must not swallow a
        // compile error into an empty matcher (which silently matches nothing,
        // reporting every input as clean). The old arm used
        // `eprintln! + empty_after_compile_failure()`; assert it is gone and a
        // fail-loud panic!() is present.
        assert!(
            !production.contains("eprintln!(\"vyre-libs GpuLiteralSet::compile failed")
                && !production.contains("empty_after_compile_failure"),
            "Fix: GpuLiteralSet::compile must not log-and-return an empty matcher on error (fail loud via panic!() so callers use try_compile)."
        );
        assert!(
            production.contains("panic!("),
            "Fix: GpuLiteralSet::compile must panic!() on an unrepresentable pattern set, never fabricate an empty matcher."
        );
        let program_debug = format!("{:#?}", GpuLiteralSet::compile(&[b"a".as_slice()]).program);
        assert!(
            !program_debug.contains("_vyre_match_leader"),
            "Fix: literal-set GPU program must use the CUDA-lowerable append primitive, not subgroup leader append."
        );
        let engine = GpuLiteralSet::compile(&[b"a".as_slice(), b"bc".as_slice()]);
        let buffer_names = engine
            .program
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect::<Vec<_>>();
        assert_eq!(
            buffer_names,
            vec![
                "haystack",
                "transitions",
                "output_offsets",
                "output_records",
                "pattern_lengths",
                "haystack_len",
                "match_count",
                "candidate_end_mask",
                "candidate_suffix2_mask",
                "candidate_suffix3_bloom",
                "matches"
            ],
            "Fix: public literal-set dispatch must run on the suffix-prefiltered bounded DFA table layout, not the old literal-byte compare ABI."
        );
        assert!(
            !program_debug.contains("pattern_bytes")
                && !program_debug.contains("pattern_offsets")
                && !program_debug.contains("_pid")
                && !program_debug.contains("_literal_matched"),
            "Fix: literal-set GPU program must not retain the per-pattern literal compare loop."
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
    fn literal_scan_uploads_dfa_tables_instead_of_literal_compare_tables() {
        let engine = GpuLiteralSet::compile(&[
            b"AKIA".as_slice(),
            b"ghp_".as_slice(),
            b"Authorization: Bearer ".as_slice(),
        ]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = Vec::new();

        engine
            .scan_into(
                &backend,
                b"prefix Authorization: Bearer token",
                4,
                &mut matches,
            )
            .expect("Fix: literal scan should dispatch with DFA table inputs");

        assert!(matches.is_empty());
        let packed_haystack_len =
            crate::scan::dispatch_io::pack_haystack_u32(b"prefix Authorization: Bearer token")
                .len();
        let prefilter = engine
            .build_prefilter_tables()
            .expect("Fix: small literal-set prefilter tables should build");
        assert_eq!(
            backend.observed_input_lengths(),
            vec![vec![
                packed_haystack_len,
                engine.dfa.transitions.len() * U32_BYTES,
                engine.dfa.output_offsets.len() * U32_BYTES,
                engine.dfa.output_records.len() * U32_BYTES,
                engine.pattern_lengths.len() * U32_BYTES,
                U32_BYTES,
                U32_BYTES,
                prefilter.candidate_end_mask.len() * U32_BYTES,
                prefilter.candidate_suffix2_mask.len() * U32_BYTES,
                prefilter.candidate_suffix3_bloom.len() * U32_BYTES,
            ]],
            "Fix: public literal-set scan must upload haystack, DFA tables, suffix-prefilter masks, haystack_len, and match_count."
        );
    }

    #[test]
    fn literal_set_dfa_program_reference_eval_matches_public_oracle() {
        let patterns: [&[u8]; 5] = [b"a", b"bc", b"abcd", b"BEGIN", b"token"];
        let haystack = b"zabcd BEGIN token abcdbc";
        let engine = GpuLiteralSet::compile(&patterns);
        let prefilter = engine
            .build_prefilter_tables()
            .expect("Fix: small literal-set prefilter tables should build");
        let inputs = vec![
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_haystack_u32(
                haystack,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.dfa.transitions,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.dfa.output_offsets,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.dfa.output_records,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.pattern_lengths,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(&[
                haystack.len() as u32,
            ])),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(&[0])),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &prefilter.candidate_end_mask,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &prefilter.candidate_suffix2_mask,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &prefilter.candidate_suffix3_bloom,
            )),
        ];
        let outputs = vyre_reference::reference_eval(&engine.program, &inputs).expect(
            "Fix: public literal-set suffix-prefiltered bounded-DFA program should evaluate in reference backend.",
        );
        let mut actual = decode_reference_matches(&outputs);
        let mut expected = engine.reference_scan(haystack);
        actual.sort_unstable();
        expected.sort_unstable();

        assert_eq!(actual, expected);
    }

    /// CPU reference oracle for the FUSED region-presence + match-positions program.
    /// One `reference_eval` of the fused suffix3 program must produce BOTH outputs
    /// recall-identically to running the two scans separately:
    ///   - the match triples must equal `reference_scan` (the linear AC oracle), and
    ///   - each region's presence bits must equal the set of pattern ids whose match
    ///     end falls in that region.
    /// This is the soundness contract a downstream GPU phase-1 fold depends on: collapsing
    /// `scan_presence_by_region` + a separate position scan into one walk must change
    /// neither output. CPU-only (no GPU) so it runs in the lib gate.
    #[test]
    fn fused_presence_and_positions_by_region_reference_eval_matches_both_oracles() {
        // patterns -> ids 0=abc, 1=xyz, 2=BEGIN (compile preserves input order).
        let patterns: [&[u8]; 3] = [b"abc", b"xyz", b"BEGIN"];
        // Two coalesced regions; no match spans the boundary.
        //   region 0 = [0, 7)  "ooabcoo"     -> holds "abc"
        //   region 1 = [7, 21) "ooxyzooBEGINoo" -> holds "xyz" and "BEGIN"
        let haystack = b"ooabcooooxyzooBEGINoo";
        let region_starts: [u32; 2] = [0, 7];
        let pattern_count: u32 = 3;
        let region_count: u32 = 2;
        let max_matches: u32 = 64;

        let engine = GpuLiteralSet::compile(&patterns);
        let prefilter = engine
            .build_prefilter_tables()
            .expect("Fix: small literal-set prefilter tables should build");
        let total_presence_words = presence_by_region_words(pattern_count, region_count) as usize;

        // Inputs in binding order (read-write buffers passed zeroed; the matches
        // output at binding 13 is backend-allocated and not passed).
        let inputs = vec![
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_haystack_u32(
                haystack,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.dfa.transitions,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.dfa.output_offsets,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.dfa.output_records,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &engine.pattern_lengths,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(&[
                haystack.len() as u32,
            ])),
            // 6: per-region presence, zeroed.
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &vec![0u32; total_presence_words],
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &prefilter.candidate_end_mask,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &prefilter.candidate_suffix2_mask,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &prefilter.candidate_suffix3_bloom,
            )),
            // 10: region_starts, 11: region_base (0), 12: match_count (zeroed).
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(
                &region_starts,
            )),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(&[0u32])),
            vyre_reference::value::Value::from(crate::scan::dispatch_io::pack_u32_slice(&[0u32])),
        ];

        let program = try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
            &engine.dfa,
            pattern_count,
            region_count,
            max_matches,
        )
        .expect("Fix: fused presence+positions program should build for a 3-pattern set");
        let outputs = vyre_reference::reference_eval(&program, &inputs).expect(
            "Fix: fused presence+positions program should evaluate in the reference backend",
        );

        // outputs[0] = presence (read_write binding 6), [1] = match_count (12),
        // [2] = matches (13).
        let presence = decode_u32_words(&outputs[0].to_bytes());
        let count = decode_u32_words(&outputs[1].to_bytes())[0] as usize;
        let mut actual_matches: Vec<Match> = decode_u32_words(&outputs[2].to_bytes())
            .into_iter()
            .take(count.saturating_mul(3))
            .collect::<Vec<_>>()
            .chunks_exact(3)
            .map(|chunk| Match::new(chunk[0], chunk[1], chunk[2]))
            .collect();

        // (1) Positions: the fused triples must equal the linear AC oracle exactly.
        let mut expected_matches = engine.reference_scan(haystack);
        actual_matches.sort_unstable();
        expected_matches.sort_unstable();
        assert_eq!(
            actual_matches, expected_matches,
            "fused match triples must equal reference_scan; got count={count}"
        );
        assert_eq!(count, 3, "exactly abc + xyz + BEGIN should match");

        // (2) Presence: derive the expected per-region bitmap from the SAME oracle
        // matches (region = largest r with region_starts[r] <= end-1; pid<32 so one
        // word per region) and require the fused presence to equal it bit-for-bit.
        assert_eq!(
            presence.len(),
            total_presence_words,
            "fused presence must be region_count x presence_words"
        );
        let presence_words = presence_bitmap_words(pattern_count) as usize;
        let mut expected_presence = vec![0u32; total_presence_words];
        for m in &expected_matches {
            let pos = m.end - 1;
            let region = region_starts
                .iter()
                .rposition(|&start| start <= pos)
                .expect("every match position lands in a region (region_starts[0]==0)");
            expected_presence[region * presence_words + (m.pattern_id >> 5) as usize] |=
                1u32 << (m.pattern_id & 31);
        }
        assert_eq!(
            presence, expected_presence,
            "fused per-region presence must equal the region-mapped firing set"
        );
        // Concrete cross-check of the derived expectation: region 0 = {abc(id0)},
        // region 1 = {xyz(id1), BEGIN(id2)}.
        assert_eq!(
            presence[0],
            1 << 0,
            "region 0 presence must be exactly {{abc}}"
        );
        assert_eq!(
            presence[1],
            (1 << 1) | (1 << 2),
            "region 1 presence must be exactly {{xyz, BEGIN}}"
        );
    }

    #[test]
    fn scan_presence_by_region_binds_dfa_and_prefilter_views_in_declared_order() {
        // Regression guard for the shared `DfaPrefilterByteViews` byte-prep: the four
        // borrowed-dispatch scan methods (`scan_presence`, this one,
        // `scan_presence_and_positions_by_region`, `scan_into_with_program`) now fill
        // their `borrowed_inputs` array by referencing the struct's fields BY NAME, so
        // a field/binding swap would silently miswire the GPU program with no compile
        // error. This is the only DEFAULT-GATE test that drives a `scan_presence*`
        // method's binding assembly end to end (the live-GPU integration tests cover
        // the megakernel, not `GpuLiteralSet`). The three patterns are chosen so the
        // four DFA tables have mutually DISTINCT byte lengths, which is what lets the
        // observed per-binding length vector detect a swap among bindings 1..=4.
        // `"bc"` is a suffix of `"abc"`, so the `abc` accepting state emits TWO pattern
        // ids, that pushes `output_records` (binding 3) past the pattern count, so it
        // differs in length from `pattern_lengths` (binding 4) and a 3<->4 swap is
        // observable (with no suffix sharing both would be `pattern_count` long).
        let patterns: [&[u8]; 3] = [b"abc", b"bc", b"xyz"];
        let engine = GpuLiteralSet::compile(&patterns);
        let prefilter = engine
            .build_prefilter_tables()
            .expect("Fix: small literal-set prefilter tables should build");

        let haystack = b"ooabcooxyzoo";
        let region_starts: [u32; 1] = [0];
        let pattern_count = patterns.len() as u32;
        let region_count = region_starts.len() as u32;
        let total_words = presence_by_region_words(pattern_count, region_count) as usize;

        // The DFA tables must be mutually distinct in length, or the length-vector
        // assertion below could not catch a swap among them.
        let dfa_lens = [
            engine.dfa.transitions.len(),
            engine.dfa.output_offsets.len(),
            engine.dfa.output_records.len(),
            engine.pattern_lengths.len(),
        ];
        for i in 0..dfa_lens.len() {
            for j in (i + 1)..dfa_lens.len() {
                assert_ne!(
                    dfa_lens[i], dfa_lens[j],
                    "Fix: test DFA tables (transitions/output_offsets/output_records/pattern_lengths) \
                     must have distinct lengths so a binding swap is observable"
                );
            }
        }

        // `RecordingCountBackend` echoes outputs[0] (the presence buffer) and records
        // every borrowed input's byte length in binding order, with no required output
        // buffer: `RecordingLiteralBackend` insists on a `matches` buffer the
        // presence-only program does not declare.
        let backend = RecordingCountBackend::new(vec![vec![0u8; total_words * 4]]);
        let presence = engine
            .scan_presence_by_region(&backend, haystack, &region_starts)
            .expect("Fix: recording backend should accept the region-presence dispatch");
        assert_eq!(
            presence.len(),
            total_words,
            "Fix: region-presence output must be region_count x presence_words"
        );

        // Expected per-binding byte layout in declared ABI order (see
        // `scan_presence_by_region_with_scratch`): 0 haystack, 1 transitions,
        // 2 output_offsets, 3 output_records, 4 pattern_lengths, 5 haystack_len,
        // 6 presence (zeroed), 7..=9 prefilter masks, 10 region_starts, 11 region_base.
        let expected = vec![
            crate::scan::dispatch_io::pack_haystack_u32(haystack).len(),
            engine.dfa.transitions.len() * 4,
            engine.dfa.output_offsets.len() * 4,
            engine.dfa.output_records.len() * 4,
            engine.pattern_lengths.len() * 4,
            4,               // haystack_len: one u32 word
            total_words * 4, // per-region presence buffer (uploaded zeroed)
            prefilter.candidate_end_mask.len() * 4,
            prefilter.candidate_suffix2_mask.len() * 4,
            prefilter.candidate_suffix3_bloom.len() * 4,
            region_starts.len() * 4,
            4, // region_base: one u32 word
        ];
        assert_eq!(
            backend.observed_input_lengths()[0],
            expected,
            "Fix: scan_presence_by_region must bind DfaPrefilterByteViews fields in declared ABI order"
        );
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
    fn literal_scan_zero_cap_fails_closed_when_a_match_is_found() {
        // Law 10 boundary: a zero-capacity position buffer that the kernel still
        // counts a match into must FAIL CLOSED, not silently return empty.
        // `scan_into` exposes only `matches`, so returning empty here would hide a
        // real match with no signal, exactly the silent false negative the capped
        // decode forbids. A caller wanting presence/count without positions uses
        // `scan_presence_by_region`; a 0-cap positions scan that finds a match has
        // dropped it, so surface that.
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(1), Vec::new()]);
        let mut matches = vec![Match::new(99, 1, 2)];

        let err = engine
            .scan_into(&backend, b"a", 0, &mut matches)
            .expect_err("zero cap with a counted match must error, not silently drop it");
        let msg = err.to_string();
        assert!(
            msg.contains("exceeds the output-buffer cap 0")
                && msg.contains("drop 1 match(es)")
                && matches.is_empty(),
            "zero-cap overflow must name the drop and expose no partial matches: {msg}"
        );
        // The dispatch still happened: the readback observed the true count and a
        // zero-length match payload before the decode failed closed.
        assert_eq!(backend.observed_matches_layouts(), vec![(1, Some(0..0))]);
    }

    #[test]
    fn literal_scan_zero_cap_with_no_matches_is_empty_ok() {
        // The benign twin: zero cap AND zero matches is not an overflow (count 0
        // is not > cap 0), so it returns empty without error.
        let engine = GpuLiteralSet::compile(&[b"a".as_slice()]);
        let backend = RecordingLiteralBackend::new(vec![match_count_bytes(0), Vec::new()]);
        let mut matches = vec![Match::new(99, 1, 2)];

        engine
            .scan_into(&backend, b"zzz", 0, &mut matches)
            .expect("zero cap with zero matches must succeed with an empty result");
        assert!(matches.is_empty());
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

    #[test]
    fn validate_region_starts_accepts_valid_and_returns_counts() {
        let lengths = [4u32, 4, 4];
        let (pattern_count, region_count) =
            validate_region_starts(&[0, 10, 20], &lengths, "ctx").expect("valid region starts");
        assert_eq!(pattern_count, 3);
        assert_eq!(region_count, 3);
    }

    #[test]
    fn validate_region_starts_rejects_empty_and_nonzero_first() {
        let lengths = [4u32];
        let empty = validate_region_starts(&[], &lengths, "literal_set region-presence")
            .expect_err("empty region_starts must be rejected");
        assert!(
            empty
                .to_string()
                .contains("region_starts must be non-empty"),
            "got: {empty}"
        );

        let nonzero = validate_region_starts(&[5, 10], &lengths, "literal_set region-presence")
            .expect_err("region_starts[0] != 0 must be rejected");
        assert!(
            nonzero.to_string().contains("region_starts[0] must be 0"),
            "got: {nonzero}"
        );
    }

    #[test]
    fn zeroed_presence_bytes_is_word_sized_and_zero() {
        let buf = zeroed_presence_bytes(5).expect("small presence buffer must allocate");
        assert_eq!(buf.len(), 5 * U32_BYTES);
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(zeroed_presence_bytes(0).expect("zero words").len(), 0);
    }

    #[test]
    fn pattern_fingerprint_shares_one_owner_with_cache_key() {
        // Dedup lock: `pattern_fingerprint` and `GpuLiteralSet::cache_key` must
        // hash the SAME three slices through the SAME `fnv1a64_word_slices`
        // owner, so the identity hash cannot drift between the two call sites.
        use crate::scan::engine::MatchScan;
        let engine = GpuLiteralSet::compile(&[b"AKIA".as_slice(), b"ghp_".as_slice()]);
        let fp = engine.pattern_fingerprint();
        assert_eq!(format!("lit-{fp:016x}"), MatchScan::cache_key(&engine));
    }
}

/// W3-1 resident POSITION-scan plumbing: `prepare_resident_scan` uploads the
/// immutable DFA + suffix-prefilter tables ONCE, and each
/// [`ResidentLiteralScan::scan_into`] re-uploads only the haystack + resets the
/// 4-byte counter, then decodes the backend's `[count, triples]` readback.
///
/// A `MockResidentMatchBackend` records resident traffic and returns a CANNED
/// two-output buffer, so the host orchestration (seven-table-upload-once,
/// per-scan haystack stage + counter reset, eleven-binding all-resident dispatch,
/// capped decode) is validated WITHOUT a GPU. Real resident-vs-borrowed match
/// parity is asserted in the integration suite where a live wgpu backend exists.
#[cfg(test)]
mod resident_match_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use vyre::DispatchConfig as Config;
    use vyre_driver::TimedDispatchResult;

    // pattern_id order matches the compile order: key=0 .. api=7.
    const LITERALS: &[&[u8]] = &[
        b"key",
        b"token",
        b"secret",
        b"AKIA",
        b"ghp_",
        b"sk_live_",
        b"password",
        b"api",
    ];

    // The literal MATCH program binds 11 buffers: inputs 0..=9 (haystack,
    // transitions, output_offsets, output_records, pattern_lengths, haystack_len,
    // match_count[6 read_write], candidate_end_mask, candidate_suffix2_mask,
    // candidate_suffix3_bloom) + the matches OUTPUT at 10. A resident dispatch
    // resolves match_count(6) -> outputs[0] and matches(10) -> outputs[1].
    const LITERAL_MATCH_BINDINGS: usize = 11;

    /// Build the backend's canned two-output readback: outputs[0] = the atomic
    /// match count (u32 LE prefix), outputs[1] = `(pattern_id, start, end)` triples
    /// (three u32 LE words each), the exact shape the borrowed match dispatch
    /// produces, so the ONE-PLACE `decode_literal_set_outputs_into` decodes it
    /// unchanged.
    fn canned_match_outputs(count: u32, triples: &[(u32, u32, u32)]) -> Vec<Vec<u8>> {
        let mut count_buf = Vec::new();
        count_buf.extend_from_slice(&count.to_le_bytes());
        let mut triples_buf = Vec::new();
        for &(pid, start, end) in triples {
            triples_buf.extend_from_slice(&pid.to_le_bytes());
            triples_buf.extend_from_slice(&start.to_le_bytes());
            triples_buf.extend_from_slice(&end.to_le_bytes());
        }
        vec![count_buf, triples_buf]
    }

    struct MockResidentMatchBackend {
        next_id: AtomicU64,
        /// (handle_id, byte_len) for every allocate_resident call, in order.
        allocations: Mutex<Vec<(u64, usize)>>,
        /// Full (immutable-table) uploads seen.
        full_uploads: AtomicUsize,
        /// Ranged (per-scan staging) uploads seen.
        ranged_uploads: AtomicUsize,
        /// Byte lengths of every ranged upload, in order.
        ranged_upload_lens: Mutex<Vec<usize>>,
        /// Canned `[count, triples]` readback returned by the resident dispatch.
        outputs: Vec<Vec<u8>>,
    }

    impl MockResidentMatchBackend {
        fn new(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                allocations: Mutex::new(Vec::new()),
                full_uploads: AtomicUsize::new(0),
                ranged_uploads: AtomicUsize::new(0),
                ranged_upload_lens: Mutex::new(Vec::new()),
                outputs,
            }
        }
    }

    impl vyre::backend::private::Sealed for MockResidentMatchBackend {}

    impl VyreBackend for MockResidentMatchBackend {
        fn id(&self) -> &'static str {
            "mock-resident-match"
        }

        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _config: &Config,
        ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
            unreachable!("resident path does not use borrowed dispatch")
        }

        fn allocate_resident(&self, byte_len: usize) -> Result<Resource, vyre::BackendError> {
            let handle = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.allocations
                .lock()
                .expect("mock allocations mutex")
                .push((handle, byte_len));
            Ok(Resource::Resident(handle))
        }

        fn upload_resident(
            &self,
            _resource: &Resource,
            _bytes: &[u8],
        ) -> Result<(), vyre::BackendError> {
            self.full_uploads.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn upload_resident_at(
            &self,
            _resource: &Resource,
            _dst_offset_bytes: usize,
            bytes: &[u8],
        ) -> Result<(), vyre::BackendError> {
            self.ranged_uploads.fetch_add(1, Ordering::Relaxed);
            self.ranged_upload_lens
                .lock()
                .expect("mock ranged-upload mutex")
                .push(bytes.len());
            Ok(())
        }

        fn free_resident(&self, _resource: Resource) -> Result<(), vyre::BackendError> {
            Ok(())
        }

        fn dispatch_resident_timed(
            &self,
            _program: &Program,
            resources: &[Resource],
            config: &Config,
        ) -> Result<TimedDispatchResult, vyre::BackendError> {
            // Contract the consumer relies on: eleven resident bindings, every one
            // resident (the CUDA resident dispatch rejects a borrowed mix, even for
            // the tiny control buffers), and a byte-scan grid override.
            assert_eq!(
                resources.len(),
                LITERAL_MATCH_BINDINGS,
                "the literal MATCH program binds eleven buffers"
            );
            for (idx, resource) in resources.iter().enumerate() {
                assert!(
                    matches!(resource, Resource::Resident(_)),
                    "binding {idx} must be resident (no borrowed mix in a resident dispatch)"
                );
            }
            assert!(
                config.grid_override.is_some(),
                "resident position scan must supply a byte-scan grid override"
            );
            Ok(TimedDispatchResult {
                outputs: self.outputs.clone(),
                wall_ns: 0,
                device_ns: None,
                enqueue_ns: None,
                wait_ns: None,
            })
        }
    }

    #[test]
    fn prepare_uploads_tables_once_then_scans_stage_only_haystack_and_counter() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        // Canned readback: two matches already in sorted (pattern_id, start, end)
        // order, the decoder sorts, so planting them sorted keeps the assertion
        // exact. max_matches = 4 so the fixed matches buffer holds 4 triples.
        let backend =
            MockResidentMatchBackend::new(canned_match_outputs(2, &[(0, 1, 2), (1, 3, 4)]));
        let max_matches = 4u32;

        let session = matcher
            .prepare_resident_scan(&backend, 4096, max_matches)
            .expect("mock backend supports resident allocation");

        // Eleven resident allocations: haystack + 7 immutable tables +
        // haystack_len + match_count + matches, in prepare order.
        {
            let allocs = backend.allocations.lock().unwrap();
            assert_eq!(
                allocs.len(),
                LITERAL_MATCH_BINDINGS,
                "haystack + 7 tables + haystack_len + match_count + matches"
            );
            // [8] haystack_len and [9] match_count are single u32 control buffers;
            // [10] matches is sized for max_matches triples (3 u32 each).
            assert_eq!(allocs[8].1, U32_BYTES, "haystack_len control is one u32");
            assert_eq!(allocs[9].1, U32_BYTES, "match_count control is one u32");
            assert_eq!(
                allocs[10].1,
                max_matches as usize * MATCH_TRIPLE_WORDS as usize * U32_BYTES,
                "matches buffer holds max_matches triples"
            );
        }
        // The seven immutable tables upload exactly once at prepare; no ranged
        // (per-scan) staging has happened yet.
        assert_eq!(
            backend.full_uploads.load(Ordering::Relaxed),
            7,
            "seven immutable tables uploaded once each at prepare"
        );
        assert_eq!(backend.ranged_uploads.load(Ordering::Relaxed), 0);

        // Three scans; each re-stages only [haystack, match_count reset,
        // haystack_len] (the immutable tables never move again).
        let haystack = b"key__api__token";
        let mut matches: Vec<Match> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();
        for _ in 0..3 {
            session
                .scan_into(&backend, haystack, &mut matches, &mut scratch)
                .expect("resident position scan decodes the canned readback");
            // Decode parity: the canned two matches surface, sorted, every scan.
            assert_eq!(
                matches,
                vec![Match::new(0, 1, 2), Match::new(1, 3, 4)],
                "the canned [count=2, triples] readback decodes to exactly two matches"
            );
        }

        assert_eq!(
            backend.full_uploads.load(Ordering::Relaxed),
            7,
            "immutable tables are NEVER re-uploaded mid-loop"
        );
        assert_eq!(
            backend.ranged_uploads.load(Ordering::Relaxed),
            9,
            "3 scans × 3 ranged uploads (haystack, match_count reset, haystack_len)"
        );
        // Per-scan upload order is [haystack, match_count reset, haystack_len].
        // The reset (2nd) and haystack_len (3rd) are each a single u32.
        let lens = backend.ranged_upload_lens.lock().unwrap();
        let nth_of_each_scan = |offset: usize| -> Vec<usize> {
            lens.iter().skip(offset).step_by(3).copied().collect()
        };
        assert_eq!(
            nth_of_each_scan(1),
            vec![U32_BYTES, U32_BYTES, U32_BYTES],
            "the match_count reset stages exactly one zeroed u32 per scan"
        );
        assert_eq!(
            nth_of_each_scan(2),
            vec![U32_BYTES, U32_BYTES, U32_BYTES],
            "haystack_len control is one u32 per scan"
        );
    }

    #[test]
    fn scan_clears_stale_matches_before_decode() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let backend = MockResidentMatchBackend::new(canned_match_outputs(1, &[(7, 0, 3)]));
        let session = matcher
            .prepare_resident_scan(&backend, 256, 8)
            .expect("prepare resident session");

        // Pre-seed stale matches; the scan must clear them, not accumulate.
        let mut matches: Vec<Match> = vec![Match::new(99, 1, 2); 5];
        let mut scratch: Vec<u8> = Vec::new();
        session
            .scan_into(&backend, b"api", &mut matches, &mut scratch)
            .expect("resident scan");
        assert_eq!(
            matches,
            vec![Match::new(7, 0, 3)],
            "stale pre-seeded matches must be replaced by the single canned match"
        );
    }

    #[test]
    fn scan_fails_closed_when_device_count_exceeds_max_matches() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        // Counter claims 5 matches but the session's matches buffer holds only 2:
        // the atomic overcounts past the fixed cap, so decoding the truncated
        // prefix would silently drop matches (Law 10). It must fail CLOSED.
        let backend =
            MockResidentMatchBackend::new(canned_match_outputs(5, &[(0, 1, 2), (1, 3, 4)]));
        let session = matcher
            .prepare_resident_scan(&backend, 256, 2)
            .expect("prepare with a 2-match cap");

        let mut matches: Vec<Match> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();
        let err = session
            .scan_into(&backend, b"key api", &mut matches, &mut scratch)
            .expect_err("count 5 over a cap of 2 must fail closed, not truncate");
        assert!(
            err.to_string().contains("exceeds the output-buffer cap"),
            "the overflow error must name the cap breach: {err}"
        );
        assert!(
            matches.is_empty(),
            "a failed-closed decode must expose no partial match set"
        );
        // The dispatch DID run (staging happened) before the decode rejected it.
        assert_eq!(
            backend.ranged_uploads.load(Ordering::Relaxed),
            3,
            "the scan staged its three per-scan uploads before the capped decode fired"
        );
    }

    #[test]
    fn scan_rejects_haystack_larger_than_resident_capacity() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let backend = MockResidentMatchBackend::new(canned_match_outputs(0, &[]));
        let session = matcher
            .prepare_resident_scan(&backend, 8, 4)
            .expect("prepare with an 8-byte haystack capacity");

        let mut matches: Vec<Match> = vec![Match::new(3, 3, 3)];
        let mut scratch: Vec<u8> = Vec::new();
        let err = session
            .scan_into(&backend, &[b'a'; 64], &mut matches, &mut scratch)
            .expect_err("a 64-byte haystack must not fit an 8-byte resident buffer");
        assert!(
            err.to_string().contains("resident buffer holds"),
            "the capacity error must name the resident-buffer limit: {err}"
        );
        // The over-capacity batch must never stage a resident upload nor dispatch.
        assert_eq!(
            backend.ranged_uploads.load(Ordering::Relaxed),
            0,
            "a rejected over-capacity batch must not stage any resident upload"
        );
    }

    #[test]
    fn free_releases_every_resident_resource() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let backend = MockResidentMatchBackend::new(canned_match_outputs(0, &[]));
        let session = matcher
            .prepare_resident_scan(&backend, 256, 4)
            .expect("prepare resident session");
        // `free` consumes the session and must succeed against a backend that
        // accepts every free (the mock's free_resident is infallible).
        session
            .free(&backend)
            .expect("every resident resource frees");
    }
}

/// W3-1 resident FUSED presence+positions plumbing: `prepare_resident_fused_scan`
/// uploads the immutable tables ONCE, and each `ResidentFusedRegionScan::scan_into`
/// re-stages only the haystack, the region controls, and the two zeroed
/// accumulators (presence prefix + match counter), then decodes the backend's
/// `[presence, count, triples]` readback. A `MockResidentFusedBackend` records
/// resident traffic and returns a CANNED three-output buffer, so the host
/// orchestration (seven-table-upload-once, per-scan staging + double reset,
/// fourteen-binding all-resident dispatch, presence + capped-match decode) is
/// validated WITHOUT a GPU.
#[cfg(test)]
mod resident_fused_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use vyre::DispatchConfig as Config;
    use vyre_driver::TimedDispatchResult;

    const LITERALS: &[&[u8]] = &[
        b"key",
        b"token",
        b"secret",
        b"AKIA",
        b"ghp_",
        b"sk_live_",
        b"password",
        b"api",
    ];

    // The fused program binds 14 buffers (0..=13): the presence common inputs
    // 0..=9 (presence read-write at 6), region_starts (10), region_base (11),
    // match_count read-write (12), and the matches OUTPUT at 13. A resident
    // dispatch resolves presence(6) -> outputs[0], match_count(12) -> outputs[1],
    // matches(13) -> outputs[2].
    const FUSED_BINDINGS: usize = 14;

    /// Build the canned `[presence, count, triples]` readback.
    fn canned_fused_outputs(
        presence_words: &[u32],
        count: u32,
        triples: &[(u32, u32, u32)],
    ) -> Vec<Vec<u8>> {
        let mut presence = Vec::new();
        for &w in presence_words {
            presence.extend_from_slice(&w.to_le_bytes());
        }
        let mut count_buf = Vec::new();
        count_buf.extend_from_slice(&count.to_le_bytes());
        let mut triples_buf = Vec::new();
        for &(pid, start, end) in triples {
            triples_buf.extend_from_slice(&pid.to_le_bytes());
            triples_buf.extend_from_slice(&start.to_le_bytes());
            triples_buf.extend_from_slice(&end.to_le_bytes());
        }
        vec![presence, count_buf, triples_buf]
    }

    struct MockResidentFusedBackend {
        next_id: AtomicU64,
        allocations: Mutex<Vec<(u64, usize)>>,
        full_uploads: AtomicUsize,
        ranged_uploads: AtomicUsize,
        outputs: Vec<Vec<u8>>,
    }

    impl MockResidentFusedBackend {
        fn new(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                allocations: Mutex::new(Vec::new()),
                full_uploads: AtomicUsize::new(0),
                ranged_uploads: AtomicUsize::new(0),
                outputs,
            }
        }
    }

    impl vyre::backend::private::Sealed for MockResidentFusedBackend {}

    impl VyreBackend for MockResidentFusedBackend {
        fn id(&self) -> &'static str {
            "mock-resident-fused"
        }

        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _config: &Config,
        ) -> Result<Vec<Vec<u8>>, vyre::BackendError> {
            unreachable!("resident path does not use borrowed dispatch")
        }

        fn allocate_resident(&self, byte_len: usize) -> Result<Resource, vyre::BackendError> {
            let handle = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.allocations
                .lock()
                .expect("mock allocations mutex")
                .push((handle, byte_len));
            Ok(Resource::Resident(handle))
        }

        fn upload_resident(
            &self,
            _resource: &Resource,
            _bytes: &[u8],
        ) -> Result<(), vyre::BackendError> {
            self.full_uploads.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn upload_resident_at(
            &self,
            _resource: &Resource,
            _dst_offset_bytes: usize,
            _bytes: &[u8],
        ) -> Result<(), vyre::BackendError> {
            self.ranged_uploads.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn free_resident(&self, _resource: Resource) -> Result<(), vyre::BackendError> {
            Ok(())
        }

        fn dispatch_resident_timed(
            &self,
            _program: &Program,
            resources: &[Resource],
            config: &Config,
        ) -> Result<TimedDispatchResult, vyre::BackendError> {
            assert_eq!(
                resources.len(),
                FUSED_BINDINGS,
                "the fused program binds fourteen buffers"
            );
            for (idx, resource) in resources.iter().enumerate() {
                assert!(
                    matches!(resource, Resource::Resident(_)),
                    "binding {idx} must be resident (no borrowed mix in a resident dispatch)"
                );
            }
            assert!(
                config.grid_override.is_some(),
                "resident fused scan must supply a byte-scan grid override"
            );
            Ok(TimedDispatchResult {
                outputs: self.outputs.clone(),
                wall_ns: 0,
                device_ns: None,
                enqueue_ns: None,
                wait_ns: None,
            })
        }
    }

    #[test]
    fn prepare_uploads_tables_once_then_scans_stage_haystack_controls_and_two_resets() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let max_regions = 4u32;
        let max_matches = 4u32;
        // 8 patterns -> 1 presence word/region. A 3-region batch -> 3 used words.
        // Plant [row0,row1,row2] + a stale 4th the 3-region decode must ignore.
        let row0 = (1 << 0) | (1 << 3); // {key, AKIA}
        let row1 = 1 << 4; // {ghp_}
        let row2 = 0u32; // {}
        let stale = 0xDEAD_BEEFu32;
        let backend = MockResidentFusedBackend::new(canned_fused_outputs(
            &[row0, row1, row2, stale],
            2,
            &[(0, 1, 2), (1, 3, 4)],
        ));

        let session = matcher
            .prepare_resident_fused_scan(&backend, 4096, max_regions, max_matches)
            .expect("mock backend supports resident allocation");
        assert_eq!(session.max_regions(), max_regions);
        assert_eq!(session.max_matches(), max_matches);

        // Fourteen resident allocations: haystack + 7 tables + presence +
        // haystack_len + region_starts + region_base + match_count + matches.
        {
            let allocs = backend.allocations.lock().unwrap();
            assert_eq!(
                allocs.len(),
                FUSED_BINDINGS,
                "haystack + 7 tables + presence + haystack_len + region_starts + region_base + match_count + matches"
            );
            // [8] presence = max_regions × 1 word × 4; [9] haystack_len u32;
            // [10] region_starts = max_regions × 4; [11] region_base u32;
            // [12] match_count u32; [13] matches = max_matches × 3 × 4.
            assert_eq!(
                allocs[8].1,
                max_regions as usize * U32_BYTES,
                "presence sized for max_regions"
            );
            assert_eq!(allocs[9].1, U32_BYTES, "haystack_len control is one u32");
            assert_eq!(
                allocs[10].1,
                max_regions as usize * U32_BYTES,
                "region_starts sized for max_regions"
            );
            assert_eq!(allocs[11].1, U32_BYTES, "region_base control is one u32");
            assert_eq!(allocs[12].1, U32_BYTES, "match_count control is one u32");
            assert_eq!(
                allocs[13].1,
                max_matches as usize * MATCH_TRIPLE_WORDS as usize * U32_BYTES,
                "matches buffer holds max_matches triples"
            );
        }
        // Seven immutable tables uploaded once at prepare; no ranged staging yet.
        assert_eq!(backend.full_uploads.load(Ordering::Relaxed), 7);
        assert_eq!(backend.ranged_uploads.load(Ordering::Relaxed), 0);

        // Three scans of a 3-region batch. Each re-stages exactly SIX ranged
        // uploads: haystack, presence reset, haystack_len, region_base,
        // match_count reset, region_starts.
        let haystack = b"key\nghp_\nzzz\n";
        let region_starts = [0u32, 4, 9];
        let mut out: Vec<u32> = Vec::new();
        let mut matches: Vec<Match> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();
        for _ in 0..3 {
            session
                .scan_into(
                    &backend,
                    haystack,
                    &region_starts,
                    0,
                    &mut out,
                    &mut matches,
                    &mut scratch,
                )
                .expect("resident fused scan decodes canned presence + matches");
            assert_eq!(
                out,
                vec![row0, row1, row2],
                "3 regions × 1 word, stale tail ignored"
            );
            assert_eq!(
                matches,
                vec![Match::new(0, 1, 2), Match::new(1, 3, 4)],
                "the canned [count=2, triples] readback decodes to exactly two matches"
            );
        }

        assert_eq!(
            backend.full_uploads.load(Ordering::Relaxed),
            7,
            "immutable tables are NEVER re-uploaded mid-loop"
        );
        assert_eq!(
            backend.ranged_uploads.load(Ordering::Relaxed),
            18,
            "3 scans × 6 ranged uploads (haystack, presence reset, haystack_len, region_base, match_count reset, region_starts)"
        );
    }

    #[test]
    fn scan_fails_closed_when_device_count_exceeds_max_matches() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        // Counter claims 5 but the matches buffer holds only 2 -> fail closed.
        let backend = MockResidentFusedBackend::new(canned_fused_outputs(
            &[0u32],
            5,
            &[(0, 1, 2), (1, 3, 4)],
        ));
        let session = matcher
            .prepare_resident_fused_scan(&backend, 256, 1, 2)
            .expect("prepare with a 2-match cap");
        let mut out: Vec<u32> = Vec::new();
        let mut matches: Vec<Match> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();
        let err = session
            .scan_into(
                &backend,
                b"key",
                &[0u32],
                0,
                &mut out,
                &mut matches,
                &mut scratch,
            )
            .expect_err("count 5 over a cap of 2 must fail closed, not truncate");
        assert!(
            err.to_string().contains("exceeds the output-buffer cap"),
            "the overflow error must name the cap breach: {err}"
        );
        assert!(
            matches.is_empty(),
            "a failed-closed decode exposes no partial matches"
        );
    }

    #[test]
    fn scan_rejects_region_count_over_the_cap() {
        let matcher = GpuLiteralSet::compile(LITERALS);
        let backend = MockResidentFusedBackend::new(canned_fused_outputs(&[0u32], 0, &[]));
        let session = matcher
            .prepare_resident_fused_scan(&backend, 256, 2, 4)
            .expect("prepare with a 2-region cap");
        let mut out: Vec<u32> = Vec::new();
        let mut matches: Vec<Match> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();
        let err = session
            .scan_into(
                &backend,
                b"a\nb\nc\n",
                &[0u32, 2, 4],
                0,
                &mut out,
                &mut matches,
                &mut scratch,
            )
            .expect_err("3 regions over a cap of 2 must error, not truncate");
        assert!(
            err.to_string()
                .contains("session was prepared for at most 2"),
            "cap error must name the limit: {err}"
        );
        assert_eq!(
            backend.ranged_uploads.load(Ordering::Relaxed),
            0,
            "a rejected over-cap batch must not stage any resident upload"
        );
    }
}

const LITERAL_SET_WIRE_MAGIC: &[u8; 4] = b"VLIT";
// v4 appended the case-insensitive flag section. v3 blobs remain readable as a
// legacy format (they predate the flag and were always case-sensitive), so
// existing on-disk caches load without a forced recompile.
pub(crate) const LITERAL_SET_WIRE_VERSION: u32 = 4;
const LITERAL_SET_LEGACY_CASE_SENSITIVE_WIRE_VERSION: u32 = 3;
const LITERAL_SET_LEGACY_BOUNDED_DFA_WIRE_VERSION: u32 = 2;
const LITERAL_SET_LEGACY_LITERAL_COMPARE_WIRE_VERSION: u32 = 1;

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

fn try_build_literal_set_program(dfa: &CompiledDfa, pattern_count: u32) -> Result<Program, String> {
    try_build_ac_bounded_ranges_suffix3_prefilter_program_ext(
        dfa,
        pattern_count,
        LITERAL_SET_DEFAULT_MAX_MATCHES,
        false,
    )
}
