//! Canonical NFA scan product lifecycle.
//!
//! [`ScanSession`] owns the substrate-neutral [`ScanProgram`]. Its
//! [`ScanSession::materialize`] boundary compiles the program as a validated graph,
//! attaches authenticated target bytes, and creates a typed artifact session.
//! [`MaterializedScanSession`] submits compiler-owned ABI values and decodes typed
//! completion readback. The explicit reference scan remains an independent CPU oracle.

use vyre_driver::BackendRegistration;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::ScanProgram;

use super::nfa;

const NFA_LANES: usize = vyre_primitives::nfa::subgroup_nfa::LANES_PER_SUBGROUP;

/// A scan program compiled, authenticated, and materialized for one registered target.
pub struct MaterializedScanSession {
    compiled: ScanProgram,
    artifact: crate::ScanArtifactSession,
}

/// Substrate-neutral scan program and immutable scan tables.
#[derive(Debug, Clone)]
pub struct ScanSession {
    /// Substrate-neutral program and immutable scan tables.
    pub compiled: ScanProgram,
}

impl ScanSession {
    /// Compile, target-compile, authenticate, and materialize this scan program.
    pub fn materialize(
        &self,
        registration: &'static BackendRegistration,
    ) -> Result<MaterializedScanSession, crate::ScanArtifactError> {
        let artifact = crate::ScanArtifactSession::compile(&self.compiled.program, registration)?;
        Ok(MaterializedScanSession {
            compiled: self.compiled.clone(),
            artifact,
        })
    }
}

impl MaterializedScanSession {
    /// Neutral artifact identity shared by every device generation.
    #[must_use]
    pub const fn artifact_digest(&self) -> vyre_megakernel::Digest {
        self.artifact.artifact_digest()
    }

    /// Authenticated target payload identity materialized by this session.
    #[must_use]
    pub const fn payload_digest(&self) -> vyre_megakernel::Digest {
        self.artifact.payload_digest()
    }

    /// Scan the complete haystack and return at most `max_matches` matches.
    pub fn scan(
        &self,
        haystack: &[u8],
        max_matches: u32,
    ) -> Result<Vec<Match>, crate::ScanArtifactError> {
        let mut matches = Vec::new();
        self.scan_into(haystack, max_matches, &mut matches)?;
        Ok(matches)
    }

    /// Scan with a per-workgroup cursor cap.
    pub fn scan_bounded(
        &self,
        haystack: &[u8],
        max_matches: u32,
        max_scan_bytes: u32,
    ) -> Result<Vec<Match>, crate::ScanArtifactError> {
        let mut matches = Vec::new();
        self.scan_bounded_into(haystack, max_matches, max_scan_bytes, &mut matches)?;
        Ok(matches)
    }

    /// Scan into caller-owned match storage.
    pub fn scan_into(
        &self,
        haystack: &[u8],
        max_matches: u32,
        matches: &mut Vec<Match>,
    ) -> Result<(), crate::ScanArtifactError> {
        self.scan_bounded_into(haystack, max_matches, u32::MAX, matches)
    }

    /// Bounded scan into caller-owned match storage.
    pub fn scan_bounded_into(
        &self,
        haystack: &[u8],
        max_matches: u32,
        max_scan_bytes: u32,
        matches: &mut Vec<Match>,
    ) -> Result<(), crate::ScanArtifactError> {
        let mut scratch = crate::dispatch_io::ScanDispatchScratch::default();
        self.scan_bounded_into_with_scratch(
            haystack,
            max_matches,
            max_scan_bytes,
            matches,
            &mut scratch,
        )
    }

    /// Bounded scan that reuses caller-owned staging storage.
    pub fn scan_bounded_into_with_scratch(
        &self,
        haystack: &[u8],
        max_matches: u32,
        max_scan_bytes: u32,
        matches: &mut Vec<Match>,
        scratch: &mut crate::dispatch_io::ScanDispatchScratch,
    ) -> Result<(), crate::ScanArtifactError> {
        use crate::dispatch_io;

        matches.clear();
        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "MaterializedScanSession::scan",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;
        let buffers = self.compiled.program.buffers();
        let input_name = buffers[0].name();
        let hit_name = buffers[3].name();
        let match_capacity = buffers[3].count().saturating_sub(1) / 3;
        if max_matches > match_capacity {
            return Err(vyre_driver::BackendError::new(format!(
                "scan artifact max_matches {max_matches} exceeds compiled hit capacity {match_capacity}. Fix: lower max_matches or build a scan program with a larger canonical hit ABI."
            ))
            .into());
        }
        zeroed_hit_buffer_into(match_capacity, &mut scratch.hit_bytes)?;
        dispatch_io::pack_haystack_u32_into(haystack, &mut scratch.haystack_bytes)?;
        let transition_bytes = dispatch_io::u32_words_as_le_bytes(&self.compiled.transition_table);
        let epsilon_bytes = dispatch_io::u32_words_as_le_bytes(&self.compiled.epsilon_table);
        let haystack_len_bytes = haystack_len.to_le_bytes();
        let max_scan_bytes_bytes = max_scan_bytes.to_le_bytes();

        let grid = dispatch_io::candidate_start_dispatch_config(haystack_len)
            .grid_override
            .ok_or_else(|| {
                vyre_driver::BackendError::new(
                    "materialized NFA scan geometry omitted its invocation grid",
                )
            })?;
        let completion = self.artifact.submit(
            [
                (input_name, scratch.haystack_bytes.as_slice()),
                ("nfa_transition", transition_bytes.as_ref()),
                ("nfa_epsilon", epsilon_bytes.as_ref()),
                (hit_name, scratch.hit_bytes.as_slice()),
                ("nfa_haystack_len", haystack_len_bytes.as_slice()),
                ("nfa_max_scan_bytes", max_scan_bytes_bytes.as_slice()),
            ],
            grid,
        )?;
        let hit_bytes = self.artifact.completion_value(&completion, hit_name)?;
        let count = dispatch_io::try_read_u32_prefix(hit_bytes, "scan artifact hit buffer")?;
        dispatch_io::try_unpack_match_triples_capped_into(
            &hit_bytes[4..],
            count,
            max_matches,
            "scan artifact hit buffer",
            matches,
        )?;
        Ok(())
    }
}

impl ScanSession {
    /// Compute matches against `haystack` on the CPU using the same NFA
    /// the GPU program runs. Mirrors [`super::GpuLiteralSet::reference_scan`]
    ///  -  same `Match` type, same sort, so any consumer can write a
    /// single parity test that swaps backends and asserts equality.
    ///
    /// This is intentionally O(n × patterns)  -  it is only meant for
    /// parity / debugging, not production scanning.
    ///
    /// # Panics
    ///
    /// Aborts when the CPU stepper cannot honor the `u32` match ABI the GPU
    /// path uses (a haystack longer than `u32::MAX` bytes). This is a LOUD
    /// failure on purpose: an empty `Vec<Match>` is indistinguishable from "no
    /// secrets here", so swallowing a scan failure into `[]` would be a silent
    /// recall lie (a >4 GiB haystack would be reported clean (Law 10)).
    /// Callers that must handle an over-`u32` haystack without unwinding use
    /// the fallible [`Self::try_reference_scan`] instead.
    #[must_use]
    pub fn reference_scan(&self, haystack: &[u8]) -> Vec<Match> {
        match self.try_reference_scan(haystack) {
            Ok(matches) => matches,
            Err(error) => {
                // Returning an empty match set would be indistinguishable from
                // "no secrets here", a total recall-loss silent fallback
                // (Law 10). Fail closed instead. Callers that must handle a
                // >u32 haystack without unwinding call try_reference_scan.
                panic!(
                    "vyre-libs ScanSession::reference_scan cannot honor the u32 match ABI for this haystack: {error}. \
                     returning an empty match set would silently report the input as clean; \
                     use try_reference_scan and split the haystack below u32::MAX bytes."
                )
            }
        }
    }

    /// Fallible CPU parity scan.
    ///
    /// # Errors
    ///
    /// Returns [`vyre_driver::BackendError`] when haystack positions cannot fit the
    /// same `u32` match ABI used by the GPU path.
    pub fn try_reference_scan(
        &self,
        haystack: &[u8],
    ) -> Result<Vec<Match>, vyre_driver::BackendError> {
        let mut results = Vec::new();
        self.try_reference_scan_into(haystack, &mut results)?;
        Ok(results)
    }

    /// CPU parity scan into caller-owned result storage.
    ///
    /// The NFA state words are stack-backed fixed arrays, so the parity oracle
    /// no longer allocates two subgroup vectors for every `(start, cursor)`
    /// pair while still mirroring the GPU transition-table semantics.
    ///
    /// # Errors
    ///
    /// Returns [`vyre_driver::BackendError`] when haystack positions cannot fit the
    /// same `u32` match ABI used by the GPU path.
    pub fn try_reference_scan_into(
        &self,
        haystack: &[u8],
        results: &mut Vec<Match>,
    ) -> Result<(), vyre_driver::BackendError> {
        crate::dispatch_io::scan_guard(haystack, "ScanSession::reference_scan", u32::MAX)?;
        results.clear();
        for start in 0..haystack.len() {
            let start_u32 = u32::try_from(start).map_err(|_| {
                vyre_driver::BackendError::new(
                    "ScanSession::reference_scan start offset exceeds u32 capacity. Fix: split the haystack before parity scanning.",
                )
            })?;
            let mut state = [0_u32; NFA_LANES];
            let mut next = [0_u32; NFA_LANES];
            state[0] = 1;
            // Regex-compiled plans wire the shared entry state to each pattern's start via
            // ε-edges (and `*`/`?`/alternation add more). Close ε from the seed before the
            // first byte, mirroring the GPU program's initial ε loop; without this the walk
            // never leaves the entry state and reports zero matches for any ε-bearing NFA.
            close_epsilon(
                &mut state,
                &self.compiled.epsilon_table,
                self.compiled.plan.num_states as usize,
            );
            for (cursor, &byte) in haystack.iter().enumerate().skip(start) {
                next.fill(0);
                for (lane, &peer) in state.iter().enumerate() {
                    for bit in 0..32 {
                        if (peer >> bit) & 1 == 0 {
                            continue;
                        }
                        let src_state = lane * 32 + bit;
                        if src_state >= self.compiled.plan.num_states as usize {
                            continue;
                        }
                        let base = src_state * 256 * NFA_LANES + (byte as usize) * NFA_LANES;
                        for (dst_lane, slot) in next.iter_mut().enumerate() {
                            *slot |= self.compiled.transition_table[base + dst_lane];
                        }
                    }
                }
                std::mem::swap(&mut state, &mut next);
                // ε-close the post-transition state set (same fixpoint the GPU eps loop runs).
                close_epsilon(
                    &mut state,
                    &self.compiled.epsilon_table,
                    self.compiled.plan.num_states as usize,
                );
                for (&accept_state, &(pattern_id, _pattern_len)) in self
                    .compiled
                    .plan
                    .accept_state_ids
                    .iter()
                    .zip(&self.compiled.plan.accept_states)
                {
                    let lane = (accept_state / 32) as usize;
                    let bit = accept_state % 32;
                    if lane < state.len() && (state[lane] & (1_u32 << bit)) != 0 {
                        let end_u32 = u32::try_from(cursor + 1).map_err(|_| {
                            vyre_driver::BackendError::new(
                                "ScanSession::reference_scan end offset exceeds u32 capacity. Fix: split the haystack before parity scanning.",
                            )
                        })?;
                        results.push(Match::new(pattern_id, start_u32, end_u32));
                    }
                }
            }
        }
        results.sort_unstable();
        Ok(())
    }
}

/// Epsilon-close an NFA state set in place: repeatedly OR in every state reachable by an
/// ε-edge until fixpoint. Lane-major `epsilon_table` layout is `[num_states × LANES]`
/// (`epsilon_table[src * LANES + dst_lane]` holds the destination bits `dst_lane` owns,
/// reachable from `src`), mirroring the byte `transition_table` minus the 256-byte axis.
///
/// No-op when there are no ε-edges (empty or all-zero table), so literal NFAs, whose
/// matches never depend on ε, are unaffected. Bounded to `num_states` iterations: each
/// pass advances the ε-frontier by one hop and OR is monotone, so the closure is complete.
fn close_epsilon(state: &mut [u32; NFA_LANES], epsilon_table: &[u32], num_states: usize) {
    if epsilon_table.is_empty() || num_states == 0 {
        return;
    }
    for _ in 0..num_states {
        let snapshot = *state;
        for (lane, &peer) in snapshot.iter().enumerate() {
            if peer == 0 {
                continue;
            }
            for bit in 0..32 {
                if (peer >> bit) & 1 == 0 {
                    continue;
                }
                let src_state = lane * 32 + bit;
                if src_state >= num_states {
                    continue;
                }
                let base = src_state * NFA_LANES;
                for (dst_lane, slot) in state.iter_mut().enumerate() {
                    if let Some(&bits) = epsilon_table.get(base + dst_lane) {
                        *slot |= bits;
                    }
                }
            }
        }
        if *state == snapshot {
            break; // fixpoint reached
        }
    }
}

/// Integrator entry point. Takes a pattern set + the input length the
/// pipeline will be dispatched against and returns everything program-analysis consumer
/// needs to issue a single dispatch.
///
/// Additional G-stack options land here as optional parameters  -
/// callers that don't opt in keep the current behaviour.
#[must_use]
pub fn build(patterns: &[&str], input_buf: &str, hit_buf: &str, input_len: u32) -> ScanSession {
    ScanSession {
        compiled: vyre_libs::scan::build_scan_program(patterns, input_buf, hit_buf, input_len),
    }
}

pub(crate) fn hit_buffer_byte_len(max_matches: u32) -> Result<usize, vyre_driver::BackendError> {
    let match_words = usize::try_from(max_matches)
        .map_err(|_| {
            vyre_driver::BackendError::new(
                "ScanSession::scan max_matches exceeds host usize capacity. Fix: reduce max_matches or shard the scan.",
            )
        })?
        .checked_mul(3)
        .and_then(|words| words.checked_add(1))
        .ok_or_else(|| {
            vyre_driver::BackendError::new(
                "ScanSession::scan hit-buffer word count overflowed. Fix: reduce max_matches or shard the scan.",
            )
        })?;
    match_words.checked_mul(4).ok_or_else(|| {
        vyre_driver::BackendError::new(
            "ScanSession::scan hit-buffer byte count overflowed. Fix: reduce max_matches or shard the scan.",
        )
    })
}

#[cfg(test)]
fn zeroed_hit_buffer(max_matches: u32) -> Result<Vec<u8>, vyre_driver::BackendError> {
    let byte_len = hit_buffer_byte_len(max_matches)?;
    let mut bytes = Vec::new();
    zeroed_hit_buffer_into(max_matches, &mut bytes)?;
    debug_assert_eq!(bytes.len(), byte_len);
    Ok(bytes)
}

fn zeroed_hit_buffer_into(
    max_matches: u32,
    bytes: &mut Vec<u8>,
) -> Result<(), vyre_driver::BackendError> {
    let byte_len = hit_buffer_byte_len(max_matches)?;
    bytes.clear();
    vyre_foundation::allocation::try_reserve_vec_to_capacity(bytes, byte_len).map_err(
        |source| {
            vyre_driver::BackendError::new(format!(
                "ScanSession::scan could not reserve {byte_len} hit-buffer byte(s): {source}. Fix: lower max_matches or shard the scan."
            ))
        },
    )?;
    bytes.resize(byte_len, 0);
    Ok(())
}

fn reserve_wire_vec<T>(
    vec: &mut Vec<T>,
    requested: usize,
    field: &'static str,
) -> Result<(), PipelineWireError> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(vec, requested).map_err(|source| {
        PipelineWireError::StorageReserveFailed {
            field,
            requested,
            message: source.to_string(),
        }
    })
}

const PIPELINE_WIRE_MAGIC: &[u8; 4] = b"VRPL";
// V4: nfa_scan added the `nfa_max_scan_bytes` storage buffer so the
// per-workgroup cursor cap is read from a 1-u32 input. Old V3 blobs
// encode a Program without that binding; decoding one and
// re-dispatching would crash on a missing-binding lookup. Bumping
// the version forces every cache consumer to re-compile against the
// V4 program shape.
//
// V3: nfa_scan added the `nfa_haystack_len` storage buffer so the
// runtime cursor bound is read from a 1-u32 input instead of baked
// into the compiled program. Old V2 blobs encode a Program without
// that binding; decoding one and re-dispatching would crash on a
// missing-binding lookup.
const PIPELINE_WIRE_VERSION: u32 = 4;

/// Errors returned by [`ScanSession::from_bytes`]. Mirrors the layered
/// error pattern of `LiteralSetWireError`  -  outer envelope failures
/// forward to `WireFraming`, inner failures keep typed variants.
#[derive(Debug)]
#[non_exhaustive]
pub enum PipelineWireError {
    /// Outer envelope (magic / version / section length) was rejected.
    WireFraming(vyre_foundation::serial::envelope::EnvelopeError),
    /// Nested vyre IR `Program` blob was rejected.
    InvalidProgram(String),
    /// One of the four `u32`-array sections had the wrong length to be
    /// consistent with the recorded `num_states` header field. Stale
    /// blob  -  recompile.
    ShapeMismatch {
        /// Static description of which section's length cross-check
        /// failed.
        reason: &'static str,
    },
    /// Serialization scratch storage could not be reserved.
    StorageReserveFailed {
        /// Scratch vector being reserved.
        field: &'static str,
        /// Requested target capacity.
        requested: usize,
        /// Allocator failure details.
        message: String,
    },
}

impl std::fmt::Display for PipelineWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WireFraming(e) => write!(f, "ScanSession wire envelope: {e}"),
            Self::InvalidProgram(msg) => {
                write!(f, "ScanSession wire blob has invalid Program: {msg}")
            }
            Self::ShapeMismatch { reason } => {
                write!(f, "ScanSession wire blob shape mismatch: {reason}")
            }
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "ScanSession wire serialization could not reserve {requested} {field} slot(s): {message}. Fix: shard the pattern pipeline before serialization."
            ),
        }
    }
}

impl std::error::Error for PipelineWireError {}

impl ScanSession {
    /// Serialize this pipeline into a self-describing binary blob
    /// suitable for on-disk caching. Built on the shared
    /// `vyre_foundation::serial::envelope` primitive  -  any future cache
    /// consumer reuses the same framing without re-implementing
    /// magic / version / truncation handling.
    ///
    /// Sections, in order:
    ///   - `u32`     : `plan.num_states`
    ///   - `u32`     : `plan.input_len`
    ///   - section 0 : vyre `Program::to_wire` payload
    ///   - words 1   : `transition_table` (lane-major)
    ///   - words 2   : `epsilon_table` (lane-major)
    ///   - words 3   : `plan.accept_states` flattened as
    ///                 `[pid_0, len_0, pid_1, len_1, …]`
    ///   - words 4   : `plan.accept_state_ids`
    ///   - words 5   : accept anchor flags, one bitset word per accept
    ///                 (`bit0=start`, `bit1=end`)
    ///
    /// Returns [`PipelineWireError::InvalidProgram`] when the dispatch IR
    /// cannot be represented in the stable program wire format, or
    /// [`PipelineWireError::WireFraming`] if an outer section exceeds the
    /// envelope's `u32` length-prefix capacity.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PipelineWireError> {
        let mut w = vyre_foundation::serial::envelope::WireWriter::new(
            PIPELINE_WIRE_MAGIC,
            PIPELINE_WIRE_VERSION,
        );
        w.write_u32(self.compiled.plan.num_states);
        w.write_u32(self.compiled.plan.input_len);
        let program_bytes = self
            .compiled
            .program
            .to_wire()
            .map_err(|error| PipelineWireError::InvalidProgram(error.to_string()))?;
        w.write_section(&program_bytes)
            .map_err(PipelineWireError::WireFraming)?;
        w.write_words(&self.compiled.transition_table)
            .map_err(PipelineWireError::WireFraming)?;
        w.write_words(&self.compiled.epsilon_table)
            .map_err(PipelineWireError::WireFraming)?;
        // Flatten accept_states tuples into a flat u32 array; each
        // accept-state contributes two consecutive words.
        let accept_flat_words = self
            .compiled
            .plan
            .accept_states
            .len()
            .checked_mul(2)
            .ok_or(PipelineWireError::ShapeMismatch {
                reason: "accept_states length overflows flattened word count",
            })?;
        let mut accept_flat: Vec<u32> = Vec::new();
        reserve_wire_vec(&mut accept_flat, accept_flat_words, "accept state word")?;
        for &(pid, len) in &self.compiled.plan.accept_states {
            accept_flat.push(pid);
            accept_flat.push(len);
        }
        w.write_words(&accept_flat)
            .map_err(PipelineWireError::WireFraming)?;
        w.write_words(&self.compiled.plan.accept_state_ids)
            .map_err(PipelineWireError::WireFraming)?;
        let mut anchor_flags: Vec<u32> = Vec::new();
        reserve_wire_vec(
            &mut anchor_flags,
            self.compiled.plan.accept_states.len(),
            "accept anchor flag",
        )?;
        for idx in 0..self.compiled.plan.accept_states.len() {
            let mut flags = 0u32;
            if self
                .compiled
                .plan
                .accept_start_anchored
                .get(idx)
                .copied()
                .unwrap_or(false)
            {
                flags |= 1;
            }
            if self
                .compiled
                .plan
                .accept_end_anchored
                .get(idx)
                .copied()
                .unwrap_or(false)
            {
                flags |= 2;
            }
            anchor_flags.push(flags);
        }
        w.write_words(&anchor_flags)
            .map_err(PipelineWireError::WireFraming)?;
        Ok(w.into_bytes())
    }

    /// Decode a `ScanSession` from a blob produced by
    /// [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`PipelineWireError`] when the envelope rejects the
    /// outer header, the nested `Program` is invalid, or the section
    /// shapes don't match the recorded `num_states`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PipelineWireError> {
        let mut r = vyre_foundation::serial::envelope::WireReader::new(
            bytes,
            PIPELINE_WIRE_MAGIC,
            PIPELINE_WIRE_VERSION,
        )
        .map_err(PipelineWireError::WireFraming)?;

        let num_states = r.read_u32().map_err(PipelineWireError::WireFraming)?;
        let input_len = r.read_u32().map_err(PipelineWireError::WireFraming)?;

        let program_bytes = r.read_section().map_err(PipelineWireError::WireFraming)?;
        let program = vyre_foundation::ir::Program::from_wire(program_bytes)
            .map_err(|e| PipelineWireError::InvalidProgram(format!("{e}")))?;

        let transition_table = r.read_words().map_err(PipelineWireError::WireFraming)?;
        let epsilon_table = r.read_words().map_err(PipelineWireError::WireFraming)?;

        // Cross-check the section shapes against the decoded `num_states` header
        // before any scan indexes them; a stale/crafted blob whose tables are
        // shorter than `num_states` implies must fail closed here, not OOB-panic
        // later in `try_reference_scan_into`/`close_epsilon`.
        let states = num_states as usize;
        let expected_trans = states
            .checked_mul(256)
            .and_then(|v| v.checked_mul(NFA_LANES))
            .ok_or(PipelineWireError::ShapeMismatch {
                reason: "num_states overflows transition-table length",
            })?;
        if transition_table.len() != expected_trans {
            return Err(PipelineWireError::ShapeMismatch {
                reason: "transition_table length disagrees with num_states",
            });
        }
        let expected_eps =
            states
                .checked_mul(NFA_LANES)
                .ok_or(PipelineWireError::ShapeMismatch {
                    reason: "num_states overflows epsilon-table length",
                })?;
        if epsilon_table.len() != expected_eps {
            return Err(PipelineWireError::ShapeMismatch {
                reason: "epsilon_table length disagrees with num_states",
            });
        }

        let accept_flat = r.read_words().map_err(PipelineWireError::WireFraming)?;
        let accept_state_ids = r.read_words().map_err(PipelineWireError::WireFraming)?;
        let anchor_flags = r.read_words().map_err(PipelineWireError::WireFraming)?;

        if accept_flat.len() % 2 != 0 {
            return Err(PipelineWireError::ShapeMismatch {
                reason: "accept_states array length is not even",
            });
        }
        let accept_states: Vec<(u32, u32)> =
            accept_flat.chunks_exact(2).map(|w| (w[0], w[1])).collect();
        if accept_state_ids.len() != accept_states.len() {
            return Err(PipelineWireError::ShapeMismatch {
                reason: "accept_state_ids length disagrees with accept_states length",
            });
        }
        if anchor_flags.len() != accept_states.len() {
            return Err(PipelineWireError::ShapeMismatch {
                reason: "accept anchor flag length disagrees with accept_states length",
            });
        }
        // Fail closed on a stale/crafted blob whose accept ids fall outside the
        // decoded state space. The reference scan guards each accept with
        // `lane < state.len()`, so an out-of-range id would be SILENTLY DROPPED
        // (a recall hole on a stale cache) instead of surfaced as corruption
        // exactly the Law-10 silent degrade the length checks above prevent for
        // the tables, now closed for the accept-id values too.
        if accept_state_ids.iter().any(|&id| id as usize >= states) {
            return Err(PipelineWireError::ShapeMismatch {
                reason: "accept_state_id out of range for num_states",
            });
        }
        let accept_start_anchored = anchor_flags.iter().map(|flags| flags & 1 != 0).collect();
        let accept_end_anchored = anchor_flags.iter().map(|flags| flags & 2 != 0).collect();

        Ok(ScanSession {
            compiled: ScanProgram {
                program,
                transition_table,
                epsilon_table,
                plan: nfa::NfaPlan {
                    num_states,
                    input_len,
                    accept_states,
                    accept_state_ids,
                    accept_start_anchored,
                    accept_end_anchored,
                },
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrator_returns_primitive_compatible_tables() {
        let pipe = build(&["abc"], "input", "hits", 16);
        let plan = nfa::compile(&["abc"]);
        let expected_trans_len = (plan.num_states as usize)
            * 256
            * vyre_primitives::nfa::subgroup_nfa::LANES_PER_SUBGROUP;
        let expected_eps_len =
            (plan.num_states as usize) * vyre_primitives::nfa::subgroup_nfa::LANES_PER_SUBGROUP;
        assert_eq!(pipe.compiled.transition_table.len(), expected_trans_len);
        assert_eq!(pipe.compiled.epsilon_table.len(), expected_eps_len);
    }

    #[test]
    fn integrator_plan_matches_compile() {
        let pipe = build(&["ab", "cd"], "input", "hits", 8);
        assert_eq!(pipe.compiled.plan.num_states, 5);
        assert_eq!(pipe.compiled.plan.input_len, 8);
        assert_eq!(pipe.compiled.plan.accept_states.len(), 2);
    }

    #[test]
    fn rule_pipeline_wire_roundtrips_and_scans_identically() {
        let pipe = build(&["ab", "bc"], "input", "hits", 16);
        let bytes = pipe.to_bytes().expect("valid pipeline must serialize");
        let decoded = ScanSession::from_bytes(&bytes).expect("roundtrip must decode");
        assert_eq!(
            decoded.compiled.plan.num_states,
            pipe.compiled.plan.num_states
        );
        assert_eq!(
            decoded.compiled.transition_table,
            pipe.compiled.transition_table
        );
        assert_eq!(decoded.compiled.epsilon_table, pipe.compiled.epsilon_table);
        assert_eq!(
            decoded.reference_scan(b"zabc"),
            vec![Match::new(0, 1, 3), Match::new(1, 2, 4)]
        );
    }

    /// WHY: nested program encoding must fail closed instead of emitting an
    /// empty section that can be mistaken for a valid cached pipeline.
    #[test]
    fn pipeline_wire_propagates_program_encoding_errors() {
        let mut pipe = build(&["ab"], "input", "hits", 16);
        let oversized_name = "x".repeat(vyre_foundation::serial::wire::MAX_STRING_LEN + 1);
        pipe.compiled.program = vyre_foundation::ir::Program::wrapped(
            vec![vyre_foundation::ir::BufferDecl::storage(
                &oversized_name,
                0,
                vyre_foundation::ir::BufferAccess::ReadOnly,
                vyre_foundation::ir::DataType::U32,
            )],
            [1, 1, 1],
            vec![],
        );

        let error = pipe
            .to_bytes()
            .expect_err("an unencodable nested Program must reject the pipeline");
        assert!(
            matches!(error, PipelineWireError::InvalidProgram(_)),
            "nested Program encoding failures must retain their typed boundary: {error}"
        );
    }

    #[test]
    fn rule_pipeline_from_bytes_rejects_num_states_larger_than_tables() {
        // A crafted-but-envelope-valid blob whose num_states header claims more
        // states than the transition/epsilon tables actually contain must fail
        // closed at decode (ShapeMismatch), NOT be accepted and OOB-panic later
        // in try_reference_scan_into's `transition_table[base + dst_lane]`.
        let mut pipe = build(&["ab"], "input", "hits", 16);
        let honest_states = pipe.compiled.plan.num_states;
        pipe.compiled.plan.num_states = honest_states + 1; // header now overstates the tables
        let bytes = pipe.to_bytes().expect("serialize with tampered header");

        let err = ScanSession::from_bytes(&bytes)
            .expect_err("num_states overstating the tables must be rejected");
        assert!(
            matches!(
                err,
                PipelineWireError::ShapeMismatch {
                    reason: "transition_table length disagrees with num_states"
                }
            ),
            "must name the transition-table shape mismatch, got: {err}"
        );
    }

    #[test]
    fn from_bytes_rejects_out_of_range_accept_state_id() {
        // Law 10 at decode: a stale/crafted blob whose accept id points past the
        // decoded state space must fail closed here. The reference scan's
        // `lane < state.len()` guard would otherwise SILENTLY DROP that accept
        // an invisible recall hole on a stale cache, not a loud corruption error.
        let mut pipe = build(&["ab", "bc"], "input", "hits", 16);
        let honest = pipe.compiled.plan.num_states;
        assert!(
            !pipe.compiled.plan.accept_state_ids.is_empty(),
            "fixture must have accepts"
        );
        // `num_states` itself is the first out-of-range id (valid ids are 0..num_states).
        pipe.compiled.plan.accept_state_ids[0] = honest;
        let bytes = pipe
            .to_bytes()
            .expect("serialize with tampered accept id (to_bytes does not validate consistency)");

        let err = ScanSession::from_bytes(&bytes)
            .expect_err("an accept_state_id >= num_states must be rejected, not silently dropped");
        assert!(
            matches!(
                err,
                PipelineWireError::ShapeMismatch {
                    reason: "accept_state_id out of range for num_states"
                }
            ),
            "must name the accept-id range violation, got: {err}"
        );
    }

    #[test]
    fn from_bytes_accepts_in_range_accept_state_ids() {
        // Positive twin: an untampered pipeline whose accept ids are all in range
        // roundtrips cleanly, proving the new guard rejects only real corruption.
        let pipe = build(&["ab", "bc"], "input", "hits", 16);
        let states = pipe.compiled.plan.num_states as usize;
        assert!(
            pipe.compiled
                .plan
                .accept_state_ids
                .iter()
                .all(|&id| (id as usize) < states),
            "honest fixture accept ids must all be < num_states"
        );
        let bytes = pipe.to_bytes().expect("valid pipeline must serialize");
        let decoded = ScanSession::from_bytes(&bytes).expect("in-range accept ids must decode");
        assert_eq!(
            decoded.compiled.plan.accept_state_ids,
            pipe.compiled.plan.accept_state_ids
        );
    }

    #[test]
    fn rule_pipeline_reference_scan_into_matches_owned_scan_and_reuses_scratch() {
        let pipe = build(&["ab", "bc"], "input", "hits", 16);
        let owned = pipe.reference_scan(b"zabc");
        let mut scratch = Vec::with_capacity(16);
        let retained_capacity = scratch.capacity();

        pipe.try_reference_scan_into(b"zabc", &mut scratch)
            .expect("Fix: ScanSession CPU oracle should scan small haystacks");

        assert_eq!(scratch, owned);
        assert!(scratch.capacity() >= retained_capacity);
        assert_eq!(scratch, vec![Match::new(0, 1, 3), Match::new(1, 2, 4)]);
    }

    #[test]
    fn rule_pipeline_reference_scan_matches_fallible_variant() {
        // The infallible wrapper and inherent fallible scan must yield the same
        // concrete triples. Production artifact execution has a separate typed
        // session and does not share this explicit reference oracle.

        let pipe = build(&["ab", "bc"], "input", "hits", 16);
        let haystack = b"zabc";
        let expected = vec![Match::new(0, 1, 3), Match::new(1, 2, 4)];

        let infallible = pipe.reference_scan(haystack);
        let fallible = pipe
            .try_reference_scan(haystack)
            .expect("Fix: small-haystack reference scan must succeed");

        assert_eq!(infallible, expected, "infallible reference_scan content");
        assert_eq!(
            fallible, expected,
            "inherent try_reference_scan must match infallible result verbatim"
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rule_pipeline_reference_scan_fails_loud_not_empty_on_oversized_haystack() {
        // Law 10: a >u32 haystack must NOT be swallowed into an empty match set
        // (which reads as "clean"); it must surface loudly. The infallible
        // wrapper aborts; the fallible variant returns Err, neither returns an
        // empty Vec.
        //
        // The guard rejects on length BEFORE the scan loop reads a single byte,
        // so the 4 GiB+1 backing allocation stays lazily zero-paged (only page
        // tables are committed, not 4 GiB of RAM).
        let oversized = vec![0u8; (u32::MAX as usize) + 1];
        let pipe = build(&["ab"], "input", "hits", 16);

        // Fallible path: a real error, not a fabricated-empty success.
        let err = pipe
            .try_reference_scan(&oversized)
            .expect_err("Fix: an over-u32 haystack must error, never report zero matches");
        let msg = err.to_string();
        assert!(
            msg.contains("ScanSession::reference_scan") && msg.contains("u32"),
            "error must name the scan and the u32 ABI limit, got: {msg}"
        );

        // Infallible path: aborts loudly instead of swallowing to `[]`.
        let prior_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // keep the test log quiet
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pipe.reference_scan(&oversized)
        }));
        std::panic::set_hook(prior_hook);

        let panic_payload = outcome.expect_err(
            "Fix: reference_scan must panic on an over-u32 haystack, never return empty",
        );
        let panic_msg = panic_payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic_payload.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            panic_msg.contains("try_reference_scan"),
            "panic must point operators at the fallible variant, got: {panic_msg}"
        );
    }

    #[test]
    fn rule_pipeline_hit_buffer_allocation_is_checked_and_zeroed() {
        let bytes =
            super::zeroed_hit_buffer(2).expect("Fix: small ScanSession hit buffer should allocate");

        assert_eq!(bytes.len(), (2 * 3 + 1) * 4);
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn rule_pipeline_hit_buffer_into_reuses_and_zeroes_scratch() {
        let mut scratch = vec![0xAA; 128];
        let retained = scratch.capacity();

        super::zeroed_hit_buffer_into(3, &mut scratch)
            .expect("Fix: ScanSession hit buffer scratch should reserve");

        assert_eq!(scratch.len(), (3 * 3 + 1) * 4);
        assert!(scratch.iter().all(|&byte| byte == 0));
        assert!(scratch.capacity() >= retained);
    }

    /// Contract: the compiled program declares the canonical
    /// `nfa_haystack_len` 1-u32 storage buffer so the runtime cursor
    /// loop can read the actual haystack byte count without a
    /// recompile. The presence of this buffer is the wire-level
    /// guarantee that any haystack ≤ declared capacity can dispatch
    /// against the same program. Removing this buffer would silently
    /// re-introduce the "input expected N bytes but received M" hard
    /// error on every short-input dispatch - locking it as a contract.
    #[test]
    fn rule_pipeline_program_declares_haystack_len_buffer() {
        let pipe = build(&["ab"], "input", "hits", 1024);
        let names: Vec<&str> = pipe
            .compiled
            .program
            .buffers
            .iter()
            .map(|b| b.name())
            .collect();
        assert!(
            names.iter().any(|n| *n == super::nfa::HAYSTACK_LEN_BUF),
            "Fix: nfa_scan must declare `{}` so the cursor loop bound \
             is runtime-supplied; without it, ScanSession can only \
             dispatch at exactly its compile-time input_len.",
            super::nfa::HAYSTACK_LEN_BUF
        );
    }

    /// Contract: the compiled program declares the canonical
    /// `nfa_max_scan_bytes` 1-u32 storage buffer so the per-workgroup
    /// cursor cap is runtime-supplied. Without this buffer the cursor
    /// loop runs unbounded per workgroup, making the kernel O(N²) on
    /// large inputs - the discord-bot-token-on-62 MiB case that drove
    /// the bound into existence. Removing this buffer would silently
    /// reintroduce that perf cliff.
    #[test]
    fn rule_pipeline_program_declares_max_scan_bytes_buffer() {
        let pipe = build(&["ab"], "input", "hits", 1024);
        let names: Vec<&str> = pipe
            .compiled
            .program
            .buffers
            .iter()
            .map(|b| b.name())
            .collect();
        assert!(
            names.iter().any(|n| *n == super::nfa::MAX_SCAN_BYTES_BUF),
            "Fix: nfa_scan must declare `{}` so the per-workgroup \
             cursor cap is runtime-supplied; without it, ScanSession \
             dispatches at O(N²) per shard.",
            super::nfa::MAX_SCAN_BYTES_BUF
        );
    }
}
