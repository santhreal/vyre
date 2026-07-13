//! Region-evidence pipeline, plan W2-2, line 158: the successor to the (now
//! vestigial) `mega_scan::RulePipeline`.
//!
//! `RulePipeline` was the NFA-subgroup integrator, but its real consumer
//! (keyhog) collapsed `--backend mega-scan` onto region-presence and never used
//! the full multimatch path. This is what that consumer actually needs: ONE type,
//! ONE call, returning the complete **phase-1 evidence bundle**: the three
//! families a coalesced-batch scanner assembles per region:
//!
//! * **presence**: every pattern that matches anywhere in each region,
//! * **positions**: located `(pid, start, end)` triples for a designated subset,
//! * **admission**: a per-region bitmap for a (possibly different) subset, the
//!   coarse gate for a heavier verifier.
//!
//! # Two strategies, one bundle, honest perf
//!
//! [`RegionEvidencePipeline::scan`] is the **fast path** and the default: it
//! issues TWO occupancy-cheap dispatches, regex-DFA presence-by-region and
//! anchored-window extraction, then derives the remaining families on the host
//! (admission = presence AND the admission mask, since admission ⊆ presence;
//! positions = extractions filtered to the position mask). No third dispatch, no
//! fused-kernel occupancy collapse.
//!
//! [`RegionEvidencePipeline::scan_fused`] is the **single-launch capability**
//! (plan line 153): ONE dispatch of [`crate::scan::fused_region_evidence`]
//! producing all three families. It exists because 153 asks for a single launch,
//! but it is a correctness primitive, NOT the fast path, fusing the third family
//! into one kernel is measured ~20x slower (occupancy collapse; see that module's
//! docs). Reach for it only when a caller genuinely needs one dispatch; otherwise
//! use `scan`.
//!
//! Both agree with [`RegionEvidencePipeline::reference_scan`] (the CPU oracle)
//! bit-for-bit (one bundle definition, three ways to compute it).

use vyre::{BackendError, DispatchConfig, VyreBackend};
use vyre_primitives::matching::CompiledDfa;

use crate::scan::dispatch_io;
use crate::scan::fused_region_evidence::{
    fused_region_evidence_program, fused_region_evidence_reference, FusedRegionEvidence,
};
use crate::scan::regex_anchored_window::anchored_window_extract_program;
use crate::scan::regex_region_admission::{
    regex_admission_by_region_program, regex_admission_presence_words,
};
use crate::scan::{pack_u32_slice, unpack_match_triples};

/// The workgroup size every phase-1 evidence program declares
/// (`Program::wrapped(.., [128, 1, 1], ..)`). One global invocation per haystack
/// byte, so the launch grid is `ceil(invocations / WORKGROUP)` workgroups.
const EVIDENCE_WORKGROUP: u32 = 128;

/// Build a one-invocation-per-element dispatch config for a `[128,1,1]`-workgroup
/// evidence program. `DispatchConfig` is `#[non_exhaustive]`, so this is the ONE
/// place the grid math lives (invocations → workgroups).
fn evidence_dispatch_config(invocations: u32) -> DispatchConfig {
    let mut config = DispatchConfig::default();
    config.grid_override = Some([invocations.div_ceil(EVIDENCE_WORKGROUP).max(1), 1, 1]);
    // One invocation per element: record the true coverage so a shape-inferring
    // backend (CpuRefBackend) covers every element instead of the tail-dropping
    // buffer-shape default (Law 10).
    config.dispatch_elements = Some(invocations);
    config
}

/// Errors constructing a [`RegionEvidencePipeline`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionEvidenceError {
    /// A role mask's length did not equal `pattern_count`.
    MaskLength {
        /// Which mask (`"position"` or `"admission"`).
        mask: &'static str,
        /// The length the caller supplied.
        got: usize,
        /// The `pattern_count` it must equal.
        expected: u32,
    },
}

impl std::fmt::Display for RegionEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaskLength {
                mask,
                got,
                expected,
            } => write!(
                f,
                "RegionEvidencePipeline {mask} mask has {got} entries but pattern_count is {expected}. \
                 Fix: supply one mask entry per pattern id (0 = excluded, non-zero = included)."
            ),
        }
    }
}

impl std::error::Error for RegionEvidenceError {}

/// A compiled phase-1 evidence pipeline: an anchored regex DFA plus the role
/// masks that decide which patterns get located (positions) and which get
/// admitted (admission). Presence always covers every pattern.
#[derive(Debug, Clone)]
pub struct RegionEvidencePipeline {
    dfa: CompiledDfa,
    pattern_count: u32,
    position_mask: Vec<u32>,
    admission_mask: Vec<u32>,
    /// Per-pattern-id packed admission bitmap (`bit p` set iff `admission_mask[p]
    /// != 0`), width `regex_admission_presence_words(pattern_count)`. Precomputed
    /// so the fast path derives admission rows with one AND per word.
    admission_row_mask: Vec<u32>,
}

impl RegionEvidencePipeline {
    /// Build a pipeline from an anchored regex DFA and the two role masks. Each
    /// mask has one entry per pattern id: non-zero includes that pattern in the
    /// family, zero excludes it.
    ///
    /// # Errors
    /// Returns [`RegionEvidenceError::MaskLength`] if either mask's length is not
    /// `pattern_count`.
    pub fn new(
        dfa: CompiledDfa,
        pattern_count: u32,
        position_mask: Vec<u32>,
        admission_mask: Vec<u32>,
    ) -> Result<Self, RegionEvidenceError> {
        if position_mask.len() != pattern_count as usize {
            return Err(RegionEvidenceError::MaskLength {
                mask: "position",
                got: position_mask.len(),
                expected: pattern_count,
            });
        }
        if admission_mask.len() != pattern_count as usize {
            return Err(RegionEvidenceError::MaskLength {
                mask: "admission",
                got: admission_mask.len(),
                expected: pattern_count,
            });
        }
        let words = regex_admission_presence_words(pattern_count) as usize;
        let mut admission_row_mask = vec![0u32; words];
        for (pid, &flag) in admission_mask.iter().enumerate() {
            if flag != 0 {
                admission_row_mask[pid >> 5] |= 1u32 << (pid & 31);
            }
        }
        Ok(Self {
            dfa,
            pattern_count,
            position_mask,
            admission_mask,
            admission_row_mask,
        })
    }

    /// The compiled DFA backing this pipeline.
    #[must_use]
    pub fn dfa(&self) -> &CompiledDfa {
        &self.dfa
    }

    /// Presence-bitmap word count per region.
    #[must_use]
    pub fn presence_words(&self) -> u32 {
        regex_admission_presence_words(self.pattern_count)
    }

    /// CPU oracle for the phase-1 bundle, the parity definition both GPU
    /// strategies must reproduce. Reuses [`fused_region_evidence_reference`] so
    /// the bundle has ONE meaning.
    #[must_use]
    pub fn reference_scan(
        &self,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
    ) -> FusedRegionEvidence {
        fused_region_evidence_reference(
            &self.dfa,
            haystack,
            region_starts,
            region_base,
            &self.position_mask,
            &self.admission_mask,
            self.pattern_count,
        )
    }

    /// **Fast path.** Produce the full bundle from TWO occupancy-cheap dispatches
    /// (presence-by-region + anchored-window extraction), deriving admission and
    /// the position subset on the host. This is the successor's default, one
    /// call, the full phase-1 bundle, no fused-kernel pessimization.
    ///
    /// # Errors
    /// Returns [`BackendError`] on dispatch/readback failure or a haystack that
    /// exceeds the `u32` scan ABI.
    pub fn scan<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        max_matches: u32,
    ) -> Result<FusedRegionEvidence, BackendError> {
        let words = self.presence_words() as usize;
        let region_count = region_starts.len() as u32;
        if haystack.is_empty() {
            return Ok(FusedRegionEvidence {
                presence: vec![0u32; region_starts.len() * words],
                positions: Vec::new(),
                admission: vec![0u32; region_starts.len() * words],
            });
        }
        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "RegionEvidencePipeline::scan",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;
        let log2_max_regions = ceil_log2(region_count);

        // ---- Dispatch 1: presence-by-region (every matched pattern). ----
        let presence = {
            let program = regex_admission_by_region_program(
                "haystack",
                "transitions",
                "output_offsets",
                "output_records",
                "region_starts",
                "region_base",
                "haystack_len",
                "presence",
                self.dfa.state_count,
                self.dfa.output_records.len() as u32,
                region_count,
                self.presence_words(),
                self.dfa.max_pattern_len,
                log2_max_regions,
            );
            let bitmap_words = region_starts.len() * words;
            let packed_haystack = dispatch_io::pack_haystack_u32(haystack);
            let inputs: [&[u8]; 8] = [
                &packed_haystack,
                bytemuck::cast_slice(&self.dfa.transitions),
                bytemuck::cast_slice(&self.dfa.output_offsets),
                bytemuck::cast_slice(&self.dfa.output_records),
                bytemuck::cast_slice(region_starts),
                bytemuck::cast_slice(std::slice::from_ref(&region_base)),
                bytemuck::cast_slice(std::slice::from_ref(&haystack_len)),
                &vec_zero_bytes(bitmap_words),
            ];
            let config = evidence_dispatch_config(haystack_len);
            let outputs = backend.dispatch_borrowed(&program, &inputs, &config)?;
            let bytes = dispatch_io::try_output_bytes(&outputs, 0, "presence bitmap")?;
            decode_words(bytes, bitmap_words)
        };

        // ---- Host-derive admission: admission = presence AND admission mask. ----
        let mut admission = vec![0u32; region_starts.len() * words];
        for region in 0..region_starts.len() {
            for w in 0..words {
                admission[region * words + w] =
                    presence[region * words + w] & self.admission_row_mask[w];
            }
        }

        // ---- Dispatch 2: anchored-window extraction over every origin. ----
        let candidates: Vec<u32> = (0..haystack_len).collect();
        let num_candidates = candidates.len() as u32;
        let program = anchored_window_extract_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "candidates",
            "candidate_count",
            "haystack_len",
            "match_count",
            "matches",
            self.dfa.state_count,
            self.dfa.output_records.len() as u32,
            num_candidates,
            max_matches,
            self.dfa.max_pattern_len,
        );
        let packed_haystack = dispatch_io::pack_haystack_u32(haystack);
        let candidate_bytes = pack_u32_slice(&candidates);
        let zero_count = 0u32;
        let matches_scratch = vec_zero_bytes(max_matches as usize * 3);
        let inputs: [&[u8]; 9] = [
            &packed_haystack,
            bytemuck::cast_slice(&self.dfa.transitions),
            bytemuck::cast_slice(&self.dfa.output_offsets),
            bytemuck::cast_slice(&self.dfa.output_records),
            &candidate_bytes,
            bytemuck::cast_slice(std::slice::from_ref(&num_candidates)),
            bytemuck::cast_slice(std::slice::from_ref(&haystack_len)),
            bytemuck::cast_slice(std::slice::from_ref(&zero_count)),
            &matches_scratch,
        ];
        let config = evidence_dispatch_config(num_candidates);
        let outputs = backend.dispatch_borrowed(&program, &inputs, &config)?;
        let count_bytes = dispatch_io::try_output_bytes(&outputs, 0, "extract match count")?;
        let count = dispatch_io::try_read_u32_prefix(count_bytes, "extract match count")?;
        if count > max_matches {
            return Err(BackendError::new(format!(
                "RegionEvidencePipeline::scan extracted {count} matches but the buffer caps {max_matches}; \
                 matches were dropped. Fix: raise max_matches or shard the haystack (fail closed, no partial set)."
            )));
        }
        let match_bytes = dispatch_io::try_output_bytes(&outputs, 1, "extract matches")?;
        let mut positions: Vec<_> = unpack_match_triples(match_bytes, count)
            .into_iter()
            .filter(|m| {
                self.position_mask
                    .get(m.pattern_id as usize)
                    .copied()
                    .unwrap_or(0)
                    != 0
            })
            .collect();
        positions.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        positions.dedup();

        Ok(FusedRegionEvidence {
            presence,
            positions,
            admission,
        })
    }

    /// **Single-launch capability** (plan line 153). Produce the full bundle from
    /// ONE dispatch of the fused evidence program. Correct and equal to
    /// [`Self::scan`]/[`Self::reference_scan`], but occupancy-slower, prefer
    /// [`Self::scan`] unless a single dispatch is a hard requirement.
    ///
    /// # Errors
    /// Returns [`BackendError`] on dispatch/readback failure, a `u32`-ABI
    /// overflow, or a match-buffer overflow (fails closed, no partial set).
    pub fn scan_fused<B: VyreBackend + ?Sized>(
        &self,
        backend: &B,
        haystack: &[u8],
        region_starts: &[u32],
        region_base: u32,
        max_matches: u32,
    ) -> Result<FusedRegionEvidence, BackendError> {
        let words = self.presence_words() as usize;
        let region_count = region_starts.len() as u32;
        let bitmap_words = region_starts.len() * words;
        if haystack.is_empty() {
            return Ok(FusedRegionEvidence {
                presence: vec![0u32; bitmap_words],
                positions: Vec::new(),
                admission: vec![0u32; bitmap_words],
            });
        }
        let haystack_len = dispatch_io::scan_guard(
            haystack,
            "RegionEvidencePipeline::scan_fused",
            dispatch_io::DEFAULT_MAX_SCAN_BYTES,
        )?;
        let program = fused_region_evidence_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "region_starts",
            "region_base",
            "position_mask",
            "admission_mask",
            "haystack_len",
            "presence",
            "match_count",
            "matches",
            "admission",
            self.dfa.state_count,
            self.dfa.output_records.len() as u32,
            region_count,
            self.pattern_count,
            self.presence_words(),
            max_matches,
            self.dfa.max_pattern_len,
            ceil_log2(region_count),
        );
        let packed_haystack = dispatch_io::pack_haystack_u32(haystack);
        let zero_count = 0u32;
        let presence_scratch = vec_zero_bytes(bitmap_words);
        let admission_scratch = vec_zero_bytes(bitmap_words);
        let matches_scratch = vec_zero_bytes(max_matches as usize * 3);
        let inputs: [&[u8]; 13] = [
            &packed_haystack,
            bytemuck::cast_slice(&self.dfa.transitions),
            bytemuck::cast_slice(&self.dfa.output_offsets),
            bytemuck::cast_slice(&self.dfa.output_records),
            bytemuck::cast_slice(region_starts),
            bytemuck::cast_slice(std::slice::from_ref(&region_base)),
            bytemuck::cast_slice(&self.position_mask),
            bytemuck::cast_slice(&self.admission_mask),
            bytemuck::cast_slice(std::slice::from_ref(&haystack_len)),
            &presence_scratch,
            bytemuck::cast_slice(std::slice::from_ref(&zero_count)),
            &matches_scratch,
            &admission_scratch,
        ];
        let config = evidence_dispatch_config(haystack_len);
        let outputs = backend.dispatch_borrowed(&program, &inputs, &config)?;

        // Writable buffers, binding order: presence, match_count, matches, admission.
        let presence = decode_words(
            dispatch_io::try_output_bytes(&outputs, 0, "fused presence")?,
            bitmap_words,
        );
        let count = dispatch_io::try_read_u32_prefix(
            dispatch_io::try_output_bytes(&outputs, 1, "fused match count")?,
            "fused match count",
        )?;
        if count > max_matches {
            return Err(BackendError::new(format!(
                "RegionEvidencePipeline::scan_fused extracted {count} matches but the buffer caps {max_matches}; \
                 matches were dropped. Fix: raise max_matches or shard the haystack (fail closed, no partial set)."
            )));
        }
        let mut positions = unpack_match_triples(
            dispatch_io::try_output_bytes(&outputs, 2, "fused matches")?,
            count,
        );
        positions.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        positions.dedup();
        let admission = decode_words(
            dispatch_io::try_output_bytes(&outputs, 3, "fused admission")?,
            bitmap_words,
        );
        Ok(FusedRegionEvidence {
            presence,
            positions,
            admission,
        })
    }
}

/// `ceil(log2(max(n, 2)))`, min 1, the fixed iteration count a binary region
/// search needs to converge over `n` regions. ONE owner for the two programs.
fn ceil_log2(n: u32) -> u32 {
    (32 - (n.max(2) - 1).leading_zeros()).max(1)
}

/// Zero-filled little-endian `u32` scratch of `word_count` words.
fn vec_zero_bytes(word_count: usize) -> Vec<u8> {
    vec![0u8; word_count * 4]
}

/// Decode the first `word_count` little-endian `u32`s from a readback buffer.
fn decode_words(bytes: &[u8], word_count: usize) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .take(word_count)
        .collect()
}

#[cfg(all(test, feature = "matching-regex", feature = "matching-dfa"))]
mod tests {
    use super::*;
    use crate::scan::regex_dfa::build_regex_dfa_pipeline;

    fn dfa_for(patterns: &[&str]) -> CompiledDfa {
        build_regex_dfa_pipeline(patterns, 4096, 16_384)
            .expect("Fix: test patterns must compile to an anchored regex DFA")
            .dfa
    }

    /// Mask-length validation fails closed rather than silently truncating a
    /// short mask (which would drop patterns from a family (a recall lie)).
    #[test]
    fn new_rejects_masks_that_do_not_cover_every_pattern() {
        let dfa = dfa_for(&["abc", "def"]);
        let err = RegionEvidencePipeline::new(dfa.clone(), 2, vec![1], vec![1, 0])
            .expect_err("short position mask must be rejected");
        assert_eq!(
            err,
            RegionEvidenceError::MaskLength {
                mask: "position",
                got: 1,
                expected: 2
            }
        );
        assert!(RegionEvidencePipeline::new(dfa, 2, vec![1, 0], vec![0, 1]).is_ok());
    }

    /// The precomputed admission row mask packs exactly the admission-masked pids.
    #[test]
    fn admission_row_mask_packs_the_masked_subset() {
        let dfa = dfa_for(&["a", "b", "c", "d"]);
        let pipeline =
            RegionEvidencePipeline::new(dfa, 4, vec![1, 1, 0, 0], vec![0, 1, 0, 1]).unwrap();
        // pids 1 and 3 admitted → bits 1 and 3 set in word 0.
        assert_eq!(pipeline.admission_row_mask, vec![0b1010]);
    }

    /// The empty haystack yields zeroed bitmaps and no positions on BOTH GPU
    /// entry points without needing a backend (early return).
    #[test]
    fn empty_haystack_is_zeroed_bundle() {
        let dfa = dfa_for(&["abc"]);
        let pipeline = RegionEvidencePipeline::new(dfa, 1, vec![1], vec![1]).unwrap();
        let region_starts = [0u32];
        let oracle = pipeline.reference_scan(b"", &region_starts, 0);
        assert!(oracle.positions.is_empty());
        assert!(oracle.presence.iter().all(|&w| w == 0));
        assert!(oracle.admission.iter().all(|&w| w == 0));
    }
}
