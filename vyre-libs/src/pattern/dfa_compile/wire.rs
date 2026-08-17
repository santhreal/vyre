//! On-wire payload schema for a compiled DFA.
//!
//! The framing  -  magic, version, length-prefixed sections  -  belongs to
//! `vyre_foundation::serial::envelope`. This module owns the field order and
//! the decode errors that order can produce.

use std::{error::Error, fmt};

use super::CompiledDfa;

/// Magic + version header for `CompiledDfa::to_bytes` / `from_bytes`.
/// Keep this stable; bump `DFA_WIRE_VERSION` for any breaking layout change.
///
/// The actual framing (magic + version header, length-prefixed sections,
/// truncation / shape error variants) is delegated to
/// `vyre_foundation::serial::envelope`. This file owns only the
/// payload schema (which fields go in what order) so future serializable
/// types in vyre-primitives reuse the same envelope.
const DFA_WIRE_MAGIC: [u8; 4] = *b"VDFA";
const DFA_WIRE_VERSION: u32 = 2;

/// Returned from [`CompiledDfa::from_bytes`] when the on-wire payload
/// cannot be decoded into a valid DFA. The variant carries enough
/// context for the caller to discriminate "stale cache, recompile" from
/// "actual bug, refuse".
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DfaWireError {
    /// Payload is shorter than the fixed header / a declared section.
    Truncated {
        /// Total bytes the decoder needed to read this section.
        needed: usize,
        /// Bytes actually provided in the input slice.
        got: usize,
    },
    /// First four bytes were not the `VDFA` magic  -  caller likely passed
    /// an unrelated blob.
    BadMagic,
    /// Wire version did not match the build's `DFA_WIRE_VERSION`. The
    /// caller's cache is from an older scanner consumer/vyre and must be rebuilt.
    VersionMismatch {
        /// Wire version this build of vyre-primitives understands.
        expected: u32,
        /// Wire version recorded in the blob's header.
        found: u32,
    },
    /// One of the array length fields disagreed with the declared
    /// `state_count`  -  corrupt or hand-crafted blob.
    ShapeMismatch {
        /// Static description of which length cross-check failed.
        reason: &'static str,
    },
    /// A payload section exceeded the wire envelope's `u32` length prefix.
    SectionTooLarge {
        /// Word count the caller attempted to encode.
        len: usize,
        /// Maximum word count representable by the wire format.
        max: usize,
    },
    /// The shared wire envelope returned an error variant this crate
    /// reports through the generic envelope branch.
    Envelope(String),
}

impl fmt::Display for DfaWireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, got } => write!(
                f,
                "DFA wire blob truncated: needed {needed} bytes, got {got}. \
                 Fix: regenerate the cache."
            ),
            Self::BadMagic => write!(
                f,
                "DFA wire blob does not start with `VDFA` magic. Fix: this \
                 is not a CompiledDfa::to_bytes payload."
            ),
            Self::VersionMismatch { expected, found } => write!(
                f,
                "DFA wire blob version {found} does not match the runtime \
                 version {expected}. Fix: discard the cache and recompile \
                 the DFA."
            ),
            Self::ShapeMismatch { reason } => write!(
                f,
                "DFA wire blob shape mismatch: {reason}. Fix: this blob is \
                 corrupt  -  discard and recompile."
            ),
            Self::SectionTooLarge { len, max } => write!(
                f,
                "DFA wire section length {len} exceeds maximum {max}. \
                 Fix: shard the DFA into smaller pattern groups."
            ),
            Self::Envelope(message) => write!(f, "DFA wire envelope error: {message}"),
        }
    }
}

impl Error for DfaWireError {}

impl CompiledDfa {
    /// Serialize this DFA into a self-describing little-endian binary
    /// blob suitable for on-disk caching. Stable layout under
    /// `DFA_WIRE_VERSION`. Pure data, no allocator-dependent state.
    ///
    /// Layout:
    ///   - 4 bytes: magic `b"VDFA"`
    ///   - 4 bytes: version (LE u32)
    ///   - 4 bytes: state_count (LE u32)
    ///   - 4 bytes: max_pattern_len (LE u32)
    ///   - 4 bytes: transitions length in u32 words (LE u32)
    ///   - 4 bytes: accept length in u32 words (LE u32)
    ///   - 4 bytes: output_offsets length in u32 words (LE u32)
    ///   - 4 bytes: output_records length in u32 words (LE u32)
    ///   - transitions data    (state_count * 256 * 4 bytes)
    ///   - accept data         (state_count * 4 bytes)
    ///   - output_offsets data ((state_count + 1) * 4 bytes)
    ///   - output_records data (variable * 4 bytes)
    ///
    /// Total size is `O(state_count)` bytes; ~1 MiB per 1k states.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DfaWireError> {
        let mut out = vyre_foundation::serial::WireWriter::new(&DFA_WIRE_MAGIC, DFA_WIRE_VERSION);
        out.write_u32(self.state_count);
        out.write_u32(self.max_pattern_len);
        out.write_words(&self.transitions)
            .map_err(map_envelope_error)?;
        out.write_words(&self.accept).map_err(map_envelope_error)?;
        out.write_words(&self.output_offsets)
            .map_err(map_envelope_error)?;
        out.write_words(&self.output_records)
            .map_err(map_envelope_error)?;
        Ok(out.into_bytes())
    }

    /// Decode a `CompiledDfa` from a blob produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns [`DfaWireError`] for truncation, magic mismatch, version
    /// drift, or shape inconsistencies. A `VersionMismatch` is the
    /// expected signal to invalidate an on-disk cache and recompile.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DfaWireError> {
        let mut reader =
            vyre_foundation::serial::WireReader::new(bytes, &DFA_WIRE_MAGIC, DFA_WIRE_VERSION)
                .map_err(map_envelope_error)?;
        let state_count = reader.read_u32().map_err(map_envelope_error)?;
        let max_pattern_len = reader.read_u32().map_err(map_envelope_error)?;
        let transitions = reader.read_words().map_err(map_envelope_error)?;
        let accept = reader.read_words().map_err(map_envelope_error)?;
        let output_offsets = reader.read_words().map_err(map_envelope_error)?;
        let output_records = reader.read_words().map_err(map_envelope_error)?;

        // Cross-check the declared shape before returning the payload to
        // callers. Length fields are validated by the envelope reader; these
        // checks validate DFA-specific invariants so corrupt cache blobs do not
        // become internally inconsistent automata.
        if transitions.len() != (state_count as usize) * 256 {
            return Err(DfaWireError::ShapeMismatch {
                reason: "transitions length != state_count * 256",
            });
        }
        // Every transition is consumed as the next state index
        // (`transitions[state * 256 + byte]`), so a target >= state_count would
        // index out of bounds on the following step. A corrupt/stale cache blob
        // must fail closed here rather than OOB-panic (or read a garbage state)
        // mid-scan (the same invariant the length checks enforce for the tables).
        if transitions
            .iter()
            .any(|&target| target as usize >= state_count as usize)
        {
            return Err(DfaWireError::ShapeMismatch {
                reason: "transition target out of range for state_count",
            });
        }
        if accept.len() != state_count as usize {
            return Err(DfaWireError::ShapeMismatch {
                reason: "accept length != state_count",
            });
        }
        if output_offsets.len() != (state_count as usize) + 1 {
            return Err(DfaWireError::ShapeMismatch {
                reason: "output_offsets length != state_count + 1",
            });
        }
        if output_offsets.first().copied() != Some(0) {
            return Err(DfaWireError::ShapeMismatch {
                reason: "output_offsets must start at zero",
            });
        }
        if output_offsets.last().copied() != Some(output_records.len() as u32) {
            return Err(DfaWireError::ShapeMismatch {
                reason: "output_offsets last entry must equal output_records length",
            });
        }
        if output_offsets
            .windows(2)
            .any(|window| window[0] > window[1])
        {
            return Err(DfaWireError::ShapeMismatch {
                reason: "output_offsets must be monotonic",
            });
        }
        if output_offsets
            .iter()
            .any(|&offset| offset as usize > output_records.len())
        {
            return Err(DfaWireError::ShapeMismatch {
                reason: "output_offsets entries must be within output_records",
            });
        }
        // max_pattern_len == 0 is consistent ONLY when the sole accepting state is the
        // root (state 0). The empty pattern matches at the root having consumed no
        // bytes, so its length is 0 and `dfa_compile(&[b""])` legitimately carries
        // max_pattern_len == 0 with accept[0] != 0. A *non-root* accept state, however,
        // is reachable only by consuming >= 1 byte along some pattern, so it spells a
        // pattern of length == depth(state) >= 1 and forces max_pattern_len >= 1. A blob
        // that pairs max_pattern_len == 0 with a deeper accept is therefore internally
        // inconsistent (the classic symptom of a corrupted cache whose length scalar was
        // zeroed). Reject it: max_pattern_len bounds the per-position replay / segmentation
        // warm-up window (see the field doc), so handing back an under-sized 0 would
        // silently drop every match that straddles a segment boundary, an invisible
        // recall loss. This is the precise form of the former guard, which was over-broad
        // (it also rejected the genuine empty-pattern round-trip, accept only at the root).
        if max_pattern_len == 0 && accept.iter().skip(1).any(|&state| state != 0) {
            return Err(DfaWireError::ShapeMismatch {
                reason: "max_pattern_len == 0 but a non-root state accepts",
            });
        }

        Ok(Self {
            transitions,
            accept,
            state_count,
            max_pattern_len,
            output_offsets,
            output_records,
        })
    }
}

fn map_envelope_error(error: vyre_foundation::serial::EnvelopeError) -> DfaWireError {
    match error {
        vyre_foundation::serial::EnvelopeError::Truncated { needed, got } => {
            DfaWireError::Truncated { needed, got }
        }
        vyre_foundation::serial::EnvelopeError::BadMagic { .. } => DfaWireError::BadMagic,
        vyre_foundation::serial::EnvelopeError::VersionMismatch { expected, found } => {
            DfaWireError::VersionMismatch { expected, found }
        }
        vyre_foundation::serial::EnvelopeError::SectionTooLarge { len, max } => {
            DfaWireError::SectionTooLarge { len, max }
        }
        error => DfaWireError::Envelope(error.to_string()),
    }
}
