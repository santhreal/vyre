//! Character-class lowering: the single-byte ASCII fast path and the UTF-8
//! byte-sequence expansion path.

use regex_syntax::hir::Class;

use super::compile_error::FEATURE_UNICODE_CLASS_CAP;
use super::nfa_builder::{ByteSet, Fragment, NfaBuilder};
use super::RegexCompileError;

/// Lower a regex character class into an NFA fragment, taking the
/// single-byte fast path when the class fits in 0..=127 and the
/// UTF-8-alternation expansion path otherwise.
///
/// The single-byte path is identical to the original implementation:
/// one ByteSet, one transition, `match_len = 1`. The expansion path
/// emits one byte-chain fragment per codepoint (or per pre-existing
/// multi-byte range like `\u{0100}-\u{01FF}` enumerated codepoint-by-
/// codepoint) and ε-merges them via a shared end state.
///
/// `match_len` for the expansion case is the MAX byte length across
/// arms - anchored extraction uses `match_len` only to position
/// the post-process window, not to extract the credential text, and
/// over-sizing the window is harmless (the real regex re-extracts the
/// exact match inside it).
///
/// To keep state-budget worst case bounded, expansion is capped at
/// `MAX_CLASS_EXPANSION_CODEPOINTS = 256` enumerated codepoints (a
/// `[\u{0100}-\u{017F}]` Latin-Extended block sits at 128, which is
/// well within budget; a class spanning a full CJK block would refuse).
pub(super) fn build_class(
    b: &mut NfaBuilder,
    cls: &Class,
    pid: usize,
) -> Result<Fragment, RegexCompileError> {
    if let Some(set) = try_class_as_ascii_byte_set(cls) {
        let start = b.fresh_state()?;
        let end = b.fresh_state()?;
        b.add_byte_transition(start, set, end);
        return Ok(Fragment {
            start,
            end,
            match_len: 1,
        });
    }
    let sequences = class_to_utf8_sequences(cls, pid)?;
    if sequences.is_empty() {
        return Err(RegexCompileError::Unsupported {
            pattern_index: pid,
            feature: "empty character class after Unicode expansion",
        });
    }
    let start = b.fresh_state()?;
    let end = b.fresh_state()?;
    let mut max_len = 1usize;
    for seq in &sequences {
        if seq.is_empty() {
            continue;
        }
        // Build a sequential chain start ε→ s0 -b0-> s1 -b1-> ... -bN-> end
        // for this UTF-8 byte sequence.
        let arm_start = b.fresh_state()?;
        b.add_epsilon(start, arm_start);
        let mut prev = arm_start;
        for &byte in seq {
            let next = b.fresh_state()?;
            b.add_byte_transition(prev, ByteSet::from_byte(byte), next);
            prev = next;
        }
        b.add_epsilon(prev, end);
        if seq.len() > max_len {
            max_len = seq.len();
        }
    }
    Ok(Fragment {
        start,
        end,
        match_len: max_len,
    })
}

/// Returns `Some(ByteSet)` when every member of the class fits in
/// 0..=127 (i.e. the original single-byte ASCII fast path). Otherwise
/// returns None so the caller takes the UTF-8 expansion path.
pub(super) fn try_class_as_ascii_byte_set(cls: &Class) -> Option<ByteSet> {
    let mut out = ByteSet::new();
    match cls {
        Class::Bytes(byte_class) => {
            // Byte classes are already at the byte level - every member
            // is a u8, no codepoint expansion involved. The legacy fast
            // path always applies.
            for r in byte_class.iter() {
                let merged = ByteSet::from_range(r.start(), r.end());
                for w in 0..4 {
                    out.bits[w] |= merged.bits[w];
                }
            }
            Some(out)
        }
        Class::Unicode(uni) => {
            // ASCII-only fast path. The moment any range escapes
            // 0..=0x7F, fall through to UTF-8 expansion.
            for r in uni.iter() {
                if (r.end() as u32) > 0x7F {
                    return None;
                }
                let merged = ByteSet::from_range(r.start() as u8, r.end() as u8);
                for w in 0..4 {
                    out.bits[w] |= merged.bits[w];
                }
            }
            Some(out)
        }
    }
}

/// Cap on enumerated codepoints during UTF-8 expansion. A class like
/// `[\u{0100}-\u{017F}]` (Latin Extended-A) expands to 128 sequences,
/// well within the cap. A class spanning a full CJK block (~20 000
/// codepoints) would blow past it - the byte-state automaton can't
/// represent that cleanly, so the consumer should keep that pattern on
/// the CPU regex path.
const MAX_CLASS_EXPANSION_CODEPOINTS: usize = 256;

/// Enumerate every codepoint in `cls`, encode each into UTF-8, and
/// return the resulting `Vec<Vec<u8>>` so the caller can build an
/// alternation of byte-chain fragments. ASCII members come back as
/// 1-byte sequences; non-ASCII as 2-4 byte sequences.
pub(super) fn class_to_utf8_sequences(
    cls: &Class,
    pid: usize,
) -> Result<Vec<Vec<u8>>, RegexCompileError> {
    let mut sequences: Vec<Vec<u8>> = Vec::new();
    let mut budget = MAX_CLASS_EXPANSION_CODEPOINTS;
    match cls {
        Class::Bytes(byte_class) => {
            for r in byte_class.iter() {
                for byte in r.start()..=r.end() {
                    if budget == 0 {
                        return Err(RegexCompileError::Unsupported {
                            pattern_index: pid,
                            feature: "byte character class exceeded expansion cap",
                        });
                    }
                    sequences.push(vec![byte]);
                    budget -= 1;
                }
            }
        }
        Class::Unicode(uni) => {
            for r in uni.iter() {
                let lo = r.start() as u32;
                let hi = r.end() as u32;
                for cp in lo..=hi {
                    if budget == 0 {
                        return Err(RegexCompileError::Unsupported {
                            pattern_index: pid,
                            feature: FEATURE_UNICODE_CLASS_CAP,
                        });
                    }
                    // Use a small buffer + `char::encode_utf8` to avoid
                    // pulling in a heavyweight UTF-8 dependency. Invalid
                    // codepoints (surrogates) are silently skipped -
                    // regex-syntax shouldn't emit them in a parsed HIR
                    // for character classes, but the `char::from_u32`
                    // guard catches the corner case if it ever does.
                    if let Some(c) = char::from_u32(cp) {
                        let mut buf = [0u8; 4];
                        let encoded = c.encode_utf8(&mut buf);
                        sequences.push(encoded.as_bytes().to_vec());
                        budget -= 1;
                    }
                }
            }
        }
    }
    Ok(sequences)
}

#[cfg(test)]
mod tests {
    use super::super::{compile_regex_set, RegexCompileError};

    /// Contract: non-ASCII codepoints inside a character class no longer
    /// abort compile. They expand into a UTF-8 byte-sequence alternation
    /// the byte-NFA can represent. Mirrors the homoglyph-expanded
    /// detector patterns consumers feed in (e.g. openai `[hнһｈ]f_...`)
    /// that used to fall on the floor with "unicode character classes
    /// outside ASCII".
    #[test]
    fn unicode_class_outside_ascii_compiles_via_utf8_expansion() {
        // `н` (U+043D) and `һ` (U+04BB) are 2-byte UTF-8; `ｈ` (U+FF48)
        // is 3-byte UTF-8; `h` (U+0068) is 1-byte. All four must be
        // representable.
        let pat = "[hнһｈ]f_[a-zA-Z0-9]{4}";
        let result = compile_regex_set(&[pat]);
        let compiled = match result {
            Ok(c) => c,
            Err(e) => {
                panic!("unicode-extended character class must compile via UTF-8 expansion; got {e}")
            }
        };
        // 4 alternation arms (one per codepoint) × varying byte length
        // + chain states + literal `f_` chain + bounded repetition
        // states - the exact count is implementation-dependent, but
        // every successfully-compiled regex must produce >=2 accept-
        // state-ids worth of state graph.
        assert!(
            compiled.plan.num_states > 4,
            "expanded NFA must have non-trivial state count"
        );
        // accept_state_ids carries one entry per accept (one pattern,
        // so one accept) regardless of arm count; the load-bearing
        // assertion is that compile didn't error.
        assert_eq!(compiled.plan.accept_states.len(), 1);
    }

    /// Contract: classes containing ONLY ASCII still take the fast
    /// single-byte-transition path. Without this guarantee, every AC
    /// detector regex would pay the multi-state expansion cost.
    #[test]
    fn ascii_only_class_keeps_single_byte_transition_path() {
        // Single state for entry + 2 for `[ab]` (start + end) = 3.
        // Anything larger means we accidentally took the expansion arm.
        let r = compile_regex_set(&["[ab]"]).unwrap();
        assert_eq!(
            r.plan.num_states, 3,
            "[ab] must stay on the single-transition fast path (entry + 2 class states); got {} states",
            r.plan.num_states
        );
    }

    /// Contract: massive Unicode ranges that would blow past the
    /// expansion cap return a structured error instead of consuming
    /// unbounded memory.
    #[test]
    fn unicode_class_above_expansion_cap_errors_cleanly() {
        // 257 codepoints - one above MAX_CLASS_EXPANSION_CODEPOINTS = 256.
        let pat = "[\u{0100}-\u{0200}]";
        let err = compile_regex_set(&[pat]).unwrap_err();
        match err {
            RegexCompileError::Unsupported { feature, .. } => {
                assert!(
                    feature.contains("expansion cap"),
                    "over-cap expansion must name the cap in its diagnostic: {feature}"
                );
            }
            other => panic!("expected Unsupported expansion-cap error, got {other:?}"),
        }
    }
}
