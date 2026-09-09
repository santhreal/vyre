//! Worst-case match-length analysis used to size the replay window.

use regex_syntax::hir::{Hir, HirKind};

use super::char_class::{class_to_utf8_sequences, try_class_as_ascii_byte_set};
use super::RegexCompileError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MatchExtent {
    pub(super) min: usize,
    pub(super) max: Option<usize>,
}

fn extent_overflow(pattern_index: usize) -> RegexCompileError {
    RegexCompileError::MatchLengthArithmeticOverflow { pattern_index }
}

fn checked_extent_add(
    left: usize,
    right: usize,
    pattern_index: usize,
) -> Result<usize, RegexCompileError> {
    left.checked_add(right)
        .ok_or_else(|| extent_overflow(pattern_index))
}

fn checked_extent_mul(
    left: usize,
    right: usize,
    pattern_index: usize,
) -> Result<usize, RegexCompileError> {
    left.checked_mul(right)
        .ok_or_else(|| extent_overflow(pattern_index))
}

pub(super) fn analyze_match_extent(
    hir: &Hir,
    pattern_index: usize,
) -> Result<MatchExtent, RegexCompileError> {
    match hir.kind() {
        HirKind::Empty | HirKind::Look(_) => Ok(MatchExtent {
            min: 0,
            max: Some(0),
        }),
        HirKind::Literal(literal) => Ok(MatchExtent {
            min: literal.0.len(),
            max: Some(literal.0.len()),
        }),
        HirKind::Class(class) => {
            if try_class_as_ascii_byte_set(class).is_some() {
                return Ok(MatchExtent {
                    min: 1,
                    max: Some(1),
                });
            }
            let sequences = class_to_utf8_sequences(class, pattern_index)?;
            let min = sequences.iter().map(Vec::len).min().unwrap_or(0);
            let max = sequences.iter().map(Vec::len).max().unwrap_or(0);
            Ok(MatchExtent {
                min,
                max: Some(max),
            })
        }
        HirKind::Capture(capture) => analyze_match_extent(&capture.sub, pattern_index),
        HirKind::Concat(parts) => {
            let mut extent = MatchExtent {
                min: 0,
                max: Some(0),
            };
            for part in parts {
                let next = analyze_match_extent(part, pattern_index)?;
                extent.min = checked_extent_add(extent.min, next.min, pattern_index)?;
                extent.max = match (extent.max, next.max) {
                    (Some(left), Some(right)) => {
                        Some(checked_extent_add(left, right, pattern_index)?)
                    }
                    _ => None,
                };
            }
            Ok(extent)
        }
        HirKind::Alternation(alternatives) => {
            let mut min = usize::MAX;
            let mut max = Some(0usize);
            for alternative in alternatives {
                let extent = analyze_match_extent(alternative, pattern_index)?;
                min = min.min(extent.min);
                max = match (max, extent.max) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    _ => None,
                };
            }
            Ok(MatchExtent {
                min: if alternatives.is_empty() { 0 } else { min },
                max,
            })
        }
        HirKind::Repetition(repetition) => {
            let sub = analyze_match_extent(&repetition.sub, pattern_index)?;
            let min = checked_extent_mul(sub.min, repetition.min as usize, pattern_index)?;
            let max = match repetition.max {
                Some(0) => Some(0),
                Some(count) => match sub.max {
                    Some(sub_max) => {
                        Some(checked_extent_mul(sub_max, count as usize, pattern_index)?)
                    }
                    None => None,
                },
                None if sub.max == Some(0) => Some(0),
                None => None,
            };
            Ok(MatchExtent { min, max })
        }
    }
}
