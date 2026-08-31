//! The 4 MiB irregular haystack and the literal set planted into it.
//!
//! The workload's whole point is unaligned, varied-length literals in noise, so
//! the fixture is deliberate: every planted offset is non-multiple-of-32 and
//! each pattern uses its own stride and phase.

use crate::api::case::BenchError;
use crate::cases::mix32;

use super::PATTERNS;

pub(super) fn pattern_lengths() -> Result<Vec<u32>, BenchError> {
    PATTERNS
        .iter()
        .map(|pattern| {
            u32::try_from(pattern.len()).map_err(|_| {
                BenchError::EnvironmentInvalid(
                    "irregular AC pattern length exceeded u32. Fix: split oversized literals."
                        .to_string(),
                )
            })
        })
        .collect()
}

pub(super) fn build_irregular_haystack(len: usize) -> (Vec<u8>, u32) {
    let mut haystack = vec![0_u8; len];
    for (index, byte) in haystack.iter_mut().enumerate() {
        let mixed = mix32(index as u32);
        *byte = 33 + (mixed % 90) as u8;
    }

    let mut planted = 0_u32;
    for (pattern_index, pattern) in PATTERNS.iter().enumerate() {
        let stride = 8_191 + pattern_index * 271;
        let phase = 17 + pattern_index * 113;
        let mut offset = phase;
        while offset + pattern.len() <= haystack.len() {
            if (offset & 31) != 0 {
                haystack[offset..offset + pattern.len()].copy_from_slice(pattern);
                planted += 1;
            }
            offset += stride;
        }
    }
    (haystack, planted)
}
