//! The 4 MiB irregular haystack and the literal set planted into it.
//!
//! The workload's whole point is unaligned, varied-length literals in noise, so
//! the fixture is deliberate: every planted offset is non-multiple-of-32 and
//! each pattern uses its own stride and phase.

use crate::api::case::BenchError;
use crate::cases::mix32;
use vyre_foundation::ir::Program;

use super::PATTERNS;

/// Bind the compiled scan program's haystack declaration to the packed word
/// count this fixture uploads.
///
/// The scan builders declare the haystack with no count, because the same
/// program scans any length. A declaration without a count leaves the logical
/// extent of the graph value unresolved, which the megakernel refuses, and
/// leaves the launch span derived from the widest remaining declaration, which
/// is the bounded match output and covers a fraction of the input. Stating the
/// count the caller actually binds resolves both. The byte length is unchanged:
/// four bytes per word is the layout `pack_haystack_u32` writes.
pub(super) fn with_haystack_extent(
    program: Program,
    haystack_bytes: usize,
) -> Result<Program, BenchError> {
    let words = u32::try_from(haystack_bytes.div_ceil(4)).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "irregular AC haystack of {haystack_bytes} bytes exceeds a u32 packed word count. Fix: shard the haystack."
        ))
    })?;
    let mut found = false;
    let buffers = program
        .buffers()
        .iter()
        .cloned()
        .map(|buffer| {
            if buffer.name() == "haystack" {
                found = true;
                buffer.with_count(words)
            } else {
                buffer
            }
        })
        .collect::<Vec<_>>();
    if !found {
        return Err(BenchError::ExecutionFailed(
            "irregular AC scan program did not declare the haystack input buffer. Fix: preserve the bounded-ranges scan buffer layout before extent binding."
                .to_string(),
        ));
    }
    Ok(program.with_rewritten_buffers(buffers))
}

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
