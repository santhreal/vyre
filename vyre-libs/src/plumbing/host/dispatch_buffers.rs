//! Byte-buffer helpers for semantic execution.
//!
//! A wrapper builds a schedule-free `Program` and crosses the semantic boundary
//! with byte buffers. Shared shape checks and little-endian u32 marshalling
//! prevent divergent host-side contracts.
//!
//! The marshalling delegates to `vyre_primitives::wire`, which is why this sits
//! here rather than beside the seam in `vyre-foundation`: foundation is below
//! `vyre-primitives` and cannot reach the packing it would have to duplicate.

use vyre_megakernel::SemanticExecutionError;

/// Graph node identity for a single-program wrapper dispatch.
///
/// `ProgramGraph::from_program` shares one name space between the node and
/// every graph value, and a graph value takes the Program buffer name. A
/// wrapper that passes its readback label instead collides with whichever
/// buffer that label names.
pub const HOST_WRAPPER_NODE: &str = "vyre-libs-host-wrapper";

/// Return `n * n` as `usize`, rejecting zero and overflow with an actionable
/// dispatcher error.
pub fn checked_square_cells(n: u32, context: &str) -> Result<usize, SemanticExecutionError> {
    if n == 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires n > 0."
        )));
    }
    let n_us = n as usize;
    n_us.checked_mul(n_us).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} n*n overflows usize for n={n}."
        ))
    })
}

/// Return `left * right` as `usize`, rejecting zeros and overflow with an
/// actionable dispatcher error.
pub fn checked_product_count(
    left: u32,
    right: u32,
    left_name: &str,
    right_name: &str,
    context: &str,
) -> Result<usize, SemanticExecutionError> {
    if left == 0 || right == 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires {left_name} > 0 and {right_name} > 0, got {left_name}={left}, {right_name}={right}."
        )));
    }
    let left_us = left as usize;
    let right_us = right as usize;
    left_us.checked_mul(right_us).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} {left_name}*{right_name} overflows usize for {left_name}={left}, {right_name}={right}."
        ))
    })
}

/// Encode a u32 slice as little-endian bytes for dispatcher input buffers.
///
/// Routes through the canonical `vyre-primitives::wire::pack_u32_slice`
/// LEGO primitive (with `bytemuck::cast_slice` fast path on LE hosts).
/// Dispatcher input-buffer encoding now matches every other GPU upload
/// path's throughput floor instead of running its own scalar `extend`
/// loop.
pub use vyre_primitives::wire::pack_u32_slice as u32_slice_to_le_bytes;

/// Ensure a dispatcher input-vector shell has exactly `count` reusable slots.
///
/// Dispatcher calls consume the whole `Vec<Vec<u8>>`. Leaving stale slots after
/// a scratch object moves from a wider primitive to a narrower primitive silently
/// changes the backend ABI and can force needless uploads. Active slots keep
/// their allocation; inactive slots are dropped instead of being passed on.
pub fn ensure_input_slots(inputs: &mut Vec<Vec<u8>>, count: usize) {
    if inputs.len() < count {
        inputs.resize_with(count, Vec::new);
    } else if inputs.len() > count {
        inputs.truncate(count);
    }
}

/// Fill a reusable dispatcher byte buffer with zeros without replacing the
/// allocation.
pub fn write_zero_bytes(out: &mut Vec<u8>, len: usize) {
    if out.len() == len {
        if out.iter().any(|&byte| byte != 0) {
            out.fill(0);
        }
    } else {
        out.clear();
        out.resize(len, 0);
    }
}

/// Return the exact byte count needed for `count` u32 words.
pub fn u32_word_bytes(count: usize, context: &str) -> Result<usize, SemanticExecutionError> {
    count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: {context} byte count overflows usize for {count} u32 word(s)."
            ))
        })
}

/// Fill a reusable dispatcher byte buffer with `count` zeroed u32 words.
pub fn write_zero_u32_words(
    out: &mut Vec<u8>,
    count: usize,
    context: &str,
) -> Result<(), SemanticExecutionError> {
    let bytes = u32_word_bytes(count, context)?;
    write_zero_bytes(out, bytes);
    Ok(())
}

/// Encode a u32 slice as little-endian bytes into caller-owned dispatcher
/// input storage. Routes through `vyre-primitives::wire::pack_u32_slice_into`
/// so dispatcher writes use the same LE-host `bytemuck::cast_slice` fast
/// path as every other GPU-upload site.
pub fn write_u32_slice_le_bytes(out: &mut Vec<u8>, values: &[u32]) {
    vyre_primitives::wire::pack_u32_slice_into(values, out);
}

/// Encode an f32 slice as little-endian bytes for dispatcher input buffers.
///
/// Routes through the canonical `vyre-primitives::wire::pack_f32_slice`
/// LEGO primitive (with `bytemuck::cast_slice` fast path on LE hosts).
pub use vyre_primitives::wire::pack_f32_slice as f32_slice_to_le_bytes;

/// Decode an aligned u32 input buffer for a CPU-parity dispatcher.
pub fn decode_u32_input_aligned(
    bytes: &[u8],
    context: &str,
) -> Result<Vec<u32>, SemanticExecutionError> {
    if bytes.len() % std::mem::size_of::<u32>() != 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} input byte count {} is not divisible by 4.",
            bytes.len()
        )));
    }
    Ok(vyre_primitives::wire::decode_u32_le_bytes_all(bytes))
}

/// Decode an aligned f32 input buffer for a CPU-parity dispatcher.
pub fn decode_f32_input_aligned(
    bytes: &[u8],
    context: &str,
) -> Result<Vec<f32>, SemanticExecutionError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} input byte count {} is not divisible by 4.",
            bytes.len()
        )));
    }
    Ok(vyre_primitives::wire::decode_f32_le_bytes_all(bytes))
}

/// Decode a u32 byte buffer for a CPU-parity dispatcher that intentionally
/// validates through the same lenient scalar oracle used before centralization.
#[must_use]
pub fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    vyre_primitives::wire::decode_u32_le_bytes_all(bytes)
}

/// Decode an f32 byte buffer for a CPU-parity dispatcher that intentionally
/// validates through the same lenient scalar oracle used before centralization.
#[must_use]
pub fn read_f32s(bytes: &[u8]) -> Vec<f32> {
    vyre_primitives::wire::decode_f32_le_bytes_all(bytes)
}

/// Encode an f32 slice as little-endian bytes into caller-owned dispatcher
/// input storage.
pub fn write_f32_slice_le_bytes(out: &mut Vec<u8>, values: &[f32]) {
    vyre_primitives::wire::pack_f32_slice_into(values, out);
}

/// Return the sole dispatcher output buffer and reject missing or surplus buffers.
pub(crate) fn require_exactly_one_output<'a>(
    outputs: &'a [Vec<u8>],
    context: &str,
) -> Result<&'a [u8], SemanticExecutionError> {
    let [output] = outputs else {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {context} expected exactly one output buffer, got {}.",
            outputs.len()
        )));
    };
    Ok(output)
}

/// Position of the writable buffer `name` within returned dispatcher outputs.
///
/// Output buffers correspond to non-Workgroup buffers that are either backend-allocated
/// or declared with `BufferAccess::ReadWrite`, in `program.buffers()` order.
#[must_use]
pub(crate) fn output_buffer_index(
    program: &vyre_foundation::ir::Program,
    name: &str,
) -> Option<usize> {
    program.output_buffer_indices().iter().position(|&index| {
        program
            .buffers()
            .get(index as usize)
            .is_some_and(|decl| decl.name() == name)
    })
}

/// Return the output buffer corresponding to `name` from a program's returned outputs.
///
/// Accepts programs that return multiple output buffers (such as scratch or write-complete
/// sibling buffers) and extracts the requested buffer by name.
pub(crate) fn require_named_output<'a>(
    outputs: &'a [Vec<u8>],
    program: &vyre_foundation::ir::Program,
    name: &str,
    context: &str,
) -> Result<&'a [u8], SemanticExecutionError> {
    let index = output_buffer_index(program, name).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} program does not declare an output buffer named `{name}`."
        ))
    })?;

    outputs.get(index).map(|buf| buf.as_slice()).ok_or_else(|| {
        SemanticExecutionError::Backend(format!(
            "Fix: {context} expected output buffer `{name}` at index {index}, but dispatcher returned {} output buffer(s).",
            outputs.len()
        ))
    })
}

/// Decode a dispatcher u32 output buffer with exact byte-count validation.
pub fn decode_u32_output_exact(
    bytes: &[u8],
    expected_words: usize,
    context: &str,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let expected_bytes = expected_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            SemanticExecutionError::Backend(format!(
                "Fix: {context} output byte count overflowed usize."
            ))
        })?;
    if bytes.len() != expected_bytes {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {context} expected {expected_bytes} output bytes, got {}.",
            bytes.len()
        )));
    }

    vyre_primitives::wire::unpack_u32_slice_into(bytes, expected_words, context, out)
        .map_err(SemanticExecutionError::Backend)
}

/// Decode a dispatcher i32 output buffer with exact byte-count validation.
pub fn decode_i32_output_exact(
    bytes: &[u8],
    expected_words: usize,
    context: &str,
    out: &mut Vec<i32>,
) -> Result<(), SemanticExecutionError> {
    let expected_bytes = expected_words
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| {
            SemanticExecutionError::Backend(format!(
                "Fix: {context} output byte count overflowed usize."
            ))
        })?;
    if bytes.len() != expected_bytes {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {context} expected {expected_bytes} output bytes, got {}.",
            bytes.len()
        )));
    }

    out.clear();
    out.reserve(expected_words);
    out.extend(vyre_primitives::wire::decode_i32_le_bytes_all(bytes));
    Ok(())
}

/// Decode a dispatcher f32 output buffer with exact byte-count validation.
pub fn decode_f32_output_exact(
    bytes: &[u8],
    expected_words: usize,
    context: &str,
    out: &mut Vec<f32>,
) -> Result<(), SemanticExecutionError> {
    let expected_bytes = expected_words
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            SemanticExecutionError::Backend(format!(
                "Fix: {context} output byte count overflowed usize."
            ))
        })?;
    if bytes.len() != expected_bytes {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {context} expected {expected_bytes} output bytes, got {}.",
            bytes.len()
        )));
    }

    vyre_primitives::wire::unpack_f32_slice_into(bytes, expected_words, context, out)
        .map_err(SemanticExecutionError::Backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_word_bytes_rejects_usize_overflow() {
        let overflowing_words = usize::MAX / std::mem::size_of::<u32>() + 1;
        let err = u32_word_bytes(overflowing_words, "dispatch-buffer test")
            .expect_err("overflowing u32 word count must be rejected");
        assert!(
            matches!(&err, SemanticExecutionError::InvalidRequest(message) if message.contains("dispatch-buffer test")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn zero_u32_words_preserves_allocation_and_exact_byte_count() {
        let mut bytes = Vec::with_capacity(64);
        let ptr = bytes.as_ptr();
        write_zero_u32_words(&mut bytes, 3, "zero test").expect("Fix: zeroing succeeds");
        assert_eq!(bytes, vec![0; 12]);
        assert_eq!(bytes.as_ptr(), ptr);
    }

    #[test]
    fn zero_bytes_reuses_already_sized_zero_buffer_without_reallocation() {
        let mut bytes = vec![0u8; 32];
        let ptr = bytes.as_ptr();

        write_zero_bytes(&mut bytes, 32);

        assert_eq!(bytes, vec![0; 32]);
        assert_eq!(bytes.as_ptr(), ptr);
    }

    #[test]
    fn zero_bytes_clears_dirty_same_size_buffer_without_reallocation() {
        let mut bytes = vec![0xA5u8; 32];
        let ptr = bytes.as_ptr();

        write_zero_bytes(&mut bytes, 32);

        assert_eq!(bytes, vec![0; 32]);
        assert_eq!(bytes.as_ptr(), ptr);
    }

    #[test]
    fn dispatcher_output_count_is_exact() {
        let empty: Vec<Vec<u8>> = Vec::new();
        let missing = require_exactly_one_output(&empty, "dispatch-buffer test")
            .expect_err("missing output must fail");
        assert!(missing
            .to_string()
            .contains("exactly one output buffer, got 0"));

        let extra = vec![vec![1], vec![2]];
        let surplus = require_exactly_one_output(&extra, "dispatch-buffer test")
            .expect_err("surplus output must fail");
        assert!(surplus
            .to_string()
            .contains("exactly one output buffer, got 2"));

        let one = vec![vec![3]];
        assert_eq!(
            require_exactly_one_output(&one, "dispatch-buffer test").expect("one output is exact"),
            [3]
        );
    }

    #[test]
    fn require_named_output_extracts_by_name_and_accepts_sibling_scratch() {
        use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};

        let program = Program::wrapped(
            vec![
                BufferDecl::storage("in_ro", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("primary_out", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("scratch_sibling", 2, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(8),
            ],
            [1, 1, 1],
            Vec::new(),
        );

        assert_eq!(output_buffer_index(&program, "primary_out"), Some(0));
        assert_eq!(output_buffer_index(&program, "scratch_sibling"), Some(1));
        assert_eq!(output_buffer_index(&program, "in_ro"), None);
        assert_eq!(output_buffer_index(&program, "missing"), None);

        let outputs = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8, 9, 10, 11, 12]];

        let primary = require_named_output(&outputs, &program, "primary_out", "test_extract")
            .expect("primary output extracted");
        assert_eq!(primary, &[1, 2, 3, 4]);

        let scratch = require_named_output(&outputs, &program, "scratch_sibling", "test_extract")
            .expect("scratch output extracted");
        assert_eq!(scratch, &[5, 6, 7, 8, 9, 10, 11, 12]);

        let missing_decl = require_named_output(&outputs, &program, "missing", "test_extract")
            .expect_err("undeclared buffer must fail with InvalidRequest");
        assert!(matches!(
            missing_decl,
            SemanticExecutionError::InvalidRequest(_)
        ));

        let empty_outputs: Vec<Vec<u8>> = Vec::new();
        let missing_slot =
            require_named_output(&empty_outputs, &program, "primary_out", "test_extract")
                .expect_err("missing dispatcher slot must fail with BackendError");
        assert!(matches!(missing_slot, SemanticExecutionError::Backend(_)));
    }
}
