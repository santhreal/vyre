//! Overflow-checked sizing for frontier queue buffers, and the invalid-program
//! diagnostic emitted when a size cannot be represented.

use std::fmt;

use vyre_foundation::ir::{DataType, Program};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FrontierQueueSizingError {
    pub(super) message: String,
}

impl FrontierQueueSizingError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

impl fmt::Display for FrontierQueueSizingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

pub(super) fn checked_frontier_u32_product(
    lhs: u32,
    rhs: u32,
    context: &str,
) -> Result<u32, FrontierQueueSizingError> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FrontierQueueSizingError::new(format!(
            "Fix: {context} overflows u32 word count for lhs={lhs} rhs={rhs}. Shard the frontier queue before GPU dispatch."
        ))
    })
}

pub(super) fn try_u32_byte_range(
    words: u32,
    context: &str,
) -> Result<usize, FrontierQueueSizingError> {
    try_u32_byte_range_with_word_size(words, std::mem::size_of::<u32>(), context)
}

pub(super) fn try_u32_byte_range_with_word_size(
    words: u32,
    word_size: usize,
    context: &str,
) -> Result<usize, FrontierQueueSizingError> {
    let count = usize::try_from(words).map_err(|_| {
        FrontierQueueSizingError::new(format!(
            "Fix: {context} words={words} cannot fit usize on this target. Shard the frontier queue before GPU dispatch."
        ))
    })?;
    count.checked_mul(word_size).ok_or_else(|| {
        FrontierQueueSizingError::new(format!(
            "Fix: {context} words={words} word_size={word_size} overflows output byte range. Shard the frontier queue before GPU dispatch."
        ))
    })
}

pub(super) fn invalid_frontier_queue_sizing_program(
    op_id: &'static str,
    output: &str,
    error: FrontierQueueSizingError,
) -> Program {
    crate::invalid_output_program(op_id, output, DataType::U32, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{checked_frontier_u32_product, try_u32_byte_range_with_word_size};

    #[test]
    fn frontier_queue_sizing_overflow_returns_error_without_panic() {
        let byte_result = std::panic::catch_unwind(|| {
            try_u32_byte_range_with_word_size(2, usize::MAX, "test frontier queue bytes")
        });
        assert!(
            byte_result.is_ok(),
            "checked frontier byte sizing must return an error instead of panicking"
        );
        let err = byte_result.unwrap().unwrap_err().to_string();
        assert!(
            err.contains("overflows output byte range"),
            "Fix: byte sizing overflow must name the byte-range contract, got: {err}"
        );
        assert!(
            err.contains("Shard the frontier queue"),
            "Fix: byte sizing overflow must tell the operator how to recover, got: {err}"
        );

        let product_result = std::panic::catch_unwind(|| {
            checked_frontier_u32_product(u32::MAX, 2, "test partial word count")
        });
        assert!(
            product_result.is_ok(),
            "checked frontier word products must return an error instead of panicking"
        );
        let err = product_result.unwrap().unwrap_err().to_string();
        assert!(
            err.contains("overflows u32 word count"),
            "Fix: word-count overflow must name the u32 product contract, got: {err}"
        );
    }
}
