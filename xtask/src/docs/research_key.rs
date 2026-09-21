//! Recognize an optimization research key.
//!
//! A research key is the uppercase identifier that ties an optimizer pass to the
//! source it came from, and it is cited in documentation between backticks.

/// Whether `value` has the shape of a research key: uppercase, digits and
/// underscores, and not empty.
pub fn is_research_key(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_keys_allow_uppercase_underscores_and_digits() {
        assert!(is_research_key("FLASH_ATTN2"));
        assert!(is_research_key("XAV_FPGA"));
        assert!(is_research_key("CUDA_GRAPHS"));
        assert!(!is_research_key(""));
        assert!(!is_research_key("Flash_Attn2"));
        assert!(!is_research_key("FLASH-ATTN2"));
    }
}
