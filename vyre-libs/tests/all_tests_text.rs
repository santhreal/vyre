//! One binary for every text integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/ir_shape/mod.rs`.
#[cfg(feature = "text")]
#[allow(clippy::needless_range_loop)]
#[path = "ir_shape/mod.rs"]
pub mod ir_shape;

/// Shared fixture module from `tests/text_char_class_runner/mod.rs`.
#[cfg(feature = "text")]
#[allow(clippy::needless_range_loop)]
#[path = "text_char_class_runner/mod.rs"]
pub mod text_char_class_runner;

/// Integration tests from `tests/adversarial_text_byte_histogram.rs`.
#[cfg(feature = "text")]
#[allow(clippy::needless_range_loop)]
#[path = "adversarial_text_byte_histogram.rs"]
pub mod adversarial_text_byte_histogram;

/// Integration tests from `tests/adversarial_text_char_class.rs`.
#[cfg(feature = "text")]
#[path = "adversarial_text_char_class.rs"]
pub mod adversarial_text_char_class;

/// Integration tests from `tests/adversarial_text_line_index.rs`.
#[cfg(feature = "text")]
#[path = "adversarial_text_line_index.rs"]
pub mod adversarial_text_line_index;

/// Integration tests from `tests/adversarial_text_utf8_shape_counts.rs`.
#[cfg(feature = "text")]
#[allow(clippy::needless_range_loop)]
#[path = "adversarial_text_utf8_shape_counts.rs"]
pub mod adversarial_text_utf8_shape_counts;

/// Integration tests from `tests/adversarial_text_utf8_validate.rs`.
#[cfg(feature = "text")]
#[path = "adversarial_text_utf8_validate.rs"]
pub mod adversarial_text_utf8_validate;

/// Integration tests from `tests/proptest_text_encoding_classify.rs`.
#[cfg(feature = "text")]
#[path = "proptest_text_encoding_classify.rs"]
pub mod proptest_text_encoding_classify;

/// Integration tests from `tests/proptest_text_line_index.rs`.
#[cfg(feature = "text")]
#[path = "proptest_text_line_index.rs"]
pub mod proptest_text_line_index;

/// Integration tests from `tests/proptest_text_utf8_validate.rs`.
#[cfg(feature = "text")]
#[path = "proptest_text_utf8_validate.rs"]
pub mod proptest_text_utf8_validate;

/// Integration tests from `tests/sweep_text_utf8_oracle_matrix.rs`.
#[cfg(feature = "text")]
#[path = "sweep_text_utf8_oracle_matrix.rs"]
pub mod sweep_text_utf8_oracle_matrix;
