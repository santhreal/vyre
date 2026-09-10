//! One binary for every integration test in this crate.

#[path = "ir_shape/mod.rs"]
pub mod ir_shape;

#[path = "text_char_class_runner/mod.rs"]
pub mod text_char_class_runner;

#[path = "adversarial_text_byte_histogram.rs"]
pub mod adversarial_text_byte_histogram;

#[path = "adversarial_text_char_class.rs"]
pub mod adversarial_text_char_class;

#[path = "adversarial_text_line_index.rs"]
pub mod adversarial_text_line_index;

#[path = "adversarial_text_utf8_shape_counts.rs"]
pub mod adversarial_text_utf8_shape_counts;

#[path = "adversarial_text_utf8_validate.rs"]
pub mod adversarial_text_utf8_validate;

#[path = "proptest_text_byte_histogram.rs"]
pub mod proptest_text_byte_histogram;

#[path = "proptest_text_char_class.rs"]
pub mod proptest_text_char_class;

#[path = "proptest_text_encoding_classify.rs"]
pub mod proptest_text_encoding_classify;

#[path = "proptest_text_line_index.rs"]
pub mod proptest_text_line_index;

#[path = "proptest_text_utf8_validate.rs"]
pub mod proptest_text_utf8_validate;

#[path = "sweep_text_utf8_oracle_matrix.rs"]
pub mod sweep_text_utf8_oracle_matrix;

#[path = "utf8_shape_counts_ir_parity_proptest.rs"]
pub mod utf8_shape_counts_ir_parity_proptest;
