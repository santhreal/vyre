//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for text::char_class

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]

fn reference_char_class(source: &[u8], table: &[u32; 256]) -> Vec<u32> {
    source
        .iter()
        .map(|byte| table[usize::from(*byte)])
        .collect()
}

mod adversarial_text_char_class_part1 {

    include!("__split/adversarial_text_char_class_part1.rs");
}
mod adversarial_text_char_class_part2 {
    include!("__split/adversarial_text_char_class_part2.rs");
}
mod adversarial_text_char_class_part3 {
    include!("__split/adversarial_text_char_class_part3.rs");
}
