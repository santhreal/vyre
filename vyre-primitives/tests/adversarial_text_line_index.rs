//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for text::line_index

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]

use vyre_primitives::text::line_index::*;

fn reference_line_index(source: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(source.len());
    let mut line = 0u32;
    let mut prev_was_cr = false;
    for byte in source.iter().copied() {
        if prev_was_cr && byte != b'\n' {
            line = line.wrapping_add(1);
        }
        out.push(line);
        if byte == b'\n' {
            line = line.wrapping_add(1);
            prev_was_cr = false;
        } else {
            prev_was_cr = byte == b'\r';
        }
    }
    out
}

mod adversarial_text_line_index_part1 {

    include!("__split/adversarial_text_line_index_part1.rs");
}
mod adversarial_text_line_index_part2 {
    include!("__split/adversarial_text_line_index_part2.rs");
}
mod adversarial_text_line_index_part3 {
    include!("__split/adversarial_text_line_index_part3.rs");
}
