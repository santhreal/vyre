//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for text::utf8_validate

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]

use vyre_primitives::text::utf8_validate::*;

fn reference_utf8_validate(source: &[u8]) -> Vec<u32> {
    (0..source.len())
        .map(|idx| cpu_class_at(source, idx))
        .collect()
}

fn cpu_is_cont(byte: u8) -> bool {
    matches!(byte, 0x80..=0xBF)
}

fn cpu_valid_lead2(source: &[u8], idx: usize) -> bool {
    matches!(source[idx], 0xC2..=0xDF) && source.get(idx + 1).copied().is_some_and(cpu_is_cont)
}

fn cpu_valid_lead3(source: &[u8], idx: usize) -> bool {
    let Some(&b1) = source.get(idx + 1) else {
        return false;
    };
    let Some(&b2) = source.get(idx + 2) else {
        return false;
    };
    let first_ok = match source[idx] {
        0xE0 => matches!(b1, 0xA0..=0xBF),
        0xE1..=0xEC | 0xEE..=0xEF => cpu_is_cont(b1),
        0xED => matches!(b1, 0x80..=0x9F),
        _ => false,
    };
    first_ok && cpu_is_cont(b2)
}

fn cpu_valid_lead4(source: &[u8], idx: usize) -> bool {
    let Some(&b1) = source.get(idx + 1) else {
        return false;
    };
    let Some(&b2) = source.get(idx + 2) else {
        return false;
    };
    let Some(&b3) = source.get(idx + 3) else {
        return false;
    };
    let first_ok = match source[idx] {
        0xF0 => matches!(b1, 0x90..=0xBF),
        0xF1..=0xF3 => cpu_is_cont(b1),
        0xF4 => matches!(b1, 0x80..=0x8F),
        _ => false,
    };
    first_ok && cpu_is_cont(b2) && cpu_is_cont(b3)
}

fn cpu_valid_cont_position(source: &[u8], idx: usize) -> bool {
    idx.checked_sub(1).is_some_and(|lead| {
        cpu_valid_lead2(source, lead)
            || cpu_valid_lead3(source, lead)
            || cpu_valid_lead4(source, lead)
    }) || idx
        .checked_sub(2)
        .is_some_and(|lead| cpu_valid_lead3(source, lead) || cpu_valid_lead4(source, lead))
        || idx
            .checked_sub(3)
            .is_some_and(|lead| cpu_valid_lead4(source, lead))
}

fn cpu_class_at(source: &[u8], idx: usize) -> u32 {
    match source[idx] {
        0x00..=0x7F => UTF8_ASCII,
        0x80..=0xBF if cpu_valid_cont_position(source, idx) => UTF8_CONT,
        0x80..=0xBF => UTF8_INVALID,
        0xC2..=0xDF if cpu_valid_lead2(source, idx) => UTF8_LEAD_2,
        0xE0..=0xEF if cpu_valid_lead3(source, idx) => UTF8_LEAD_3,
        0xF0..=0xF4 if cpu_valid_lead4(source, idx) => UTF8_LEAD_4,
        _ => UTF8_INVALID,
    }
}

mod adversarial_text_utf8_validate_part1 {

    include!("__split/adversarial_text_utf8_validate_part1.rs");
}
mod adversarial_text_utf8_validate_part2 {
    include!("__split/adversarial_text_utf8_validate_part2.rs");
}
mod adversarial_text_utf8_validate_part3 {
    include!("__split/adversarial_text_utf8_validate_part3.rs");
}
