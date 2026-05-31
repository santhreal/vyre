//! Parity test: vyre-primitives char_class + bracket_match match
//! their reference oracles.

#![cfg(test)]

mod common;

use common::{bytes_u32, u32_bytes, with_live_backend};
use vyre::DispatchConfig;
use vyre_primitives::matching::bracket_match::{
    bracket_match, bracket_match_dispatch_grid, cpu_ref as bracket_cpu, CLOSE_BRACE, MATCH_NONE,
    OPEN_BRACE, OTHER,
};
use vyre_primitives::text::char_class::{
    build_char_class_table, char_class, char_class_dispatch_grid, reference_char_class,
};

fn bytes_to_u32_per_lane(source: &[u8]) -> Vec<u32> {
    source.iter().map(|&b| b as u32).collect()
}

fn run_char_class(source: &[u8], table: &[u32; 256]) -> Vec<u32> {
    let n = source.len() as u32;
    let program = char_class("source", "classified", n);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(&bytes_to_u32_per_lane(source)), u32_bytes(table)];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(char_class_dispatch_grid(n));
    let outputs = with_live_backend("char class", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA char-class dispatch failed: {error}"))
    });
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(n as usize);
    out
}

#[test]
fn cuda_char_class_alpha_digit_ws() {
    let table = build_char_class_table();
    let source = b"A1 ";
    let cpu = reference_char_class(source, &table);
    let gpu = run_char_class(source, &table);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_char_class_mixed_ascii() {
    let table = build_char_class_table();
    let source = b"Hello, World!";
    let cpu = reference_char_class(source, &table);
    let gpu = run_char_class(source, &table);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_char_class_underscore_treated_as_alpha() {
    let table = build_char_class_table();
    let source = b"foo_bar123";
    let cpu = reference_char_class(source, &table);
    let gpu = run_char_class(source, &table);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_char_class_all_byte_values_past_first_workgroup() {
    let table = build_char_class_table();
    let source: Vec<u8> = (0u8..=255).cycle().take(1029).collect();
    let cpu = reference_char_class(&source, &table);
    let gpu = run_char_class(&source, &table);

    assert_eq!(gpu, cpu);
    for (idx, byte) in source.iter().copied().enumerate() {
        assert_eq!(gpu[idx], table[usize::from(byte)]);
    }
}

fn run_bracket_match(kinds: &[u32], max_depth: u32) -> Vec<u32> {
    let n = kinds.len() as u32;
    let program = bracket_match("kinds", "stack", "match_pairs", n, max_depth);
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(kinds),
        // stack scratch: zero-init.
        vec![0u8; max_depth as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(bracket_match_dispatch_grid(n, max_depth));
    let outputs = with_live_backend("bracket match", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA bracket-match dispatch failed: {error}"))
    });
    // Buffer order: 0:kinds(RO) 1:stack(RW) 2:match_pairs(output).
    // outputs[0]=stack, outputs[1]=match_pairs.
    let mut out = bytes_u32(&outputs[1]);
    out.truncate(n as usize);
    out
}

#[test]
fn cuda_bracket_match_simple_pair() {
    // {x} → indices 0 OPEN, 1 OTHER, 2 CLOSE.
    let kinds = vec![OPEN_BRACE, OTHER, CLOSE_BRACE];
    let cpu = bracket_cpu(&kinds, 3);
    let gpu = run_bracket_match(&kinds, 3);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_bracket_match_nested_pairs() {
    // {{}} → 0 OPEN, 1 OPEN, 2 CLOSE, 3 CLOSE.
    let kinds = vec![OPEN_BRACE, OPEN_BRACE, CLOSE_BRACE, CLOSE_BRACE];
    let cpu = bracket_cpu(&kinds, 4);
    let gpu = run_bracket_match(&kinds, 4);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_bracket_match_unbalanced_open_left_unmatched() {
    // {{} → 0 OPEN, 1 OPEN, 2 CLOSE. Inner pair 1↔2; outer 0 unmatched.
    let kinds = vec![OPEN_BRACE, OPEN_BRACE, CLOSE_BRACE];
    let cpu = bracket_cpu(&kinds, 3);
    let gpu = run_bracket_match(&kinds, 3);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_bracket_match_extra_close_dropped() {
    // }{} → 0 CLOSE (no opening), 1 OPEN, 2 CLOSE.
    let kinds = vec![CLOSE_BRACE, OPEN_BRACE, CLOSE_BRACE];
    let cpu = bracket_cpu(&kinds, 3);
    let gpu = run_bracket_match(&kinds, 3);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_bracket_match_parallel_crosses_workgroup_boundaries() {
    let mut kinds = vec![OTHER; 513];
    kinds[0] = OPEN_BRACE;
    kinds[300] = OPEN_BRACE;
    kinds[301] = CLOSE_BRACE;
    kinds[512] = CLOSE_BRACE;

    let cpu = bracket_cpu(&kinds, kinds.len() as u32);
    let gpu = run_bracket_match(&kinds, kinds.len() as u32);

    assert_eq!(
        bracket_match_dispatch_grid(kinds.len() as u32, kinds.len() as u32),
        [3, 1, 1]
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu[0], 512);
    assert_eq!(gpu[512], 0);
    assert_eq!(gpu[300], 301);
    assert_eq!(gpu[301], 300);
}

#[test]
fn cuda_bracket_match_bounded_depth_stays_exact_for_overflow_opens() {
    let kinds = vec![
        OPEN_BRACE,
        OPEN_BRACE,
        OPEN_BRACE,
        CLOSE_BRACE,
        CLOSE_BRACE,
        CLOSE_BRACE,
    ];
    let cpu = bracket_cpu(&kinds, 2);
    let gpu = run_bracket_match(&kinds, 2);

    assert_eq!(
        bracket_match_dispatch_grid(kinds.len() as u32, 2),
        [1, 1, 1]
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![4, 3, MATCH_NONE, 1, 0, MATCH_NONE]);
}

#[test]
fn cuda_bracket_match_parallel_generated_mixed_tokens() {
    let mut state = 0xBADC_0DEu32;
    let mut kinds = Vec::with_capacity(1029);
    for index in 0..1029u32 {
        state = state.rotate_left(7) ^ index.wrapping_mul(0x9E37_79B9);
        let kind = match state % 6 {
            0 | 1 => OPEN_BRACE,
            2 | 3 => CLOSE_BRACE,
            _ => OTHER,
        };
        kinds.push(kind);
    }

    let cpu = bracket_cpu(&kinds, kinds.len() as u32);
    let gpu = run_bracket_match(&kinds, kinds.len() as u32);

    assert_eq!(
        bracket_match_dispatch_grid(kinds.len() as u32, kinds.len() as u32),
        [5, 1, 1]
    );
    assert_eq!(gpu, cpu);
}
