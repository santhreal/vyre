//! Hostile-input half of the lexer oracle: the substrate lexer must fail loudly
//! on bytes it cannot tokenize, never report a silent wrong Match and never panic.
//!
//! WHY: `lexer_oracle.rs` drives the differential oracle (substrate lexer vs
//! `rustc_lexer`) over a clean nano-subset corpus, so it can only observe
//! divergence on input the lexer already handles. The failure mode it cannot see
//! is a byte outside the subset that the lexer accepts anyway and tokenizes
//! wrongly: rustc and the substrate then disagree on content the caller never
//! learns about. This target supplies the inputs that must be rejected, plus the
//! valid cases whose agreement proves the rejections are not the lexer refusing
//! everything.

#![forbid(unsafe_code)]

mod oracle_support;
use oracle_support::{lexer_parity, OracleResult};

/// Valid nano-subset programs: the oracle must report byte agreement.
const VALID: &[&str] = &[
    "",
    "fn f(){}",
    "fn f() -> i32 { return 0; }",
    "fn f(a: &mut i32) -> bool { let mut x: bool = true; return x; }",
    "fn f(a: i32, b: i32) -> i32 { if a == b { return a; } else { return a + b * 2 / 1 - 0; }; }",
    "// comment\nfn f() { /* block */ }",
    // `%` is a wired subset operator (lexer -> parse -> typeck -> lower).
    "fn f(a: i32) -> i32 { return a % 3; }",
];

/// Hostile or out-of-subset inputs: the substrate lexer must reject them
/// loudly (no clean Match, no panic). These are bytes the lexer genuinely
/// cannot tokenize (an unrecognised punctuation byte or invalid UTF-8), so it
/// returns a `Lex` error that surfaces as a Mismatch. Note: operators that the
/// lexer *can* tokenize and that agree with rustc byte-for-byte (e.g. `%`, or
/// `>>` as two `>`) are not lexer-hostile even when they are rejected later by
/// the parser; those belong to parser/sema tests, not this lexer oracle.
const HOSTILE: &[&[u8]] = &[
    b"@",                           // unsupported punctuation
    b"\xff\xfe",                    // invalid utf-8
    b"fn f() { let s = \"str\"; }", // string literal: `\"` is un-lexable
    b"#[attr] fn f(){}",            // attribute: `#` is un-lexable
    b"fn f() { a.b }",              // field/method access: `.` is un-lexable
    b"fn f() { let a = [1]; }",     // array literal: `[` is un-lexable
];

#[test]
fn valid_nano_subset_agrees_with_rustc_byte_for_byte() {
    for (i, src) in VALID.iter().enumerate() {
        match lexer_parity(src.as_bytes()) {
            OracleResult::Match => {}
            OracleResult::Mismatch(why) => {
                panic!("Fix: substrate lexer diverged from rustc on valid case[{i}] {src:?}: {why}")
            }
        }
    }
}

#[test]
fn hostile_inputs_fail_loudly_without_silent_match() {
    for (i, bytes) in HOSTILE.iter().enumerate() {
        assert!(
            bytes.len() <= 4096,
            "Fix: hostile case {i} must stay bounded (len={})",
            bytes.len()
        );
        if let OracleResult::Match = lexer_parity(bytes) {
            panic!("Fix: hostile case {i} ({bytes:?}) must not silently match rustc");
        }
    }
}
