//! Differential parity gate: our frontend's accept/reject verdict must match
//! real `rustc` on a curated nano-subset corpus.
//!
//! Our verdict = lex + parse + resolve + typeck + check_mutability + check_escape
//! all succeed. rustc verdict = `rustc --crate-type lib` compiles (lints capped
//! to allow, so only hard errors reject). The corpus is restricted to the rules
//! we implement (name resolution, type errors, E0596, E0597) plus clean
//! programs; conflicting-borrow programs (E0499/E0502) are excluded because that
//! rule is not yet wired (it would be a known divergence, not a bug).

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU32, Ordering};

use vyre_libs::parsing::rust::lex::lexer::core::lex;
use vyre_libs::parsing::rust::parse::parse;
use vyre_libs::parsing::rust::sema::{check_escape, check_mutability, resolve, typeck};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Our verdict: does the implemented frontend accept this program?
fn ours_accepts(src: &str) -> bool {
    let bytes = src.as_bytes();
    let tokens = match lex(bytes) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let module = match parse(bytes, &tokens) {
        Ok(m) => m,
        Err(_) => return false,
    };
    let resolution = match resolve(&module, bytes) {
        Ok(r) => r,
        Err(_) => return false,
    };
    typeck(&module, bytes, &resolution).is_ok()
        && check_mutability(&module, &resolution).is_ok()
        && check_escape(&module, &resolution).is_ok()
}

/// rustc verdict: does `rustc --crate-type lib` accept this program?
fn rustc_accepts(src: &str) -> bool {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let stem = format!("vyre_diff_{}_{}", std::process::id(), n);
    let rs = dir.join(format!("{stem}.rs"));
    let meta = dir.join(format!("{stem}.rmeta"));
    std::fs::write(&rs, src).expect("Fix: must be able to write a temp source file for the rustc differential");
    // current_dir(temp) dodges the workspace rust-toolchain.toml so the default
    // installed toolchain is used; cap-lints=allow so only hard errors reject.
    let output = std::process::Command::new("rustc")
        .current_dir(&dir)
        .args(["--edition", "2021", "--crate-type", "lib", "--cap-lints", "allow", "--emit", "metadata"])
        .arg("-o")
        .arg(&meta)
        .arg(&rs)
        .output()
        .expect("Fix: rustc must be available on PATH; it is the project toolchain");
    let _ = std::fs::remove_file(&rs);
    let _ = std::fs::remove_file(&meta);
    output.status.success()
}

/// Programs rustc accepts and we must accept.
const ACCEPT: &[&str] = &[
    "fn f() {}",
    "fn add(a: i32, b: i32) -> i32 { return a + b; }",
    "fn lt(a: i32, b: i32) -> bool { return a < b; }",
    "fn eq(a: i32, b: i32) -> bool { return a == b; }",
    "fn deref(r: &i32) -> i32 { return *r; }",
    "fn ret_ref(r: &i32) -> &i32 { return r; }",
    "fn deref_ref(r: &i32) -> &i32 { return &*r; }",
    "fn use_let() -> i32 { let x: i32 = 5; return x; }",
    "fn mut_let() -> i32 { let mut x: i32 = 5; return x; }",
    "fn branchy(a: i32, b: i32) -> i32 { if a < b { return b; } else { return a; }; }",
    "fn one() -> i32 { return 1; } fn g() -> i32 { return one(); }",
    "fn mut_borrow() { let mut x: i32 = 0; let r: &mut i32 = &mut x; }",
    "fn shared(x: i32) { let r: &i32 = &x; }",
    "fn arith() -> i32 { return 3 * 4 / 2 - 1 + 0; }",
    "fn boolret() -> bool { return true; }",
    "fn id(x: i32) -> i32 { return x; }",
    "fn two_lets() -> i32 { let a: i32 = 1; let b: i32 = a + 1; return b; }",
    "fn shadow(x: i32) -> i32 { let x: i32 = x + 1; return x; }",
    "fn deref_mut(r: &mut i32) -> i32 { return *r; }",
    "fn mutref(r: &mut i32) -> &mut i32 { return r; }",
];

/// Programs rustc rejects and we must reject.
const REJECT: &[&str] = &[
    "fn f() -> i32 { return true; }",                                  // E0308
    "fn f() { let x: bool = 5; }",                                     // E0308
    "fn f() -> i32 { return missing; }",                               // E0425
    "fn f() -> i32 { return ghost(); }",                               // E0425
    "fn f() { let x: i32 = 0; let r: &mut i32 = &mut x; }",            // E0596
    "fn f(r: &i32) -> &i32 { let x: i32 = 0; return &x; }",            // E0515/E0597
    "fn f() -> i32 { let x: i32 = 0; return *x; }",                    // E0614
    "fn f() -> i32 { return true + 1; }",                              // E0308/E0277
    "fn f() -> i32 { let x: i32 = 0; }",                               // E0308 missing return
    "fn g(x: i32) -> i32 { return x; } fn f() -> i32 { return g(true); }", // E0308 arg
    "fn g(x: i32) -> i32 { return x; } fn f() -> i32 { return g(1, 2); }", // E0061 arity
    "fn f() { if 1 { let x: i32 = 0; }; }",                            // E0308 non-bool cond
    "fn f(x: i32) -> &i32 { return &x; }",                             // E0106/E0515
];

#[test]
fn ours_agrees_with_rustc_on_accept_corpus() {
    for (i, src) in ACCEPT.iter().enumerate() {
        let ours = ours_accepts(src);
        let rustc = rustc_accepts(src);
        assert!(
            rustc,
            "Fix: ACCEPT corpus[{i}] must compile under rustc (adjust the corpus): {src}"
        );
        assert_eq!(ours, rustc, "Fix: verdict mismatch on ACCEPT corpus[{i}] (ours={ours}, rustc={rustc}): {src}");
    }
}

#[test]
fn ours_agrees_with_rustc_on_reject_corpus() {
    for (i, src) in REJECT.iter().enumerate() {
        let ours = ours_accepts(src);
        let rustc = rustc_accepts(src);
        assert!(
            !rustc,
            "Fix: REJECT corpus[{i}] must be rejected by rustc (adjust the corpus): {src}"
        );
        assert_eq!(ours, rustc, "Fix: verdict mismatch on REJECT corpus[{i}] (ours={ours}, rustc={rustc}): {src}");
    }
}
