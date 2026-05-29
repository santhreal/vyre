//! Differential oracle for the Rust nano-subset borrow checks against rustc.
//!
//! The sema borrow checks (`check_mutability` E0596, `check_escape` E0597,
//! `check_conflicts` E0499/E0502 via the CFG dataflow engine) are *sound but
//! incomplete*: they must never reject a program rustc accepts, and must catch
//! the conflict classes they target. This drives the full CPU pipeline
//! (lex -> parse -> resolve -> typeck -> borrow checks) on a generated
//! nano-subset corpus and a curated corpus, and compares the verdict to a real
//! rustc.
//!
//! Requires `rustc` on PATH (the oracle fundamentally depends on it).

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU32, Ordering};

use vyre_libs::parsing::rust::lex::lexer::core::lex;
use vyre_libs::parsing::rust::parse::parse;
use vyre_libs::parsing::rust::sema::{check_conflicts, check_escape, check_mutability, resolve, typeck};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Whether rustc compiles `src` (a borrow-check accept). The corpus is
/// type-correct, so rustc's verdict is purely a borrow-check decision.
fn rustc_accepts(src: &str) -> bool {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("vyre_sema_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&base).expect("create temp dir");
    let src_path = base.join("lib.rs");
    std::fs::write(&src_path, src).expect("write temp source");
    let status = std::process::Command::new("rustc")
        .args(["--edition", "2021", "--crate-type", "lib", "--cap-lints", "allow"])
        .args(["--emit", "metadata", "-o"])
        .arg(base.join("out.rmeta"))
        .arg(&src_path)
        .output()
        .expect("Fix: rustc must be on PATH for the sema borrow oracle");
    let _ = std::fs::remove_dir_all(&base);
    status.status.success()
}

/// Whether the sema pipeline accepts `src`: every stage returns `Ok`. A parse
/// or resolve/type error counts as a (non-borrow) rejection, consistent with
/// rustc rejecting the same program.
fn sema_accepts(src: &str) -> bool {
    let bytes = src.as_bytes();
    let Ok(tokens) = lex(bytes) else { return false };
    let Ok(module) = parse(bytes, &tokens) else { return false };
    let Ok(resolution) = resolve(&module, bytes) else { return false };
    typeck(&module, bytes, &resolution).is_ok()
        && check_mutability(&module, &resolution).is_ok()
        && check_escape(&module, &resolution).is_ok()
        && check_conflicts(&module, &resolution).is_ok()
}

/// Deterministic type-correct nano-subset borrow program (no `pub`, so the
/// nano-subset parser accepts it). Some `mut`/non-`mut` `i32` vars, then a
/// sequence of `&`/`&mut` borrows and deref-uses. `&mut` only targets `mut`
/// vars, so the program never trips E0596; the verdict is purely conflicting
/// borrows.
fn gen_straight(seed: u64) -> String {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let nvars = 2 + (next() % 3) as usize;
    let var_mut: Vec<bool> = (0..nvars).map(|_| next() % 2 == 0).collect();
    let mut s = String::from("fn f() {");
    for (i, &m) in var_mut.iter().enumerate() {
        s.push_str(&format!(" let {}v{i}: i32 = {i};", if m { "mut " } else { "" }));
    }
    let mut borrows = 0usize;
    let mut uses = 0u32;
    for _ in 0..(next() % 8) {
        if next() % 2 == 0 {
            let vk = (next() as usize) % nvars;
            let m = next() % 2 == 0 && var_mut[vk];
            let kw = if m { "mut " } else { "" };
            s.push_str(&format!(" let r{borrows}: &{kw}i32 = &{kw}v{vk};"));
            borrows += 1;
        } else if borrows > 0 {
            let bk = (next() as usize) % borrows;
            s.push_str(&format!(" let u{uses}: i32 = *r{bk};"));
            uses += 1;
        }
    }
    s.push_str(" }");
    s
}

/// Like [`gen_straight`] but borrows are taken at the top level then deref-used
/// inside `if`/`else` arms, exercising cross-branch liveness.
fn gen_branch(seed: u64) -> String {
    let mut state = seed ^ 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let nvars = 2 + (next() % 3) as usize;
    let var_mut: Vec<bool> = (0..nvars).map(|_| next() % 2 == 0).collect();
    let mut s = String::from("fn f() {");
    for (i, &m) in var_mut.iter().enumerate() {
        s.push_str(&format!(" let {}v{i}: i32 = {i};", if m { "mut " } else { "" }));
    }
    let nborrows = 1 + (next() % 4) as usize;
    let mut borrows = 0usize;
    for _ in 0..nborrows {
        let vk = (next() as usize) % nvars;
        let m = next() % 2 == 0 && var_mut[vk];
        let kw = if m { "mut " } else { "" };
        s.push_str(&format!(" let r{borrows}: &{kw}i32 = &{kw}v{vk};"));
        borrows += 1;
    }
    let mut uses = 0u32;
    let mut arm = |s: &mut String, n: u32, next: &mut dyn FnMut() -> u32| {
        for _ in 0..n {
            if borrows > 0 {
                let bk = (next() as usize) % borrows;
                s.push_str(&format!(" let u{uses}: i32 = *r{bk};"));
                uses += 1;
            }
        }
    };
    s.push_str(" if true {");
    arm(&mut s, next() % 3, &mut next);
    s.push_str(" } else {");
    arm(&mut s, next() % 3, &mut next);
    s.push_str(" }; }");
    s
}

/// The core soundness gate: the sema borrow checks must never reject a program
/// rustc accepts (no false positives). Any divergence is a real soundness bug.
fn assert_no_false_reject(cases: u64, gen: impl Fn(u64) -> String) {
    let mut false_rejects = Vec::new();
    for seed in 0..cases {
        let src = gen(seed);
        if rustc_accepts(&src) && !sema_accepts(&src) {
            false_rejects.push(src);
        }
    }
    assert!(
        false_rejects.is_empty(),
        "sema borrow checks rejected {} programs rustc accepts (false positives):\n{}",
        false_rejects.len(),
        false_rejects.join("\n")
    );
}

#[test]
fn sema_never_false_rejects_straight_line_programs() {
    assert_no_false_reject(200, gen_straight);
}

#[test]
fn sema_never_false_rejects_branch_programs() {
    assert_no_false_reject(150, gen_branch);
}

/// Programs with a real conflicting-borrow error: rustc rejects, and the sema
/// conflict engine must catch it too (proving the check is effective, not
/// vacuously sound).
const CONFLICT_REJECT: &[&str] = &[
    "fn f() { let mut x: i32 = 0; let a: &mut i32 = &mut x; let b: &mut i32 = &mut x; let _c: i32 = *a + *b; }",
    "fn f() { let mut x: i32 = 0; let a: &i32 = &x; let b: &mut i32 = &mut x; let _c: i32 = *b + *a; }",
    "fn f() { let mut x: i32 = 0; let a: &mut i32 = &mut x; let b: &mut i32 = &mut x; if true { let _p: i32 = *a; } else { let _q: i32 = *b; }; }",
];

#[test]
fn sema_catches_conflicts_rustc_rejects() {
    for (i, src) in CONFLICT_REJECT.iter().enumerate() {
        assert!(!rustc_accepts(src), "CONFLICT_REJECT[{i}] must be rejected by rustc: {src}");
        assert!(
            !sema_accepts(src),
            "CONFLICT_REJECT[{i}]: sema accepted a conflict rustc rejects: {src}"
        );
    }
}

/// Clean borrow programs rustc accepts; the sema checks must accept them too.
const ACCEPT: &[&str] = &[
    "fn f() { let x: i32 = 0; let a: &i32 = &x; let b: &i32 = &x; let _c: i32 = *a + *b; }",
    "fn f() { let mut x: i32 = 0; let a: &mut i32 = &mut x; let _p: i32 = *a; let b: &mut i32 = &mut x; let _q: i32 = *b; }",
    "fn f() { let mut x: i32 = 0; if true { let a: &mut i32 = &mut x; let _p: i32 = *a; } else { let b: &mut i32 = &mut x; let _q: i32 = *b; }; }",
];

#[test]
fn sema_accepts_what_rustc_accepts() {
    for (i, src) in ACCEPT.iter().enumerate() {
        assert!(rustc_accepts(src), "ACCEPT[{i}] must compile under rustc: {src}");
        assert!(sema_accepts(src), "ACCEPT[{i}]: sema rejected what rustc accepts: {src}");
    }
}
