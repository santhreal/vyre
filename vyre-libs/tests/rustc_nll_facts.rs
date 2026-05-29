//! Validation: load rustc's real `-Znll-facts` dump for a Rust function and
//! check that our NLL loan-liveness verdict ([`RustcNllFacts::accepts`]) agrees
//! with rustc's own borrow-check verdict. This proves the producer runs the
//! borrow check on real MIR-level facts, not just our nano-subset's AST.
//!
//! Requires a nightly rustc (the only toolchain that emits `-Znll-facts`); the
//! producer fundamentally depends on it, so absence is a loud failure, not a
//! silent skip.

#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use vyre_libs::borrowck::rustc_facts::load_facts;

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn read_relation(fn_dir: &Path, name: &str) -> String {
    std::fs::read_to_string(fn_dir.join(format!("{name}.facts"))).unwrap_or_default()
}

/// Dump rustc nll-facts for `src`, returning `(rustc_accepts, our_accepts)`.
/// `our_accepts` is true iff every dumped function borrow-checks under our NLL
/// loan-liveness rule.
fn verdicts(src: &str) -> (bool, bool) {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("vyre_nll_{}_{}", std::process::id(), n));
    let facts_dir = base.join("facts");
    std::fs::create_dir_all(&facts_dir).expect("create temp facts dir");
    let src_path = base.join("lib.rs");
    std::fs::write(&src_path, src).expect("write temp source");

    let output = std::process::Command::new("rustc")
        .arg("+nightly")
        .args(["--edition", "2021", "--crate-type", "lib", "--cap-lints", "allow"])
        .arg("-Znll-facts")
        .arg(format!("-Znll-facts-dir={}", facts_dir.display()))
        .args(["--emit", "metadata"])
        .arg("-o")
        .arg(base.join("out.rmeta"))
        .arg(&src_path)
        .output()
        .expect("Fix: rustc must be on PATH; the nll-facts producer test needs it");
    let rustc_accepts = output.status.success();

    let function_dirs: Vec<_> = std::fs::read_dir(&facts_dir)
        .expect("read facts dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    assert!(
        !function_dirs.is_empty(),
        "rustc emitted no nll-facts directories; is the active rustc a nightly with -Znll-facts? \
         stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut our_accepts = true;
    for fn_dir in &function_dirs {
        let facts = load_facts(|name| read_relation(fn_dir, name));
        if !facts.accepts() {
            our_accepts = false;
        }
    }

    let _ = std::fs::remove_dir_all(&base);
    (rustc_accepts, our_accepts)
}

/// Type-correct functions whose accept/reject is purely a borrow-check decision,
/// so rustc's verdict is exactly what our NLL rule must reproduce.
const ACCEPT: &[&str] = &[
    "pub fn f() { let x: i32 = 0; let a: &i32 = &x; let b: &i32 = &x; let _c: i32 = *a + *b; }",
    "pub fn f() { let mut x: i32 = 0; let a: &mut i32 = &mut x; let _p: i32 = *a; let b: &mut i32 = &mut x; let _q: i32 = *b; }",
    "pub fn f() { let mut x: i32 = 0; if true { let a: &mut i32 = &mut x; let _p: i32 = *a; } else { let b: &mut i32 = &mut x; let _q: i32 = *b; } }",
    "pub fn f() { let mut x: i32 = 0; let a: &mut i32 = &mut x; let _b: &mut i32 = &mut x; let _c: i32 = *_b; }",
    // NLL across a loop: each iteration's &mut is dead before the next.
    "pub fn f() { let mut x: i32 = 0; let mut i: i32 = 0; while i < 10 { let a: &mut i32 = &mut x; *a = i; i = i + 1; } }",
    // Reborrow of a &mut parameter.
    "pub fn f(p: &mut i32) { let a: &mut i32 = &mut *p; *a = 1; }",
    // Aliasing shared borrows through a call is allowed.
    "pub fn g(a: &i32, b: &i32) -> i32 { return *a + *b; } pub fn f() { let x: i32 = 0; let r: &i32 = &x; let _z: i32 = g(r, r); }",
];

const REJECT: &[&str] = &[
    "pub fn f() { let mut x: i32 = 0; let a: &mut i32 = &mut x; let b: &mut i32 = &mut x; let _c: i32 = *a + *b; }",
    "pub fn f() { let mut x: i32 = 0; let a: &i32 = &x; let b: &mut i32 = &mut x; let _c: i32 = *a + *b; }",
    "pub fn f() { let mut x: i32 = 0; let a: &mut i32 = &mut x; let b: &mut i32 = &mut x; if true { let _p: i32 = *a; } else { let _q: i32 = *b; } }",
    // Assign to a place while it is shared-borrowed and the borrow is later used.
    "pub fn f() { let mut x: i32 = 0; let a: &i32 = &x; x = 1; let _b: i32 = *a; }",
];

#[test]
fn our_nll_verdict_matches_rustc_on_accept_corpus() {
    for (i, src) in ACCEPT.iter().enumerate() {
        let (rustc_accepts, our_accepts) = verdicts(src);
        assert!(rustc_accepts, "ACCEPT[{i}] must compile under rustc: {src}");
        assert!(
            our_accepts,
            "ACCEPT[{i}]: our NLL rule rejected what rustc accepts: {src}"
        );
    }
}

#[test]
fn our_nll_verdict_matches_rustc_on_reject_corpus() {
    for (i, src) in REJECT.iter().enumerate() {
        let (rustc_accepts, our_accepts) = verdicts(src);
        assert!(!rustc_accepts, "REJECT[{i}] must be rejected by rustc: {src}");
        assert!(
            !our_accepts,
            "REJECT[{i}]: our NLL rule accepted what rustc rejects: {src}"
        );
    }
}
