//! Differential fuzzer: hundreds of generated, type- and name-correct borrow
//! programs run through both our checker and real rustc; their accept/reject
//! verdicts must agree. The only verdict-affecting variation is the borrow
//! pattern (mutability + conflicts), so this fuzzes exactly the borrow rules
//! against the compiler at scale and surfaces any divergence.

#![forbid(unsafe_code)]

mod diff_support;

use diff_support::{ours_accepts, rustc_accepts};
use proptest::prelude::*;

/// Render a nano program from a borrow plan: some `i32` vars (each maybe `mut`),
/// then a sequence of borrow/use ops over the in-scope vars and borrows. Every
/// program is type- and name-correct by construction; the borrow pattern varies.
fn render(var_mut: &[bool], ops: &[(u8, usize, bool)]) -> String {
    let mut s = String::from("fn f() {");
    for (i, &m) in var_mut.iter().enumerate() {
        s.push_str(&format!(" let {}v{}: i32 = {};", if m { "mut " } else { "" }, i, i));
    }
    let mut borrow_count = 0usize;
    let mut next_use = 0u32;
    for &(kind, a, b) in ops {
        if kind == 0 {
            // Borrow an in-scope var (shared or mutable).
            let vk = a % var_mut.len();
            let m = if b { "mut " } else { "" };
            s.push_str(&format!(" let r{}: &{}i32 = &{}v{};", borrow_count, m, m, vk));
            borrow_count += 1;
        } else if borrow_count > 0 {
            // Use (deref) an existing borrow, extending its live range.
            let bid = a % borrow_count;
            s.push_str(&format!(" let u{}: i32 = *r{};", next_use, bid));
            next_use += 1;
        }
    }
    s.push_str(" }");
    s
}

/// Like `render`, but the borrows are top-level and then used inside the arms
/// of an `if`/`else`, fuzzing cross-branch liveness: borrows used in separate
/// reachable arms are live across the branch point (so two `&mut` of one place
/// conflict), exercising the engine's CFG dataflow against rustc.
fn render_branch(
    var_mut: &[bool],
    top: &[(u8, usize, bool)],
    then_uses: &[usize],
    else_uses: &[usize],
) -> String {
    let mut s = String::from("fn f() {");
    for (i, &m) in var_mut.iter().enumerate() {
        s.push_str(&format!(" let {}v{}: i32 = {};", if m { "mut " } else { "" }, i, i));
    }
    let mut borrow_count = 0usize;
    let mut next_use = 0u32;
    for &(kind, a, b) in top {
        if kind == 0 {
            let vk = a % var_mut.len();
            let m = if b { "mut " } else { "" };
            s.push_str(&format!(" let r{}: &{}i32 = &{}v{};", borrow_count, m, m, vk));
            borrow_count += 1;
        } else if borrow_count > 0 {
            let bid = a % borrow_count;
            s.push_str(&format!(" let u{}: i32 = *r{};", next_use, bid));
            next_use += 1;
        }
    }
    s.push_str(" if true {");
    if borrow_count > 0 {
        for &i in then_uses {
            s.push_str(&format!(" let u{}: i32 = *r{};", next_use, i % borrow_count));
            next_use += 1;
        }
    }
    s.push_str(" } else {");
    if borrow_count > 0 {
        for &i in else_uses {
            s.push_str(&format!(" let u{}: i32 = *r{};", next_use, i % borrow_count));
            next_use += 1;
        }
    }
    s.push_str(" }; }");
    s
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn ours_matches_rustc_on_generated_borrow_programs(
        var_mut in prop::collection::vec(any::<bool>(), 2..=4),
        ops in prop::collection::vec((0u8..2u8, any::<usize>(), any::<bool>()), 0..=8),
    ) {
        let src = render(&var_mut, &ops);
        let ours = ours_accepts(&src);
        let rustc = rustc_accepts(&src);
        prop_assert_eq!(
            ours, rustc,
            "borrow-verdict mismatch (ours={}, rustc={}):\n  {}",
            ours, rustc, src
        );
    }

    #[test]
    fn ours_matches_rustc_on_generated_branch_programs(
        var_mut in prop::collection::vec(any::<bool>(), 2..=4),
        top in prop::collection::vec((0u8..2u8, any::<usize>(), any::<bool>()), 0..=6),
        then_uses in prop::collection::vec(any::<usize>(), 0..=4),
        else_uses in prop::collection::vec(any::<usize>(), 0..=4),
    ) {
        let src = render_branch(&var_mut, &top, &then_uses, &else_uses);
        let ours = ours_accepts(&src);
        let rustc = rustc_accepts(&src);
        prop_assert_eq!(
            ours, rustc,
            "branch borrow-verdict mismatch (ours={}, rustc={}):\n  {}",
            ours, rustc, src
        );
    }
}
