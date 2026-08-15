//! Counted proof that the PTX cache key's digest is computed once per program
//! value rather than once per dispatch.
//!
//! `CudaBackend::ptx_for_program_cached_with_key` derives its cache key from a
//! program, and the normalized digest feeding that key is memoized ON THE
//! PROGRAM VALUE. The defect locked out here is neither a wrong digest nor a
//! broken memo: both were correct in isolation. It is that the key used to be
//! derived from `lower_subgroup_reductions(program.clone(), caps)`, a value
//! created and dropped inside a single dispatch, so the memo's only writer was
//! a temporary and the memo could never be read. A memo whose only writer is a
//! temporary is a memo that cannot ever be read. Measured before the fix: 79 ns
//! per IR node, 92 percent of the host PTX phase, and 6 recomputes per `cjk`
//! encode that should have been 1.
//!
//! HOW A MEMO READ IS OBSERVED HERE, since it decides whether these tests can
//! fail. The memo is private, this crate forbids `unsafe` so an allocation
//! counter is unavailable, and a duration would be a timing on a contended box.
//! What remains is exact and better than all three: write a program field that
//! the digest READS but that direct field assignment does not invalidate, then
//! ask for the digest. A warm program returns its pre-mutation bytes, which a
//! real computation could not produce; a cold program returns post-mutation
//! bytes. Every test below pairs that observation with a control proving the
//! mutation IS visible to a real computation, so "the digest did not change"
//! can only mean the memo was read and never that the mutation was inert.
//!
//! The direct field write is a MEASUREMENT DEVICE, not a supported way to edit
//! a program. Production mutation goes through the setters, which clear all six
//! memos on purpose. Nothing here should be copied as a mutation pattern.

use std::sync::Arc;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::lower::lower_subgroup_reductions;
use vyre_foundation::optimizer::AdapterCaps;

fn caps() -> AdapterCaps {
    AdapterCaps {
        supports_subgroup_ops: true,
        subgroup_size: 32,
        ..AdapterCaps::default()
    }
}

/// A program with `statements` body nodes and no subgroup reduction, so the
/// subgroup lowering pass is a no-op on it.
fn wide_program(statements: u32) -> Program {
    let body = (0..statements)
        .map(|index| Node::store("out", Expr::u32(index), Expr::u32(index)))
        .collect::<Vec<_>>();
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(statements)],
        [64, 1, 1],
        body,
    )
}

fn digest_of(program: &Program) -> [u8; 32] {
    vyre_driver::try_normalized_program_cache_digest(program)
        .expect("Fix: fixture program must produce a normalized cache digest")
}

/// PRECONDITION for every memo test below: the field used as the probe MUST be
/// one the digest actually reads.
///
/// Runs first in spirit, not in order, because the whole technique collapses if
/// `workgroup_size` were absent from the digest input. Then a warm and a cold
/// program would agree for the trivial reason that the mutation is invisible,
/// and every "the memo was read" assertion would pass while proving nothing.
/// This is the negative control that licenses the others.
#[test]
fn the_probe_field_is_visible_to_a_real_digest_computation() {
    let baseline = digest_of(&wide_program(64));

    let mut mutated_before_any_digest = wide_program(64);
    mutated_before_any_digest.workgroup_size = [32, 1, 1];

    assert_ne!(
        digest_of(&mutated_before_any_digest),
        baseline,
        "Fix: the normalized digest no longer reads workgroup_size, so it cannot \
         be used to observe a memo read and every memo test in this file is \
         silently vacuous. Pick another field the digest reads."
    );
}

/// A long-lived program MUST compute its digest exactly ONCE, no matter how many
/// dispatches follow.
///
/// This is the fix, stated as a count of computations rather than as a duration.
/// The program is warmed once, then the probe field is written seven times and
/// the digest requested after each write. A recompute would see the new
/// workgroup size and return different bytes, so seven identical answers means
/// seven memo reads and exactly one computation. Before the fix this path
/// computed on every dispatch: 6 times per `cjk` encode.
#[test]
fn a_long_lived_program_computes_its_digest_once_across_many_dispatches() {
    let mut program = wide_program(512);
    let warm = digest_of(&program);

    for dispatch in 2..=8u32 {
        program.workgroup_size = [dispatch, 1, 1];
        assert_eq!(
            digest_of(&program),
            warm,
            "Fix: dispatch {dispatch} recomputed the normalized digest instead of \
             reading the memo, so the PTX cache key is being derived from a \
             value that does not outlive one dispatch."
        );
    }
}

/// NEGATIVE CONTROL: the pre-fix shape, a fresh clone per dispatch off a COLD
/// original, MUST recompute every single time.
///
/// This is what makes the test above meaningful. `Program::clone` forwards a
/// WARM memo, so this control never warms the original: a permanently cold
/// original is exactly the state the old dispatch path was stuck in, and it is
/// why reading the `Clone` impl could never reveal the defect. Each iteration
/// mutates the probe field before digesting, so a recompute is observable as a
/// CHANGED digest. All four answers must differ from the first, and from each
/// other, proving four separate computations.
#[test]
fn a_per_dispatch_clone_off_a_cold_original_recomputes_every_time() {
    let program = wide_program(512);
    let adapter = caps();

    let mut digests = Vec::new();
    for dispatch in 1..=4u32 {
        let mut lowered = lower_subgroup_reductions(program.clone(), &adapter);
        assert!(
            Arc::ptr_eq(&program.entry, &lowered.entry),
            "Fix: fixture must be a lowering no-op, or this control is measuring \
             a rewrite rather than a redundant recompute."
        );
        lowered.workgroup_size = [dispatch, 1, 1];
        digests.push(digest_of(&lowered));
    }

    assert_eq!(
        digests.len(),
        4,
        "Fix: the control must observe every dispatch it claims to."
    );
    for (index, digest) in digests.iter().enumerate() {
        for (other_index, other) in digests.iter().enumerate().skip(index + 1) {
            assert_ne!(
                digest, other,
                "Fix: dispatches {index} and {other_index} returned the same \
                 digest despite different workgroup sizes, so the pre-fix shape \
                 stopped recomputing and the memo-lifetime test above is no \
                 longer proving anything."
            );
        }
    }
}

/// A warm original MUST hand its memo to a clone, so the fix compounds instead
/// of only helping the program it warmed.
///
/// The mechanism half of the fix: once the key is derived from the long-lived
/// program, that program is warm, and every later per-dispatch clone inherits
/// the digest for free. If `Clone` stopped forwarding the memo this would fail
/// while the tests above still passed, because they never clone after warming.
#[test]
fn a_clone_of_a_warm_program_inherits_the_digest_without_recomputing() {
    let program = wide_program(256);
    let warm = digest_of(&program);

    let mut clone = program.clone();
    clone.workgroup_size = [7, 1, 1];

    assert_eq!(
        digest_of(&clone),
        warm,
        "Fix: a clone of a warm program recomputed its digest, so warming the \
         caller's program stops paying off for the per-dispatch clone and the \
         saving does not compound across dispatches."
    );
}

/// A program the lowering pass actually REWRITES must not share the unlowered
/// program's digest.
///
/// The false-no-op guard, restated where the digests themselves are visible.
/// Keying a rewritten program on its unlowered digest would file lowered PTX
/// under the unlowered program's identity, so a later dispatch of the unlowered
/// form would be served a kernel containing subgroup reductions it never asked
/// for. Unlike the memo defect, that is a wrong-kernel bug and would not show
/// up as a slowdown at all.
#[test]
fn a_rewritten_program_does_not_share_the_unlowered_digest() {
    let program = Program::wrapped(
        vec![BufferDecl::output("scratch", 0, DataType::U32).with_count(64)],
        [64, 1, 1],
        vec![Node::Region {
            generator: "vyre-primitives::reduce::workgroup_sum_u32".into(),
            source_region: None,
            body: Arc::new(vec![Node::store("scratch", Expr::u32(0), Expr::u32(7))]),
        }],
    );
    let lowered = lower_subgroup_reductions(program.clone(), &caps());

    assert!(
        !Arc::ptr_eq(&program.entry, &lowered.entry),
        "Fix: fixture must actually be rewritten by subgroup lowering, or this \
         test cannot see a false no-op."
    );
    assert_ne!(
        digest_of(&program),
        digest_of(&lowered),
        "Fix: subgroup lowering rewrote the program without moving its digest, \
         so the PTX cache cannot tell the lowered and unlowered forms apart."
    );
}
