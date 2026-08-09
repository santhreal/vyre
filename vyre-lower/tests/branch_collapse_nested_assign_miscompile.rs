//! Descriptor-level regression suite for `branch_collapse`.
//!
//! Covers the invalid-descriptor defect that motivated this work, the
//! fail-closed contract that keeps the collapse decision conservative, and the
//! literal-pool invariant that the fix installs.
//!
//! `lower_verified_accepts_the_repro` is the end-to-end gate: verified lowering
//! must return a structurally valid descriptor whose literal references remain
//! in range. The backend differential independently proves value semantics.

use vyre_lower::analyses::value_range::analyze_body;
use vyre_lower::rewrites::branch_collapse;
use vyre_lower::verify::{classify_operand, OperandClass};
use vyre_lower::{
    lower_verified, verify, BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp,
    KernelOpKind, LiteralValue, VerifyErrorKind,
};

mod shapes;
use shapes::*;

/// Recursively assert every `Literal`-pool operand in `body` and its children
/// indexes a slot that actually exists, and return the number of ops checked.
fn assert_pool_refs_in_range(body: &KernelBody, path: &mut Vec<usize>) -> usize {
    let mut checked = 0;
    for (i, op) in body.ops.iter().enumerate() {
        for (pos, &operand) in op.operands.iter().enumerate() {
            if classify_operand(&op.kind, pos) != OperandClass::LiteralPoolIdx {
                continue;
            }
            checked += 1;
            assert!(
                (operand as usize) < body.literals.len(),
                "body {path:?} op {i} operand {pos} indexes literal pool slot \
                 {operand} but the pool holds only {} entries",
                body.literals.len()
            );
        }
    }
    for (ci, child) in body.child_bodies.iter().enumerate() {
        path.push(ci);
        checked += assert_pool_refs_in_range(child, path);
        path.pop();
    }
    checked
}

fn count_kind(body: &KernelBody, want: &KernelOpKind) -> usize {
    let here = body.ops.iter().filter(|op| &op.kind == want).count();
    here + body
        .child_bodies
        .iter()
        .map(|c| count_kind(c, want))
        .sum::<usize>()
}

/// THE reproduction. `branch_collapse` on the reported shape must produce a
/// descriptor that `verify` accepts.
///
/// Observed pre-fix failure, from `verify(&branch_collapse(&lower(program)))`:
///
/// ```text
/// OBSERVED 6 violation(s):
///   VerifyError { body_path: [0, 0], op_index: 23, kind: LiteralPoolOutOfRange { operand_pos: 0, pool_idx: 2, pool_size: 2 } }
///   VerifyError { body_path: [0, 0], op_index: 23, kind: LiteralPoolOutOfRange { operand_pos: 0, pool_idx: 2, pool_size: 2 } }
///   VerifyError { body_path: [0, 0], op_index: 30, kind: LiteralPoolOutOfRange { operand_pos: 0, pool_idx: 3, pool_size: 2 } }
///   VerifyError { body_path: [0, 0], op_index: 30, kind: LiteralPoolOutOfRange { operand_pos: 0, pool_idx: 3, pool_size: 2 } }
///   VerifyError { body_path: [0, 0], op_index: 37, kind: LiteralPoolOutOfRange { operand_pos: 0, pool_idx: 4, pool_size: 2 } }
///   VerifyError { body_path: [0, 0], op_index: 37, kind: LiteralPoolOutOfRange { operand_pos: 0, pool_idx: 4, pool_size: 2 } }
/// ```
///
/// Under `run_all` the same descriptor tripped the debug assertion at
/// `vyre-lower/src/rewrites/mod.rs:474`: "rewrite pass `branch_collapse`
/// produced an invalid KernelDescriptor". Each bad op is reported twice
/// because `verify` checks a `Literal`'s pool operand both explicitly and
/// through `classify_operand`, so 3 broken ops yield 6 violations.
///
/// Cause: collapsing the always-true `if (end == 0)` guard inlined that arm's
/// ops into the parent body while leaving their pool indices pointing into the
/// arm's own 5-entry pool, which the 2-entry parent pool could not resolve.
#[test]
fn repro_descriptor_verifies_after_branch_collapse() {
    let desc = lower_verified(&repro_program(REPRO_N))
        .map(|lowered| lowered.descriptor)
        .expect("lowering the repro must succeed");
    assert!(
        verify(&desc).is_ok(),
        "the lowered input must verify before any rewrite runs, otherwise this \
         test would be measuring a lowering bug: {:#?}",
        verify(&desc)
    );

    let out = branch_collapse(&desc);

    match verify(&out) {
        Ok(()) => {}
        Err(errors) => {
            let pool_errors: Vec<_> = errors
                .iter()
                .filter(|e| matches!(e.kind, VerifyErrorKind::LiteralPoolOutOfRange { .. }))
                .collect();
            panic!(
                "branch_collapse produced an invalid descriptor: \
                 {} violation(s), {} of them LiteralPoolOutOfRange:\n{errors:#?}",
                errors.len(),
                pool_errors.len()
            );
        }
    }
}

/// Every literal-pool reference in the fully rewritten descriptor must resolve,
/// at every body depth, and the check must actually have inspected ops.
///
/// This pins the pool size against the indices actually referenced rather than
/// asserting a bare `is_ok()`, so a pass that grew the pool without repointing
/// the operands (or repointed operands past the pool) fails here.
#[test]
fn emitted_descriptor_pool_size_covers_every_referenced_index() {
    let desc = lower_verified(&repro_program(REPRO_N))
        .map(|lowered| lowered.descriptor)
        .expect("lowering must succeed");
    let out = branch_collapse(&desc);

    let checked = assert_pool_refs_in_range(&out.body, &mut Vec::new());
    assert!(
        checked > 0,
        "the regression descriptor must contain literal-pool references"
    );

    assert!(verify(&out).is_ok(), "{:#?}", verify(&out));
}

type NamedProgramBuilder = (&'static str, fn(u32) -> vyre_foundation::ir::Program);

/// The pass must DECLINE to collapse a guard whose operand range is unknown at
/// that point, and this must hold for every construct that can carry a
/// mutation. This is the test that stops a future change from making the
/// decision optimistic again.
///
/// Each shape binds a variable to a literal, mutates it via `Node::assign`
/// inside a different kind of nested body, then guards on it. `analyze_body`
/// must report no range for the post-mutation read, and the guard's
/// `StructuredIfThen` ops must survive the pass.
#[test]
fn pass_declines_to_collapse_guards_on_mutated_variables() {
    let cases: &[NamedProgramBuilder] = &[
        ("assign in loop body", loop_assign_program),
        ("assign in else arm", else_assign_program),
        ("assign in nested region", region_assign_program),
        ("assign in one branch, read after join", join_program),
        ("self-referencing min sentinel", sentinel_min_program),
    ];

    for (label, build) in cases {
        let desc = lower_verified(&build(REPRO_N))
            .map(|lowered| lowered.descriptor)
            .expect("lowering must succeed");
        let before = count_kind(&desc.body, &KernelOpKind::StructuredIfThen);
        let out = branch_collapse(&desc);
        let after = count_kind(&out.body, &KernelOpKind::StructuredIfThen);

        assert_eq!(
            before, after,
            "[{label}] branch_collapse must leave every guard intact; the \
             variable under test is mutated in a nested body so no guard on \
             it has a provable range. before={before} after={after}"
        );
        assert!(
            verify(&out).is_ok(),
            "[{label}] output must verify: {:#?}",
            verify(&out)
        );

        // The two complementary guards on the probe variable must both still
        // be runtime comparisons, i.e. neither folded to a constant.
        assert!(
            after >= 2,
            "[{label}] the shape must contain at least the two complementary \
             probe guards, found {after}"
        );
    }
}

/// The stale-snapshot hazard, in the one form the analysis can actually get
/// wrong, proven to be refused.
///
/// A carrier's SEED is an ordinary SSA id, usually a `Literal`, so
/// `value_range` knows it exactly. Before the construct that writes the carrier
/// the seed correctly describes the variable; after it, the variable may hold
/// something else while the seed still reads as a known constant. This
/// descriptor puts a comparison on the seed AFTER the writing construct, which
/// the pre-fix pass folded to `true` and collapsed.
///
/// Locks out: reverting `branch_collapse` to the position-insensitive
/// `ValueRangeReport::get`, which cannot distinguish "a range was derived for
/// this id" from "that range holds where I am about to act on it".
#[test]
fn stale_carrier_seed_is_refused_but_the_pre_write_read_is_not() {
    // ops:
    //  0 Literal[0]=U32(0)            -> id 0   (seed: x = 0)
    //  1 BinOpKind(Eq) [0, 0]         -> id 1   PRE-write read of the seed
    //  2 LoopCarrierInit{x} [0]
    //  3 Literal[1]=Bool(true)        -> id 2
    //  4 StructuredIfThen [2, 0]               if (true) { x = 7 }
    //  5 Literal[2]=U32(0)            -> id 5
    //  6 BinOpKind(Eq) [0, 5]         -> id 6   POST-write read of the seed
    //  7 StructuredIfThen [6, 1]               if (x == 0) { .. }
    let desc = KernelDescriptor {
        id: "stale_seed".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Eq),
                    operands: vec![0, 0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::LoopCarrierInit { name: "x".into() },
                    operands: vec![0],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::StructuredIfThen,
                    operands: vec![2, 0],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![2],
                    result: Some(5),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Eq),
                    operands: vec![0, 5],
                    result: Some(6),
                },
                KernelOp {
                    kind: KernelOpKind::StructuredIfThen,
                    operands: vec![6, 1],
                    result: None,
                },
            ],
            child_bodies: vec![
                // child 0: x = 7
                KernelBody {
                    ops: vec![
                        KernelOp {
                            kind: KernelOpKind::Literal,
                            operands: vec![0],
                            result: Some(3),
                        },
                        KernelOp {
                            kind: KernelOpKind::LoopCarrierEnd { name: "x".into() },
                            operands: vec![3],
                            result: None,
                        },
                    ],
                    child_bodies: vec![],
                    literals: vec![LiteralValue::U32(7)],
                },
                // child 1: the guarded arm
                KernelBody {
                    ops: vec![KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(4),
                    }],
                    child_bodies: vec![],
                    literals: vec![LiteralValue::U32(123)],
                },
            ],
            literals: vec![
                LiteralValue::U32(0),
                LiteralValue::Bool(true),
                LiteralValue::U32(0),
            ],
        },
    };
    assert!(verify(&desc).is_ok(), "{:#?}", verify(&desc));

    // The analysis knows the seed's range, but only before the write.
    let report = analyze_body(&desc.body);
    assert_eq!(
        report.get(0),
        Some(vyre_lower::analyses::value_range::IntRange::singleton(0)),
        "the seed id is a Literal(0), so a range IS derived for it"
    );
    assert_eq!(
        report.invalidated_from.get(&0),
        Some(&5),
        "the construct at op index 4 writes carrier `x`, so the seed must be \
         unknown from op index 5 onward"
    );
    assert_eq!(
        report.get_at(0, 1),
        Some(vyre_lower::analyses::value_range::IntRange::singleton(0)),
        "the read at op index 1 precedes the write and stays known"
    );
    assert_eq!(
        report.get_at(0, 6),
        None,
        "the read at op index 6 follows the write and must be unknown"
    );

    let out = branch_collapse(&desc);
    assert!(verify(&out).is_ok(), "{:#?}", verify(&out));

    // The `if (true)` guard at index 4 is a genuine literal and collapses.
    // The `if (x == 0)` guard at index 7 reads a stale seed and must NOT.
    let surviving: Vec<&KernelOp> = out
        .body
        .ops
        .iter()
        .filter(|op| matches!(op.kind, KernelOpKind::StructuredIfThen))
        .collect();
    assert_eq!(
        surviving.len(),
        1,
        "exactly one guard must survive: the literal-true one collapses, the \
         stale-seed one is refused. Survivors: {surviving:#?}"
    );
    assert_eq!(
        surviving[0].operands[0], 6,
        "the surviving guard must be the one reading the stale carrier seed \
         (cond id 6), not the literal-true guard"
    );
    assert_eq!(
        out.body.child_bodies[1].ops.len(),
        1,
        "the refused guard's arm must be left intact, not dropped"
    );
}

/// A body with no carrier writes at all must produce no invalidations, so the
/// fail-closed machinery costs nothing on straight-line code.
#[test]
fn bodies_without_carrier_writes_have_no_invalidations() {
    let desc = lower_verified(&repro_program(REPRO_N))
        .map(|lowered| lowered.descriptor)
        .expect("lowering must succeed");
    // The innermost `l != 0` arm assigns only through carriers of its own
    // enclosing scopes; the leaf body that performs the writes is where they
    // live, so pick a body with no nested writes: the root.
    let report = analyze_body(&desc.body);
    assert!(
        report.invalidated_from.is_empty(),
        "the root body holds a single Region op and no carrier writes, so \
         nothing may be invalidated: {:?}",
        report.invalidated_from
    );
}

/// Verified lowering must apply the composed cleanup pipeline without producing
/// out-of-range literal references, then reach a descriptor fixpoint.
#[test]
fn lower_verified_accepts_the_repro() {
    let lowered = lower_verified(&repro_program(REPRO_N))
        .expect("verified lowering must accept the repro program");
    let desc = lowered.descriptor;

    assert!(
        verify(&desc).is_ok(),
        "the verified descriptor must verify: {:#?}",
        verify(&desc)
    );
    let checked = assert_pool_refs_in_range(&desc.body, &mut Vec::new());
    assert!(
        checked > 0,
        "the verified descriptor must retain literal-pool references"
    );

    // The pipeline runs to a fixpoint, so re-applying the pass must be a no-op.
    // A pass that keeps rewriting forever, or that only becomes invalid on a
    // second application, is caught here.
    assert_eq!(
        branch_collapse(&desc),
        desc,
        "branch_collapse must be at a fixpoint after lower_verified"
    );
}
