//! Descriptor-level regression suite for `branch_collapse`.
//!
//! Covers the invalid-descriptor defect that motivated this work, the
//! fail-closed contract that keeps the collapse decision conservative, and the
//! literal-pool invariant that the fix installs.
//!
//! This file, not the backend differential, is the gate for the reported
//! defect. See `lower_for_emit_of_the_repro_does_not_panic` for why: the
//! defect's symptom is a REJECTED descriptor, so a consumer either panics in
//! `rewrites::run_all` or gets an `Err` from its codegen gate. It never
//! reaches the device as a wrong number, so no CPU-versus-CUDA value
//! comparison can witness it. The differential in
//! `branch_collapse_backend_differential.rs` covers the OTHER half of the
//! contract: that collapsing never picks the wrong arm.

use vyre_lower::analyses::value_range::analyze_body;
use vyre_lower::rewrites::branch_collapse;
use vyre_lower::verify::{classify_operand, OperandClass};
use vyre_lower::{
    lower, lower_for_emit, verify, BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp,
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

/// Collect the highest literal-pool index referenced anywhere in `body`.
fn max_pool_ref(body: &KernelBody) -> Option<u32> {
    let mut best: Option<u32> = None;
    for op in &body.ops {
        for (pos, &operand) in op.operands.iter().enumerate() {
            if classify_operand(&op.kind, pos) == OperandClass::LiteralPoolIdx {
                best = Some(best.map_or(operand, |b: u32| b.max(operand)));
            }
        }
    }
    best
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
    let desc = lower(&repro_program(REPRO_N)).expect("lowering the repro must succeed");
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

/// The collapse the repro depends on must still FIRE. Without this, the pool
/// fix could be "achieved" by declining to collapse anything, which would pass
/// the verify test above and silently retire the optimization.
///
/// `if (end == 0)` where `end` is the literal 0 at that program point is a
/// genuine compile-time constant: the assignment to `end` lives in a nested
/// body that has not executed yet. Collapsing it is sound and must be kept.
#[test]
fn legitimate_constant_guard_is_still_collapsed_and_absorbs_the_pool() {
    let desc = lower(&repro_program(REPRO_N)).expect("lowering must succeed");
    let out = branch_collapse(&desc);

    // The arm's body sat at child_bodies[0] of body path [0, 0] and held 5
    // literals; the parent held 2. After a correct inline the parent's pool
    // must have grown to cover the relocated ops.
    let inner = &desc.body.child_bodies[0].child_bodies[0];
    let arm = &inner.child_bodies[0];
    assert_eq!(
        inner.literals.len(),
        2,
        "pre-collapse parent pool size is the premise of this test"
    );
    assert_eq!(
        arm.literals.len(),
        5,
        "pre-collapse arm pool size is the premise of this test"
    );

    let inner_after = &out.body.child_bodies[0].child_bodies[0];
    assert_eq!(
        inner_after.literals.len(),
        8,
        "the parent pool must absorb one slot per distinct arm literal that \
         was relocated into it"
    );

    // The collapsed guard's `StructuredIfThen` is gone from that body, while
    // the guards whose operands are opaque carrier reads survive.
    let ifs_before = inner
        .ops
        .iter()
        .filter(|op| matches!(op.kind, KernelOpKind::StructuredIfThen))
        .count();
    let ifs_after = inner_after
        .ops
        .iter()
        .filter(|op| matches!(op.kind, KernelOpKind::StructuredIfThen))
        .count();
    assert_eq!(
        ifs_before, 1,
        "pre-collapse the inner body holds exactly the one collapsible guard"
    );
    assert_eq!(
        ifs_after, 4,
        "collapsing the one constant guard must splice in the arm's own 4 \
         guards, none of which are collapsible"
    );
}

/// Every literal-pool reference in the fully rewritten descriptor must resolve,
/// at every body depth, and the check must actually have inspected ops.
///
/// This pins the pool size against the indices actually referenced rather than
/// asserting a bare `is_ok()`, so a pass that grew the pool without repointing
/// the operands (or repointed operands past the pool) fails here.
#[test]
fn emitted_descriptor_pool_size_covers_every_referenced_index() {
    let desc = lower(&repro_program(REPRO_N)).expect("lowering must succeed");
    let out = branch_collapse(&desc);

    let checked = assert_pool_refs_in_range(&out.body, &mut Vec::new());
    assert_eq!(
        checked, 20,
        "the repro descriptor carries 20 literal-pool references after \
         collapse; a different count means the shape drifted and the pinned \
         pool assertions below no longer describe it"
    );

    let inner_after = &out.body.child_bodies[0].child_bodies[0];
    assert_eq!(
        max_pool_ref(inner_after),
        Some(7),
        "highest pool index referenced by the collapsed body"
    );
    assert_eq!(
        inner_after.literals.len(),
        8,
        "pool size must be exactly one past the highest referenced index"
    );
    assert_eq!(
        inner_after.literals,
        vec![LiteralValue::U32(0); 8],
        "every relocated literal in this program is U32(0); a different value \
         means a relocated operand resolved to the wrong pool entry"
    );

    assert!(verify(&out).is_ok(), "{:#?}", verify(&out));
}

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
    let cases: &[(&str, fn(u32) -> vyre_foundation::ir::Program)] = &[
        ("assign in loop body", loop_assign_program),
        ("assign in else arm", else_assign_program),
        ("assign in nested region", region_assign_program),
        ("assign in one branch, read after join", join_program),
        ("self-referencing min sentinel", sentinel_min_program),
    ];

    for (label, build) in cases {
        let desc = lower(&build(REPRO_N)).expect("lowering must succeed");
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

/// A guard read STRICTLY BEFORE any write to the carrier must still collapse.
///
/// Locks out over-conservatism: deleting the seed's range outright instead of
/// invalidating it from the write point would also kill this collapse, which is
/// exactly the one the repro program depends on.
#[test]
fn pre_write_carrier_seed_read_still_collapses() {
    let desc = lower(&repro_program(REPRO_N)).expect("lowering must succeed");
    let inner = &desc.body.child_bodies[0].child_bodies[0];

    let report = analyze_body(inner);
    // op 0 = Literal(0) -> id 3 (the `end` seed), op 2 = Eq(3, 4), op 3 =
    // LoopCarrierInit{end}, op 4 = the construct that writes `end`.
    assert_eq!(
        report.invalidated_from.get(&3),
        Some(&5),
        "`end`'s seed must go unknown from op index 5, one past the writing \
         construct at index 4"
    );
    assert!(
        report.get_at(3, 2).is_some(),
        "the `end == 0` guard sits at op index 2, before the write, so its \
         operand range must still be available"
    );
}

/// A body with no carrier writes at all must produce no invalidations, so the
/// fail-closed machinery costs nothing on straight-line code.
#[test]
fn bodies_without_carrier_writes_have_no_invalidations() {
    let desc = lower(&repro_program(REPRO_N)).expect("lowering must succeed");
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

/// The reported panic, at its real entry point.
///
/// `lower_for_emit` is what every emitting consumer calls, and it runs the
/// canonical pass list through `rewrites::run_all`, which debug-asserts that
/// each pass produced a valid descriptor. With the pool bug present this call
/// aborted with the exact reported message:
///
/// ```text
/// thread 'probe' panicked at vyre-lower/src/rewrites/mod.rs:475:9:
/// rewrite pass `branch_collapse` produced an invalid KernelDescriptor
///   - 6 violation(s): LiteralPoolOutOfRange { pool_idx: 2, pool_size: 2 } x2,
///     { pool_idx: 3, pool_size: 2 } x2, { pool_idx: 4, pool_size: 2 } x2
/// ```
///
/// This is the strongest lock in the suite because it exercises the composed
/// pipeline rather than `branch_collapse` in isolation: the passes that run
/// before it reshape the descriptor, so a fix that only satisfies the isolated
/// call could still leave the real pipeline broken.
///
/// Note for anyone extending the backend differential: `CudaBackend::dispatch`
/// does NOT reach this state on this program, because it subgroup-lowers the
/// `Program` first (`vyre-driver-cuda/src/codegen.rs:55` passes
/// `subgroup_lowered` into the descriptor gate) and the perturbed shape stops
/// `branch_collapse` from firing at all. Verified empirically: with the bug
/// reintroduced, `lower_for_emit(repro_program(REPRO_N))` panicked while
/// `CudaBackend::dispatch` on the same program returned correct output. Do not
/// treat a green differential as coverage for this defect.
#[test]
fn lower_for_emit_of_the_repro_does_not_panic() {
    let lowered = lower_for_emit(&repro_program(REPRO_N))
        .expect("the canonical pre-emit pipeline must accept the repro program");
    let desc = lowered.descriptor;

    assert!(
        verify(&desc).is_ok(),
        "the pre-emit descriptor must verify: {:#?}",
        verify(&desc)
    );
    let checked = assert_pool_refs_in_range(&desc.body, &mut Vec::new());
    assert!(
        checked > 0,
        "the pre-emit descriptor must still contain literal-pool references, \
         otherwise this test would pass vacuously on an empty pool"
    );

    // The pipeline runs to a fixpoint, so re-applying the pass must be a no-op.
    // A pass that keeps rewriting forever, or that only becomes invalid on a
    // second application, is caught here.
    assert_eq!(
        branch_collapse(&desc),
        desc,
        "branch_collapse must be at a fixpoint after lower_for_emit"
    );
}
