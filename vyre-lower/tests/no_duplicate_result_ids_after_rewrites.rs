//! Pin-test: every shipped rewrite combination must preserve the
//! "every result_id is unique across the body tree" invariant.
//!
//! emit-naga's `bind_result` uses last-write-wins on a single
//! `values: BTreeMap<u32, Handle>`  -  when two ops in different bodies
//! produce the same result_id, the second overwrites the first and any
//! cross-block read of either binding dangles in the WGSL output
//! (naga's parser rejects with `no definition in scope for identifier
//! _eN`). The licm-shared-allocator regression that produced 32
//! duplicate ids on P-6 (`semantic_pg`) only surfaced after the lex
//! fixes landed and the pipeline reached P-6  -  it had been silently
//! corrupting earlier IRs too. This test would have caught it at
//! commit time.
//!
//! Strategy: build a small kernel that exercises every shape known
//! to trip an id-allocation discipline failure (nested loops, hoist-
//! eligible invariants in inner loops, multiple if-then branches
//! contributing merge-Selects, structured-block scopes), run
//! `vyre_lower::rewrites::run_all_with_stats` on it, and walk the
//! resulting body asserting that every result_id appears exactly
//! once.

use std::collections::BTreeMap;

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::{
    rewrites::run_all_with_stats, BindingLayout, BindingSlot, BindingVisibility, Dispatch,
    KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, MemoryClass,
};

fn count_result_ids(body: &KernelBody) -> BTreeMap<u32, u32> {
    fn walk(body: &KernelBody, out: &mut BTreeMap<u32, u32>) {
        for op in &body.ops {
            if let Some(r) = op.result {
                *out.entry(r).or_insert(0) += 1;
            }
        }
        for child in &body.child_bodies {
            walk(child, out);
        }
    }
    let mut out = BTreeMap::new();
    walk(body, &mut out);
    out
}

fn assert_no_duplicates(body: &KernelBody, label: &str) {
    let counts = count_result_ids(body);
    let dups: Vec<_> = counts.iter().filter(|(_, &n)| n > 1).collect();
    assert!(
        dups.is_empty(),
        "{label}: {} duplicate result_ids after rewrites: {:?}",
        dups.len(),
        dups.iter().take(8).collect::<Vec<_>>(),
    );
}

fn nested_loop_with_hoistable_invariant_descriptor() -> KernelDescriptor {
    // Outer loop 0..N, body contains:
    //   - an inner loop 0..M whose body has a Literal op (hoist
    //     candidate at depth-2)
    //   - an if-then branch that conditionally rebinds a value
    //     (forces merge-Select emission in the parent body)
    // Plus a top-level loop and an outer-scope hoist candidate.
    // This is the shape that previously produced duplicate ids when
    // licm's per-recursion subtree-only id-max scan missed sibling
    // hoists.
    let bindings = vec![BindingSlot {
        slot: 0,
        element_type: DataType::U32,
        element_count: Some(64),
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: "out".into(),
    }];

    // SSA id allocation (manual, must match a plausible lower output)
    // 0..3: pre-loop literals + GlobalInvocationId
    // 10..14: inner loop body computations
    // 20..24: outer loop body computations
    let ops = vec![
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        },
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![1],
            result: Some(1),
        },
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![2],
            result: Some(2),
        },
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![3],
            result: Some(3),
        },
        KernelOp {
            kind: KernelOpKind::GlobalInvocationId,
            operands: vec![0],
            result: Some(4),
        },
        KernelOp {
            kind: KernelOpKind::StructuredForLoop {
                loop_var: "i".into(),
            },
            operands: vec![0, 1, 0],
            result: None,
        },
        KernelOp {
            kind: KernelOpKind::StoreGlobal,
            operands: vec![0, 4, 24],
            result: None,
        },
    ];
    let inner_body = KernelBody {
        ops: vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(10),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![10, 2],
                result: Some(11),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Mul),
                operands: vec![11, 3],
                result: Some(12),
            },
        ],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(7)],
    };
    let outer_body = KernelBody {
        ops: vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(20),
            },
            KernelOp {
                kind: KernelOpKind::StructuredForLoop {
                    loop_var: "j".into(),
                },
                operands: vec![2, 3, 0],
                result: None,
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![20, 12],
                result: Some(24),
            },
        ],
        child_bodies: vec![inner_body],
        literals: vec![LiteralValue::U32(5)],
    };
    KernelDescriptor {
        id: "nested_loop_hoist".into(),
        bindings: BindingLayout { slots: bindings },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![outer_body],
            literals: vec![
                LiteralValue::U32(0),
                LiteralValue::U32(8),
                LiteralValue::U32(0),
                LiteralValue::U32(4),
            ],
        },
    }
}

#[test]
fn input_descriptor_starts_with_unique_ids() {
    let desc = nested_loop_with_hoistable_invariant_descriptor();
    assert_no_duplicates(&desc.body, "input descriptor");
}

#[test]
fn nested_loop_hoist_produces_no_duplicate_result_ids() {
    let desc = nested_loop_with_hoistable_invariant_descriptor();
    let (out, _stats) = run_all_with_stats(&desc);
    assert_no_duplicates(&out.body, "post-rewrites");
}

/// Shape that used to make `loop_unroll` mint colliding ids.
///
/// The top-level body holds a loop too long to unroll (0..100 against
/// `MAX_UNROLL_COUNT` of 4), so `loop_unroll` leaves it alone and
/// recurses into its child body. That child holds a SHORT inner loop
/// (0..2) that does get unrolled. The ids inside the child are small
/// (3..5) while the top-level body already uses 6 and 7, so a free-id
/// counter reseeded from the child's own subtree hands the unrolled
/// copies ids 6 and 7 and stamps them on top of the enclosing body's
/// `GlobalInvocationId` and store value.
fn short_loop_nested_under_long_loop_descriptor() -> KernelDescriptor {
    let bindings = vec![BindingSlot {
        slot: 0,
        element_type: DataType::U32,
        element_count: Some(64),
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: "out".into(),
    }];

    // Unrolled twice, so the two copies want two fresh ids.
    let unrollable_inner_body = KernelBody {
        ops: vec![
            KernelOp {
                kind: KernelOpKind::LoopIndex {
                    loop_var: "inner".into(),
                },
                operands: vec![],
                result: Some(5),
            },
            KernelOp {
                kind: KernelOpKind::StoreGlobal,
                operands: vec![0, 5, 5],
                result: None,
            },
        ],
        child_bodies: vec![],
        literals: vec![],
    };
    let long_loop_body = KernelBody {
        ops: vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(3),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(4),
            },
            KernelOp {
                kind: KernelOpKind::StructuredForLoop {
                    loop_var: "inner".into(),
                },
                operands: vec![3, 4, 0],
                result: None,
            },
        ],
        child_bodies: vec![unrollable_inner_body],
        literals: vec![LiteralValue::U32(0), LiteralValue::U32(2)],
    };
    KernelDescriptor {
        id: "short_loop_under_long_loop".into(),
        bindings: BindingLayout { slots: bindings },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![2],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::GlobalInvocationId,
                    operands: vec![0],
                    result: Some(6),
                },
                KernelOp {
                    kind: KernelOpKind::StructuredForLoop {
                        loop_var: "outer".into(),
                    },
                    operands: vec![0, 2, 0],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(7),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 6, 7],
                    result: None,
                },
            ],
            child_bodies: vec![long_loop_body],
            literals: vec![
                LiteralValue::U32(0),
                LiteralValue::U32(2),
                LiteralValue::U32(100),
            ],
        },
    }
}

#[test]
fn short_loop_under_long_loop_input_is_well_formed() {
    let desc = short_loop_nested_under_long_loop_descriptor();
    assert_no_duplicates(&desc.body, "input descriptor");
    assert_eq!(vyre_lower::verify::verify(&desc), Ok(()));
}

/// `loop_unroll` in isolation must not reuse an id the enclosing body
/// already owns. Before the fix this produced `%6` and `%7` twice: once
/// in the top-level body and once inside each unrolled copy.
#[test]
fn loop_unroll_does_not_reuse_enclosing_body_ids() {
    let desc = short_loop_nested_under_long_loop_descriptor();
    let unrolled = vyre_lower::rewrites::loop_unroll(&desc);
    assert_no_duplicates(&unrolled.body, "post loop_unroll");
    assert_eq!(vyre_lower::verify::verify(&unrolled), Ok(()));
}

/// The unrolled copies must be genuinely new ids, not a renumbering that
/// merely happens to avoid the two the test fixture checks. Every id the
/// unrolled body introduces sits strictly above the input's maximum.
#[test]
fn loop_unroll_allocates_ids_above_the_whole_descriptor_maximum() {
    let desc = short_loop_nested_under_long_loop_descriptor();
    let input_ids = count_result_ids(&desc.body);
    let input_max = *input_ids.keys().max().expect("fixture produces results");
    assert_eq!(input_max, 7, "fixture's highest input id");

    let unrolled = vyre_lower::rewrites::loop_unroll(&desc);
    let output_ids = count_result_ids(&unrolled.body);
    let introduced: Vec<u32> = output_ids
        .keys()
        .copied()
        .filter(|id| !input_ids.contains_key(id))
        .collect();
    assert_eq!(
        introduced,
        vec![8, 9],
        "two unrolled iterations, each taking the next descriptor-wide free id"
    );
}

/// End to end through the full rewrite pipeline, which is what the
/// backends actually run.
#[test]
fn short_loop_under_long_loop_survives_the_full_pipeline() {
    let desc = short_loop_nested_under_long_loop_descriptor();
    let (out, _stats) = run_all_with_stats(&desc);
    assert_no_duplicates(&out.body, "post-rewrites");
    assert_eq!(vyre_lower::verify::verify(&out), Ok(()));
}

#[test]
fn idempotent_rewrites_preserve_unique_ids() {
    // Run the rewrite suite TWICE  -  id allocation must remain stable
    // across re-rewrite. Previously a stale `next_free_id` could
    // collide with ids freshly allocated by the previous iteration.
    let desc = nested_loop_with_hoistable_invariant_descriptor();
    let (once, _) = run_all_with_stats(&desc);
    assert_no_duplicates(&once.body, "post-rewrites (1st pass)");
    let (twice, _) = run_all_with_stats(&once);
    assert_no_duplicates(&twice.body, "post-rewrites (2nd pass)");
}
