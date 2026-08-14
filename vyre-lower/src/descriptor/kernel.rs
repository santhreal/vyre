//! Kernel descriptor behavior: dispatch construction, body hashing, and the
//! read-only analyses emitters run over nested bodies.

use super::{Dispatch, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, KernelOpsIter};

impl Dispatch {
    /// Create a workgroup dispatch shape.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            workgroup_size: [x, y, z],
        }
    }
}

impl Eq for KernelBody {}

impl std::hash::Hash for KernelBody {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ops.hash(state);
        self.child_bodies.hash(state);
        for lit in &self.literals {
            lit.hash(state);
        }
    }
}

impl Eq for KernelDescriptor {}

impl std::hash::Hash for KernelDescriptor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.bindings.hash(state);
        self.dispatch.hash(state);
        self.body.hash(state);
    }
}

impl KernelDescriptor {
    /// One-line human-readable summary. Useful for diagnostic output.
    /// Format: `"<id>: N ops, M bindings, K child bodies, L literals,
    /// dispatch [x, y, z]"`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}: {} ops, {} bindings, {} child bodies, {} literals, dispatch {:?}",
            crate::pattern_audit::display_kernel_id(&self.id),
            self.body.ops.len(),
            self.bindings.slots.len(),
            self.body.child_bodies.len(),
            self.body.literals.len(),
            self.dispatch.workgroup_size,
        )
    }

    /// Terser alternative to [`Self::summary`]. Format: `"<id>(N ops, M bindings)"`.
    /// Useful for compact terminal output where the full summary is
    /// too noisy.
    #[must_use]
    pub fn summary_compact(&self) -> String {
        format!(
            "{}({} ops, {} bindings)",
            crate::pattern_audit::display_kernel_id(&self.id),
            self.body.ops.len(),
            self.bindings.slots.len(),
        )
    }

    /// Total op count across the parent body AND every nested child
    /// body, recursively. The parent-only `body.ops.len()` is the
    /// flat count; this is the deep count.
    #[must_use]
    pub fn total_ops(&self) -> usize {
        fn walk(b: &KernelBody) -> usize {
            b.ops.len() + b.child_bodies.iter().map(walk).sum::<usize>()
        }
        walk(&self.body)
    }

    /// Total number of bodies (the parent counts as 1, plus each
    /// nested child recursively). Useful for "how nested is this
    /// kernel?" telemetry  -  a kernel with one big flat body has
    /// `body_count() == 1`; one with deep control flow has more.
    #[must_use]
    pub fn body_count(&self) -> usize {
        fn walk(b: &KernelBody) -> usize {
            1 + b.child_bodies.iter().map(walk).sum::<usize>()
        }
        walk(&self.body)
    }

    /// Maximum nesting depth of child bodies. A flat kernel returns
    /// `0`. A kernel with one If returns `1`. An If-inside-an-If
    /// returns `2`. Useful for routing decisions (deeply-nested
    /// kernels may need a different optimization strategy).
    #[must_use]
    pub fn max_body_depth(&self) -> usize {
        fn walk(b: &KernelBody) -> usize {
            b.child_bodies
                .iter()
                .map(|c| 1 + walk(c))
                .max()
                .unwrap_or(0)
        }
        walk(&self.body)
    }

    /// Look up a body by its path (a Vec of child-body indices).
    /// Empty path returns the parent body. Each element of `path`
    /// indexes into the child_bodies of the body it descends into.
    /// Returns None if any index is out of range.
    ///
    /// Matches the `body_path` shape used by `verify::VerifyError`,
    /// so tooling can take a verify error and resolve it to the
    /// actual body the error refers to.
    #[must_use]
    pub fn body_at(&self, path: &[usize]) -> Option<&KernelBody> {
        let mut current = &self.body;
        for &idx in path {
            current = current.child_bodies.get(idx)?;
        }
        Some(current)
    }

    /// True iff the descriptor has no ops at all (no parent ops AND
    /// no ops in any child body). The dispatch geometry and bindings
    /// can still be populated  -  this only asks about op content.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_ops() == 0
    }

    /// True iff the descriptor is pure  -  no side-effecting ops anywhere.
    /// Inverse of `has_side_effects`. Pure kernels can be safely
    /// cached by descriptor identity since they produce no observable
    /// output (the only "result" is whatever value-flow the consumer
    /// inspects, which is fully determined by the descriptor).
    #[must_use]
    pub fn is_pure(&self) -> bool {
        !self.has_side_effects()
    }

    /// Iterator over every `KernelOp` in the descriptor (parent body
    /// + every nested child body, depth-first pre-order). Useful for
    /// tooling that wants to walk all ops without writing the
    /// recursion themselves.
    pub fn ops_iter(&self) -> KernelOpsIter<'_> {
        KernelOpsIter {
            stack: vec![(&self.body, 0)],
        }
    }

    /// Find the first op anywhere in the descriptor whose `result`
    /// matches `id`. Per-body id space means an id may be reused
    /// across child bodies  -  this returns the FIRST match in DFS
    /// pre-order. For a given body's view, callers should iterate
    /// `body.ops` directly.
    #[must_use]
    pub fn find_op_by_id(&self, id: u32) -> Option<&KernelOp> {
        self.ops_iter().find(|op| op.result == Some(id))
    }

    /// Total threads per workgroup (the product of `dispatch.workgroup_size`).
    /// Saturates on overflow rather than wrapping. Useful for
    /// per-dispatch resource calculations (shared memory budget,
    /// register pressure, etc.).
    #[must_use]
    pub fn dispatch_total_threads(&self) -> u32 {
        let wg = self.dispatch.workgroup_size;
        wg[0].saturating_mul(wg[1]).saturating_mul(wg[2])
    }

    /// Return a clone of this descriptor with a new `id` field.
    /// Body, bindings, dispatch all unchanged. Useful for tooling
    /// that wants to fork a descriptor for ablation testing or
    /// versioning.
    #[must_use]
    pub fn with_id(&self, id: impl Into<String>) -> Self {
        let mut clone = self.clone();
        clone.id = id.into();
        clone
    }

    /// True iff the descriptor has at least one side-effecting op (memory
    /// write, atomic, sync/async op, barrier, control exit, indirect dispatch,
    /// call, or opaque extension). A pure descriptor with no side effects
    /// produces no observable output, so a caller is free to drop it entirely.
    ///
    /// "Side effect" here means observable-or-cross-thread: `AsyncLoad` writes
    /// shared memory other threads read, `AsyncWait`/`Barrier` are sync points,
    /// and `IndirectDispatch` reconfigures the grid, all are unsafe to drop
    /// even though none produces a global-buffer write.
    #[must_use]
    pub fn has_side_effects(&self) -> bool {
        fn walk(b: &KernelBody) -> bool {
            for op in &b.ops {
                use KernelOpKind::*;
                // EXHAUSTIVE ON PURPOSE (no `_` wildcard): a future KernelOpKind
                // with an observable or cross-thread effect must be classified
                // here, never silently default to "droppable".
                let effecting = match op.kind {
                    StoreGlobal
                    | StoreShared
                    | LoopCarrierInit { .. }
                    | LoopCarrierEnd { .. }
                    | Atomic { .. }
                    | AsyncLoad { .. }
                    | AsyncStore { .. }
                    | AsyncWait { .. }
                    | Barrier { .. }
                    | Trap { .. }
                    | Resume { .. }
                    | Return
                    | IndirectDispatch { .. }
                    | Call { .. }
                    | OpaqueExpr(..)
                    | OpaqueNode(..) => true,
                    // Structured control flow / Region carry no direct effect:
                    // their child bodies are walked separately below.
                    StructuredIfThen
                    | StructuredIfThenElse
                    | StructuredForLoop { .. }
                    | StructuredBlock
                    | Region { .. } => false,
                    // Pure value/builtin/arith ops and READS (loads, carrier
                    // read, buffer length): no observable or cross-thread effect.
                    Literal
                    | Copy
                    | LocalInvocationId
                    | GlobalInvocationId
                    | WorkgroupId
                    | SubgroupLocalId
                    | SubgroupSize
                    | LoopIndex { .. }
                    | LoopCarrier { .. }
                    | LoadGlobal
                    | LoadShared
                    | LoadConstant
                    | BufferLength
                    | BinOpKind(_)
                    | UnOpKind(_)
                    | Fma
                    | MatrixMma { .. }
                    | Select
                    | Cast { .. }
                    | SubgroupBallot
                    | SubgroupShuffle
                    | SubgroupBroadcast
                    | SubgroupReduce { .. } => false,
                };
                if effecting {
                    return true;
                }
            }
            b.child_bodies.iter().any(walk)
        }
        walk(&self.body)
    }
}

impl<'a> Iterator for KernelOpsIter<'a> {
    type Item = &'a KernelOp;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (body, idx) = self.stack.last_mut()?;
            if let Some(op) = body.ops.get(*idx) {
                *idx += 1;
                return Some(op);
            }
            // Body exhausted  -  push children and pop self.
            let body = *body;
            self.stack.pop();
            for child in body.child_bodies.iter().rev() {
                self.stack.push((child, 0));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::test_descriptors::build;
    use crate::descriptor::{
        BindingLayout, BindingSlot, BindingVisibility, LiteralValue, MemoryClass,
    };
    use vyre_foundation::ir::DataType;
    use vyre_foundation::memory_model::MemoryOrdering;

    fn binding(slot: u32, element: DataType, mc: MemoryClass) -> BindingSlot {
        BindingSlot {
            slot,
            element_type: element,
            element_count: None,
            memory_class: mc,
            visibility: BindingVisibility::ReadWrite,
            name: format!("b{slot}"),
        }
    }

    #[test]
    fn summary_includes_all_counts() {
        let d = build(vec![], vec![]);
        let s = d.summary();
        assert!(s.contains("k:"));
        assert!(s.contains("0 ops"));
        assert!(s.contains("1 bindings"));
        assert!(s.contains("0 child bodies"));
        assert!(s.contains("1 literals"));
        assert!(s.contains("[64, 1, 1]"));
    }

    #[test]
    fn summary_compact_terser_form() {
        let d = build(
            vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            }],
            vec![],
        );
        let s = d.summary_compact();
        assert_eq!(s, "k(1 ops, 1 bindings)");
    }

    #[test]
    fn unnamed_descriptor_uses_unnamed_label() {
        let mut d = build(vec![], vec![]);
        d.id = String::new();
        let s = d.summary();
        assert!(s.contains("<unnamed>"));
    }

    #[test]
    fn total_ops_recurses_into_child_bodies() {
        let child = KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(5)],
        };
        let parent_ops = vec![KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        }];
        let d = build(parent_ops, vec![child]);
        assert_eq!(d.body.ops.len(), 1); // shallow
        assert_eq!(d.total_ops(), 3); // 1 parent + 2 child
    }

    #[test]
    fn body_at_empty_path_returns_parent() {
        let d = build(
            vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(7),
            }],
            vec![],
        );
        let body = d.body_at(&[]).unwrap();
        assert_eq!(body.ops.len(), 1);
        assert_eq!(body.ops[0].result, Some(7));
    }

    #[test]
    fn body_at_descends_into_children() {
        let grandchild = KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(99),
            }],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        };
        let child = KernelBody {
            ops: vec![],
            child_bodies: vec![grandchild],
            literals: vec![],
        };
        let d = build(vec![], vec![child]);
        // Path [0]: first child of parent  -  empty body with one grandchild.
        let b = d.body_at(&[0]).unwrap();
        assert!(b.ops.is_empty());
        // Path [0, 0]: grandchild  -  has the Literal with result 99.
        let b = d.body_at(&[0, 0]).unwrap();
        assert_eq!(b.ops[0].result, Some(99));
    }

    #[test]
    fn body_at_out_of_range_returns_none() {
        let d = build(vec![], vec![]);
        assert!(d.body_at(&[5]).is_none());
        assert!(d.body_at(&[0, 0]).is_none());
    }

    #[test]
    fn body_count_includes_parent_plus_recursive_children() {
        let nested = KernelBody {
            ops: vec![],
            child_bodies: vec![KernelBody {
                ops: vec![],
                child_bodies: vec![],
                literals: vec![],
            }],
            literals: vec![],
        };
        let d = build(vec![], vec![nested]);
        // Parent (1) + first child (1) + grandchild (1) = 3.
        assert_eq!(d.body_count(), 3);
    }

    #[test]
    fn body_count_flat_kernel_is_one() {
        let d = build(vec![], vec![]);
        assert_eq!(d.body_count(), 1);
    }

    #[test]
    fn max_body_depth_flat_is_zero() {
        let d = build(vec![], vec![]);
        assert_eq!(d.max_body_depth(), 0);
    }

    #[test]
    fn max_body_depth_one_if_is_one() {
        let child = KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        };
        let d = build(vec![], vec![child]);
        assert_eq!(d.max_body_depth(), 1);
    }

    #[test]
    fn max_body_depth_two_levels() {
        let grandchild = KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        };
        let child = KernelBody {
            ops: vec![],
            child_bodies: vec![grandchild],
            literals: vec![],
        };
        let d = build(vec![], vec![child]);
        assert_eq!(d.max_body_depth(), 2);
    }

    #[test]
    fn total_ops_zero_for_empty_kernel() {
        let d = build(vec![], vec![]);
        assert_eq!(d.total_ops(), 0);
    }

    #[test]
    fn is_empty_true_when_no_ops() {
        let d = build(vec![], vec![]);
        assert!(d.is_empty());
    }

    #[test]
    fn is_empty_false_when_parent_has_ops() {
        let d = build(
            vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            }],
            vec![],
        );
        assert!(!d.is_empty());
        assert_eq!(d.total_ops(), 1);
    }

    #[test]
    fn is_empty_false_when_child_has_ops() {
        let child = KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            }],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(1)],
        };
        let d = build(vec![], vec![child]);
        assert!(!d.is_empty());
        assert_eq!(d.total_ops(), 1);
    }

    #[test]
    fn has_side_effects_true_with_store() {
        let d = build(
            vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 0, 0],
                    result: None,
                },
            ],
            vec![],
        );
        assert!(d.has_side_effects());
    }

    #[test]
    fn has_side_effects_false_with_only_arithmetic() {
        let d = build(
            vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                    operands: vec![0, 0],
                    result: Some(1),
                },
            ],
            vec![],
        );
        assert!(!d.has_side_effects());
    }

    #[test]
    fn has_side_effects_true_for_async_and_indirect_dispatch_ops() {
        // Regression: AsyncLoad writes shared memory other threads read,
        // AsyncWait is a sync point, and IndirectDispatch reconfigures the grid
        //: all cross-thread/dispatch effects (like the already-listed Barrier /
        // AsyncStore), so a descriptor containing one is NOT droppable. They
        // were omitted from the side-effecting set before the exhaustive-match
        // change, which would have let a "drop pure descriptor" caller drop one.
        for kind in [
            KernelOpKind::AsyncLoad { tag: "t".into() },
            KernelOpKind::AsyncWait { tag: "t".into() },
            KernelOpKind::IndirectDispatch { count_offset: 0 },
        ] {
            let d = build(
                vec![KernelOp {
                    kind: kind.clone(),
                    operands: vec![0],
                    result: None,
                }],
                vec![],
            );
            assert!(
                d.has_side_effects(),
                "{kind:?} is a cross-thread/dispatch effect and must not be droppable"
            );
            assert!(!d.is_pure(), "{kind:?} must not be classified pure");
        }
    }

    #[test]
    fn ops_iter_visits_parent_then_children_in_order() {
        let child0 = KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(10),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(11),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(1)],
        };
        let child1 = KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(20),
            }],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(2)],
        };
        let d = build(
            vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            vec![child0, child1],
        );
        let visited: Vec<u32> = d.ops_iter().map(|o| o.result.unwrap()).collect();
        // Parent ops (0, 1) first, then child0 (10, 11), then child1 (20).
        assert_eq!(visited, vec![0, 1, 10, 11, 20]);
    }

    #[test]
    fn ops_iter_count_matches_total_ops() {
        let child = KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        };
        let d = build(
            vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            }],
            vec![child],
        );
        assert_eq!(d.ops_iter().count(), d.total_ops());
    }

    #[test]
    fn dispatch_total_threads_multiplies_dims() {
        let d = build(vec![], vec![]);
        assert_eq!(d.dispatch_total_threads(), 64); // build() uses Dispatch::new(64, 1, 1)

        let mut d2 = build(vec![], vec![]);
        d2.dispatch = Dispatch::new(8, 8, 4);
        assert_eq!(d2.dispatch_total_threads(), 256);
    }

    #[test]
    fn with_id_preserves_everything_else() {
        let d = build(
            vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            }],
            vec![],
        );
        let renamed = d.with_id("renamed");
        assert_eq!(renamed.id, "renamed");
        assert_eq!(d.id, "k"); // original unchanged
        assert_eq!(renamed.body.ops.len(), d.body.ops.len());
        assert_eq!(renamed.bindings, d.bindings);
        assert_eq!(renamed.dispatch, d.dispatch);
    }

    #[test]
    fn dispatch_total_threads_saturates_on_overflow() {
        let mut d = build(vec![], vec![]);
        d.dispatch = Dispatch::new(u32::MAX, u32::MAX, u32::MAX);
        // Saturating multiplication means we get u32::MAX rather than wrap.
        assert_eq!(d.dispatch_total_threads(), u32::MAX);
    }

    #[test]
    fn find_op_by_id_in_parent() {
        let d = build(
            vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(7),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(42),
                },
            ],
            vec![],
        );
        let op = d.find_op_by_id(42).expect("Fix: found");
        assert_eq!(op.result, Some(42));
        assert!(d.find_op_by_id(99).is_none());
    }

    #[test]
    fn find_op_by_id_finds_in_child() {
        let child = KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(100),
            }],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        };
        let d = build(vec![], vec![child]);
        assert!(d.find_op_by_id(100).is_some());
    }

    #[test]
    fn ops_iter_empty_descriptor_yields_none() {
        let d = build(vec![], vec![]);
        assert!(d.ops_iter().next().is_none());
    }

    #[test]
    fn is_pure_inverse_of_has_side_effects() {
        let pure_kernel = build(
            vec![KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            }],
            vec![],
        );
        assert!(pure_kernel.is_pure());
        assert!(!pure_kernel.has_side_effects());

        let impure = build(
            vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 0, 0],
                    result: None,
                },
            ],
            vec![],
        );
        assert!(!impure.is_pure());
        assert!(impure.has_side_effects());
    }

    #[test]
    fn empty_descriptor_round_trips_serde_byte_stable() {
        let k = KernelDescriptor {
            id: "test".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(1, 1, 1),
            body: KernelBody {
                ops: vec![],
                child_bodies: vec![],
                literals: vec![],
            },
        };
        let json1 = serde_json::to_string(&k).unwrap();
        let parsed: KernelDescriptor = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2);
        assert_eq!(k, parsed);
    }

    #[test]
    fn one_store_kernel_round_trips_byte_stable() {
        let k = KernelDescriptor {
            id: "store_one".into(),
            bindings: BindingLayout {
                slots: vec![binding(0, DataType::U32, MemoryClass::Global)],
            },
            dispatch: Dispatch::new(1, 1, 1),
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
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
            },
        };
        let json1 = serde_json::to_string(&k).unwrap();
        let parsed: KernelDescriptor = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2);
    }

    #[test]
    fn nested_if_then_body_round_trips() {
        let inner = KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::Barrier {
                    ordering: MemoryOrdering::SeqCst,
                },
                operands: vec![],
                result: None,
            }],
            child_bodies: vec![],
            literals: vec![],
        };
        let outer = KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::StructuredIfThen,
                    operands: vec![0, 0],
                    result: None,
                },
            ],
            child_bodies: vec![inner],
            literals: vec![LiteralValue::Bool(true)],
        };
        let k = KernelDescriptor {
            id: "if_then".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(1, 1, 1),
            body: outer,
        };
        let json1 = serde_json::to_string(&k).unwrap();
        let parsed: KernelDescriptor = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2);
    }

    #[test]
    fn for_loop_with_var_name_round_trips() {
        let body = KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        };
        let outer = KernelBody {
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
                    kind: KernelOpKind::StructuredForLoop {
                        loop_var: "i".into(),
                    },
                    operands: vec![0, 1, 0],
                    result: None,
                },
            ],
            child_bodies: vec![body],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(64)],
        };
        let k = KernelDescriptor {
            id: "for_i".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: outer,
        };
        let json = serde_json::to_string(&k).unwrap();
        let parsed: KernelDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(k, parsed);
    }

    #[test]
    fn async_load_wait_carry_tag() {
        let body = KernelBody {
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
                    kind: KernelOpKind::AsyncLoad {
                        tag: "chunk-0".into(),
                    },
                    operands: vec![0, 1, 0, 1],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::AsyncWait {
                        tag: "chunk-0".into(),
                    },
                    operands: vec![],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(16)],
        };
        let k = KernelDescriptor {
            id: "async".into(),
            bindings: BindingLayout {
                slots: vec![
                    binding(0, DataType::U32, MemoryClass::Global),
                    binding(1, DataType::U32, MemoryClass::Shared),
                ],
            },
            dispatch: Dispatch::new(64, 1, 1),
            body,
        };
        let json = serde_json::to_string(&k).unwrap();
        let parsed: KernelDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(k, parsed);
    }

    #[test]
    fn dispatch_constructor_preserves_axes() {
        let d = Dispatch::new(64, 4, 2);
        assert_eq!(d.workgroup_size, [64, 4, 2]);
    }
}
