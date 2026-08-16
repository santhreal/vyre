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

// Inline: covers the crate-private `descriptor::kernel` module, which no integration test can reach.
#[cfg(test)]
mod tests;
