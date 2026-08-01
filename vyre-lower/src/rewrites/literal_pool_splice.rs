//! The one sound way to move ops between `KernelBody` literal-pool namespaces.
//!
//! A `Literal` op's operand 0 is an index into the literal pool of the body
//! that **contains** it, not a globally meaningful number. Any rewrite that
//! relocates ops out of one body and into another (inlining a collapsed
//! branch arm, unrolling a loop body, hoisting a loop-invariant op, fusing
//! two loop bodies) therefore changes what that index means. Relocating the
//! op without re-pointing the index at an equivalent entry in the
//! destination pool leaves the descriptor either
//!
//! - **invalid**, when the index exceeds the destination pool  -  `verify`
//!   reports `LiteralPoolOutOfRange` and the debug assertion in
//!   [`crate::rewrites`] fires; or
//! - **silently wrong**, when the index happens to be in range but names a
//!   different value. This is the worse outcome: the descriptor verifies and
//!   every backend emits a kernel that computes with the wrong constant.
//!
//! This bug was found and fixed independently in `loop_unroll` and `licm`,
//! and missed in `branch_collapse`, because relocation and pool maintenance
//! were separate steps that a caller could do one of. This type makes them
//! inseparable: the only way to get a relocated op out of it is to have its
//! pool operand rewritten. A future fifth inliner cannot get this wrong by
//! omission, only by not using this type at all.
//!
//! Scope note: `Literal` operand 0 is the sole literal-pool operand in the
//! whole op surface (`crate::verify::classify_operand` is the source of
//! truth, and it classifies exactly that one position as
//! `OperandClass::LiteralPoolIdx`). Result-id renumbering, child-body index
//! rebasing, and `LoopIndex` substitution are orthogonal concerns and stay
//! with the caller that knows its own relocation scheme.

use rustc_hash::FxHashMap;

use crate::{KernelOp, KernelOpKind, LiteralValue};

/// An in-progress relocation of ops from `source`'s pool into `dest`'s pool.
///
/// Entries are appended to `dest` lazily, on first reference, and
/// de-duplicated per splice: two relocated ops that read the same source
/// slot share one destination slot.
pub(crate) struct LiteralPoolSplice<'src, 'dest> {
    source: &'src [LiteralValue],
    dest: &'dest mut Vec<LiteralValue>,
    /// source pool index → destination pool index, for slots already copied.
    mapped: FxHashMap<u32, u32>,
}

impl<'src, 'dest> LiteralPoolSplice<'src, 'dest> {
    /// Begin relocating ops that currently index into `source`, into a body
    /// whose pool is `dest`.
    pub(crate) fn new(source: &'src [LiteralValue], dest: &'dest mut Vec<LiteralValue>) -> Self {
        Self {
            source,
            dest,
            mapped: FxHashMap::default(),
        }
    }

    /// Relocate one op, re-pointing its literal-pool operand if it has one.
    ///
    /// An op whose pool index is out of range in `source` is returned
    /// untouched. That op was already invalid in the source body, and
    /// fabricating a destination slot for it would convert a defect `verify`
    /// reports into a silently wrong constant.
    #[must_use]
    pub(crate) fn relocate(&mut self, mut op: KernelOp) -> KernelOp {
        if !matches!(op.kind, KernelOpKind::Literal) {
            return op;
        }
        let Some(&source_idx) = op.operands.first() else {
            return op;
        };
        let Some(value) = self.source.get(source_idx as usize) else {
            return op;
        };
        let dest_idx = match self.mapped.get(&source_idx) {
            Some(&idx) => idx,
            None => {
                let idx = self.dest.len() as u32;
                self.dest.push(value.clone());
                self.mapped.insert(source_idx, idx);
                idx
            }
        };
        op.operands[0] = dest_idx;
        op
    }

    /// Relocate a sequence of ops, preserving order.
    pub(crate) fn relocate_all<I>(&mut self, ops: I) -> Vec<KernelOp>
    where
        I: IntoIterator<Item = KernelOp>,
    {
        let iter = ops.into_iter();
        let mut out = Vec::with_capacity(iter.size_hint().0);
        for op in iter {
            out.push(self.relocate(op));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(pool_idx: u32, result: u32) -> KernelOp {
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![pool_idx],
            result: Some(result),
        }
    }

    /// Locks out the `branch_collapse` defect directly: relocating a
    /// `Literal` op whose source index exceeds the destination pool must
    /// re-point the operand at a freshly appended destination slot holding
    /// the SAME value, never leave the out-of-range index in place.
    #[test]
    fn relocated_literal_points_at_an_equal_value_in_the_destination_pool() {
        let source = vec![
            LiteralValue::U32(10),
            LiteralValue::U32(11),
            LiteralValue::U32(12),
            LiteralValue::U32(13),
            LiteralValue::U32(14),
        ];
        let mut dest = vec![LiteralValue::U32(0), LiteralValue::U32(1)];

        let relocated = {
            let mut splice = LiteralPoolSplice::new(&source, &mut dest);
            splice.relocate_all(vec![lit(2, 100), lit(3, 101), lit(4, 102)])
        };

        assert_eq!(relocated[0].operands, vec![2]);
        assert_eq!(relocated[1].operands, vec![3]);
        assert_eq!(relocated[2].operands, vec![4]);
        assert_eq!(
            dest,
            vec![
                LiteralValue::U32(0),
                LiteralValue::U32(1),
                LiteralValue::U32(12),
                LiteralValue::U32(13),
                LiteralValue::U32(14),
            ]
        );
        for op in &relocated {
            let idx = op.operands[0] as usize;
            assert!(idx < dest.len(), "relocated index {idx} must be in range");
        }
    }

    /// A relocated index that is coincidentally in range in the destination
    /// pool must still be re-pointed. This is the silent-miscompile case: an
    /// unremapped index 1 would resolve to U32(99) instead of U32(7).
    #[test]
    fn in_range_source_index_naming_a_different_value_is_still_remapped() {
        let source = vec![LiteralValue::U32(5), LiteralValue::U32(7)];
        let mut dest = vec![LiteralValue::U32(98), LiteralValue::U32(99)];

        let relocated = {
            let mut splice = LiteralPoolSplice::new(&source, &mut dest);
            splice.relocate_all(vec![lit(1, 100)])
        };

        assert_eq!(relocated[0].operands, vec![2]);
        assert_eq!(dest[2], LiteralValue::U32(7));
    }

    /// Two relocated ops reading one source slot must share one destination
    /// slot, so inlining does not grow the pool by one entry per reference.
    #[test]
    fn repeated_source_slot_is_deduplicated_into_one_destination_slot() {
        let source = vec![LiteralValue::U32(42)];
        let mut dest = Vec::new();

        let relocated = {
            let mut splice = LiteralPoolSplice::new(&source, &mut dest);
            splice.relocate_all(vec![lit(0, 1), lit(0, 2), lit(0, 3)])
        };

        assert_eq!(relocated[0].operands, vec![0]);
        assert_eq!(relocated[1].operands, vec![0]);
        assert_eq!(relocated[2].operands, vec![0]);
        assert_eq!(dest, vec![LiteralValue::U32(42)]);
    }

    /// An op whose source index is already out of range must pass through
    /// untouched, so the pre-existing defect stays visible to `verify`
    /// instead of being laundered into a valid-looking wrong constant.
    #[test]
    fn out_of_range_source_index_passes_through_without_fabricating_a_slot() {
        let source = vec![LiteralValue::U32(1)];
        let mut dest = vec![LiteralValue::U32(9)];

        let relocated = {
            let mut splice = LiteralPoolSplice::new(&source, &mut dest);
            splice.relocate_all(vec![lit(7, 100)])
        };

        assert_eq!(relocated[0].operands, vec![7]);
        assert_eq!(dest, vec![LiteralValue::U32(9)]);
    }

    /// Non-`Literal` ops carry no pool operand and must be returned
    /// byte-identical, including ops whose operand 0 is a binding slot that
    /// would look like a pool index to a careless remapper.
    #[test]
    fn non_literal_ops_are_untouched() {
        let source = vec![LiteralValue::U32(1), LiteralValue::U32(2)];
        let mut dest = Vec::new();
        let store = KernelOp {
            kind: KernelOpKind::StoreGlobal,
            operands: vec![1, 40, 41],
            result: None,
        };

        let relocated = {
            let mut splice = LiteralPoolSplice::new(&source, &mut dest);
            splice.relocate_all(vec![store.clone()])
        };

        assert_eq!(relocated[0], store);
        assert!(dest.is_empty());
    }
}
