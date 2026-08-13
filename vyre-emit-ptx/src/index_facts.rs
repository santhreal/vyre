//! Body-local index arithmetic facts for PTX emission.
//!
//! `verified-intentional`: this index survives alongside
//! `vyre_lower::analyses::def_use` rather than consuming it. The target
//! constraint is what PTX asks of an index. `ld.global.v2/v4` is only legal
//! when the base address is naturally aligned to the vector width, and
//! `ldmatrix` addresses a fragment row by its offset within a 32-lane group,
//! so the emitter needs `index mod k` for a symbolic index expression, not a
//! list of use sites. That is what [`IndexFacts::index_modulo`] and
//! `normalized_index` compute, and it has no meaning off the PTX register
//! model. `def_use` answers a different question (which operand position of
//! which op in which body references an id) over the whole descriptor tree,
//! and its per-position `UseSite` allocation cannot be narrowed to the single
//! body an emitter has in hand.
//!
//! What is NOT re-derived here: which operand positions name an SSA result,
//! and which ids an op defines. Both come from `vyre-lower`, their one owner.
//! A private copy of either table drifted out of agreement once already.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::BinOp;
use vyre_lower::operand_semantics::operand_is_result_reference;
use vyre_lower::{KernelBody, KernelOp, KernelOpKind, LiteralValue};

pub(crate) struct IndexFacts {
    producer: FxHashMap<u32, usize>,
    consumer_indices: FxHashMap<u32, Vec<usize>>,
    lit_u32: FxHashMap<u32, u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NormalizedIndex {
    root: Option<u32>,
    offset: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AffineModulo {
    root: Option<u32>,
    coeff: u32,
    offset: u32,
}

const AFFINE_ROOT_GLOBAL_INVOCATION: u32 = u32::MAX - 2;
const AFFINE_ROOT_LOCAL_INVOCATION: u32 = u32::MAX - 5;
const AFFINE_ROOT_WORKGROUP: u32 = u32::MAX - 8;

impl IndexFacts {
    pub(crate) fn new(body: &KernelBody) -> Self {
        let mut producer = FxHashMap::with_capacity_and_hasher(body.ops.len(), Default::default());
        let mut consumer_indices: FxHashMap<u32, Vec<usize>> =
            FxHashMap::with_capacity_and_hasher(body.ops.len(), Default::default());
        let mut lit_u32 = FxHashMap::with_capacity_and_hasher(body.ops.len(), Default::default());
        for (idx, op) in body.ops.iter().enumerate() {
            for result_id in op.result_ids() {
                producer.insert(result_id, idx);
            }
            let Some(result_id) = op.result else {
                continue;
            };
            if !matches!(op.kind, KernelOpKind::Literal) {
                continue;
            }
            let Some(&pool_idx) = op.operands.first() else {
                continue;
            };
            let Some(literal) = body.literals.get(pool_idx as usize) else {
                continue;
            };
            let value = match literal {
                LiteralValue::U32(value) => Some(*value),
                LiteralValue::I32(value) => Some(*value as u32),
                _ => None,
            };
            if let Some(value) = value {
                lit_u32.insert(result_id, value);
            }
        }
        for (op_idx, op) in body.ops.iter().enumerate() {
            for (pos, &operand) in op.operands.iter().enumerate() {
                if !operand_is_result_reference(&op.kind, pos) || !producer.contains_key(&operand) {
                    continue;
                }
                consumer_indices.entry(operand).or_default().push(op_idx);
            }
        }
        Self {
            producer,
            consumer_indices,
            lit_u32,
        }
    }

    pub(crate) fn is_index_plus_one(
        &self,
        body: &KernelBody,
        candidate_id: u32,
        prev_id: u32,
    ) -> bool {
        if let (Some(candidate), Some(prev)) =
            (self.lit_u32.get(&candidate_id), self.lit_u32.get(&prev_id))
        {
            return prev.checked_add(1) == Some(*candidate);
        }
        if let (Some(candidate), Some(prev)) = (
            self.normalized_index(body, candidate_id),
            self.normalized_index(body, prev_id),
        ) {
            if candidate.root == prev.root && prev.offset.checked_add(1) == Some(candidate.offset) {
                return true;
            }
        }
        let Some(&op_idx) = self.producer.get(&candidate_id) else {
            return false;
        };
        let op = &body.ops[op_idx];
        let KernelOpKind::BinOpKind(BinOp::Add) = op.kind else {
            return false;
        };
        if op.operands.len() != 2 {
            return false;
        }
        let lhs = op.operands[0];
        let rhs = op.operands[1];
        let is_one = |id: u32| self.lit_u32.get(&id) == Some(&1);
        (lhs == prev_id && is_one(rhs)) || (rhs == prev_id && is_one(lhs))
    }

    #[cfg(test)]
    pub(crate) fn index_is_multiple_of(
        &self,
        body: &KernelBody,
        result_id: u32,
        modulus: u32,
    ) -> bool {
        if modulus <= 1 {
            return true;
        }
        self.index_mod(body, result_id, modulus, 0) == Some(0)
    }

    pub(crate) fn index_modulo(
        &self,
        body: &KernelBody,
        result_id: u32,
        modulus: u32,
    ) -> Option<u32> {
        self.index_mod(body, result_id, modulus, 0)
    }

    fn index_mod(&self, body: &KernelBody, result_id: u32, modulus: u32, depth: u8) -> Option<u32> {
        if modulus == 0 || depth > 8 {
            return None;
        }
        let affine = self.affine_mod(body, result_id, modulus, depth)?;
        (affine.coeff == 0).then_some(affine.offset % modulus)
    }

    fn affine_mod(
        &self,
        body: &KernelBody,
        result_id: u32,
        modulus: u32,
        depth: u8,
    ) -> Option<AffineModulo> {
        if modulus == 0 || depth > 8 {
            return None;
        }
        if let Some(value) = self.lit_u32.get(&result_id).copied() {
            return Some(AffineModulo {
                root: None,
                coeff: 0,
                offset: value % modulus,
            });
        }
        let Some(&op_idx) = self.producer.get(&result_id) else {
            return Some(AffineModulo {
                root: Some(result_id),
                coeff: 1 % modulus,
                offset: 0,
            });
        };
        let op = &body.ops[op_idx];
        if op.operands.len() != 2 {
            return Some(AffineModulo {
                root: Some(symbolic_affine_root(op, result_id)),
                coeff: 1 % modulus,
                offset: 0,
            });
        }
        let lhs = op.operands[0];
        let rhs = op.operands[1];
        match op.kind {
            KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd) => {
                let lhs_mod = self.affine_mod(body, lhs, modulus, depth + 1)?;
                let rhs_mod = self.affine_mod(body, rhs, modulus, depth + 1)?;
                combine_affine_add(lhs_mod, rhs_mod, modulus)
            }
            KernelOpKind::BinOpKind(BinOp::Mul) => {
                if let Some(value) = self.lit_u32.get(&lhs).copied() {
                    let rhs_mod = self.affine_mod(body, rhs, modulus, depth + 1)?;
                    return Some(scale_affine(rhs_mod, value, modulus));
                }
                if let Some(value) = self.lit_u32.get(&rhs).copied() {
                    let lhs_mod = self.affine_mod(body, lhs, modulus, depth + 1)?;
                    return Some(scale_affine(lhs_mod, value, modulus));
                }
                Some(AffineModulo {
                    root: Some(symbolic_affine_root(op, result_id)),
                    coeff: 1 % modulus,
                    offset: 0,
                })
            }
            KernelOpKind::BinOpKind(BinOp::Shl) => {
                let shift = self.lit_u32.get(&rhs).copied()? & 31;
                let factor = 1u32 << shift;
                let lhs_mod = self.affine_mod(body, lhs, modulus, depth + 1)?;
                Some(scale_affine(lhs_mod, factor, modulus))
            }
            _ => Some(AffineModulo {
                root: Some(symbolic_affine_root(op, result_id)),
                coeff: 1 % modulus,
                offset: 0,
            }),
        }
    }

    fn normalized_index(&self, body: &KernelBody, result_id: u32) -> Option<NormalizedIndex> {
        self.normalized_index_inner(body, result_id, 0)
    }

    fn normalized_index_inner(
        &self,
        body: &KernelBody,
        result_id: u32,
        depth: u8,
    ) -> Option<NormalizedIndex> {
        if depth > 8 {
            return Some(NormalizedIndex {
                root: Some(result_id),
                offset: 0,
            });
        }
        if let Some(value) = self.lit_u32.get(&result_id).copied() {
            return Some(NormalizedIndex {
                root: None,
                offset: value,
            });
        }
        let Some(&op_idx) = self.producer.get(&result_id) else {
            return Some(NormalizedIndex {
                root: Some(result_id),
                offset: 0,
            });
        };
        let op = &body.ops[op_idx];
        if !matches!(
            op.kind,
            KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd)
        ) || op.operands.len() != 2
        {
            return Some(NormalizedIndex {
                root: Some(result_id),
                offset: 0,
            });
        }

        let lhs = op.operands[0];
        let rhs = op.operands[1];
        if let Some(delta) = self.lit_u32.get(&rhs).copied() {
            let base = self.normalized_index_inner(body, lhs, depth + 1)?;
            return Some(NormalizedIndex {
                root: base.root,
                offset: base.offset.checked_add(delta)?,
            });
        }
        if let Some(delta) = self.lit_u32.get(&lhs).copied() {
            let base = self.normalized_index_inner(body, rhs, depth + 1)?;
            return Some(NormalizedIndex {
                root: base.root,
                offset: base.offset.checked_add(delta)?,
            });
        }

        Some(NormalizedIndex {
            root: Some(result_id),
            offset: 0,
        })
    }

    pub(crate) fn producer_idx(&self, result_id: u32) -> Option<usize> {
        self.producer.get(&result_id).copied()
    }

    pub(crate) fn result_use_count(&self, result_id: u32) -> usize {
        self.consumer_indices
            .get(&result_id)
            .map(Vec::len)
            .unwrap_or(0)
    }

    pub(crate) fn consumer_indices(&self, result_id: u32) -> Option<&[usize]> {
        self.consumer_indices.get(&result_id).map(Vec::as_slice)
    }

    #[cfg(test)]
    pub(crate) fn single_consumer_idx(&self, result_id: u32) -> Option<usize> {
        match self.consumer_indices(result_id)? {
            [index] => Some(*index),
            _ => None,
        }
    }
}

fn combine_affine_add(lhs: AffineModulo, rhs: AffineModulo, modulus: u32) -> Option<AffineModulo> {
    let root = match (lhs.root, rhs.root) {
        (None, root) | (root, None) => root,
        (Some(left), Some(right)) if left == right => Some(left),
        _ => return None,
    };
    Some(AffineModulo {
        root,
        coeff: ((u64::from(lhs.coeff) + u64::from(rhs.coeff)) % u64::from(modulus)) as u32,
        offset: ((u64::from(lhs.offset) + u64::from(rhs.offset)) % u64::from(modulus)) as u32,
    })
}

fn scale_affine(value: AffineModulo, factor: u32, modulus: u32) -> AffineModulo {
    AffineModulo {
        root: value.root,
        coeff: ((u64::from(value.coeff) * u64::from(factor % modulus)) % u64::from(modulus)) as u32,
        offset: ((u64::from(value.offset) * u64::from(factor % modulus)) % u64::from(modulus))
            as u32,
    }
}

fn symbolic_affine_root(op: &KernelOp, result_id: u32) -> u32 {
    let axis = op.operands.first().copied().unwrap_or(0).min(2);
    match op.kind {
        KernelOpKind::GlobalInvocationId => AFFINE_ROOT_GLOBAL_INVOCATION + axis,
        KernelOpKind::LocalInvocationId => AFFINE_ROOT_LOCAL_INVOCATION + axis,
        KernelOpKind::WorkgroupId => AFFINE_ROOT_WORKGROUP + axis,
        _ => result_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_lower::descriptor_builder::{body, effect, lit, op};
    use vyre_lower::{
        KernelBody, KernelOp, LiteralValue, MatrixMmaElement, MatrixMmaLayout, MatrixMmaShape,
    };

    fn body_with_add(
        operands: Vec<u32>,
        result: Option<u32>,
        literals: Vec<LiteralValue>,
    ) -> KernelBody {
        body()
            .ops([
                lit(0, 1),
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands,
                    result,
                },
            ])
            .literals(literals).build()
    }

    #[test]
    fn detects_unit_stride_add_in_either_operand_order() {
        let body = body_with_add(vec![7, 1], Some(9), vec![LiteralValue::U32(1)]);
        let facts = IndexFacts::new(&body);
        assert!(facts.is_index_plus_one(&body, 9, 7));

        let body = body_with_add(vec![1, 7], Some(9), vec![LiteralValue::I32(1)]);
        let facts = IndexFacts::new(&body);
        assert!(facts.is_index_plus_one(&body, 9, 7));
    }

    #[test]
    fn detects_adjacent_folded_literal_indices() {
        let body = body()
            .ops([lit(0, 10), lit(1, 11)])
            .literals([LiteralValue::U32(8), LiteralValue::U32(9)]).build();
        let facts = IndexFacts::new(&body);
        assert!(facts.is_index_plus_one(&body, 11, 10));
        assert!(!facts.is_index_plus_one(&body, 10, 11));
    }

    #[test]
    fn detects_adjacent_dynamic_indices_after_affine_reassociation() {
        let body = body()
            .ops([
                lit(0, 1),
                lit(1, 2),
                op(KernelOpKind::BinOpKind(BinOp::Add), [7, 1], 9),
                op(KernelOpKind::BinOpKind(BinOp::Add), [7, 2], 10),
            ])
            .literals([LiteralValue::U32(1), LiteralValue::U32(2)]).build();
        let facts = IndexFacts::new(&body);
        assert!(facts.is_index_plus_one(&body, 10, 9));
        assert!(!facts.is_index_plus_one(&body, 9, 10));
    }

    #[test]
    fn detects_adjacent_dynamic_indices_after_chained_reassociation() {
        let body = body()
            .ops([
                lit(0, 1),
                op(KernelOpKind::BinOpKind(BinOp::Add), [7, 1], 9),
                op(KernelOpKind::BinOpKind(BinOp::WrappingAdd), [9, 1], 10),
            ])
            .literal(LiteralValue::U32(1)).build();
        let facts = IndexFacts::new(&body);
        assert!(facts.is_index_plus_one(&body, 10, 9));
    }

    #[test]
    fn rejects_missing_producer_non_add_and_non_one_literals() {
        let body = body_with_add(vec![7, 1], Some(9), vec![LiteralValue::U32(2)]);
        let facts = IndexFacts::new(&body);
        assert!(!facts.is_index_plus_one(&body, 9, 7));
        assert!(!facts.is_index_plus_one(&body, 99, 7));
    }

    #[test]
    fn literal_pool_indices_do_not_count_as_result_consumers() {
        let body = body().op(lit(0, 0)).literal(LiteralValue::U32(9)).build();
        let facts = IndexFacts::new(&body);
        assert_eq!(facts.result_use_count(0), 0);
        assert_eq!(facts.single_consumer_idx(0), None);
    }

    #[test]
    fn store_binding_slots_do_not_count_as_result_consumers() {
        let body = body()
            .ops([
                lit(0, 0),
                lit(1, 1),
                lit(2, 2),
                effect(KernelOpKind::StoreGlobal, [0, 1, 2]),
            ])
            .literals([LiteralValue::U32(0), LiteralValue::U32(1), LiteralValue::U32(2)]).build();
        let facts = IndexFacts::new(&body);
        assert_eq!(facts.result_use_count(0), 0);
        assert_eq!(facts.result_use_count(1), 1);
        assert_eq!(facts.result_use_count(2), 1);
    }

    #[test]
    fn matrix_mma_consecutive_fragment_results_are_producers() {
        let body = body()
            .ops([
                op(KernelOpKind::MatrixMma {
                        shape: MatrixMmaShape::M16N8K16,
                        a_layout: MatrixMmaLayout::RowMajor,
                        b_layout: MatrixMmaLayout::ColMajor,
                        a_type: MatrixMmaElement::F16,
                        b_type: MatrixMmaElement::F16,
                        accum_type: MatrixMmaElement::F32,
                    }, [0; 10], 10),
                op(KernelOpKind::Copy, [12], 20),
            ]).build();
        let facts = IndexFacts::new(&body);
        assert_eq!(facts.producer_idx(12), Some(0));
        assert_eq!(facts.result_use_count(12), 1);
        assert_eq!(facts.single_consumer_idx(12), Some(1));
    }

    #[test]
    fn index_multiple_detects_dynamic_constant_stride_alignment() {
        let body = body()
            .ops([
                lit(0, 1),
                lit(1, 2),
                lit(2, 3),
                op(KernelOpKind::BinOpKind(BinOp::Mul), [99, 1], 10),
                op(KernelOpKind::BinOpKind(BinOp::Mul), [99, 2], 11),
                op(KernelOpKind::BinOpKind(BinOp::Add), [10, 3], 12),
            ])
            .literals([
                LiteralValue::U32(4),
                LiteralValue::U32(10),
                LiteralValue::U32(1),
            ]).build();
        let facts = IndexFacts::new(&body);
        assert!(facts.index_is_multiple_of(&body, 10, 4));
        assert!(facts.index_is_multiple_of(&body, 11, 2));
        assert!(!facts.index_is_multiple_of(&body, 11, 4));
        assert!(!facts.index_is_multiple_of(&body, 12, 2));
    }

    #[test]
    fn index_multiple_detects_strength_reduced_shift_alignment() {
        let body = body()
            .ops([
                lit(0, 1),
                lit(1, 2),
                lit(2, 3),
                op(KernelOpKind::BinOpKind(BinOp::Shl), [99, 1], 10),
                op(KernelOpKind::BinOpKind(BinOp::Shl), [99, 2], 11),
                op(KernelOpKind::BinOpKind(BinOp::Add), [10, 3], 12),
            ])
            .literals([LiteralValue::U32(2), LiteralValue::U32(1), LiteralValue::U32(1)]).build();
        let facts = IndexFacts::new(&body);
        assert!(facts.index_is_multiple_of(&body, 10, 4));
        assert!(facts.index_is_multiple_of(&body, 11, 2));
        assert!(!facts.index_is_multiple_of(&body, 11, 4));
        assert!(!facts.index_is_multiple_of(&body, 12, 2));
    }

    /// Resolution guard for the operand-namespace clone family.
    ///
    /// This crate used to carry its own operand classifier. Step 0 proved it
    /// had drifted from `vyre_lower::operand_semantics`: on operand counts the
    /// op contracts forbid, the local copy dropped trailing operands from the
    /// use map (`take(2)` on `StructuredForLoop`, `skip(1)` on the loads)
    /// while the owner kept them. The owner won, because under-counting uses
    /// is the unsafe direction: a value that is still read looks dead to a
    /// hoisting or elimination decision.
    ///
    /// A `StructuredForLoop` whose operands run past `[lo, hi, body]` is
    /// out of contract, and every operand past the body index must still be
    /// counted as a use.
    #[test]
    fn out_of_contract_loop_operands_still_count_as_uses() {
        let body = body()
            .ops([
                lit(0, 1),
                lit(1, 2),
                lit(0, 3),
                effect(KernelOpKind::StructuredForLoop {
                        loop_var: "i".into(),
                    }, [1, 2, 0, 3]),
            ])
            .child(body())
            .literals([LiteralValue::U32(0), LiteralValue::U32(4)]).build();
        let facts = IndexFacts::new(&body);
        assert_eq!(facts.result_use_count(1), 1, "lo bound is a use");
        assert_eq!(facts.result_use_count(2), 1, "hi bound is a use");
        assert_eq!(
            facts.result_use_count(3),
            1,
            "an operand past the body index is still an SSA reference"
        );
    }

    /// The MMA accumulator tuple is `vyre-lower`'s rule, not this crate's.
    /// Every fragment id must resolve to the producing op without this file
    /// restating how wide the tuple is.
    #[test]
    fn mma_fragment_ids_resolve_to_the_producing_op() {
        use vyre_lower::{MatrixMmaElement, MatrixMmaLayout, MatrixMmaShape};
        let mma = op(KernelOpKind::MatrixMma {
                shape: MatrixMmaShape::M16N8K16,
                a_layout: MatrixMmaLayout::RowMajor,
                b_layout: MatrixMmaLayout::ColMajor,
                a_type: MatrixMmaElement::F16,
                b_type: MatrixMmaElement::F16,
                accum_type: MatrixMmaElement::F32,
            }, [], 10);
        let expected: Vec<u32> = mma.result_ids().collect();
        assert_eq!(expected, vec![10, 11, 12, 13]);
        let body = body().op(mma).build();
        let facts = IndexFacts::new(&body);
        for id in expected {
            assert_eq!(facts.producer_idx(id), Some(0), "fragment id {id}");
        }
        assert_eq!(facts.producer_idx(14), None, "tuple must not run past 4");
    }
}
