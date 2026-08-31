//! The recursive descriptor walk.
//!
//! Owns the traversal that visits every body in lexical order carrying the
//! results visible from enclosing scopes, and the per-op checks it runs at
//! each stop. It owns no error vocabulary and no operand classification;
//! both come from siblings.

use rustc_hash::FxHashSet;

use super::{VerifyError, VerifyErrorKind};
use crate::operand_class::{classify_operand, OperandClass};
use crate::{KernelBody, KernelOpKind};

/// Check invariant 6: result ids are unique across the WHOLE descriptor,
/// not merely within one body.
///
/// `verify_body` collects `produced` fresh per body, so it only ever sees
/// duplicates that sit side by side in one op list. Sibling and
/// ancestor/descendant reuse slipped through, which is exactly what
/// `loop_unroll` used to emit when it reseeded its free-id counter at
/// every recursion level.
pub(super) fn verify_result_ids_unique_descriptor_wide(
    body: &KernelBody,
    errors: &mut Vec<VerifyError>,
) {
    use rustc_hash::FxHashMap;

    let mut owners: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    fn walk(
        body: &KernelBody,
        path: &mut Vec<usize>,
        owners: &mut FxHashMap<u32, Vec<usize>>,
        errors: &mut Vec<VerifyError>,
    ) {
        for (op_index, op) in body.ops.iter().enumerate() {
            for result in op.result_ids() {
                match owners.get(&result) {
                    // A duplicate inside one body is already reported as
                    // `DuplicateResultId`; do not double-report it here.
                    Some(first) if first == path => {}
                    Some(first) => errors.push(VerifyError {
                        body_path: path.clone(),
                        op_index,
                        kind: VerifyErrorKind::ResultIdReusedAcrossBodies {
                            result,
                            first_body_path: first.clone(),
                        },
                    }),
                    None => {
                        owners.insert(result, path.clone());
                    }
                }
            }
        }
        for (child_index, child) in body.child_bodies.iter().enumerate() {
            path.push(child_index);
            walk(child, path, owners, errors);
            path.pop();
        }
    }
    walk(body, &mut Vec::new(), &mut owners, errors);
}

pub(super) fn verify_body(
    body: &KernelBody,
    path: &mut Vec<usize>,
    inherited_results: &FxHashSet<u32>,
    declared_slots: &FxHashSet<u32>,
    errors: &mut Vec<VerifyError>,
) {
    use rustc_hash::FxHashSet;

    // 1. Collect produced result-ids, flagging duplicates.
    let mut produced: FxHashSet<u32> = FxHashSet::default();
    for (i, op) in body.ops.iter().enumerate() {
        for r in op.result_ids() {
            if !produced.insert(r) {
                errors.push(VerifyError {
                    body_path: path.clone(),
                    op_index: i,
                    kind: VerifyErrorKind::DuplicateResultId(r),
                });
            }
        }
    }

    // 2 & 3 & 4 & 5: per-op operand checks.
    let mut produced_so_far: FxHashSet<u32> = FxHashSet::default();
    let child_results: Vec<FxHashSet<u32>> =
        body.child_bodies.iter().map(collect_body_results).collect();
    let mut completed_child_results: FxHashSet<u32> = FxHashSet::default();
    let mut child_scopes = vec![FxHashSet::default(); body.child_bodies.len()];
    for (i, op) in body.ops.iter().enumerate() {
        // Literal ops must have at least one operand (pool index).
        if matches!(op.kind, KernelOpKind::Literal) {
            if op.operands.is_empty() {
                errors.push(VerifyError {
                    body_path: path.clone(),
                    op_index: i,
                    kind: VerifyErrorKind::LiteralOpMissingPoolOperand,
                });
            } else {
                let pool_idx = op.operands[0];
                if (pool_idx as usize) >= body.literals.len() {
                    errors.push(VerifyError {
                        body_path: path.clone(),
                        op_index: i,
                        kind: VerifyErrorKind::LiteralPoolOutOfRange {
                            operand_pos: 0,
                            pool_idx,
                            pool_size: body.literals.len(),
                        },
                    });
                }
            }
        }

        // Per-position classification.
        for (pos, &val) in op.operands.iter().enumerate() {
            let cls = classify_operand(&op.kind, pos);
            match cls {
                OperandClass::ResultRef => {
                    if !produced_so_far.contains(&val)
                        && !produced.contains(&val)
                        && !inherited_results.contains(&val)
                        && !completed_child_results.contains(&val)
                    {
                        errors.push(VerifyError {
                            body_path: path.clone(),
                            op_index: i,
                            kind: VerifyErrorKind::DanglingResultRef {
                                operand_pos: pos,
                                ref_id: val,
                            },
                        });
                    }
                }
                OperandClass::ChildBodyIdx => {
                    if (val as usize) >= body.child_bodies.len() {
                        errors.push(VerifyError {
                            body_path: path.clone(),
                            op_index: i,
                            kind: VerifyErrorKind::ChildBodyIndexOutOfRange {
                                operand_pos: pos,
                                body_idx: val,
                                child_count: body.child_bodies.len(),
                            },
                        });
                    } else {
                        let child_scope = &mut child_scopes[val as usize];
                        child_scope.extend(inherited_results.iter().copied());
                        child_scope.extend(produced_so_far.iter().copied());
                        child_scope.extend(completed_child_results.iter().copied());
                    }
                }
                OperandClass::LiteralPoolIdx => {
                    if (val as usize) >= body.literals.len() {
                        errors.push(VerifyError {
                            body_path: path.clone(),
                            op_index: i,
                            kind: VerifyErrorKind::LiteralPoolOutOfRange {
                                operand_pos: pos,
                                pool_idx: val,
                                pool_size: body.literals.len(),
                            },
                        });
                    }
                }
                OperandClass::BindingSlot => {
                    if !declared_slots.contains(&val) {
                        errors.push(VerifyError {
                            body_path: path.clone(),
                            op_index: i,
                            kind: VerifyErrorKind::UndeclaredBindingSlot {
                                operand_pos: pos,
                                slot: val,
                            },
                        });
                    }
                }
                OperandClass::Other => {}
            }
        }

        // Minimum operand count per kind. Conservative  -  we just check
        // shapes the rewrites actually produce.
        let min_required = min_operand_count(&op.kind);
        if op.operands.len() < min_required {
            errors.push(VerifyError {
                body_path: path.clone(),
                op_index: i,
                kind: VerifyErrorKind::OperandCountTooShort {
                    expected_min: min_required,
                    got: op.operands.len(),
                },
            });
        }

        // A matrix multiply-accumulate states its own arity through its
        // declared fragments, so it is checked exactly rather than by a
        // minimum: an operand list of a different length means the op and its
        // specification state different fragments.
        if let KernelOpKind::MatrixMma(spec) = &op.kind {
            match spec.operand_count() {
                Err(reason) => errors.push(VerifyError {
                    body_path: path.clone(),
                    op_index: i,
                    kind: VerifyErrorKind::MatrixFragmentUnstatable { reason },
                }),
                Ok(expected) if expected as usize != op.operands.len() => {
                    errors.push(VerifyError {
                        body_path: path.clone(),
                        op_index: i,
                        kind: VerifyErrorKind::MatrixOperandCountMismatch {
                            expected,
                            got: op.operands.len(),
                        },
                    });
                }
                Ok(_) => {}
            }
        }

        // An asynchronous transfer states its own ring slot, visibility and
        // fence. A declaration that pairs nothing, addresses a slot outside
        // its ring, or fences narrower than the visibility it claims cannot
        // be carried by any target, whichever mechanism the target picks.
        let declaration = match &op.kind {
            KernelOpKind::AsyncLoad(transaction) | KernelOpKind::AsyncStore(transaction) => {
                Some(transaction.validate())
            }
            KernelOpKind::AsyncWait(wait) => Some(wait.validate()),
            _ => None,
        };
        if let Some(Err(reason)) = declaration {
            errors.push(VerifyError {
                body_path: path.clone(),
                op_index: i,
                kind: VerifyErrorKind::AsyncTransactionUnstatable { reason },
            });
        }

        for r in op.result_ids() {
            produced_so_far.insert(r);
        }
        for child_idx in child_body_operands(op) {
            if let Some(results) = child_results.get(child_idx as usize) {
                completed_child_results.extend(results.iter().copied());
            }
        }
    }

    // Recurse.
    for (idx, child) in body.child_bodies.iter().enumerate() {
        path.push(idx);
        verify_body(child, path, &child_scopes[idx], declared_slots, errors);
        path.pop();
    }
}

fn collect_body_results(body: &KernelBody) -> FxHashSet<u32> {
    crate::analyses::body_result_ids(body)
}

fn child_body_operands(op: &crate::KernelOp) -> impl Iterator<Item = u32> + '_ {
    op.operands
        .iter()
        .enumerate()
        .filter_map(|(pos, &operand)| {
            (classify_operand(&op.kind, pos) == OperandClass::ChildBodyIdx).then_some(operand)
        })
}

fn min_operand_count(kind: &KernelOpKind) -> usize {
    use KernelOpKind::*;
    match kind {
        Literal => 1,
        Copy => 1,
        LocalInvocationId | GlobalInvocationId | WorkgroupId => 0,
        SubgroupLocalId | SubgroupSize => 0,
        LoopIndex { .. } => 0,
        BufferLength => 1,
        LoadGlobal | LoadShared | LoadConstant | VectorLoadGlobal { .. } => 2,
        StoreGlobal | StoreShared => 3,
        VectorStoreGlobal { width } => 2 + (*width as usize),
        ExtractLane { .. } => 1,
        BinOpKind(_) => 2,
        UnOpKind(_) | Cast { .. } => 1,
        Fma => 3,
        MatrixMma(spec) => spec.operand_count().unwrap_or(0) as usize,
        Select => 3,
        Atomic { .. } => 2,
        SubgroupBallot | SubgroupShuffle | SubgroupBroadcast | SubgroupReduce { .. } => 1,
        StructuredIfThen => 2,
        StructuredIfThenElse => 3,
        StructuredForLoop { .. } => 3,
        StructuredBlock => 1,
        Region { .. } => 1,
        Return => 0,
        Barrier { .. } => 0,
        AsyncLoad(_) | AsyncStore(_) => 4,
        AsyncWait(_) => 0,
        Trap { .. } => 1,
        Resume { .. } => 0,
        IndirectDispatch { .. } => 0,
        Call { .. } => 0,
        OpaqueExpr(..) | OpaqueNode(..) => 0,
        LoopCarrier { .. } => 0,
        LoopCarrierInit { .. } | LoopCarrierEnd { .. } => 1,
    }
}

/// Check that every wait completes a transfer this descriptor issues.
///
/// Pairing is descriptor-wide: an issue inside a loop body is waited on by an
/// op of the enclosing body, so a per-body walk cannot see both ends. A wait
/// that pairs with nothing leaves a target draining an unrelated transfer, and
/// a wait whose ring slot no issue filled completes a slot that was never
/// occupied.
pub(super) fn verify_async_pairing(body: &KernelBody, errors: &mut Vec<VerifyError>) {
    use crate::descriptor::StageSlot;
    use crate::Name;
    use rustc_hash::FxHashMap;

    let mut issued: FxHashMap<&Name, Vec<Option<StageSlot>>> = FxHashMap::default();
    collect_issues(body, &mut issued);

    fn collect_issues<'a>(
        body: &'a KernelBody,
        issued: &mut FxHashMap<&'a Name, Vec<Option<StageSlot>>>,
    ) {
        for op in &body.ops {
            if let KernelOpKind::AsyncLoad(transaction) | KernelOpKind::AsyncStore(transaction) =
                &op.kind
            {
                issued
                    .entry(transaction.tag())
                    .or_default()
                    .push(transaction.stage());
            }
        }
        for child in &body.child_bodies {
            collect_issues(child, issued);
        }
    }

    fn walk(
        body: &KernelBody,
        path: &mut Vec<usize>,
        issued: &FxHashMap<&Name, Vec<Option<StageSlot>>>,
        errors: &mut Vec<VerifyError>,
    ) {
        for (op_index, op) in body.ops.iter().enumerate() {
            let KernelOpKind::AsyncWait(wait) = &op.kind else {
                continue;
            };
            let transaction = wait.transaction();
            let tag = transaction.tag();
            let kind = match issued.get(tag) {
                None => VerifyErrorKind::AsyncWaitUnmatched {
                    tag: tag.to_string(),
                },
                Some(stages) if !stages.contains(&transaction.stage()) => {
                    VerifyErrorKind::AsyncStageDisagreement {
                        tag: tag.to_string(),
                    }
                }
                Some(_) => continue,
            };
            errors.push(VerifyError {
                body_path: path.clone(),
                op_index,
                kind,
            });
        }
        for (child_index, child) in body.child_bodies.iter().enumerate() {
            path.push(child_index);
            walk(child, path, issued, errors);
            path.pop();
        }
    }

    walk(body, &mut Vec::new(), &issued, errors);
}
