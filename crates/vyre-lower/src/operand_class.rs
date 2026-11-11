//! The operand namespace of a lowered kernel op.
//!
//! A `KernelOp` operand list is a flat `Vec<u32>`, but a position can be an
//! SSA result reference, an index into the body's literal pool, an index into
//! its child-body table, a declared binding slot, or a plain number such as an
//! axis or a width. This module is the one owner of that table. Every analysis,
//! rewrite, verifier, and emitter that must tell those apart asks here instead
//! of re-deriving per-kind skip offsets, because two copies drift and a copy
//! that treats metadata as an SSA value miscompiles in silence.

use crate::KernelOpKind;

/// Semantic class assigned to one operation operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandClass {
    /// Reference to a previously produced result.
    ResultRef,
    /// Index into the current body's child-body table.
    ChildBodyIdx,
    /// Index into the current body's literal pool.
    LiteralPoolIdx,
    /// Declared binding slot the op addresses storage through.
    ///
    /// Separate from [`OperandClass::Other`] because a slot resolves against
    /// the descriptor's binding layout: the verifier rejects an op addressing a
    /// slot nothing declares, and the storage planner reads the same positions
    /// to derive how long each region is live.
    BindingSlot,
    /// Opaque tag, axis, width, etc.  -  not validated structurally.
    Other,
}

/// Classify one operand position for structural verification.
pub fn classify_operand(kind: &KernelOpKind, pos: usize) -> OperandClass {
    use KernelOpKind::*;
    match kind {
        Literal => {
            if pos == 0 {
                OperandClass::LiteralPoolIdx
            } else {
                OperandClass::Other
            }
        }
        LocalInvocationId | GlobalInvocationId | WorkgroupId => OperandClass::Other,
        SubgroupLocalId | SubgroupSize => OperandClass::Other,
        LoopIndex { .. } => OperandClass::Other,
        BufferLength => {
            if pos == 0 {
                OperandClass::BindingSlot
            } else {
                OperandClass::Other
            }
        }
        LoadGlobal | LoadShared | LoadConstant | VectorLoadGlobal { .. } => {
            if pos == 0 {
                OperandClass::BindingSlot
            } else {
                OperandClass::ResultRef
            }
        }
        StoreGlobal | StoreShared | VectorStoreGlobal { .. } => {
            if pos == 0 {
                OperandClass::BindingSlot
            } else {
                OperandClass::ResultRef
            }
        }
        ExtractLane { .. } => {
            if pos == 0 {
                OperandClass::ResultRef
            } else {
                OperandClass::Other
            }
        }
        Copy | BinOpKind(_) | UnOpKind(_) | Fma | MatrixMma(_) | Select | Cast { .. } => {
            OperandClass::ResultRef
        }
        Atomic { .. } => {
            if pos == 0 {
                OperandClass::BindingSlot
            } else {
                OperandClass::ResultRef
            }
        }
        SubgroupBallot | SubgroupShuffle | SubgroupBroadcast | SubgroupReduce { .. } => {
            OperandClass::ResultRef
        }
        StructuredIfThen => {
            if pos == 0 {
                OperandClass::ResultRef
            } else if pos == 1 {
                OperandClass::ChildBodyIdx
            } else {
                OperandClass::Other
            }
        }
        StructuredIfThenElse => {
            if pos == 0 {
                OperandClass::ResultRef
            } else if pos == 1 || pos == 2 {
                OperandClass::ChildBodyIdx
            } else {
                OperandClass::Other
            }
        }
        // `[lo, hi, body]` is the contract. An operand past the body index
        // is out of contract, and stays an SSA reference: under-counting a
        // use is the unsafe direction, because a value that is still read
        // looks dead to an elimination or hoisting decision.
        StructuredForLoop { .. } => {
            if pos == 2 {
                OperandClass::ChildBodyIdx
            } else {
                OperandClass::ResultRef
            }
        }
        StructuredBlock => {
            if pos == 0 {
                OperandClass::ChildBodyIdx
            } else {
                OperandClass::Other
            }
        }
        Region { .. } => {
            if pos == 0 {
                OperandClass::ChildBodyIdx
            } else {
                OperandClass::Other
            }
        }
        Return | Barrier { .. } => OperandClass::Other,
        AsyncLoad(_) | AsyncStore(_) => {
            if pos < 2 {
                OperandClass::BindingSlot
            } else {
                OperandClass::ResultRef
            }
        }
        AsyncWait(_) => OperandClass::Other,
        Trap { .. } => {
            if pos == 0 {
                OperandClass::ResultRef
            } else {
                OperandClass::Other
            }
        }
        Resume { .. } => OperandClass::Other,
        IndirectDispatch { .. } => OperandClass::Other,
        Call { .. } => OperandClass::ResultRef,
        OpaqueExpr(..) | OpaqueNode(..) => OperandClass::ResultRef,
        LoopCarrierInit { .. } | LoopCarrier { .. } | LoopCarrierEnd { .. } => {
            OperandClass::ResultRef
        }
    }
}

/// True when `kind.operands[pos]` is a result-id reference in the lowered
/// kernel SSA namespace.
///
/// A shorthand for the [`OperandClass::ResultRef`] case of
/// [`classify_operand`], for callers that only need data dependencies and
/// have no use for the literal-pool and child-body distinctions.
#[must_use]
pub fn operand_is_result_reference(kind: &KernelOpKind, pos: usize) -> bool {
    matches!(classify_operand(kind, pos), OperandClass::ResultRef)
}
