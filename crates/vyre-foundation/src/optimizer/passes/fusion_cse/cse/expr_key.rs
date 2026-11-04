use super::TypeKey;
use crate::ir::Ident;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CseExprId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum ExprKey {
    LitU32(u32),
    LitI32(i32),
    /// Stores the IEEE 754 bits so `Eq`/`Hash`/`Ord` work correctly.
    LitF32(u32),
    LitBool(bool),
    Var(Ident),
    Load(Ident, CseExprId),
    BufferRef(Ident),
    BufLen(Ident),
    InvocationId(u8),
    WorkgroupId(u8),
    LocalId(u8),
    LogicalIndex(u8),
    LogicalTileId(u8),
    LogicalWithinTileId(u8),
    BinOp(u8, CseExprId, CseExprId),
    UnOp(u8, CseExprId),
    /// CSE key for `BinOp::Opaque(id)`  -  stores the extension u32 id so
    /// two extensions with distinct ids hash to distinct keys. Without
    /// this, every `BinOp::Opaque(_)` collapsed onto a single key via
    /// `bin_op_key`'s wildcard fallback  -  silently merging unrelated
    /// extensions in CSE. (§1 injectivity contract.)
    BinOpOpaque(u32, CseExprId, CseExprId),
    /// CSE key for `UnOp::Opaque(id)`  -  same rationale as `BinOpOpaque`.
    UnOpOpaque(u32, CseExprId),
    Call(Ident, SmallVec<[CseExprId; 4]>),
    Fma(CseExprId, CseExprId, CseExprId),
    Select(CseExprId, CseExprId, CseExprId),
    Cast(TypeKey, CseExprId),
    Atomic,
    /// Subgroup intrinsics are effectful + lane-correlated; CSE must not
    /// merge them, so every instance gets a unique counter-keyed identity.
    Subgroup(u32),
    SubgroupLocalId,
    SubgroupSize,
    Opaque(&'static str, [u8; 32]),
}
